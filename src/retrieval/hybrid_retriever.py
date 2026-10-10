"""混合检索：向量召回 + BM25 召回 + RRF 融合（模块一 5.1.4）。

为什么需要混合检索
------------------
- **纯向量**擅长语义泛化（"过拟合怎么办" ↔ "正则化策略"），
  但对专有名词、缩写、代码符号（"BatchNorm"、"mAP@0.5"）匹配不稳，
  且向量模型对训练语料外的术语表征可能很弱。
- **纯 BM25**擅长精确词面匹配，但完全无法处理同义改写与跨语言语义。

二者互补。本项目采用 **RRF（Reciprocal Rank Fusion，倒数排名融合）** 融合：

.. math::

    \\text{RRF}(d) = \\sum_{r \\in R} \\frac{w_r}{k + \\text{rank}_r(d)}

选择 RRF 而非"分数加权求和"的关键原因：**向量相似度（0~1）与 BM25 分数
（0~几十，无上界）量纲完全不同**，直接加权需要归一化，而归一化会引入
对分数分布敏感的超参。RRF 只用**排名**，天然免疫量纲问题，
且被证明在异构检索器融合上稳健性最好（Cormack et al., SIGIR 2009）。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.logger import get_logger
from ..utils.types import Chunk, RetrievedChunk
from .bm25_retriever import BM25Retriever
from .vector_store import BaseVectorStore

__all__ = ["HybridRetriever", "rrf_fuse", "RetrievalTrace"]

_log = get_logger("retrieval.hybrid")


def rrf_fuse(
    ranked_lists: Sequence[Tuple[str, Sequence[Any]]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    key: str = "chunk_id",
) -> List[Tuple[str, float, Dict[str, int]]]:
    """RRF 融合多路排名列表。

    Parameters
    ----------
    ranked_lists:
        ``[(route_name, [item, item, ...]), ...]``，列表内顺序即排名（从 1 开始）。
    k: RRF 平滑常数（论文推荐 60；越小越放大头部排名差异）
    weights: 各路权重，与 ``ranked_lists`` 等长
    key: 从 item 中提取唯一标识的属性名

    Returns
    -------
    ``[(item_key, fused_score, {route: rank}), ...]`` 按融合分数降序。
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}
    ranks: Dict[str, Dict[str, int]] = {}

    for (route, items), weight in zip(ranked_lists, weights):
        if not weight:
            continue
        for rank, item in enumerate(items, start=1):
            # 支持三种入参形态：字符串 ID、RetrievedChunk（取 .chunk.chunk_id）、dict
            if isinstance(item, str):
                item_key = item
            else:
                chunk = getattr(item, "chunk", None)
                item_key = getattr(chunk, "chunk_id", None) if chunk is not None else None
                if item_key is None:
                    item_key = getattr(item, key, None)
                if item_key is None and isinstance(item, dict):
                    item_key = item.get(key)
            if not item_key:
                continue
            scores[item_key] = scores.get(item_key, 0.0) + weight / (k + rank)
            ranks.setdefault(item_key, {})[route] = rank

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(item_key, score, ranks.get(item_key, {})) for item_key, score in ordered]


@dataclass
class RetrievalTrace:
    """一次检索的可观测记录（写入 traces.jsonl，供评测与 Bad Case 分析）。"""

    query: str
    num_vector: int = 0
    num_bm25: int = 0
    num_fused: int = 0
    num_returned: int = 0
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    vector_ms: float = 0.0
    bm25_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    reranked: bool = False
    top1_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "num_vector": self.num_vector,
            "num_bm25": self.num_bm25,
            "num_fused": self.num_fused,
            "num_returned": self.num_returned,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "vector_ms": round(self.vector_ms, 2),
            "bm25_ms": round(self.bm25_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "reranked": self.reranked,
            "top1_score": round(self.top1_score, 4),
        }


class HybridRetriever:
    """混合检索器：向量 + BM25 → RRF → （可选）重排序。

    Attributes
    ----------
    vector_store: 向量库（提供语义召回）
    bm25: BM25 检索器（提供关键词召回）
    reranker: 重排序器（可为 None）
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25: Optional[BM25Retriever] = None,
        reranker: Optional[Any] = None,
        config: Any = None,
        embedder: Optional[Any] = None,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        enable_bm25: Optional[bool] = None,
        enable_rerank: Optional[bool] = None,
        low_relevance_threshold: Optional[float] = None,
        min_score: Optional[float] = None,
    ) -> None:
        from ..utils.config import get_config

        cfg = config or get_config()
        node = cfg.retrieval

        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.embedder = embedder

        self.top_k = int(top_k if top_k is not None else node.top_k)
        self.candidate_k = int(candidate_k if candidate_k is not None else node.candidate_k)
        self.rrf_k = int(rrf_k if rrf_k is not None else node.rrf_k)
        self.vector_weight = float(
            vector_weight if vector_weight is not None else node.vector_weight
        )
        self.bm25_weight = float(bm25_weight if bm25_weight is not None else node.bm25_weight)
        self.enable_bm25 = bool(enable_bm25 if enable_bm25 is not None else node.enable_bm25)
        self.enable_rerank = bool(enable_rerank if enable_rerank is not None else node.enable_rerank)
        self.low_relevance_threshold = float(
            low_relevance_threshold
            if low_relevance_threshold is not None
            else node.low_relevance_threshold
        )
        # 绝对相关度门槛：作用在 confidence 上，而非归一化后的 score。
        # 详见 retrieve() 中过滤处的说明。默认 0.0 = 不过滤。
        self.min_score = float(min_score if min_score is not None else node.get("min_score", 0.0) or 0.0)
        self._last_trace: Optional[RetrievalTrace] = None

    # -- 主检索入口 -------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        candidate_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """执行检索。

        Parameters
        ----------
        query: 检索问题
        top_k: 返回条数（覆盖配置）
        mode: 强制指定检索档位，用于**三档对比实验**：

              - ``vector``  纯向量检索
              - ``hybrid``  向量 + BM25 + RRF
              - ``full``    混合 + 重排序（默认，等价于配置全开）
        candidate_k: 每路召回候选数（覆盖配置）

        Returns
        -------
        按相关性降序的 :class:`RetrievedChunk` 列表
        """
        start = time.perf_counter()
        final_k = int(top_k or self.top_k)
        pool_k = int(candidate_k or max(self.candidate_k, final_k))
        trace = RetrievalTrace(query=query)

        use_bm25 = self.enable_bm25 and mode != "vector"
        use_rerank = self.enable_rerank and mode in (None, "full") and self.reranker is not None

        # ---- 1) 两路并行召回 ----
        vector_hits: List[RetrievedChunk] = []
        bm25_hits: List[RetrievedChunk] = []

        def run_vector() -> Tuple[List[RetrievedChunk], float]:
            began = time.perf_counter()
            hits: List[RetrievedChunk] = []
            if self.embedder is not None:
                vector = self.embedder.encode(query)
                hits = self.vector_store.search(vector, top_k=pool_k)
            return hits, (time.perf_counter() - began) * 1000

        def run_bm25() -> Tuple[List[RetrievedChunk], float]:
            began = time.perf_counter()
            hits: List[RetrievedChunk] = []
            if use_bm25 and self.bm25 is not None and self.bm25.size:
                hits = self.bm25.search(query, top_k=pool_k)
            return hits, (time.perf_counter() - began) * 1000

        if use_bm25:
            with ThreadPoolExecutor(max_workers=2) as pool:
                vector_future = pool.submit(run_vector)
                bm25_future = pool.submit(run_bm25)
                vector_hits, vector_ms = vector_future.result()
                bm25_hits, bm25_ms = bm25_future.result()
        else:
            vector_hits, vector_ms = run_vector()
            bm25_ms = 0.0

        trace.vector_ms, trace.bm25_ms = vector_ms, bm25_ms
        trace.num_vector, trace.num_bm25 = len(vector_hits), len(bm25_hits)

        # ---- 2) 融合 ----
        if use_bm25 and bm25_hits:
            fused = self._fuse(vector_hits, bm25_hits, pool_k)
        else:
            # 单路：直接使用向量结果，但把分数归一化到 0~1 便于阈值判断
            fused = self._normalize_single_route(vector_hits)
        trace.num_fused = len(fused)

        # ---- 3) 重排序 ----
        if use_rerank and fused:
            rerank_start = time.perf_counter()
            results = self.reranker.rerank(query, fused, top_n=final_k)
            trace.rerank_ms = (time.perf_counter() - rerank_start) * 1000
            trace.reranked = True
        else:
            results = fused[:final_k]

        # 相关性过滤作用在 **confidence**（绝对相关度）上，而不是 score。
        #
        # 为什么不能用 score 过滤：score 是 min-max 归一化后的**相对**分数
        # （见 _finalize_scores / BaseReranker._normalize），本批最差的那条恒为
        # 0.0。早期写成 `r.score > 0`，于是"所有并列最低分"的候选被整批删掉 ——
        # 实测 20 个片段的语料要求 top_k=20 时只返回 2 条，且并列项在启发式
        # 重排序下非常常见（不相关片段得分完全相同）。这会让 top_k 形同虚设，
        # 并连带污染 Hit@K / MRR / Recall 等所有评测指标。
        #
        # confidence 才是"到底有没有检索到"的绝对度量，min_score 的语义
        # （config.yaml: "低于该分数的结果被丢弃"）本就该用它来判。
        if self.min_score > 0:
            results = [r for r in results if r.confidence >= self.min_score]
        trace.num_returned = len(results)
        trace.retrieved_chunk_ids = [r.chunk.chunk_id for r in results]
        trace.top1_score = results[0].score if results else 0.0
        trace.total_ms = (time.perf_counter() - start) * 1000
        self._last_trace = trace

        _log.debug(
            "检索完成 query=%r vector=%d bm25=%d returned=%d 耗时=%.1fms",
            query[:40],
            trace.num_vector,
            trace.num_bm25,
            trace.num_returned,
            trace.total_ms,
        )
        return results

    # -- 融合实现 ---------------------------------------------------------
    def _fuse(
        self,
        vector_hits: Sequence[RetrievedChunk],
        bm25_hits: Sequence[RetrievedChunk],
        limit: int,
    ) -> List[RetrievedChunk]:
        """RRF 融合两路召回结果。

        原始 RRF 分数暂存到 ``rerank_score`` 字段（融合阶段该字段尚未使用），
        供 :meth:`_finalize_scores` 计算绝对置信度后清除。
        """
        by_id: Dict[str, RetrievedChunk] = {}
        # 分路保存各路的原始分：同一个 chunk 被两路都召回时，by_id 只能留一个
        # 对象（早期实现用 setdefault，留下向量路的对象），另一路的分数会丢失 ——
        # 表现为"BM25 排第 1 的片段 bm25_score=0.0"。分数需按路合并取最大值。
        vector_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}
        for hit in vector_hits:
            cid = hit.chunk.chunk_id
            by_id.setdefault(cid, hit)
            vector_scores[cid] = max(vector_scores.get(cid, 0.0), float(hit.vector_score or 0.0))
        for hit in bm25_hits:
            cid = hit.chunk.chunk_id
            by_id.setdefault(cid, hit)
            bm25_scores[cid] = max(bm25_scores.get(cid, 0.0), float(hit.bm25_score or 0.0))

        fused = rrf_fuse(
            [("vector", list(vector_hits)), ("bm25", list(bm25_hits))],
            k=self.rrf_k,
            weights=[self.vector_weight, self.bm25_weight],
        )

        results: List[RetrievedChunk] = []
        for chunk_id, rrf_score, ranks in fused[:limit]:
            hit = by_id.get(chunk_id)
            if hit is None:
                continue
            # 复制一份，避免污染原始召回结果
            results.append(
                RetrievedChunk(
                    chunk=hit.chunk,
                    score=rrf_score,          # 暂存原始 RRF 分
                    rerank_score=rrf_score,   # 暂存副本，供置信度换算
                    vector_score=vector_scores.get(chunk_id, hit.vector_score),
                    bm25_score=bm25_scores.get(chunk_id, hit.bm25_score),
                    vector_rank=ranks.get("vector", 0),
                    bm25_rank=ranks.get("bm25", 0),
                    source_route="hybrid",
                )
            )
        return self._finalize_scores(results, router_count=2, rrf_k=self.rrf_k)

    @staticmethod
    def _normalize_single_route(hits: Sequence[RetrievedChunk]) -> List[RetrievedChunk]:
        """单路检索（纯向量）时：``confidence`` 直接取向量余弦相似度。

        单路场景不需要 RRF —— 余弦相似度本身就是可用的绝对置信度
        （0~1，且与文档完全无关的查询会得到接近 0 的值）。
        """
        if not hits:
            return []
        has_vector = any(h.vector_score for h in hits)
        bm25_top = max((h.bm25_score for h in hits), default=0.0) or 1.0

        copied: List[RetrievedChunk] = []
        for h in hits:
            if has_vector:
                confidence = max(0.0, min(1.0, h.vector_score))
            else:
                # 纯 BM25 单路：分数无上界，用本路 Top-1 归一作为保守置信度
                confidence = max(0.0, min(1.0, h.bm25_score / bm25_top))
            copied.append(
                RetrievedChunk(
                    chunk=h.chunk,
                    score=h.score,
                    confidence=confidence,
                    vector_score=h.vector_score,
                    bm25_score=h.bm25_score,
                    rerank_score=h.rerank_score,
                    vector_rank=h.vector_rank,
                    bm25_rank=h.bm25_rank,
                    source_route=h.source_route or "vector",
                )
            )
        # confidence 已由本方法给定，只做展示分归一化
        return HybridRetriever._finalize_scores(copied, has_confidence=True)

    @staticmethod
    def _finalize_scores(
        hits: List[RetrievedChunk],
        router_count: int = 2,
        has_confidence: bool = False,
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """统一计算两个分数（所有检索路径的最后一步）。

        ``score``（相对展示分）
            min-max 归一化到 0~1，Top-1 恒为 1.0。回答"这条比那条好多少"，
            供前端画相关度条。**不可用于降级判定。**

        ``confidence``（绝对置信度）
            融合分数 ÷ 理论上限（``router_count / (k + 1)``，即每个召回器都把它
            排在第 1 位时的最大可能融合分）。回答"这条到底有多相关"，
            **降级判定与阈值过滤必须用它**。

        为什么必须分开（真实踩坑记录）
            早期版本把 min-max 归一化后的 ``score`` 直接当置信度，导致
            "今天天气怎么样"这类与知识库完全无关的问题，Top-1 也拿到 1.00 的
            "置信度"，低相关性降级永不触发，系统会拿无关片段去生成答案。

        举例（k=60，两路召回，理论上限 = 2/61 ≈ 0.0328）：
        某 chunk 仅被向量路排第 1（RRF = 1/61 ≈ 0.0164）→ confidence ≈ 0.50；
        两路都排第 1（RRF = 2/61）→ confidence ≈ 1.00；
        若两路都是无关结果（RRF 很小）→ confidence 同样很小，正确触发降级。
        """
        if not hits:
            return []

        # RRF 理论最大值：每条召回路径都贡献 rank=1，即 router_count / (k + 1)
        theoretical_max = max(1e-9, router_count / float(rrf_k + 1))

        scores = [h.score for h in hits]
        top, bottom = max(scores), min(scores)
        span = top - bottom

        for hit in hits:
            raw = hit.score
            hit.score = (
                (1.0 if top > 0 else 0.0)
                if span <= 1e-12
                else round((raw - bottom) / span, 6)
            )
            if not has_confidence:
                hit.confidence = round(min(1.0, raw / theoretical_max), 6)
                hit.rerank_score = 0.0  # 清除暂存的 RRF 原始分
        return hits

    # -- 可观测性 ---------------------------------------------------------
    @property
    def last_trace(self) -> Optional[RetrievalTrace]:
        return self._last_trace

    def stats(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "enable_bm25": self.enable_bm25,
            "enable_rerank": self.enable_rerank and self.reranker is not None,
            "low_relevance_threshold": self.low_relevance_threshold,
            "min_score": self.min_score,
            "vector_store": self.vector_store.stats(),
            "bm25": self.bm25.stats() if self.bm25 else None,
            "reranker": self.reranker.stats() if self.reranker else None,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "vector_store": self.vector_store.health_check(),
            "bm25": self.bm25.health_check() if self.bm25 else {"ok": True, "enabled": False},
            "reranker": self.reranker.health_check() if self.reranker else {"ok": True, "enabled": False},
        }
