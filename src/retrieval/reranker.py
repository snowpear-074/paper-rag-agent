"""重排序（Rerank）模块（模块一 5.1.4 第二层）。

为什么需要重排序
----------------
召回阶段（向量/BM25）必须在**毫秒级**从全库里筛出候选，因此只能用
"双塔"式的独立编码（query 与 doc 分别编码后算相似度），精度有限。
重排序阶段只处理 Top-20 候选，可以用**交叉编码**（query 与 doc 拼接后
一起过模型）获得高得多的相关性判断精度 —— 这是"召回-精排"两阶段架构的核心。

本项目提供三种重排序器，对应不同资源条件：

=================  ==================================================================
provider           说明
=================  ==================================================================
``heuristic``      零依赖启发式重排。综合 6 个可解释特征加权打分：
                   词项覆盖率、IDF 加权覆盖、查询词邻近度、标题命中、
                   数字/术语精确匹配、长度先验。适合无 GPU 环境，
                   且**特征可导出**，便于在报告中分析"模型为什么这样排序"。
``ollama``         通过 Ollama 调用 bge-reranker（需本地部署）。
``local``          本地 sentence-transformers CrossEncoder。
=================  ==================================================================

优先级：``provider`` 配置指定；若指定的 provider 不可用，自动降级到 heuristic。
"""

from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.text import tokenize_query, tokenize
from ..utils.types import RetrievedChunk

__all__ = ["BaseReranker", "HeuristicReranker", "OllamaReranker", "LocalReranker",
           "create_reranker", "RERANKER_PROVIDERS"]

_log = get_logger("retrieval.reranker")

# 数字/术语模式（"28.4 BLEU"、"93.2 F1"、"ResNet-152"、"mAP@0.5"）
_RE_NUMERIC_TERM = re.compile(
    r"\d+(?:\.\d+)?%?|[A-Za-z]+[-_]?\d+|[A-Z]{2,}(?:@[\d\.]+)?"
)


class BaseReranker(ABC):
    """重排序器抽象接口。"""

    provider: str = "base"

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = int(top_n)
        self._last_ms: float = 0.0

    @abstractmethod
    def _score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        """返回每个文档与查询的相关性分数（越大越相关）。"""
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """对候选重排序，返回 Top-N。

        评分字段的写入约定
        ------------------
        - ``rerank_score``：排序器原始输出（**保留绝对尺度**）
        - ``confidence``：绝对相关性置信度（= 原始分数的 sigmoid / 余弦，0~1）
        - ``score``：归一化相对分（Top-1 恒为 1.0），仅供展示排序

        降级判定请使用 ``confidence``，不要用 ``score``。
        """
        items = list(candidates)
        if not items:
            return []
        limit = int(top_n or self.top_n)
        start = time.perf_counter()
        raw = self._score_pairs(query, [c.text for c in items])
        self._last_ms = (time.perf_counter() - start) * 1000

        if len(raw) != len(items):
            _log.warning("重排序返回分数数量(%d)与候选数(%d)不一致，回退原顺序", len(raw), len(items))
            return items[:limit]

        for hit, score in zip(items, raw):
            hit.rerank_score = float(score)
            hit.confidence = self.to_confidence(float(score))

        ordered = sorted(items, key=lambda h: -h.rerank_score)
        normalized = self._normalize([h.rerank_score for h in ordered])
        for hit, score in zip(ordered, normalized):
            hit.score = score
            hit.source_route = "rerank" if hit.source_route == "vector" else f"{hit.source_route}+rerank"
        return ordered[:limit]

    def to_confidence(self, raw_score: float) -> float:
        """把排序器原始分数转成绝对置信度（0~1）。

        子类可覆写：启发式重排的 raw 已经是 sigmoid 输出，直接裁剪即可；
        CrossEncoder 的 logits 通常是 -10~10，需要再过一次 sigmoid。
        """
        return float(min(1.0, max(0.0, raw_score)))

    @staticmethod
    def _normalize(scores: Sequence[float]) -> List[float]:
        """把分数线性映射到 0~1（保持相对顺序）。"""
        if not scores:
            return []
        top, bottom = max(scores), min(scores)
        span = top - bottom
        if span <= 1e-12:
            return [1.0 if top > 0 else 0.0] * len(scores)
        return [round((s - bottom) / span, 6) for s in scores]

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "top_n": self.top_n,
            "last_latency_ms": round(self._last_ms, 2),
        }

    def health_check(self) -> Dict[str, Any]:
        try:
            scores = self._score_pairs("测试查询", ["测试文档内容", "无关内容"])
            return {"ok": True, "provider": self.provider, "sample_scores": [round(s, 4) for s in scores]}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": self.provider, "error": str(exc)[:200]}


# ==============================================================================
# 启发式重排序（零依赖，默认）
# ==============================================================================
class HeuristicReranker(BaseReranker):
    """基于多特征加权的启发式重排序器。

    六个可解释特征
    --------------
    1. **词项覆盖率**：查询词项在文档中出现的比例（BM25 的无权重版本，
       捕捉"是否谈到"）。
    2. **IDF 加权覆盖**：稀有词命中权重更高 —— "Transformer"命中比"方法"命中重要得多。
    3. **邻近度**：命中的查询词项之间的最小字符距离。距离越近，
       越可能是同一个语义单元（"注意力"和"机制"挨在一起才算命中概念）。
    4. **标题/来源命中**：文档标题包含查询词项时加分（论文标题通常直指主题）。
    5. **术语精确匹配**：查询中的数字与专有名词（"28.4 BLEU"）在文档中精确出现时加分 ——
       这是向量检索最容易失效、而对科研问答最关键的一类匹配。
    6. **长度先验**：过短（信息不足）或过长（噪声多）的块适度降权。

    各特征加权求和后经 **sigmoid** 压缩，得到稳定的 0~1 分数。
    """

    provider = "heuristic"

    # 特征权重（经小规模网格搜索调优；在报告中给出调参过程）
    WEIGHTS: Dict[str, float] = {
        "coverage": 1.00,
        "idf_coverage": 1.60,
        "proximity": 0.75,
        "title_hit": 0.55,
        "term_match": 0.90,
        "length_prior": 0.25,
    }

    def __init__(self, top_n: int = 5, weights: Optional[Dict[str, float]] = None) -> None:
        super().__init__(top_n)
        self.weights = dict(self.WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._idf: Dict[str, float] = {}
        self._last_contributions: List[Dict[str, float]] = []

    # -- 语料统计（IDF）---------------------------------------------------
    def fit(self, chunks: Sequence[Any]) -> "HeuristicReranker":
        """用全量语料统计 IDF（用于"IDF 加权覆盖"特征）。

        不调用 fit 也能工作（退化为等权覆盖），但 fit 后区分度明显更好。
        """
        documents = [c.text for c in chunks]
        total = max(1, len(documents))
        doc_freq: Dict[str, int] = {}
        for text in documents:
            for term in set(tokenize(text)):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        self._idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        _log.info("重排序器 IDF 统计完成：词表 %d（语料 %d 块）", len(self._idf), total)
        return self

    # -- 打分 -------------------------------------------------------------
    def _score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        """打分并记录绝对置信度。

        返回的是"绝对相关性置信度"（0~1），已做**相关性门控**：
        若候选与查询**没有任何词项交集**，即使长度先验等辅助特征得分不低，
        置信度也会被压到接近 0。这一步是降级判定能正常工作的前提 ——
        否则任何长度适中的无关段落都会拿到 ~0.3 的分数而逃过低相关性检查。

        相关公式：``confidence = sigmoid(Σw·f - 1.6) × gate``，
        其中 ``gate = 0.05 + 0.95 × √coverage``。
        """
        query_terms = tokenize_query(query)
        query_set = list(dict.fromkeys(query_terms))
        query_lower = query.lower()
        numeric_terms = {t.lower() for t in _RE_NUMERIC_TERM.findall(query) if len(t) > 1}
        max_idf = max(self._idf.values()) if self._idf else 1.0
        contributions: List[Dict[str, float]] = []
        scores: List[float] = []

        for document in documents:
            doc_lower = document.lower()
            doc_tokens = tokenize(document)
            doc_token_set = set(doc_tokens)

            features = self._features(
                document=document,
                doc_lower=doc_lower,
                doc_token_set=doc_token_set,
                query=query,
                query_lower=query_lower,
                query_set=query_set,
                numeric_terms=numeric_terms,
                max_idf=max_idf,
            )
            contributions.append(features)
            weighted = sum(self.weights[name] * value for name, value in features.items())
            raw = 1.0 / (1.0 + math.exp(-(weighted - 1.6)))
            coverage = features.get("coverage", 0.0)
            # 相关性门控：无词项交集 → 置信度趋近 0
            gate = 0.05 + 0.95 * math.sqrt(max(0.0, min(1.0, coverage)))
            scores.append(raw * gate)

        self._last_contributions = contributions
        return scores

    def _features(
        self,
        document: str,
        doc_lower: str,
        doc_token_set: set,
        query: str,
        query_lower: str,
        query_set: Sequence[str],
        numeric_terms: set,
        max_idf: float,
    ) -> Dict[str, float]:
        """计算六个特征（每个都归一化到 0~1）。"""
        features: Dict[str, float] = {}
        if not query_set:
            return {name: 0.0 for name in self.WEIGHTS}

        # 1) 词项覆盖率
        hits = [t for t in query_set if t in doc_token_set]
        coverage = len(hits) / len(query_set)
        features["coverage"] = coverage

        # 2) IDF 加权覆盖（归一化到 0~1）
        if self._idf:
            total_idf = sum(self._idf.get(t, max_idf * 0.5) for t in query_set)
            hit_idf = sum(self._idf.get(t, max_idf * 0.5) for t in hits)
            features["idf_coverage"] = hit_idf / total_idf if total_idf > 0 else 0.0
        else:
            features["idf_coverage"] = coverage

        # 3) 邻近度：命中词项在原文档中的最小跨度
        features["proximity"] = self._proximity(document, hits)

        # 4) 标题/来源命中（文档首行常为标题或章节名）
        head = document[:120].lower()
        head_hits = sum(1 for t in query_set if t in head and len(t) > 1)
        features["title_hit"] = min(1.0, head_hits / max(1, min(4, len(query_set))))

        # 5) 术语精确匹配（数字 + 专有名词），这是科研问答的关键信号
        if numeric_terms:
            matched = sum(1 for t in numeric_terms if t in doc_lower)
            features["term_match"] = matched / len(numeric_terms)
        else:
            # 无数字术语时，用"连续短语命中"替代（query 的 2-gram 是否原样出现）
            features["term_match"] = self._phrase_hit_ratio(query_lower, doc_lower)

        # 6) 长度先验：理想区间 150~900 字符
        size = len(document)
        if size < 80:
            features["length_prior"] = 0.2
        elif size < 150:
            features["length_prior"] = 0.6
        elif size <= 900:
            features["length_prior"] = 1.0
        elif size <= 1600:
            features["length_prior"] = 0.75
        else:
            features["length_prior"] = 0.5
        return features

    @staticmethod
    def _proximity(document: str, hits: Sequence[str]) -> float:
        """命中词项的最小字符跨度 → 归一化邻近度分数。

        所有命中词都挤在 30 字符内 → 1.0；分散在全文 → 接近 0。
        """
        if len(hits) < 2:
            return 0.5 if hits else 0.0
        positions: List[int] = []
        lowered = document.lower()
        for term in hits:
            pos = lowered.find(term.lower())
            if pos >= 0:
                positions.append(pos)
        if len(positions) < 2:
            return 0.3
        positions.sort()
        span = positions[-1] - positions[0]
        # 跨度 0 → 1.0；跨度 400+ → 接近 0
        return math.exp(-span / 260.0)

    @staticmethod
    def _phrase_hit_ratio(query_lower: str, doc_lower: str) -> float:
        """查询清洗后的连续片段在文档中出现的比例。"""
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", query_lower).strip()
        phrases = [p for p in cleaned.split() if len(p) > 3]
        if not phrases:
            return 0.0
        return sum(1 for p in phrases if p in doc_lower) / len(phrases)

    # -- 可解释性 ---------------------------------------------------------
    @property
    def last_contributions(self) -> List[Dict[str, float]]:
        """最近一次重排中每个候选的特征明细（Bad Case 分析用）。"""
        return self._last_contributions

    def explain(self, query: str, candidates: Sequence[RetrievedChunk]) -> List[Dict[str, Any]]:
        """输出可读的特征解释表。"""
        scores = self._score_pairs(query, [c.text for c in candidates])
        rows: List[Dict[str, Any]] = []
        for hit, features, confidence in zip(candidates, self._last_contributions, scores):
            weighted = sum(self.weights[k] * v for k, v in features.items())
            rows.append(
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "source": hit.chunk.source,
                    "page": hit.chunk.page,
                    "features": {k: round(v, 4) for k, v in features.items()},
                    "weighted_sum": round(weighted, 4),
                    "gate": round(
                        0.05 + 0.95 * math.sqrt(max(0.0, min(1.0, features.get("coverage", 0.0)))), 4
                    ),
                    "confidence": round(confidence, 4),
                }
            )
        return sorted(rows, key=lambda r: -r["confidence"])

    def to_confidence(self, raw_score: float) -> float:
        """启发式打分器的输出已经是 0~1 的置信度（含门控），直接返还。"""
        return float(min(1.0, max(0.0, raw_score)))


# ==============================================================================
# Ollama Cross-Encoder
# ==============================================================================
class OllamaReranker(BaseReranker):
    """通过 Ollama 调用 bge-reranker（需 ``ollama pull bge-reranker-base``）。

    使用 Ollama 的 ``/api/embed`` 对 "query</s>document" 拼接串编码，
    取相似度作为相关性分数（bge-reranker 的标准用法）。
    """

    provider = "ollama"

    def __init__(self, model: str = "bge-reranker-base", base_url: str = "http://127.0.0.1:11434",
                 top_n: int = 5, timeout: float = 60.0) -> None:
        super().__init__(top_n)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        import json
        import urllib.request

        payload = {"model": self.model, "input": list(texts)}
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        vectors = np.asarray(body.get("embeddings") or [], dtype=np.float32)
        if vectors.size == 0:
            raise RuntimeError(f"Ollama reranker 模型 {self.model} 返回为空，请先 ollama pull")
        return vectors

    def _score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        pairs = [f"{query}</s>{doc}" for doc in documents]
        vectors = self._embed(pairs)
        # bge-reranker 输出单向量，用与 query 部分的相似度近似；此处直接用模长归一化后的
        # query 向量与 pair 向量的点积作为相关性分数
        query_vector = self._embed([query])[0]
        norms = np.linalg.norm(vectors, axis=1) * (np.linalg.norm(query_vector) + 1e-12)
        norms[norms == 0] = 1e-12
        return [float(s) for s in (vectors @ query_vector) / norms]

    def to_confidence(self, raw_score: float) -> float:
        """余弦相似度范围约 -1~1，线性映射到 0~1 作为绝对置信度。"""
        return float(min(1.0, max(0.0, (raw_score + 1.0) / 2.0)))


# ==============================================================================
# 本地 CrossEncoder
# ==============================================================================
class LocalReranker(BaseReranker):
    """本地 sentence-transformers CrossEncoder（需 ``pip install sentence-transformers``）。"""

    provider = "local"

    def __init__(self, model: str = "BAAI/bge-reranker-base", top_n: int = 5,
                 device: str = "cpu", max_length: int = 512) -> None:
        super().__init__(top_n)
        self.model_name = model
        self.device = device
        self.max_length = max_length
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 sentence-transformers，请执行 `pip install sentence-transformers`，"
                "或把 reranker.provider 改为 heuristic"
            ) from exc
        self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        _log.info("CrossEncoder 重排序模型已加载: %s", self.model_name)
        return self._model

    def _score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        model = self._load()
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in np.asarray(scores).reshape(-1)]

    def to_confidence(self, raw_score: float) -> float:
        """CrossEncoder 输出是未归一化 logits，用 sigmoid 压到 0~1。"""
        try:
            return float(1.0 / (1.0 + math.exp(-raw_score)))
        except OverflowError:
            return 0.0 if raw_score < 0 else 1.0


# ==============================================================================
# 工厂
# ==============================================================================
RERANKER_PROVIDERS: Dict[str, type] = {
    "heuristic": HeuristicReranker,
    "none": type("_NoopReranker", (BaseReranker,), {
        "provider": "none",
        "_score_pairs": lambda self, q, docs: [1.0] * len(docs),
    }),
    "ollama": OllamaReranker,
    "local": LocalReranker,
    "cross-encoder": LocalReranker,
}


def create_reranker(
    config: Any = None,
    provider: Optional[str] = None,
    corpus: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> Optional[BaseReranker]:
    """按配置创建重排序器。

    Parameters
    ----------
    config: 全局配置对象（None 表示使用进程级单例配置）。
            **必须是第一个参数** —— 与其他工厂函数（create_embedder /
            create_vector_store / create_retrieval_engine）保持一致的
            "配置优先"签名，避免 ``create_reranker(cfg)`` 把配置对象误传给
            ``provider`` 这类难以定位的错误。
    provider: 覆盖配置中的 ``reranker.provider``；``none`` 表示关闭重排序
    corpus: 传入全量 chunk 会用其统计 IDF，显著提升启发式重排区分度

    Examples
    --------
    >>> create_reranker(cfg)                    # 用配置
    >>> create_reranker(cfg, provider="none")   # 关闭重排序
    >>> create_reranker(cfg, corpus=chunks)     # 用语料拟合 IDF
    """
    from ..utils.config import get_config

    cfg = config or get_config()
    name = (provider or str(cfg.get("reranker.provider", "heuristic")) or "heuristic").lower()
    if name == "none":
        _log.info("重排序已关闭（reranker.provider=none）")
        return None

    # 优先读 reranker.top_n；未配置时回退到 retrieval.top_k。
    # 早期直接读 retrieval.top_k，导致 config.yaml 里的 reranker.top_n 是死配置，
    # 改它没有任何效果。
    top_n = int(kwargs.pop("top_n", cfg.get("reranker.top_n", cfg.retrieval.top_k)))
    cls = RERANKER_PROVIDERS.get(name)
    if cls is None:
        _log.warning("未知 reranker.provider=%s，回退 heuristic", name)
        cls = HeuristicReranker

    try:
        if cls is HeuristicReranker:
            reranker = HeuristicReranker(top_n=top_n, **kwargs)
            if corpus:
                reranker.fit(corpus)
            return reranker
        if cls is OllamaReranker:
            return OllamaReranker(
                model=str(cfg.get("reranker.model", "bge-reranker-base")),
                base_url=str(cfg.get("reranker.base_url", "http://127.0.0.1:11434")),
                top_n=top_n,
                **kwargs,
            )
        if cls is LocalReranker:
            return LocalReranker(
                model=str(cfg.get("reranker.model", "BAAI/bge-reranker-base")),
                top_n=top_n,
                **kwargs,
            )
        return cls(top_n=top_n, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.warning("重排序器 %s 初始化失败（%s），回退 heuristic", name, exc)
        reranker = HeuristicReranker(top_n=top_n)
        if corpus:
            reranker.fit(corpus)
        return reranker
