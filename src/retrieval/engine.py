"""检索层统一门面（模块一的总装）。

把四个组件组装成一条完整链路，对上提供稳定 API：

.. code-block:: text

    LoadedDocument ──[chunker]──> List[Chunk]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
             [embedder 批量向量化]            [BM25 索引构建]
                    ▼                               ▼
              [向量库 add/delete]          [BM25Retriever.index]
                    └───────────────┬───────────────┘
                                    ▼
                    [HybridRetriever: 向量 ‖ BM25 → RRF]
                                    ▼
                        [Reranker Top-20 → Top-5]

面向业务的能力
--------------
- ``index_document()``  单文档索引入库（加载 → 分块 → 向量化 → 落库），支持增量
- ``index_documents()`` 批量导入 + 进度回调
- ``retrieve()``        检索（mode 可切 vector / hybrid / full，用于三档对比实验）
- ``delete_document()`` 按文档删除（向量库与 BM25 同步）
- ``evaluate()``        检索质量评估（Hit@K / MRR / Recall@K / NDCG@K）
"""

from __future__ import annotations

import functools
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..chunking import create_chunker
from ..embeddings import create_embedder
from ..utils.logger import get_logger
from ..utils.storage import read_json, write_json
from ..utils.types import Chunk, LoadedDocument, RetrievedChunk
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .reranker import create_reranker
from .vector_store import BaseVectorStore, create_vector_store

__all__ = ["RetrievalEngine", "IndexReport", "EvalMetrics", "create_retrieval_engine"]

_log = get_logger("retrieval.engine")


# ==============================================================================
# 数据结构
# ==============================================================================
class IndexReport(dict):
    """索引进度/结果报告（dict 子类，可直接序列化给前端）。"""


class EvalMetrics(dict):
    """检索评估指标（dict 子类）。"""


def _hit_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    gold = set(gold_ids)
    return 1.0 if any(cid in gold for cid in ranked_ids[:k]) else 0.0


def _mrr(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def _recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return len(gold & set(ranked_ids[:k])) / len(gold)


def _precision_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """P@k：分母恒为 k（标准定义）。

    早期实现除以 ``min(k, len(ranked_ids))``，在返回条数少于 k 时会**虚高** ——
    例如只返回 2 条且都相关，会得到 1.0 而不是 2/k。评测中返回条数不足是常态
    （配置了 min_score、语料本身就小），所以这个偏差会系统性抬高精度指标，也让
    "检索层是否真的返回了 k 条"这个问题在指标里彻底不可见。
    """
    gold = set(gold_ids)
    if k <= 0:
        return 0.0
    return len(gold & set(ranked_ids[:k])) / k


def _ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    gold = set(gold_ids)
    dcg = 0.0
    for idx, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            dcg += 1.0 / np.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    """把对索引有读写的方法串行化（``self._lock`` 是可重入锁）。

    HTTP 服务是 ThreadingHTTPServer，上传/删除与检索天然并发。锁必须在
    **检索引擎**这一层，因为它是唯一同时持有 ``_chunks``、BM25 与向量库的对象；
    只在 HTTP 层加锁挡不住同一进程内的其它调用方（CLI、评测脚本、后台任务）。
    """

    @functools.wraps(method)
    def wrapper(self: "RetrievalEngine", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


# ==============================================================================
# 检索引擎
# ==============================================================================
class RetrievalEngine:
    """检索层总装门面。

    典型用法::

        engine = create_retrieval_engine()
        engine.index_documents(documents)
        hits = engine.retrieve("Transformer 用了什么数据集？", mode="full")
        engine.evaluate(eval_set)      # Hit@5 / MRR ...
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        bm25: Optional[BM25Retriever] = None,
        retriever: Optional[HybridRetriever] = None,
        embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
        config: Any = None,
        index_dir: Optional[Any] = None,
    ) -> None:
        from ..utils.config import get_config, PROJECT_ROOT

        self.config = config or get_config()
        self.embedder = embedder or create_embedder(self.config)
        self.vector_store = vector_store or create_vector_store(
            self.config, dimension=self.embedder.dimension
        )
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()
        self.reranker = reranker
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            bm25=self.bm25,
            reranker=self.reranker,
            embedder=self.embedder,
            config=self.config,
        )
        # 索引目录：存 chunk 清单与文档清单（BM25 需要，且便于删除/重建）
        base = Path(index_dir) if index_dir else self.config.resolve_path(str(self.config.paths.index_dir))
        self.index_dir = Path(base)
        self.chunks_path = self.index_dir / "chunks.json"
        self.documents_path = self.index_dir / "documents.json"
        self._chunks: List[Chunk] = []
        self._documents: Dict[str, Dict[str, Any]] = {}
        # 可重入锁：HTTP 服务是多线程的（ThreadingHTTPServer），上传/删除与
        # 检索会并发访问 _chunks / BM25 / 向量库。没有这把锁时，一边重建 BM25
        # 一边检索会读到半成品索引，落盘也可能互相覆盖。
        self._lock = threading.RLock()
        # 批处理期间递增：>0 时把 BM25/重排序器的重建推迟到批末一次完成
        self._defer_sparse = 0
        self._load_manifests()

    # ------------------------------------------------------------------
    # 稀疏索引（BM25 + 重排序 IDF）维护
    # ------------------------------------------------------------------
    def _rebuild_sparse_indexes(self) -> None:
        """按当前 ``_chunks`` 重建 BM25 与重排序器的 IDF 统计。"""
        self.bm25.index(self._chunks, rebuild=True)
        if self.reranker is not None and hasattr(self.reranker, "fit"):
            self.reranker.fit(self._chunks)

    @contextmanager
    def _deferred_sparse_rebuild(self) -> Iterator[None]:
        """把批内逐篇的稀疏索引重建合并为一次。

        每次 :meth:`index_document` 都 ``rebuild=True`` 会让批量导入退化成
        O(D²)：每加一篇文档就重新分词整个语料 + 重建稀疏矩阵 + 重估 IDF。
        实测 40 篇 × 50 片段逐篇添加 1.77s → 批末一次重建 0.10s（18×），
        且差距随文档数线性放大（1000 篇时约 250×）。
        """
        self._defer_sparse += 1
        try:
            yield
        finally:
            self._defer_sparse -= 1
            if self._defer_sparse == 0:
                self._rebuild_sparse_indexes()

    # ------------------------------------------------------------------
    # 清单持久化
    # ------------------------------------------------------------------
    def _load_manifests(self) -> None:
        """从磁盘恢复 chunk 清单与文档清单，并重建 BM25 索引。

        这一步是"重启不丢索引"的关键：向量库自己会加载向量，
        但 BM25 与文档元信息需要从这里恢复。
        """
        raw_chunks = read_json(self.chunks_path, default=[]) or []
        self._chunks = [Chunk.from_dict(item) for item in raw_chunks]
        self._documents = read_json(self.documents_path, default={}) or {}

        # 以向量库的实际内容为准（防止两份数据不一致）
        store_chunks = {c.chunk_id: c for c in self.vector_store.all_chunks()}
        if store_chunks:
            self._chunks = [store_chunks.get(c.chunk_id, c) for c in self._chunks if c.chunk_id in store_chunks]
            # 集合必须在循环外建好：写成内联推导式会让这里变成 O(n²) ——
            # 每遍历一个 store_chunk 都重建一次全量 id 集合。实测 1 万 chunk 的
            # 语料下这一步 6.06s → 1.9ms（3200×），而它每次构造检索引擎都会跑。
            manifest_ids = {x.chunk_id for x in self._chunks}
            missing = [c for cid, c in store_chunks.items() if cid not in manifest_ids]
            self._chunks.extend(missing)

        if self._chunks:
            self.bm25.index(self._chunks, rebuild=True)
            # 用全量语料拟合重排序器的 IDF 统计
            if self.reranker is not None and hasattr(self.reranker, "fit"):
                self.reranker.fit(self._chunks)
            _log.info(
                "检索索引已恢复：%d chunk / %d 文档",
                len(self._chunks),
                len({c.doc_id for c in self._chunks}),
            )

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.chunks_path, [c.to_dict() for c in self._chunks])
        write_json(self.documents_path, self._documents)
        self.vector_store.persist()

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    @_synchronized
    def index_document(
        self,
        document: LoadedDocument,
        strategy: Optional[str] = None,
        reindex: bool = True,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """把单个文档索引入库（增量：只对新 chunk 做向量化）。

        Parameters
        ----------
        strategy: 分块策略，默认取 ``chunking.strategy``
        reindex: 同一 doc_id 已存在时是否先删除旧数据（保证幂等）
        progress: 进度回调，收到 ``{"stage": ..., "percent": ..., "message": ...}``
        """
        began = time.perf_counter()

        def notify(stage: str, percent: float, message: str = "") -> None:
            if progress:
                try:
                    progress({"stage": stage, "percent": round(percent, 1), "message": message})
                except Exception as exc:  # noqa: BLE001
                    _log.warning("索引进度回调异常: %s", exc)

        strategy = strategy or str(self.config.chunking.strategy)
        notify("chunking", 5.0, f"正在分块（策略={strategy}）")

        chunker = create_chunker(self.config, strategy=strategy, embedder=self.embedder)
        chunks = chunker.split(document)
        if not chunks:
            return {
                "doc_id": document.doc_id,
                "ok": False,
                "error": "分块结果为空（文档可能没有可提取文本）",
                "num_chunks": 0,
            }

        if reindex:
            removed = self.delete_document(document.doc_id, persist=False)
            if removed:
                _log.info("重新索引：先移除旧数据 %d 条", removed)

        notify("embedding", 25.0, f"正在向量化 {len(chunks)} 个片段")

        def vector_progress(done: int, total: int) -> None:
            notify("embedding", 25.0 + 55.0 * (done / max(1, total)), f"向量化 {done}/{total}")

        embeddings = self.embedder.encode_batch(
            [c.to_embed_text() for c in chunks], progress=vector_progress
        )

        notify("storing", 85.0, "正在写入向量库")
        self.vector_store.add(chunks, embeddings)
        self._chunks.extend(chunks)
        if self._defer_sparse > 0:
            pass  # 批处理：留给 _deferred_sparse_rebuild 在批末重建一次
        else:
            self._rebuild_sparse_indexes()

        notify("persisting", 95.0, "正在落盘")
        self._documents[document.doc_id] = {
            **document.meta.to_dict(),
            "status": "indexed",
            "num_chunks": len(chunks),
            "strategy": strategy,
            "indexed_at": time.time(),
            "text_chars": len(document.text),
        }
        self._persist()
        notify("done", 100.0, f"索引完成：{len(chunks)} 个片段")

        return {
            "doc_id": document.doc_id,
            "ok": True,
            "num_chunks": len(chunks),
            "num_vectors": len(chunks),
            "strategy": strategy,
            "title": document.meta.title,
            "latency_ms": round((time.perf_counter() - began) * 1000, 1),
            "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / len(chunks), 1),
        }

    @_synchronized
    def index_documents(
        self,
        documents: Iterable[LoadedDocument],
        strategy: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """批量索引多个文档。"""
        items = list(documents)
        reports: List[Dict[str, Any]] = []
        began = time.perf_counter()
        with self._deferred_sparse_rebuild():
            for idx, document in enumerate(items, start=1):
                if on_progress:
                    on_progress(idx, len(items), document.meta.title)
                reports.append(self.index_document(document, strategy=strategy))
        succeeded = sum(1 for r in reports if r.get("ok"))
        return {
            "num_documents": len(items),
            "succeeded": succeeded,
            "failed": len(items) - succeeded,
            "num_chunks": sum(int(r.get("num_chunks", 0)) for r in reports),
            "latency_ms": round((time.perf_counter() - began) * 1000, 1),
            "reports": reports,
        }

    @_synchronized
    def delete_document(self, doc_id: str, persist: bool = True) -> int:
        """删除某文档的全部索引数据（向量库 + BM25 + 清单）。"""
        removed_vectors = self.vector_store.delete(doc_id)
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        removed_chunks = before - len(self._chunks)
        if removed_chunks and self._defer_sparse == 0:
            self._rebuild_sparse_indexes()
        self._documents.pop(doc_id, None)
        if persist:
            self._persist()
        _log.info("删除文档 %s：移除 %d 向量 / %d 分块", doc_id, removed_vectors, removed_chunks)
        return max(removed_vectors, removed_chunks)

    @_synchronized
    def reindex_all(self, documents: Iterable[LoadedDocument], strategy: Optional[str] = None) -> Dict[str, Any]:
        """清空并全量重建索引（配置变更后使用）。"""
        self.clear()
        return self.index_documents(documents, strategy=strategy)

    @_synchronized
    def clear(self) -> None:
        """清空整个索引。"""
        self.vector_store.clear()
        self.bm25.clear()
        self._chunks = []
        self._documents = {}
        for path in (self.chunks_path, self.documents_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    @_synchronized
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        candidate_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """检索（mode: ``vector`` / ``hybrid`` / ``full``）。"""
        if not self._chunks:
            return []
        return self.retriever.retrieve(query, top_k=top_k, mode=mode, candidate_k=candidate_k)

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """检索并返回前端/日志所需的完整上下文。"""
        began = time.perf_counter()
        hits = self.retrieve(query, top_k=top_k, mode=mode)
        trace = self.retriever.last_trace
        return {
            "query": query,
            "mode": mode or "full",
            "num": len(hits),
            "results": [
                {
                    "index": idx,
                    "chunk_id": h.chunk.chunk_id,
                    "doc_id": h.chunk.doc_id,
                    "source": h.chunk.source,
                    "page": h.chunk.page,
                    # score = 相对展示分（Top-1 恒为 1.0）；confidence = 绝对相关度（前端应展示这个）
                    "score": round(h.score, 4),
                    "confidence": round(h.confidence, 4),
                    "vector_score": round(h.vector_score, 4),
                    "bm25_score": round(h.bm25_score, 4),
                    "rerank_score": round(h.rerank_score, 4),
                    "route": h.source_route,
                    "preview": h.text[:300],
                }
                for idx, h in enumerate(hits, start=1)
            ],
            "latency_ms": round((time.perf_counter() - began) * 1000, 2),
            "trace": trace.to_dict() if trace else None,
        }

    # ------------------------------------------------------------------
    # 评估（模块五的检索指标来源）
    # ------------------------------------------------------------------
    @_synchronized
    def evaluate(
        self,
        eval_set: Sequence[Dict[str, Any]],
        mode: str = "full",
        top_k: int = 5,
        match: str = "chunk_id",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """在评测集上计算检索指标。

        Parameters
        ----------
        eval_set:
            评测样本列表，每条至少包含：

            - ``question``：问题
            - ``gold_chunk_ids``（match="chunk_id"）或 ``gold_texts``（match="text"）

            兼容字段名：``gold`` / ``relevant_chunk_ids`` / ``gold_answers``。
        mode: 检索档位（``vector`` / ``hybrid`` / ``full``）
        top_k: 计算 Hit@k / Recall@k 的 k
        match: 相关性判定方式

               - ``chunk_id``：金标准给出 chunk_id（精确）
               - ``text``：金标准给出参考文本片段，用字符重合度判定
               - ``doc_id``：金标准给出文档 ID（粗粒度）

        Returns
        -------
        含 Hit@k / MRR / Recall@k / Precision@k / NDCG@k 与逐条明细的字典。
        """
        hits_at_k: List[float] = []
        mrrs: List[float] = []
        recalls: List[float] = []
        precisions: List[float] = []
        ndcgs: List[float] = []
        latencies: List[float] = []
        details: List[Dict[str, Any]] = []

        total = len(eval_set)
        for idx, sample in enumerate(eval_set, start=1):
            question = str(sample.get("question") or sample.get("query") or "")
            if not question:
                continue
            began = time.perf_counter()
            retrieved = self.retrieve(question, top_k=max(top_k, 10), mode=mode)
            latency = (time.perf_counter() - began) * 1000
            latencies.append(latency)

            ranked_ids = self._resolve_ranked_ids(retrieved, sample, match)
            gold_ids = self._resolve_gold_ids(sample, match, retrieved)

            h = _hit_at_k(ranked_ids, gold_ids, top_k)
            m = _mrr(ranked_ids, gold_ids)
            r = _recall_at_k(ranked_ids, gold_ids, top_k)
            p = _precision_at_k(ranked_ids, gold_ids, top_k)
            n = _ndcg_at_k(ranked_ids, gold_ids, top_k)

            hits_at_k.append(h)
            mrrs.append(m)
            recalls.append(r)
            precisions.append(p)
            ndcgs.append(n)
            details.append(
                {
                    "question": question,
                    "hit": bool(h),
                    "mrr": round(m, 4),
                    "recall": round(r, 4),
                    "ndcg": round(n, 4),
                    "retrieved_ids": ranked_ids[:top_k],
                    "gold_ids": list(gold_ids)[:10],
                    "latency_ms": round(latency, 1),
                    "top1_score": round(retrieved[0].score, 4) if retrieved else 0.0,
                }
            )
            if on_progress and (idx % 10 == 0 or idx == total):
                on_progress(idx, total)

        n_samples = max(1, len(hits_at_k))
        metrics = EvalMetrics(
            mode=mode,
            top_k=top_k,
            match=match,
            num_samples=len(hits_at_k),
            hit_at_k=round(sum(hits_at_k) / n_samples, 4),
            mrr=round(sum(mrrs) / n_samples, 4),
            recall_at_k=round(sum(recalls) / n_samples, 4),
            precision_at_k=round(sum(precisions) / n_samples, 4),
            ndcg_at_k=round(sum(ndcgs) / n_samples, 4),
            avg_latency_ms=round(sum(latencies) / n_samples, 2),
            p95_latency_ms=round(
                float(np.percentile(latencies, 95)) if latencies else 0.0, 2
            ),
        )
        metrics["details"] = details
        return metrics

    def compare_modes(
        self,
        eval_set: Sequence[Dict[str, Any]],
        modes: Sequence[str] = ("vector", "hybrid", "full"),
        top_k: int = 5,
        match: str = "chunk_id",
    ) -> Dict[str, Any]:
        """三档检索对比实验（课程要求 5.1.4 的产出）。

        对应"纯向量 → 混合检索 → 混合+重排序"三档，输出可直接绘图的对比表。
        """
        table: List[Dict[str, Any]] = []
        for mode in modes:
            _log.info("开始评估检索档位: %s", mode)
            metrics = self.evaluate(eval_set, mode=mode, top_k=top_k, match=match)
            table.append(
                {
                    "mode": mode,
                    "hit@k": metrics["hit_at_k"],
                    "mrr": metrics["mrr"],
                    "recall@k": metrics["recall_at_k"],
                    "precision@k": metrics["precision_at_k"],
                    "ndcg@k": metrics["ndcg_at_k"],
                    "avg_latency_ms": metrics["avg_latency_ms"],
                    "p95_latency_ms": metrics["p95_latency_ms"],
                }
            )

        baseline = table[0] if table else {}
        for row in table:
            row["hit_gain_vs_vector"] = round(row["hit@k"] - baseline.get("hit@k", 0.0), 4)
            row["mrr_gain_vs_vector"] = round(row["mrr"] - baseline.get("mrr", 0.0), 4)
            if baseline.get("avg_latency_ms"):
                row["latency_ratio"] = round(row["avg_latency_ms"] / baseline["avg_latency_ms"], 2)
        return {"top_k": top_k, "num_samples": len(eval_set), "table": table}

    # -- 评估内部工具 -----------------------------------------------------
    @staticmethod
    def _resolve_ranked_ids(
        retrieved: Sequence[RetrievedChunk],
        sample: Dict[str, Any],
        match: str,
    ) -> List[str]:
        """把检索结果映射为用于比对的 ID 序列。"""
        if match == "doc_id":
            return [h.chunk.doc_id for h in retrieved]
        return [h.chunk.chunk_id for h in retrieved]

    def _resolve_gold_ids(
        self,
        sample: Dict[str, Any],
        match: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> List[str]:
        """解析金标准 ID；支持 chunk_id / doc_id / 文本三种形式。"""
        raw = (
            sample.get("gold_chunk_ids")
            or sample.get("relevant_chunk_ids")
            or sample.get("gold_ids")
            or sample.get("gold")
            or []
        )
        if isinstance(raw, str):
            raw = [raw]

        if match == "text":
            return self._match_texts(
                list(raw) + list(sample.get("gold_texts") or []) + list(sample.get("gold_answers") or []),
                retrieved,
            )

        ids = [str(item) for item in raw]
        if match == "doc_id":
            return ids
        # chunk_id 模式下，若金标准给的是 doc_id，则把所有属于该文档的 chunk 视为相关
        if ids and not any(c.chunk_id in ids for c in retrieved):
            expanded: List[str] = []
            for hit in retrieved:
                if hit.chunk.doc_id in ids or hit.chunk.source in ids:
                    expanded.append(hit.chunk.chunk_id)
            if expanded:
                return expanded
        return ids

    @staticmethod
    def _match_texts(gold_texts: Sequence[str], retrieved: Sequence[RetrievedChunk]) -> List[str]:
        """文本金标准 → 命中 chunk 的 ID 列表。

        判定方式：金标准文本被检索片段包含，或与之的字符 n-gram 重合度 ≥ 0.5。
        """
        matched: List[str] = []

        def ngrams(text: str, n: int = 4) -> set:
            cleaned = "".join(ch for ch in text if ch.strip())
            return {cleaned[i : i + n] for i in range(max(0, len(cleaned) - n + 1))}

        for gold in gold_texts:
            gold_clean = str(gold).strip()
            if not gold_clean:
                continue
            gold_grams = ngrams(gold_clean)
            for hit in retrieved:
                text = hit.text
                if gold_clean in text:
                    matched.append(hit.chunk.chunk_id)
                    continue
                if gold_grams:
                    overlap = len(gold_grams & ngrams(text)) / len(gold_grams)
                    if overlap >= 0.5:
                        matched.append(hit.chunk.chunk_id)
        # 保序去重
        return list(dict.fromkeys(matched))

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    @property
    def chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def documents(self) -> List[Dict[str, Any]]:
        """返回文档清单（前端知识库面板消费）。"""
        rows: List[Dict[str, Any]] = []
        for doc_id, meta in self._documents.items():
            row = dict(meta)
            row["doc_id"] = doc_id
            row.setdefault("status", "indexed")
            row.setdefault("num_vectors", row.get("num_chunks", 0))
            rows.append(row)
        rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        return rows

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        meta = self._documents.get(doc_id)
        if meta is None:
            return None
        return {**meta, "doc_id": doc_id}

    def stats(self) -> Dict[str, Any]:
        return {
            "num_documents": self.num_documents,
            "num_chunks": self.num_chunks,
            "embedding": {
                "provider": self.embedder.provider,
                "model": self.embedder.model,
                "dimension": self.embedder.dimension,
            },
            "vector_store": self.vector_store.stats(),
            "bm25": self.bm25.stats(),
            "reranker": self.reranker.stats() if self.reranker else None,
            "chunking": {
                "strategy": str(self.config.chunking.strategy),
                "chunk_size": int(self.config.chunking.chunk_size),
                "chunk_overlap": int(self.config.chunking.chunk_overlap),
            },
            "retrieval": {
                "top_k": self.retriever.top_k,
                "candidate_k": self.retriever.candidate_k,
                "rrf_k": self.retriever.rrf_k,
                "enable_bm25": self.retriever.enable_bm25,
                "enable_rerank": self.retriever.enable_rerank,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "document_count": self.num_documents,
            "chunk_count": self.num_chunks,
            "vector_store": self.vector_store.health_check(),
            "bm25": self.bm25.health_check(),
            "embedding": self.embedder.health_check(),
            "reranker": self.reranker.health_check() if self.reranker else {"ok": True, "enabled": False},
        }


# ==============================================================================
# 工厂
# ==============================================================================
def create_retrieval_engine(
    config: Any = None,
    embedder: Optional[Any] = None,
    reranker: Optional[Any] = None,
    vector_store: Optional[BaseVectorStore] = None,
    chunk_corpus: Optional[Sequence[Chunk]] = None,
    **kwargs: Any,
) -> RetrievalEngine:
    """按配置组装完整的检索引擎。

    Parameters
    ----------
    chunk_corpus: 预先存在的 chunk 语料（用于拟合重排序器 IDF）
    """
    from ..utils.config import get_config

    cfg = config or get_config()
    embedder = embedder or create_embedder(cfg)
    store = vector_store or create_vector_store(cfg, dimension=embedder.dimension)
    bm25 = BM25Retriever(
        k1=float(cfg.get("retrieval.bm25_k1", 1.5)),
        b=float(cfg.get("retrieval.bm25_b", 0.75)),
    )
    reranker = reranker if reranker is not None else create_reranker(cfg, corpus=chunk_corpus)
    return RetrievalEngine(
        vector_store=store,
        bm25=bm25,
        embedder=embedder,
        reranker=reranker,
        config=cfg,
        **kwargs,
    )
