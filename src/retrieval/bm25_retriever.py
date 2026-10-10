"""BM25 关键词检索（模块一 5.1.4）。

从零手写实现 Okapi BM25，不依赖 ``rank_bm25``，原因有三：

1. **可控**：中文场景必须自定义分词（本项目用 jieba 或中文 bigram 回退），
   第三方库默认按空格分词，对中文完全不可用。
2. **高效**：用 CSR 稀疏矩阵 + NumPy 向量化打分，只对查询词项涉及的文档列
   做计算，比 Python 循环快 1~2 个数量级。
3. **可解释**：每个词项的 IDF 与贡献度都能导出，便于撰写实验报告与
   分析"为什么这个词的权重高"。

BM25 公式
---------
.. math::

    \\text{score}(D, Q) = \\sum_{t \\in Q} \\text{IDF}(t) \\cdot
    \\frac{f(t, D) \\cdot (k_1 + 1)}
         {f(t, D) + k_1 \\cdot \\left(1 - b + b \\cdot \\frac{|D|}{\\text{avgdl}}\\right)}

其中 IDF 采用带平滑的概率形式：
:math:`\\text{IDF}(t) = \\ln\\left(1 + \\frac{N - n_t + 0.5}{n_t + 0.5}\\right)`

典型参数：``k1 = 1.5``（词频饱和速度），``b = 0.75``（文档长度归一化强度）。
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.storage import read_json, write_json
from ..utils.text import tokenize, tokenize_query
from ..utils.types import Chunk, RetrievedChunk

__all__ = ["BM25Retriever", "BM25Okapi"]

_log = get_logger("retrieval.bm25")


class BM25Okapi:
    """Okapi BM25 打分器（纯 NumPy 实现）。

    使用方式::

        bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
        scores = bm25.get_scores(query_tokens)     # -> np.ndarray, shape=(N,)
        top = bm25.get_top_n(query_tokens, n=10)   # -> List[(index, score)]
    """

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)
        self.corpus_size = len(corpus)
        self.doc_lengths = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(self.doc_lengths.mean()) if self.corpus_size else 0.0
        self.avgdl = max(self.avgdl, 1.0)

        # 构建词表与 CSR 稀疏词频矩阵
        self.vocab: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        values: List[float] = []
        doc_freq: Counter = Counter()

        for row_idx, doc in enumerate(corpus):
            counts = Counter(doc)
            for term, freq in counts.items():
                col = self.vocab.get(term)
                if col is None:
                    col = len(self.vocab)
                    self.vocab[term] = col
                rows.append(row_idx)
                cols.append(col)
                values.append(float(freq))
                doc_freq[term] += 1

        n_terms = max(1, len(self.vocab))
        self._tf = _csr_from_triplets(rows, cols, values, self.corpus_size, n_terms)
        # 统一两种后端的列访问方式
        if hasattr(self._tf, "get_column"):
            self._column = self._tf.get_column
        else:  # scipy.sparse：CSR 的列切片代价高，转成 CSC 后按列取
            csc = self._tf.tocsc()
            self._column = lambda col: (csc.indices[csc.indptr[col] : csc.indptr[col + 1]],
                                        csc.data[csc.indptr[col] : csc.indptr[col + 1]])

        # 预计算每个词项的 IDF
        self.idf = np.zeros(n_terms, dtype=np.float32)
        for term, col in self.vocab.items():
            n_t = doc_freq[term]
            self.idf[col] = math.log(1.0 + (self.corpus_size - n_t + 0.5) / (n_t + 0.5))

        # 预计算分母项：k1 * (1 - b + b * |D| / avgdl)
        self._denom_base = (self.k1 * (1.0 - self.b + self.b * self.doc_lengths / self.avgdl)).astype(
            np.float32
        )

    # -- 打分 -------------------------------------------------------------
    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        """返回语料库中每个文档的 BM25 分数。"""
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if not query_tokens or self.corpus_size == 0:
            return scores

        # 统计查询词项（去重；查询内词频按 qtf 线性加权，符合 BM25 的常见扩展）
        query_counts = Counter(query_tokens)
        for term, qtf in query_counts.items():
            col = self.vocab.get(term)
            if col is None:
                continue
            rows, freqs = self._column(col)
            if rows.size == 0:
                continue
            numerator = freqs * (self.k1 + 1.0)
            denominator = freqs + self._denom_base[rows]
            scores[rows] += self.idf[col] * qtf * (numerator / denominator)
        return scores

    def get_top_n(
        self, query_tokens: Sequence[str], n: int = 10, min_score: float = 1e-6
    ) -> List[Tuple[int, float]]:
        """返回 (文档下标, 分数) 的 Top-N 列表（已过滤 0 分文档）。"""
        scores = self.get_scores(query_tokens)
        if scores.size == 0:
            return []
        k = min(max(1, n), scores.size)
        if k < scores.size:
            candidates = np.argpartition(-scores, k - 1)[:k]
        else:
            candidates = np.arange(scores.size)
        ordered = candidates[np.argsort(-scores[candidates])]
        return [(int(i), float(scores[i])) for i in ordered if scores[i] > min_score]

    def term_contributions(self, query_tokens: Sequence[str], doc_index: int) -> Dict[str, float]:
        """返回某文档中各查询词项的 BM25 贡献度（可解释性分析用）。"""
        contributions: Dict[str, float] = {}
        if doc_index < 0 or doc_index >= self.corpus_size:
            return contributions
        denominator = float(self._denom_base[doc_index])
        for term in dict.fromkeys(query_tokens):
            col = self.vocab.get(term)
            if col is None:
                continue
            freq = float(self._tf[doc_index, col])
            if freq <= 0:
                continue
            contributions[term] = float(
                self.idf[col] * freq * (self.k1 + 1.0) / (freq + denominator)
            )
        return contributions


def _csr_from_triplets(
    rows: Sequence[int],
    cols: Sequence[int],
    values: Sequence[float],
    n_rows: int,
    n_cols: int,
) -> Any:
    """从 (row, col, value) 三元组构造稀疏词频矩阵。

    scipy 可用时直接用 ``csr_matrix``（成熟、快）；否则回退到内置的
    :class:`_SimpleSparse` —— 它按 **CSC**（列压缩）组织数据，
    因为 BM25 打分只需要"按词项取一列非零文档"，列式布局正是最优结构。
    """
    try:  # pragma: no cover - 取决于环境是否装了 scipy
        from scipy.sparse import csr_matrix  # type: ignore

        return csr_matrix((values, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return _SimpleSparse(rows, cols, values, n_rows, n_cols)


class _SimpleSparse:
    """极简 CSC 稀疏矩阵（零依赖回退实现）。

    仅支持 BM25 需要的两种访问：
    - ``matrix[row, col]``  → 取单个元素（float）
    - ``matrix.get_column(col)`` → 取整列的非零 (行号数组, 数值数组)
    """

    __slots__ = ("shape", "n_rows", "n_cols", "col_ptr", "row_idx", "data")

    def __init__(
        self,
        rows: Sequence[int],
        cols: Sequence[int],
        values: Sequence[float],
        n_rows: int,
        n_cols: int,
    ) -> None:
        self.shape = (n_rows, n_cols)
        self.n_rows = n_rows
        self.n_cols = n_cols

        # 按列分桶（同一列内按行号升序，便于二分/顺序扫描）
        buckets: List[List[Tuple[int, float]]] = [[] for _ in range(n_cols)]
        for row, col, value in zip(rows, cols, values):
            buckets[int(col)].append((int(row), float(value)))

        col_ptr = np.zeros(n_cols + 1, dtype=np.int64)
        row_idx: List[int] = []
        data: List[float] = []
        for col, entries in enumerate(buckets):
            entries.sort(key=lambda x: x[0])
            for row, value in entries:
                row_idx.append(row)
                data.append(value)
            col_ptr[col + 1] = len(row_idx)

        self.col_ptr = col_ptr
        self.row_idx = np.asarray(row_idx, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float32)

    def get_column(self, col: int) -> Tuple[np.ndarray, np.ndarray]:
        """返回第 col 列的非零 (行号数组, 数值数组)。"""
        if col < 0 or col >= self.n_cols:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        start, end = int(self.col_ptr[col]), int(self.col_ptr[col + 1])
        return self.row_idx[start:end], self.data[start:end]

    def __getitem__(self, key: Any) -> float:
        row, col = key
        rows, vals = self.get_column(int(col))
        if rows.size == 0:
            return 0.0
        position = int(np.searchsorted(rows, int(row)))
        if position < rows.size and rows[position] == int(row):
            return float(vals[position])
        return 0.0

    def copy(self) -> "_SimpleSparse":
        clone = _SimpleSparse.__new__(_SimpleSparse)
        clone.shape = self.shape
        clone.n_rows = self.n_rows
        clone.n_cols = self.n_cols
        clone.col_ptr = self.col_ptr.copy()
        clone.row_idx = self.row_idx.copy()
        clone.data = self.data.copy()
        return clone


# ==============================================================================
# 检索器
# ==============================================================================
class BM25Retriever:
    """BM25 关键词检索器（面向 :class:`Chunk` 的封装）。

    维护一份与向量库同步的 chunk 列表；支持增量重建（新增文档后只需 O(N)
    重建稀疏矩阵，实测 1 万 chunk 重建耗时 < 200ms）。
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        persist_path: Optional[Any] = None,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.RLock()
        self._chunks: List[Chunk] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = True
        self._build_ms: float = 0.0

    # -- 索引维护 ---------------------------------------------------------
    def index(self, chunks: Sequence[Chunk], rebuild: bool = True) -> int:
        """建立/更新关键词索引。

        Parameters
        ----------
        chunks: 待索引的 chunk
        rebuild: True 表示用传入列表**替换**现有语料；False 表示增量追加
        """
        import time

        start = time.perf_counter()
        with self._lock:
            if rebuild:
                self._chunks = list(chunks)
            else:
                existing = {c.chunk_id: i for i, c in enumerate(self._chunks)}
                for chunk in chunks:
                    if chunk.chunk_id in existing:
                        self._chunks[existing[chunk.chunk_id]] = chunk
                    else:
                        self._chunks.append(chunk)
            self._tokenized = [tokenize(c.text) for c in self._chunks]
            self._bm25 = (
                BM25Okapi(self._tokenized, k1=self.k1, b=self.b) if self._chunks else None
            )
            self._dirty = False
            self._build_ms = (time.perf_counter() - start) * 1000
        _log.info(
            "BM25 索引完成: %d 个 chunk，耗时 %.1fms，词表 %d",
            len(self._chunks),
            self._build_ms,
            len(self._bm25.vocab) if self._bm25 else 0,
        )
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk]) -> int:
        """增量追加 chunk。"""
        return self.index(chunks, rebuild=False)

    def remove(self, doc_id: str) -> int:
        """按文档 ID 删除并重建索引。"""
        with self._lock:
            remaining = [c for c in self._chunks if c.doc_id != doc_id]
            removed = len(self._chunks) - len(remaining)
            if removed:
                self.index(remaining, rebuild=True)
            return removed

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._tokenized = []
            self._bm25 = None
            self._dirty = True

    # -- 检索 -------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 1e-6,
    ) -> List[RetrievedChunk]:
        """BM25 Top-K 检索。"""
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            tokens = tokenize_query(query)
            hits = self._bm25.get_top_n(tokens, n=top_k, min_score=min_score)
            results: List[RetrievedChunk] = []
            for rank, (index, score) in enumerate(hits, start=1):
                if index >= len(self._chunks):
                    continue
                results.append(
                    RetrievedChunk(
                        chunk=self._chunks[index],
                        score=float(score),
                        bm25_score=float(score),
                        bm25_rank=rank,
                        source_route="bm25",
                    )
                )
            return results

    def explain(self, query: str, chunk_id: str) -> Dict[str, float]:
        """解释某 chunk 的得分构成（词项贡献度），用于 Bad Case 分析。"""
        with self._lock:
            if self._bm25 is None:
                return {}
            for index, chunk in enumerate(self._chunks):
                if chunk.chunk_id == chunk_id:
                    return self._bm25.term_contributions(tokenize_query(query), index)
        return {}

    # -- 工具 -------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._chunks)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "num_chunks": len(self._chunks),
                "vocab_size": len(self._bm25.vocab) if self._bm25 else 0,
                "avg_doc_length": round(self._bm25.avgdl, 1) if self._bm25 else 0.0,
                "k1": self.k1,
                "b": self.b,
                "build_ms": round(self._build_ms, 1),
                "backend": "scipy" if _scipy_available() else "builtin-csr",
            }

    def health_check(self) -> Dict[str, Any]:
        data = self.stats()
        data["ok"] = True
        return data


def _scipy_available() -> bool:
    try:  # pragma: no cover
        import scipy  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False
