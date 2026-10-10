"""向量数据库封装（模块一 5.1.3）。

支持三种后端，统一接口 :class:`BaseVectorStore`：

============  ====================================================================
backend       说明
============  ====================================================================
``numpy``     纯 NumPy 内存矩阵 + 磁盘持久化。零依赖、可控、便于教学理解
              底层原理。默认后端。
``chroma``    ChromaDB 轻量级向量库（需 ``pip install chromadb``），
              自带持久化与元数据过滤。
``faiss``     FAISS 高性能向量检索（需 ``pip install faiss-cpu``），
              适合百万级向量。
============  ====================================================================

关键能力
--------
- **增量更新**：``add()`` 只对新 chunk 做向量化并追加，``delete()`` 支持按
  ``doc_id`` 批量删除 —— 新文档加入时**无需重建全量索引**。
- **幂等**：同一 ``chunk_id`` 重复 add 会覆盖而非重复插入。
- **Top-K 检索**：余弦 / 内积 / L2 三种距离度量。
- **持久化**：向量与元数据分离存储（``.npy`` + ``.json``），便于人工检查。
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.storage import ensure_dir, read_json, write_json
from ..utils.types import Chunk, RetrievedChunk

__all__ = [
    "BaseVectorStore",
    "NumpyVectorStore",
    "ChromaVectorStore",
    "FaissVectorStore",
    "create_vector_store",
    "VECTOR_STORE_BACKENDS",
]

_log = get_logger("retrieval.vector_store")


def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """余弦相似度（矩阵行已归一化时退化为点积；此处做完整计算以兼容未归一化向量）。"""
    denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) + 1e-12)
    denom[denom == 0] = 1e-12
    return (matrix @ query) / denom


def _l2_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """L2 距离转相似度（越小越相似 → 取负值排序）。"""
    distances = np.linalg.norm(matrix - query, axis=1)
    return -distances


class BaseVectorStore(ABC):
    """向量存储抽象接口。"""

    backend: str = "base"

    def __init__(self, dimension: int = 512, collection: str = "papers", distance: str = "cosine") -> None:
        self.dimension = int(dimension)
        self.collection = collection
        self.distance = distance
        self._lock = threading.RLock()
        # 后端加载元数据失败时写入错误信息（而不是静默把知识库显示为空）
        self._load_error = ""

    # -- 子类实现 ---------------------------------------------------------
    @abstractmethod
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        """写入（或覆盖）chunk 及其向量，返回实际写入条数。"""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[RetrievedChunk]:
        """向量相似度 Top-K 检索。"""

    @abstractmethod
    def delete(self, doc_id: str) -> int:
        """删除某文档的所有 chunk，返回删除条数。"""

    @abstractmethod
    def all_chunks(self) -> List[Chunk]:
        """返回全部 chunk（BM25 索引与统计需要）。"""

    @abstractmethod
    def count(self) -> int:
        """当前向量总数。"""

    @abstractmethod
    def persist(self) -> None:
        """落盘。"""

    # -- 通用能力 ---------------------------------------------------------
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        for chunk in self.all_chunks():
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def clear(self) -> None:
        """清空索引（测试与"重建知识库"功能用）。"""
        for doc_id in {c.doc_id for c in self.all_chunks()}:
            self.delete(doc_id)
        self.persist()

    def doc_ids(self) -> List[str]:
        """返回当前索引中的全部文档 ID。"""
        return sorted({c.doc_id for c in self.all_chunks()})

    def stats(self) -> Dict[str, Any]:
        chunks = self.all_chunks()
        return {
            "backend": self.backend,
            "collection": self.collection,
            "dimension": self.dimension,
            "distance": self.distance,
            "num_vectors": len(chunks),
            "num_documents": len({c.doc_id for c in chunks}),
        }

    def health_check(self) -> Dict[str, Any]:
        try:
            stats = self.stats()
            # 元数据加载失败时统计值会是 0，但知识库并非真的为空 —— 必须暴露出来，
            # 否则用户只看到"检索不到相关内容"，而真实原因是加载失败
            stats["ok"] = not self._load_error
            if self._load_error:
                stats["error"] = self._load_error[:200]
            return stats
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": self.backend, "error": str(exc)[:200]}


# ==============================================================================
# NumPy 后端（默认，零依赖）
# ==============================================================================
class NumpyVectorStore(BaseVectorStore):
    """纯 NumPy 向量库。

    存储布局（``persist_dir/``）::

        vectors.npy       # float32 矩阵，shape=(N, dim)
        chunks.json       # 与矩阵行一一对应的 chunk 元数据
        store_meta.json   # 后端配置快照

    检索复杂度 O(N·d)，对十万级 chunk 在 CPU 上是毫秒级，完全够用；
    超过百万级建议切换 faiss 后端。
    """

    backend = "numpy"

    def __init__(
        self,
        dimension: int = 512,
        persist_dir: Any = "./data/vector_store",
        collection: str = "papers",
        distance: str = "cosine",
        auto_load: bool = True,
    ) -> None:
        super().__init__(dimension, collection, distance)
        self.persist_dir = Path(persist_dir)
        self.vectors_path = self.persist_dir / "vectors.npy"
        self.chunks_path = self.persist_dir / "chunks.json"
        self.meta_path = self.persist_dir / "store_meta.json"

        self._vectors: np.ndarray = np.zeros((0, self.dimension), dtype=np.float32)
        self._chunks: List[Chunk] = []
        self._index_of: Dict[str, int] = {}
        if auto_load:
            self._load()

    # -- 内部 -------------------------------------------------------------
    def _load(self) -> None:
        """从磁盘恢复索引。"""
        if not self.vectors_path.exists() or not self.chunks_path.exists():
            return
        try:
            vectors = np.load(self.vectors_path)
            raw_chunks = read_json(self.chunks_path, default=[])
        except Exception as exc:  # noqa: BLE001
            _log.warning("向量索引加载失败（将使用空索引）: %s", exc)
            return

        chunks: List[Chunk] = []
        for item in raw_chunks:
            chunk = Chunk.from_dict(item)
            chunks.append(chunk)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            _log.warning(
                "向量矩阵(%s)与元数据(%d)数量不匹配，索引已丢弃",
                getattr(vectors, "shape", "?"),
                len(chunks),
            )
            return
        self._vectors = vectors.astype(np.float32)
        self._chunks = chunks
        self._index_of = {c.chunk_id: i for i, c in enumerate(chunks)}
        if self._vectors.shape[1] != self.dimension:
            self.dimension = int(self._vectors.shape[1])
        _log.info(
            "向量索引已加载: %d 个向量 / %d 篇文档 (dim=%d, backend=%s)",
            len(chunks),
            len({c.doc_id for c in chunks}),
            self.dimension,
            self.backend,
        )

    def _append(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        """追加到矩阵尾部，并更新倒排索引。"""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配：期望 {self.dimension}，收到 {embeddings.shape[1]}"
            )
        start = self._vectors.shape[0]
        self._vectors = np.vstack([self._vectors, embeddings.astype(np.float32)])
        self._chunks.extend(chunks)
        for offset, chunk in enumerate(chunks):
            self._index_of[chunk.chunk_id] = start + offset

    def _remove_indices(self, indices: Iterable[int]) -> int:
        """按行号删除（重建矩阵，O(N)）。"""
        drop = sorted(set(indices))
        if not drop:
            return 0
        keep_mask = np.ones(self._vectors.shape[0], dtype=bool)
        keep_mask[drop] = False
        self._vectors = self._vectors[keep_mask]
        self._chunks = [c for i, c in enumerate(self._chunks) if keep_mask[i]]
        self._index_of = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        return len(drop)

    # -- 接口实现 ---------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        items = list(chunks)
        if not items:
            return 0
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[0] != len(items):
            raise ValueError(f"chunk 数({len(items)})与向量数({matrix.shape[0]})不一致")

        with self._lock:
            # 幂等：先删除同 chunk_id 的旧记录
            duplicates = [self._index_of[c.chunk_id] for c in items if c.chunk_id in self._index_of]
            if duplicates:
                self._remove_indices(duplicates)
            self._append(items, matrix)
        return len(items)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[RetrievedChunk]:
        with self._lock:
            if self._vectors.shape[0] == 0:
                return []
            query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            if query.shape[0] != self.dimension:
                # 维度不一致 = 索引是用**另一个** Embedding 模型建的。
                # 早期实现截断/补零把查询"对齐"过去，结果是在无关的向量空间里
                # 算相似度：实测 4 维查询打进 8 维索引仍返回 score=0.46 的"命中"，
                # 32 维查询返回 0.65 —— 静默给出垃圾检索，比报错危险得多。
                # 这里必须显式失败，让它变成一条可操作的"请重建索引"错误。
                raise ValueError(
                    f"查询向量维度 {query.shape[0]} 与索引维度 {self.dimension} 不一致："
                    f"索引可能是用其它 Embedding 模型建立的。"
                    f"请删除并重建索引（`python run.py --reindex`），"
                    f"或把 embedding 配置改回建索引时使用的模型。"
                )
            if self.distance == "l2":
                scores = _l2_scores(self._vectors, query)
            elif self.distance == "ip":
                scores = self._vectors @ query
            else:
                scores = _cosine_scores(self._vectors, query)

            k = min(max(1, top_k), len(self._chunks))
            # argpartition 取 Top-K，再对候选排序（比全排序快）
            if k < len(scores):
                candidate_idx = np.argpartition(-scores, k - 1)[:k]
                order = candidate_idx[np.argsort(-scores[candidate_idx])]
            else:
                order = np.argsort(-scores)

            results: List[RetrievedChunk] = []
            for rank, idx in enumerate(order, start=1):
                score = float(scores[idx])
                if not np.isfinite(score):
                    continue
                results.append(
                    RetrievedChunk(
                        chunk=self._chunks[int(idx)],
                        score=score,
                        vector_score=score,
                        vector_rank=rank,
                        source_route="vector",
                    )
                )
            return results

    def _align(self, vector: np.ndarray) -> np.ndarray:
        """把查询向量对齐到索引维度。

        ⚠️ 仅用于**明确知道自己在做什么**的场景（例如导入外部已对齐向量）。
        检索路径不再使用它 —— 维度不一致必须在 :meth:`search` 里报错，
        而不是悄悄截断/补零后在错误的空间里比相似度。
        """
        if vector.shape[0] > self.dimension:
            return vector[: self.dimension]
        padded = np.zeros(self.dimension, dtype=np.float32)
        padded[: vector.shape[0]] = vector
        return padded

    def delete(self, doc_id: str) -> int:
        with self._lock:
            indices = [i for i, c in enumerate(self._chunks) if c.doc_id == doc_id]
            return self._remove_indices(indices)

    def all_chunks(self) -> List[Chunk]:
        with self._lock:
            return list(self._chunks)

    def count(self) -> int:
        return int(self._vectors.shape[0])

    def persist(self) -> None:
        with self._lock:
            ensure_dir(self.persist_dir)
            np.save(self.vectors_path, self._vectors)
            write_json(self.chunks_path, [c.to_dict() for c in self._chunks])
            write_json(
                self.meta_path,
                {
                    "backend": self.backend,
                    "collection": self.collection,
                    "dimension": self.dimension,
                    "distance": self.distance,
                    "num_vectors": len(self._chunks),
                },
            )

    def clear(self) -> None:
        with self._lock:
            self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
            self._chunks = []
            self._index_of = {}
        self.persist()

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["persist_dir"] = str(self.persist_dir)
        data["matrix_shape"] = list(self._vectors.shape)
        data["memory_mb"] = round(self._vectors.nbytes / 1024 / 1024, 2)
        return data


# ==============================================================================
# Chroma 后端
# ==============================================================================
class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 后端（可选依赖）。"""

    backend = "chroma"

    def __init__(
        self,
        dimension: int = 512,
        persist_dir: Any = "./data/vector_store/chroma",
        collection: str = "papers",
        distance: str = "cosine",
        auto_load: bool = True,
    ) -> None:
        super().__init__(dimension, collection, distance)
        self.persist_dir = Path(persist_dir)
        ensure_dir(self.persist_dir)
        self._chunks: Dict[str, Chunk] = {}
        self._client = self._open_client()
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata=self._collection_metadata()
        )
        if auto_load:
            self._reload_chunks()

    def _collection_metadata(self) -> Dict[str, str]:
        return {"hnsw:space": "cosine" if self.distance == "cosine" else "l2"}

    def _open_client(self) -> Any:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 chromadb，请执行 `pip install chromadb`，"
                "或把 vector_store.backend 改为 numpy"
            ) from exc
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _reload_chunks(self) -> None:
        """从 Chroma 集合元数据恢复 chunk 清单。

        失败时**不能静默返回**：``self._chunks`` 会保持为空，于是 search() 返回
        []、count()/stats() 报 0、delete() 删不掉 —— 用户看到的是"知识库里
        什么都没有"，而真实原因是加载失败。这里改为记录错误并置健康标志。
        """
        try:
            data = self._collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            _log.error(
                "Chroma 集合元数据加载失败，知识库将显示为空：%s（%s）",
                exc,
                self._load_error,
            )
            return
        self._load_error = ""
        for meta in data.get("metadatas") or []:
            blob = (meta or {}).get("chunk_json")
            if not blob:
                continue
            chunk = Chunk.from_dict(json.loads(blob))
            self._chunks[chunk.chunk_id] = chunk

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        items = list(chunks)
        if not items:
            return 0
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        with self._lock:
            self._collection.upsert(
                ids=[c.chunk_id for c in items],
                embeddings=[row.tolist() for row in matrix],
                documents=[c.text for c in items],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "page": c.page,
                        "index": c.index,
                        "source": c.source,
                        "strategy": c.strategy,
                        "chunk_json": c.to_json(),
                    }
                    for c in items
                ],
            )
            for chunk in items:
                self._chunks[chunk.chunk_id] = chunk
        return len(items)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[RetrievedChunk]:
        with self._lock:
            if not self._chunks:
                return []
            query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            result = self._collection.query(
                query_embeddings=[query.tolist()],
                n_results=min(max(1, top_k), len(self._chunks)),
                include=["distances", "metadatas"],
            )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        out: List[RetrievedChunk] = []
        for rank, (chunk_id, distance) in enumerate(zip(ids, distances), start=1):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            score = 1.0 - float(distance) if self.distance == "cosine" else -float(distance)
            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    vector_score=score,
                    vector_rank=rank,
                    source_route="vector",
                )
            )
        return out

    def delete(self, doc_id: str) -> int:
        with self._lock:
            victims = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
            if victims:
                self._collection.delete(ids=victims)
                for cid in victims:
                    self._chunks.pop(cid, None)
            return len(victims)

    def clear(self) -> None:
        """清空集合。

        覆写基类：基类逐文档 delete，每个文档都要在 ``_chunks`` 里做一次全量扫描
        （O(D·N)），且与 Chroma 往返 D 次。
        """
        with self._lock:
            try:
                self._client.delete_collection(self.collection)
            except Exception as exc:  # noqa: BLE001
                _log.warning("删除 Chroma 集合失败，改为逐条删除: %s", exc)
                ids = list(self._chunks)
                if ids:
                    self._collection.delete(ids=ids)
            self._collection = self._client.get_or_create_collection(
                name=self.collection, metadata=self._collection_metadata()
            )
            self._chunks = {}

    def all_chunks(self) -> List[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def count(self) -> int:
        return len(self._chunks)

    def persist(self) -> None:
        # Chroma PersistentClient 自动落盘
        return

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["persist_dir"] = str(self.persist_dir)
        return data


# ==============================================================================
# FAISS 后端
# ==============================================================================
class FaissVectorStore(BaseVectorStore):
    """FAISS 后端（可选依赖，适合大规模向量）。"""

    backend = "faiss"

    def __init__(
        self,
        dimension: int = 512,
        persist_dir: Any = "./data/vector_store/faiss",
        collection: str = "papers",
        distance: str = "cosine",
        auto_load: bool = True,
    ) -> None:
        super().__init__(dimension, collection, distance)
        self.persist_dir = Path(persist_dir)
        ensure_dir(self.persist_dir)
        self.index_path = self.persist_dir / f"{collection}.faiss"
        self.chunks_path = self.persist_dir / f"{collection}_chunks.json"
        self._chunks: List[Chunk] = []
        self._index = self._new_index()
        if auto_load:
            self._load()

    def _new_index(self) -> Any:
        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "未安装 faiss，请执行 `pip install faiss-cpu`，"
                "或把 vector_store.backend 改为 numpy"
            ) from exc
        # 内积索引 + 归一化向量 == 余弦相似度
        return faiss.IndexFlatIP(self.dimension)

    def _load(self) -> None:
        if not (self.index_path.exists() and self.chunks_path.exists()):
            return
        import faiss  # type: ignore

        self._index = faiss.read_index(str(self.index_path))
        self.dimension = int(self._index.d)
        self._chunks = [Chunk.from_dict(item) for item in read_json(self.chunks_path, default=[])]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        items = list(chunks)
        if not items:
            return 0
        matrix = self._normalize(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            duplicates = {c.chunk_id for c in items} & {c.chunk_id for c in self._chunks}
            if duplicates:
                self._rebuild_excluding(duplicates)
            self._index.add(matrix)
            self._chunks.extend(items)
        return len(items)

    def _rebuild_excluding(self, exclude: set[str]) -> None:
        """FAISS IndexFlat 不支持删除，采用"重建"策略（仅在同 ID 覆盖时触发）。"""
        import faiss  # type: ignore

        keep = [c for c in self._chunks if c.chunk_id not in exclude]
        old_vectors = self._index.reconstruct_n(0, self._index.ntotal)
        keep_ids = {c.chunk_id for c in keep}
        keep_rows = [
            old_vectors[i] for i, c in enumerate(self._chunks) if c.chunk_id in keep_ids
        ]
        self._index = faiss.IndexFlatIP(self.dimension)
        if keep_rows:
            self._index.add(np.vstack(keep_rows).astype(np.float32))
        self._chunks = keep

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[RetrievedChunk]:
        with self._lock:
            if not self._chunks:
                return []
            query = self._normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
            k = min(max(1, top_k), len(self._chunks))
            scores, indices = self._index.search(query, k)
        results: List[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._chunks):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[int(idx)],
                    score=float(score),
                    vector_score=float(score),
                    vector_rank=rank,
                    source_route="vector",
                )
            )
        return results

    def delete(self, doc_id: str) -> int:
        with self._lock:
            victims = {c.chunk_id for c in self._chunks if c.doc_id == doc_id}
            if victims:
                self._rebuild_excluding(victims)
            return len(victims)

    def clear(self) -> None:
        """清空索引。

        必须覆写基类实现：基类是"逐文档调 delete()"，而 FAISS 的 delete 每次都会
        重建整个索引并重新 add 全部向量 —— 1000 篇文档就是 1000 次全量重建，
        且每次 delete 都重新统计一遍文档集合。实测在这里会直接卡死。
        """
        with self._lock:
            self._index = self._new_index()
            self._chunks = []
        self.persist()

    def all_chunks(self) -> List[Chunk]:
        with self._lock:
            return list(self._chunks)

    def count(self) -> int:
        return int(getattr(self._index, "ntotal", 0))

    def persist(self) -> None:
        with self._lock:
            import faiss  # type: ignore

            faiss.write_index(self._index, str(self.index_path))
            write_json(self.chunks_path, [c.to_dict() for c in self._chunks])

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["persist_dir"] = str(self.persist_dir)
        return data


# ==============================================================================
# 工厂
# ==============================================================================
VECTOR_STORE_BACKENDS: Dict[str, type] = {
    "numpy": NumpyVectorStore,
    "chroma": ChromaVectorStore,
    "chromadb": ChromaVectorStore,
    "faiss": FaissVectorStore,
}


def create_vector_store(
    config: Any = None,
    *,
    backend: Optional[str] = None,
    dimension: Optional[int] = None,
    persist_dir: Optional[Any] = None,
    collection: Optional[str] = None,
    auto_load: bool = True,
) -> BaseVectorStore:
    """按配置创建向量库。

    Parameters
    ----------
    config: 全局配置对象（**第一个位置参数**，与其它工厂保持一致）
    backend: 覆盖 ``vector_store.backend``（``numpy`` / ``chroma`` / ``faiss``）
    dimension: 覆盖向量维度（通常与 Embedding 模型维度一致）
    persist_dir / collection: 覆盖持久化目录与集合名
    auto_load: 是否在构造时从磁盘恢复已有索引

    其余参数刻意设为 **keyword-only**：``create_vector_store(cfg, "chroma")`` 这种
    位置传参极易把配置对象误传给 ``backend``，强制写参数名可彻底避免该类错误。

    若首选后端依赖缺失（如未装 chromadb），自动降级到 numpy 并给出警告。
    """
    cfg = config or get_config()
    node = cfg.vector_store
    name = (backend or str(node.backend) or "numpy").lower()
    dim = int(dimension or cfg.embedding.dimension)
    target_dir = persist_dir or cfg.resolve_path(str(node.persist_dir))
    coll = collection or str(node.collection)
    distance = str(node.distance)

    cls = VECTOR_STORE_BACKENDS.get(name)
    attempted: List[str] = []
    while cls is not None:
        attempted.append(name)
        try:
            store = cls(
                dimension=dim,
                persist_dir=target_dir,
                collection=coll,
                distance=distance,
                auto_load=auto_load,
            )
            _log.info("向量库已创建: backend=%s collection=%s dim=%d", store.backend, coll, dim)
            return store
        except Exception as exc:  # noqa: BLE001
            _log.warning("向量库后端 %s 不可用（%s），尝试降级", name, exc)
            name = "numpy"
            cls = VECTOR_STORE_BACKENDS.get(name)
            if "numpy" in attempted:
                break

    store = NumpyVectorStore(
        dimension=dim,
        persist_dir=target_dir,
        collection=coll,
        distance=distance,
        auto_load=auto_load,
    )
    _log.info("向量库已创建（numpy 兜底）: collection=%s dim=%d", coll, dim)
    return store
