"""语义缓存（模块二 5.2.3）。

与普通 KV 缓存的区别
--------------------
普通缓存用**字符串精确匹配**做键，用户把"Transformer 用了什么数据集"
换成"Transformer 模型训练用了哪些数据"，就会全部击穿缓存。
语义缓存改为**向量相似度匹配**：问题向量与缓存条目的向量相似度超过阈值
（默认 0.93）即视为同一问题，直接复用之前的答案。

三层查找（从快到慢）
--------------------
1. **内存 LRU**：进程内字典，命中耗时 < 1ms，容量 512 条。
2. **SQLite 持久层**：跨进程/重启存活，向量以 float32 二进制存储。
3. **向量近邻检索**：对候选集做余弦相似度全扫（缓存规模 < 1 万条时开销可忽略；
   若需更大规模可换 FAISS，接口已预留）。

缓存键与失效
------------
- 缓存条目记录 ``model`` 与 ``top_k`` 等生成参数，参数变化时不会误用旧答案。
- 支持 TTL（默认 7 天）与容量上限（超出按 LRU 淘汰）。
- 提供按问题前缀 / 全量清空接口，便于"知识库更新后清缓存"。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.storage import open_sqlite
from ..utils.types import RAGAnswer, stable_hash, to_jsonable
from .prompt_template import build_citations

__all__ = ["SemanticCache", "CacheEntry", "create_cache", "NullCache"]

_log = get_logger("generation.cache")


@dataclass
class CacheEntry:
    """一条缓存记录。"""

    key: str
    question: str
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    vector: Optional[np.ndarray] = None
    model: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    hits: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = 0.0

    def to_row(self) -> Tuple:
        return (
            self.key,
            self.question,
            self.answer,
            json.dumps(self.citations, ensure_ascii=False),
            json.dumps(self.retrieved, ensure_ascii=False),
            self.vector.astype(np.float32).tobytes() if self.vector is not None else b"",
            self.model,
            json.dumps(self.params, ensure_ascii=False),
            self.hits,
            self.created_at,
            self.last_hit_at,
        )

    @classmethod
    def from_row(cls, row: Any) -> "CacheEntry":
        vector_bytes = row["vector"] or b""
        return cls(
            key=row["cache_key"],
            question=row["question"],
            answer=row["answer"],
            citations=json.loads(row["citations"] or "[]"),
            retrieved=json.loads(row["retrieved"] or "[]"),
            vector=np.frombuffer(vector_bytes, dtype=np.float32).copy() if vector_bytes else None,
            model=row["model"] or "",
            params=json.loads(row["params"] or "{}"),
            hits=row["hits"] or 0,
            created_at=row["created_at"] or 0.0,
            last_hit_at=row["last_hit_at"] or 0.0,
        )


class SemanticCache:
    """语义缓存实现（SQLite + 内存 LRU + 向量近邻匹配）。"""

    def __init__(
        self,
        path: Any = "./data/cache/semantic_cache.db",
        similarity_threshold: float = 0.93,
        ttl_seconds: int = 604800,
        max_entries: int = 5000,
        memory_size: int = 512,
        enabled: bool = True,
        backend: str = "sqlite",
    ) -> None:
        self.path = Path(path)
        # 阈值与容量做一次收敛，避免配置写成 1.5 / 0 / -1 时出现
        # "永远不命中"或"内存无限增长"这类静默故障
        self.threshold = min(1.0, max(0.0, float(similarity_threshold)))
        if self.threshold != float(similarity_threshold):
            _log.warning(
                "generation.cache.similarity_threshold=%.3f 超出 [0,1]，已收敛为 %.3f",
                float(similarity_threshold),
                self.threshold,
            )
        self.ttl = max(0, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self.memory_size = max(1, int(memory_size))
        self.enabled = bool(enabled)
        self.backend = backend

        self._pool = None
        self._memory: Dict[str, CacheEntry] = {}
        self._memory_order: List[str] = []
        self._lock = threading.RLock()

        self._stats = {"lookups": 0, "hits": 0, "misses": 0, "stores": 0, "evictions": 0}

        if self.enabled and backend == "sqlite":
            try:
                self._pool = open_sqlite(self.path)
                self._init_schema()
            except Exception as exc:  # noqa: BLE001
                _log.warning("语义缓存数据库初始化失败，降级为纯内存缓存: %s", exc)
                self._pool = None

    # -- 结构 -------------------------------------------------------------
    def _init_schema(self) -> None:
        assert self._pool is not None
        with self._pool.write() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    cache_key    TEXT PRIMARY KEY,
                    question     TEXT NOT NULL,
                    answer       TEXT NOT NULL,
                    citations    TEXT,
                    retrieved    TEXT,
                    vector       BLOB,
                    model        TEXT,
                    params       TEXT,
                    hits         INTEGER DEFAULT 0,
                    created_at   REAL,
                    last_hit_at  REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_created ON semantic_cache(created_at)"
            )
        self._evict_if_needed()

    # -- 键与向量 ---------------------------------------------------------
    @staticmethod
    def make_key(question: str, model: str = "", params: Optional[Dict[str, Any]] = None) -> str:
        """精确键：问题 + 生成参数 的联合哈希（用于内存层与去重）。"""
        payload = f"{question.strip()}|{model}|{json.dumps(params or {}, sort_keys=True)}"
        return stable_hash(payload, 32)

    # -- 查询 -------------------------------------------------------------
    def get(
        self,
        question: str,
        vector: Optional[np.ndarray] = None,
        model: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[CacheEntry]:
        """查找缓存。

        Parameters
        ----------
        question: 用户问题原文
        vector: 问题向量（None 时退化为精确键匹配）
        model / params: 生成参数，参与匹配条件（参数不同不复用）
        """
        if not self.enabled:
            return None
        with self._lock:
            self._stats["lookups"] += 1

        key = self.make_key(question, model, params)

        # 1) 内存精确命中
        entry = self._memory.get(key)
        if entry is not None and not self._expired(entry):
            self._touch(entry)
            with self._lock:
                self._stats["hits"] += 1
            _log.debug("语义缓存命中（内存精确）: %s", question[:40])
            return entry

        # 2) 数据库精确命中
        entry = self._load_by_key(key)
        if entry is not None and not self._expired(entry):
            self._touch(entry)
            self._memory_put(entry)
            with self._lock:
                self._stats["hits"] += 1
            _log.debug("语义缓存命中（数据库精确）: %s", question[:40])
            return entry

        # 3) 语义近邻命中
        if vector is not None:
            entry, similarity = self._search_similar(vector, model, params)
            if entry is not None:
                self._touch(entry)
                self._memory_put(entry)
                with self._lock:
                    self._stats["hits"] += 1
                _log.info(
                    "语义缓存命中（相似度 %.4f ≥ %.2f）: %r ≈ %r",
                    similarity,
                    self.threshold,
                    question[:30],
                    entry.question[:30],
                )
                return entry

        with self._lock:
            self._stats["misses"] += 1
        return None

    def _search_similar(
        self,
        vector: np.ndarray,
        model: str,
        params: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[CacheEntry], float]:
        """在候选集中做余弦相似度检索，返回最佳匹配。

        性能注意：早期实现用 ``SELECT *`` 把最近 1000 条**整行**（含答案正文、
        引用与检索片段的 JSON）读进内存，再逐条 Python 循环算余弦 —— 每次
        未命中都要反序列化几 MB 文本。现在分两步：
        先用只含打分所需列（vector/model/params/created_at）的轻量查询筛出候选，
        用一次矩阵乘法批量算余弦，**只把胜出那条**完整读回来。
        """
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0:
            return None, 0.0

        param_key = json.dumps(params or {}, sort_keys=True)

        # 打分单元：(条目标识, 向量, 已完成的 CacheEntry 或 None)
        scored: List[Tuple[str, Optional[CacheEntry], np.ndarray]] = []
        seen: set[str] = set()

        with self._lock:
            memory_snapshot = list(self._memory.values())

        for entry in memory_snapshot:
            if entry.key in seen or entry.vector is None or self._expired(entry):
                continue
            if entry.model != model or json.dumps(entry.params, sort_keys=True) != param_key:
                continue
            seen.add(entry.key)
            scored.append((entry.key, entry, np.asarray(entry.vector, dtype=np.float32).reshape(-1)))

        if self._pool is not None:
            for key, candidate_vector, created_at, entry_model, entry_params in self._load_candidates(limit=1000):
                if key in seen:
                    continue
                if entry_model != model:
                    continue
                try:
                    if json.dumps(json.loads(entry_params or "{}"), sort_keys=True) != param_key:
                        continue
                except json.JSONDecodeError:
                    continue
                if self.ttl > 0 and (time.time() - (created_at or 0.0)) > self.ttl:
                    continue
                seen.add(key)
                # DB 里的行先不反序列化正文，命中后再按 key 完整读回
                scored.append((key, None, candidate_vector))

        if not scored:
            return None, 0.0

        # 只保留维度一致的候选，拼成矩阵一次算完余弦相似度
        usable = [item for item in scored if item[2].shape[0] == query.shape[0] and item[2].size]
        if not usable:
            return None, 0.0
        matrix = np.vstack([item[2] for item in usable])
        norms = np.linalg.norm(matrix, axis=1)
        denom = norms * query_norm
        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = np.where(denom > 0, (matrix @ query) / denom, -1.0)

        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        if best_similarity < self.threshold:
            return None, max(best_similarity, 0.0)

        key, entry, _vec = usable[best_index]
        if entry is None:
            entry = self._load_by_key(key)
            if entry is None:  # 竞态：恰在打分后被删除
                return None, max(best_similarity, 0.0)
        return entry, best_similarity

    # -- 写入 -------------------------------------------------------------
    def put(
        self,
        question: str,
        answer: RAGAnswer,
        vector: Optional[np.ndarray] = None,
        model: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """写入缓存。降级/空答案不缓存（避免把失败结果固化）。"""
        if not self.enabled:
            return False
        if not answer.answer or answer.degraded:
            return False

        key = self.make_key(question, model, params)
        entry = CacheEntry(
            key=key,
            question=question.strip(),
            answer=answer.answer,
            citations=[c.to_dict() for c in answer.citations],
            retrieved=[
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "source": r.chunk.source,
                    "page": r.chunk.page,
                    "score": round(r.score, 4),
                    "text": r.text,
                }
                for r in answer.retrieved
            ],
            vector=np.asarray(vector, dtype=np.float32).reshape(-1) if vector is not None else None,
            model=model,
            params=params or {},
        )

        with self._lock:
            self._stats["stores"] += 1
        self._memory_put(entry)

        if self._pool is not None:
            try:
                with self._pool.write() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO semantic_cache
                        (cache_key, question, answer, citations, retrieved, vector,
                         model, params, hits, created_at, last_hit_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        entry.to_row(),
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning("语义缓存写入失败: %s", exc)
                return False
        self._evict_if_needed()
        return True

    # -- 内部工具 ---------------------------------------------------------
    def _load_by_key(self, key: str) -> Optional[CacheEntry]:
        if self._pool is None:
            return None
        with self._pool.read() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        return CacheEntry.from_row(row) if row else None

    def _load_candidates(self, limit: int = 1000) -> List[Tuple[str, np.ndarray, float, str, str]]:
        """加载最近的缓存条目**打分所需的最小列**，用于语义近邻匹配。

        返回 ``[(key, vector, created_at, model, params_json), ...]``。
        刻意不取 answer/citations/retrieved —— 那些是每条几 KB 的 JSON，
        在逐条打分阶段完全用不上（命中后由 :meth:`_load_by_key` 单独读回）。
        """
        if self._pool is None:
            return []
        with self._pool.read() as conn:
            rows = conn.execute(
                "SELECT cache_key, vector, created_at, model, params FROM semantic_cache "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        candidates: List[Tuple[str, np.ndarray, float, str, str]] = []
        for row in rows:
            blob = row["vector"] or b""
            if not blob:
                continue
            candidates.append(
                (
                    row["cache_key"],
                    np.frombuffer(blob, dtype=np.float32).copy(),
                    float(row["created_at"] or 0.0),
                    row["model"] or "",
                    row["params"] or "{}",
                )
            )
        return candidates

    def _memory_put(self, entry: CacheEntry) -> None:
        with self._lock:
            if entry.key not in self._memory:
                self._memory_order.append(entry.key)
            self._memory[entry.key] = entry
            while len(self._memory_order) > self.memory_size:
                oldest = self._memory_order.pop(0)
                self._memory.pop(oldest, None)

    def _touch(self, entry: CacheEntry) -> None:
        """更新命中计数与最近命中时间。"""
        entry.hits += 1
        entry.last_hit_at = time.time()
        if self._pool is None:
            return
        try:
            with self._pool.write() as conn:
                conn.execute(
                    "UPDATE semantic_cache SET hits = ?, last_hit_at = ? WHERE cache_key = ?",
                    (entry.hits, entry.last_hit_at, entry.key),
                )
        except Exception:  # noqa: BLE001
            pass

    def _expired(self, entry: CacheEntry) -> bool:
        return self.ttl > 0 and (time.time() - entry.created_at) > self.ttl

    def _evict_if_needed(self) -> int:
        """超出容量或 TTL 时清理（按"最少使用 + 最久未命中"淘汰）。"""
        if self._pool is None:
            return 0
        removed = 0
        try:
            with self._pool.write() as conn:
                if self.ttl > 0:
                    cursor = conn.execute(
                        "DELETE FROM semantic_cache WHERE created_at < ?",
                        (time.time() - self.ttl,),
                    )
                    removed += cursor.rowcount or 0
                count = conn.execute("SELECT COUNT(*) AS c FROM semantic_cache").fetchone()["c"]
                if count > self.max_entries:
                    overflow = count - self.max_entries
                    conn.execute(
                        """
                        DELETE FROM semantic_cache WHERE cache_key IN (
                            SELECT cache_key FROM semantic_cache
                            ORDER BY hits ASC, last_hit_at ASC, created_at ASC LIMIT ?
                        )
                        """,
                        (overflow,),
                    )
                    removed += overflow
        except Exception as exc:  # noqa: BLE001
            _log.warning("语义缓存清理失败: %s", exc)
        if removed:
            with self._lock:
                self._stats["evictions"] += removed
            _log.info("语义缓存清理 %d 条（TTL/容量）", removed)
        return removed

    # -- 维护 -------------------------------------------------------------
    def clear(self, question_prefix: Optional[str] = None) -> int:
        """清空缓存；给定前缀时只清匹配项。返回被清除的条目数。

        注意要同时统计**内存层**与**数据库层**的条目：纯内存后端
        （或刚写入还没落盘的部分）如果只算数据库的 rowcount，会错误地返回 0。
        """
        with self._lock:
            if question_prefix:
                victims = [
                    key for key, entry in self._memory.items()
                    if entry.question.startswith(question_prefix)
                ]
            else:
                victims = list(self._memory.keys())
            for key in victims:
                self._memory.pop(key, None)
            self._memory_order = [k for k in self._memory_order if k in self._memory]

        removed = len(victims)
        if self._pool is None:
            return removed
        with self._pool.write() as conn:
            if question_prefix:
                cursor = conn.execute(
                    "DELETE FROM semantic_cache WHERE question LIKE ?", (f"{question_prefix}%",)
                )
            else:
                cursor = conn.execute("DELETE FROM semantic_cache")
            # 两个层级可能存在重叠（同一 key 同时在内存与磁盘），取较大值更符合直觉
            return max(removed, cursor.rowcount or 0)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            lookups = self._stats["lookups"]
            snapshot = dict(self._stats)
            snapshot["hit_rate"] = round(snapshot["hits"] / lookups, 4) if lookups else 0.0
            snapshot["memory_entries"] = len(self._memory)
        if self._pool is not None:
            try:
                with self._pool.read() as conn:
                    snapshot["entries"] = conn.execute(
                        "SELECT COUNT(*) AS c FROM semantic_cache"
                    ).fetchone()["c"]
                    snapshot["answer_chars"] = (
                        conn.execute(
                            "SELECT COALESCE(SUM(LENGTH(answer)), 0) AS c FROM semantic_cache"
                        ).fetchone()["c"]
                    )
            except Exception:  # noqa: BLE001
                snapshot["entries"] = 0
        else:
            snapshot["entries"] = len(self._memory)
        snapshot.update(
            {
                "enabled": self.enabled,
                "backend": self.backend if self._pool is not None else "memory-only",
                "threshold": self.threshold,
                "ttl_seconds": self.ttl,
                "max_entries": self.max_entries,
            }
        )
        return snapshot

    def health_check(self) -> Dict[str, Any]:
        data = self.stats()
        data["ok"] = True
        return data


class NullCache:
    """空缓存（用于关闭缓存或测试）。所有写操作无效，读操作恒为 None。"""

    enabled = False

    def get(self, *args: Any, **kwargs: Any) -> None:
        return None

    def put(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def clear(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def stats(self) -> Dict[str, Any]:
        return {"enabled": False, "entries": 0, "hit_rate": 0.0}

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "enabled": False}


def create_cache(config: Any = None, **overrides: Any) -> Any:
    """按配置创建语义缓存。

    Parameters
    ----------
    config: 全局配置对象
    overrides: 覆盖 ``generation.cache.*`` 中的任意字段
    """
    from ..utils.config import get_config

    cfg = config or get_config()
    node = cfg.generation.cache
    enabled = bool(overrides.pop("enabled", cfg.get("generation.cache.enabled", True)))
    if not enabled:
        _log.info("语义缓存已关闭（generation.cache.enabled=false）")
        return NullCache()

    params: Dict[str, Any] = {
        # 一律用点号路径 + 默认值读取，避免把"整段 section"误当成字段值
        "path": str(cfg.get("generation.cache.path", "./data/cache/semantic_cache.db")),
        "similarity_threshold": float(cfg.get("generation.cache.similarity_threshold", 0.93)),
        "ttl_seconds": int(cfg.get("generation.cache.ttl_seconds", 604800)),
        "max_entries": int(cfg.get("generation.cache.max_entries", 5000)),
        "backend": str(cfg.get("generation.cache.backend", "sqlite")),
    }
    params.update(overrides)
    # 相对路径按项目根目录解析
    if not Path(params["path"]).is_absolute():
        params["path"] = cfg.resolve_path(params["path"])
    cache = SemanticCache(**params)
    _log.info(
        "语义缓存已创建: backend=%s threshold=%.2f ttl=%ds max=%d",
        params["backend"],
        params["similarity_threshold"],
        params["ttl_seconds"],
        params["max_entries"],
    )
    return cache
