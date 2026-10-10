"""Embedding 向量化抽象层。

支持四种 provider，业务代码只依赖 :class:`BaseEmbedder`：

=============  =================================================================
provider       说明
=============  =================================================================
``hash``       离线哈希 n-gram 向量（零依赖、确定性、开箱即用）。
               采用 feature hashing + 字符/词双粒度 n-gram + 亚线性 TF 加权
               + L2 归一化。虽然不是神经网络语义向量，但对中英文技术文本的
               词面相似度刻画稳定，可完整支撑检索链路的开发、测试与评测。
``ollama``     通过本地 Ollama 调用 bge-large-zh / bge-m3 / nomic-embed-text
``openai``     OpenAI 兼容 ``/v1/embeddings`` 接口
``local``      本地 sentence-transformers 模型（需自行安装）
=============  =================================================================

附带特性
--------
- **持久化缓存**：文本 SHA1 → 向量 存入 SQLite，避免重复计算（增量索引前提）。
- **批量编码**：``encode_batch`` 支持进度回调，用于批量导入文档。
- **统一维度**：所有 provider 输出固定维度（hash/ollama/openai 按配置或探测）。
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .utils.config import Config, get_config, PROJECT_ROOT
from .utils.logger import get_logger
from .utils.storage import open_sqlite
from .utils.text import _RE_CJK_RUN, _RE_LATIN, tokenize
from .utils.types import stable_hash

__all__ = ["BaseEmbedder", "HashEmbedder", "OllamaEmbedder", "OpenAIEmbedder",
           "LocalEmbedder", "create_embedder", "EmbeddingCache"]

_log = get_logger("embedding")


# ==============================================================================
# 抽象基类
# ==============================================================================
class BaseEmbedder(ABC):
    """向量化模型统一接口。"""

    provider: str = "base"

    def __init__(
        self,
        config: Optional[Config] = None,
        cache: Optional["EmbeddingCache"] = None,
    ) -> None:
        cfg = (config or get_config()).embedding
        self.model = str(cfg.model)
        self.dimension = int(cfg.dimension)
        self.batch_size = int(cfg.batch_size)
        self.normalize = bool(cfg.normalize)
        self._cache = cache

    # -- 主接口 -----------------------------------------------------------
    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """底层实现：返回 shape=(n, dim) 的 float32 矩阵。"""
        raise NotImplementedError

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """单条文本向量化，返回 shape=(dim,) 的 float32 向量。"""
        return self.encode_batch([text], use_cache=use_cache)[0]

    def encode_batch(
        self,
        texts: Sequence[str],
        use_cache: bool = True,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """批量向量化，返回 shape=(n, dim) 的 float32 矩阵。

        先查缓存 → 批量计算未命中项 → 回写缓存。这一顺序使增量索引成为可能：
        新增文档时只有新 chunk 需要真正前向计算。
        """
        items = list(texts)
        if not items:
            return np.zeros((0, self.dimension), dtype=np.float32)

        results: List[Optional[np.ndarray]] = [None] * len(items)
        pending: List[int] = []

        if use_cache and self._cache is not None:
            keys = [self.cache_key(t) for t in items]
            hits = self._cache.get_many(keys)
            for idx, key in enumerate(keys):
                vec = hits.get(key)
                if vec is not None and vec.shape[0] == self.dimension:
                    results[idx] = vec
                else:
                    pending.append(idx)
        else:
            pending = list(range(len(items)))

        cache_miss = len(pending)
        if pending:
            for start in range(0, len(pending), max(1, self.batch_size)):
                batch_idx = pending[start : start + self.batch_size]
                matrix = np.asarray(self._encode([items[i] for i in batch_idx]), dtype=np.float32)
                if matrix.ndim == 1:
                    matrix = matrix.reshape(1, -1)
                if matrix.shape[1] != self.dimension:
                    # provider 实际维度与配置不一致时自动对齐（截断或补零）
                    matrix = self._fit_dimension(matrix)
                if self.normalize:
                    matrix = _l2_normalize(matrix)
                for row, idx in enumerate(batch_idx):
                    results[idx] = matrix[row]
                if progress:
                    progress(min(start + len(batch_idx), len(pending)), len(pending))
            if use_cache and self._cache is not None:
                pairs = [
                    (self.cache_key(items[i]), results[i])
                    for i in pending
                    if results[i] is not None
                ]
                self._cache.put_many(pairs)

        filled = [
            r if r is not None else np.zeros(self.dimension, dtype=np.float32) for r in results
        ]
        stacked = np.vstack(filled).astype(np.float32)
        if cache_miss:
            _log.debug("向量化 %d 条（缓存未命中 %d 条）", len(items), cache_miss)
        return stacked

    def _fit_dimension(self, matrix: np.ndarray) -> np.ndarray:
        """把 provider 输出的维度对齐到配置维度。"""
        current = matrix.shape[1]
        if current == self.dimension:
            return matrix
        if current > self.dimension:
            return matrix[:, : self.dimension]
        pad = np.zeros((matrix.shape[0], self.dimension - current), dtype=np.float32)
        return np.hstack([matrix, pad])

    def cache_key(self, text: str) -> str:
        """缓存键：模型 + 维度 + 文本内容 的联合哈希。"""
        raw = f"{self.provider}:{self.model}:{self.dimension}:{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    # -- 工具 -------------------------------------------------------------
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度（向量已归一化时等价于点积）。"""
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def health_check(self) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            vec = self.encode("健康检查 health check", use_cache=False)
            return {
                "ok": True,
                "provider": self.provider,
                "model": self.model,
                "dimension": int(vec.shape[0]),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "cache_enabled": self._cache is not None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": self.provider, "model": self.model, "error": str(exc)[:200]}


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化（零向量保持为零）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


# ==============================================================================
# 持久化缓存
# ==============================================================================
class EmbeddingCache:
    """Embedding 向量持久化缓存（SQLite）。

    表结构：``embeddings(cache_key TEXT PRIMARY KEY, dim INT, vector BLOB, model TEXT, ts REAL)``
    向量以 float32 二进制存储，512 维仅占 2KB，比 JSON 省 5 倍空间。
    """

    def __init__(self, path: Any = "./data/cache/embeddings.db", model: str = "") -> None:
        self.path = Path(path)
        self.model = model
        self._pool = open_sqlite(self.path)
        self._init_schema()
        self._mem: Dict[str, np.ndarray] = {}
        self._mem_lock = threading.Lock()
        self._mem_order: List[str] = []

    def _init_schema(self) -> None:
        with self._pool.write() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    dim       INTEGER NOT NULL,
                    vector    BLOB    NOT NULL,
                    model     TEXT,
                    ts        REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model)")

    # -- 读写 -------------------------------------------------------------
    def get_many(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        """批量查询；先查内存 LRU，再查数据库。"""
        found: Dict[str, np.ndarray] = {}
        missing: List[str] = []
        with self._mem_lock:
            for key in keys:
                vec = self._mem.get(key)
                if vec is not None:
                    found[key] = vec
                else:
                    missing.append(key)
        if not missing:
            return found
        unique = list(dict.fromkeys(missing))
        placeholders = ",".join("?" * len(unique))
        with self._pool.read() as conn:
            rows = conn.execute(
                f"SELECT cache_key, vector FROM embeddings WHERE cache_key IN ({placeholders})",
                unique,
            ).fetchall()
        for row in rows:
            vec = np.frombuffer(row["vector"], dtype=np.float32)
            found[row["cache_key"]] = vec
            self._mem_put(row["cache_key"], vec)
        return found

    def put_many(self, pairs: Iterable[tuple[str, np.ndarray]]) -> int:
        """批量写入（内存 + 数据库）。"""
        items = [(k, np.asarray(v, dtype=np.float32)) for k, v in pairs if v is not None]
        if not items:
            return 0
        with self._pool.write() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(cache_key, dim, vector, model, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (key, int(vec.shape[0]), vec.tobytes(), self.model, time.time())
                    for key, vec in items
                ],
            )
        for key, vec in items:
            self._mem_put(key, vec)
        return len(items)

    def _mem_put(self, key: str, vec: np.ndarray) -> None:
        """写入内存 LRU（上限 5000 条，超出淘汰最旧）。"""
        with self._mem_lock:
            if key not in self._mem:
                self._mem_order.append(key)
            self._mem[key] = vec
            while len(self._mem_order) > 5000:
                oldest = self._mem_order.pop(0)
                self._mem.pop(oldest, None)

    def stats(self) -> Dict[str, Any]:
        with self._pool.read() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"]
        return {"entries": int(total), "memory_entries": len(self._mem), "path": str(self.path)}

    def clear(self) -> None:
        with self._pool.write() as conn:
            conn.execute("DELETE FROM embeddings")
        with self._mem_lock:
            self._mem.clear()
            self._mem_order.clear()


# ==============================================================================
# Hash 向量化（零依赖）
# ==============================================================================
class HashEmbedder(BaseEmbedder):
    """离线哈希 n-gram 向量化器。

    算法
    ----
    1. **多粒度特征**：中文 unigram / bigram、英文小写词、全文词袋 token。
    2. **Feature hashing**：``h = blake2b(feature) % dim``，符号位由哈希次高位决定
       （signed hashing，抵消哈希冲突带来的系统性偏差）。
    3. **亚线性 TF 加权**：``w = 1 + log(tf)``，抑制高频词主导。
    4. **IDF 近似**：对超高频特征（如"的"、"方法"）施加固定降权。
    5. **L2 归一化**：使余弦相似度可直接用点积计算。

    特点：确定性（同文本永远同向量）、O(n) 速度、零依赖，适合开发/测试/离线评测。
    """

    provider = "hash"

    # 超高频特征降权表（近似 IDF 的轻量替代）
    _LOW_IDF = frozenset(
        "的 了 和 是 在 我们 本文 方法 结果 研究 分析 基于 通过 进行 一个 可以 中 与 及".split()
    )

    def __init__(
        self,
        config: Optional[Config] = None,
        cache: Optional[EmbeddingCache] = None,
        dimension: Optional[int] = None,
    ) -> None:
        super().__init__(config, cache)
        if dimension:
            self.dimension = int(dimension)

    # -- 特征抽取 ---------------------------------------------------------
    def _features(self, text: str) -> Counter:
        """抽取 (特征名, 权重计数) 计数器。"""
        counter: Counter = Counter()
        if not text:
            return counter

        # 1) 中文 unigram / bigram
        for run in _RE_CJK_RUN.findall(text):
            if len(run) == 1:
                counter[f"c1:{run}"] += 1
                continue
            for ch in run:
                counter[f"c1:{ch}"] += 1
            for i in range(len(run) - 1):
                counter[f"c2:{run[i:i + 2]}"] += 1
            # 三字组：捕捉"注意力机制"这类术语
            for i in range(len(run) - 2):
                counter[f"c3:{run[i:i + 3]}"] += 0.5

        # 2) 英文/数字词
        for word in _RE_LATIN.findall(text):
            lowered = word.lower().strip(".-")
            if len(lowered) > 1:
                counter[f"w:{lowered}"] += 1
                if len(lowered) > 4:  # 长词加字符 trigram，缓解词形变化
                    for i in range(len(lowered) - 2):
                        counter[f"g:{lowered[i:i + 3]}"] += 0.3

        # 3) 分词器 token（jieba 可用时提供词级特征）
        for tok in tokenize(text):
            counter[f"t:{tok}"] += 0.7
        return counter

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        dim = self.dimension
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counter = self._features(text)
            if not counter:
                continue
            for feature, count in counter.items():
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % dim
                sign = 1.0 if (value >> 63) & 1 else -1.0
                weight = 1.0 + math.log(1.0 + float(count))
                # 取特征名冒号后的部分：特征名的前缀长度不一（c1: 的 / c2: 方法 /
                # t: 方法 / w: the），早期写成 feature[2:] 会把 "c1:的" 切成 ":的"，
                # 于是**所有中文特征都匹配不上降权表** —— 而 hash 是默认向量化器，
                # 等于"高频词降权"这项功能一直没生效过。
                if feature.split(":", 1)[-1] in self._LOW_IDF:
                    weight *= 0.35
                matrix[row, index] += sign * weight
        return matrix


# ==============================================================================
# Ollama 向量化
# ==============================================================================
class OllamaEmbedder(BaseEmbedder):
    """通过本地 Ollama 调用 Embedding 模型（如 bge-large-zh / bge-m3）。"""

    provider = "ollama"

    def __init__(self, config: Optional[Config] = None, cache: Optional[EmbeddingCache] = None) -> None:
        super().__init__(config, cache)
        cfg = (config or get_config()).embedding
        self.base_url = str(cfg.base_url).rstrip("/")
        self.timeout = float((config or get_config()).llm.timeout)
        self._dim_probed = False

    def _probe_dimension(self) -> None:
        """首次调用时探测真实维度并同步到 self.dimension。"""
        if self._dim_probed:
            return
        vec = self._request(["维度探测"])[0]
        self.dimension = int(len(vec))
        self._dim_probed = True

    def _request(self, texts: Sequence[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": list(texts)}
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            vectors = body.get("embeddings")
            if vectors:
                return vectors
            single = body.get("embedding")
            return [single] if single else []
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # 兼容旧版 /api/embeddings（单条）
                return [self._request_legacy(text) for text in texts]
            raise
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"无法连接 Ollama({self.base_url}) 进行向量化：{exc}。"
                f"请执行 `ollama pull {self.model}`，或把 embedding.provider 改为 hash。"
            ) from exc

    def _request_legacy(self, text: str) -> List[float]:
        payload = {"model": self.model, "prompt": text}
        request = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body.get("embedding", [])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._request(texts)
        if not vectors:
            raise RuntimeError(f"Ollama Embedding 返回为空，请确认模型 {self.model} 已拉取")
        if not self._dim_probed:
            self.dimension = len(vectors[0])
            self._dim_probed = True
        return np.asarray(vectors, dtype=np.float32)

    def health_check(self) -> Dict[str, Any]:
        try:
            vec = self.encode("健康检查", use_cache=False)
            return {
                "ok": True,
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "dimension": int(vec.shape[0]),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": self.provider, "model": self.model, "error": str(exc)[:200]}


# ==============================================================================
# OpenAI 兼容向量化
# ==============================================================================
class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 兼容 ``/v1/embeddings``（OpenAI / 硅基流动 / vLLM / 通义兼容）。"""

    provider = "openai"

    def __init__(self, config: Optional[Config] = None, cache: Optional[EmbeddingCache] = None) -> None:
        super().__init__(config, cache)
        cfg = (config or get_config()).embedding
        self.base_url = str(cfg.base_url).rstrip("/")
        if not self.base_url.endswith("/v1") and "openai.com" not in self.base_url:
            self.base_url = f"{self.base_url}/v1"
        self.api_key = str(cfg.api_key) or str((config or get_config()).llm.api_key) or "sk-no-key"
        self.timeout = float((config or get_config()).llm.timeout)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        payload = {"model": self.model, "input": list(texts), "encoding_format": "float"}
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:300]
            raise RuntimeError(f"Embedding 接口 HTTP {exc.code}: {detail}") from exc
        data = sorted(body.get("data", []), key=lambda d: d.get("index", 0))
        vectors = [d.get("embedding", []) for d in data]
        if vectors and vectors[0]:
            self.dimension = len(vectors[0])
        return np.asarray(vectors, dtype=np.float32)


# ==============================================================================
# 本地 sentence-transformers
# ==============================================================================
class LocalEmbedder(BaseEmbedder):
    """本地 sentence-transformers 模型（需 ``pip install sentence-transformers``）。"""

    provider = "local"

    def __init__(self, config: Optional[Config] = None, cache: Optional[EmbeddingCache] = None) -> None:
        super().__init__(config, cache)
        cfg = (config or get_config()).embedding
        self.cache_dir = str(cfg.cache_dir)
        self._model: Any = None
        self._load_error: str = ""

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            self._load_error = (
                "未安装 sentence-transformers，请执行 `pip install sentence-transformers`，"
                "或把 embedding.provider 改为 hash / ollama"
            )
            raise RuntimeError(self._load_error) from exc
        model_path = Path(self.cache_dir) / self.model
        source = str(model_path) if model_path.exists() else self.model
        self._model = SentenceTransformer(source, cache_folder=self.cache_dir, device="cpu")
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        _log.info("本地 Embedding 模型已加载: %s (dim=%d)", source, self.dimension)
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


# ==============================================================================
# 工厂
# ==============================================================================
_REGISTRY = {
    "hash": HashEmbedder,
    "ollama": OllamaEmbedder,
    "openai": OpenAIEmbedder,
    "local": LocalEmbedder,
    "sentence-transformers": LocalEmbedder,
}


def create_embedder(
    config: Optional[Config] = None,
    provider: Optional[str] = None,
    use_cache: bool = True,
    cache_path: Optional[str] = None,
) -> BaseEmbedder:
    """按配置创建 Embedding 模型。

    Parameters
    ----------
    provider: 显式指定，覆盖配置（用于"不同 Embedding 模型对比实验"）
    use_cache: 是否启用持久化向量缓存
    cache_path: 缓存数据库路径，默认 ``<data_dir>/cache/embeddings.db``
    """
    cfg = config or get_config()
    name = (provider or str(cfg.embedding.provider) or "hash").lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        _log.warning("未知的 embedding.provider=%s，回退到 hash 离线向量化器", name)
        cls = HashEmbedder

    cache: Optional[EmbeddingCache] = None
    if use_cache:
        if cache_path is None:
            data_dir = cfg.resolve_path(str(cfg.paths.data_dir))
            cache_path = str(data_dir / "cache" / "embeddings.db")
        try:
            cache = EmbeddingCache(cache_path, model=f"{name}:{cfg.embedding.model}")
        except Exception as exc:  # noqa: BLE001 - 缓存失败不应阻断主流程
            _log.warning("Embedding 缓存初始化失败，已禁用缓存: %s", exc)
            cache = None

    embedder = cls(cfg, cache)
    _log.info(
        "Embedding 模型已创建: provider=%s model=%s dim=%d cache=%s",
        embedder.provider,
        embedder.model,
        embedder.dimension,
        "on" if cache else "off",
    )
    return embedder
