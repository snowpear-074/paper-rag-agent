"""持久化与并发工具：JSON/NDJSON 读写、SQLite 连接管理、跨进程文件锁。

设计要点
--------
- **原子写**：先写临时文件再 ``os.replace``，避免断电/异常导致文件半写损坏。
- **SQLite 并发**：统一设置 ``WAL`` 模式 + ``busy_timeout``，并用进程内锁串行化
  写操作，规避 Windows 下 "database is locked" 报错。
- **跨进程文件锁**：Windows 用 ``msvcrt.locking``，POSIX 用 ``fcntl.flock``，
  两者都不可用时退化为进程内锁（不影响单机正确性）。
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from .logger import get_logger
from .types import to_jsonable

__all__ = [
    "ensure_dir",
    "read_json",
    "write_json",
    "append_ndjson",
    "read_ndjson",
    "open_sqlite",
    "FileLock",
    "human_size",
]

_log = get_logger("utils.storage")

PathLike = Union[str, os.PathLike]


def ensure_dir(path: PathLike) -> Path:
    """确保目录存在并返回 Path。空字符串返回当前目录。"""
    target = Path(path) if str(path) else Path(".")
    target.mkdir(parents=True, exist_ok=True)
    return target


# ------------------------------------------------------------------------------
# JSON
# ------------------------------------------------------------------------------
def read_json(path: PathLike, default: Any = None) -> Any:
    """读取 JSON 文件；不存在或损坏时返回 default。"""
    target = Path(path)
    if not target.exists():
        return default
    try:
        with target.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("读取 JSON 失败 %s: %s", target, exc)
        return default


def write_json(path: PathLike, data: Any, indent: int = 2) -> None:
    """原子写入 JSON 文件（临时文件 + os.replace）。"""
    target = Path(path)
    ensure_dir(target.parent)
    payload = json.dumps(to_jsonable(data), ensure_ascii=False, indent=indent, default=str)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def append_ndjson(path: PathLike, records: Iterable[Any]) -> int:
    """向 NDJSON 文件追加记录，返回写入条数。"""
    target = Path(path)
    ensure_dir(target.parent)
    count = 0
    with target.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(to_jsonable(record), ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def read_ndjson(path: PathLike, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """读取 NDJSON 文件；limit 表示只取最后 limit 条。"""
    target = Path(path)
    if not target.exists():
        return []
    records: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and len(records) > limit:
        return records[-limit:]
    return records


# ------------------------------------------------------------------------------
# SQLite
# ------------------------------------------------------------------------------
class _SqlitePool:
    """按线程缓存的 SQLite 连接池（sqlite3 连接默认不跨线程共享）。"""

    def __init__(self, path: PathLike) -> None:
        self.path = str(path)
        self._local = threading.local()
        self._write_lock = threading.RLock()

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            ensure_dir(Path(self.path).parent)
            conn = sqlite3.connect(self.path, timeout=15.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            with contextlib.suppress(sqlite3.DatabaseError):
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=15000")
            self._local.conn = conn
        return conn

    @contextlib.contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """写事务（进程内串行 + 自动提交/回滚）。"""
        conn = self.connection()
        with self._write_lock:
            try:
                yield conn
                conn.commit()
            except BaseException:
                with contextlib.suppress(sqlite3.DatabaseError):
                    conn.rollback()
                raise

    @contextlib.contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        yield self.connection()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with contextlib.suppress(sqlite3.DatabaseError):
                conn.close()
            self._local.conn = None


_POOLS: Dict[str, _SqlitePool] = {}
_POOL_LOCK = threading.Lock()


def open_sqlite(path: PathLike) -> _SqlitePool:
    """获取（并缓存）指定路径的连接池。"""
    key = str(Path(path).resolve())
    with _POOL_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = _SqlitePool(key)
            _POOLS[key] = pool
        return pool


# ------------------------------------------------------------------------------
# 跨进程文件锁
# ------------------------------------------------------------------------------
class FileLock:
    """简单的跨进程排他文件锁。

    主要用于：文档上传目录的并发写入、索引重建、评测脚本独占运行。
    超时后放弃获取并返回 False（调用方可选择降级运行）。
    """

    def __init__(self, path: PathLike, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self._fd: Optional[int] = None
        self._thread_lock = threading.Lock()
        self._acquired = False

    def acquire(self) -> bool:
        ensure_dir(self.path.parent)
        deadline = time.time() + self.timeout
        while True:
            if self._try_acquire():
                self._acquired = True
                return True
            if time.time() >= deadline:
                _log.warning("获取文件锁超时: %s", self.path)
                return False
            time.sleep(0.05)

    def _try_acquire(self) -> bool:
        try:
            self._fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR)
        except OSError:
            return self._thread_lock.acquire(blocking=False)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            os.close(self._fd)
            self._fd = None
            return False

    def release(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                if os.name == "nt":
                    import msvcrt

                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None
        if self._acquired:
            with contextlib.suppress(RuntimeError):
                self._thread_lock.release()
            self._acquired = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.release()
        return False


def human_size(num_bytes: int) -> str:
    """字节数转可读字符串：``1536 -> '1.5 KB'``。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"
