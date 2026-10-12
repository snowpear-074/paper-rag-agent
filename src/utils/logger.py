"""可观测性基础设施：结构化日志、请求追踪（Trace）、指标聚合（Metrics）。

三类输出
--------
1. 控制台 / 文件日志：人类可读，分级（DEBUG/INFO/WARNING/ERROR）。
2. Traces（``logs/traces.jsonl``）：每次问答的完整记录 —— 问题、检索结果、
   生成答案、耗时分解、Token 消耗、Agent 轨迹。模块五评测与 Bad Case 分析的数据来源。
3. Metrics（``logs/metrics.jsonl`` + 内存聚合）：命中率、工具成功率、延迟分位数等
   滚动指标，供前端可观测性面板展示。

线程安全：所有写盘操作均加锁；日志文件按天轮转。
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

from .types import AgentResult, RAGAnswer, TokenUsage, to_jsonable

__all__ = [
    "get_logger",
    "setup_logging",
    "TraceLogger",
    "MetricsCollector",
    "get_metrics",
    "tracer",
    "Timer",
]

# ------------------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------------------
_CONFIGURED = False
_SETUP_LOCK = threading.Lock()

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """带颜色的控制台格式化器（Windows 10+ 终端支持 ANSI）。"""

    def __init__(self, use_color: bool = True) -> None:
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s", "%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return text
        color = _LEVEL_COLORS.get(record.levelname, "")
        return f"{color}{text}{_RESET}" if color else text


def setup_logging(
    log_dir: Optional[Any] = None,
    level: str = "INFO",
    console_color: bool = True,
    force: bool = False,
) -> None:
    """初始化根日志器（幂等）。"""
    global _CONFIGURED
    with _SETUP_LOCK:
        if _CONFIGURED and not force:
            return
        root = logging.getLogger("rag_agent")
        root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        root.handlers.clear()
        root.propagate = False

        console = logging.StreamHandler()
        console.setFormatter(_ColorFormatter(console_color and _supports_color()))
        root.addHandler(console)

        if log_dir:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                directory / f"app_{datetime.now():%Y%m%d}.log", encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
                    "%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(file_handler)
        _install_secret_redaction()
        _CONFIGURED = True


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.name == "nt":
        return bool(os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM"))
    return hasattr(os, "isatty") and os.isatty(1)


def get_logger(name: str) -> logging.Logger:
    """获取子日志器：``get_logger("retrieval.hybrid")``。"""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"rag_agent.{name}" if not name.startswith("rag_agent") else name)


def _install_secret_redaction() -> None:
    """给 ``rag_agent`` 日志器及其**所有 handler** 装上密钥脱敏过滤器。

    为什么必要：异常栈里经常夹带请求头/URL（例如 ``Authorization: Bearer sk-...``），
    一旦写进日志文件就等于密钥泄露到磁盘。这个过滤器在日志落盘/上屏之前统一脱敏，
    避免"某处忘了处理"的疏漏。

    ⚠️ 必须挂在 **handler** 上，不能只挂 logger ——
    Python 的 logger 过滤器只对**直接**在该 logger 上产生的记录生效，
    子 logger（如 ``rag_agent.llm``）产生的记录会直接冒泡到祖先的 handler，
    不经过祖先 logger 的 filter。挂在 handler 上才能覆盖全部记录。
    """
    root = logging.getLogger("rag_agent")
    if not any(isinstance(f, _SecretRedactor) for f in root.filters):
        root.addFilter(_SecretRedactor())
    for handler in root.handlers:
        if not any(isinstance(f, _SecretRedactor) for f in handler.filters):
            handler.addFilter(_SecretRedactor())


class _SecretRedactor(logging.Filter):
    """把日志消息中形如 API Key 的片段替换为掩码。"""

    # sk- 开头的常见密钥格式、Bearer 令牌、api_key= / api_key: 赋值形式
    _PATTERNS = (
        re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
        re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9_\-\.]{12,}"),
        re.compile(r"(?i)\b(api[_-]?key[\"'\s:=]+)[A-Za-z0-9_\-]{12,}"),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            try:
                record.msg = redacted
                record.args = ()
            except Exception:  # noqa: BLE001
                pass
        return True


# 触发脱敏所需的"可疑子串"预筛；绝大多数文本不含它们，可直接返回，避免正则开销
_REDACT_HINTS = ("sk-", "key", "bearer")


def redact_secrets(text: str) -> str:
    """把任意文本中的 API Key 替换为掩码。

    与 :class:`_SecretRedactor` 共用同一套正则，但**可用于日志之外**的出口：
    用户可见的答案、``traces.jsonl`` / ``metrics.jsonl`` 落盘、异常详情等。
    日志过滤器只覆盖 ``logging`` 管道，这些出口必须显式调用本函数。
    """
    if not text:
        return text
    lowered = text.lower()
    if not any(hint in lowered for hint in _REDACT_HINTS):
        return text
    for pattern in _SecretRedactor._PATTERNS:
        text = pattern.sub(_mask_match, text)
    return text


def redact_value(value: Any, _depth: int = 0) -> Any:
    """递归脱敏 dict/list/str（用于 JSONL 落盘与内存记录）。"""
    if _depth > 6:
        return value
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {k: redact_value(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, _depth + 1) for v in value]
    return value


def _mask_match(match: "re.Match[str]") -> str:
    """把匹配到的密钥替换成 ``sk-abc****xyz`` 形式（保留可辨识的前后缀）。"""
    text = match.group(0)
    prefix = ""
    for lead in ("bearer ", "Bearer ", "BEARER "):
        if text.startswith(lead):
            prefix, text = lead, text[len(lead):]
            break
    for head in ("api_key=", "api_key:", "apikey=", "apikey:", "API_KEY=", "API_KEY:"):
        if text.lower().startswith(head.lower()):
            prefix, text = text[: len(head)], text[len(head):]
            break
    if len(text) <= 10:
        return prefix + "*" * len(text)
    return f"{prefix}{text[:6]}****{text[-4:]}"


# ------------------------------------------------------------------------------
# 计时器
# ------------------------------------------------------------------------------
class Timer:
    """上下文管理器计时器。

    >>> with Timer() as t:
    ...     pass
    >>> t.ms >= 0
    True
    """

    __slots__ = ("start", "end", "label")

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.start = 0.0
        self.end = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.end = time.perf_counter()
        return False

    @property
    def ms(self) -> float:
        end = self.end or time.perf_counter()
        return (end - self.start) * 1000.0

    @property
    def seconds(self) -> float:
        return self.ms / 1000.0


# ------------------------------------------------------------------------------
# Trace 日志
# ------------------------------------------------------------------------------
@dataclass
class TraceLogger:
    """把每次请求的结构化轨迹以 NDJSON 追加写入文件。

    同时保留最近 N 条在内存中，供前端"最近请求"面板与测试断言使用（避免反复读盘）。
    """

    path: Path = field(default_factory=lambda: Path("./logs/traces.jsonl"))
    memory_size: int = 200
    enabled: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _recent: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200), repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- 写入 -------------------------------------------------------------
    def _write(self, record: Dict[str, Any]) -> None:
        record.setdefault("timestamp", time.time())
        record.setdefault("datetime", datetime.now().isoformat(timespec="seconds"))
        # 落盘与入内存**之前**脱敏：答案文本里可能夹带网关回显的密钥，
        # 而 logging 过滤器只覆盖 logging 管道，管不到 NDJSON 与内存快照
        record = redact_value(record)
        with self._lock:
            self._recent.append(record)
            if not self.enabled:
                return
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass  # 日志失败绝不能影响主流程

    def log_rag(self, answer: RAGAnswer, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录一次 RAG 请求。"""
        record: Dict[str, Any] = {
            "kind": "rag",
            "trace_id": answer.trace_id,
            "question": answer.question,
            "answer": answer.answer[:2000],
            "num_retrieved": len(answer.retrieved),
            "retrieved": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "doc_id": r.chunk.doc_id,
                    "source": r.chunk.source,
                    "page": r.chunk.page,
                    "score": round(r.score, 4),
                    "confidence": round(r.confidence, 4),
                    "vector_score": round(r.vector_score, 4),
                    "bm25_score": round(r.bm25_score, 4),
                    "rerank_score": round(r.rerank_score, 4),
                    "route": r.source_route,
                    "preview": r.text[:200],
                }
                for r in answer.retrieved
            ],
            "citations": [c.to_dict() for c in answer.citations],
            "usage": answer.usage.to_dict(),
            "latency_ms": round(answer.latency_ms, 2),
            "retrieval_ms": round(answer.retrieval_ms, 2),
            "generation_ms": round(answer.generation_ms, 2),
            "cache_hit": answer.cache_hit,
            "degraded": answer.degraded,
            "degrade_reason": answer.degrade_reason,
            "model": answer.model,
            "top_score": round(answer.top_score, 4),
            "top_confidence": round(answer.top_confidence, 4),
            "hit": answer.hit,
        }
        if extra:
            record.update(extra)
        self._write(record)

    def log_agent(self, result: AgentResult, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录一次 Agent 请求。"""
        record: Dict[str, Any] = {
            "kind": "agent",
            "trace_id": result.trace_id,
            "session_id": result.session_id,
            "question": result.question,
            "answer": result.answer[:2000],
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "tool_success": result.tool_success,
            "tool_failures": result.tool_failures,
            "tool_success_rate": round(result.tool_success_rate, 4),
            "terminated_reason": result.terminated_reason,
            "routed_tools": result.routed_tools,
            "usage": result.usage.to_dict(),
            "latency_ms": round(result.latency_ms, 2),
            "trajectory": result.trajectory(),
        }
        if extra:
            record.update(extra)
        self._write(record)

    def log_event(self, kind: str, **fields: Any) -> None:
        """记录任意自定义事件（上传、索引重建、错误等）。"""
        self._write({"kind": kind, **fields})

    # -- 读取 -------------------------------------------------------------
    def recent(self, n: int = 20, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """读取最近 n 条内存中的 trace。"""
        items = list(self._recent)
        if kind:
            items = [r for r in items if r.get("kind") == kind]
        return items[-n:]

    def read_file(self, n: int = 100, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """从文件尾部读取最近 n 条 trace（大文件时只读尾部 2MB）。"""
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        read_bytes = min(size, 2 * 1024 * 1024)
        with self.path.open("rb") as fh:
            fh.seek(size - read_bytes)
            chunk = fh.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in chunk.splitlines() if ln.strip().startswith("{")]
        records: List[Dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind and rec.get("kind") != kind:
                continue
            records.append(rec)
        return records


# ------------------------------------------------------------------------------
# Metrics 聚合
# ------------------------------------------------------------------------------
@dataclass
class MetricsCollector:
    """滚动指标聚合器（线程安全，定长 deque 控制内存）。

    采集指标
    --------
    - RAG：请求数、检索命中率、缓存命中率、降级率、延迟分布、Token 消耗
    - Agent：工具调用成功率、平均推理轮次、终止原因分布
    - 工具：按工具名统计调用次数 / 成功率 / 平均耗时
    """

    window: int = 500
    path: Optional[Path] = None
    enabled: bool = True

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _rag_total: int = 0
    _rag_hit: int = 0
    _rag_cache: int = 0
    _rag_degraded: int = 0
    _rag_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _rag_retrieval_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _rag_tokens: int = 0
    _rag_top_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    # 「最近问题」由**两条链路**共同写入：RAG 流水线（record_rag）与 Agent（record_agent）。
    # 只记 RAG 那条的话，Web 端提问全走 Agent，指标面板这一块会**恒为空**。
    _recent_questions: Deque[str] = field(default_factory=lambda: deque(maxlen=200))

    _agent_total: int = 0
    _retrieval_total: int = 0
    _retrieval_hits: int = 0
    _retrieval_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _retrieval_top_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _agent_iterations: Deque[int] = field(default_factory=lambda: deque(maxlen=500))
    _agent_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _agent_tokens: int = 0
    _terminated: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    _tool_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _tool_success: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _tool_latency: Dict[str, Deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=200))
    )

    # -- 采集 -------------------------------------------------------------
    def record_rag(self, answer: RAGAnswer) -> None:
        with self._lock:
            self._rag_total += 1
            if answer.hit:
                self._rag_hit += 1
            if answer.cache_hit:
                self._rag_cache += 1
            if answer.degraded:
                self._rag_degraded += 1
            self._rag_latencies.append(answer.latency_ms)
            self._rag_retrieval_latencies.append(answer.retrieval_ms)
            self._rag_tokens += answer.usage.total_tokens
            self._rag_top_scores.append(answer.top_score)
            if answer.question:
                self._recent_questions.append(answer.question)
        self._persist({"kind": "rag", **self.rag_snapshot()})

    def record_agent(self, result: AgentResult) -> None:
        with self._lock:
            self._agent_total += 1
            self._agent_iterations.append(result.iterations)
            self._agent_latencies.append(result.latency_ms)
            self._agent_tokens += result.usage.total_tokens
            self._terminated[result.terminated_reason] += 1
            # 指标面板的「最近问题」：Agent 链路也要记，否则 Web 端提问一条都进不去
            if result.question:
                self._recent_questions.append(result.question)
            for step in result.steps:
                if step.tool_result is None:
                    continue
                name = step.tool_result.name or step.tool_name
                self._tool_calls[name] += 1
                if step.tool_result.ok:
                    self._tool_success[name] += 1
                self._tool_latency[name].append(step.tool_result.latency_ms)
        self._persist({"kind": "agent", **self.agent_snapshot()})

    def record_retrieval(self, latency_ms: float, top_score: float, hit: bool) -> None:
        """记录一次**裸检索**（不经过 RAG 流水线的那条路）。

        谁走这条路：Agent 的 ``knowledge_search`` 工具直接调
        :class:`~src.retrieval.engine.RetrievalEngine`，不构造 RAGAnswer，
        因此 :meth:`record_rag` 永远收不到它。而 Web 端的每一次提问都走 Agent ——
        少了这个入口，指标面板的「检索命中率 / 检索 P50 / 平均 Top 分数」
        在真实使用下会**恒为 0**，看起来像面板坏了。

        Parameters
        ----------
        latency_ms:
            检索耗时（毫秒）。
        top_score:
            Top-1 的**绝对置信度** ``confidence``（0~1）。
            注意不要传归一化后的 ``score`` —— 那个值只要命中就恒为 1.0，
            平均下来永远是 1，没有信息量。
        hit:
            本次检索是否返回了通过置信度门槛的结果。
        """
        with self._lock:
            self._retrieval_total += 1
            if hit:
                self._retrieval_hits += 1
            self._retrieval_latencies.append(latency_ms)
            self._retrieval_top_scores.append(top_score)

    def retrieval_snapshot(self) -> Dict[str, Any]:
        """Agent 直连检索的统计（与 RAG 流水线口径分开，便于对照）。"""
        with self._lock:
            total = self._retrieval_total
            latencies = list(self._retrieval_latencies)
            scores = list(self._retrieval_top_scores)
            return {
                "requests": total,
                "hit_rate": round(self._retrieval_hits / total, 4) if total else 0.0,
                "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
                "p50_latency_ms": self._percentile(latencies, 50),
                "p95_latency_ms": self._percentile(latencies, 95),
                "avg_top_score": round(statistics.fmean(scores), 4) if scores else 0.0,
            }

    def record_tool(self, name: str, ok: bool, latency_ms: float) -> None:
        with self._lock:
            self._tool_calls[name] += 1
            if ok:
                self._tool_success[name] += 1
            self._tool_latency[name].append(latency_ms)

    def record_event(self, kind: str, **fields: Any) -> None:
        self._persist({"kind": kind, **fields})

    def _persist(self, record: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            safe = redact_value(record)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(safe, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    # -- 快照 -------------------------------------------------------------
    @staticmethod
    def _percentile(values: Iterable[float], pct: float) -> float:
        data = sorted(v for v in values)
        if not data:
            return 0.0
        if len(data) == 1:
            return round(data[0], 2)
        index = min(len(data) - 1, max(0, int(round(pct / 100.0 * (len(data) - 1)))))
        return round(data[index], 2)

    def rag_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = self._rag_total
            latencies = list(self._rag_latencies)
            scores = list(self._rag_top_scores)
            return {
                "requests": total,
                "hit_rate": round(self._rag_hit / total, 4) if total else 0.0,
                "cache_hit_rate": round(self._rag_cache / total, 4) if total else 0.0,
                "degrade_rate": round(self._rag_degraded / total, 4) if total else 0.0,
                "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
                "p50_latency_ms": self._percentile(latencies, 50),
                "p95_latency_ms": self._percentile(latencies, 95),
                "retrieval_p50_ms": self._percentile(list(self._rag_retrieval_latencies), 50),
                "avg_top_score": round(statistics.fmean(scores), 4) if scores else 0.0,
                "total_tokens": self._rag_tokens,
                "avg_tokens": round(self._rag_tokens / total, 1) if total else 0.0,
            }

    def agent_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = self._agent_total
            iterations = list(self._agent_iterations)
            latencies = list(self._agent_latencies)
            calls = sum(self._tool_calls.values())
            success = sum(self._tool_success.values())
            return {
                "requests": total,
                "avg_iterations": round(statistics.fmean(iterations), 2) if iterations else 0.0,
                "max_iterations": max(iterations) if iterations else 0,
                "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
                "p95_latency_ms": self._percentile(latencies, 95),
                "total_tool_calls": calls,
                "tool_success_rate": round(success / calls, 4) if calls else 1.0,
                "total_tokens": self._agent_tokens,
                "terminated_reasons": dict(self._terminated),
            }

    def tool_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = []
            for name, calls in sorted(self._tool_calls.items()):
                latencies = list(self._tool_latency[name])
                rows.append(
                    {
                        "tool": name,
                        "calls": calls,
                        "success": self._tool_success[name],
                        "success_rate": round(self._tool_success[name] / calls, 4) if calls else 0.0,
                        "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
                    }
                )
            return rows

    def full_snapshot(self) -> Dict[str, Any]:
        """前端可观测性面板的完整数据。"""
        return {
            "rag": self.rag_snapshot(),
            "agent": self.agent_snapshot(),
            "retrieval": self.retrieval_snapshot(),
            "tools": self.tool_snapshot(),
            "recent_questions": list(self._recent_questions)[-10:],
            "timestamp": time.time(),
        }

    def reset(self) -> None:
        """清空所有指标（测试 / 前端"重置"按钮用）。"""
        with self._lock:
            self._rag_total = self._rag_hit = self._rag_cache = self._rag_degraded = 0
            self._rag_tokens = 0
            self._agent_total = self._agent_tokens = 0
            self._retrieval_total = self._retrieval_hits = 0
            for dq in (
                self._rag_latencies,
                self._rag_retrieval_latencies,
                self._rag_top_scores,
                self._recent_questions,
                self._agent_iterations,
                self._agent_latencies,
                self._retrieval_latencies,
                self._retrieval_top_scores,
            ):
                dq.clear()
            self._terminated.clear()
            self._tool_calls.clear()
            self._tool_success.clear()
            self._tool_latency.clear()


# ------------------------------------------------------------------------------
# 进程级单例
# ------------------------------------------------------------------------------
_tracer: Optional[TraceLogger] = None
_metrics: Optional[MetricsCollector] = None
_SINGLETON_LOCK = threading.Lock()


def configure(log_dir: Any, trace_file: Any, metrics_file: Any, level: str = "INFO",
              console_color: bool = True) -> None:
    """按配置初始化全局 tracer / metrics（由容器启动时调用一次）。"""
    global _tracer, _metrics
    with _SINGLETON_LOCK:
        setup_logging(log_dir, level=level, console_color=console_color, force=True)
        _tracer = TraceLogger(path=Path(trace_file))
        _metrics = MetricsCollector(path=Path(metrics_file))


def configure_from_config(config: Any, force: bool = False) -> None:
    """按 :class:`~src.utils.config.Config` 的 ``observability`` 段配置全局单例。

    与 :func:`configure` 的区别：本函数是**幂等且路径感知**的 ——
    若当前 tracer 已经在写同一个 trace 文件就跳过，避免每次构造组件都重建句柄。
    同时它会把完整配置传给 :func:`setup_logging`（含日志级别与彩色开关）。

    这样各入口（Web 服务、CLI、RAG 流水线单独使用、评测脚本）都能保证
    "日志写到配置指定的位置"，而不会各自持有互不相同的默认路径。
    """
    global _tracer, _metrics
    node = config.observability
    trace_path = Path(config.resolve_path(str(node.trace_file)))
    metrics_path = Path(config.resolve_path(str(node.metrics_file)))
    log_dir = config.resolve_path(str(node.log_dir))

    with _SINGLETON_LOCK:
        already = (
            _tracer is not None
            and Path(_tracer.path) == trace_path
            and _metrics is not None
            and _metrics.path is not None
            and Path(_metrics.path) == metrics_path
        )
        if already and not force:
            return
        setup_logging(
            log_dir,
            level=str(node.level),
            console_color=bool(node.console_color),
            force=True,
        )
        _tracer = TraceLogger(path=trace_path, enabled=True)
        _metrics = MetricsCollector(path=metrics_path, enabled=True)


def tracer() -> TraceLogger:
    global _tracer
    if _tracer is None:
        with _SINGLETON_LOCK:
            if _tracer is None:
                _tracer = TraceLogger()
    return _tracer


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        with _SINGLETON_LOCK:
            if _metrics is None:
                _metrics = MetricsCollector()
    return _metrics
