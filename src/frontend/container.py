"""系统容器 —— 全链路依赖装配与服务门面（模块四 5.4.1）。

为什么需要容器
--------------
前四个模块各自独立可用，但"系统集成"要求它们能装配成一个整体，并且：

1. **初始化顺序有依赖**：Embedding → 向量库 → 检索器 → RAG 流水线 → 工具 → Agent。
   散落在各处的全局变量会让测试与替换变得困难。
2. **资源要单例**：LLM 客户端、向量库、SQLite 连接池都必须进程内唯一，
   否则会出现"两个向量库实例各自持有一半索引"这类诡异问题。
3. **前端/CLI/评测都需要同一套服务**：把它们统一到 :class:`SystemContainer`
   的方法上，三种入口共享完全相同的业务逻辑，不会出现"CLI 能跑、Web 不能跑"。

容器职责
--------
- :meth:`initialize` —— 按序装配全部组件，返回健康状态
- :meth:`index_documents` / :meth:`index_files` —— 文档入库（带进度回调）
- :meth:`ask` —— 非流式问答（走 Agent）
- :meth:`ask_stream` —— 流式问答（SSE 事件流）
- :meth:`health` / :meth:`metrics` / :meth:`stats` —— 可观测性
- 会话管理代理（新建/列表/删除/反馈）
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..utils.config import Config, get_config
from ..utils.logger import configure as configure_logging
from ..utils.logger import get_logger, get_metrics, tracer
from ..utils.storage import ensure_dir
from ..utils.types import AgentResult, LoadedDocument, RAGAnswer, new_id

__all__ = ["SystemContainer", "get_container", "reset_container"]

_log = get_logger("frontend.container")

# 进程级单例
_CONTAINER: Optional["SystemContainer"] = None
_CONTAINER_LOCK = threading.Lock()


class SystemContainer:
    """全系统服务容器。"""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.started_at = time.time()
        self._initialized = False
        self._init_error = ""
        self._lock = threading.RLock()

        # 装配顺序即字段顺序，便于阅读
        self.loader: Any = None
        self.embedder: Any = None
        self.vector_store: Any = None
        self.retrieval: Any = None
        self.llm: Any = None
        self.cache: Any = None
        self.rag: Any = None
        self.memory: Any = None
        self.router: Any = None
        self.registry: Any = None
        self.agent: Any = None

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def initialize(self, force: bool = False) -> Dict[str, Any]:
        """按依赖顺序装配全部组件。

        任一非关键组件失败都不会中断装配（例如 chromadb 未安装 → 降级 numpy；
        联网搜索未配置 → 该工具标记不可用），保证系统总能启动。
        """
        with self._lock:
            if self._initialized and not force:
                return self.health()

            self.config.ensure_dirs()
            configure_logging(
                self.config.observability.log_dir,
                self.config.observability.trace_file,
                self.config.observability.metrics_file,
                level=str(self.config.observability.level),
                console_color=bool(self.config.observability.console_color),
            )

            errors: List[str] = []

            # 1) 文档加载器
            try:
                from ..data_loader import loader_from_config

                self.loader = loader_from_config(self.config)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"文档加载器: {exc}")
                _log.error("文档加载器初始化失败: %s", exc)

            # 2) LLM
            try:
                from ..llm import create_llm

                self.llm = create_llm(self.config)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"LLM: {exc}")
                _log.error("LLM 初始化失败: %s", exc)

            # 3) Embedding + 向量库 + 检索引擎
            try:
                from ..retrieval import create_retrieval_engine

                self.retrieval = create_retrieval_engine(self.config)
                self.embedder = self.retrieval.embedder
                self.vector_store = self.retrieval.vector_store
            except Exception as exc:  # noqa: BLE001
                errors.append(f"检索引擎: {exc}")
                _log.exception("检索引擎初始化失败")

            # 4) 语义缓存 + RAG 流水线
            try:
                from ..generation import create_cache, create_rag_pipeline

                self.cache = create_cache(self.config)
                self.rag = create_rag_pipeline(
                    config=self.config,
                    retrieval_engine=self.retrieval,
                    llm=self.llm,
                    cache=self.cache,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"RAG 流水线: {exc}")
                _log.exception("RAG 流水线初始化失败")

            # 5) 记忆（需要 LLM 做摘要压缩）
            try:
                from ..agent import create_memory

                self.memory = create_memory(self.config, llm=self.llm)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"对话记忆: {exc}")
                _log.error("对话记忆初始化失败: %s", exc)

            # 6) 工具注册表
            try:
                from ..agent.tools import build_default_registry

                self.registry = build_default_registry(
                    retrieval_engine=self.retrieval,
                    config=self.config,
                    rag_pipeline=self.rag,
                    llm=self.llm,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"工具注册表: {exc}")
                _log.exception("工具注册表初始化失败")

            # 7) 意图路由
            try:
                from ..agent import create_router

                self.router = create_router(self.config, llm=self.llm, registry=self.registry)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"意图路由: {exc}")
                self.router = None

            # 8) ReAct Agent
            try:
                from ..agent import create_agent

                self.agent = create_agent(
                    config=self.config,
                    registry=self.registry,
                    llm=self.llm,
                    memory=self.memory,
                    router=self.router,
                    retrieval_engine=self.retrieval,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Agent: {exc}")
                _log.exception("Agent 初始化失败")

            self._initialized = True
            self._init_error = "; ".join(errors)
            if errors:
                _log.warning("系统装配完成，但有 %d 个组件降级: %s", len(errors), self._init_error)
            else:
                _log.info("系统装配完成，全部组件正常")
            return self.health()

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------
    def index_documents(
        self,
        documents: Iterable[LoadedDocument],
        strategy: Optional[str] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """把已加载的文档索引入库。"""
        if self.retrieval is None:
            raise RuntimeError("检索引擎未初始化，无法建立索引")
        items = list(documents)
        reports: List[Dict[str, Any]] = []
        total = len(items)
        for idx, document in enumerate(items, start=1):
            if on_progress:
                on_progress(
                    {
                        "stage": "indexing",
                        "percent": round(100.0 * (idx - 1) / max(1, total), 1),
                        "message": f"[{idx}/{total}] 正在索引《{document.meta.title or document.doc_id}》",
                        "current": idx,
                        "total": total,
                    }
                )
            report = self.retrieval.index_document(
                document,
                strategy=strategy,
                progress=(
                    lambda info, _idx=idx: on_progress(
                        {
                            **info,
                            "percent": round(100.0 * ((_idx - 1) + info.get("percent", 0) / 100.0) / max(1, total), 1),
                            "current": _idx,
                            "total": total,
                        }
                    )
                    if on_progress
                    else None
                ),
            )
            reports.append(report)
        if on_progress:
            on_progress({"stage": "done", "percent": 100.0, "message": "索引完成", "current": total, "total": total})
        return {
            "num_documents": total,
            "succeeded": sum(1 for r in reports if r.get("ok")),
            "num_chunks": sum(int(r.get("num_chunks", 0)) for r in reports),
            "reports": reports,
        }

    def index_files(
        self,
        paths: Sequence[Any],
        strategy: Optional[str] = None,
        max_retries: int = 1,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """加载文件并索引入库（前端上传/批量导入的统一入口）。

        流程：批量加载（带重试）→ 逐篇索引（带进度）。
        """
        from ..data_loader import BatchImporter

        if self.loader is None:
            raise RuntimeError("文档加载器未初始化")
        importer = BatchImporter(self.loader, max_retries=max_retries)
        files: List[Path] = []
        for path in paths:
            files.extend(importer.collect_files(path))
        if not files:
            return {
                "ok": False,
                "error": "没有找到受支持的文件（PDF/DOCX/TXT/MD/CSV/JSON/BIB）",
                "num_documents": 0,
                "num_chunks": 0,
            }

        total = len(files)
        if on_progress:
            on_progress({"stage": "loading", "percent": 0.0, "message": f"开始解析 {total} 个文件"})

        documents: List[LoadedDocument] = []
        failures: List[Dict[str, Any]] = []
        for idx, path in enumerate(files, start=1):
            if on_progress:
                on_progress(
                    {
                        "stage": "parsing",
                        "percent": round(100.0 * (idx - 1) / total * 0.35, 1),
                        "message": f"[{idx}/{total}] 正在解析 {path.name}",
                        "current": idx,
                        "total": total,
                        "filename": path.name,
                    }
                )
            job, loaded = importer.import_documents([path], on_progress=None)
            for result in job.results:
                if result.ok and result.document is not None:
                    documents.append(result.document)
                else:
                    failures.append({"path": result.path, "error": result.error})

        index_report: Dict[str, Any] = {"num_chunks": 0, "reports": []}
        if documents:
            index_report = self.index_documents(
                documents,
                strategy=strategy,
                on_progress=(
                    lambda info: on_progress(
                        {
                            **info,
                            "percent": round(35.0 + 0.65 * float(info.get("percent", 0)), 1),
                        }
                    )
                    if on_progress
                    else None
                ),
            )

        return {
            "ok": not failures or bool(documents),
            "num_documents": len(documents),
            "num_chunks": int(index_report.get("num_chunks", 0)),
            "failed": failures,
            "reports": index_report.get("reports", []),
        }

    def index_bytes(
        self,
        filename: str,
        content: bytes,
        strategy: Optional[str] = None,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """把上传的字节内容落盘后索引（前端 base64 上传用）。"""
        upload_dir = self.config.resolve_path(str(self.config.paths.upload_dir))
        ensure_dir(upload_dir)
        safe_name = _safe_filename(filename)
        target = upload_dir / safe_name
        # 同名文件加时间戳后缀，避免覆盖
        if target.exists():
            target = upload_dir / f"{target.stem}_{int(time.time())}{target.suffix}"
        target.write_bytes(content)
        _log.info("已保存上传文件: %s (%d bytes)", target.name, len(content))
        return self.index_files([target], strategy=strategy, on_progress=on_progress)

    def delete_document(self, doc_id: str) -> int:
        """删除文档索引（向量库 + BM25 + 清单）。"""
        if self.retrieval is None:
            return 0
        return self.retrieval.delete_document(doc_id)

    def documents(self) -> List[Dict[str, Any]]:
        return self.retrieval.documents() if self.retrieval is not None else []

    def clear_knowledge_base(self) -> None:
        if self.retrieval is not None:
            self.retrieval.clear()
        if self.cache is not None:
            self.cache.clear()

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        session_id: str = "",
        use_agent: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """非流式问答，返回可直接 JSON 序列化的字典。

        ``use_agent=True``（默认）走完整 ReAct 决策链；
        为 False 时直接走 RAG 流水线（用于对比实验与降级场景）。
        """
        began = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return {"ok": False, "error": "问题不能为空", "answer": ""}

        if use_agent and self.agent is not None:
            result: AgentResult = self.agent.run(
                question, session_id=session_id or new_id("sess_"), context=context
            )
            return {
                "ok": True,
                "mode": "agent",
                "question": result.question,
                "answer": result.answer,
                "citations": [c.to_dict() for c in result.citations],
                "trajectory": result.trajectory(),
                "iterations": result.iterations,
                "tool_calls": result.tool_calls,
                "tool_success": result.tool_success,
                "tool_failures": result.tool_failures,
                "tool_success_rate": round(result.tool_success_rate, 4),
                "terminated_reason": result.terminated_reason,
                "routed_tools": result.routed_tools,
                "usage": result.usage.to_dict(),
                "latency_ms": round(result.latency_ms, 2),
                "session_id": result.session_id,
                "trace_id": result.trace_id,
                "degraded": result.degraded,
                "cache_hit": result.cache_hit,
            }

        if self.rag is None:
            return {"ok": False, "error": "RAG 流水线未初始化", "answer": ""}
        answer: RAGAnswer = self.rag.answer(question)
        return {
            "ok": True,
            "mode": "rag",
            "question": answer.question,
            "answer": answer.answer,
            "citations": [c.to_dict() for c in answer.citations],
            "retrieved": [
                {
                    "index": idx,
                    "source": r.chunk.source,
                    "page": r.chunk.page,
                    "score": round(r.score, 4),
                    "confidence": round(r.confidence, 4),
                    "preview": r.text[:200],
                }
                for idx, r in enumerate(answer.retrieved, start=1)
            ],
            "usage": answer.usage.to_dict(),
            "latency_ms": round(answer.latency_ms, 2),
            "retrieval_ms": round(answer.retrieval_ms, 2),
            "generation_ms": round(answer.generation_ms, 2),
            "cache_hit": answer.cache_hit,
            "degraded": answer.degraded,
            "degrade_reason": answer.degrade_reason,
            "trace_id": answer.trace_id,
        }

    def ask_stream(
        self,
        question: str,
        session_id: str = "",
        use_agent: bool = True,
    ) -> Iterable[Any]:
        """流式问答，产出 :class:`~src.generation.streaming.StreamEvent` 序列。

        实现方式：Agent 是**同步阻塞**的（ReAct 循环要等 LLM 与工具），
        因此把它放到工作线程执行，主线程从队列消费事件 —— 这样 SSE 能
        边跑边推，而不是等整个推理结束才一次性返回。
        """
        from ..generation.streaming import EventType, StreamEvent

        question = (question or "").strip()
        if not question:
            yield StreamEvent(EventType.ERROR, {"message": "问题不能为空"})
            return

        session_id = session_id or new_id("sess_")
        yield StreamEvent(EventType.START, {"session_id": session_id, "question": question})

        if not use_agent or self.agent is None:
            # 直接走 RAG 流式流水线
            if self.rag is None:
                yield StreamEvent(EventType.ERROR, {"message": "RAG 流水线未初始化"})
                return
            for event in self.rag.answer_stream(question):
                yield event
            return

        yield from self._agent_event_stream(question, session_id)

    def _agent_event_stream(self, question: str, session_id: str) -> Iterable[Any]:
        """把 Agent 的同步执行转换为事件流（工作线程 + 队列）。"""
        import queue
        import threading as _threading

        from ..generation.streaming import EventType, StreamEvent
        from ..utils.types import StepType

        event_queue: "queue.Queue[Any]" = queue.Queue()
        sentinel = object()

        def worker() -> None:
            try:
                def on_step(step: Any) -> None:
                    """把 Agent 步骤实时翻译成 SSE 事件。"""
                    if step.type == StepType.ROUTE:
                        event_queue.put(
                            StreamEvent(
                                EventType.ROUTE,
                                {"tools": [t for t in (step.tool_name or "").split(",") if t],
                                 "reason": step.content},
                            )
                        )
                    elif step.type == StepType.THOUGHT:
                        event_queue.put(
                            StreamEvent(EventType.THOUGHT, {"round": step.round, "content": step.content})
                        )
                    elif step.type == StepType.ACTION:
                        event_queue.put(
                            StreamEvent(
                                EventType.ACTION,
                                {
                                    "round": step.round,
                                    "tool": step.tool_name,
                                    "args": step.tool_args,
                                },
                            )
                        )
                    elif step.type == StepType.OBSERVATION:
                        payload = {
                            "round": step.round,
                            "tool": step.tool_name,
                            "status": step.tool_result.status.value if step.tool_result else "success",
                            "latency_ms": round(step.latency_ms, 2),
                            "preview": (step.content or "")[:300],
                            "content": step.content or "",
                        }
                        data = step.tool_result.data if step.tool_result else None
                        if isinstance(data, dict) and data.get("results"):
                            # 检索类工具：额外推送检索结果事件，前端可立刻渲染来源
                            event_queue.put(
                                StreamEvent(
                                    EventType.RETRIEVAL,
                                    {
                                        "num": len(data["results"]),
                                        "results": [
                                            {
                                                "index": item.get("index", i + 1),
                                                "chunk_id": item.get("chunk_id", ""),
                                                "doc_id": item.get("doc_id", ""),
                                                "source": item.get("source", ""),
                                                "page": item.get("page", 0),
                                                "score": item.get("score", 0.0),
                                                "confidence": item.get("confidence", 0.0),
                                                "route": item.get("route", ""),
                                                "preview": item.get("preview", ""),
                                            }
                                            for i, item in enumerate(data["results"])
                                        ],
                                    },
                                )
                            )
                        event_queue.put(StreamEvent(EventType.OBSERVATION, payload))
                    elif step.type == StepType.ERROR:
                        event_queue.put(
                            StreamEvent(EventType.ERROR, {"message": step.content, "recoverable": True})
                        )

                result = self.agent.run(question, session_id=session_id, on_step=on_step)
                event_queue.put(("result", result))
            except Exception as exc:  # noqa: BLE001
                _log.exception("Agent 流式执行失败")
                event_queue.put(("error", exc))
            finally:
                event_queue.put(sentinel)

        thread = _threading.Thread(target=worker, name="agent-stream", daemon=True)
        thread.start()

        collected_steps: List[Any] = []
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            if isinstance(item, tuple):
                kind, payload = item
                if kind == "error":
                    yield StreamEvent(
                        EventType.ERROR, {"message": "推理过程发生错误", "detail": str(payload)[:300]}
                    )
                    continue
                result = payload
                # 逐字推送最终答案，保持与 RAG 流式一致的打字机体验
                answer = result.answer or ""
                for chunk in _chunk_text(answer, 24):
                    yield StreamEvent(EventType.TOKEN, {"text": chunk})
                for citation in result.citations:
                    yield StreamEvent(
                        EventType.CITATION,
                        {
                            "index": citation.index,
                            "chunk_id": citation.chunk_id,
                            "doc_id": citation.doc_id,
                            "source": citation.source,
                            "page": citation.page,
                            "snippet": citation.snippet,
                            "score": citation.score,
                        },
                    )
                yield StreamEvent(
                    EventType.FINAL,
                    {
                        "answer": answer,
                        "citations": [c.to_dict() for c in result.citations],
                        "trajectory": result.trajectory(),
                        "iterations": result.iterations,
                        "tool_calls": result.tool_calls,
                        "tool_success": result.tool_success,
                        "tool_failures": result.tool_failures,
                        "tool_success_rate": round(result.tool_success_rate, 4),
                        "terminated_reason": result.terminated_reason,
                        "routed_tools": result.routed_tools,
                        "usage": result.usage.to_dict(),
                        "latency_ms": round(result.latency_ms, 2),
                        "session_id": result.session_id,
                        "degraded": result.degraded,
                        "cache_hit": result.cache_hit,
                        "trace_id": result.trace_id,
                    },
                )
                yield StreamEvent(EventType.DONE, {"trace_id": result.trace_id})
                continue
            yield item

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def create_session(self, title: str = "") -> Dict[str, Any]:
        if self.memory is None:
            return {"session_id": new_id("sess_"), "title": title or "新会话", "num_turns": 0}
        session = self.memory.create_session(title=title or "新会话")
        return {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "num_turns": session.num_turns,
        }

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.memory.list_sessions(limit=limit) if self.memory is not None else []

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """读取会话详情（含全部轮次，供前端恢复历史对话）。"""
        if self.memory is None:
            return None
        session = self.memory.get_session(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "summary": session.summary,
            "num_turns": session.num_turns,
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "question": turn.question,
                    "answer": turn.answer,
                    "citations": [c.to_dict() for c in turn.citations],
                    "trajectory": turn.trajectory,
                    "usage": turn.usage.to_dict(),
                    "latency_ms": turn.latency_ms,
                    "timestamp": turn.timestamp,
                    "feedback": turn.feedback,
                    # 恢复历史会话时前端需要知道"这轮是不是缓存命中/降级"，
                    # 否则只能靠"没有引用也没有轨迹"来猜（会把降级答案也标成缓存命中）
                    "metadata": turn.metadata,
                }
                for turn in session.turns
            ],
        }

    def delete_session(self, session_id: str) -> bool:
        return self.memory.delete_session(session_id) if self.memory is not None else False

    def rename_session(self, session_id: str, title: str) -> bool:
        return (
            self.memory.rename_session(session_id, title) if self.memory is not None else False
        )

    def set_feedback(self, turn_id: str, feedback: str) -> bool:
        return self.memory.set_feedback(turn_id, feedback) if self.memory is not None else False

    # ------------------------------------------------------------------
    # 可观测性
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """健康检查：逐组件探测（LLM 服务、向量库、Embedding 等）。"""
        health: Dict[str, Any] = {
            "ok": self._initialized,
            "version": "1.0.0",
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "init_errors": self._init_error,
        }
        if self.llm is not None:
            health["llm"] = self.llm.health_check()
        if self.embedder is not None:
            health["embedding"] = self.embedder.health_check()
        if self.retrieval is not None:
            health["retrieval"] = self.retrieval.health_check()
            health["num_documents"] = self.retrieval.num_documents
            health["num_chunks"] = self.retrieval.num_chunks
        if self.vector_store is not None:
            health["vector_store"] = self.vector_store.health_check()
        if self.agent is not None:
            health["tools"] = self.registry.health_check() if self.registry else {}
        if self.cache is not None:
            health["cache"] = self.cache.health_check()
        if self.memory is not None:
            health["memory"] = self.memory.health_check()
        if self.loader is not None:
            health["loader"] = self.loader.capability_report()
        health["ok"] = bool(self._initialized and self.retrieval is not None)
        return health

    def metrics(self) -> Dict[str, Any]:
        """指标快照（前端可观测性面板）。"""
        snapshot = get_metrics().full_snapshot()
        if self.cache is not None:
            snapshot["cache"] = self.cache.stats()
        if self.retrieval is not None:
            snapshot["knowledge_base"] = {
                "num_documents": self.retrieval.num_documents,
                "num_chunks": self.retrieval.num_chunks,
            }
        if self.agent is not None:
            snapshot["agent_detail"] = {
                k: v for k, v in self.agent.stats().items() if k not in ("tools", "config")
            }
        return snapshot

    def stats(self) -> Dict[str, Any]:
        return {
            "config": {
                "llm": {
                    "provider": str(self.config.llm.provider),
                    "model": str(self.config.llm.model),
                },
                "embedding": {
                    "provider": str(self.config.embedding.provider),
                    "model": str(self.config.embedding.model),
                    "dimension": int(self.config.embedding.dimension),
                },
                "vector_store": {"backend": str(self.config.vector_store.backend)},
                "retrieval": {
                    "top_k": int(self.config.retrieval.top_k),
                    "candidate_k": int(self.config.retrieval.candidate_k),
                    "enable_bm25": bool(self.config.retrieval.enable_bm25),
                    "enable_rerank": bool(self.config.retrieval.enable_rerank),
                    "rrf_k": int(self.config.retrieval.rrf_k),
                },
                "chunking": {
                    "strategy": str(self.config.chunking.strategy),
                    "chunk_size": int(self.config.chunking.chunk_size),
                    "chunk_overlap": int(self.config.chunking.chunk_overlap),
                },
                "agent": {
                    "max_iterations": int(self.config.agent.max_iterations),
                    "enable_router": bool(self.config.agent.enable_router),
                    "enable_parallel": bool(self.config.agent.enable_parallel),
                },
                "generation": {
                    "top_k": int(self.config.retrieval.top_k),
                    "cache_enabled": bool(self.config.get("generation.cache.enabled", True)),
                },
            },
            "retrieval": self.retrieval.stats() if self.retrieval is not None else {},
            "rag": self.rag.stats() if self.rag is not None else {},
            "agent": (
                {k: v for k, v in self.agent.stats().items() if k != "tools"}
                if self.agent is not None
                else {}
            ),
            "tools": self.registry.stats() if self.registry is not None else [],
            "memory": self.memory.stats() if self.memory is not None else {},
            "cache": self.cache.stats() if self.cache is not None else {},
            "router": self.router.stats() if self.router is not None else {},
        }

    def tools(self) -> List[Dict[str, Any]]:
        return self.registry.descriptions() if self.registry is not None else []

    def traces(self, n: int = 20, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        return tracer().read_file(n=n, kind=kind)


# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------
_RE_UNSAFE_FILENAME = __import__("re").compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_filename(filename: str) -> str:
    """清洗上传文件名，防止路径穿越与非法字符。"""
    name = Path(str(filename or "upload.bin")).name
    name = _RE_UNSAFE_FILENAME.sub("_", name).strip(". ")
    return name or f"upload_{int(time.time())}.bin"


def _chunk_text(text: str, size: int) -> Iterable[str]:
    """把长文本切成小块，用于模拟打字机流式输出。"""
    for start in range(0, len(text), size):
        yield text[start : start + size]


# ------------------------------------------------------------------------------
# 单例
# ------------------------------------------------------------------------------
def get_container(config: Optional[Config] = None, auto_init: bool = True) -> SystemContainer:
    """获取（并初始化）进程级容器单例。"""
    global _CONTAINER
    if _CONTAINER is None:
        with _CONTAINER_LOCK:
            if _CONTAINER is None:
                _CONTAINER = SystemContainer(config)
                if auto_init:
                    _CONTAINER.initialize()
    return _CONTAINER


def reset_container() -> None:
    """销毁单例（测试用）。"""
    global _CONTAINER
    with _CONTAINER_LOCK:
        _CONTAINER = None
