"""RAG 生成流水线（模块二总装）。

完整链路
--------
.. code-block:: text

    question
       │
       ├─[1] 语义缓存查询 ──命中──> 直接返回（cache_hit=True，跳过检索与生成）
       │
       ├─[2] 检索（模块一 HybridRetriever：向量 ‖ BM25 → RRF → Rerank）
       │
       ├─[3] 降级判定
       │       ├─ 无结果        → no_result     → 走 NO_CONTEXT_PROMPT（或直接提示）
       │       ├─ 相关性过低     → low_relevance → 提示用户并展示相关内容
       │       └─ 正常          → 继续
       │
       ├─[4] 上下文构建（按相关性排序 + 动态截断）
       │
       ├─[5] LLM 生成（同步 / 流式）
       │       └─ 调用失败 → llm_error 降级（返回友好提示 + 重试建议）
       │
       ├─[6] 引用解析（只保留答案中真正引用到的来源）
       │
       ├─[7] 写缓存 + 记录 Trace 日志（Token / 耗时 / 命中率）
       │
       └─> RAGAnswer

关键设计
--------
- **降级不是异常，而是正常返回值**：``RAGAnswer.degraded=True`` + ``degrade_reason``，
  上层（Agent / 前端）据此决定如何呈现，不会因检索空结果而抛异常。
- **可观测性内建**：每次调用都写 ``traces.jsonl`` 并更新内存指标，
  无需业务代码额外埋点。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from ..llm import estimate_usage
from ..utils.logger import get_logger, redact_secrets, tracer, get_metrics
from ..utils.types import (
    Citation,
    Message,
    RAGAnswer,
    RetrievedChunk,
    TokenUsage,
    stable_hash,
)

_log = get_logger("generation.rag")

__all__ = ["RAGPipeline", "RAGConfig", "create_rag_pipeline"]


# ------------------------------------------------------------------------------
# 生成参数
# ------------------------------------------------------------------------------
@dataclass
class RAGConfig:
    """生成层参数（由 ``config.generation`` 段构造）。"""

    top_k: int = 5
    retrieval_mode: str = "full"
    max_context_chars: int = 6000
    max_chunk_chars: int = 1500
    fallback_to_llm: bool = True
    low_relevance_threshold: float = 0.25
    temperature: float = 0.3
    max_tokens: int = 1024
    prompt_name: str = "rag_qa"
    use_cache: bool = True
    include_context_meta: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "RAGConfig":
        node = config.generation
        return cls(
            top_k=int(config.retrieval.top_k),
            retrieval_mode=str(config.get("generation.retrieval_mode", "full")),
            max_context_chars=int(node.max_context_chars),
            max_chunk_chars=int(node.max_chunk_chars),
            fallback_to_llm=bool(node.fallback_to_llm),
            low_relevance_threshold=float(config.retrieval.low_relevance_threshold),
            # temperature / max_tokens 属于 ``llm`` 段（生成参数由模型决定），
            # 早期实现遗漏了这两行，导致 config.yaml 里的 ``llm.max_tokens: 2048``
            # 永远被 dataclass 默认值 1024 覆盖 —— 长答案会被静默截断。
            temperature=float(config.get("llm.temperature", cls.temperature)),
            max_tokens=int(config.get("llm.max_tokens", cls.max_tokens)),
            prompt_name=str(config.get("generation.prompt_name", "rag_qa")),
            use_cache=bool(config.get("generation.cache.enabled", True)),
            include_context_meta=bool(config.get("generation.include_context_meta", True)),
        )


# ==============================================================================
# 流水线
# ==============================================================================
class RAGPipeline:
    """RAG 问答流水线。

    Parameters
    ----------
    retrieval_engine: 模块一的 :class:`~src.retrieval.engine.RetrievalEngine`
    llm: LLM 客户端
    cache: 语义缓存（可为 None 或 NullCache）
    embedder: Embedding 模型（用于语义缓存的问题向量）
    """

    def __init__(
        self,
        retrieval_engine: Any,
        llm: Any,
        cache: Optional[Any] = None,
        embedder: Optional[Any] = None,
        config: Any = None,
        rag_config: Optional[RAGConfig] = None,
    ) -> None:
        from ..utils.config import get_config
        from ..utils.logger import configure_from_config

        self.config = config or get_config()
        # 让可观测性输出遵循本流水线所使用的配置（若全局单例已指向同一路径则跳过）
        try:
            configure_from_config(self.config)
        except Exception as exc:  # noqa: BLE001
            _log.debug("可观测性配置同步失败: %s", exc)
        self.retrieval = retrieval_engine
        self.llm = llm
        self.embedder = embedder or getattr(retrieval_engine, "embedder", None)
        self.rag = rag_config or RAGConfig.from_config(self.config)

        if cache is None:
            try:
                from .cache import create_cache

                cache = create_cache(self.config)
            except Exception as exc:  # noqa: BLE001
                _log.warning("语义缓存初始化失败，已禁用: %s", exc)
                cache = None
        self.cache = cache
        self._stats = {"total": 0, "cache_hits": 0, "degraded": 0, "errors": 0}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        prompt_name: Optional[str] = None,
        use_cache: Optional[bool] = None,
        history: Optional[Sequence[Message]] = None,
        trace_extra: Optional[Dict[str, Any]] = None,
    ) -> RAGAnswer:
        """执行一次完整 RAG 问答（非流式）。"""
        began = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return RAGAnswer(question="", answer="请输入有效的问题。", degraded=True,
                             degrade_reason="empty_question")

        top_k = int(top_k or self.rag.top_k)
        mode = mode or self.rag.retrieval_mode
        use_cache = self.rag.use_cache if use_cache is None else use_cache
        self._stats["total"] += 1

        result = RAGAnswer(question=question, model=getattr(self.llm, "model", ""))

        # ---- [1] 语义缓存 ----
        query_vector: Optional[Any] = None
        if use_cache and self.cache is not None and self.embedder is not None:
            query_vector = self._embed_question(question)
            entry = self.cache.get(
                question,
                vector=query_vector,
                model=result.model,
                params=self._cache_params(mode, top_k, prompt_name, history),
            )
            if entry is not None:
                return self._from_cache(entry, question, result, began)

        # ---- [2] 检索 ----
        retrieval_started = time.perf_counter()
        retrieved = self._retrieve(question, top_k, mode)
        result.retrieved = retrieved
        result.retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        # ---- [3] 降级判定 ----
        degrade = self._check_degradation(retrieved)
        if degrade:
            result.degraded = True
            result.degrade_reason = degrade
            self._stats["degraded"] += 1

        # ---- [4] 上下文构建 ----
        context, selected = self._build_context(retrieved)
        result.retrieved = selected or retrieved
        result.citations = self._build_citations(selected)

        # ---- [5] 生成 ----
        messages = self._build_messages(question, context, degrade, prompt_name, history)
        result.prompt_preview = messages[-1].content[:600]
        generation_started = time.perf_counter()
        try:
            answer_text, usage = self.llm.chat_with_usage(
                messages, temperature=self.rag.temperature, max_tokens=self.rag.max_tokens
            )
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            _log.error("LLM 生成失败: %s", exc)
            answer_text, usage = self._llm_error_message(exc), TokenUsage()
            result.degraded = True
            result.degrade_reason = result.degrade_reason or "llm_error"
        result.generation_ms = (time.perf_counter() - generation_started) * 1000
        result.answer = answer_text
        result.usage = usage

        # ---- [6] 引用解析 ----
        from .prompt_template import parse_citations

        result.citations = parse_citations(answer_text, result.citations) or result.citations

        # ---- [7] 缓存与可观测 ----
        result.latency_ms = (time.perf_counter() - began) * 1000
        if use_cache and self.cache is not None and not result.degraded:
            self.cache.put(
                question,
                result,
                vector=query_vector,
                model=result.model,
                params=self._cache_params(mode, top_k, prompt_name, history, degrade),
            )
        self._observe(result, trace_extra)
        return result

    def answer_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        prompt_name: Optional[str] = None,
        use_cache: Optional[bool] = None,
        history: Optional[Sequence[Message]] = None,
        trace_extra: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Any]:
        """执行一次流式 RAG 问答。

        产出 :class:`~src.generation.streaming.StreamEvent` 序列：
        先是 ``retrieval``（若命中缓存则直接是 ``token``），随后 ``token`` /
        ``citation`` 交替，最后 ``final``。
        """
        from .streaming import EventType, StreamEvent, stream_events

        began = time.perf_counter()
        question = (question or "").strip()
        top_k = int(top_k or self.rag.top_k)
        mode = mode or self.rag.retrieval_mode
        use_cache = self.rag.use_cache if use_cache is None else use_cache
        self._stats["total"] += 1
        model = getattr(self.llm, "model", "")

        if not question:
            yield StreamEvent(EventType.ERROR, {"message": "请输入有效的问题。"})
            return

        # ---- 缓存 ----
        query_vector = None
        if use_cache and self.cache is not None and self.embedder is not None:
            query_vector = self._embed_question(question)
            entry = self.cache.get(
                question,
                vector=query_vector,
                model=model,
                params=self._cache_params(mode, top_k, prompt_name, history),
            )
            if entry is not None:
                _log.info("流式问答命中语义缓存: %r", question[:40])
                yield StreamEvent(EventType.TOKEN, {"text": entry.answer})
                yield StreamEvent(
                    EventType.FINAL,
                    {
                        "answer": entry.answer,
                        "citations": entry.citations,
                        "cache_hit": True,
                        "degraded": False,
                        "latency_ms": round((time.perf_counter() - began) * 1000, 2),
                        "usage": {"total_tokens": 0, "estimated": True},
                    },
                )
                return

        # ---- 检索 ----
        retrieval_started = time.perf_counter()
        retrieved = self._retrieve(question, top_k, mode)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        degrade = self._check_degradation(retrieved)
        if degrade:
            self._stats["degraded"] += 1

        context, selected = self._build_context(retrieved)
        citations = self._build_citations(selected)
        messages = self._build_messages(question, context, degrade, prompt_name, history)

        # 先推 retrieval，让前端立刻渲染来源列表
        yield StreamEvent(
            EventType.RETRIEVAL,
            {
                "num": len(selected or retrieved),
                "degraded": bool(degrade),
                "degrade_reason": degrade or "",
                "retrieval_ms": round(retrieval_ms, 2),
                "results": [
                    {
                        "index": idx,
                        "chunk_id": r.chunk.chunk_id,
                        "doc_id": r.chunk.doc_id,
                        "source": r.chunk.source,
                        "page": r.chunk.page,
                        "score": round(r.score, 4),
                        "route": r.source_route,
                        "preview": r.text[:200].replace("\n", " "),
                    }
                    for idx, r in enumerate(selected or retrieved, start=1)
                ],
            },
        )

        # ---- 生成 ----
        generation_started = time.perf_counter()
        answer_parts: List[str] = []
        raw_answer = ""          # 含引用标记的"权威答案"（落库/缓存/评测/最终渲染）
        stream_citations: List[Dict[str, Any]] = []
        error: Optional[Exception] = None

        try:
            token_iter = self.llm.stream(
                messages, temperature=self.rag.temperature, max_tokens=self.rag.max_tokens
            )
            for event in stream_events(token_iter, citations=citations):
                if event.type == EventType.TOKEN:
                    answer_parts.append(event.data.get("text", ""))
                elif event.type == EventType.CITATION:
                    stream_citations.append(event.data)
                elif event.type == EventType.DONE:
                    # DONE 不转发给前端，但要取出**含引用标记**的权威答案：
                    # 早期实现丢掉它、改用去标记的 TOKEN 拼接，导致流式回答
                    # 行内引用消失、写进缓存的答案也没有标记。
                    raw_answer = event.data.get("answer_raw") or event.data.get("answer") or ""
                    continue
                elif event.type == EventType.ERROR:
                    # 流中途断开：已产出的片段是"半句话"，直接落库会让用户
                    # 以为模型就是这么答的。这里补一段友好说明，并把失败计入
                    # 统计（早期实现既不补说明也不计 errors，静默交付残句）。
                    self._stats["errors"] += 1
                    degrade = degrade or "llm_error"
                    detail = str(event.data.get("detail") or event.data.get("message") or "")
                    error = RuntimeError(detail or "生成过程中断")
                    note = self._llm_error_message(error)
                    answer_parts.append("\n\n" + note)
                    raw_answer = ""  # 没有完整 DONE → 退回 TOKEN 拼接（含上面的说明）
                    yield event
                    yield StreamEvent(EventType.TOKEN, {"text": "\n\n" + note})
                    continue
                yield event
        except Exception as exc:  # noqa: BLE001
            error = exc
            self._stats["errors"] += 1
            message = self._llm_error_message(exc)
            answer_parts = [message]
            raw_answer = message
            degrade = degrade or "llm_error"
            yield StreamEvent(EventType.TOKEN, {"text": message})
            yield StreamEvent(EventType.ERROR, {"message": "生成失败", "detail": str(exc)[:300]})

        # 权威答案优先；没有 DONE（例如流被中断）时退回 TOKEN 拼接
        answer_text = raw_answer or "".join(answer_parts)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        usage = estimate_usage(
            "\n".join(m.content for m in messages), answer_text, model
        )

        # ---- 收尾：构造 RAGAnswer 并写缓存 / Trace ----
        final = RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved=list(selected or retrieved),
            usage=usage,
            latency_ms=(time.perf_counter() - began) * 1000,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            degraded=bool(degrade) or error is not None,
            degrade_reason=degrade or ("llm_error" if error else ""),
            model=model,
        )
        from .prompt_template import parse_citations

        parsed = parse_citations(answer_text, citations)
        if parsed:
            final.citations = parsed

        if use_cache and self.cache is not None and not final.degraded and final.answer:
            self.cache.put(
                question,
                final,
                vector=query_vector,
                model=model,
                params=self._cache_params(mode, top_k, prompt_name, history, degrade),
            )
        self._observe(final, trace_extra)

        yield StreamEvent(
            EventType.FINAL,
            {
                "answer": final.answer,
                "citations": [c.to_dict() for c in final.citations],
                "usage": final.usage.to_dict(),
                "latency_ms": round(final.latency_ms, 2),
                "retrieval_ms": round(final.retrieval_ms, 2),
                "generation_ms": round(final.generation_ms, 2),
                "cache_hit": False,
                "degraded": final.degraded,
                "degrade_reason": final.degrade_reason,
                "num_retrieved": len(final.retrieved),
                "trace_id": final.trace_id,
            },
        )

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------
    def _embed_question(self, question: str) -> Optional[Any]:
        if self.embedder is None:
            return None
        try:
            return self.embedder.encode(question)
        except Exception as exc:  # noqa: BLE001
            _log.debug("问题向量化失败（不影响主流程）: %s", exc)
            return None

    def _retrieve(self, question: str, top_k: int, mode: str) -> List[RetrievedChunk]:
        if self.retrieval is None:
            return []
        try:
            return self.retrieval.retrieve(question, top_k=top_k, mode=mode)
        except Exception as exc:  # noqa: BLE001
            _log.error("检索失败: %s", exc)
            return []

    def _check_degradation(self, retrieved: Sequence[RetrievedChunk]) -> str:
        """判定是否需要降级，返回降级原因（空字符串表示不降级）。

        ⚠️ 必须用 ``confidence``（绝对相关性置信度）而不是 ``score``：
        ``score`` 是 min-max 归一化后的相对分，只要检索有结果，Top-1 就恒为 1.0 ——
        用它判断会让"知识库中完全没有相关内容"的问题（如"今天天气怎么样"）
        也被判定为高相关，从而跳过降级、直接拿无关片段去生成答案。
        """
        if not retrieved:
            return "no_result"
        top_confidence = max(r.confidence for r in retrieved)
        if top_confidence < self.rag.low_relevance_threshold:
            return "low_relevance"
        return ""

    def _build_context(
        self, retrieved: Sequence[RetrievedChunk]
    ) -> tuple[str, List[RetrievedChunk]]:
        from .prompt_template import build_context

        return build_context(
            retrieved,
            max_chars=self.rag.max_context_chars,
            max_chunk_chars=self.rag.max_chunk_chars,
            include_meta=self.rag.include_context_meta,
        )

    @staticmethod
    def _build_citations(chunks: Sequence[RetrievedChunk]) -> List[Citation]:
        from .prompt_template import build_citations

        return build_citations(chunks)

    def _build_messages(
        self,
        question: str,
        context: str,
        degrade: str,
        prompt_name: Optional[str],
        history: Optional[Sequence[Message]],
    ) -> List[Message]:
        """组装发给 LLM 的消息列表。"""
        from .prompt_template import get_prompt

        # 模板选择与缓存键共用同一处判断（见 _effective_prompt_name）
        template = get_prompt(self._effective_prompt_name(prompt_name, degrade))

        if degrade == "no_result" and not self.rag.fallback_to_llm:
            # 明确不回退纯模型生成
            context = "（未检索到任何相关文档）"

        if template.name == "no_context":
            rendered = template.render(
                question=question,
                retrieval_note="检索结果：0 条（知识库可能为空，或问题与已上传论文无关）",
            )
        else:
            if degrade == "no_result":
                context = "（未检索到任何相关文档）"
            elif degrade == "low_relevance":
                context = (
                    "（注意：以下片段的检索相关度较低，请谨慎使用，"
                    "若无法支撑结论请明确说明）\n\n" + context
                )
            rendered = template.render(context=context or "（无可用上下文）", question=question)

        messages: List[Message] = [Message.system(rendered["system"])]
        # 注入历史（记忆管理已在 Agent 层做滑动窗口与摘要压缩，这里只做安全截断）
        if history:
            for message in list(history)[-8:]:
                if message.role.value in ("user", "assistant") and message.content:
                    messages.append(message)
        messages.append(Message.user(rendered["user"]))
        return messages

    def _effective_prompt_name(self, prompt_name: Optional[str], degrade: str = "") -> str:
        """返回本次请求**实际使用**的提示词模板名。

        ``_build_messages`` 在"零检索结果 + 关闭纯模型回退"时会临时切到
        ``no_context`` 模板。缓存键与消息组装必须共用这一处判断，
        否则"没有资料、模型凭自身知识作答"的结果会被当成"有资料支撑的答案"
        在后续请求里复用。
        """
        from .prompt_template import get_prompt

        if degrade == "no_result" and not self.rag.fallback_to_llm:
            return "no_context"
        try:
            return get_prompt(prompt_name or self.rag.prompt_name).name
        except Exception as exc:  # noqa: BLE001
            _log.debug("提示词模板 %r 解析失败，退回配置值: %s", prompt_name, exc)
            return str(prompt_name or self.rag.prompt_name)

    def _history_fingerprint(self, history: Optional[Sequence[Message]]) -> str:
        """把注入 LLM 的历史消息压成短指纹，参与缓存键。

        多轮追问（"它和 ResNet 相比如何？"）的**问题原文完全相同**，答案却
        取决于上文；不带历史维度的缓存键会让第二轮复用第一轮的答案。
        这里只取 ``_build_messages`` 真正会注入的那几条（末尾 8 条），
        避免历史里无关的早期轮次无谓地降低命中率。
        """
        if not history:
            return ""
        injected = [
            m for m in list(history)[-8:]
            if getattr(m, "role", None) is not None and getattr(m, "content", "")
        ]
        if not injected:
            return ""
        payload = "\n".join(f"{m.role.value if hasattr(m.role, 'value') else m.role}:{m.content}" for m in injected)
        return stable_hash(payload, 16)

    def _cache_params(
        self,
        mode: str,
        top_k: int,
        prompt_name: Optional[str] = None,
        history: Optional[Sequence[Message]] = None,
        degrade: str = "",
    ) -> Dict[str, Any]:
        """构造缓存键的"生成条件"分量。

        必须覆盖所有会影响答案文本的输入，否则会出现**错误的缓存命中**：

        - ``prompt``: 用**实际生效**的模板名（早期实现写死配置里的默认模板名，
          调用方显式传 ``prompt_name="paper_qa"`` 时，两种模板的答案会互相串用）；
        - ``hist``: 历史指纹（见 :meth:`_history_fingerprint`）；
        - ``max_tokens``: 截断上限不同则答案长度不同（见 RAGConfig.from_config）。
        """
        return {
            "mode": mode,
            "top_k": top_k,
            "prompt": self._effective_prompt_name(prompt_name, degrade),
            "hist": self._history_fingerprint(history),
            "temp": self.rag.temperature,
            "max_tokens": self.rag.max_tokens,
        }

    def _from_cache(
        self, entry: Any, question: str, result: RAGAnswer, began: float
    ) -> RAGAnswer:
        """由缓存条目还原 RAGAnswer。"""
        self._stats["cache_hits"] += 1
        result.answer = entry.answer
        result.cache_hit = True
        result.latency_ms = (time.perf_counter() - began) * 1000
        result.citations = [Citation(**c) for c in entry.citations if isinstance(c, dict)]
        result.retrieved = _rebuild_retrieved(entry.retrieved)
        _log.info("RAG 命中语义缓存: %r", question[:40])
        self._observe(result, {"cache_entry_question": entry.question})
        return result

    @staticmethod
    def _llm_error_message(exc: Exception) -> str:
        """把 LLM 异常转成对用户友好的提示（含可操作建议）。"""
        # 网关/代理的报错正文常把 Authorization 头原样回显，这里必须先脱敏再拼进
        # **用户可见的答案**；logging 的脱敏过滤器管不到这个出口
        detail = redact_secrets(str(exc))
        hint = "请稍后重试。"
        if "Ollama" in detail or "Connection" in detail or "无法连接" in detail:
            hint = (
                "本地模型服务未响应。请确认已执行 `ollama serve`，"
                "并已 `ollama pull qwen2.5:7b`；或在 config.yaml 中把 "
                "`llm.provider` 改为 `mock` 使用离线模拟模式。"
            )
        elif "401" in detail or "403" in detail or "api_key" in detail.lower():
            hint = "API Key 无效或未配置。请检查环境变量 `APP_LLM__API_KEY`。"
        elif "timeout" in detail.lower() or "超时" in detail:
            hint = "模型响应超时。可调大 `llm.timeout`，或减小 `generation.max_context_chars`。"
        elif "rate" in detail.lower() or "429" in detail:
            hint = "触发接口限流，请稍后重试或降低请求频率。"
        return (
            f"⚠️ 生成回答时发生错误，本次未能给出完整答案。\n\n"
            f"**原因**：{detail[:200]}\n\n"
            f"**建议**：{hint}\n\n"
            f"（检索本身已成功完成，你可以点击下方引用来源直接查看原文片段。）"
        )

    def _observe(self, result: RAGAnswer, extra: Optional[Dict[str, Any]]) -> None:
        """写入 Trace 日志与指标。"""
        try:
            tracer().log_rag(result, extra)
            get_metrics().record_rag(result)
        except Exception as exc:  # noqa: BLE001 - 观测失败绝不影响主流程
            _log.debug("可观测性记录失败: %s", exc)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        total = max(1, self._stats["total"])
        return {
            **self._stats,
            "cache_hit_rate": round(self._stats["cache_hits"] / total, 4),
            "degrade_rate": round(self._stats["degraded"] / total, 4),
            "config": {
                "top_k": self.rag.top_k,
                "retrieval_mode": self.rag.retrieval_mode,
                "max_context_chars": self.rag.max_context_chars,
                "prompt": self.rag.prompt_name,
                "use_cache": self.rag.use_cache,
            },
            "cache": self.cache.stats() if self.cache is not None else None,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "llm": self.llm.health_check() if self.llm is not None else {"ok": False},
            "cache": self.cache.health_check() if self.cache is not None else {"ok": True, "enabled": False},
            "pipeline": self.stats(),
        }

    def clear_cache(self) -> int:
        return self.cache.clear() if self.cache is not None else 0


def _rebuild_retrieved(records: Sequence[Dict[str, Any]]) -> List[RetrievedChunk]:
    """把缓存里的检索记录还原成 RetrievedChunk（用于前端展示引用片段）。"""
    from ..utils.types import Chunk

    out: List[RetrievedChunk] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        chunk = Chunk(
            chunk_id=str(record.get("chunk_id", "")),
            doc_id=str(record.get("doc_id", "")),
            text=str(record.get("text", "")),
            source=str(record.get("source", "")),
            page=int(record.get("page", 1) or 1),
        )
        out.append(
            RetrievedChunk(chunk=chunk, score=float(record.get("score", 0.0)), source_route="cache")
        )
    return out


# ==============================================================================
# 工厂
# ==============================================================================
def create_rag_pipeline(
    config: Any = None,
    retrieval_engine: Any = None,
    llm: Any = None,
    cache: Optional[Any] = None,
    **kwargs: Any,
) -> RAGPipeline:
    """组装 RAG 流水线（缺省参数按配置自动创建依赖）。"""
    from ..utils.config import get_config

    cfg = config or get_config()
    if retrieval_engine is None:
        from ..retrieval import create_retrieval_engine

        retrieval_engine = create_retrieval_engine(cfg)
    if llm is None:
        from ..llm import create_llm

        llm = create_llm(cfg)
    return RAGPipeline(
        retrieval_engine=retrieval_engine,
        llm=llm,
        cache=cache,
        embedder=getattr(retrieval_engine, "embedder", None),
        config=cfg,
        **kwargs,
    )
