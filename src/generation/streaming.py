"""流式输出（模块二 5.2.2）。

两种流式形态
------------
1. **纯文本流** ``stream_text()``：直接产出增量文本，供 SSE / 打字机效果使用。
2. **事件流** ``stream_events()``：产出结构化事件（token / citation / done / error），
   这是前端真正消费的形态 —— 它让"答案文本"与"引用卡片"能分别、分时机地渲染：

   - ``retrieval`` 先到 → 前端立刻显示"检索到 5 个片段"与来源列表；
   - ``token`` 持续到 → 打字机效果；
   - ``citation`` 在正文出现引用标记时推送 → 引用卡片**动态出现**，
     而不是等答案全部生成完才一起显示。

引用标记的增量识别
------------------
难点：``【论文名 p.3】`` 可能被切成 ``【论`` + ``文名 p.`` + ``3】`` 多个 token。
本模块用**缓冲区 + 正则**处理：把新到的 token 累积进缓冲区，只把"确定不再是
引用开头"的部分吐出，遇到完整引用标记则转成 ``citation`` 事件。
这样既保证引用不漏，又不会阻塞打字机节奏。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from ..llm import estimate_usage
from ..utils.logger import get_logger
from ..utils.types import Citation, RetrievedChunk, TokenUsage

__all__ = ["StreamEvent", "EventType", "stream_text", "stream_events", "StreamBuffer"]

_log = get_logger("generation.streaming")

# 引用标记的起始字符（可能是【或半角 [）
_CITATION_OPEN = "【"
_PARTIAL_CITATION_MAX = 140  # 缓冲区上限，防止模型输出大量 "【" 导致卡顿


class EventType:
    """SSE 事件类型常量（与前端 app.js 的事件名严格对应）。"""

    # 通用生命周期
    START = "start"
    DONE = "done"
    ERROR = "error"
    METADATA = "metadata"

    # 生成相关
    RETRIEVAL = "retrieval"
    TOKEN = "token"
    CITATION = "citation"
    FINAL = "final"

    # Agent 轨迹相关
    ROUTE = "route"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"

    # 文档上传/索引进度相关
    PROGRESS = "progress"
    INDEXED = "indexed"
    WARNING = "warning"


@dataclass
class StreamEvent:
    """流式事件。``to_sse()`` 输出符合 SSE 协议的文本块。"""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """序列化为 SSE 报文：``event: xxx\\ndata: {...}\\n\\n``。"""
        import json

        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"event: {self.type}\ndata: {payload}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "data": self.data, "timestamp": self.timestamp}


@dataclass
class StreamBuffer:
    """增量引用识别缓冲区。

    工作方式：每次 ``feed()`` 返回"可以安全输出为普通文本"的前缀，
    并把可能是引用标记开头的尾巴留在缓冲区里等待后续 token。
    """

    pattern: re.Pattern
    pending: str = ""
    emitted_chars: int = 0

    def feed(self, piece: str) -> List[tuple[str, Optional[re.Match]]]:
        """喂入一个 token，返回 [(输出文本, 匹配到的引用或 None), ...]。

        正常情况下返回 1 个元素；当 token 补全了一个引用标记时返回 2 个元素
        （引用之前的文本 + 引用匹配）。
        """
        self.pending += piece
        results: List[tuple[str, Optional[re.Match]]] = []

        while True:
            match = self.pattern.search(self.pending)
            if match:
                before = self.pending[: match.start()]
                # 引用标记前的文本先输出（若为空则跳过，避免产生空 token）
                results.append((before, match))
                self.pending = self.pending[match.end() :]
                self.emitted_chars += len(before)
                continue
            break

        # 判断尾巴里是否有"可能是引用开头"的未完成片段
        open_index = self.pending.rfind(_CITATION_OPEN)
        if open_index == -1:
            # 没有未闭合的【 → 全部可安全输出
            safe, self.pending = self.pending, ""
        elif len(self.pending) - open_index > _PARTIAL_CITATION_MAX:
            # 缓冲区过长，判定它不是引用（模型可能只是打了半个方括号），直接输出
            safe, self.pending = self.pending, ""
        else:
            safe = self.pending[:open_index]
            self.pending = self.pending[open_index:]

        if safe:
            results.append((safe, None))
            self.emitted_chars += len(safe)
        return results

    def flush(self) -> str:
        """流结束时把缓冲区剩余内容吐出。"""
        rest, self.pending = self.pending, ""
        self.emitted_chars += len(rest)
        return rest


def stream_events(
    token_iter: Iterable[str],
    citations: Sequence[Citation] = (),
    emit_retrieval: bool = False,
    retrieved: Sequence[RetrievedChunk] = (),
    on_error: Optional[Callable[[Exception], None]] = None,
) -> Iterator[StreamEvent]:
    """把 token 迭代器转换为结构化事件流。

    Parameters
    ----------
    token_iter: LLM 客户端的 ``stream()`` 产出（增量文本）
    citations: 可用的引用列表（用于把 ``【来源 p.N】`` 映射成结构化引用卡片）
    emit_retrieval: 是否在流开头先推送一次 retrieval 事件
    retrieved: 检索结果（emit_retrieval=True 时使用）

    Yields
    ------
    :class:`StreamEvent`

    两个版本的答案
    --------------
    ``token`` 事件里的文本**去掉了引用标记**（标记改由独立的 ``citation``
    事件表达），这样"打字机"过程中用户不会看到裸露的 ``【论文.pdf p.3】``。
    但 ``done`` 事件同时给出 ``answer``（去标记）与 ``answer_raw``（**原样保留
    标记**）：落库、写缓存、评测与前端最终渲染都必须用后者 —— 前端 markdown
    渲染器正是靠 ``【来源 p.3】`` 生成可点击的行内引用角标（见 markdown.js）。
    早期实现只保留去标记版本，导致流式问答的最终答案**行内引用全部消失**，
    与非流式的 ``/api/ask`` 两条路径render结果不一致。
    """
    from .prompt_template import CITATION_PATTERN

    # 引用映射表：支持精确匹配与模糊匹配（模型可能写简称）
    by_source_page = {(c.source, c.page): c for c in citations}
    by_source: Dict[str, Citation] = {}
    for citation in citations:
        by_source.setdefault(citation.source.lower().strip(), citation)

    if emit_retrieval:
        yield StreamEvent(
            EventType.RETRIEVAL,
            {
                "num": len(retrieved),
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
                    for idx, r in enumerate(retrieved, start=1)
                ],
            },
        )

    buffer = StreamBuffer(pattern=CITATION_PATTERN)
    seen_citations: set[tuple[str, int]] = set()
    full_text: List[str] = []   # 去标记版（推给前端的"打字机"文本）
    raw_text: List[str] = []    # 原样版（含引用标记，落库/缓存/评测/最终渲染用）

    try:
        for piece in token_iter:
            if not piece:
                continue
            for text, match in buffer.feed(piece):
                if text:
                    full_text.append(text)
                    raw_text.append(text)
                    yield StreamEvent(EventType.TOKEN, {"text": text})
                if match is not None:
                    raw_text.append(match.group(0))  # 标记只保留在原文版里
                    source = match.group("source").strip()
                    page = int(match.group("page"))
                    key = (source, page)
                    if key in seen_citations:
                        continue
                    seen_citations.add(key)
                    citation = (
                        by_source_page.get(key)
                        or by_source.get(source.lower().strip())
                        or Citation(index=0, source=source, page=page)
                    )
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
                            "raw": match.group(0),
                        },
                    )
    except Exception as exc:  # noqa: BLE001
        if on_error:
            on_error(exc)
        _log.warning("流式生成中断: %s", exc)
        yield StreamEvent(EventType.ERROR, {"message": "生成过程中断", "detail": str(exc)[:300]})
        return

    tail = buffer.flush()
    if tail:
        full_text.append(tail)
        raw_text.append(tail)
        yield StreamEvent(EventType.TOKEN, {"text": tail})

    yield StreamEvent(
        EventType.DONE,
        {
            "answer": "".join(full_text),          # 去标记（兼容旧调用方）
            "answer_raw": "".join(raw_text),       # 权威版本：含引用标记
            "num_citations": len(seen_citations),
        },
    )


def stream_text(token_iter: Iterable[str]) -> Iterator[str]:
    """纯文本流（不做引用解析），用于 CLI 与简单场景。"""
    for piece in token_iter:
        if piece:
            yield piece


def collect_stream(events: Iterable[StreamEvent]) -> Dict[str, Any]:
    """把事件流聚合成最终结果（服务端需要保存完整答案时使用）。"""
    answer_parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None

    for event in events:
        if event.type == EventType.TOKEN:
            answer_parts.append(event.data.get("text", ""))
        elif event.type == EventType.CITATION:
            if event.data not in citations:
                citations.append(event.data)
        elif event.type == EventType.RETRIEVAL:
            retrieval = event.data
        elif event.type == EventType.ERROR:
            error = event.data
        elif event.type == EventType.DONE:
            # DONE 事件里的 answer 是权威版本（包含缓冲区尾部）。
            # 优先取 answer_raw：只有它保留了引用标记，与"落库/前端最终渲染"
            # 所依赖的版本一致。
            authoritative = event.data.get("answer_raw") or event.data.get("answer")
            if authoritative:
                answer_parts = [authoritative]

    answer = "".join(answer_parts)
    return {
        "answer": answer,
        "citations": citations,
        "retrieval": retrieval,
        "error": error,
        "usage": estimate_usage("", answer).to_dict(),
    }
