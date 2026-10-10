"""全系统统一的数据结构契约（Data Contract）。

所有模块（检索层 / 生成层 / Agent 层 / 前端）都通过这些 dataclass 交换数据，
以此保证模块间松耦合 —— 任一模块的内部实现可以整体替换，只要维持这些结构不变。

层级关系
--------
Document ──1:N──> Chunk ──(检索)──> RetrievedChunk ──(生成)──> Citation / RAGAnswer
                                                          │
Agent: Session ─> ConversationTurn ─> AgentStep(Thought/Action/Observation) ─> AgentResult
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "Role",
    "Message",
    "DocumentMeta",
    "LoadedDocument",
    "Chunk",
    "RetrievedChunk",
    "Citation",
    "RAGAnswer",
    "ToolStatus",
    "ToolCall",
    "ToolResult",
    "StepType",
    "AgentStep",
    "AgentResult",
    "ConversationTurn",
    "Session",
    "TokenUsage",
    "new_id",
    "stable_hash",
    "to_jsonable",
    "from_dict",
]


# ==============================================================================
# 通用工具
# ==============================================================================
def new_id(prefix: str = "") -> str:
    """生成短随机 ID，如 ``paper_3f2a1b9c``。"""
    raw = uuid.uuid4().hex[:8]
    return f"{prefix}{raw}" if prefix else raw


def stable_hash(text: str, length: int = 16) -> str:
    """稳定内容哈希：同一文本永远得到同一 ID（用于幂等去重 / 增量索引）。"""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


def to_jsonable(obj: Any) -> Any:
    """递归把 dataclass / Enum / Path 转成可 JSON 序列化的结构。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_jsonable(obj.to_dict())
    return str(obj)


def from_dict(cls: type, data: Optional[Dict[str, Any]]) -> Any:
    """从 dict 还原 dataclass（忽略多余字段，容忍缺字段）。"""
    if not data:
        return cls()
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})  # type: ignore[arg-type]


class _JsonMixin:
    """为 dataclass 提供统一的序列化能力。"""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)  # type: ignore[return-value]

    def to_json(self, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]):
        return from_dict(cls, data)


# ==============================================================================
# 对话消息
# ==============================================================================
class Role(str, Enum):
    """对话角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class TokenUsage(_JsonMixin):
    """Token 消耗统计（无真实 tokenizer 时用字符数估算）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = True
    cost_yuan: float = 0.0
    # 价格是否已收录（未知模型时为 False，此时 cost_yuan 无意义，应显示"未收录"）
    pricing_known: bool = True

    def add(self, other: "TokenUsage") -> "TokenUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_yuan += other.cost_yuan
        self.pricing_known = self.pricing_known and other.pricing_known
        return self

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated=self.estimated and other.estimated,
            cost_yuan=self.cost_yuan + other.cost_yuan,
            pricing_known=self.pricing_known and other.pricing_known,
        )


@dataclass
class Message(_JsonMixin):
    """统一消息结构（OpenAI Chat 风格）。"""

    role: Role = Role.USER
    content: str = ""
    name: Optional[str] = None            # 工具名 / 发言者名
    tool_call_id: Optional[str] = None    # 工具结果回填时对应调用 ID
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            self.role = Role(self.role)

    @classmethod
    def system(cls, content: str, **meta: Any) -> "Message":
        return cls(role=Role.SYSTEM, content=content, metadata=meta)

    @classmethod
    def user(cls, content: str, **meta: Any) -> "Message":
        return cls(role=Role.USER, content=content, metadata=meta)

    @classmethod
    def assistant(cls, content: str, **meta: Any) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, metadata=meta)

    @classmethod
    def tool(cls, content: str, name: str, tool_call_id: str = "") -> "Message":
        return cls(role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id)

    def to_llm_dict(self) -> Dict[str, Any]:
        """转成发给 LLM 的最小字典（去掉本地字段，节省 token）。"""
        payload: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name and self.role == Role.TOOL:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


# ==============================================================================
# 文档与分块
# ==============================================================================
@dataclass
class DocumentMeta(_JsonMixin):
    """论文元信息（模块三"论文元信息提取工具"的数据来源）。"""

    doc_id: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    doi: str = ""
    venue: str = ""                       # 发表会议/期刊
    source_path: str = ""
    file_type: str = ""                   # pdf | docx | txt | md
    file_size: int = 0
    num_pages: int = 0
    num_chunks: int = 0
    language: str = "zh"
    created_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def short_ref(self) -> str:
        """生成简短引用名，如 "Attention Is All You Need (2017)"。"""
        title = self.title or self.source_path or self.doc_id
        if len(title) > 40:
            title = title[:37] + "..."
        return f"{title} ({self.year})" if self.year else title


@dataclass
class LoadedDocument(_JsonMixin):
    """文档加载器的统一输出。"""

    doc_id: str = ""
    text: str = ""                         # 全文纯文本
    pages: List[str] = field(default_factory=list)   # 按页/节的文本（用于页码溯源）
    meta: DocumentMeta = field(default_factory=DocumentMeta)
    load_errors: List[str] = field(default_factory=list)
    loader: str = ""                       # 实际使用的加载器名（便于对比实验）

    def page_of_offset(self, offset: int) -> int:
        """把全文偏移量映射回页码（1-based）。"""
        cursor = 0
        for idx, page in enumerate(self.pages, start=1):
            cursor += len(page) + 1
            if offset < cursor:
                return idx
        return max(1, len(self.pages))


@dataclass
class Chunk(_JsonMixin):
    """文本分块——索引与检索的最小单位。"""

    chunk_id: str = ""
    doc_id: str = ""
    text: str = ""
    index: int = 0                        # 在文档内的顺序号
    page: int = 1
    start_offset: int = 0
    end_offset: int = 0
    strategy: str = "recursive"           # 产生该块的切分策略
    source: str = ""                      # 文档标题/文件名（展示用）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = stable_hash(f"{self.doc_id}:{self.index}:{self.text[:200]}")
        if not self.source:
            self.source = self.metadata.get("title") or self.doc_id

    @property
    def num_chars(self) -> int:
        return len(self.text)

    def to_embed_text(self) -> str:
        """送入 Embedding 模型的文本（可加标题前缀提升召回）。"""
        title = self.metadata.get("title") or ""
        return f"{title}\n{self.text}" if title else self.text


@dataclass
class RetrievedChunk(_JsonMixin):
    """检索结果——带打分的 Chunk。

    关于两个"分数"的区别（重要）
    ----------------------------
    - :attr:`score`：**相对展示分**，经 min-max 归一化到 0~1，Top-1 恒为 1.0。
      它回答"这条比那条好多少"，适合前端画相关度进度条。
      但它**不能**用来判断"到底有没有检索到相关内容" —— 哪怕库里全是无关文档，
      Top-1 也仍然是 1.0。
    - :attr:`confidence`：**绝对相关性置信度**，保留排序器原始输出
      （启发式重排的 sigmoid 值 / 向量余弦相似度）。它回答"这条到底有多相关"，
      因此**降级判定（低相关性）必须用它**。

    早期版本只有 ``score`` 一个字段，导致"今天天气怎么样"这类与知识库完全无关的
    问题也会得到 1.00 的高分而无法触发降级 —— 这正是拆成两个字段的原因。
    """

    chunk: Chunk = field(default_factory=Chunk)
    score: float = 0.0                    # 归一化相对分（0~1），用于展示与排序
    confidence: float = 0.0               # 绝对置信度，用于降级判定与阈值过滤
    vector_score: float = 0.0             # 向量相似度
    bm25_score: float = 0.0               # BM25 分数（归一化后）
    rerank_score: float = 0.0             # 重排序原始分数
    vector_rank: int = 0                  # 向量路排名（0 表示未命中）
    bm25_rank: int = 0                    # BM25 路排名（0 表示未命中）
    source_route: str = ""                # vector | bm25 | hybrid | rerank

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def citation_label(self) -> str:
        return f"{self.chunk.source} p.{self.chunk.page}"


@dataclass
class Citation(_JsonMixin):
    """引用溯源信息。"""

    index: int = 0                        # 答案中的引用编号 [1][2]
    chunk_id: str = ""
    doc_id: str = ""
    source: str = ""
    page: int = 0
    snippet: str = ""
    score: float = 0.0

    def label(self) -> str:
        return f"{self.source} p.{self.page}"


# ==============================================================================
# RAG 生成层输出
# ==============================================================================
@dataclass
class RAGAnswer(_JsonMixin):
    """RAG 问答的完整输出。"""

    question: str = ""
    answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    cache_hit: bool = False
    degraded: bool = False                # 是否触发降级策略
    degrade_reason: str = ""              # no_result | low_relevance | llm_error
    model: str = ""
    prompt_preview: str = ""
    trace_id: str = field(default_factory=lambda: new_id("trace_"))

    @property
    def top_score(self) -> float:
        """Top-1 的相对展示分（归一化，恒为 1.0 当有结果时）。"""
        return self.retrieved[0].score if self.retrieved else 0.0

    @property
    def top_confidence(self) -> float:
        """Top-1 的绝对相关性置信度 —— **降级判定与质量评估请用这个**。"""
        return self.retrieved[0].confidence if self.retrieved else 0.0

    @property
    def hit(self) -> bool:
        """检索是否命中（有结果且分数不低于阈值）。"""
        return bool(self.retrieved)


# ==============================================================================
# Agent 决策层
# ==============================================================================
class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class StepType(str, Enum):
    """Agent 单步类型——对应 ReAct 的三种动作与起止标记。"""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"
    ERROR = "error"
    ROUTE = "route"          # 规则路由决策
    SUMMARY = "summary"      # 记忆压缩摘要


@dataclass
class ToolCall(_JsonMixin):
    """一次工具调用请求。"""

    call_id: str = field(default_factory=lambda: new_id("call_"))
    name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""                     # 触发该调用的推理
    round: int = 0
    parallel_group: str = ""              # 同一批并行调用的组 ID

    def signature(self) -> str:
        """调用指纹，用于死循环检测。"""
        try:
            args_repr = json.dumps(self.args, sort_keys=True, ensure_ascii=False)
        except TypeError:
            args_repr = str(self.args)
        return f"{self.name}::{args_repr}"


@dataclass
class ToolResult(_JsonMixin):
    """一次工具调用的结果。"""

    call_id: str = ""
    name: str = ""
    status: ToolStatus = ToolStatus.SUCCESS
    content: str = ""                     # 文本化观察结果（回填给 LLM）
    data: Any = None                      # 结构化结果（供前端渲染）
    error: str = ""
    latency_ms: float = 0.0
    retries: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS


@dataclass
class AgentStep(_JsonMixin):
    """Agent 轨迹树中的一个节点：Thought → Action → Observation。"""

    index: int = 0
    type: StepType = StepType.THOUGHT
    content: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[ToolResult] = None
    latency_ms: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    round: int = 0

    def render(self) -> str:
        """人类可读的单行摘要（终端调试 / 日志用）。"""
        if self.type == StepType.THOUGHT:
            return f"[Thought] {self.content}"
        if self.type == StepType.ACTION:
            return f"[Action] {self.tool_name}({json.dumps(self.tool_args, ensure_ascii=False)})"
        if self.type == StepType.OBSERVATION:
            preview = (self.content or "")[:160].replace("\n", " ")
            return f"[Observation] {preview}"
        return f"[{self.type.value}] {self.content[:160]}"


@dataclass
class AgentResult(_JsonMixin):
    """Agent 一次完整问答的输出。"""

    question: str = ""
    answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    iterations: int = 0
    tool_calls: int = 0
    tool_success: int = 0
    tool_failures: int = 0
    latency_ms: float = 0.0
    terminated_reason: str = "final_answer"   # final_answer | max_iterations | loop_detected | error
    routed_tools: List[str] = field(default_factory=list)
    cache_hit: bool = False
    degraded: bool = False
    session_id: str = ""
    trace_id: str = field(default_factory=lambda: new_id("atrace_"))

    @property
    def tool_success_rate(self) -> float:
        return self.tool_success / self.tool_calls if self.tool_calls else 1.0

    def trajectory(self) -> List[Dict[str, Any]]:
        """前端轨迹树所需的精简结构。"""
        return [
            {
                "index": s.index,
                "round": s.round,
                "type": s.type.value,
                "content": s.content,
                "tool": s.tool_name,
                "args": s.tool_args,
                "status": s.tool_result.status.value if s.tool_result else None,
                "latency_ms": round(s.latency_ms, 1),
            }
            for s in self.steps
        ]


# ==============================================================================
# 会话与记忆
# ==============================================================================
@dataclass
class ConversationTurn(_JsonMixin):
    """一轮完整交互（用户问题 + 助手回答 + 轨迹）。"""

    turn_id: str = field(default_factory=lambda: new_id("turn_"))
    session_id: str = ""
    question: str = ""
    answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    feedback: Optional[str] = None        # up | down
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session(_JsonMixin):
    """会话（会话隔离的载体）。"""

    session_id: str = field(default_factory=lambda: new_id("sess_"))
    title: str = "新会话"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turns: List[ConversationTurn] = field(default_factory=list)
    summary: str = ""                     # 长对话摘要压缩结果
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)


def dump_ndjson(path: Any, records: Iterable[Any]) -> None:
    """把一批 dataclass 以 NDJSON 形式落盘（可观测性日志用）。"""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
