"""多轮对话记忆管理（模块三 5.3.4）。

三项能力（对应课程要求）
------------------------
1. **会话隔离**：每个 ``session_id`` 独立存储消息、摘要与统计，互不干扰。
2. **滑动窗口 Token 管理**：历史消息按"最近优先"装入上下文预算，
   超预算的旧消息被挤出窗口（而不是把整段历史一股脑塞给模型导致截断）。
3. **对话摘要压缩**：当轮次超过阈值时，把被挤出窗口的历史交给 LLM 压缩成
   一段摘要，替换原始消息。这样既能支持数十轮长对话，
   又能保住"用户关注什么论文、已确认过什么结论"这类关键上下文。

存储后端
--------
- ``sqlite``（默认）：跨进程/重启存活，WAL 模式，写操作串行化。
- ``memory``：纯内存，用于测试与无持久化场景。

数据结构
--------
``sessions`` 表：会话元信息 + 摘要
``turns``    表：每轮问答（问题/答案/引用/轨迹/Token/耗时/反馈）
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..utils.logger import get_logger
from ..utils.storage import open_sqlite
from ..utils.text import estimate_tokens
from ..utils.types import (
    Citation,
    ConversationTurn,
    Message,
    Role,
    Session,
    TokenUsage,
    new_id,
)

__all__ = ["ConversationMemory", "MemoryConfig", "create_memory", "NullMemory"]

_log = get_logger("agent.memory")


@dataclass
class MemoryConfig:
    """记忆参数（由 ``config.memory`` 段构造）。"""

    backend: str = "sqlite"
    path: str = "./data/memory/chat.db"
    max_turns: int = 20
    max_tokens: int = 3000
    summary_threshold: int = 12
    keep_recent_after_summary: int = 6

    @classmethod
    def from_config(cls, config: Any) -> "MemoryConfig":
        # 一律用点号路径 + 默认值读取：直接 str(config.memory) 之类写法会把
        # 整段 section 当成字段值，属于典型误用，故此处显式取值。
        return cls(
            backend=str(config.get("memory.backend", "sqlite")),
            path=str(config.get("memory.path", "./data/memory/chat.db")),
            max_turns=int(config.get("memory.max_turns", 20)),
            max_tokens=int(config.get("memory.max_tokens", 3000)),
            summary_threshold=int(config.get("memory.summary_threshold", 12)),
            keep_recent_after_summary=int(config.get("memory.keep_recent_after_summary", 6)),
        )


class ConversationMemory:
    """会话隔离的多轮对话记忆（SQLite / 内存双后端）。"""

    def __init__(self, config: Optional[MemoryConfig] = None, llm: Any = None) -> None:
        self.cfg = config or MemoryConfig()
        self.llm = llm
        self._lock = threading.RLock()
        self._sessions: Dict[str, Session] = {}
        self._pool = None

        if self.cfg.backend == "sqlite":
            try:
                self._pool = open_sqlite(self.cfg.path)
                self._init_schema()
            except Exception as exc:  # noqa: BLE001
                _log.warning("记忆数据库初始化失败，降级为内存后端: %s", exc)
                self._pool = None
        self._stats = {"turns": 0, "summaries": 0, "truncations": 0}

    # ------------------------------------------------------------------
    # 表结构
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        assert self._pool is not None
        with self._pool.write() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title      TEXT,
                    summary    TEXT,
                    created_at REAL,
                    updated_at REAL,
                    metadata   TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id     TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    seq         INTEGER,
                    question    TEXT,
                    answer      TEXT,
                    citations   TEXT,
                    trajectory  TEXT,
                    usage       TEXT,
                    latency_ms  REAL,
                    timestamp   REAL,
                    feedback    TEXT,
                    metadata    TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, seq)")

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------
    def create_session(
        self,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: str = "",
    ) -> Session:
        """新建会话。标题为空时用首轮问题自动命名。

        Parameters
        ----------
        session_id: 指定会话 ID（用于"调用方已生成 ID"的场景，保证幂等）。
        """
        session = Session(
            session_id=session_id or new_id("sess_"),
            title=title or "新会话",
            metadata=metadata or {},
        )
        with self._lock:
            self._sessions[session.session_id] = session
        if self._pool is not None:
            with self._pool.write() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sessions"
                    "(session_id, title, summary, created_at, updated_at, metadata) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        session.session_id,
                        session.title,
                        session.summary,
                        session.created_at,
                        session.updated_at,
                        json.dumps(session.metadata, ensure_ascii=False),
                    ),
                )
        _log.info("新建会话: %s (%s)", session.session_id, session.title)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """读取会话（含全部轮次）。"""
        if not session_id:
            return None
        with self._lock:
            cached = self._sessions.get(session_id)
            if cached is not None and cached.turns:
                return cached
        if self._pool is None:
            return self._sessions.get(session_id)

        with self._pool.read() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            turn_rows = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY seq ASC", (session_id,)
            ).fetchall()

        session = Session(
            session_id=row["session_id"],
            title=row["title"] or "新会话",
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
            summary=row["summary"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
            turns=[self._row_to_turn(r) for r in turn_rows],
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def ensure_session(self, session_id: str = "", title: str = "") -> Session:
        """获取会话；不存在则创建（``session_id`` 为空时也返回一个可用会话）。"""
        if session_id:
            existing = self.get_session(session_id)
            if existing is not None:
                return existing
            return self.create_session(title=title or "新会话", session_id=session_id)
        return self.create_session(title=title or "新会话")

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出会话摘要（前端会话列表）。"""
        if self._pool is None:
            with self._lock:
                sessions = list(self._sessions.values())
            sessions.sort(key=lambda s: -s.updated_at)
            return [
                {
                    "session_id": s.session_id,
                    "title": s.title,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "num_turns": s.num_turns,
                    "summary": s.summary[:120],
                }
                for s in sessions[:limit]
            ]

        with self._pool.read() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at, s.summary,
                       (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS num_turns
                FROM sessions s ORDER BY s.updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": row["title"] or "未命名会话",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "num_turns": row["num_turns"],
                "summary": (row["summary"] or "")[:120],
            }
            for row in rows
        ]

    def rename_session(self, session_id: str, title: str) -> bool:
        if not session_id or not title:
            return False
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].title = title
                self._sessions[session_id].updated_at = time.time()
        if self._pool is None:
            return session_id in self._sessions
        with self._pool.write() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, time.time(), session_id),
            )
            return (cursor.rowcount or 0) > 0

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其全部轮次。"""
        with self._lock:
            removed = self._sessions.pop(session_id, None) is not None
        if self._pool is None:
            return removed
        with self._pool.write() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return removed or (cursor.rowcount or 0) > 0

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
        if self._pool is not None:
            with self._pool.write() as conn:
                conn.execute("DELETE FROM turns")
                conn.execute("DELETE FROM sessions")

    # ------------------------------------------------------------------
    # 轮次写入
    # ------------------------------------------------------------------
    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        citations: Optional[Sequence[Any]] = None,
        trajectory: Optional[Sequence[Dict[str, Any]]] = None,
        usage: Optional[TokenUsage] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """记录一轮问答，并在必要时触发摘要压缩。"""
        session = self.ensure_session(session_id, title=question[:30] or "新会话")
        session_id = session.session_id

        turn = ConversationTurn(
            session_id=session_id,
            question=question,
            answer=answer,
            citations=[
                c if isinstance(c, Citation) else Citation(**c) if isinstance(c, dict) else Citation()
                for c in (citations or [])
            ],
            trajectory=list(trajectory or []),
            usage=usage or TokenUsage(),
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        with self._lock:
            session.turns.append(turn)
            session.updated_at = time.time()
            if session.title in ("新会话", "") and question:
                session.title = question[:30]
            self._stats["turns"] += 1

        if self._pool is not None:
            with self._pool.write() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO turns
                    (turn_id, session_id, seq, question, answer, citations, trajectory,
                     usage, latency_ms, timestamp, feedback, metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        turn.turn_id,
                        session_id,
                        len(session.turns) - 1,
                        turn.question,
                        turn.answer,
                        json.dumps([c.to_dict() for c in turn.citations], ensure_ascii=False),
                        json.dumps(turn.trajectory, ensure_ascii=False, default=str),
                        json.dumps(turn.usage.to_dict(), ensure_ascii=False),
                        turn.latency_ms,
                        turn.timestamp,
                        turn.feedback,
                        json.dumps(turn.metadata, ensure_ascii=False, default=str),
                    ),
                )
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (session.title, session.updated_at, session_id),
                )

        # 超阈值 → 摘要压缩
        if session.num_turns > self.cfg.summary_threshold:
            self.compress(session_id, force=False)
        return turn

    def set_feedback(self, turn_id: str, feedback: str) -> bool:
        """记录用户反馈（up/down）——模块五 Bad Case 分析的数据来源。"""
        with self._lock:
            for session in self._sessions.values():
                for turn in session.turns:
                    if turn.turn_id == turn_id:
                        turn.feedback = feedback
        if self._pool is None:
            return False
        with self._pool.write() as conn:
            cursor = conn.execute(
                "UPDATE turns SET feedback = ? WHERE turn_id = ?", (feedback, turn_id)
            )
            return (cursor.rowcount or 0) > 0

    # ------------------------------------------------------------------
    # 上下文构建（滑动窗口 + 摘要）
    # ------------------------------------------------------------------
    def _summary_upto(self, session: Session) -> int:
        """摘要已覆盖到的轮次下标（``turns[:upto]`` 已折叠进 summary）。

        为什么要单独记这个边界
        ----------------------
        滑动窗口若直接从 ``turns`` 末尾取 ``max_turns`` 条，会把**已经压缩进
        摘要的旧轮次**再原样送一遍：模型同时看到摘要和摘要的来源，既浪费
        Token 预算，又容易把同一件事当成两次对话。因此窗口必须从摘要边界之后
        开始。边界优先读 ``metadata["summary_upto"]``；旧数据没有该字段时，
        按压缩算法的不变式（总保留最后 ``keep_recent_after_summary`` 轮原文）推算。
        """
        if not session.summary:
            return 0
        stored = session.metadata.get("summary_upto")
        if isinstance(stored, int) and 0 <= stored <= len(session.turns):
            return stored
        keep = max(1, self.cfg.keep_recent_after_summary)
        return max(0, len(session.turns) - keep)

    def get_context(self, session_id: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """构建给 LLM 的对话上下文。

        Returns
        -------
        ``{"messages": [...], "summary": str, "hint": str, "num_turns": int,
           "truncated": bool, "tokens": int}``

        滑动窗口逻辑：先跳过已压缩进摘要的旧轮次，再从**最新一轮往前**累加消息，
        直到 Token 预算耗尽；被裁掉的部分记为 ``truncated``，并在 ``hint`` 中
        提示模型"更早的对话已压缩为摘要"。摘要本身也计入 Token 预算 —— 否则
        ``tokens`` 会系统性低报，让调用方误以为还有余量。
        """
        session = self.get_session(session_id) if session_id else None
        if session is None:
            return {"messages": [], "summary": "", "hint": "", "num_turns": 0, "truncated": False, "tokens": 0}

        budget = int(max_tokens or self.cfg.max_tokens)
        # 摘要已覆盖的旧轮次不再重复送出
        covered = self._summary_upto(session)
        available = session.turns[covered:]
        # 每个历史轮次贡献 2 条消息（user + assistant），assistant 只保留摘要化的片段
        window_turns = available[-self.cfg.max_turns :]

        picked: List[Message] = []
        used = estimate_tokens(session.summary) if session.summary else 0
        truncated = False
        for turn in reversed(window_turns):
            pair = [
                Message.user(turn.question),
                Message.assistant(self._shorten_answer(turn.answer)),
            ]
            pair_tokens = sum(estimate_tokens(m.content) for m in pair)
            if used + pair_tokens > budget and picked:
                truncated = True
                break
            picked = pair + picked
            used += pair_tokens

        hint_parts: List[str] = []
        if session.summary:
            hint_parts.append("更早的对话已压缩为摘要（见上方摘要）。")
        if truncated:
            self._stats["truncations"] += 1
            # 只统计"被 Token 预算挤掉"的轮次；已进摘要的不算省略
            hint_parts.append(
                f"为控制上下文长度，{len(available) - len(picked) // 2} 条更早的对话已省略，"
                "如用户提到早前内容请主动询问。"
            )

        return {
            "messages": picked,
            "summary": session.summary,
            "hint": " ".join(hint_parts),
            "num_turns": session.num_turns,
            "truncated": truncated,
            "tokens": used,
        }

    @staticmethod
    def _shorten_answer(answer: str, limit: int = 600) -> str:
        """历史答案只保留开头（结论通常在前），避免历史占用过多预算。"""
        text = (answer or "").strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "…（历史回答已截断）"

    # ------------------------------------------------------------------
    # 摘要压缩
    # ------------------------------------------------------------------
    def compress(self, session_id: str, force: bool = False) -> str:
        """把较早的对话压缩成摘要，替换原始历史。

        - 保留最近 ``keep_recent_after_summary`` 轮原始消息（细节不能丢）；
        - **只把尚未进摘要的那部分**交给 LLM 生成摘要（增量，不重复喂历史）；
        - 摘要与既有摘要**累积合并**（多次压缩不会丢历史要点）。

        为什么必须增量
        --------------
        早期实现每次都把 ``turns[:-keep]`` 全量重摘一遍，并把上一版摘要一起塞进
        提示词。于是第 N 次压缩要读 N 倍的历史，长会话的摘要成本是 O(n²)，而且
        模型会"反复总结摘要"—— 摘要越滚越抽象，早期细节逐轮丢失。改成只处理
        新跨越边界的那几轮后，成本回到线性，且旧要点以原文摘要形式原样保留。
        """
        session = self.get_session(session_id)
        if session is None or not session.turns:
            return ""

        keep = max(1, self.cfg.keep_recent_after_summary)
        if not force and session.num_turns <= self.cfg.summary_threshold:
            return session.summary
        if session.num_turns <= keep:
            return session.summary

        covered = self._summary_upto(session)          # 已被摘要覆盖的轮次
        target = len(session.turns) - keep             # 本次压缩后的新边界
        if target <= covered:
            # 没有新的轮次跨越边界（例如连续两次 force 压缩）
            return session.summary

        older = session.turns[covered:target]
        history_text = "\n".join(
            f"用户：{t.question}\n助手：{self._shorten_answer(t.answer, 300)}" for t in older
        )
        if session.summary:
            history_text = f"（已有摘要）{session.summary}\n\n【本次新增对话】\n{history_text}"

        summary = self._summarize(history_text, previous=session.summary)
        if not summary:
            return session.summary

        with self._lock:
            session.summary = summary
            # 记录边界：下一轮 get_context/compress 从这里继续
            session.metadata["summary_upto"] = target
            session.updated_at = time.time()
            self._stats["summaries"] += 1

        if self._pool is not None:
            with self._pool.write() as conn:
                conn.execute(
                    "UPDATE sessions SET summary = ?, updated_at = ?, metadata = ? "
                    "WHERE session_id = ?",
                    (
                        summary,
                        session.updated_at,
                        json.dumps(session.metadata, ensure_ascii=False, default=str),
                        session_id,
                    ),
                )
        _log.info("会话 %s 摘要压缩完成（新增 %d 轮，累计覆盖 %d/%d 轮）：%s",
                  session_id, len(older), target, session.num_turns, summary[:80])
        return summary

    def _summarize(self, history_text: str, previous: str = "") -> str:
        """调用 LLM 生成摘要；失败时用规则摘要兜底。

        ``previous`` 是上一版摘要，规则兜底时用它保住旧要点 —— 纯抽取式的
        兜底摘要只看得到本次新增的对话，不合并就等于把历史要点删了。
        """
        if self.llm is not None:
            try:
                from ..generation.prompt_template import MEMORY_SUMMARY_PROMPT

                rendered = MEMORY_SUMMARY_PROMPT.render(history=history_text[:6000])
                summary = self.llm.chat(
                    [Message.system(rendered["system"]), Message.user(rendered["user"])],
                    temperature=0.2,
                    max_tokens=300,
                )
                if summary and summary.strip():
                    return summary.strip()[:600]
            except Exception as exc:  # noqa: BLE001
                _log.warning("LLM 摘要生成失败，使用规则摘要: %s", exc)

        # 规则兜底：抽取历史中的问题作为要点
        questions = [t for t in history_text.split("\n") if t.startswith("用户：")]
        topics = list(dict.fromkeys(q[3:].strip()[:40] for q in questions))[-6:]
        if not topics:
            return previous
        points = "；".join(f"用户曾询问「{t}」" for t in topics)
        if not previous:
            return f"历史对话要点：{points}。"[:600]
        # 合并旧摘要：旧摘要已带"历史对话要点："标签，这里直接续写要点，
        # 否则会出现两个重复的标签。预算优先留给新要点，必要时截断旧摘要尾部。
        head = previous.strip().rstrip("。").strip()
        budget = max(0, 600 - len(points) - 2)
        if len(head) > budget:
            head = head[: max(0, budget - 1)].rstrip() + "…"
        return f"{head}；{points}。"[:600]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_turn(row: Any) -> ConversationTurn:
        return ConversationTurn(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            question=row["question"] or "",
            answer=row["answer"] or "",
            citations=[
                Citation(**c) for c in json.loads(row["citations"] or "[]") if isinstance(c, dict)
            ],
            trajectory=json.loads(row["trajectory"] or "[]"),
            usage=TokenUsage(**(json.loads(row["usage"] or "{}") or {})),
            latency_ms=row["latency_ms"] or 0.0,
            timestamp=row["timestamp"] or 0.0,
            feedback=row["feedback"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def stats(self) -> Dict[str, Any]:
        total_sessions = len(self.list_sessions(limit=10000))
        return {
            "backend": "sqlite" if self._pool is not None else "memory",
            "path": self.cfg.path if self._pool is not None else "",
            "sessions": total_sessions,
            **self._stats,
            "config": {
                "max_turns": self.cfg.max_turns,
                "max_tokens": self.cfg.max_tokens,
                "summary_threshold": self.cfg.summary_threshold,
                "keep_recent_after_summary": self.cfg.keep_recent_after_summary,
            },
        }

    def health_check(self) -> Dict[str, Any]:
        data = self.stats()
        data["ok"] = True
        return data


class NullMemory:
    """空记忆（关闭记忆功能时使用）。接口兼容，行为全部为空操作。"""

    enabled = False

    def create_session(
        self,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: str = "",
    ) -> Session:
        return Session(session_id=session_id or new_id("sess_"), title=title or "临时会话")

    def get_session(self, session_id: str) -> Optional[Session]:
        return None

    def ensure_session(self, session_id: str = "", title: str = "") -> Session:
        return Session(session_id=session_id or new_id("sess_"), title=title or "临时会话")

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return []

    def rename_session(self, session_id: str, title: str) -> bool:
        return False

    def delete_session(self, session_id: str) -> bool:
        return False

    def add_turn(
        self,
        session_id: str,
        question: str = "",
        answer: str = "",
        citations: Optional[Sequence[Any]] = None,
        trajectory: Optional[Sequence[Dict[str, Any]]] = None,
        usage: Optional[TokenUsage] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        **_extra: Any,
    ) -> ConversationTurn:
        """空记忆的 ``add_turn``：签名与真实实现保持一致（只是不落盘）。

        签名兼容性很重要 —— 否则"关闭记忆"这一配置会让 Agent 在调用时抛
        TypeError，把"可选功能关闭"变成"系统不可用"。
        """
        return ConversationTurn(
            session_id=session_id, question=question, answer=answer, latency_ms=latency_ms
        )

    def set_feedback(self, turn_id: str, feedback: str) -> bool:
        return False

    def get_context(self, session_id: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        return {"messages": [], "summary": "", "hint": "", "num_turns": 0, "truncated": False, "tokens": 0}

    def compress(self, session_id: str, force: bool = False) -> str:
        return ""

    def clear(self) -> None:
        return

    def stats(self) -> Dict[str, Any]:
        return {"backend": "none", "enabled": False, "sessions": 0, "turns": 0}

    def health_check(self) -> Dict[str, Any]:
        return {"ok": True, "enabled": False}


# ==============================================================================
# 工厂
# ==============================================================================
def create_memory(config: Any = None, llm: Any = None, **overrides: Any) -> Any:
    """按配置创建记忆模块。``memory.backend=none`` 时返回 :class:`NullMemory`。"""
    from ..utils.config import get_config

    cfg = config or get_config()
    backend = str(overrides.pop("backend", cfg.get("memory.backend", "sqlite"))).lower()
    if backend in ("none", "off", "disabled"):
        _log.info("对话记忆已关闭（memory.backend=none）")
        return NullMemory()

    mem_cfg = MemoryConfig.from_config(cfg)
    mem_cfg.backend = "sqlite" if backend == "sqlite" else "memory"
    mem_cfg.path = str(overrides.pop("path", cfg.resolve_path(mem_cfg.path)))
    mem_cfg.max_turns = int(overrides.pop("max_turns", mem_cfg.max_turns))
    mem_cfg.max_tokens = int(overrides.pop("max_tokens", mem_cfg.max_tokens))
    mem_cfg.summary_threshold = int(overrides.pop("summary_threshold", mem_cfg.summary_threshold))
    mem_cfg.keep_recent_after_summary = int(
        overrides.pop("keep_recent_after_summary", mem_cfg.keep_recent_after_summary)
    )
    memory = ConversationMemory(mem_cfg, llm=llm)
    _log.info(
        "对话记忆已创建: backend=%s max_turns=%d max_tokens=%d summary_threshold=%d",
        memory.cfg.backend,
        memory.cfg.max_turns,
        memory.cfg.max_tokens,
        memory.cfg.summary_threshold,
    )
    return memory
