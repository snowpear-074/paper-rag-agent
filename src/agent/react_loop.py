"""ReAct 核心循环（模块三 5.3.1）—— 手写实现，不依赖 LangChain 等框架的高层封装。

为什么手写
----------
框架把"推理-行动-观察"循环藏起来了，学生只能调用而无法理解：
提示词怎么组装、轨迹怎么维护、什么时候停、失败了怎么办。
本项目把这段逻辑完整摊开，每个决策点都有注释说明权衡。

循环结构
--------
.. code-block:: text

    ┌─ 1. 规则路由（可选，廉价意图识别，跳过多余推理轮次）
    │
    ├─ 2. 组装 System Prompt（角色 + 工具清单 + 格式约束）
    │      + Scratchpad（历史轨迹）+ 当前问题
    │
    ├─ 3. LLM 输出决策 JSON  {thought, action, action_input}
    │      └─ 解析失败 → 追加纠正提示重试（最多 2 次）
    │
    ├─ 4. action == final_answer ?
    │      是 → 结束，返回答案
    │      否 → 执行工具
    │            ├─ 同名同参重复调用 → 死循环计数 +1
    │            ├─ 超时 → 重试（最多 N 次）→ 仍失败则记录错误观察
    │            ├─ 异常 → 记录错误观察，提示模型改用其它工具
    │            └─ 并行：一轮多个独立工具时并发执行
    │
    ├─ 5. 观察结果截断后追加进 Scratchpad，回到第 3 步
    │
    └─ 6. 终止条件：final_answer / 达到最大轮次 / 检测到死循环 / 不可恢复错误
            达到最大轮次时，用"强制收尾"生成基于已有信息的答案

四大决策优化（课程要求 5.3.3）
------------------------------
1. **工具选择路由** ``router.py``：先用一次轻量分类决定候选工具集，
   缩小给模型的工具清单，减少选错概率与 Token 消耗。
2. **并行工具调用**：模型可在一次响应中给出 ``actions`` 数组（或路由判定
   需要多个工具时），循环用线程池并发执行。
3. **错误恢复**：工具超时/异常 → 重试 → 换工具 → 记录错误观察让模型自行调整。
4. **死循环检测**：滑动窗口内 ``(action, args)`` 指纹完全重复即强制终止，
   并用已有信息收尾。
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..llm import LLMError, extract_json
from ..utils.logger import get_logger, get_metrics, tracer
from ..utils.types import (
    AgentResult,
    AgentStep,
    Citation,
    Message,
    StepType,
    TokenUsage,
    ToolCall,
    ToolResult,
    ToolStatus,
    new_id,
)
from .base_tool import BaseTool, ToolRegistry

__all__ = ["ReActAgent", "AgentConfig", "create_agent"]

_log = get_logger("agent.react")


# ==============================================================================
# 循环状态与工具
# ==============================================================================
def _action_signature(action: Dict[str, Any]) -> str:
    """动作指纹：``工具名::参数JSON``（参数按键名排序，保证可比）。"""
    payload = action.get("action_input")
    try:
        args = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        args = str(payload)
    return f"{action.get('action')}::{args}"


class ToolTimeoutError(TimeoutError):
    """工具执行超时。

    不用 ``concurrent.futures.TimeoutError`` 的原因见
    :meth:`ReActAgent._call_with_timeout`：这里的超时必须在**不阻塞调用方**
    的前提下生效，因此由守护线程 + ``join(timeout)`` 实现。
    """


@dataclass
class _RunState:
    """单次 ``run()`` 的循环状态。

    为什么必须是**每次运行独立**的对象
    ----------------------------------
    早期实现把它挂在实例属性上（``self._tool_query_history``），并在 ``run()``
    开头重置。单线程下没问题，但前端服务是 ``ThreadingHTTPServer``：两个会话
    并发提问时，A 会话的重置会清空 B 会话刚记录的历史，B 会话随后的查询又会
    写进 A 的列表 —— 结果是**跨会话污染**，正常提问被误判成"语义重复"而提前
    终止（``loop_detected``）。把状态收进 per-run 对象后，并发运行天然隔离。
    """

    # 各工具已用过的查询文本（语义重复检测）
    tool_query_history: Dict[str, List[str]] = field(default_factory=dict)
    # 已出现过的 (action, args) 指纹（精确重复检测）
    seen_signatures: List[str] = field(default_factory=list)
    # 已就"重复调用"提醒过模型的次数（先提醒、再终止，避免误杀）
    loop_warnings: int = 0


# ==============================================================================
# 配置
# ==============================================================================
@dataclass
class AgentConfig:
    """Agent 行为参数（由 ``config.agent`` 段构造）。"""

    max_iterations: int = 8
    max_parallel_tools: int = 3
    tool_timeout: float = 45.0
    max_retries_per_tool: int = 2
    enable_router: bool = True
    enable_parallel: bool = True
    loop_detect_window: int = 2
    # 语义重复检测回看窗口：只在"该工具最近 N 条查询"里比对。
    # 早期实现拿全部历史比对，长会话末尾随口一问就可能撞上开头的旧查询而被误判成循环。
    loop_semantic_window: int = 6
    # 观察到重复调用时，先提醒模型几次；超过后才强制终止（误判的代价远高于多跑一轮）
    loop_max_warnings: int = 1
    max_observation_chars: int = 2000
    # 兜底答案/强制收尾时汇总观察结果的总预算（防止把 4 万字符塞进答案或提示词）
    max_aggregate_chars: int = 6000
    json_retries: int = 2
    temperature: float = 0.1
    max_tokens: int = 1024
    force_final_on_max_iterations: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "AgentConfig":
        node = config.agent
        return cls(
            max_iterations=int(node.max_iterations),
            max_parallel_tools=int(node.max_parallel_tools),
            tool_timeout=float(node.tool_timeout),
            max_retries_per_tool=int(node.max_retries_per_tool),
            enable_router=bool(node.enable_router),
            enable_parallel=bool(node.enable_parallel),
            loop_detect_window=int(node.loop_detect_window),
            loop_semantic_window=int(config.get("agent.loop_semantic_window", 6)),
            loop_max_warnings=int(config.get("agent.loop_max_warnings", 1)),
            max_observation_chars=int(node.max_observation_chars),
            max_aggregate_chars=int(config.get("agent.max_aggregate_chars", 6000)),
            temperature=float(config.get("agent.temperature", 0.1)),
            max_tokens=int(config.get("agent.max_tokens", 1024)),
        )


# ==============================================================================
# ReAct Agent
# ==============================================================================
class ReActAgent:
    """手写 ReAct Agent。

    典型用法::

        agent = create_agent(retrieval_engine=engine)
        result = agent.run("Transformer 用了什么数据集？", session_id="sess_1")
        print(result.answer)
        for step in result.steps:
            print(step.render())
    """

    def __init__(
        self,
        registry: ToolRegistry,
        llm: Any,
        memory: Optional[Any] = None,
        router: Optional[Any] = None,
        config: Any = None,
        agent_config: Optional[AgentConfig] = None,
    ) -> None:
        from ..utils.config import get_config

        self.config = config or get_config()
        self.registry = registry
        self.llm = llm
        self.memory = memory
        self.router = router
        self.cfg = agent_config or AgentConfig.from_config(self.config)
        self._stats = {"runs": 0, "tool_calls": 0, "failures": 0, "loops_detected": 0, "json_retries": 0}
        # 注意：循环状态（重复检测历史等）**不是**实例属性 —— 见 _RunState 的说明。
        # 同一个 Agent 实例可以被多个会话/线程并发调用，任何可变状态都必须按 run 隔离。
        self._stats_lock = threading.Lock()

    def _bump(self, key: str, delta: int = 1) -> None:
        """线程安全地累加统计计数。

        ``dict[key] += 1`` 是"读-改-写"三步，CPython 的 GIL 可能在任意两条
        字节码之间切换线程，并发调用时会丢计数（前端服务就是多线程的）。
        """
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + delta

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        session_id: str = "",
        history: Optional[Sequence[Message]] = None,
        context: Optional[Dict[str, Any]] = None,
        on_step: Optional[Callable[[AgentStep], None]] = None,
    ) -> AgentResult:
        """执行一次完整的 ReAct 推理循环。

        Parameters
        ----------
        question: 用户问题
        session_id: 会话 ID（用于记忆读写与可观测性分组）
        history: 历史消息（若提供 memory 则自动获取）
        context: 附加上下文（透传到 Trace 日志）
        on_step: 每产生一个步骤就回调（前端 SSE 实时推送轨迹）

        Returns
        -------
        :class:`AgentResult` —— 含答案、完整轨迹、Token 统计与终止原因。
        """
        began = time.perf_counter()
        question = (question or "").strip()
        result = AgentResult(question=question, session_id=session_id)
        self._bump("runs")
        # 本次运行的全部可变循环状态（并发安全的前提，见 _RunState）
        state = _RunState()

        if not question:
            result.answer = "请输入有效的问题。"
            result.terminated_reason = "error"
            return result

        # ---- 0) 会话与记忆 ----
        # 关键：session_id 必须在**进入循环前**就确定下来。
        # 早期实现等到收尾阶段才调用 ensure_session，导致每轮问答都会新建一个
        # 会话（因为传入的 session_id 为空），多轮对话的记忆完全无法累积。
        if self.memory is not None:
            try:
                session = self.memory.ensure_session(session_id, title=question[:30])
                session_id = getattr(session, "session_id", session_id) or session_id
            except Exception as exc:  # noqa: BLE001
                _log.warning("会话初始化失败（继续无记忆推理）: %s", exc)
        result.session_id = session_id

        memory_context = self._load_memory(session_id) if self.memory is not None else {}
        if history is None:
            history = memory_context.get("messages") or []

        # ---- 1) 规则路由 ----
        steps: List[AgentStep] = []
        candidate_tools: Optional[List[str]] = None
        if self.cfg.enable_router and self.router is not None:
            route = self.router.route(question, history=history)
            result.routed_tools = list(route.get("routes") or [])
            candidate_tools = route.get("tools")
            step = AgentStep(
                index=len(steps),
                type=StepType.ROUTE,
                content=f"路由判断：{route.get('reason', '')}（置信度 {route.get('confidence', 0):.2f}）",
                tool_name=",".join(result.routed_tools),
                round=0,
            )
            steps.append(step)
            self._emit(step, on_step)

            # 路由判定"无需工具"且置信度高 → 直接生成答案，跳过整个循环
            if route.get("direct") and float(route.get("confidence", 0)) >= 0.75:
                direct = self._direct_answer(question, history)
                if direct is not None:
                    answer, usage = direct
                    result.answer = answer
                    result.usage = usage
                    result.steps = steps
                    result.iterations = 0
                    result.terminated_reason = "direct_answer"
                    result.latency_ms = (time.perf_counter() - began) * 1000
                    self._finalize(result, context)
                    return result

        # ---- 2) 组装首轮提示词 ----
        system_prompt = self._build_system_prompt(candidate_tools)
        base_messages: List[Message] = [Message.system(system_prompt)]
        base_messages.extend(self._history_messages(history, memory_context))
        base_messages.append(
            Message.user(
                self._build_user_prompt(question, memory_context)
            )
        )

        # ---- 3) 推理循环 ----
        scratchpad: List[AgentStep] = []
        total_usage = TokenUsage()
        iterations = 0
        terminated = "final_answer"
        final_answer_text = ""

        for iteration in range(1, self.cfg.max_iterations + 1):
            iterations = iteration
            decision, decision_usage = self._decide(base_messages, scratchpad)
            total_usage = total_usage + decision_usage

            if decision is None:
                # JSON 解析彻底失败 → 终止
                terminated = "parse_error"
                final_answer_text = self._fallback_from_scratchpad(scratchpad, question)
                break

            thought = str(decision.get("thought") or "").strip()
            if thought:
                step = AgentStep(
                    index=len(steps), type=StepType.THOUGHT, content=thought,
                    round=iteration, usage=decision_usage,
                )
                steps.append(step)
                scratchpad.append(step)
                self._emit(step, on_step)

            actions = self._extract_actions(decision)

            # -- 决策里没有可识别的动作 --
            # 注意：这里**不能**把 thought 当成答案返回（早期实现这么做）——
            # 模型只要把 actions 写成字符串、或漏掉 action_input，它的"内心独白"
            # 就会直接变成给用户看的回答，既答非所问又暴露推理过程。
            # 正确做法是回一条纠错观察，让模型按格式重发。
            if not actions:
                observation = (
                    "错误：你的 JSON 中没有可识别的动作。请**只**输出形如 "
                    '{"thought": "...", "action": "工具名", "action_input": {...}} 的对象，'
                    '或在信息足够时输出 {"action": "final_answer", "action_input": {"answer": "..."}}。'
                )
                step = AgentStep(
                    index=len(steps), type=StepType.ERROR, content=observation, round=iteration
                )
                steps.append(step)
                scratchpad.append(step)
                self._emit(step, on_step)
                continue

            # -- 终局：本轮动作**全部**是 final_answer（混合批次不算终局）--
            terminal = [a for a in actions if a.get("action") == "final_answer"]
            if terminal and len(terminal) == len(actions):
                final_answer_text = self._extract_final_text(decision, actions)
                if final_answer_text:
                    result.terminated_reason = "final_answer"
                    break
                # action=final_answer 但没有内容 → 记一轮错误观察，继续循环
                observation = "错误：你选择了 final_answer，但 action_input.answer 为空。请在 answer 字段中给出完整回答。"
                step = AgentStep(
                    index=len(steps), type=StepType.ERROR, content=observation, round=iteration
                )
                steps.append(step)
                scratchpad.append(step)
                self._emit(step, on_step)
                continue
            if terminal:
                # 混合批次（例如同时给出 knowledge_search 与 final_answer）：
                # 丢掉 final_answer，只执行真正的工具调用，避免半句话就收尾。
                _log.info("本轮决策混用了 final_answer 与工具调用，仅执行工具调用")
                actions = [a for a in actions if a.get("action") != "final_answer"]

            # -- 死循环检测 --
            signatures = [self._signature(a) for a in actions]
            if self._is_looping(state, signatures):
                self._bump("loops_detected")
                if state.loop_warnings < self.cfg.loop_max_warnings:
                    # 先提醒、再终止：模型往往只是忘记了自己查过什么，
                    # 直接掐断会让一次本可答对的问题降级成"原始结果堆砌"。
                    state.loop_warnings += 1
                    _log.warning("检测到重复调用，先提醒模型换策略：%s", signatures)
                    observation = (
                        "错误：你正在重复调用已经执行过的工具与参数，它的结果就在上面的 Observation 里。\n"
                        "请换一种做法：① 直接基于已有信息给出 final_answer；"
                        "② 改变检索词/参数；③ 换用其它工具。"
                    )
                    step = AgentStep(
                        index=len(steps), type=StepType.ERROR, content=observation, round=iteration
                    )
                    steps.append(step)
                    scratchpad.append(step)
                    self._emit(step, on_step)
                    continue

                _log.warning("重复调用提醒无效，强制终止：%s", signatures)
                terminated = "loop_detected"
                step = AgentStep(
                    index=len(steps),
                    type=StepType.ERROR,
                    content="检测到重复调用相同工具与参数，已终止循环改用现有信息作答。",
                    round=iteration,
                )
                steps.append(step)
                scratchpad.append(step)
                self._emit(step, on_step)
                final_answer_text = self._fallback_from_scratchpad(scratchpad, question)
                break
            state.seen_signatures.extend(signatures)

            # -- 执行工具（可并行）--
            tool_calls = [self._to_tool_call(a, iteration) for a in actions]
            if self.cfg.enable_parallel and len(tool_calls) > 1:
                observations = self._execute_parallel(
                    tool_calls, steps, scratchpad, on_step, state
                )
            else:
                observations = [
                    self._execute_single(call, steps, scratchpad, on_step, state)
                    for call in tool_calls
                ]

            result.tool_calls += len(observations)
            result.tool_success += sum(1 for o in observations if o.ok)
            result.tool_failures += sum(1 for o in observations if not o.ok)
            self._bump("tool_calls", len(observations))
        else:
            # for 循环正常结束（未 break）→ 达到最大轮次
            terminated = "max_iterations"
            step = AgentStep(
                index=len(steps),
                type=StepType.ERROR,
                content=f"已达到最大推理轮次（{self.cfg.max_iterations}），将基于已收集的信息作答。",
                round=iterations,
            )
            steps.append(step)
            scratchpad.append(step)
            self._emit(step, on_step)

        # ---- 4) 收尾 ----
        if not final_answer_text:
            if terminated == "final_answer":
                final_answer_text = self._fallback_from_scratchpad(scratchpad, question)
            elif self.cfg.force_final_on_max_iterations:
                final_answer_text, force_usage = self._force_final_answer(question, scratchpad)
                total_usage = total_usage + force_usage
            else:
                final_answer_text = self._fallback_from_scratchpad(scratchpad, question)

        if not final_answer_text:
            final_answer_text = (
                "抱歉，我未能从知识库中收集到足够的信息来回答这个问题。\n\n"
                "可能的原因：知识库中还没有与问题相关的论文，或问题表述与论文原文差异较大。\n"
                "建议：先上传相关论文，或换用论文中出现过的术语重新提问。"
            )

        # 追加最终答案步骤
        final_step = AgentStep(
            index=len(steps), type=StepType.FINAL, content=final_answer_text, round=iterations,
            usage=total_usage,
        )
        steps.append(final_step)
        self._emit(final_step, on_step)

        # ---- 5) 汇总 ----
        result.answer = final_answer_text
        result.steps = steps
        result.iterations = iterations
        result.usage = total_usage
        result.terminated_reason = terminated
        result.citations = self._collect_citations(steps, final_answer_text)
        result.latency_ms = (time.perf_counter() - began) * 1000
        if terminated in ("max_iterations", "loop_detected", "parse_error"):
            result.degraded = True

        self._finalize(result, context)
        return result

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------
    def _decide(
        self, base_messages: Sequence[Message], scratchpad: Sequence[AgentStep]
    ) -> Tuple[Optional[Dict[str, Any]], TokenUsage]:
        """请求 LLM 给出下一步决策（带 JSON 容错重试）。"""
        from .prompts import render_scratchpad

        messages = list(base_messages)
        trail = render_scratchpad(scratchpad)
        if trail:
            messages.append(Message.assistant(trail))
            messages.append(
                Message.user(
                    "以上是你已经执行的步骤。请基于当前掌握的信息，"
                    "输出下一步的 JSON（只输出 JSON）。"
                )
            )

        usage = TokenUsage()
        for attempt in range(self.cfg.json_retries + 1):
            try:
                text, call_usage = self.llm.chat_with_usage(
                    messages, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens
                )
                usage = usage + call_usage
            except Exception as exc:  # noqa: BLE001
                # 必须捕获**所有**异常：不同 LLM 客户端（或测试替身）抛出的
                # 类型各不相同（LLMError / RuntimeError / URLError / 超时…），
                # 只捕获 LLMError 会让 Agent 在模型服务异常时直接崩溃，
                # 而"错误恢复"正是 Agent 必须具备的能力。
                _log.error("LLM 决策调用失败: %s", exc)
                return None, usage

            decision = extract_json(text)
            if decision is not None and ("action" in decision or "actions" in decision):
                return decision, usage

            self._stats["json_retries"] += 1
            _log.warning("决策 JSON 解析失败（第 %d 次）：%s", attempt + 1, text[:200].replace("\n", " "))
            if attempt < self.cfg.json_retries:
                messages = messages + [
                    Message.assistant(text),
                    Message.user(
                        "你的回复不是合法的决策 JSON。请**只**输出一个 JSON 对象，"
                        '形如 {"thought": "...", "action": "工具名", "action_input": {...}}，'
                        "不要包含任何其他文字或代码围栏。"
                    ),
                ]
        return None, usage

    @staticmethod
    def _extract_actions(decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从决策 JSON 中提取动作列表。

        支持两种形态：

        - 单动作：``{"action": "...", "action_input": {...}}``
        - 多动作（并行）：``{"actions": [{"action": "...", "action_input": {...}}, ...]}``

        两种形态同时出现时**只取 ``actions`` 数组**。早期实现把两者都追加进结果，
        于是模型给出 ``{"actions": [X], "action": X}``（同一动作写两遍的冗余输出）
        时 X 会被执行两次 —— 检索翻倍、引用重复、并行去重也失效。
        """
        raw: List[Any] = []
        if isinstance(decision.get("actions"), list):
            raw.extend(decision["actions"])
        elif decision.get("action"):
            # 仅在 actions 数组缺失/非法时才回退到单动作字段
            raw.append(
                {"action": decision["action"], "action_input": decision.get("action_input")}
            )
        elif isinstance(decision.get("actions"), dict):
            raw.append(decision["actions"])

        actions: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict) or not item.get("action"):
                continue
            action = {
                "action": str(item["action"]),
                "action_input": item.get("action_input") or item.get("input") or {},
            }
            fingerprint = _action_signature(action)
            if fingerprint in seen:  # 同一批内完全相同的动作只保留一次
                continue
            seen.add(fingerprint)
            actions.append(action)
        return actions

    @staticmethod
    def _extract_final_text(
        decision: Dict[str, Any], actions: Sequence[Dict[str, Any]]
    ) -> str:
        """从 final_answer 决策中取出答案文本（兼容多种字段写法）。"""
        payload = decision.get("action_input")
        if not isinstance(payload, dict):
            payload = {}
        for action in actions:
            if action.get("action") == "final_answer" and isinstance(action.get("action_input"), dict):
                payload = {**payload, **action["action_input"]}
        for key in ("answer", "final_answer", "response", "text", "content"):
            value = payload.get(key) or decision.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # 兜底：answer 字段缺失，但 final_answer 动作的 action_input 本身就是一段文本
        for action in actions:
            if action.get("action") != "final_answer":
                continue
            raw = action.get("action_input")
            if isinstance(raw, str) and len(raw.strip()) >= 2:
                return raw.strip()
        # 注意：**不要**把 thought 当答案返回。
        # 早期实现有这一步"兜底"，结果是模型只要把 action 写成不可解析的形式
        # （例如 actions 传成字符串），它的推理独白就会被当成正式回答发给用户：
        # 既答非所问，又把内部推理过程暴露出去。宁可返回空串，让主循环回一条
        # 格式纠错观察，逼模型重发一次。
        return ""

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    def _to_tool_call(self, action: Dict[str, Any], round_no: int) -> ToolCall:
        args = action.get("action_input")
        if not isinstance(args, dict):
            args = {}
        return ToolCall(
            call_id=new_id("call_"),
            name=str(action.get("action") or ""),
            args=args,
            round=round_no,
        )

    def _execute_single(
        self,
        call: ToolCall,
        steps: List[AgentStep],
        scratchpad: List[AgentStep],
        on_step: Optional[Callable[[AgentStep], None]],
        state: _RunState,
    ) -> ToolResult:
        """执行单个工具（含超时控制与重试）。"""
        self._record_tool_query(state, call)
        action_step = AgentStep(
            index=len(steps),
            type=StepType.ACTION,
            content="",
            tool_name=call.name,
            tool_args=call.args,
            round=call.round,
        )
        steps.append(action_step)
        # Action 步骤也必须进 Scratchpad：下一轮提示词由 Scratchpad 渲染，
        # 缺了它模型只看得到一串 Observation，无从判断"这个结果是哪个查询换来的"，
        # 于是反复换词重搜同一个问题（正是死循环检测第二级要拦的行为）。
        scratchpad.append(action_step)
        self._emit(action_step, on_step)

        tool_result = self._run_with_retry(call)

        observation_step = AgentStep(
            index=len(steps),
            type=StepType.OBSERVATION if tool_result.ok else StepType.ERROR,
            content=self._truncate_observation(tool_result),
            tool_name=call.name,
            tool_args=call.args,
            tool_result=tool_result,
            latency_ms=tool_result.latency_ms,
            round=call.round,
        )
        steps.append(observation_step)
        scratchpad.append(observation_step)
        self._emit(observation_step, on_step)
        return tool_result

    def _execute_parallel(
        self,
        calls: Sequence[ToolCall],
        steps: List[AgentStep],
        scratchpad: List[AgentStep],
        on_step: Optional[Callable[[AgentStep], None]],
        state: _RunState,
    ) -> List[ToolResult]:
        """并发执行多个工具（课程要求 5.3.3 的"并行工具调用"）。

        为保持**轨迹顺序可读**，并行执行本身是并发的，但结果按原始顺序
        依次追加到轨迹中（先发 Action 列表，再按顺序落 Observation）。

        ``max_parallel_tools`` 约束的是**并发度**（线程池大小），不是"接单上限"：
        早期实现写成 ``calls[:max_parallel_tools]``，模型一次给出 4 个调用时
        第 4 个被静默丢弃 —— 模型等不到它的 Observation，只能再问一遍，
        白白多烧一轮。超出并发度的调用改为排队执行（由线程池自动调度）。
        """
        limited = list(calls)
        if len(limited) > self.cfg.max_parallel_tools:
            _log.info(
                "模型本轮给出 %d 个工具调用，并发度限制为 %d（其余排队执行）",
                len(limited), self.cfg.max_parallel_tools,
            )
        group_id = new_id("grp_")

        for call in limited:
            call.parallel_group = group_id
            self._record_tool_query(state, call)
            action_step = AgentStep(
                index=len(steps), type=StepType.ACTION, content="",
                tool_name=call.name, tool_args=call.args, round=call.round,
            )
            steps.append(action_step)
            scratchpad.append(action_step)  # 与串行路径一致：Action 也要进 Scratchpad
            self._emit(action_step, on_step)

        results: List[ToolResult] = [None] * len(limited)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=min(len(limited), self.cfg.max_parallel_tools)) as pool:
            futures = {
                pool.submit(self._run_with_retry, call): idx for idx, call in enumerate(limited)
            }
            for future, idx in futures.items():
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[idx] = ToolResult(
                        call_id=limited[idx].call_id,
                        name=limited[idx].name,
                        status=ToolStatus.ERROR,
                        content=f"并行执行异常：{exc}",
                        error=str(exc),
                    )

        for call, tool_result in zip(limited, results):
            observation_step = AgentStep(
                index=len(steps),
                type=StepType.OBSERVATION if tool_result.ok else StepType.ERROR,
                content=self._truncate_observation(tool_result),
                tool_name=call.name,
                tool_args=call.args,
                tool_result=tool_result,
                latency_ms=tool_result.latency_ms,
                round=call.round,
            )
            steps.append(observation_step)
            scratchpad.append(observation_step)
            self._emit(observation_step, on_step)
        return list(results)

    def _call_with_timeout(self, func: Callable[[], Any], timeout: float) -> Any:
        """在守护线程中执行 ``func``，超时立即抛 :class:`ToolTimeoutError`。

        为什么不用 ``ThreadPoolExecutor``
        --------------------------------
        直觉写法是::

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(fn).result(timeout=timeout)

        但 ``with`` 退出时会执行 ``shutdown(wait=True)``，而它**无视 timeout**：
        ``result()`` 抛了 TimeoutError，`with` 块却仍在原地等那个卡死的任务跑完。
        实测一个 sleep(3s) 的工具在 ``tool_timeout=0.4`` 下照样让 Agent 卡满 3 秒，
        超时形同虚设。守护线程 + ``join(timeout)`` 没有这个收尾等待。
        """
        box: Dict[str, Any] = {}

        def runner() -> None:
            try:
                box["value"] = func()
            except BaseException as exc:  # noqa: BLE001 - 原样回抛给调用方
                box["error"] = exc

        thread = threading.Thread(target=runner, name="tool-call", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            # 线程是守护态：即便工具永远不返回，也不会拖住进程退出
            raise ToolTimeoutError(f"tool call exceeded {timeout:.2f}s")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # 这些字样通常意味着"稍后重试可能成功"（网络抖动、限流、服务端 5xx、超时）
    _RETRYABLE_HINTS = (
        "timeout", "timed out", "超时", "connection", "connect ", "temporarily",
        "unavailable", "reset by peer", "network", "网络", "429", "502", "503", "504",
    )

    @classmethod
    def _is_retryable(cls, result: ToolResult) -> bool:
        """判断失败是否值得重试。

        早期实现"只要失败就重试"。但像"缺少必填参数 query"这类**参数错误**，
        重试 2 次不仅毫无意义，还会白等 ``0.25s+0.5s`` 并让模型误以为工具不稳。
        只有超时与疑似临时性故障才重试。
        """
        if result.status == ToolStatus.TIMEOUT:
            return True
        blob = f"{result.error or ''} {result.content or ''}".lower()
        return any(hint in blob for hint in cls._RETRYABLE_HINTS)

    def _run_with_retry(self, call: ToolCall) -> ToolResult:
        """带超时与重试的工具调用（错误恢复机制的核心）。"""
        attempts = max(1, self.cfg.max_retries_per_tool)
        last: Optional[ToolResult] = None
        exhausted = False  # 是否"重试次数用尽"才失败（决定给模型哪条提示）

        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                result = self._call_with_timeout(
                    lambda: self.registry.execute(call), self.cfg.tool_timeout
                )
                if not isinstance(result, ToolResult):
                    result = ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        status=ToolStatus.ERROR,
                        content=f"工具 {call.name} 返回了非法结果类型：{type(result).__name__}",
                        error="invalid_result_type",
                    )
            except ToolTimeoutError:
                result = ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    status=ToolStatus.TIMEOUT,
                    content=(
                        f"工具 {call.name} 执行超时（>{self.cfg.tool_timeout:.0f}s）。"
                        f"{'将重试。' if attempt + 1 < attempts else '请改用其它工具或简化查询。'}"
                    ),
                    error="timeout",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except Exception as exc:  # noqa: BLE001
                result = ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    status=ToolStatus.ERROR,
                    content=f"工具 {call.name} 调用异常：{exc}",
                    error=str(exc),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )

            result.retries = attempt
            last = result
            if result.ok:
                return result

            self._bump("failures")
            if attempt + 1 >= attempts:
                exhausted = True
                break
            if not self._is_retryable(result):
                break
            _log.info("工具 %s 第 %d 次失败，重试中", call.name, attempt + 1)
            time.sleep(0.25 * (attempt + 1))

        assert last is not None
        # 失败收尾：补充可操作的提示，帮助模型改道
        if exhausted:
            last.content += (
                f"\n\n【错误恢复提示】工具 {call.name} 连续 {attempts} 次失败。"
                f"请换用其它工具，或把参数调整得更简单后重试。"
            )
        else:
            # 非临时性失败（参数缺失、工具不存在…）：重试没有意义，
            # 直接提示模型改道，省掉两轮无用的等待。
            last.content += (
                f"\n\n【错误恢复提示】工具 {call.name} 调用失败，重试该参数不会成功。"
                f"请检查参数格式，或改用其它工具。"
            )
        return last

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _build_system_prompt(self, candidate_tools: Optional[Sequence[str]] = None) -> str:
        """组装 System Prompt；路由给出候选工具时只渲染这些工具（省 Token、降误选）。"""
        from .prompts import build_react_system_prompt

        if candidate_tools:
            # 始终保留"收尾"能力所需的工具与元信息
            tools = [t for t in self.registry.enabled_tools() if t.name in set(candidate_tools)]
            if not tools:
                tools = self.registry.enabled_tools()
        else:
            tools = self.registry.enabled_tools()

        catalog = "\n\n".join(tool.render_for_prompt() for tool in tools)
        router_hint = ""
        if candidate_tools:
            router_hint = (
                "## 路由建议\n"
                f"初步意图识别认为本轮最可能需要这些工具：{'、'.join(candidate_tools)}。\n"
                "这只是建议 —— 如果实际需要其它工具或无需工具，请按你的判断执行。"
            )
        return build_react_system_prompt(
            catalog, max_tool_iterations=self.cfg.max_iterations, router_hint=router_hint
        )

    def _build_user_prompt(self, question: str, memory_context: Dict[str, Any]) -> str:
        from .prompts import build_react_user_prompt

        return build_react_user_prompt(
            question,
            history_summary=memory_context.get("summary", ""),
            memory_hint=memory_context.get("hint", ""),
        )

    def _history_messages(
        self, history: Optional[Sequence[Message]], memory_context: Dict[str, Any]
    ) -> List[Message]:
        """把历史对话转成消息列表（仅保留 user/assistant 且限制条数）。"""
        messages: List[Message] = []
        for message in list(history or [])[-8:]:
            if getattr(message, "role", None) is None:
                continue
            role_value = message.role.value if hasattr(message.role, "value") else str(message.role)
            if role_value in ("user", "assistant") and message.content:
                messages.append(Message(role=message.role, content=message.content[:1500]))
        return messages

    def _truncate_observation(self, result: ToolResult) -> str:
        """截断观察结果，防止单条 Observation 撑爆上下文。"""
        content = result.content or ""
        limit = self.cfg.max_observation_chars
        if len(content) <= limit:
            return content
        return content[:limit].rstrip() + f"\n...（结果过长已截断，原文共 {len(content)} 字符）"

    @staticmethod
    def _signature(action: Dict[str, Any]) -> str:
        return _action_signature(action)

    def _is_looping(self, state: _RunState, new_signatures: Sequence[str]) -> bool:
        """判断是否陷入死循环（两级判定）。

        **第一级：精确重复** —— 把新签名与"最近 window 次调用"比较，
        若每个新签名都在最近 window 内出现过，即判为死循环。

        **第二级：换词重试** —— 这是实测发现必须补上的一级。
        真实模型很少一字不差地重复调用；它更常见的退化行为是
        **把同一个查询换几种说法反复搜**（例如"RAG 2026 新进展" →
        "agentic RAG 2026 advances" → "RETRIEVAL augmented generation 2026 survey"），
        每次都"看起来不一样"，于是精确匹配永远抓不到，白白烧掉全部轮次，
        最后用户只拿到一坨原始工具输出。因此这一级用**词项重合度**判定语义重复。

        第二级只在**每个工具最近 ``loop_semantic_window`` 条查询**里比对。
        早期实现拿整个 run 的全部历史比对，长会话末尾随口一问就可能撞上开头
        早已问过的旧查询，被误判成"重复"而提前收尾。
        """
        window = max(1, self.cfg.loop_detect_window)
        seen = state.seen_signatures
        if not seen:
            return False
        recent = list(seen)[-window * max(1, len(new_signatures)) :]
        if new_signatures and all(sig in recent for sig in new_signatures):
            return True

        # -- 第二级：语义重复检测（只看最近若干条，避免误判）--
        semantic_window = max(1, self.cfg.loop_semantic_window)
        for signature in new_signatures:
            name, _, args = signature.partition("::")
            text = self._extract_query_text(args)
            if not text:
                continue
            history = (state.tool_query_history.get(name) or [])[-semantic_window:]
            for previous in history:
                if self._is_semantically_similar(text, previous):
                    _log.info(
                        "检测到语义重复调用：%s(%r) ≈ 先前的 %r",
                        name,
                        text[:40],
                        previous[:40],
                    )
                    return True
        return False

    @staticmethod
    def _extract_query_text(args_json: str) -> str:
        """从工具参数 JSON 中取出用于相似度比较的查询文本。

        不同工具的检索词字段名不同（query / text / q / papers / expression…），
        这里统一提取，拼成一段可比对的文本。
        """
        try:
            payload = json.loads(args_json)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        parts: List[str] = []
        for key in ("query", "text", "q", "keyword", "keywords", "expression", "paper", "title", "doc_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, (list, tuple)):
                parts.extend(str(v).strip() for v in value if str(v).strip())
        if not parts:
            # papers 之类字段之外，退化为整个参数串
            return " ".join(str(v) for v in payload.values() if isinstance(v, str))
        return " ".join(parts)

    @staticmethod
    def _is_semantically_similar(
        left: str, right: str, threshold: float = 0.52, jaccard_floor: float = 0.53
    ) -> bool:
        """判断两次工具查询是否"其实是同一次"（换词重试）。

        距离度量
        --------
        主判据取 ``min(Jaccard, coverage)``，其中

        - **Jaccard** = ``|A∩B| / |A∪B|`` —— 惩罚"两边都加了很多不相干的新词"
        - **coverage** = ``|A∩B| / min(|A|,|B|)`` —— 奖励"短查询被长查询完全包含"

        用两者的**较小值**，是为了同时挡住两类误判：

        1. 只看 Jaccard 会漏掉"A 是 B 的子集"这种典型换词重试
           （``"RAG 2026 最新进展"`` ⊂ ``"2026 RAG ... 最新进展 新方法"``，Jaccard 仅 0.56）；
        2. 只看 coverage 会把"短词碰巧被包含"误判为重复
           （``attention mechanism`` vs ``attention mechanism survey 2023`` 的 coverage = 1.0，
           但这是两个不同意图，不能当重复）。

        另设 ``jaccard_floor`` 下限作为**兜底护栏**：``min()`` 在"一方是另一方子集"时会等于
        Jaccard，此时若 Jaccard 偏低说明两边长度差异过大（短查询只是长查询的一小部分），
        那种情况更像"换了个话题"而不是重试。

        阈值标定（12 组实测对照）
        ------------------------
        重复类 ``min()`` 区间 0.533 ~ 0.769，非重复类 0.000 ~ 0.500，取 0.52 可干净分开。
        误判为重复的代价更高（会让 Agent 过早放弃检索），因此宁可保守。
        """
        from ..utils.text import tokenize

        a, b = set(tokenize(left)), set(tokenize(right))
        if not a or not b:
            return False
        union = len(a | b)
        smaller = min(len(a), len(b))
        if union == 0 or smaller == 0:
            return False
        jaccard = len(a & b) / union
        coverage = len(a & b) / smaller
        return min(jaccard, coverage) >= threshold and jaccard >= jaccard_floor

    def _record_tool_query(self, state: _RunState, call: ToolCall) -> None:
        """记录一次工具查询，供语义重复检测使用（写入 per-run 状态）。"""
        name = call.name or "unknown"
        text = self._extract_query_text(
            json.dumps(call.args, sort_keys=True, ensure_ascii=False, default=str)
        )
        if text:
            state.tool_query_history.setdefault(name, []).append(text)

    def _load_memory(self, session_id: str) -> Dict[str, Any]:
        """从记忆模块取上下文（摘要 + 最近消息 + 提示）。"""
        if self.memory is None or not session_id:
            return {}
        try:
            return self.memory.get_context(session_id) or {}
        except Exception as exc:  # noqa: BLE001
            _log.warning("读取对话记忆失败: %s", exc)
            return {}

    def _direct_answer(
        self, question: str, history: Optional[Sequence[Message]]
    ) -> Optional[Tuple[str, TokenUsage]]:
        """路由判定"无需工具"时直接生成答案（跳过整个 ReAct 循环）。"""
        try:
            messages = [
                Message.system(
                    "你是智能科研助理。用户的问题不需要查阅论文知识库，"
                    "请直接用简洁准确的语言回答。"
                    "如果这属于需要论文依据的问题，请明确说明你未查阅知识库。"
                )
            ]
            messages.extend(self._history_messages(history, {}))
            messages.append(Message.user(question))
            answer, usage = self.llm.chat_with_usage(
                messages, temperature=0.3, max_tokens=self.cfg.max_tokens
            )
            if answer and answer.strip():
                _log.info("路由判定直接作答，跳过 ReAct 循环")
                return answer.strip(), usage
        except Exception as exc:  # noqa: BLE001
            _log.warning("直接作答失败，回退到完整循环: %s", exc)
        return None

    def _collect_observations(self, scratchpad: Sequence[AgentStep]) -> str:
        """把轨迹中的观察结果拼成文本（供强制收尾与兜底使用）。

        单条 Observation 已在 ``_truncate_observation`` 里限长（默认 2000 字符），
        但**总长度**没有任何上限：8 轮 × 3 个并行工具 × 2000 字符 = 48k 字符，
        既可能撑爆收尾提示词的上下文，也会让"兜底答案"变成一坨没法读的原始输出。
        因此这里按 ``max_aggregate_chars`` 做总量预算：**优先保留最近的观察**
        （最相关的证据通常在最后一轮），放不下的直接从最新往回截。
        """
        blocks: List[str] = []
        budget = max(500, int(self.cfg.max_aggregate_chars))
        for step in reversed(list(scratchpad)):
            if step.type == StepType.OBSERVATION and step.content:
                block = f"[工具 {step.tool_name} 返回]\n{step.content}"
            elif step.type == StepType.ERROR and step.content:
                block = f"[错误]\n{step.content}"
            else:
                continue
            if len(block) > budget:
                block = block[:budget].rstrip() + "\n...（内容过长已截断）"
            blocks.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        blocks.reverse()
        return "\n\n".join(blocks)

    def _force_final_answer(
        self, question: str, scratchpad: Sequence[AgentStep]
    ) -> Tuple[str, TokenUsage]:
        """达到最大轮次时的强制收尾：用已收集信息生成答案。

        直接抛"推理轮次超限"对用户毫无价值 —— 检索到的资料明明就在手里。
        这里再调用一次 LLM 让它基于已有观察作答，保证用户总能拿到有依据的结果。
        """
        from .prompts import build_final_answer_prompt

        observations = self._collect_observations(scratchpad)
        if not observations:
            return "", TokenUsage()

        try:
            messages = [Message.user(build_final_answer_prompt(question, observations))]
            answer, usage = self.llm.chat_with_usage(
                messages, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens
            )
            if answer and answer.strip():
                _log.info("已达最大轮次，基于已有观察强制收尾生成答案")
                return answer.strip(), usage
        except Exception as exc:  # noqa: BLE001
            _log.warning("强制收尾生成失败: %s", exc)
        return self._fallback_from_scratchpad(scratchpad, question), TokenUsage()

    def _fallback_from_scratchpad(self, scratchpad: Sequence[AgentStep], question: str) -> str:
        """最终兜底：直接汇总观察结果（不调用 LLM），保证永不空手而归。"""
        observations = self._collect_observations(scratchpad)
        if not observations:
            return (
                f"抱歉，我未能就「{question}」从知识库中收集到有效信息。\n\n"
                "可能原因：知识库中还没有相关论文，或问题与已上传论文的内容差异较大。\n"
                "建议：先上传相关论文 PDF，或用论文中出现过的术语重新提问。"
            )
        return (
            f"以下是就「{question}」从知识库中检索到的原始信息"
            f"（Agent 推理未正常收尾，此处直接汇总工具结果供你参考）：\n\n{observations}"
        )

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """把工具返回的任意字段安全转成 int。

        工具结果是模型/检索层给的**外部数据**：``page`` 可能是 ``"12"``、
        ``"p.12"``、``None`` 甚至 ``[12]``。早期实现直接 ``int(item.get("page"))``，
        一个脏字段就让整次 ``run()`` 在汇总阶段抛异常 —— 用户在拿到答案的最后
        一步收到 500，前面所有推理全部作废。
        """
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            import re as _re

            match = _re.search(r"-?\d+", value)
            if match:
                try:
                    return int(match.group())
                except ValueError:
                    return default
        return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip().rstrip("%"))
            except ValueError:
                return default
        return default

    def _citation_from_item(
        self, item: Dict[str, Any], index: int
    ) -> Optional[tuple[tuple[str, int], Citation]]:
        """把一条工具结果转换成 (去重键, 引用)；字段非法时返回 None。"""
        source = str(item.get("source", "") or "").strip()
        if not source:
            return None
        page = self._safe_int(item.get("page", 0))
        key = (source, page)
        return key, Citation(
            index=index,
            chunk_id=str(item.get("chunk_id", "") or ""),
            doc_id=str(item.get("doc_id", "") or ""),
            source=source,
            page=page,
            snippet=str(item.get("preview", "") or "")[:200],
            score=self._safe_float(item.get("confidence", item.get("score", 0.0))),
        )

    def _collect_citations(
        self, steps: Sequence[AgentStep], answer: str
    ) -> List[Citation]:
        """从工具结果与答案文本中汇总引用来源。"""
        citations: List[Citation] = []
        seen: set[tuple[str, int]] = set()

        # 来源 1：检索类工具返回的结构化结果
        for step in steps:
            if step.tool_result is None or not step.tool_result.ok:
                continue
            data = step.tool_result.data
            if not isinstance(data, dict):
                continue
            for container in (data.get("results") or [], data.get("evidence") or {}):
                if isinstance(container, dict):
                    # 对比 / 摘要 / 元信息工具的证据结构：evidence -> {方面: {列表名: [...]}}
                    items_iter: Iterable[Any] = [
                        item
                        for group in container.values()
                        if isinstance(group, dict)
                        for values in group.values()
                        for item in (values or [])
                    ]
                else:
                    items_iter = container
                for item in items_iter:
                    if not isinstance(item, dict):
                        continue
                    converted = self._citation_from_item(item, len(citations) + 1)
                    if converted is None:
                        continue
                    key, citation = converted
                    if key in seen:
                        continue
                    seen.add(key)
                    citations.append(citation)

        # 来源 2：答案正文里出现的引用标记（模型可能引用了未在结构化结果中的来源）
        from ..generation.prompt_template import CITATION_PATTERN

        known = {c.source.lower().strip() for c in citations}
        for match in CITATION_PATTERN.finditer(answer or ""):
            source = match.group("source").strip()
            page = self._safe_int(match.group("page"))
            if (source, page) in seen:
                continue
            if source.lower().strip() in known:
                continue
            seen.add((source, page))
            citations.append(
                Citation(index=len(citations) + 1, source=source, page=page, score=0.0)
            )
        return citations

    def _emit(self, step: AgentStep, callback: Optional[Callable[[AgentStep], None]]) -> None:
        if callback is None:
            return
        try:
            callback(step)
        except Exception as exc:  # noqa: BLE001 - 回调异常不影响推理
            _log.warning("步骤回调异常: %s", exc)

    def _finalize(self, result: AgentResult, context: Optional[Dict[str, Any]]) -> None:
        """写入记忆、Trace 日志与指标。"""
        # 记忆
        if self.memory is not None and result.session_id:
            try:
                self.memory.add_turn(
                    result.session_id,
                    question=result.question,
                    answer=result.answer,
                    # 引用必须一起落库：否则重新打开历史会话时，答案还在、
                    # 引用卡片全没了（前端只能从轨迹里猜，猜出来的是假的）
                    citations=result.citations,
                    trajectory=result.trajectory(),
                    usage=result.usage,
                    latency_ms=result.latency_ms,
                    metadata={
                        "cache_hit": bool(result.cache_hit),
                        "degraded": bool(result.degraded),
                        "terminated_reason": result.terminated_reason or "",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("写入对话记忆失败: %s", exc)
        # 可观测性
        try:
            tracer().log_agent(result, context)
            get_metrics().record_agent(result)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Agent 可观测性记录失败: %s", exc)

        _log.info(
            "Agent 完成：iterations=%d tools=%d/%d terminated=%s latency=%.0fms tokens=%d",
            result.iterations,
            result.tool_success,
            result.tool_calls,
            result.terminated_reason,
            result.latency_ms,
            result.usage.total_tokens,
        )

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        runs = max(1, self._stats["runs"])
        return {
            **self._stats,
            "avg_tool_calls_per_run": round(self._stats["tool_calls"] / runs, 2),
            "loop_rate": round(self._stats["loops_detected"] / runs, 4),
            "json_retry_rate": round(self._stats["json_retries"] / runs, 4),
            "config": {
                "max_iterations": self.cfg.max_iterations,
                "enable_router": self.cfg.enable_router,
                "enable_parallel": self.cfg.enable_parallel,
                "tool_timeout": self.cfg.tool_timeout,
                "loop_detect_window": self.cfg.loop_detect_window,
            },
            "tools": self.registry.stats(),
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "llm": self.llm.health_check() if self.llm is not None else {"ok": False},
            "tools": self.registry.health_check(),
            "memory": self.memory.health_check() if self.memory is not None else {"ok": True, "enabled": False},
            "router": self.router.stats() if self.router is not None else {"enabled": False},
            "agent": self.stats(),
        }


# ==============================================================================
# 工厂
# ==============================================================================
def create_agent(
    config: Any = None,
    registry: Optional[ToolRegistry] = None,
    llm: Any = None,
    memory: Optional[Any] = None,
    router: Optional[Any] = None,
    retrieval_engine: Any = None,
    **kwargs: Any,
) -> ReActAgent:
    """组装 ReAct Agent（缺省依赖按配置自动创建）。"""
    from ..utils.config import get_config

    cfg = config or get_config()
    if llm is None:
        from ..llm import create_llm

        llm = create_llm(cfg)
    if registry is None:
        from .tools import build_default_registry

        registry = build_default_registry(retrieval_engine=retrieval_engine, config=cfg)
    if memory is None:
        try:
            from .memory import create_memory

            memory = create_memory(cfg)
        except Exception as exc:  # noqa: BLE001
            _log.warning("对话记忆初始化失败，已禁用: %s", exc)
            memory = None
    if router is None:
        try:
            from .router import create_router

            router = create_router(cfg, llm=llm, registry=registry)
        except Exception as exc:  # noqa: BLE001
            _log.warning("意图路由初始化失败，已禁用: %s", exc)
            router = None
    return ReActAgent(
        registry=registry,
        llm=llm,
        memory=memory,
        router=router,
        config=cfg,
        **kwargs,
    )
