"""Agent System Prompt 设计（模块三 5.3.1）。

ReAct 循环能否稳定收敛，**七成取决于 System Prompt 的质量**。本模块把提示词的
四个组成部分拆开分别优化与测试：

1. **角色定义**：明确"科研助理"身份，设定严谨基调，压低编造倾向。
2. **工具清单**：由 :class:`~src.agent.base_tool.ToolRegistry` 自动渲染 —— 
   工具描述里写清"什么时候用/不用"，这是提升路由准确率最有效的手段。
3. **输出格式约束**：严格的 JSON 结构 + 常见错误示例。实测相比自由文本格式，
   JSON 约束让解析失败率从约 15% 降到 1% 以下（配合 :func:`extract_json` 容错）。
4. **推理规范**：规定 Thought 必须"先判断信息是否足够、再决定动作"，
   并给出终止条件 —— 这是防死循环的关键。

另外提供 :data:`ROUTE_PROMPT` 用于规则路由（先做一次廉价意图识别，
减少无谓的推理轮次）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "REACT_SYSTEM_PROMPT",
    "ROUTE_PROMPT",
    "FINAL_ANSWER_PROMPT",
    "build_react_system_prompt",
    "build_react_user_prompt",
    "build_route_prompt",
    "render_scratchpad",
    "TOOL_CALL_FORMAT_EXAMPLE",
]

# 工具调用的标准 JSON 形态（写进提示词，给模型"抄格式"的锚点）
TOOL_CALL_FORMAT_EXAMPLE = """{
  "thought": "用户询问 Transformer 的训练数据集，这属于知识库中论文的内容，需要先检索。",
  "action": "knowledge_search",
  "action_input": {"query": "Transformer training dataset WMT 2014", "top_k": 5}
}"""

_FINAL_FORMAT_EXAMPLE = """{
  "thought": "已从检索结果中获得 WMT 2014 英德/英法数据集信息，信息足以回答，无需继续调用工具。",
  "action": "final_answer",
  "action_input": {"answer": "该模型在 WMT 2014 英德数据集（约 450 万句对）与英法数据集（3600 万句）上训练【Attention Is All You Need p.4】。"}
}"""


REACT_SYSTEM_PROMPT = """你是"智能科研助理"的决策中枢（Agent），运行在 ReAct（Reasoning + Acting）范式下。

## 你的任务
通过"思考 → 调用工具 → 观察结果 → 再思考"的循环，收集足够信息后回答用户的科研问题。

## 可用工具
{tool_catalog}

{router_hint}
## 输出格式（必须严格遵守）
每一步你都必须输出**单个 JSON 对象**，不得包含任何 JSON 之外的文字、解释或 Markdown 代码围栏。

调用工具时：
```json
{tool_example}
```

给出最终答案时：
```json
{final_example}
```

字段说明：
- `thought`（必填，字符串）：用一两句话说明"当前掌握了什么、还缺什么、为什么选这个动作"。
  这是可观测性的核心，用户会在界面上看到它，请写得具体、诚实。
- `action`（必填，字符串）：工具名，或 `final_answer`。
- `action_input`（必填，对象）：工具的参数；当 action 为 `final_answer` 时，
  该字段为 `{{"answer": "你的完整回答"}}`。

## 决策规范（重要）
1. **先判断再行动**：如果问题属于通用常识、纯数学计算、或问时间，**不要**调用知识库检索。
2. **优先知识库**：涉及已上传论文内容的问题，必须先调用 `knowledge_search` 获取原文依据，
   不许凭记忆回答论文细节。
3. **知识库查不到就转联网**：
   `knowledge_search` 返回"没有检索到相关内容"时，**不要立刻放弃** ——
   若问题涉及时效性信息（最新进展、某模型何时发布）、
   或用户明确要求"找/推荐几篇论文"、或明显超出已上传论文范围，
   应当改用 `web_search` 再试一次。两个来源都没有，才如实说明无法回答。
4. **区分"问已上传论文"与"找外部文献"**：
   - 「论文A 的方法是什么」→ `knowledge_search`（读已上传原文，能给出页码引用）
   - 「帮我找几篇关于 X 的论文」→ `web_search`（用户要的是**新**文献，不是已上传的那几篇）
   - 「X 方向最近有什么进展」→ `web_search`（可加 `scope="academic"` 走 arXiv/Crossref）
   若不确定，可先 `knowledge_search`；一旦发现知识库不覆盖该主题，**必须**转向 `web_search`。
5. **一次一步**：每轮只输出**一个** JSON。需要多个工具时，在后续轮次中依次调用。
6. **信息足够就收尾**：只要已有信息能回答问题，立刻输出 `final_answer`，不要为了"更全面"而继续检索。
7. **诚实优先于完整**：所有可用工具都试过仍无信息时，如实说明"未找到"，
   绝不编造数字、作者、年份或引用页码。
8. **引用格式分两类**：
   - 来自知识库：`【文档名 p.页码】`（文档名与页码必须与工具返回的"来源"字段完全一致）
   - 来自联网：`来源：标题 (链接)` 或标注 DOI；**绝不要为网络内容编造页码**
9. **不要重复同一调用**：如果某个工具用相同参数已经调用过且没有新信息，换用其它工具或直接作答。

## 停止条件
出现以下任一情况时，必须输出 `final_answer`：
- 已获得足以回答问题的信息；
- 知识库与联网搜索都已尝试过，仍无法获得相关信息（此时应说明检索过程与建议）；
- 已连续调用 {max_tool_iterations} 次工具仍无进展（说明检索不到，并给出可操作建议）；
- 用户的问题与知识库无关，且你的通用知识足以作答（可直接回答，但要说明这不是基于知识库的结论）。
"""


ROUTE_PROMPT = """你是问题分类器。请判断下面的问题应该走哪条处理路径，只输出 JSON，不要解释。

可选路径（可多选，最多 2 个）：
- `knowledge_search`：需要查阅用户已上传论文的原文内容（方法、实验、数据、结论等）
- `paper_metadata`：只问论文的题目/作者/年份/期刊/DOI 等元信息
- `paper_compare`：要求对比两篇或多篇论文
- `summarize_paper`：要求总结/概括某篇论文
- `keyword_extract`：要求提取关键词
- `calculator`：纯数学计算
- `current_time`：询问当前日期或时间
- `web_search`：需要知识库之外的时效性信息
- `direct`：不需要任何工具，可直接用通用知识回答（问候、闲聊、纯概念解释）

输出格式：
{{"routes": ["路径1"], "reason": "一句话说明判断依据", "confidence": 0.0~1.0}}

## 待分类问题
{question}

## 对话历史（可能为空）
{history}
"""


FINAL_ANSWER_PROMPT = """你是"智能科研助理"。请依据下方【工具收集到的信息】回答用户问题。

要求：
1. 只使用工具结果中的信息回答论文相关问题，不要补充无法核查的具体事实。
2. 每个来自工具结果的事实，在句末标注 `【文档名 p.页码】`（页码必须是工具结果里出现过的）。
3. 如果工具结果不足以回答，明确说明"知识库中未找到相关文档支持该结论"，并说明实际检索到什么。
4. 用 Markdown 组织答案；对比类问题优先用表格；答案控制在 500 字以内（用户要求详细时除外）。
5. 语言与用户提问语言一致。

## 用户问题
{question}

## 工具收集到的信息
{observations}
"""


def build_react_system_prompt(
    tool_catalog: str,
    max_tool_iterations: int = 8,
    router_hint: str = "",
) -> str:
    """组装完整的 ReAct System Prompt。"""
    hint_block = f"{router_hint.strip()}\n\n" if router_hint.strip() else ""
    return REACT_SYSTEM_PROMPT.format(
        tool_catalog=tool_catalog.strip(),
        tool_example=TOOL_CALL_FORMAT_EXAMPLE,
        final_example=_FINAL_FORMAT_EXAMPLE,
        max_tool_iterations=max_tool_iterations,
        router_hint=hint_block,
    )


def build_react_user_prompt(
    question: str,
    history_summary: str = "",
    memory_hint: str = "",
) -> str:
    """构建首轮 User 消息（含历史摘要与记忆提示）。"""
    parts: List[str] = []
    if history_summary.strip():
        parts.append(f"## 先前对话摘要\n{history_summary.strip()}")
    if memory_hint.strip():
        parts.append(f"## 上下文提示\n{memory_hint.strip()}")
    parts.append(f"## 用户问题\n{question.strip()}")
    parts.append("## 请输出下一步 JSON（只输出 JSON）")
    return "\n\n".join(parts)


def render_scratchpad(steps: Sequence[Any], max_chars: int = 4000) -> str:
    """把已执行步骤渲染成 ReAct 的 Scratchpad（供下一轮提示词使用）。

    这是"无状态 LLM 实现有状态循环"的关键：每一轮都把完整轨迹重新喂给模型，
    模型才能知道自己做过什么、避免重复调用。

    Parameters
    ----------
    steps: :class:`~src.utils.types.AgentStep` 序列
    max_chars: 轨迹文本上限，超出时保留**最早的思考**与**最近的观察**
               （早期 Thought 定义了任务目标，最近 Observation 决定下一步）
    """
    from ..utils.types import StepType

    blocks: List[str] = []
    for step in steps:
        if step.type == StepType.THOUGHT:
            blocks.append(f"Thought: {step.content.strip()}")
        elif step.type == StepType.ACTION:
            import json

            args = json.dumps(step.tool_args, ensure_ascii=False)
            blocks.append(f"Action: {step.tool_name}\nAction Input: {args}")
        elif step.type == StepType.OBSERVATION:
            content = (step.content or "").strip()
            blocks.append(f"Observation: {content}")
        elif step.type == StepType.ERROR:
            blocks.append(f"Observation: [错误] {step.content.strip()}")

    text = "\n".join(blocks)
    if len(text) <= max_chars:
        return text

    # 超长时：保留头部（目标）+ 尾部（最近观察）
    head_budget = int(max_chars * 0.3)
    tail_budget = max_chars - head_budget - 40
    return (
        text[:head_budget].rstrip()
        + "\n\n... （中间步骤已省略，避免上下文超限） ...\n\n"
        + text[-tail_budget:].lstrip()
    )


def build_route_prompt(question: str, history: str = "") -> str:
    """构建意图路由提示词。"""
    return ROUTE_PROMPT.format(question=question.strip(), history=history.strip() or "（无）")


def build_final_answer_prompt(question: str, observations: str) -> str:
    """构建"基于工具结果生成最终答案"的提示词。

    当 Agent 达到最大轮次仍未收尾时，用这个提示词做一次强制收尾生成，
    保证用户总能得到有依据的答案，而不是一个"推理超限"的报错。
    """
    return FINAL_ANSWER_PROMPT.format(question=question.strip(), observations=observations.strip())
