"""模块二 —— RAG 生成层的 Prompt 工程（对应课程要求 5.2.1）。

设计要点
--------
1. **结构化 System Prompt**：角色设定 / 可用上下文 / 回答规则 / 输出格式四段式。
   四段职责分离，便于单独调参与 A/B 对比。
2. **上下文按相关性排序拼接**：Top-1 放在最前面（大模型对首尾信息更敏感，
   即 "lost in the middle" 现象），并对超长上下文按相关性动态截断。
3. **强制引用溯源**：要求模型在每个事实性陈述后标注 ``【来源 p.页码】``，
   格式与前端引用角标解析规则严格一致，形成"生成→解析→跳转"闭环。
4. **明确的拒答与降级指令**：上下文不足时不许编造，必须明说"知识库中未找到"，
   这是科研场景可信度的底线。

提供三套模板，可按场景切换（也用于 Prompt 对比实验）：

- :data:`RAG_QA_PROMPT`      默认问答（带引用、可推理）
- :data:`RAG_STRICT_PROMPT`  严格模式（只依据上下文，不做任何常识补充）
- :data:`RAG_SUMMARY_PROMPT` 论文结构化摘要（背景-方法-结果-结论）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..utils.types import Citation, RetrievedChunk

__all__ = [
    "PromptTemplate",
    "RAG_QA_PROMPT",
    "RAG_STRICT_PROMPT",
    "RAG_SUMMARY_PROMPT",
    "NO_CONTEXT_PROMPT",
    "MEMORY_SUMMARY_PROMPT",
    "PROMPT_REGISTRY",
    "get_prompt",
    "build_context",
    "format_context_block",
    "build_citations",
    "CITATION_PATTERN",
]

# 引用标记正则：与前端 markdown.js 的解析规则保持一致
# 匹配 【来源 p.3】 / 【论文名 p.12】 / 【论文名 第3页】
CITATION_PATTERN = re.compile(
    r"【\s*(?P<source>[^】\n]{1,120}?)\s*(?:p\.|P\.|第)\s*(?P<page>\d{1,4})\s*页?\s*】"
)


# ==============================================================================
# 模板定义
# ==============================================================================
@dataclass
class PromptTemplate:
    """Prompt 模板。

    ``system`` 与 ``user`` 中都可用花括号占位符，由 :meth:`render` 填充。
    """

    name: str
    system: str
    user: str
    description: str = ""
    # 允许占位符缺失时保留原文（避免 KeyError 打断生成）
    strict_keys: bool = False

    def render(self, **values: Any) -> Dict[str, str]:
        """渲染模板，返回 ``{"system": ..., "user": ...}``。"""
        from string import Formatter

        def fill(text: str) -> str:
            if self.strict_keys:
                return text.format(**values)
            try:
                return text.format(**values)
            except KeyError as exc:
                # 缺字段时给出可读提示而不是崩溃
                raise KeyError(
                    f"Prompt 模板 '{self.name}' 缺少占位符 {exc}；"
                    f"可用字段：{sorted(values.keys())}"
                ) from exc

        return {"system": fill(self.system), "user": fill(self.user)}


_SYSTEM_COMMON = """你是"智能科研助理"，一个面向学术论文的检索增强问答（RAG）系统。

## 你的角色
你帮助科研人员快速理解论文内容：解释方法、对比工作、定位实验数据、澄清概念。
你的回答必须严谨、准确、可核查 —— 科研场景下，"不确定"远好于"编造"。

## 回答规则
1. **只依据给定资料**：下方【参考资料】是从用户论文知识库中检索出的原文片段，
   这是你回答的主要依据。不要引入资料之外的、你无法核查的具体事实。
2. **必须标注引用**：凡是来自参考资料的具体事实、数字、结论、方法名称，
   都要在该句末尾标注来源，格式严格为 `【文档名 p.页码】`（页码用阿拉伯数字）。
   例：`该模型在 WMT 2014 英德任务上取得 28.4 BLEU【Attention Is All You Need p.4】。`
3. **资料不足时明说**：如果参考资料无法回答该问题，就明确回答
   "当前知识库中未找到相关文档支持该结论"，并说明你的资料覆盖到什么程度。
   严禁凭印象补全实验数字、作者、年份、DOI 等可核查信息。
4. **区分事实与推断**：属于你的推理或常识补充时，用"（推断）"标注。
5. **结构化表达**：多要点时使用 Markdown 列表或小标题；对比类问题优先用表格。
6. **语言一致**：用户用中文提问就用中文回答，用英文提问就用英文回答。
7. **数字精确**：指标数字、超参数、数据集名称、模型名必须与参考资料完全一致，不要四舍五入或改写。

## 输出格式
- 直接给出答案，不要复述问题，不要写"根据提供的资料"这类空话开头。
- 引用标记写在句末标点之前。
- 保持简洁：一般情况下控制在 400 字以内；用户明确要求详细时再展开。"""


RAG_QA_PROMPT = PromptTemplate(
    name="rag_qa",
    description="默认 RAG 问答模板：依据检索上下文作答并强制引用溯源。",
    system=_SYSTEM_COMMON,
    user="""## 参考资料
{context}

## 用户问题
{question}

## 请回答
先在心里判断资料是否足以回答（不要输出判断过程），然后直接给出答案，
并按要求标注 `【文档名 p.页码】` 引用。""",
)


RAG_STRICT_PROMPT = PromptTemplate(
    name="rag_strict",
    description="严格模式：完全不允许上下文之外的补充，用于事实核验类问题。",
    system="""你是严谨的学术文献事实核验助手。

规则（必须逐条遵守）：
1. 你的答案中**每一句**都必须能在【参考资料】中找到直接依据。
2. 不允许使用任何外部知识，不允许推测，不允许补全数字。
3. 每句话末尾必须标注 `【文档名 p.页码】`。
4. 如果参考资料无法支撑答案，只回复：`当前知识库中未找到相关文档支持该结论。`
   并列出你实际检索到的相关片段主题（不标注引用）。
5. 不做任何格式化修饰以外的扩写。""",
    user="""## 参考资料
{context}

## 待核验问题
{question}

## 核验结论""",
)


RAG_SUMMARY_PROMPT = PromptTemplate(
    name="rag_summary",
    description="论文结构化摘要：按『背景-方法-结果-结论』四段式输出。",
    system="""你是学术论文摘要生成助手。请依据给定资料，为目标论文生成结构化摘要。

输出必须严格使用以下四个小标题（Markdown 三级标题），每部分 2-5 句话：

### 研究背景与动机
（该工作要解决什么问题，为什么重要）

### 方法与技术路线
（核心方法、模型结构、关键设计选择）

### 实验结果
（数据集、对比基线、关键指标数值 —— 必须与资料一致）

### 结论与局限
（作者结论、适用边界、可能的局限）

要求：每个事实性陈述后标注 `【文档名 p.页码】`；资料缺失的部分写"资料未覆盖"，
不要编造。""",
    user="""## 参考资料
{context}

## 目标论文/问题
{question}

## 结构化摘要""",
)


NO_CONTEXT_PROMPT = PromptTemplate(
    name="no_context",
    description="检索无结果时的降级模板：明确告知用户知识库未命中。",
    system="""你是"智能科研助理"。当前**没有**从用户的知识库中检索到相关内容。

要求：
1. 首先明确说明：当前知识库中未找到与问题相关的文档，因此以下内容**不是**基于用户论文的回答。
2. 如果这个问题属于通用常识或领域基础知识，你可以在明确标注"以下为通用知识，非知识库内容"
   的前提下简要回答；否则请直接说明无法回答。
3. 给出一条可操作的建议（例如：请先上传相关论文 PDF，或换用更具体的关键词）。
4. 不要编造任何文献引用、页码或实验数据。""",
    user="""## 用户问题
{question}

## 检索情况
{retrieval_note}

## 请回复""",
)


MEMORY_SUMMARY_PROMPT = PromptTemplate(
    name="memory_summary",
    description="长对话摘要压缩模板（模块三记忆管理使用）。",
    system="""你是对话摘要压缩器。请把给出的多轮对话压缩成一段不超过 200 字的摘要，
用于替代原始历史以节省上下文预算。

必须保留：
- 用户关注的核心论文 / 主题
- 已经确认的关键结论与数字
- 用户明确表达过的偏好或约束（例如"只要方法部分"）

不要保留：寒暄、重复内容、已被推翻的中间结论。
直接输出摘要正文，不要加"摘要："前缀。""",
    user="""## 历史对话
{history}

## 压缩摘要""",
)


PROMPT_REGISTRY: Dict[str, PromptTemplate] = {
    "rag_qa": RAG_QA_PROMPT,
    "default": RAG_QA_PROMPT,
    "rag_strict": RAG_STRICT_PROMPT,
    "strict": RAG_STRICT_PROMPT,
    "rag_summary": RAG_SUMMARY_PROMPT,
    "summary": RAG_SUMMARY_PROMPT,
    "no_context": NO_CONTEXT_PROMPT,
    "memory_summary": MEMORY_SUMMARY_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """按名称取模板（未知名称回退默认问答模板）。"""
    return PROMPT_REGISTRY.get(str(name).lower(), RAG_QA_PROMPT)


# ==============================================================================
# 上下文拼接
# ==============================================================================
def format_context_block(
    chunk: RetrievedChunk,
    index: int,
    max_chars: int = 1500,
    include_meta: bool = True,
) -> str:
    """把单个检索片段格式化为上下文块。

    格式（与 MockLLM 的解析规则对齐，便于离线联调）::

        [1] 来源：Attention Is All You Need p.4 | 相关度 0.87 | 路由 hybrid+rerank
        正文……

    Parameters
    ----------
    max_chars: 单块最大字符数（超出在句子边界截断）
    include_meta: 是否附带相关度等元信息（省 token 时可关闭）
    """
    text = chunk.text.strip()
    if len(text) > max_chars:
        text = _truncate_at_sentence(text, max_chars)
    source = chunk.chunk.source or chunk.chunk.doc_id
    if include_meta:
        header = (
            f"[{index}] 来源：{source} p.{chunk.chunk.page} "
            f"| 相关度 {chunk.confidence:.2f} | 召回 {chunk.source_route or 'vector'}"
        )
    else:
        header = f"[{index}] 来源：{source} p.{chunk.chunk.page}"
    return f"{header}\n{text}"


def _truncate_at_sentence(text: str, limit: int) -> str:
    """在句子边界截断，避免切断半个句子造成语义畸变。"""
    if len(text) <= limit:
        return text
    window = text[:limit]
    for marker in ("。", "！", "？", ". ", "! ", "? ", "\n"):
        position = window.rfind(marker)
        if position > limit * 0.55:
            return window[: position + len(marker)].rstrip()
    return window.rstrip() + "…"


def build_context(
    chunks: Sequence[RetrievedChunk],
    max_chars: int = 6000,
    max_chunk_chars: int = 1500,
    include_meta: bool = True,
    min_score: float = 0.0,
) -> tuple[str, List[RetrievedChunk]]:
    """按相关性顺序拼接上下文，并做动态截断。

    截断策略
    --------
    1. 丢弃低于 ``min_score`` 的片段；
    2. 按**相关性降序**逐个尝试加入（Top-1 一定在上下文最前面 —— 这利用了
       大模型对上下文首尾位置更敏感的 "lost in the middle" 特性）；
    3. 累计字符数超过 ``max_chars`` 即停止，但**至少保留 1 个片段**
       （即使它超长，也会被 ``max_chunk_chars`` 截断后放入）。

    Returns
    -------
    (上下文文本, 实际纳入的片段列表)
    """
    # 先按绝对相关度降序排列（不假设调用方已排序）。
    # 优先用 confidence（绝对相关度），缺失时退回 score，再退回原始顺序，
    # 保证"最相关的片段一定出现在上下文最前面"。
    ordered = sorted(
        enumerate(chunks),
        key=lambda pair: (-(pair[1].confidence or pair[1].score or 0.0), pair[0]),
    )

    selected: List[RetrievedChunk] = []
    blocks: List[str] = []
    used = 0

    for _original_index, chunk in ordered:
        if chunk.score < min_score and chunk.confidence < min_score:
            continue
        block = format_context_block(
            chunk, len(selected) + 1, max_chars=max_chunk_chars, include_meta=include_meta
        )
        if selected and used + len(block) > max_chars:
            break
        blocks.append(block)
        selected.append(chunk)
        used += len(block) + 2

    if not blocks:
        return "", []
    return "\n\n".join(blocks), selected


def build_citations(
    chunks: Sequence[RetrievedChunk],
    snippets: int = 200,
) -> List[Citation]:
    """由检索结果构造引用列表（引用编号与上下文块编号一一对应）。"""
    citations: List[Citation] = []
    for index, chunk in enumerate(chunks, start=1):
        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk.chunk_id,
                doc_id=chunk.chunk.doc_id,
                source=chunk.chunk.source,
                page=chunk.chunk.page,
                snippet=chunk.text[:snippets].replace("\n", " "),
                score=round(chunk.score, 4),
            )
        )
    return citations


def parse_citations(answer: str, available: Sequence[Citation]) -> List[Citation]:
    """从答案文本中解析实际被引用的来源，只保留"答案里真的引用了"的条目。

    这一步让前端展示的引用列表与答案正文严格对应；
    同时可用于**引用准确性**评估（模块五）：模型标注的来源是否真实存在。
    """
    if not answer:
        return []
    used: List[Citation] = []
    seen: set[tuple[str, int]] = set()
    by_key = {(c.source, c.page): c for c in available}
    # 模糊匹配：模型可能写简称（"Attention Is All You Need" vs "attention is all you need.pdf"）
    normalized = {
        _normalize_source(c.source): c for c in available
    }

    for match in CITATION_PATTERN.finditer(answer):
        source = match.group("source").strip()
        page = int(match.group("page"))
        key = (source, page)
        if key in seen:
            continue
        citation = by_key.get(key) or normalized.get(_normalize_source(source))
        if citation is None:
            # 模型引用了不存在的来源 → 记为"疑似幻觉引用"，仍保留供人工核查
            citation = Citation(
                index=0, source=source, page=page, snippet="", score=0.0
            )
            citation.chunk_id = ""
            citation.doc_id = ""
        seen.add(key)
        used.append(citation)

    # 补上参考答案里的引用编号（模型可能用 [1] 形式）
    for match in re.finditer(r"(?:^|[^\w\[])\[(\d{1,2})\]", answer):
        idx = int(match.group(1))
        for citation in available:
            if citation.index == idx and (citation.source, citation.page) not in seen:
                seen.add((citation.source, citation.page))
                used.append(citation)
                break
    for order, citation in enumerate(used, start=1):
        citation.index = order
    return used


def _normalize_source(source: str) -> str:
    """来源名归一化：小写、去扩展名与空白，用于模糊匹配。"""
    text = source.lower().strip()
    text = re.sub(r"\.(pdf|docx?|txt|md|markdown)$", "", text)
    text = re.sub(r"[\s_\-]+", "", text)
    return text


def validate_citation_coverage(answer: str, min_ratio: float = 0.5) -> Dict[str, Any]:
    """评估答案的引用覆盖率（用于模块五的"引用准确性"人工/自动评分辅助）。

    判定逻辑：把答案按句切分，统计包含具体数字或断言（长度 > 20 字）的句子中，
    有多大比例带了引用标记。
    """
    sentences = [s for s in re.split(r"(?<=[。！？!?；;])\s*|\n", answer) if len(s.strip()) > 20]
    if not sentences:
        return {"sentences": 0, "cited": 0, "coverage": 0.0, "passed": False}
    cited = sum(1 for s in sentences if CITATION_PATTERN.search(s))
    coverage = cited / len(sentences)
    return {
        "sentences": len(sentences),
        "cited": cited,
        "coverage": round(coverage, 4),
        "passed": coverage >= min_ratio,
    }
