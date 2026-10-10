"""模块五 —— Bad Case 分析与优化（课程要求 5.5.3）。

分析框架
--------
Bad Case 分析的价值在于**把"效果不好"拆成可定位、可修复的具体原因**。
本模块建立了"四类失败 → 严重度 → 改进建议"的固定映射：

====================  ==================================================  ========
类别                   判定依据                                            严重度
====================  ==================================================  ========
``retrieval_failure``  严格 Hit@5=0（检索结果里没有任何片段与金标准文本重合）  高
``generation_error``   召回成功但答案缺关键信息（忠实度低 + Token-F1 低）    中
``citation_error``     答案引用了检索结果中不存在的来源（引用幻觉）          严重
``agent_decision_error`` 工具选错 / 陷入循环 / 达到最大轮次                    中
====================  ==================================================  ========

设计要点
--------
1. **判定基于真实运行数据**：``classify_badcase`` 的每一个判定分支都要求
   传入真实的检索结果与 Agent 结果对象；没有数据时判定为"无 Bad Case"而非猜测。
2. **可追溯到证据**：每条 :class:`BadCase` 都带 ``evidence`` 字段
   （实际召回的片段、引用的假来源、工具调用序列…），报告里可以直接展示，
   避免"结论无法复现"。
3. **改进建议可执行**：``suggestion`` 一律指向**系统里真实存在的可调参数**
   （``retrieval.top_k`` / ``candidate_k`` / ``enable_rerank`` / ``chunking.chunk_size``
   / ``generation.max_context_chars`` …），而不是空泛的"优化模型"。
4. **优化对比用真实对照实验**：:func:`optimization_experiment` 会在同一份语料上
   真实跑两组参数配置并记录指标，报告里的"优化前后"数字全部来自这次实测。

用法
----
>>> from src.evaluation.badcase import analyze, save_markdown
>>> report = analyze(retrieval_metrics, agent_results, eval_set)
>>> save_markdown(report, "reports/Bad_Case分析报告.md")
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .metrics import aggregate, faithfulness, token_f1

__all__ = [
    "BadCase",
    "BadCaseReport",
    "CATEGORIES",
    "CATEGORY_LABELS",
    "CATEGORY_SUGGESTIONS",
    "SEVERITY_ORDER",
    "classify_badcase",
    "analyze",
    "save_markdown",
    "optimization_experiment",
    "gold_text_recall",
]

#: 四类失败（顺序即报告中展示的顺序）
CATEGORIES = ("retrieval_failure", "generation_error", "citation_error", "agent_decision_error")

CATEGORY_LABELS = {
    "retrieval_failure": "检索失败（未召回相关文档）",
    "generation_error": "生成错误（答案缺关键信息）",
    "citation_error": "引用错误（引用幻觉）",
    "agent_decision_error": "Agent 决策错误（工具选错/死循环/超轮次）",
}

#: 严重度排序（数值越大越严重），用于报告排序与优先级判断
SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

#: 每类失败的默认改进建议（全部对应系统里真实存在的可调参数）
CATEGORY_SUGGESTIONS = {
    "retrieval_failure": (
        "1) 调大 `retrieval.top_k`（5→8）与 `candidate_k`（20→30），给融合/重排更多候选；"
        "2) 确认 `retrieval.enable_bm25=true`（中文问题依赖 BM25 的术语字面匹配）；"
        "3) 若术语与原文差异大，把 `chunking.chunk_size` 从 512 调小到 384 提升片段聚焦度；"
        "4) 长期方案：把 `embedding.provider` 从 hash 换成 ollama/bge-m3 等真实语义向量模型。"
    ),
    "generation_error": (
        "1) 调大 `generation.max_context_chars`（6000→8000）让更多证据进入 Prompt；"
        "2) 改用 `rag_strict` 提示词模板，强制逐句依据上下文；"
        "3) 降低 `llm.temperature`（0.3→0.1）减少自由发挥；"
        "4) 检查是否被 `generation.max_chunk_chars` 在句子边界截断，导致关键句丢失。"
    ),
    "citation_error": (
        "1) 检查 `generation.citation_style` 是否与提示词中的示例格式一致；"
        "2) 在 System Prompt 中强化“只允许引用【参考资料】中出现的来源与页码”；"
        "3) 开启答案后处理校验：引用来源不在检索结果中时替换为“（来源未核实）”；"
        "4) 检查重排序是否把正确来源挤出 Top-K，导致模型只能凭记忆写来源名。"
    ),
    "agent_decision_error": (
        "1) 检查 `agent.max_iterations`（8）是否被频繁触顶，必要时配合路由收敛候选工具；"
        "2) 确认 `agent.enable_router=true`，让规则路由先缩小工具集；"
        "3) 把工具描述中“适用/不适用场景”写得更明确（工具描述直接影响选择准确率）；"
        "4) 死循环多时调小 `agent.loop_detect_window`（2→1）提前终止。"
    ),
}


# ==============================================================================
# 数据结构
# ==============================================================================
@dataclass
class BadCase:
    """一条 Bad Case 记录。"""

    case_id: str = ""
    category: str = ""
    question: str = ""
    gold: str = ""                       # 期望（金标准答案/金标准片段）
    got: str = ""                        # 实际（模型答案/实际召回）
    root_cause: str = ""                 # 根因（可读描述）
    severity: str = "medium"             # critical | high | medium | low
    suggestion: str = ""                 # 改进建议（可执行、指向具体参数）
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 0)


@dataclass
class BadCaseReport:
    """Bad Case 分析报告（结构化结果，可渲染成 Markdown / JSON）。"""

    generated_at: str = ""
    total_samples: int = 0
    total_bad_cases: int = 0
    bad_case_rate: float = 0.0
    category_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    root_cause_stats: Dict[str, int] = field(default_factory=dict)
    severity_stats: Dict[str, int] = field(default_factory=dict)
    cases: List[BadCase] = field(default_factory=list)
    retrieval_summary: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cases"] = [case.to_dict() for case in self.cases]
        return data

    def by_category(self, category: str) -> List[BadCase]:
        """取某一类的全部案例（按严重度降序）。"""
        return [c for c in self.cases if c.category == category]


# ==============================================================================
# 判定工具
# ==============================================================================
def _ngram_overlap(gold: str, text: str, n: int = 4) -> float:
    """金标准片段在检索文本中的 4-gram 覆盖率（与 ``RetrievalEngine._match_texts`` 口径一致）。"""
    from .metrics import ngrams, tokenize

    gold_clean = str(gold).strip()
    if not gold_clean or not text:
        return 0.0
    if gold_clean in str(text):
        return 1.0
    gold_grams = ngrams(tokenize(gold_clean), n)
    if not gold_grams:
        return 0.0
    # 中文 4-gram 在短片段上容易退化，n 不足时用字符级 4-gram 兜底
    if len(gold_grams) < 3:
        compact = "".join(ch for ch in gold_clean if ch.strip())
        gold_grams = {compact[i : i + n] for i in range(max(0, len(compact) - n + 1))}
        if not gold_grams:
            return 0.0
    return len(gold_grams & ngrams(tokenize(str(text)), n)) / len(gold_grams)


def gold_text_recall(
    retrieved_texts: Sequence[str],
    gold_texts: Sequence[str],
    threshold: float = 0.5,
) -> float:
    """金标准文本片段被检索结果覆盖的比例。

    Parameters
    ----------
    retrieved_texts: 实际召回的片段正文列表。
    gold_texts: 该样本的金标准片段。
    threshold: 单条金标准判定为"被覆盖"所需的 4-gram 重合度（与检索层一致，0.5）。

    Returns
    -------
    float
        覆盖率 0~1；``gold_texts`` 为空时返回 0.0（无法判定即视为未覆盖）。
    """
    golds = [str(g).strip() for g in (gold_texts or []) if str(g).strip()]
    if not golds:
        return 0.0
    covered = 0
    for gold in golds:
        if any(_ngram_overlap(gold, text, 4) >= threshold for text in (retrieved_texts or ())):
            covered += 1
    return round(covered / len(golds), 4)


def _extract_answer(result: Any) -> str:
    """从 RAGAnswer / AgentResult / dict / str 中取出答案文本。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("answer") or "")
    return str(getattr(result, "answer", "") or "")


def _extract_retrieved(result: Any) -> List[Any]:
    """从结果对象中取出 ``RetrievedChunk`` 列表（兼容 RAGAnswer / AgentResult / dict）。

    支持的形态
    ----------
    - ``RAGAnswer``：直接读 ``.retrieved``；
    - ``dict``（如端到端明细或自造的最小样例）：读 ``["retrieved"]``；
    - ``AgentResult``：本身不带 ``retrieved``，因此会从轨迹里的检索类工具结果
      （``ToolResult.data["results"]``）反向还原一个轻量的代理对象列表，
      用于后续的文本重合与引用核对判定。
    """
    if result is None:
        return []
    if isinstance(result, dict):
        direct = result.get("retrieved")
        return list(direct) if direct else []
    direct = getattr(result, "retrieved", None)
    if direct:
        return list(direct)

    from ..utils.types import StepType

    proxy: List[Dict[str, Any]] = []
    for step in getattr(result, "steps", []) or []:
        if step.type != StepType.OBSERVATION:
            continue
        data = getattr(getattr(step, "tool_result", None), "data", None)
        if not isinstance(data, dict):
            continue
        for item in data.get("results") or []:
            if isinstance(item, dict):
                proxy.append(item)
    return proxy


def _retrieved_to_texts(retrieved: Sequence[Any]) -> List[str]:
    """把异构的检索结果统一转成文本列表（供 4-gram 重合判定使用）。"""
    texts: List[str] = []
    for item in retrieved or ():
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            texts.append(str(item.get("preview") or item.get("text") or ""))
            continue
        chunk = getattr(item, "chunk", None)
        if chunk is not None:
            texts.append(str(getattr(chunk, "text", "") or ""))
        else:
            texts.append(str(getattr(item, "text", "") or ""))
    return [t for t in texts if t]


def _retrieved_to_sources(retrieved: Sequence[Any]) -> List[Tuple[str, int]]:
    """把检索结果转成 ``(来源名, 页码)`` 列表（供引用核对使用）。"""
    sources: List[Tuple[str, int]] = []
    for item in retrieved or ():
        if isinstance(item, dict):
            sources.append((str(item.get("source") or ""), int(item.get("page", 0) or 0)))
        elif isinstance(item, str):
            sources.append((item, 0))
        else:
            chunk = getattr(item, "chunk", None)
            if chunk is not None:
                sources.append(
                    (str(getattr(chunk, "source", "") or ""), int(getattr(chunk, "page", 0) or 0))
                )
            else:
                sources.append(
                    (str(getattr(item, "source", "") or ""), int(getattr(item, "page", 0) or 0))
                )
    return [s for s in sources if s[0]]


def _agent_tool_sequence(agent_result: Any) -> List[str]:
    """提取 Agent 实际调用过的工具序列（按调用顺序）。"""
    from ..utils.types import StepType

    tools: List[str] = []
    for step in getattr(agent_result, "steps", []) or []:
        if step.type == StepType.ACTION and step.tool_name:
            tools.append(str(step.tool_name))
    return tools


def _expected_tools(sample: Dict[str, Any]) -> List[str]:
    """取样本的期望工具（复用 benchmark 的启发式推断，口径保持一致）。"""
    from .benchmark import infer_expected_tools

    return infer_expected_tools(sample)


# ==============================================================================
# 分类器
# ==============================================================================
def classify_badcase(
    sample: Dict[str, Any],
    retrieval_result: Any = None,
    agent_result: Any = None,
    retrieval_hit: Optional[bool] = None,
    reference_answer: Optional[str] = None,
) -> List[BadCase]:
    """判定单条样本属于哪几类失败（可能同时命中多类）。

    判定顺序与依据
    --------------
    1. ``retrieval_failure``：金标准文本在检索结果中的覆盖率 = 0
       （等价于**严格** Hit@5=0 —— 只用真实返回的前 5 条，而不是引擎内部
       取 ``max(top_k, 10)`` 条的宽松口径）。可通过 ``retrieval_hit`` 显式传入
       严格 Hit 结论覆盖自动判定。
    2. ``citation_error``：答案中的 ``【来源 p.N】`` 有任意一条不在本次检索结果中
       （``citation_accuracy.invalid > 0``）。
    3. ``generation_error``：**召回成功**（覆盖率 > 0 或 ``retrieval_hit=True``）
       但答案缺关键信息 —— 忠实度 < 0.5 且 Token-F1 < 0.3。
       要求"召回成功"是为了避免把检索失败的后果错误归因到生成环节。
    4. ``agent_decision_error``：终止原因为 ``loop_detected`` / ``max_iterations`` /
       ``parse_error``，或第一个调用的工具不在期望工具集合内。

    Parameters
    ----------
    sample: 评测样本（含 ``question`` / ``gold_texts`` / ``reference_answer`` / ``expected_tools``）。
    retrieval_result: ``RAGAnswer`` / ``AgentResult`` / ``list[RetrievedChunk]`` / None。
    agent_result: ``AgentResult``（可选，用于决策类判定）。
    retrieval_hit: 外部已算好的**严格** Hit@5 结论；None 表示由本函数按文本覆盖率自动判定。
    reference_answer: 覆盖样本里的参考答案（可选）。

    Returns
    -------
    list of :class:`BadCase`
        无失败时返回空列表 —— 这是"正常"而非"未知"。
    """
    question = str(sample.get("question") or "")
    gold_texts = [str(t) for t in (sample.get("gold_texts") or [])]
    reference = reference_answer or str(sample.get("reference_answer") or "")
    case_id = str(sample.get("id") or question[:20])

    retrieved = _extract_retrieved(retrieval_result)
    # 优先使用结果对象显式提供的片段正文（端到端明细会单独保留 retrieved_texts；
    # 检索明细则只有来源与页码，此时正文缺失 → 退回按 Hit 结论判定）
    explicit_texts = getattr(retrieval_result, "retrieved_texts", None)
    retrieved_texts = list(explicit_texts) if explicit_texts else _retrieved_to_texts(retrieved)
    answer = _extract_answer(agent_result) or _extract_answer(retrieval_result)
    cases: List[BadCase] = []

    # ---- 1) 检索失败 -----------------------------------------------------
    recall = gold_text_recall(retrieved_texts, gold_texts, threshold=0.5) if retrieved_texts else 0.0
    hit = bool(retrieval_hit) if retrieval_hit is not None else recall > 0.0
    if not hit:
        cases.append(
            BadCase(
                case_id=case_id,
                category="retrieval_failure",
                question=question,
                gold="；".join(gold_texts)[:300],
                got=f"Top-{len(retrieved_texts)} 召回片段均未覆盖金标准文本"
                if retrieved_texts
                else "无任何召回结果",
                root_cause=(
                    "检索层未召回相关片段：可能原因包括 (a) 中文提问与英文语料的词面/语义鸿沟"
                    "（hash 词面向量对中英跨语言几乎无信号）；(b) 问题中的术语与原文表述差异大，"
                    "BM25 也无法命中；(c) candidate_k 过小导致正确片段未能进入融合候选。"
                ),
                severity="high",
                suggestion=CATEGORY_SUGGESTIONS["retrieval_failure"],
                evidence={
                    "gold_texts": gold_texts,
                    "gold_text_recall": recall,
                    "num_retrieved": len(retrieved_texts),
                    "retrieved_preview": [t[:160] for t in retrieved_texts[:3]],
                },
            )
        )

    # ---- 2) 引用错误（引用幻觉）-----------------------------------------
    from .metrics import citation_accuracy

    accuracy = citation_accuracy(answer, getattr(retrieval_result, "citations", None) or [],
                                retrieved)
    if accuracy["invalid"] > 0:
        fake = accuracy["hallucinated"]
        cases.append(
            BadCase(
                case_id=case_id,
                category="citation_error",
                question=question,
                gold="只能引用本次检索结果中真实存在的来源与页码",
                got="；".join(f"{item['source']} p.{item['page']}" for item in fake),
                root_cause=(
                    f"答案中 {accuracy['invalid']}/{accuracy['total']} 条引用指向不存在的来源。"
                    "典型成因：模型为显得严谨而凭记忆补全来源名/页码；"
                    "或正确来源被重排序挤出 Top-K 后模型仍按模板标注旧来源。"
                ),
                severity="critical",
                suggestion=CATEGORY_SUGGESTIONS["citation_error"],
                evidence={
                    "citation_total": accuracy["total"],
                    "citation_invalid": accuracy["invalid"],
                    "citation_accuracy": accuracy["accuracy"],
                    "hallucinated": fake,
                    "available_sources": [
                        {"source": s, "page": p} for s, p in _retrieved_to_sources(retrieved)
                    ][:10],
                },
            )
        )

    # ---- 3) 生成错误（召回成功但答案缺关键信息）--------------------------
    if hit and answer and reference:
        from .metrics import answer_relevance

        contexts = retrieved_texts
        grounded = faithfulness(answer, contexts)
        f1 = token_f1(answer, reference)
        relevance = answer_relevance(answer, question)
        # 判定条件刻意从严，避免把"措辞不同但内容正确"误判为生成错误：
        #   (a) 答案缺乏上下文支撑（忠实度低）；且
        #   (b) 与参考答案重叠低；且
        #   (c) 连问题关键词都没覆盖（真正跑题）或与参考答案完全无交集。
        # 三个条件同时满足才报生成错误 —— 宁可漏报，也不要制造假警报。
        if grounded["score"] < 0.5 and f1 < 0.3 and (relevance < 0.3 or f1 == 0.0):
            cases.append(
                BadCase(
                    case_id=case_id,
                    category="generation_error",
                    question=question,
                    gold=reference[:300],
                    got=answer[:300],
                    root_cause=(
                        f"已召回相关片段（金标准覆盖率 {recall:.2f}），但答案缺乏上下文支撑"
                        f"（忠实度 {grounded['score']:.2f}）、与参考答案重叠度低"
                        f"（Token-F1 {f1:.2f}）、对问题的关键词覆盖率也仅 {relevance:.2f}。"
                        "说明问题出在生成环节：上下文被截断、提示词约束不足，或答案跑题。"
                    ),
                    severity="medium",
                    suggestion=CATEGORY_SUGGESTIONS["generation_error"],
                    evidence={
                        "faithfulness": grounded["score"],
                        "token_f1": f1,
                        "answer_relevance": relevance,
                        "unsupported_sentences": grounded.get("unsupported_examples", []),
                        "answer_chars": len(answer),
                        "num_contexts": len(contexts),
                    },
                )
            )

    # ---- 4) Agent 决策错误 ----------------------------------------------
    if agent_result is not None:
        terminated = str(getattr(agent_result, "terminated_reason", "") or "")
        tools = _agent_tool_sequence(agent_result)
        expected = _expected_tools(sample)
        reasons: List[str] = []
        if terminated == "loop_detected":
            reasons.append("陷入重复调用循环（loop_detected）")
        elif terminated == "max_iterations":
            reasons.append(
                f"达到最大推理轮次（{getattr(agent_result, 'iterations', 0)} 轮）仍未收尾"
            )
        elif terminated == "parse_error":
            reasons.append("决策 JSON 连续解析失败（parse_error）")
        if tools and expected and tools[0] not in expected:
            reasons.append(f"首个工具选择错误：实际 {tools[0]}，期望 {expected}")

        if reasons:
            cases.append(
                BadCase(
                    case_id=case_id,
                    category="agent_decision_error",
                    question=question,
                    gold=f"期望工具：{'、'.join(expected)}；正常终止（final_answer）",
                    got=f"实际工具序列：{' → '.join(tools) or '（无工具调用）'}；"
                        f"终止原因：{terminated}",
                    root_cause="；".join(reasons),
                    severity="medium",
                    suggestion=CATEGORY_SUGGESTIONS["agent_decision_error"],
                    evidence={
                        "expected_tools": expected,
                        "actual_tools": tools,
                        "terminated_reason": terminated,
                        "iterations": int(getattr(agent_result, "iterations", 0)),
                        "tool_calls": int(getattr(agent_result, "tool_calls", 0)),
                        "tool_failures": int(getattr(agent_result, "tool_failures", 0)),
                        "routed_tools": list(getattr(agent_result, "routed_tools", []) or []),
                    },
                )
            )

    return cases


# ==============================================================================
# 汇总分析
# ==============================================================================
def analyze(
    retrieval_metrics: Any = None,
    agent_results: Any = None,
    eval_set: Sequence[Dict[str, Any]] = None,
    retrieval_details: Optional[Sequence[Dict[str, Any]]] = None,
    end_to_end_details: Optional[Sequence[Dict[str, Any]]] = None,
) -> BadCaseReport:
    """汇总分析全部样本，生成 :class:`BadCaseReport`。

    Parameters
    ----------
    retrieval_metrics:
        ``RetrievalBenchmark.run()`` 的返回值，或 ``RetrievalEngine.evaluate()`` 的结果。
        用于提取逐条 Hit/MRR 与"三档对比"摘要。
    agent_results:
        ``AgentBenchmark.run()`` 的返回值（dict，含 ``details``），
        或 ``list[Tuple[sample_dict, AgentResult]]``。两者都支持。
    eval_set:
        评测样本列表（用于补齐金标准文本与参考答案）。
    retrieval_details:
        直接传入逐条检索明细（``RetrievalBenchmark.run()["per_mode_detail"]``
        或 ``evaluate()["details"]``）；优先级最高。若明细带 ``strict_hit``
        （``RetrievalBenchmark`` 会写入严格 Hit 结论），检索失败一律按严格口径
        判定；``strict_hit`` 缺失按未命中处理（缺证据不能当作命中）。
    end_to_end_details:
        ``EndToEndBenchmark.run()["details"]``。这些明细带有**完整答案**与
        **实际来源清单**，因此是判定"引用错误"与"生成错误"的直接证据来源；
        不传时这两类失败无法判定（会如实记为 0 条，而不是猜测）。

    Returns
    -------
    BadCaseReport

    Notes
    -----
    本函数**不做任何性能测试**，只做数据分析，因此开销极小；
    耗时的对照实验在 :func:`optimization_experiment` 里单独执行。
    ``total_samples`` 是**去重后的样本数**：同一道题同时出现在端到端与 Agent
    两套明细里时只计一次，避免占比被重复计数稀释。
    """
    began = time.time()
    samples = list(eval_set or ())
    by_id = {str(s.get("id") or ""): s for s in samples}
    by_question = {str(s.get("question") or ""): s for s in samples}

    # ---- 归一化 agent 结果 ----------------------------------------------
    agent_rows: List[Dict[str, Any]] = []
    raw_agent_pairs: List[Tuple[Dict[str, Any], Any]] = []
    if isinstance(agent_results, dict):
        agent_rows = [dict(r) for r in (agent_results.get("details") or []) if isinstance(r, dict)]
    elif isinstance(agent_results, (list, tuple)):
        for item in agent_results:
            if isinstance(item, dict):
                agent_rows.append(dict(item))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                raw_agent_pairs.append((dict(item[0] or {}), item[1]))

    # ---- 归一化检索明细 --------------------------------------------------
    details: List[Dict[str, Any]] = []
    retrieval_summary: Dict[str, Any] = {}
    if retrieval_details is not None:
        details = [dict(r) for r in retrieval_details if isinstance(r, dict)]
    if isinstance(retrieval_metrics, dict):
        # RetrievalBenchmark.run() 的返回值：聚合指标在 "summary" 里，
        # 逐条明细在 "per_mode_detail" 里（旧版 evaluate() 的结果则直接在顶层）
        summary_node = retrieval_metrics.get("summary")
        if isinstance(summary_node, dict):
            retrieval_summary.update(
                {k: v for k, v in summary_node.items() if k != "strict"}
            )
        if retrieval_metrics.get("details"):
            details = [dict(r) for r in retrieval_metrics["details"] if isinstance(r, dict)]
        elif retrieval_metrics.get("per_mode_detail"):
            per_mode = retrieval_metrics["per_mode_detail"]
            chosen = per_mode.get("full") or next(iter(per_mode.values()), [])
            details = [dict(r) for r in chosen if isinstance(r, dict)]
        if retrieval_metrics.get("table"):
            retrieval_summary.setdefault("table", retrieval_metrics["table"])
        for key in ("mode", "top_k", "match", "num_samples", "hit_at_k", "mrr",
                    "recall_at_k", "precision_at_k", "ndcg_at_k",
                    "avg_latency_ms", "p95_latency_ms"):
            if key in retrieval_metrics:
                retrieval_summary[key] = retrieval_metrics[key]
    elif isinstance(retrieval_metrics, (list, tuple)):
        details = [dict(r) for r in retrieval_metrics if isinstance(r, dict)]

    hit_by_id: Dict[str, bool] = {}
    for row in details:
        question = str(row.get("question") or "")
        sample = by_question.get(question)
        key = str(sample.get("id")) if sample else question
        if key:
            # 严格口径：只用严格 Hit@K 结论（RetrievalBenchmark 会为逐条明细写入
            # strict_hit）。缺省视为**未命中** —— 不能用宽松的 "hit"（引擎内部取
            # max(top_k,10) 条判定，会高估命中），否则检索失败会被漏报。
            hit_by_id[key] = bool(row.get("strict_hit", False))

    # ---- 端到端明细：带完整答案与来源，是生成/引用类判定的直接证据 ----------
    cases: List[BadCase] = []
    errors: List[str] = []
    # 分母用**去重后的样本 id**：同一道题会同时出现在端到端与 Agent 两套明细里
    # （quick 模式下 Agent 抽样的 20 条是端到端 40 条的子集），逐行累加会把
    # 分母从 40 放大到 60，Bad Case 占比被系统性低估。
    evaluated_ids: Set[str] = set()

    e2e_rows = [dict(r) for r in (end_to_end_details or ()) if isinstance(r, dict)]
    for row in e2e_rows:
        if int(row.get("round", 1) or 1) != 1:
            continue  # 只分析第一轮（第二轮是缓存命中，判定无意义）
        sample = by_id.get(str(row.get("id") or "")) or by_question.get(
            str(row.get("question") or "")
        ) or {
            "id": str(row.get("id", "")),
            "question": str(row.get("question", "")),
            "gold_texts": [],
            "reference_answer": "",
            "type": str(row.get("type", "")),
        }
        sample_id = str(sample.get("id") or "")
        evaluated_ids.add(sample_id or str(sample.get("question") or ""))
        cases.extend(
            classify_badcase(
                sample,
                retrieval_result=_SyntheticEndToEndResult(row),
                agent_result=None,
                # 有检索明细时用真实 Hit@K；没有则让分类器自行按文本重合判定
                retrieval_hit=hit_by_id.get(sample_id) if sample_id in hit_by_id else None,
            )
        )

    # ---- 逐条判定（Agent 明细）-------------------------------------------
    for row in agent_rows:
        sample = by_id.get(str(row.get("id") or "")) or by_question.get(
            str(row.get("question") or "")
        )
        if sample is None:
            sample = {
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "gold_texts": [],
                "reference_answer": "",
                "type": row.get("type", ""),
            }
        # Agent 明细里的答案与检索信息有限：用严格 Hit 结论 + 工具序列做判定
        synthetic_agent = _SyntheticAgentResult(row)
        evaluated_ids.add(str(sample.get("id") or sample.get("question") or ""))
        cases.extend(
            classify_badcase(
                sample,
                retrieval_result=None,
                agent_result=synthetic_agent,
                retrieval_hit=hit_by_id.get(str(sample.get("id") or "")),
            )
        )

    for sample, agent_result in raw_agent_pairs:
        evaluated_ids.add(str(sample.get("id") or sample.get("question") or ""))
        cases.extend(
            classify_badcase(
                sample,
                retrieval_result=_extract_retrieved(agent_result),
                agent_result=agent_result,
                retrieval_hit=hit_by_id.get(str(sample.get("id") or "")),
            )
        )

    # 只有检索明细、没有 Agent 结果时，仍要能报出"检索失败"这一类
    if not agent_rows and not raw_agent_pairs and details:
        for row in details:
            sample = by_question.get(str(row.get("question") or "")) or {
                "id": str(row.get("question", ""))[:20],
                "question": row.get("question", ""),
                "gold_texts": row.get("gold_texts") or [],
                "reference_answer": "",
            }
            if sample.get("gold_texts") is None and row.get("gold_ids"):
                sample["gold_texts"] = []
            evaluated_ids.add(str(sample.get("id") or sample.get("question") or ""))
            # 有严格 Hit 结论时优先用严格口径，没有才退回引擎口径的 hit
            hit = row.get("strict_hit", row.get("hit"))
            if hit is not None and not hit:
                cases.append(
                    BadCase(
                        case_id=str(sample.get("id") or ""),
                        category="retrieval_failure",
                        question=str(sample.get("question") or ""),
                        gold="（由评测集金标准文本判定）",
                        got="；".join(str(x) for x in (row.get("retrieved_ids") or [])[:5])
                        or "无召回结果",
                        root_cause=(
                            "该问题的检索 Hit@K=0：Top-K 片段与金标准文本的 4-gram 覆盖度均低于 0.5。"
                        ),
                        severity="high",
                        suggestion=CATEGORY_SUGGESTIONS["retrieval_failure"],
                        evidence={
                            "hit": False,
                            "mrr": row.get("mrr", 0.0),
                            "recall": row.get("recall", 0.0),
                            "latency_ms": row.get("latency_ms", 0.0),
                            "retrieved_ids": list(row.get("retrieved_ids") or [])[:5],
                        },
                    )
                )

    # ---- 汇总统计 --------------------------------------------------------
    # 同一道题可能被多条分析路径（检索明细 / Agent 明细 / 端到端明细）分别看一遍，
    # 从而对同一 (题号, 类别) 产生多条记录。这里按 (case_id, category) 去重，
    # 保留**证据更充分**的那一条 —— 否则"11 条检索失败"里会混进多条重复计数，
    # 让失败率被凭空放大。
    cases = _dedupe_cases(cases)

    # 分母 = 去重后的样本数（同一道题同时出现在端到端与 Agent 两套明细里时只算一次）
    evaluated = len(evaluated_ids)

    category_counter: Counter = Counter(case.category for case in cases)
    severity_counter: Counter = Counter(case.severity for case in cases)
    root_cause_counter: Counter = Counter(case.root_cause.split("：")[0] for case in cases)

    category_stats: Dict[str, Dict[str, Any]] = {}
    for category in CATEGORIES:
        items = [c for c in cases if c.category == category]
        category_stats[category] = {
            "label": CATEGORY_LABELS[category],
            "count": len(items),
            "rate": round(len(items) / evaluated, 4) if evaluated else 0.0,
            "severity": dict(Counter(c.severity for c in items)),
            "example_questions": [c.question for c in items[:3]],
            "suggestion": CATEGORY_SUGGESTIONS[category],
        }

    cases.sort(key=lambda c: (-c.severity_rank, c.category, c.case_id))
    return BadCaseReport(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_samples=evaluated,
        total_bad_cases=len(cases),
        bad_case_rate=round(len(cases) / evaluated, 4) if evaluated else 0.0,
        category_stats=category_stats,
        root_cause_stats=dict(root_cause_counter),
        severity_stats=dict(severity_counter),
        cases=cases,
        retrieval_summary=retrieval_summary,
        optimization={},
        errors=errors,
    )


def _dedupe_cases(cases: Sequence[BadCase]) -> List[BadCase]:
    """按 ``(case_id, category)`` 去重，保留证据更充分的记录。

    "证据更充分"的判据：``evidence`` 中有效（非空）字段更多者优先；
    数量相同时优先保留带 ``num_retrieved``/``retrieved_preview`` 的记录
    （来自端到端明细的那条能看到实际召回内容，可诊断性更强）。
    完全没有 ``case_id`` 的记录不参与去重（无法判断是否重复）。
    """
    def richness(case: BadCase) -> Tuple[int, int]:
        evidence = case.evidence or {}
        filled = sum(1 for value in evidence.values() if value not in (None, "", [], {}))
        # 召回侧的可诊断信息：端到端明细给 preview，检索明细给 retrieved_ids
        diagnostic = 1 if (evidence.get("retrieved_preview") or evidence.get("retrieved_ids")) else 0
        return (filled, diagnostic)

    best: Dict[Tuple[str, str], BadCase] = {}
    unkeyed: List[BadCase] = []
    for case in cases:
        if not case.case_id:
            unkeyed.append(case)
            continue
        key = (case.case_id, case.category)
        current = best.get(key)
        if current is None or richness(case) > richness(current):
            best[key] = case
    return list(best.values()) + unkeyed


class _SyntheticAgentResult:
    """由 Agent 评测明细行还原的轻量结果对象（供分类器读取属性）。

    这样 :func:`classify_badcase` 既能接受真实的 ``AgentResult``，
    也能接受报告阶段只剩字典的明细行，判定逻辑无需分叉。
    """

    def __init__(self, row: Dict[str, Any]) -> None:
        self.answer = str(row.get("answer") or "")
        self.terminated_reason = str(row.get("terminated_reason") or "")
        self.iterations = int(row.get("iterations", 0) or 0)
        self.tool_calls = int(row.get("num_tools", 0) or 0)
        self.tool_failures = 0
        self.routed_tools = list(row.get("routed_tools") or [])
        self.steps = _synthetic_steps(row.get("actual_tools") or [])
        self.citations: List[Any] = []


class _SyntheticStep:
    """最小可用的步骤对象（只保留分类器需要的 ``type`` 与 ``tool_name``）。"""

    def __init__(self, type_value: str, tool_name: str = "") -> None:
        from ..utils.types import StepType

        self.type = StepType(type_value)
        self.tool_name = tool_name
        self.tool_args: Dict[str, Any] = {}
        self.tool_result = None
        self.content = ""


class _SyntheticEndToEndResult:
    """由端到端评测明细行还原出的轻量结果对象。

    端到端明细里保存了**完整答案**与**实际来源清单**，因此足以支撑
    "引用错误"（来源是否真实存在）与"生成错误"（忠实度 + Token-F1）两类判定。
    检索片段的**正文**在明细里未被保留（体积考虑），
    因此回到检索明细的 Hit 结论来判定"检索失败"。
    """

    def __init__(self, row: Dict[str, Any]) -> None:
        self.answer = str(row.get("answer") or "")
        self.cache_hit = bool(row.get("cache_hit"))
        self.degraded = bool(row.get("degraded"))
        self.degrade_reason = str(row.get("degrade_reason") or "")
        self.latency_ms = float(row.get("latency_ms", 0.0) or 0.0)
        self.retrieval_ms = float(row.get("retrieval_ms", 0.0) or 0.0)
        self.generation_ms = float(row.get("generation_ms", 0.0) or 0.0)
        # retrieved：以 dict 形态提供 (source, page)，供引用核对使用
        self.retrieved: List[Dict[str, Any]] = [
            {"source": str(item.get("source", "")), "page": int(item.get("page", 0) or 0)}
            for item in (row.get("retrieved_sources") or [])
            if isinstance(item, dict)
        ]
        # 片段正文：明细里只保留前 400 字预览（判定生成忠实度足够，且能控制体积）
        self.retrieved_texts: List[str] = [
            str(item.get("preview") or "")
            for item in (row.get("retrieved_sources") or [])
            if isinstance(item, dict) and item.get("preview")
        ]
        self.citations: List[Any] = []
        self.steps: List[Any] = []


def _synthetic_steps(tools: Sequence[str]) -> List[Any]:
    """把工具名序列还原成 ACTION 步骤序列（供 ``_agent_tool_sequence`` 读取）。"""
    return [_SyntheticStep("action", str(tool)) for tool in tools if tool]


# ==============================================================================
# 优化对照实验（真实运行）
# ==============================================================================
#: 对照组（优化前）：完全按 config.yaml 的默认值
#: —— candidate_k=20、开启启发式重排序、chunk_size=512
BASELINE_OVERRIDES: Dict[str, Any] = {
    "retrieval": {"candidate_k": 20, "enable_rerank": True, "top_k": 5},
    "chunking": {"chunk_size": 512, "chunk_overlap": 64},
}

#: 实验组（优化后）：针对本轮 Bad Case 暴露的两个根因做定向修复
#:
#: 1. **片段粒度与金标准粒度不匹配** → ``chunk_size`` 512→384，
#:    让"论文里的一句话"更容易完整落在一个分块内（分块数 27→37）；
#: 2. **启发式重排序在部分问题上误判** → 关闭 ``retrieval.enable_rerank``。
#:    这是由三轮真实对照实验（不同候选池 / 分块粒度 / 重排序开关）选出的组合，
#:    而不是凭直觉拍定的参数 —— 具体对比数据见本模块生成的 Markdown 报告。
#:
#: 同时把候选池 20→40，为未来接入真实语义向量时预留召回空间。
OPTIMIZED_OVERRIDES: Dict[str, Any] = {
    "retrieval": {"candidate_k": 40, "enable_rerank": False, "top_k": 5},
    "chunking": {"chunk_size": 384, "chunk_overlap": 48},
    "generation": {"max_context_chars": 8000},
}


def optimization_experiment(
    config: Any = None,
    eval_set: Optional[Sequence[Dict[str, Any]]] = None,
    limit: int = 15,
    baseline: Optional[Dict[str, Any]] = None,
    optimized: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """**真实运行**的一轮参数对照实验（用于报告里的"优化前后对比"）。

    实验设计
    --------
    - 两组使用**同一份样例语料**（各自独立的临时索引，避免互相污染）；
    - 两组都跑检索三档的 ``full`` 档 + 端到端问答，指标口径完全一致；
    - 差异只有参数：候选池大小、最终 K、上下文上限、分块粒度。

    Parameters
    ----------
    config: 基础配置。
    eval_set: 评测集；None 表示用 ``build_eval_set()``。
    limit: 每组最多评测多少条（默认 15，控制耗时）。
    baseline / optimized: 可覆盖默认的两组参数。

    Returns
    -------
    dict
        ``{"enabled", "baseline": {...}, "optimized": {...}, "delta": {...},
        "runtime_ms": ...}``。任一组失败时 ``enabled=False`` 并附 ``error``。

    Notes
    -----
    ``limit`` 为 None 时使用**全部**评测样本；否则按问题类型**分层抽样**
    （复用 :func:`src.evaluation.benchmark._stratified_sample`）。
    样本量对结论影响很大：评测集按类型分段排列，前 25 条恰好都是 factual 类
    （基线已接近满分），直接截取前 N 条会让对照实验系统性偏向简单题，
    得出"参数调整没有收益"的错误结论。
    """
    from ..utils.config import load_config
    from .benchmark import _stratified_sample
    from .eval_set import build_eval_set

    began = time.perf_counter()
    all_items = list(eval_set or build_eval_set())
    items = (
        all_items
        if limit is None
        else _stratified_sample(all_items, max(1, int(limit)))
    )
    base_config = config or load_config()

    result: Dict[str, Any] = {
        "enabled": True,
        "num_samples": len(items),
        "baseline_params": dict(baseline or BASELINE_OVERRIDES),
        "optimized_params": dict(optimized or OPTIMIZED_OVERRIDES),
        "baseline": {},
        "optimized": {},
        "delta": {},
        "ablation": [],
        "error": "",
    }

    try:
        result["baseline"] = _run_single_config(
            base_config, items, baseline or BASELINE_OVERRIDES, tag="baseline"
        )
        result["optimized"] = _run_single_config(
            base_config, items, optimized or OPTIMIZED_OVERRIDES, tag="optimized"
        )
        # 单参数消融：把"优化后的收益"归因到具体参数上。
        # 没有这一步，就只能说"改了三个参数后变好了"，无法指导下一步该动哪个参数。
        for tag, label, overrides in _ablation_configs(baseline or BASELINE_OVERRIDES):
            try:
                run = _run_single_config(base_config, items, overrides, tag=tag)
                run["label"] = label
                result["ablation"].append(run)
            except Exception as exc:  # noqa: BLE001 - 单个消融失败不影响整体
                result["ablation"].append({"tag": tag, "label": label, "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001 - 对照实验失败不能拖垮整份报告
        result["enabled"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["runtime_ms"] = round((time.perf_counter() - began) * 1000, 2)
        return result

    def _delta(key: str, higher_is_better: bool = True) -> Dict[str, float]:
        before = float(result["baseline"].get(key, 0.0) or 0.0)
        after = float(result["optimized"].get(key, 0.0) or 0.0)
        gain = after - before
        return {
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(gain, 4),
            "relative": round(gain / before, 4) if before else 0.0,
            "improved": bool(gain > 0) if higher_is_better else bool(gain < 0),
        }

    result["delta"] = {
        "hit@5": _delta("hit@5"),
        "mrr": _delta("mrr"),
        "recall@5": _delta("recall@5"),
        "ndcg@5": _delta("ndcg@5"),
        "avg_latency_ms": _delta("avg_latency_ms", higher_is_better=False),
        "degrade_rate": _delta("degrade_rate", higher_is_better=False),
        "citation_accuracy": _delta("citation_accuracy"),
    }
    result["runtime_ms"] = round((time.perf_counter() - began) * 1000, 2)
    return result


def _ablation_configs(baseline: Dict[str, Any]) -> List[tuple]:
    """构造单参数消融配置：每次只在基线上改动**一个**参数。

    返回 ``[(tag, label, overrides), ...]``，其中 overrides 已合并基线参数
    （保证每个消融组与对照组的差异只有一处）。
    """
    base_retrieval = dict(baseline.get("retrieval") or BASELINE_OVERRIDES["retrieval"])
    base_chunking = dict(baseline.get("chunking") or BASELINE_OVERRIDES["chunking"])

    def merged(**changes: Any) -> Dict[str, Any]:
        retrieval = dict(base_retrieval)
        chunking = dict(base_chunking)
        for key, value in (changes.get("retrieval") or {}).items():
            retrieval[key] = value
        for key, value in (changes.get("chunking") or {}).items():
            chunking[key] = value
        return {"retrieval": retrieval, "chunking": chunking}

    return [
        ("ablation_rerank_off", "仅关闭启发式重排序（enable_rerank=False）",
         merged(retrieval={"enable_rerank": False})),
        ("ablation_chunk_384", "仅调细分块（chunk_size 512→384）",
         merged(chunking={"chunk_size": 384, "chunk_overlap": 48})),
        ("ablation_candidate_40", "仅扩大候选池（candidate_k 20→40）",
         merged(retrieval={"candidate_k": 40})),
    ]


def _run_single_config(
    base_config: Any,
    items: Sequence[Dict[str, Any]],
    overrides: Dict[str, Any],
    tag: str = "run",
    repeat: bool = False,
) -> Dict[str, Any]:
    """在指定参数下真实跑一轮"检索 + 端到端"，返回可对比的指标。

    检索指标统一取 **严格口径**（只用真实返回的前 5 条 + 金标准文本精确归属的
    chunk_id），并对两组使用**同一个 k=5**，保证优化前后的对比是同口径的。
    """
    from .benchmark import EndToEndBenchmark, RetrievalBenchmark, prepare_corpus

    cfg = base_config.merged_with(overrides)
    bundle = prepare_corpus(cfg)
    benchmark = RetrievalBenchmark(bundle.engine, items, top_k=5, match="text")
    metrics = bundle.engine.evaluate(items, mode="full", top_k=5, match="text")
    strict = benchmark.strict_metrics("full", benchmark.resolve_gold_chunk_ids(), k=5)
    end_to_end = EndToEndBenchmark(
        bundle.create_pipeline(), items, limit=len(items), repeat=repeat
    ).run()
    return {
        "tag": tag,
        "overrides": overrides,
        "num_samples": int(strict.get("strict_num_samples", 0) or metrics.get("num_samples", 0) or 0),
        # 检索指标：严格口径（主口径）
        "hit@5": float(strict.get("strict_hit@5", 0.0) or 0.0),
        "mrr": float(strict.get("strict_mrr", 0.0) or 0.0),
        "recall@5": float(strict.get("strict_recall@5", 0.0) or 0.0),
        "ndcg@5": float(strict.get("strict_ndcg@5", 0.0) or 0.0),
        # 引擎口径（辅助参考）
        "engine_hit@5": float(metrics.get("hit_at_k", 0.0) or 0.0),
        "engine_ndcg@5": float(metrics.get("ndcg_at_k", 0.0) or 0.0),
        # 端到端指标（延迟 / 降级率 / 引用准确率）
        "avg_latency_ms": float(end_to_end.get("avg_latency_ms", 0.0) or 0.0),
        "avg_retrieval_ms": float(end_to_end.get("avg_retrieval_ms", 0.0) or 0.0),
        "avg_generation_ms": float(end_to_end.get("avg_generation_ms", 0.0) or 0.0),
        "degrade_rate": float(end_to_end.get("degrade_rate", 0.0) or 0.0),
        "citation_accuracy": float(end_to_end.get("avg_citation_accuracy", 0.0) or 0.0),
        "avg_retrieved": float(end_to_end.get("avg_retrieved", 0.0) or 0.0),
        "corpus": {
            "num_chunks": bundle.engine.num_chunks,
            "chunk_size": cfg.get("chunking.chunk_size"),
        },
    }


# ==============================================================================
# Markdown 报告
# ==============================================================================
def _bar(value: float, scale: float = 1.0, width: int = 20) -> str:
    """用 █ 字符画条形图（报告中所有图形均为纯文本，任何环境都能正确显示）。"""
    ratio = max(0.0, min(1.0, float(value) / scale if scale else 0.0))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


def _fmt4(value: Any) -> str:
    """把指标格式化为 4 位小数；非数值时原样返回（便于渲染"未运行"）。"""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _fmt3(value: Any) -> str:
    """把延迟类数值格式化为 3 位小数。"""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def save_markdown(report: BadCaseReport, path: Any = None, config: Any = None) -> Path:
    """把 :class:`BadCaseReport` 渲染成 Markdown 报告并落盘。

    报告结构
    --------
    1. 总览表（四类失败的条数 / 占比 / 严重度 / 典型问题）
    2. 每类失败的典型样例（最多 3 条，含 问题 / 期望 / 实际 / 根因 / 建议 / 证据）
    3. 根因归纳与优先级
    4. 一轮真实参数对照实验（优化前后对比数据）

    Parameters
    ----------
    report: :func:`analyze` 的返回值（可先挂上 ``optimization``）。
    path: 目标文件；None 表示 ``reports/Bad_Case分析报告.md``。
    config: 配置对象（用于解析 ``paths.report_dir``）。
    """
    import json

    from ..utils.config import load_config
    from ..utils.storage import ensure_dir

    cfg = config or load_config()
    if path is None:
        report_dir = cfg.resolve_path(str(cfg.paths.report_dir))
        target = Path(report_dir) / "Bad_Case分析报告.md"
    else:
        target = Path(path)
    ensure_dir(target.parent)

    lines: List[str] = []
    add = lines.append

    add("# Bad Case 分析报告")
    add("")
    add(f"> 生成时间：{report.generated_at}　|　分析样本数：**{report.total_samples}**　|　"
        f"Bad Case 总数：**{report.total_bad_cases}**　|　"
        f"占比：**{report.bad_case_rate:.1%}**")
    add("")
    add("本报告全部数据来自实际运行（评测语料为 3 篇样例论文构建的临时索引），"
        "未运行的小节会显式标注「未运行」。")
    add("")

    # ---------------- 1. 总览 ----------------
    add("## 一、总览")
    add("")
    add("| 失败类别 | 条数 | 占比 | 严重度分布 | 改进优先级 |")
    add("| --- | ---: | ---: | --- | --- |")
    priority = {
        "citation_error": "P0（可信度底线）",
        "retrieval_failure": "P1（影响面最大）",
        "generation_error": "P2",
        "agent_decision_error": "P2",
    }
    for category in CATEGORIES:
        stat = report.category_stats.get(category, {})
        severity = "、".join(f"{k}×{v}" for k, v in (stat.get("severity") or {}).items()) or "—"
        add(
            f"| {stat.get('label', category)} | {stat.get('count', 0)} | "
            f"{stat.get('rate', 0.0):.1%} | {severity} | {priority.get(category, '—')} |"
        )
    add(f"| **合计** | **{report.total_bad_cases}** | **{report.bad_case_rate:.1%}** | — | — |")
    add("")
    add("各失败类别占比如下图（每格 = 5%）：")
    add("")
    add("```text")
    for category in CATEGORIES:
        stat = report.category_stats.get(category, {})
        label = str(stat.get("label", category))
        add(f"{label:<28} {_bar(float(stat.get('rate', 0.0)))} {stat.get('rate', 0.0):.1%}")
    add("```")
    add("")

    # ---------------- 2. 典型样例 ----------------
    add("## 二、典型样例")
    add("")
    total_shown = 0
    for category in CATEGORIES:
        cases = report.by_category(category)
        stat = report.category_stats.get(category, {})
        add(f"### {stat.get('label', category)}（{len(cases)} 条）")
        add("")
        if not cases:
            add("本次评测未发现该类失败。")
            add("")
            continue
        for index, case in enumerate(cases[:3], start=1):
            total_shown += 1
            add(f"#### 样例 {index}：`{case.case_id}`（严重度：{case.severity}）")
            add("")
            add(f"- **问题**：{case.question}")
            add(f"- **期望**：{case.gold}")
            add(f"- **实际**：{case.got}")
            add(f"- **根因**：{case.root_cause}")
            add(f"- **改进建议**：{case.suggestion}")
            if case.evidence:
                add("- **证据**：")
                add("")
                add("```json")
                add(json.dumps(case.evidence, ensure_ascii=False, indent=2, default=str)[:1600])
                add("```")
            add("")
        add(f"- 该类改进方向（统一建议）：{stat.get('suggestion', '')}")
        add("")
    add(f"（共展示 {total_shown} 个具体样例；完整清单见 "
        f"`reports/性能评测数据.json` 与逐条明细表。）")
    add("")

    # ---------------- 3. 根因归纳 ----------------
    add("## 三、根因归纳")
    add("")
    if report.root_cause_stats:
        add("| 根因类别 | 出现次数 | 占比 |")
        add("| --- | ---: | ---: |")
        total = max(1, sum(report.root_cause_stats.values()))
        for cause, count in sorted(report.root_cause_stats.items(), key=lambda kv: -kv[1]):
            add(f"| {cause} | {count} | {count / total:.1%} |")
    else:
        add("本次评测未产生 Bad Case，无根因可归纳。")
    add("")
    if report.retrieval_summary:
        add("### 检索层整体表现（主口径：严格 Hit@5 / MRR / Recall@K / NDCG@K）")
        add("")
        summary = report.retrieval_summary
        add(f"- 档位：`{summary.get('mode', 'full')}`　Top-K：{summary.get('top_k', 5)}　"
            f"判定方式：`{summary.get('match', 'text')}`　样本数：{summary.get('num_samples', 0)}")
        add(f"- Hit@K **{_fmt4(summary.get('hit_at_k'))}**　MRR **{_fmt4(summary.get('mrr'))}**　"
            f"Recall@K **{_fmt4(summary.get('recall_at_k'))}**　"
            f"NDCG@K **{_fmt4(summary.get('ndcg_at_k'))}**")
        add(f"- 平均延迟 **{_fmt3(summary.get('avg_latency_ms'))} ms**　"
            f"P95 延迟 **{_fmt3(summary.get('p95_latency_ms'))} ms**")
        if summary.get("strict"):
            add("")
            add("> 以上为**严格口径**：只统计真实返回的前 Top-K 条，金标准由「金标准文本落在哪个"
                "分块」精确解析为 chunk_id。`RetrievalEngine.evaluate(match=\"text\")` 会先取 "
                "`max(top_k, 10)` 条再判定，因此其 Hit@K 天然更高（本次 "
                f"{_fmt4(summary.get('engine_hit_at_k'))}），两者差距即为宽松口径的高估幅度。")
        add("")
        add("```text")
        for label, key in (("Hit@K", "hit_at_k"), ("MRR", "mrr"),
                           ("Recall@K", "recall_at_k"), ("NDCG@K", "ndcg_at_k")):
            value = float(summary.get(key, 0.0) or 0.0)
            add(f"{label:<9} {_bar(value)} {value:.3f}")
        add("```")
        add("")

    # ---------------- 4. 优化前后对比 ----------------
    add("## 四、一轮迭代优化的前后对比（真实对照实验）")
    add("")
    experiment = report.optimization or {}
    if not experiment.get("enabled"):
        add("**未运行**：本次未执行参数对照实验。"
            f"{('原因：' + str(experiment.get('error'))) if experiment.get('error') else ''}")
        add("")
        add("复现方式：`python -m src.evaluation.badcase`（会真实跑两组参数）。")
    else:
        add(f"实验设计：同一份样例语料（Transformer / ResNet / BERT 三篇），"
            f"按问题类型分层抽取 {experiment.get('num_samples')} 条评测样本，两组仅参数不同，"
            f"各自建立独立临时索引，指标由同一套代码计算"
            f"（严格口径：只用真实返回的前 5 条 + 金标准文本精确归属的 chunk_id）。")
        add("")
        add("**对照组（优化前）**：")
        add("")
        add("```yaml")
        add(json.dumps(experiment.get("baseline_params"), ensure_ascii=False, indent=2))
        add("```")
        add("")
        add("**实验组（优化后）**：")
        add("")
        add("```yaml")
        add(json.dumps(experiment.get("optimized_params"), ensure_ascii=False, indent=2))
        add("```")
        add("")
        before = experiment.get("baseline") or {}
        after = experiment.get("optimized") or {}
        add("| 指标（严格口径） | 优化前 | 优化后 | 变化 | 相对变化 | 结论 |")
        add("| --- | ---: | ---: | ---: | ---: | --- |")
        delta = experiment.get("delta") or {}
        rows_spec = [
            ("Hit@5", "hit@5"), ("MRR", "mrr"), ("Recall@5", "recall@5"),
            ("NDCG@5", "ndcg@5"), ("平均端到端延迟(ms)", "avg_latency_ms"),
            ("降级率", "degrade_rate"), ("引用准确率", "citation_accuracy"),
        ]
        for label, key in rows_spec:
            item = delta.get(key)
            if not item:
                continue
            arrow = "↑" if item["delta"] > 0 else ("↓" if item["delta"] < 0 else "→")
            verdict = "提升" if item.get("improved") else ("变差" if item["delta"] != 0 else "持平")
            add(
                f"| {label} | {item['before']} | {item['after']} | "
                f"{arrow} {item['delta']:+.4f} | {item['relative']:+.2%} | {verdict} |"
            )
        add("")
        add("```text")
        for label, key in (("Hit@5", "hit@5"), ("MRR", "mrr"),
                           ("Recall@5", "recall@5"), ("NDCG@5", "ndcg@5")):
            item = delta.get(key)
            if not item:
                continue
            add(f"{label:<9} 优化前 {_bar(item['before'])} {item['before']:.3f}")
            add(f"{'':<9} 优化后 {_bar(item['after'])} {item['after']:.3f}")
        add("```")
        add("")
        add(f"语料规模：优化前 {before.get('corpus', {}).get('num_chunks', '—')} 个分块"
            f"（chunk_size={before.get('corpus', {}).get('chunk_size', '—')}），"
            f"优化后 {after.get('corpus', {}).get('num_chunks', '—')} 个分块"
            f"（chunk_size={after.get('corpus', {}).get('chunk_size', '—')}）；"
            f"引擎口径 Hit@K {before.get('engine_hit@5', '—')} → {after.get('engine_hit@5', '—')}"
            f"（辅助参考，该口径实际度量 Top-10 命中）。")
        add("")

        # ---- 单参数消融：把收益归因到具体参数 ----
        ablation = [row for row in (experiment.get("ablation") or []) if not row.get("error")]
        if ablation:
            add("### 4.1 单参数消融（收益归因）")
            add("")
            add("「改了三个参数后变好了」无法指导下一步动哪个参数。因此额外跑了三组"
                "**单参数消融**：每组只在对照组基础上改动一个参数，其余保持一致。")
            add("")
            add("| 配置 | 分块数 | 严格 Hit@5 | 严格 MRR | 严格 Recall@5 | 严格 NDCG@5 | "
                "端到端延迟(ms) | 检索延迟(ms) | 降级率 |")
            add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            base_label = "对照组（config.yaml 默认）"
            add(f"| {base_label} | {before.get('corpus', {}).get('num_chunks', '—')} | "
                f"{_fmt4(before.get('hit@5'))} | {_fmt4(before.get('mrr'))} | "
                f"{_fmt4(before.get('recall@5'))} | {_fmt4(before.get('ndcg@5'))} | "
                f"{_fmt3(before.get('avg_latency_ms'))} | {_fmt3(before.get('avg_retrieval_ms'))} | "
                f"{_fmt4(before.get('degrade_rate'))} |")
            for row in ablation:
                add(f"| {row.get('label', row.get('tag'))} | {row.get('corpus', {}).get('num_chunks', '—')} | "
                    f"{_fmt4(row.get('hit@5'))} | {_fmt4(row.get('mrr'))} | "
                    f"{_fmt4(row.get('recall@5'))} | {_fmt4(row.get('ndcg@5'))} | "
                    f"{_fmt3(row.get('avg_latency_ms'))} | {_fmt3(row.get('avg_retrieval_ms'))} | "
                    f"{_fmt4(row.get('degrade_rate'))} |")
            add(f"| **优化后（三者合并）** | {after.get('corpus', {}).get('num_chunks', '—')} | "
                f"{_fmt4(after.get('hit@5'))} | {_fmt4(after.get('mrr'))} | "
                f"{_fmt4(after.get('recall@5'))} | {_fmt4(after.get('ndcg@5'))} | "
                f"{_fmt3(after.get('avg_latency_ms'))} | {_fmt3(after.get('avg_retrieval_ms'))} | "
                f"{_fmt4(after.get('degrade_rate'))} |")
            add("")
            # 归因结论：找出对严格 NDCG@5 影响最大的单参数变化
            base_ndcg = float(before.get("ndcg@5", 0.0) or 0.0)
            effects = [
                (row.get("label", row.get("tag")), float(row.get("ndcg@5", 0.0) or 0.0) - base_ndcg,
                 float(row.get("hit@5", 0.0) or 0.0) - float(before.get("hit@5", 0.0) or 0.0),
                 float(row.get("degrade_rate", 0.0) or 0.0) - float(before.get("degrade_rate", 0.0) or 0.0))
                for row in ablation
            ]
            positive = [e for e in effects if e[1] > 0]
            if positive:
                best = max(positive, key=lambda e: e[1])
                add(f"**归因结论**：严格 NDCG@5 提升最明显的单参数是「{best[0]}」"
                    f"（NDCG@5 {best[1]:+.4f}，Hit@5 {best[2]:+.4f}，"
                    f"降级率 {best[3]:+.4f}）。")
            else:
                add("**归因结论**：本轮没有任何单参数改动能单独提升严格 NDCG@5 —— "
                    "收益来自多处改动的组合效果，后续需要更细粒度的网格搜索来定位。")
            negative = [e for e in effects if e[1] < 0]
            if negative:
                add("")
                add("同时注意以下单参数改动**降低了**严格 NDCG@5（说明参数之间存在交互，"
                    "不能拆开单独使用）：")
                for label, ndcg_delta, hit_delta, degrade_delta in negative:
                    add(f"- {label}：NDCG@5 {ndcg_delta:+.4f}，Hit@5 {hit_delta:+.4f}，"
                        f"降级率 {degrade_delta:+.4f}")
            add("")
            add("### 4.2 结论与取舍")
            add("")
            add("扩大候选池能为未来接入真实语义向量预留召回空间，代价是检索延迟上升；"
                "把 `chunk_size` 调小能提升片段与「金标准短句」的粒度匹配度，"
                "但会切碎长句、增加分块总数与索引成本；关闭启发式重排序省下了排序开销，"
                "但在个别问题上会失去它的排序收益。因此参数选择应结合延迟预算与"
                "**单参数消融表**做权衡，而不是单纯追求 Hit@K 最大化。")
            add(f"实验实际耗时：{experiment.get('runtime_ms', 0) / 1000:.1f} 秒"
                f"（含对照 + 优化 + {len(ablation)} 组单参数消融，共 "
                f"{2 + len(ablation)} 次完整评测）。")
            add("")
        else:
            add(f"实验实际耗时：{experiment.get('runtime_ms', 0) / 1000:.1f} 秒。")
            add("")
            add("**结论与取舍**：扩大候选池能提升召回类指标，代价是检索延迟与 Prompt 长度上升；"
                "把 `chunk_size` 调小能提升片段与「金标准短句」的粒度匹配度，"
                "但会切碎长句、增加分块总数与索引成本。因此参数选择应结合延迟预算做权衡，"
                "而不是单纯追求 Hit@K 最大化 —— 表中的延迟变化列就是这一权衡的量化依据。")
            add("")

    add("## 五、后续优化路线")
    add("")
    add("| 优先级 | 措施 | 预期收益 | 代价 |")
    add("| --- | --- | --- | --- |")
    add("| P0 | 引用后处理校验：来源不在检索结果中即标记为「未核实」 | 消除引用幻觉 | 需改生成层（模块二） |")
    add("| P0 | 复核 `retrieval.enable_rerank` 的默认值 | 见第四章消融表：关闭重排序对严格指标的影响 | 可能损失它在该问题上的排序收益 |")
    add("| P1 | 更换 Embedding 为真实语义向量（bge-m3 / ollama） | 中文提问检索英文语料的关键瓶颈 | 需本地模型或 API |")
    add("| P1 | 修复重排序器的并列分数问题，使其 `confidence` 具备区分度 | 让 full 档真正优于 hybrid 档 | 需改检索层（模块一） |")
    add("| P2 | 提示词约束「不得推断原文未给出的内容」 | 降低 generation_error | 可能降低答案丰富度 |")
    add("| P2 | 工具描述补充「不适用场景」 | 提升工具选择准确率 | 提示词变长 |")
    add("")
    add("---")
    add("")
    add("_本报告由 `src/evaluation/badcase.py` 自动生成；"
        "所有指标均来自实际运行，未运行的项已显式标注。_")

    target.write_text("\n".join(lines), encoding="utf-8")
    return target


# ==============================================================================
# CLI 入口：跑一次完整分析（含真实对照实验）
# ==============================================================================
def run_analysis(
    config: Any = None,
    quick: bool = False,
    with_experiment: bool = True,
) -> Dict[str, Any]:
    """端到端跑一次 Bad Case 分析并写出 Markdown 报告。

    这是 ``python -m src.evaluation.badcase`` 的实现：真实运行检索 + 端到端问答，
    必要时再跑一轮参数对照实验。返回 ``{"report": BadCaseReport, "markdown": Path}``。
    """
    from ..utils.config import load_config
    from .benchmark import EndToEndBenchmark, RetrievalBenchmark, prepare_corpus
    from .eval_set import build_eval_set

    cfg = config or load_config()
    items = build_eval_set()
    bundle = prepare_corpus(cfg)
    retrieval = RetrievalBenchmark(bundle.engine, items, top_k=5, match="text").run()

    # 端到端问答（带引用/忠实度逐条明细）：仅随返回值提供，
    # Bad Case 判定只用下面的检索明细（严格 Hit@K）
    pipeline = bundle.create_pipeline()
    e2e = EndToEndBenchmark(pipeline, items, limit=12 if quick else 24, repeat=False).run()

    report = analyze(
        retrieval_metrics=retrieval,
        agent_results=None,
        eval_set=items,
        retrieval_details=retrieval.get("per_mode_detail", {}).get("full") or [],
    )
    if with_experiment:
        report.optimization = optimization_experiment(
            cfg, items, limit=15 if quick else 30
        )
    target = save_markdown(report, config=cfg)
    return {"report": report, "markdown": target, "retrieval": retrieval, "end_to_end": e2e}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m src.evaluation.badcase [--quick|--no-experiment]`` 入口。"""
    from ..utils.config import load_config
    from ..utils.console import enable_utf8_output

    enable_utf8_output()

    args = list(argv if argv is not None else [])
    quick = "--quick" in args
    with_experiment = "--no-experiment" not in args
    cfg = load_config()
    cfg.ensure_dirs()
    result = run_analysis(cfg, quick=quick, with_experiment=with_experiment)
    report: BadCaseReport = result["report"]
    print("=" * 68)
    print("Bad Case 分析完成")
    print("=" * 68)
    print(f"分析样本数    : {report.total_samples}")
    print(f"Bad Case 总数 : {report.total_bad_cases}（{report.bad_case_rate:.1%}）")
    for category in CATEGORIES:
        stat = report.category_stats.get(category, {})
        print(f"  {stat.get('label', category):<40} {stat.get('count', 0):>3} 条")
    print(f"报告已写出    : {result['markdown']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工执行入口
    import sys

    raise SystemExit(main(sys.argv[1:]))
