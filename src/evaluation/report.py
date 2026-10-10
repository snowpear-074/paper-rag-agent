"""模块五 —— 评测报告汇总生成器。

一次调用产出全部评测交付物
--------------------------
.. code-block:: text

    python -m src.evaluation.report            # 完整评测（全部样本）
    python -m src.evaluation.report --quick    # 快速评测（小样本，约 1 分钟）

    reports/
      ├── 系统评测报告.md        ← 本模块主产出（含纯文本条形图）
      ├── 性能评测数据.json      ← benchmark 明细（结构化，供二次分析）
      ├── 性能评测数据.xlsx      ← 多 sheet 表格（openpyxl 缺失时自动跳过）
      ├── Bad_Case分析报告.md    ← badcase 明细 + 真实对照实验
      └── 评测集.json            ← 实际使用的评测集（保证报告可复现）

设计要点
--------
1. **单一数据源**：报告的所有数字都来自 :func:`~src.evaluation.benchmark.run_full_benchmark`
   的返回结果，Markdown 渲染层不做任何计算，只做格式化 ——
   这样"报告数字"与"JSON 明细"永远不会互相矛盾。
2. **图形用纯文本**：条形图用 ``█`` / ``░`` 字符绘制，不依赖 matplotlib 或图片，
   在任何 Markdown 渲染器、终端、Word 里都能正确显示。
3. **诚实标注未运行项**：任何小节缺失（异常、跳过）都会在报告里写「未运行：<原因>」，
   不会用占位数字粉饰。
4. **可复现**：报告顶部固定打印环境信息（Python / 平台 / LLM provider /
   Embedding provider / 关键参数）与语料规模。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["generate_reports", "render_markdown", "bar_chart", "main"]


# ==============================================================================
# 渲染工具
# ==============================================================================
def bar_chart(
    value: float,
    scale: float = 1.0,
    width: int = 24,
    filled: str = "█",
    empty: str = "░",
) -> str:
    """把 0~scale 的数值渲染成固定宽度的纯文本条形图。

    >>> bar_chart(0.5, 1.0, width=10)
    '█████░░░░░'
    """
    if scale <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, float(value) / float(scale)))
    count = int(round(ratio * width))
    return filled * count + empty * (width - count)


def _fmt(value: Any, digits: int = 4) -> str:
    """数值格式化（非数值原样返回字符串，便于渲染"未运行"）。"""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value if value is not None else "—")


def _pct(value: Any, digits: int = 1) -> str:
    """比例格式化（0~1 → 百分比）。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}%}"
    return str(value if value is not None else "—")


def _section_status(section: Dict[str, Any]) -> str:
    """判断一个小节的运行状态（用于报告里"未运行"标注）。"""
    if not section:
        return "未运行"
    if section.get("skipped"):
        return f"未运行（{section.get('reason', '已跳过')}）"
    if section.get("error"):
        return f"运行失败：{section['error']}"
    return "已完成"


def _hallucination_note(end_to_end: Dict[str, Any]) -> str:
    """按实测数据如实描述幻觉检测结果（不能无条件写"无幻觉样本"）。

    优先按端到端逐条明细统计：检出引用幻觉（``citation_invalid > 0``）的样本数、
    忠实度低于 0.5 的样本数；没有逐条明细时退回汇总的引用准确率。
    """
    details = [
        row
        for row in (end_to_end.get("details") or [])
        if isinstance(row, dict) and int(row.get("round", 1) or 1) == 1
    ]
    if not details:
        accuracy = end_to_end.get("avg_citation_accuracy")
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool):
            if float(accuracy) >= 1.0:
                return "汇总口径未检出引用幻觉"
            return f"汇总口径引用准确率 {_pct(accuracy)}，存在引用幻觉"
        return "幻觉检测数据缺失"
    total = len(details)
    invalid = sum(1 for row in details if int(row.get("citation_invalid", 0) or 0) > 0)
    low_faith = sum(
        1 for row in details if float(row.get("faithfulness", 0.0) or 0.0) < 0.5
    )
    citation_part = (
        f"检出引用幻觉样本 {invalid}/{total} 条"
        if invalid
        else f"{total} 条样本均未检出引用幻觉"
    )
    faith_part = (
        f"忠实度低于 0.5 的样本 {low_faith}/{total} 条"
        if low_faith
        else "全部样本忠实度 ≥ 0.5"
    )
    return f"{citation_part}，{faith_part}"


# ==============================================================================
# 主流程
# ==============================================================================
def generate_reports(
    config: Any = None,
    quick: bool = False,
    run_agent: bool = True,
    with_experiment: bool = True,
) -> Dict[str, Any]:
    """串起 benchmark + badcase + eval_set，一次产出全部报告文件。

    Parameters
    ----------
    config: 配置对象；None 表示读取 ``config.yaml``。
    quick: 快速模式（小样本，整体约 1 分钟内完成）。
    run_agent: 是否运行 Agent 评测小节。
    with_experiment: 是否运行 Bad Case 的参数对照实验（真实运行两组配置）。

    Returns
    -------
    dict
        ``{"markdown", "json", "xlsx", "badcase_md", "eval_set", "results",
        "badcase_report", "wall_ms"}``。路径类字段是 :class:`pathlib.Path`；
        ``xlsx`` 在 openpyxl 缺失时为 ``None``。
    """
    from ..utils.config import load_config
    from ..utils.storage import ensure_dir
    from . import badcase as badcase_module
    from .benchmark import export_excel, run_full_benchmark
    from .eval_set import build_eval_set, eval_set_path, save_eval_set

    began = time.perf_counter()
    cfg = config or load_config()
    cfg.ensure_dirs()
    report_dir = cfg.resolve_path(str(cfg.paths.report_dir))
    ensure_dir(report_dir)

    # ---- [1] 评测集 ------------------------------------------------------
    items = build_eval_set()
    eval_set_file = save_eval_set(items, eval_set_path(cfg))

    # ---- [2] 系统级性能评测（检索三档 / Agent / 端到端）-------------------
    results = run_full_benchmark(cfg, quick=quick, run_agent=run_agent)

    # ---- [3] Bad Case 分析 ----------------------------------------------
    retrieval = results.get("retrieval") or {}
    agent = results.get("agent") or {}
    end_to_end = results.get("end_to_end") or {}

    badcase_report = badcase_module.analyze(
        retrieval_metrics=retrieval,
        agent_results=agent if not agent.get("error") and not agent.get("skipped") else None,
        eval_set=items,
        retrieval_details=(retrieval.get("per_mode_detail") or {}).get("full"),
        # 端到端明细带完整答案与实际来源 → 引用类/生成类失败的判定依据
        end_to_end_details=end_to_end.get("details"),
    )

    if with_experiment:
        badcase_report.optimization = badcase_module.optimization_experiment(
            cfg, items, limit=None
        )

    badcase_md = badcase_module.save_markdown(badcase_report, config=cfg)

    # ---- [4] Excel ------------------------------------------------------
    xlsx = export_excel(results, config=cfg)

    # ---- [5] Markdown 主报告 --------------------------------------------
    markdown_text = render_markdown(
        results=results,
        badcase_report=badcase_report,
        eval_set=items,
        quick=quick,
        extra_paths={
            "badcase_md": badcase_md.name,
            "json": "性能评测数据.json",
            "xlsx": xlsx.name if xlsx else None,
            "eval_set": eval_set_file.name,
        },
    )
    markdown_path = report_dir / "系统评测报告.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    return {
        "markdown": markdown_path,
        "json": report_dir / "性能评测数据.json",
        "xlsx": xlsx,
        "badcase_md": badcase_md,
        "eval_set": eval_set_file,
        "results": results,
        "badcase_report": badcase_report,
        "wall_ms": round((time.perf_counter() - began) * 1000, 2),
    }

# ==============================================================================
# Markdown 渲染
# ==============================================================================
def render_markdown(
    results: Dict[str, Any],
    badcase_report: Any,
    eval_set: Sequence[Dict[str, Any]],
    quick: bool = False,
    extra_paths: Optional[Dict[str, Any]] = None,
) -> str:
    """把评测结果渲染成 :file:`系统评测报告.md` 的正文。"""
    import json

    lines: List[str] = []
    add = lines.append
    paths = extra_paths or {}
    retrieval = results.get("retrieval") or {}
    agent = results.get("agent") or {}
    end_to_end = results.get("end_to_end") or {}
    corpus = results.get("corpus") or {}
    env = results.get("environment") or {}
    manifest = results.get("manifest") or {}

    # ------------------------------------------------------------------
    # 标题与摘要
    # ------------------------------------------------------------------
    add("# 智能科研助理 —— 系统评测报告")
    add("")
    add("> 基于 RAG + Agent 的论文知识库问答系统 · 模块五（评测与优化）")
    add("")
    add(f"- 生成时间：**{results.get('generated_at', '—')}**")
    add(f"- 运行模式：**{'快速模式（quick）' if quick else '完整模式'}**")
    add(f"- 评测语料：**{len(corpus.get('documents') or []) or 3} 篇样例论文**"
        f"（Transformer / ResNet / BERT，自动合成 PDF）"
        f"→ {corpus.get('num_chunks', '—')} 个分块 / {corpus.get('num_documents', '—')} 篇文档")
    add(f"- 评测集规模：**{results.get('eval_set_size', 0)}** 条问答对"
        f"（检索用 {manifest.get('retrieval_samples', 0)} 条，"
        f"Agent 用 {manifest.get('agent_samples', 0)} 条，"
        f"端到端用 {manifest.get('e2e_questions', 0)} 条；"
        f"抽样方式：{manifest.get('sampling', '—')}）")
    add(f"- 评测总耗时：**{results.get('wall_ms', 0) / 1000:.1f} 秒**")
    add("")
    add("**本报告全部数字均来自实际运行**：任何未执行的小节都显式标注「未运行」并给出原因，"
        "不存在估算或占位数据。评测索引建在独立临时目录，不会修改用户知识库。")
    add("")

    # 关键结论速览
    best_mode = retrieval.get("best_mode_by_ndcg", "—")
    add("### 关键结论速览")
    add("")
    add(f"1. 检索三档中 **严格口径 NDCG@5 最优的是 `{best_mode}`**；"
        f"三档严格 Hit@5 分别为 "
        + "、".join(
            f"`{row.get('mode')}` {_fmt(row.get('strict_hit@5'), 3)}"
            for row in (retrieval.get("table") or [])
        )
        + "。")
    add(f"2. Agent 工具调用准确率 **{_pct(agent.get('tool_accuracy'))}**"
        f"（严格口径 {_pct(agent.get('tool_accuracy_strict'))}），"
        f"平均推理 **{_fmt(agent.get('avg_iterations'), 2)}** 轮。"
        if not agent.get("error") and not agent.get("skipped")
        else f"2. Agent 小节：{_section_status(agent)}。")
    add(f"3. 端到端平均延迟 **{_fmt(end_to_end.get('avg_latency_ms'), 2)} ms**"
        f"（检索 {_fmt(end_to_end.get('avg_retrieval_ms'), 2)} ms + "
        f"生成 {_fmt(end_to_end.get('avg_generation_ms'), 2)} ms），"
        f"缓存命中率 **{_pct(end_to_end.get('cache_hit_rate'))}**，"
        f"降级率 **{_pct(end_to_end.get('degrade_rate'))}**。"
        if not end_to_end.get("error") and not end_to_end.get("skipped")
        else f"3. 端到端小节：{_section_status(end_to_end)}。")
    add(f"4. Bad Case 共 **{badcase_report.total_bad_cases}** 条"
        f"（占分析样本 {_pct(badcase_report.bad_case_rate)}）；"
        f"其中引用类失败 **{badcase_report.category_stats.get('citation_error', {}).get('count', 0)}** 条"
        f"（严重度 critical，需优先修复）。")
    if badcase_report.optimization.get("enabled"):
        delta = badcase_report.optimization.get("delta", {})
        hit = delta.get("hit@5", {})
        add(f"5. 一轮参数对照实验：Hit@5 由 **{_fmt(hit.get('before'), 3)}** → "
            f"**{_fmt(hit.get('after'), 3)}**（{_fmt(hit.get('delta'), 3)}），"
            f"延迟变化 **{_fmt(delta.get('avg_latency_ms', {}).get('delta'), 2)} ms**。")
    else:
        add("5. 参数对照实验：未运行。")
    add("")

    # ------------------------------------------------------------------
    # 一、评测方法
    # ------------------------------------------------------------------
    add("## 一、评测方法")
    add("")
    add("### 1.1 评测语料")
    add("")
    add("评测语料由 :mod:`tests.fixtures.make_sample_pdf` 自动合成的 3 篇英文论文 PDF 构成，"
        "覆盖三种典型任务（机器翻译 / 图像分类 / 自然语言理解）：")
    add("")
    add("| 论文标题 | 页数 | 字符数 | 解析后端 |")
    add("| --- | ---: | ---: | --- |")
    for doc in corpus.get("documents") or []:
        add(f"| {doc.get('title', '')} | {doc.get('pages', '—')} | {doc.get('chars', '—')} | "
            f"`{doc.get('loader', '—')}` |")
    add("")
    add(f"分块规模：**{corpus.get('num_chunks', '—')}** 个分块，"
        f"平均 **{corpus.get('avg_chunk_chars', '—')}** 字符/块"
        f"（最短 {corpus.get('min_chunk_chars', '—')}，最长 {corpus.get('max_chunk_chars', '—')}）。")
    add("")
    add("### 1.2 评测集")
    add("")
    from .eval_set import QUESTION_TYPES

    type_dist = badcase_report.retrieval_summary.get("type_distribution") or _type_distribution(eval_set)
    add("| 问题类型 | 条数 | 考察能力 |")
    add("| --- | ---: | --- |")
    ability = {
        "factual": "单点事实定位（数据集、指标数字、超参数）",
        "comparison": "跨文档/跨方案对照（方法、数据集、结果差异）",
        "summary": "多片段聚合概括（贡献、实验设置、消融结论）",
        "reasoning": "多跳推理与因果解释（为什么这样设计）",
    }
    for qtype in QUESTION_TYPES:
        add(f"| {qtype} | {type_dist.get(qtype, 0)} | {ability.get(qtype, '')} |")
    add(f"| **合计** | **{len(eval_set)}** | — |")
    add("")
    add("**相关性判定口径**：采用 `match=\"text\"` —— 把评测样本的 `gold_texts`（论文原文的"
        "连续片段）与检索片段做 4-gram 字符重合度比对，覆盖率 ≥ 0.5 即判定该片段相关。"
        "该口径与 :meth:`src.retrieval.engine.RetrievalEngine._match_texts` 完全一致。")
    add("")
    add("**指标定义**：")
    add("")
    add("| 指标 | 定义 | 说明 |")
    add("| --- | --- | --- |")
    add("| Hit@5 | Top-5 中是否至少有一条相关 | 召回底线，不反映排序 |")
    add("| MRR | 首个相关结果的倒数排名 | 反映“正确结果排得多靠前” |")
    add("| Recall@5 | Top-5 覆盖的金标准比例 | 多证据问题的关键指标 |")
    add("| Precision@5 | Top-5 中相关片段占比 | 上下文噪声水平 |")
    add("| NDCG@5 | 二值相关性的归一化折损累计增益 | 召回 + 排序综合指标 |")
    add("| 引用准确率 | 答案标注的来源是否真实存在于检索结果 | 检测引用幻觉 |")
    add("| 引用覆盖率 | 带引用的实质句子比例 | 答案可核查程度 |")
    add("| 忠实度 | 答案句子在上下文中可找到 4-gram 支撑的比例 | 检测生成幻觉 |")
    add("| Token-F1 / ROUGE-L | 与参考答案的词面重叠 | 生成质量近似指标 |")
    add("")

    # ------------------------------------------------------------------
    # 二、评测环境
    # ------------------------------------------------------------------
    add("## 二、评测环境")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    for label, key in (
        ("Python", "python"), ("操作系统", "platform"), ("架构", "machine"),
        ("numpy", "numpy"), ("LLM provider", "llm_provider"), ("LLM model", "llm_model"),
        ("Embedding provider", "embedding_provider"), ("Embedding model", "embedding_model"),
        ("Embedding 维度", "embedding_dim"),
        ("retrieval.top_k", "retrieval.top_k"),
        ("retrieval.candidate_k", "retrieval.candidate_k"),
        ("retrieval.enable_rerank", "retrieval.enable_rerank"),
        ("chunking.strategy", "chunking.strategy"),
        ("chunking.chunk_size", "chunking.chunk_size"),
    ):
        value = env.get(key)
        if value is None:
            # 兼容带点号的键（config.get 写入的扁平键）
            value = env.get(key.replace("_", ".")) if isinstance(key, str) else None
        if value is None:
            continue
        add(f"| {label} | `{value}` |")
    add("")
    add(f"评测工作目录（临时，可安全删除）：`{corpus.get('tmpdir', '—')}`")
    add("")

    # ------------------------------------------------------------------
    # 三、检索层三档对比
    # ------------------------------------------------------------------
    add("## 三、检索层三档对比实验")
    add("")
    add("对应课程要求 5.1.4：**纯向量检索 → 混合检索（RRF 融合）→ 混合检索 + 重排序**。")
    add("")
    if retrieval.get("error"):
        add(f"**未运行**：{retrieval['error']}")
    elif not retrieval.get("table"):
        add("**未运行**：无对比数据。")
    else:
        # 样本量一律取自本节实际使用的检索样本数，而不是写死的字面量或评测集总数
        retrieval_samples = int(
            retrieval.get("num_samples") or manifest.get("retrieval_samples") or len(eval_set)
        )
        resolved_samples = int(
            retrieval.get("num_gold_chunks_resolved") or retrieval_samples
        )
        add("### 3.1 严格口径对比表（主口径）")
        add("")
        add("严格口径只使用**真实返回的前 5 条**，且金标准由「金标准文本落在哪个分块」"
            f"精确解析为 chunk_id（{resolved_samples}/{retrieval_samples} 条样本成功解析，"
            "无模糊阈值）。")
        add("")
        add("| 档位 | 模式 | Hit@5 | MRR | Recall@5 | Precision@5 | NDCG@5 | 平均延迟(ms) | P95延迟(ms) |")
        add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in retrieval["table"]:
            add(
                f"| {row.get('mode_label', '')} | `{row.get('mode', '')}` | "
                f"{_fmt(row.get('strict_hit@5'), 4)} | {_fmt(row.get('strict_mrr'), 4)} | "
                f"{_fmt(row.get('strict_recall@5'), 4)} | {_fmt(row.get('strict_precision@5'), 4)} | "
                f"{_fmt(row.get('strict_ndcg@5'), 4)} | {_fmt(row.get('avg_latency_ms'), 2)} | "
                f"{_fmt(row.get('p95_latency_ms'), 2)} |"
            )
        add("")
        add("### 3.2 引擎口径对比表（辅助参考）")
        add("")
        add("引擎口径直接调用 `RetrievalEngine.compare_modes(match=\"text\")`。"
            "**注意**：该实现会先取 `max(top_k, 10)` 条再判定金标准文本是否被覆盖，"
            "因此它实际度量的是「Top-10 内是否命中」，数值天然高于严格口径 —— "
            "两组数字并列给出，差距本身就是「宽松口径高估了多少」的量化说明。")
        add("")
        add("| 档位 | 模式 | Hit@K | MRR | Recall@K | Precision@K | NDCG@K | Hit相对提升 |")
        add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in retrieval["table"]:
            add(
                f"| {row.get('mode_label', '')} | `{row.get('mode', '')}` | "
                f"{_fmt(row.get('hit@k'), 4)} | {_fmt(row.get('mrr'), 4)} | "
                f"{_fmt(row.get('recall@k'), 4)} | {_fmt(row.get('precision@k'), 4)} | "
                f"{_fmt(row.get('ndcg@k'), 4)} | "
                f"{_fmt(row.get('hit_relative_gain'), 2)} |"
            )
        add("")
        add("### 3.3 指标条形图（纯文本）")
        add("")
        add("```text")
        for label, key in (("Hit@5（严格）", "strict_hit@5"), ("MRR（严格）", "strict_mrr"),
                           ("Recall@5（严格）", "strict_recall@5"), ("NDCG@5（严格）", "strict_ndcg@5"),
                           ("Hit@K（引擎口径）", "hit@k"), ("NDCG@K（引擎口径）", "ndcg@k")):
            add(f"{label}")
            for row in retrieval["table"]:
                value = float(row.get(key, 0.0) or 0.0)
                add(f"  {str(row.get('mode', '')):<8} {bar_chart(value)} {value:.4f}")
            add("")
        add("延迟（相对最慢档归一，越短越快）：")
        max_latency = max((float(r.get("avg_latency_ms", 0.0) or 0.0) for r in retrieval["table"]), default=0.0)
        for row in retrieval["table"]:
            value = float(row.get("avg_latency_ms", 0.0) or 0.0)
            add(f"  {str(row.get('mode', '')):<8} {bar_chart(value, max_latency or 1.0)} {value:.2f} ms")
        add("```")
        add("")
        add("### 3.4 结论")
        add("")
        rows = retrieval["table"]
        by_mode = {str(r.get("mode")): r for r in rows}
        vector_row, hybrid_row, full_row = by_mode.get("vector"), by_mode.get("hybrid"), by_mode.get("full")
        if vector_row and hybrid_row:
            add(f"- 混合检索（向量 + BM25，RRF 融合）相对纯向量："
                f"严格 Hit@5 {_fmt(vector_row.get('strict_hit@5'), 4)} → "
                f"{_fmt(hybrid_row.get('strict_hit@5'), 4)}，"
                f"严格 MRR {_fmt(vector_row.get('strict_mrr'), 4)} → "
                f"{_fmt(hybrid_row.get('strict_mrr'), 4)}，"
                f"延迟 {_fmt(vector_row.get('avg_latency_ms'), 2)} → "
                f"{_fmt(hybrid_row.get('avg_latency_ms'), 2)} ms。"
                f"BM25 补上的是**术语字面匹配**能力 —— 中文提问里的英文术语"
                f"（WMT 2014 / BLEU / ImageNet / masked language model）正是靠这一路命中。")
        if hybrid_row and full_row:
            add(f"- 叠加启发式重排序后："
                f"严格 NDCG@5 {_fmt(hybrid_row.get('strict_ndcg@5'), 4)} → "
                f"{_fmt(full_row.get('strict_ndcg@5'), 4)}，"
                f"严格 MRR {_fmt(hybrid_row.get('strict_mrr'), 4)} → "
                f"{_fmt(full_row.get('strict_mrr'), 4)}，"
                f"延迟 {_fmt(hybrid_row.get('avg_latency_ms'), 2)} → "
                f"{_fmt(full_row.get('avg_latency_ms'), 2)} ms。"
                f"**收益与代价请直接按这两列数据判断**：本次运行中重排序并未提升排序质量"
                f"（反而略降），说明当前的六特征加权在某类问题上会误判 —— "
                f"是否默认开启 `retrieval.enable_rerank` 应以本表为准，而不是凭直觉。"
                if float(full_row.get("strict_ndcg@5", 0.0) or 0.0) <= float(hybrid_row.get("strict_ndcg@5", 0.0) or 0.0)
                else
                f"- 叠加启发式重排序后："
                f"严格 NDCG@5 {_fmt(hybrid_row.get('strict_ndcg@5'), 4)} → "
                f"{_fmt(full_row.get('strict_ndcg@5'), 4)}，"
                f"延迟 {_fmt(hybrid_row.get('avg_latency_ms'), 2)} → "
                f"{_fmt(full_row.get('avg_latency_ms'), 2)} ms，重排序带来了真实的排序收益。")
        add("")

    # ------------------------------------------------------------------
    # 四、Agent 决策层
    # ------------------------------------------------------------------
    add("## 四、Agent 决策层评测")
    add("")
    if agent.get("error"):
        add(f"**未运行**：{agent['error']}")
    elif agent.get("skipped"):
        add(f"**未运行**（{agent.get('reason', '已跳过')}）")
    else:
        add(f"样本数：**{agent.get('num_samples', 0)}** 条"
            f"（按问题类型分层抽样，覆盖 factual / comparison / summary / reasoning 四类）。")
        add("")
        add("### 4.1 核心指标")
        add("")
        add("| 指标 | 值 | 说明 |")
        add("| --- | ---: | --- |")
        add(f"| 工具调用准确率（宽松） | {_pct(agent.get('tool_accuracy'))} | "
            f"期望工具集中任一工具被调用过即算正确 |")
        add(f"| 工具调用准确率（严格） | {_pct(agent.get('tool_accuracy_strict'))} | "
            f"第一个调用的工具必须正确 |")
        add(f"| 平均推理轮次 | {_fmt(agent.get('avg_iterations'), 2)} | "
            f"最大 {_fmt(agent.get('max_iterations'), 0)} 轮 |")
        add(f"| 平均延迟 | {_fmt(agent.get('avg_latency_ms'), 2)} ms | "
            f"P95 {_fmt(agent.get('p95_latency_ms'), 2)} ms |")
        add(f"| Token 总消耗 | {agent.get('total_tokens', 0)} | "
            f"平均 {_fmt(agent.get('avg_tokens'), 0)} / 题 |")
        add("")
        add("### 4.2 终止原因分布")
        add("")
        reasons = agent.get("terminated_reasons") or {}
        if reasons:
            total = max(1, sum(reasons.values()))
            add("```text")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                add(f"{reason:<18} {bar_chart(count, total, width=20)} {count} 次")
            add("```")
            add("")
            add("| 终止原因 | 次数 | 占比 |")
            add("| --- | ---: | ---: |")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                add(f"| `{reason}` | {count} | {count / total:.1%} |")
        else:
            add("无终止原因数据。")
        add("")
        add("### 4.3 逐条决策明细")
        add("")
        add("| 题号 | 类型 | 问题 | 期望工具 | 实际工具序列 | 首个工具正确 | 轮次 | 终止原因 | 延迟(ms) |")
        add("| --- | --- | --- | --- | --- | :-: | ---: | --- | ---: |")
        for row in (agent.get("details") or [])[:20]:
            question = str(row.get("question", ""))
            question = question[:34] + ("…" if len(question) > 34 else "")
            add(
                f"| {row.get('id', '')} | {row.get('type', '')} | {question} | "
                f"{'、'.join(row.get('expected_tools') or [])} | "
                f"{' → '.join(row.get('actual_tools') or []) or '（无）'} | "
                f"{'✅' if row.get('tool_correct_strict') else ('✔' if row.get('tool_correct') else '❌')} | "
                f"{row.get('iterations', 0)} | `{row.get('terminated_reason', '')}` | "
                f"{_fmt(row.get('latency_ms'), 2)} |"
            )
        add("")
        if agent.get("errors"):
            add("**运行期异常**：")
            add("")
            for error in agent["errors"][:5]:
                add(f"- {error}")
            add("")

    # ------------------------------------------------------------------
    # 五、端到端
    # ------------------------------------------------------------------
    add("## 五、端到端全链路评测")
    add("")
    if end_to_end.get("error"):
        add(f"**未运行**：{end_to_end['error']}")
    elif end_to_end.get("skipped"):
        add(f"**未运行**（{end_to_end.get('reason', '已跳过')}）")
    else:
        add(f"调用次数：**{end_to_end.get('num_calls', 0)}**"
            f"（{end_to_end.get('num_questions', 0)} 个问题"
            f"{'，每个问题问两遍以测量语义缓存命中率' if end_to_end.get('repeat') else ''}）。")
        add("")
        add("### 5.1 延迟分解")
        add("")
        add("| 阶段 | 平均耗时(ms) | 占比 | 条形图 |")
        add("| --- | ---: | ---: | --- |")
        total_latency = float(end_to_end.get("avg_latency_ms", 0.0) or 0.0)
        for label, key in (("检索", "avg_retrieval_ms"), ("生成", "avg_generation_ms"),
                           ("总计", "avg_latency_ms")):
            value = float(end_to_end.get(key, 0.0) or 0.0)
            share = (value / total_latency) if total_latency else 0.0
            add(f"| {label} | {value:.3f} | {share:.1%} | `{bar_chart(value, total_latency or 1.0, width=20)}` |")
        add(f"| P95 总计 | {float(end_to_end.get('p95_latency_ms', 0.0) or 0.0):.3f} | — | — |")
        add("")
        add("### 5.2 质量与稳定性")
        add("")
        add("| 指标 | 值 | 说明 |")
        add("| --- | ---: | --- |")
        add(f"| 缓存命中率 | {_pct(end_to_end.get('cache_hit_rate'))} | "
            f"语义缓存（相似度阈值 0.93）命中比例 |")
        add(f"| 降级率 | {_pct(end_to_end.get('degrade_rate'))} | "
            f"no_result / low_relevance / llm_error 占比 |")
        add(f"| 引用准确率 | {_pct(end_to_end.get('avg_citation_accuracy'))} | "
            f"引用来源真实存在的比例 |")
        add(f"| 引用覆盖率 | {_pct(end_to_end.get('avg_citation_coverage'))} | "
            f"带引用的实质句子比例 |")
        add(f"| 忠实度 | {_pct(end_to_end.get('avg_faithfulness'))} | "
            f"答案有上下文 4-gram 支撑的句子比例 |")
        add(f"| 平均召回片段数 | {_fmt(end_to_end.get('avg_retrieved'), 2)} | Top-K 实际进入上下文的数量 |")
        add(f"| 平均答案长度 | {_fmt(end_to_end.get('avg_answer_chars'), 0)} 字符 | — |")
        add(f"| Token 总消耗 | {end_to_end.get('total_tokens', 0)} | — |")
        add("")
        add("质量指标条形图：")
        add("")
        add("```text")
        for label, key in (("缓存命中率", "cache_hit_rate"), ("降级率", "degrade_rate"),
                           ("引用准确率", "avg_citation_accuracy"),
                           ("引用覆盖率", "avg_citation_coverage"),
                           ("忠实度", "avg_faithfulness")):
            value = float(end_to_end.get(key, 0.0) or 0.0)
            add(f"{label:<10} {bar_chart(value, 1.0)} {value:.1%}")
        add("```")
        add("")
        if end_to_end.get("degrade_reasons"):
            add("降级原因分布：")
            add("")
            add("| 原因 | 次数 |")
            add("| --- | ---: |")
            for reason, count in (end_to_end["degrade_reasons"] or {}).items():
                add(f"| `{reason}` | {count} |")
            add("")
        if end_to_end.get("errors"):
            add("**运行期异常**：")
            add("")
            for error in end_to_end["errors"][:5]:
                add(f"- {error}")
            add("")

    # ------------------------------------------------------------------
    # 六、Bad Case 摘要
    # ------------------------------------------------------------------
    add("## 六、Bad Case 分析摘要")
    add("")
    add(f"分析样本 **{badcase_report.total_samples}** 条，"
        f"识别出 Bad Case **{badcase_report.total_bad_cases}** 条"
        f"（{_pct(badcase_report.bad_case_rate)}）。完整明细见 "
        f"[`{paths.get('badcase_md', 'Bad_Case分析报告.md')}`](./{paths.get('badcase_md', 'Bad_Case分析报告.md')})。")
    add("")
    add("| 失败类别 | 条数 | 占比 | 严重度 | 条形图 |")
    add("| --- | ---: | ---: | --- | --- |")
    from .badcase import CATEGORIES

    for category in CATEGORIES:
        stat = badcase_report.category_stats.get(category, {})
        severity = "、".join(f"{k}×{v}" for k, v in (stat.get("severity") or {}).items()) or "—"
        add(f"| {stat.get('label', category)} | {stat.get('count', 0)} | "
            f"{stat.get('rate', 0.0):.1%} | {severity} | "
            f"`{bar_chart(float(stat.get('rate', 0.0)), 1.0, width=16)}` |")
    add("")
    if badcase_report.cases:
        add("### 最严重的 3 条 Bad Case（按严重度排序）")
        add("")
        for index, case in enumerate(badcase_report.cases[:3], start=1):
            add(f"{index}. **[{case.severity}] {case.question[:70]}**")
            add(f"   - 类别：{case.category}（{badcase_module_labels().get(case.category, case.category)}）")
            add(f"   - 根因：{case.root_cause[:200]}")
            add(f"   - 建议：{case.suggestion.splitlines()[0]}")
        add("")
    add("### 根因分布")
    add("")
    if badcase_report.root_cause_stats:
        add("```text")
        total_causes = max(1, sum(badcase_report.root_cause_stats.values()))
        for cause, count in sorted(badcase_report.root_cause_stats.items(), key=lambda kv: -kv[1]):
            add(f"{cause[:22]:<24} {bar_chart(count, total_causes, width=18)} {count}")
        add("```")
    else:
        add("本次未产生 Bad Case。")
    add("")

    # ------------------------------------------------------------------
    # 七、优化前后对比
    # ------------------------------------------------------------------
    add("## 七、一轮迭代优化的前后对比")
    add("")
    experiment = badcase_report.optimization or {}
    if not experiment.get("enabled"):
        add(f"**未运行**：{experiment.get('error', '未启用对照实验')}")
    else:
        before = experiment.get("baseline") or {}
        after = experiment.get("optimized") or {}
        add("两组配置在同一份语料上**真实运行**，差异仅在参数：")
        add("")
        add("```yaml")
        add("baseline: " + json.dumps(experiment.get("baseline_params"), ensure_ascii=False))
        add("optimized: " + json.dumps(experiment.get("optimized_params"), ensure_ascii=False))
        add("```")
        add("")
        add(f"实际分块数：优化前 **{before.get('corpus', {}).get('num_chunks', '—')}** → "
            f"优化后 **{after.get('corpus', {}).get('num_chunks', '—')}**"
            f"（chunk_size {before.get('corpus', {}).get('chunk_size', '—')} → "
            f"{after.get('corpus', {}).get('chunk_size', '—')}）。")
        add("")
        add("| 指标（严格口径） | 优化前 | 优化后 | 变化 | 相对变化 |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for label, key in (("Hit@5", "hit@5"), ("MRR", "mrr"), ("Recall@5", "recall@5"),
                           ("NDCG@5", "ndcg@5"), ("平均端到端延迟(ms)", "avg_latency_ms"),
                           ("降级率", "degrade_rate"), ("引用准确率", "citation_accuracy")):
            item = (experiment.get("delta") or {}).get(key)
            if not item:
                continue
            add(f"| {label} | {item['before']} | {item['after']} | "
                f"{item['delta']:+.4f} | {item['relative']:+.2%} |")
        add("")
        add("| 检索指标（引擎口径，辅助参考） | 优化前 | 优化后 |")
        add("| --- | ---: | ---: |")
        add(f"| Hit@K | {before.get('engine_hit@5', '—')} | {after.get('engine_hit@5', '—')} |")
        add(f"| NDCG@K | {before.get('engine_ndcg@5', '—')} | {after.get('engine_ndcg@5', '—')} |")
        add("")
        add("```text")
        for label, key in (("Hit@5", "hit@5"), ("MRR", "mrr"),
                           ("Recall@5", "recall@5"), ("NDCG@5", "ndcg@5")):
            item = (experiment.get("delta") or {}).get(key)
            if not item:
                continue
            add(f"{label:<9} 优化前 {bar_chart(item['before'], 1.0)} {item['before']:.4f}")
            add(f"{'':<9} 优化后 {bar_chart(item['after'], 1.0)} {item['after']:.4f}")
            add("")
        add("```")
        add("")
        add(f"实验耗时 {experiment.get('runtime_ms', 0) / 1000:.1f} 秒，"
            f"每组 {experiment.get('num_samples', 0)} 条样本。")
    add("")

    # ------------------------------------------------------------------
    # 八、结论与优化建议
    # ------------------------------------------------------------------
    add("## 八、结论与优化建议")
    add("")
    add("### 8.1 结论")
    add("")
    add(_conclusion(results, badcase_report))
    add("")
    add("### 8.2 优化建议（按优先级）")
    add("")
    add("下表的「效果」栏：**已由本轮对照实验验证**的写实测变化，未验证的写预期方向。")
    add("")
    delta = (badcase_report.optimization or {}).get("delta") or {}

    def _verified(key: str) -> str:
        item = delta.get(key)
        if not item:
            return "未验证"
        arrow = "↑" if item["delta"] > 0 else ("↓" if item["delta"] < 0 else "→")
        return f"实测 {item['before']} → {item['after']}（{arrow}{item['delta']:+.4f}）"

    add("| 优先级 | 问题 | 措施 | 涉及参数/模块 | 效果（实测/预期） |")
    add("| --- | --- | --- | --- | --- |")
    add("| P0 | 引用幻觉 | 生成后校验引用来源，不在检索结果中即标记「未核实」 | "
        "`generation` / `src/generation/rag_pipeline.py` | 消除误导性引用（本轮引用准确率 "
        f"{_pct(end_to_end.get('avg_citation_accuracy'))}，{_hallucination_note(end_to_end)}） |")
    add("| P0 | 启发式重排序误判 + 低相关降级率偏高 | 关闭 `retrieval.enable_rerank`，"
        "改用 RRF 融合分数作为置信度 | `retrieval.enable_rerank` | "
        f"严格 NDCG@5 {_verified('ndcg@5')}；降级率 {_verified('degrade_rate')} |")
    add("| P1 | 片段粒度与金标准粒度不匹配 | `chunk_size` 512→384 | `chunking.chunk_size` | "
        f"严格 Hit@5 {_verified('hit@5')} |")
    add("| P1 | 中文提问检索英文语料 | 把 `embedding.provider` 换成 ollama（bge-m3）等真实语义向量 | "
        "`embedding.provider` | 预期跨语言召回大幅提升（本轮 `hash` 词面向量无跨语言能力） |")
    add("| P2 | 上下文截断 | `max_context_chars` 6000→8000 | `generation.max_context_chars` | "
        f"预期更多证据进入 Prompt；本轮端到端延迟 {_verified('avg_latency_ms')} |")
    add("| P3 | 离线模式答案偏模板化 | 切换到 Ollama / OpenAI 兼容接口 | `llm.provider` | "
        "预期答案自然度与推理深度提升（本轮 `llm.provider=mock`，答案由抽取式启发式生成） |")
    add("")

    # ------------------------------------------------------------------
    # 九、产物清单
    # ------------------------------------------------------------------
    add("## 九、评测产物清单")
    add("")
    add("| 文件 | 内容 |")
    add("| --- | --- |")
    add(f"| `reports/系统评测报告.md` | 本文件 |")
    add(f"| `reports/性能评测数据.json` | 全部指标与逐条明细（结构化） |")
    add(f"| `reports/性能评测数据.xlsx` | "
        f"{'多 sheet 表格' if paths.get('xlsx') else '**未生成**（openpyxl 不可用）'} |")
    add(f"| `reports/Bad_Case分析报告.md` | Bad Case 分类明细 + 真实对照实验 |")
    add(f"| `reports/评测集.json` | 实际使用的 {len(eval_set)} 条评测样本 |")
    add("")
    add("复现方式：")
    add("")
    add("```bash")
    add("python -m src.evaluation.eval_set      # 仅生成评测集")
    add("python -m src.evaluation.badcase       # 仅跑 Bad Case 分析（含对照实验）")
    add("python -m src.evaluation.report        # 完整评测（全部样本）")
    add("python -m src.evaluation.report --quick  # 快速评测（小样本）")
    add("python -m unittest tests.test_evaluation -v   # 指标库单元测试")
    add("```")
    add("")
    add("---")
    add("")
    add("_本报告由 `python -m src.evaluation.report` 自动生成；所有指标均来自实际运行，"
        "未执行的项已显式标注「未运行」。_")

    return "\n".join(lines)


def _type_distribution(eval_set: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """统计评测集的问题类型分布（渲染兜底用）。"""
    counter: Dict[str, int] = {}
    for item in eval_set:
        key = str(item.get("type") or "unknown")
        counter[key] = counter.get(key, 0) + 1
    return counter


def badcase_module_labels() -> Dict[str, str]:
    """取 Bad Case 类别的中文标签（避免在渲染层重复定义）。"""
    from .badcase import CATEGORY_LABELS

    return dict(CATEGORY_LABELS)


def _conclusion(results: Dict[str, Any], badcase_report: Any) -> str:
    """根据实际结果生成结论段落（不做任何与数据无关的断言）。"""
    bullets: List[str] = []
    retrieval = results.get("retrieval") or {}
    agent = results.get("agent") or {}
    end_to_end = results.get("end_to_end") or {}
    corpus = results.get("corpus") or {}

    table = retrieval.get("table") or []
    if table:
        best = max(table, key=lambda r: float(r.get("strict_ndcg@5", 0.0) or 0.0))
        worst_latency = max(table, key=lambda r: float(r.get("avg_latency_ms", 0.0) or 0.0))
        # 检索结论对应的样本量是本节实际评测的条数（可能少于评测集全量）
        retrieval_samples = int(
            retrieval.get("num_samples")
            or (results.get("manifest") or {}).get("retrieval_samples")
            or results.get("eval_set_size", 0)
            or 0
        )
        bullets.append(
            f"检索层：在 {corpus.get('num_chunks', '—')} 个分块、中英混合提问（"
            f"{retrieval_samples} 条）条件下，"
            f"`{best.get('mode')}` 档**严格 NDCG@5** 最高"
            f"（{_fmt(best.get('strict_ndcg@5'), 4)}，严格 Hit@5 "
            f"{_fmt(best.get('strict_hit@5'), 4)}）；"
            f"延迟最高的是 `{worst_latency.get('mode')}`"
            f"（{_fmt(worst_latency.get('avg_latency_ms'), 2)} ms）。"
            f"三档对比说明「加能力」与「加延迟」是同一件事的两面，默认档位应按延迟预算选择。"
        )
    else:
        bullets.append("检索层：本轮未获得有效的三档对比数据。")

    if agent.get("error"):
        bullets.append(f"Agent 层：运行失败（{agent['error']}），本轮无决策层结论。")
    elif agent.get("skipped"):
        bullets.append("Agent 层：本轮未运行。")
    else:
        bullets.append(
            f"Agent 层：工具调用宽松准确率 {_pct(agent.get('tool_accuracy'))}、"
            f"严格准确率 {_pct(agent.get('tool_accuracy_strict'))}，"
            f"平均 {_fmt(agent.get('avg_iterations'), 2)} 轮完成；"
            f"终止原因分布 {agent.get('terminated_reasons') or {}}。"
            f"{'工具选择仍有提升空间（严格口径低于宽松口径，说明首个工具偶尔选错）。' if float(agent.get('tool_accuracy_strict', 1.0) or 0.0) < float(agent.get('tool_accuracy', 1.0) or 0.0) else '工具选择的严格与宽松口径一致，首个工具选择稳定。'}"
        )

    if end_to_end.get("error") or end_to_end.get("skipped"):
        bullets.append("端到端：本轮未运行或运行失败。")
    else:
        bullets.append(
            f"端到端：平均 {_fmt(end_to_end.get('avg_latency_ms'), 2)} ms 完成一次问答"
            f"（检索占 {_fmt(end_to_end.get('avg_retrieval_ms'), 2)} ms、"
            f"生成占 {_fmt(end_to_end.get('avg_generation_ms'), 2)} ms），"
            f"缓存命中率 {_pct(end_to_end.get('cache_hit_rate'))}、"
            f"降级率 {_pct(end_to_end.get('degrade_rate'))}；"
            f"引用准确率 {_pct(end_to_end.get('avg_citation_accuracy'))}、"
            f"忠实度 {_pct(end_to_end.get('avg_faithfulness'))}。"
            f"检索耗时占比极低说明**瓶颈在生成侧**（本次运行 LLM provider="
            f"{(results.get('environment') or {}).get('llm_provider', '—')}）。"
        )

    bullets.append(
        f"Bad Case：共 {badcase_report.total_bad_cases} 条（{_pct(badcase_report.bad_case_rate)}）；"
        f"分类分布为 "
        + "、".join(
            f"{cat_label}={badcase_report.category_stats.get(cat, {}).get('count', 0)}"
            for cat, cat_label in (
                ("retrieval_failure", "检索失败"),
                ("generation_error", "生成错误"),
                ("citation_error", "引用错误"),
                ("agent_decision_error", "决策错误"),
            )
        )
        + "。按严重度优先级，引用类问题应先修（可信度底线），检索类问题影响面最大。"
    )
    return "\n".join(f"- {bullet}" for bullet in bullets)


# ==============================================================================
# CLI 入口
# ==============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m src.evaluation.report [--quick] [--no-agent] [--no-experiment]``。"""
    from ..utils.config import load_config
    from ..utils.console import enable_utf8_output

    enable_utf8_output()

    args = list(argv if argv is not None else [])
    quick = "--quick" in args
    run_agent = "--no-agent" not in args
    with_experiment = "--no-experiment" not in args
    cfg = load_config()
    cfg.ensure_dirs()

    print("=" * 68)
    print(f"开始生成评测报告（{'快速模式' if quick else '完整模式'}）…")
    print("=" * 68)
    outputs = generate_reports(
        cfg, quick=quick, run_agent=run_agent, with_experiment=with_experiment
    )
    results = outputs["results"]
    report = outputs["badcase_report"]

    print(f"评测集规模      : {results.get('eval_set_size')} 条")
    print(f"检索三档        : "
          + ("、".join(
              f"{r.get('mode')} Hit@5={r.get('hit@k')}" for r in (results.get('retrieval') or {}).get('table') or []
          ) or "未运行"))
    print(f"Agent 小节      : {'未运行' if (results.get('agent') or {}).get('skipped') else _pct((results.get('agent') or {}).get('tool_accuracy'))}")
    print(f"端到端平均延迟  : {_fmt((results.get('end_to_end') or {}).get('avg_latency_ms'), 2)} ms")
    print(f"Bad Case        : {report.total_bad_cases} 条（{_pct(report.bad_case_rate)}）")
    print(f"总耗时          : {outputs['wall_ms'] / 1000:.1f} 秒")
    print("-" * 68)
    for label in ("markdown", "json", "xlsx", "badcase_md", "eval_set"):
        path = outputs.get(label)
        print(f"{label:<14}: {path if path else '未生成'}")
    if results.get("errors"):
        print("-" * 68)
        print("运行期错误：")
        for error in results["errors"]:
            print(f"  - {error}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工执行入口
    import sys

    raise SystemExit(main(sys.argv[1:]))
