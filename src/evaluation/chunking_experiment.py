"""分块策略对比实验（课程要求 5.1.2 的产出物《分块策略对比实验报告》）。

为什么单独做这个实验
--------------------
分块是检索链路上**唯一一个"免费"的杠杆**：它不增加任何推理开销，却同时决定
了两件事 —— (a) 召回单元的大小，(b) 命中单元与金标准文本片段的粒度是否匹配。
粒度太粗，一个块里混进多个主题，向量被平均掉、BM25 词频被稀释；粒度太细，
一句话被切散，答案所需的上下文凑不齐。**"最优块长"没法靠直觉拍，只能实测。**

实验设计
--------
控制变量：语料（同一批样例论文）、评测集（同一批问答对）、检索配置
（``mode`` / ``top_k`` 全部固定），唯一变量是 ``chunking.strategy``。

四种策略各建一次独立索引（各自落在独立临时目录，互不污染），在同一批问题上
跑同一套严格口径指标（只用真实返回的前 ``top_k`` 条，金标准精确解析到 chunk_id），
外加建索引耗时与平均检索延迟 —— 因为"效果好但慢一倍"和"效果好且一样快"
是两个不同的结论。

指标口径与 :mod:`src.evaluation.benchmark` 完全一致（复用
:class:`~src.evaluation.benchmark.RetrievalBenchmark`），避免出现"换个报告换个
口径、数字对不上"的情况。

运行方式
--------
.. code-block:: bash

    python -m src.evaluation.chunking_experiment            # 全部策略，完整评测集
    python -m src.evaluation.chunking_experiment --quick    # 分层抽样 20 条，约 10 秒
    python -m src.evaluation.chunking_experiment --strategies fixed,recursive
"""

from __future__ import annotations

import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils.logger import get_logger
from .benchmark import RetrievalBenchmark, prepare_corpus
from .eval_set import build_eval_set, eval_set_path, load_eval_set
from .report import bar_chart

__all__ = [
    "STRATEGY_LABELS",
    "STRATEGY_NOTES",
    "run_experiment",
    "render_markdown",
    "save_markdown",
    "main",
]

_log = get_logger("evaluation.chunking")

#: 中文标签（报告里不出现裸英文策略名，便于教师直接阅读）
STRATEGY_LABELS: Dict[str, str] = {
    "fixed": "固定窗口切分",
    "recursive": "递归字符切分",
    "semantic": "语义切分",
    "paper": "论文专用切分",
}

#: 每种策略的原理与代价，写在报告里作为结果解读的依据
STRATEGY_NOTES: Dict[str, str] = {
    "fixed": "按固定字符窗口滑动切分，不识别任何语义边界。作为对照基线存在："
             "它的值代表「完全不考虑语义」时能拿到什么效果。",
    "recursive": "按 段落 → 换行 → 句末 → 从句 → 空格 的优先级递归二分，"
                 "尽量在语义边界断开；不需要 Embedding，是默认策略。",
    "semantic": "先切句、再用 Embedding 计算相邻句相似度，在相似度骤降处（话题切换）"
                "断开。块内语义一致性最高，代价是要为每个句子算一次向量。",
    "paper": "在 recursive 之上叠加学术论文专项处理：跨页段落合并、公式行保护、"
             "章节标题吸附、参考文献段剔除、跨行断词修复。",
}

#: quick 模式的样本上限（与系统评测报告的分层抽样口径保持一致）
QUICK_LIMIT = 20


# ==============================================================================
# 一、实验执行
# ==============================================================================
def _quick_subset(items: Sequence[Dict[str, Any]], limit: int = QUICK_LIMIT) -> List[Dict[str, Any]]:
    """按问题类型分层抽样：每类均匀取样，避免"只测到最简单的那一段"。

    直接复用 :func:`src.evaluation.benchmark._stratified_sample` —— 抽样口径
    若与主报告不一致，两份报告的数字就没法放在一起看。
    """
    from .benchmark import _stratified_sample

    return list(_stratified_sample(items, limit))


def _strategy_config(base: Any, strategy: str, workdir: Path) -> Any:
    """在基础配置上叠加"本次实验该策略"的覆盖项。

    索引与向量库都落在各自的临时子目录：四种策略必须各建各的索引，
    共用目录会让后一种策略读到前一种的分块，实验立刻失去意义。
    """
    return base.merged_with(
        {
            "chunking": {"strategy": strategy},
            "vector_store": {"persist_dir": str(workdir / strategy / "vector_store")},
            "paths": {"index_dir": str(workdir / strategy / "index")},
        }
    )


def run_experiment(
    config: Any = None,
    quick: bool = False,
    strategies: Sequence[str] = ("fixed", "recursive", "semantic", "paper"),
    mode: str = "hybrid",
    top_k: int = 5,
) -> Dict[str, Any]:
    """跑完整个分块策略对比实验。

    Parameters
    ----------
    config:
        基础配置；None 表示读取项目根目录 ``config.yaml``。
    quick:
        是否使用分层抽样的小样本（快，用于反复调试；正式报告请用完整评测集）。
    strategies:
        要对比的策略名序列，取值见 :data:`STRATEGIES`。
    mode:
        固定使用的检索模式（默认 ``hybrid``，即 RRF 融合）。
        这里**不固定为 full**：重排序会掩盖分块粒度的差异，主报告已显示
        启发式重排在部分问题上会误判，用它会污染本实验的结论。
    top_k:
        计算 Hit@K / Recall@K / NDCG@K 的 k，默认 5。

    Returns
    -------
    dict
        ``{"generated_at", "quick", "mode", "top_k", "num_samples",
        "rows": [...], "corpus": {...}, "wall_ms": float}``。
    """
    began = time.perf_counter()
    if config is None:
        from ..utils.config import load_config

        config = load_config()

    # 语料与策略无关：三篇样例论文只加载一次，四种策略共用同一份 LoadedDocument
    bundle = prepare_corpus(config)
    base_cfg = bundle.config

    items = _load_items(base_cfg)
    if quick:
        items = _quick_subset(items)
    if not items:
        raise RuntimeError("评测集为空：请先运行 python -m src.evaluation.eval_set 生成评测集")

    workdir = Path(tempfile.mkdtemp(prefix="rag_chunking_"))
    rows: List[Dict[str, Any]] = []
    for strategy in strategies:
        strategy = str(strategy)
        try:
            rows.append(
                _evaluate_strategy(base_cfg, strategy, workdir, bundle, items, mode, top_k)
            )
        except Exception as exc:  # noqa: BLE001 - 单个策略失败不应中断整个实验
            _log.warning("策略 %s 实验失败：%s", strategy, exc)
            rows.append(
                {
                    "strategy": strategy,
                    "label": STRATEGY_LABELS.get(strategy, strategy),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quick": bool(quick),
        "mode": str(mode),
        "top_k": int(top_k),
        "num_samples": len(items),
        "strategies": [str(s) for s in strategies],
        "corpus": bundle.corpus_stats(),
        "rows": rows,
        "workdir": str(workdir),
        "wall_ms": (time.perf_counter() - began) * 1000.0,
    }


def _load_items(config: Any) -> List[Dict[str, Any]]:
    """读取评测集；文件不存在时**现场构建**（而不是让实验直接失败）。"""
    path = eval_set_path(config)
    items = load_eval_set(path) if Path(path).exists() else []
    if not items:
        items = build_eval_set()
    return list(items)


def _evaluate_strategy(
    base_cfg: Any,
    strategy: str,
    workdir: Path,
    bundle: Any,
    items: Sequence[Dict[str, Any]],
    mode: str,
    top_k: int,
) -> Dict[str, Any]:
    """建索引 → 跑严格口径指标 → 汇总一行结果。"""
    from ..retrieval import create_retrieval_engine

    cfg = _strategy_config(base_cfg, strategy, workdir)
    cfg.ensure_dirs()

    engine = create_retrieval_engine(cfg)
    began = time.perf_counter()
    engine.index_documents(bundle.documents)
    index_ms = (time.perf_counter() - began) * 1000.0

    chunks = list(engine.chunks)
    lengths = [len(c.text) for c in chunks] or [0]

    bench = RetrievalBenchmark(engine, items, top_k=top_k, match="text")
    gold = bench.resolve_gold_chunk_ids()
    began = time.perf_counter()
    # 第一次调用才会真正检索（内部按 (mode, top_k) 记忆化），因此这里的耗时
    # 就是"这批问题在该策略下的检索总耗时"
    metrics = bench.strict_metrics(mode, gold, top_k)
    retrieval_ms = (time.perf_counter() - began) * 1000.0
    # 分母必须取严格口径下的**有效样本数**（金标准解析失败的样本不计入指标，
    # 也不该被算进延迟的分母）；键名由 strict_metric_keys 生成，不要写死。
    num_samples = max(1, int(metrics.get("strict_num_samples") or 0))

    row: Dict[str, Any] = {
        "strategy": strategy,
        "label": STRATEGY_LABELS.get(strategy, strategy),
        "num_chunks": engine.num_chunks,
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 1),
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
        "index_ms": round(index_ms, 1),
        "retrieval_ms_total": round(retrieval_ms, 1),
        "avg_latency_ms": round(retrieval_ms / num_samples, 3),
        "resolved_gold": len([k for k, v in gold.items() if v]),
        **metrics,
    }
    _log.info(
        "策略 %s：%d 块 / 平均 %.0f 字符 / 严格 Hit@%d=%.4f",
        strategy, row["num_chunks"], row["avg_chunk_chars"], top_k,
        row.get(f"strict_hit@{top_k}", 0.0),
    )
    return row


# ==============================================================================
# 二、报告渲染
# ==============================================================================
def _fmt(value: Any, digits: int = 4, dash: str = "—") -> str:
    """数值格式化：非数值原样返回字符串，便于渲染"未运行"。"""
    if value is None:
        return dash
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _best(rows: Sequence[Dict[str, Any]], key: str, *, largest: bool = True) -> Optional[Dict[str, Any]]:
    """在成功记录里挑出某指标的最优行（全部缺失或全为 0 时返回 None）。"""
    ok = [r for r in rows if not r.get("error") and isinstance(r.get(key), (int, float))]
    if not ok:
        return None
    best = (max if largest else min)(ok, key=lambda r: r[key])
    if largest and float(best[key]) <= 0.0:
        return None
    return best


def render_markdown(result: Dict[str, Any]) -> str:
    """把实验结果渲染成 Markdown（与系统评测报告同风格，全纯文本）。"""
    rows: List[Dict[str, Any]] = list(result.get("rows") or [])
    ok = [r for r in rows if not r.get("error")]
    k = int(result.get("top_k") or 5)
    hit_key, ndcg_key = f"strict_hit@{k}", f"strict_ndcg@{k}"
    corpus = result.get("corpus") or {}
    docs = corpus.get("documents") or []

    out: List[str] = []
    out.append("# 分块策略对比实验报告")
    out.append("")
    out.append("> 智能科研助理 —— 基于 RAG + Agent 的论文知识库问答系统 · 模块一（5.1.2 文本分块策略实验）")
    out.append("")
    out.append(f"- 生成时间：**{result.get('generated_at')}**")
    out.append(
        f"- 运行模式：**{'快速模式（分层抽样）' if result.get('quick') else '完整模式'}**"
        f"，样本 **{result.get('num_samples')}** 条"
    )
    out.append(
        f"- 固定变量：检索模式 `{result.get('mode')}`、`top_k={k}`、语料与评测集不变"
        f"；**唯一变量是分块策略**"
    )
    out.append("- 指标口径：严格口径（只用真实返回的前 "
               f"{k} 条，金标准精确解析到 chunk_id），与《系统评测报告》一致")
    out.append("")
    out.append("**本报告全部数字均来自实际运行**：四种策略各自在独立临时目录建立索引后真实检索，"
               "不存在估算或占位数据。")
    out.append("")

    if not ok:
        out.append("## 实验结果")
        out.append("")
        out.append("所有策略均未成功运行，无法给出对比结论。失败原因：")
        out.append("")
        for row in rows:
            out.append(f"- `{row.get('strategy')}`：{row.get('error')}")
        return "\n".join(out) + "\n"

    # -- 语料 ---------------------------------------------------------------
    out.append("## 一、实验语料")
    out.append("")
    out.append(
        f"共 **{corpus.get('num_documents')}** 篇文档，加载一次后交由 "
        f"{len(result.get('strategies') or ())} 种策略分别切分。"
    )
    out.append("")
    if docs:
        out.append("| 论文标题 | 页数 | 字符数 | 解析后端 |")
        out.append("| --- | ---: | ---: | --- |")
        for d in docs:
            out.append(f"| {d.get('title')} | {d.get('pages')} | {d.get('chars')} | `{d.get('loader')}` |")
        out.append("")

    # -- 主表 ---------------------------------------------------------------
    out.append("## 二、对比结果")
    out.append("")
    out.append(
        f"| 分块策略 | 分块数 | 平均块长(字符) | 最短/最长 | 建索引耗时(ms) "
        f"| Hit@{k} | MRR | Recall@{k} | Precision@{k} | NDCG@{k} | 平均检索延迟(ms) |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        if row.get("error"):
            out.append(
                f"| {row.get('label')}（`{row.get('strategy')}`） | 未运行 | — | — | — | — | — | — | — | — | — |"
            )
            continue
        out.append(
            "| {label}（`{strategy}`） | {num_chunks} | {avg} | {mn}/{mx} | {im} "
            "| {hit} | {mrr} | {rec} | {prec} | {ndcg} | {lat} |".format(
                label=row.get("label"),
                strategy=row.get("strategy"),
                num_chunks=row.get("num_chunks"),
                avg=_fmt(row.get("avg_chunk_chars"), 1),
                mn=row.get("min_chunk_chars"),
                mx=row.get("max_chunk_chars"),
                im=_fmt(row.get("index_ms"), 1),
                hit=_fmt(row.get(hit_key)),
                mrr=_fmt(row.get("strict_mrr")),
                rec=_fmt(row.get(f"strict_recall@{k}")),
                prec=_fmt(row.get(f"strict_precision@{k}")),
                ndcg=_fmt(row.get(ndcg_key)),
                lat=_fmt(row.get("avg_latency_ms"), 3),
            )
        )
    out.append("")
    if any(row.get("error") for row in rows):
        out.append("未成功运行的策略：")
        out.append("")
        for row in rows:
            if row.get("error"):
                out.append(f"- `{row.get('strategy')}`：{row.get('error')}")
        out.append("")
    out.append(
        f"> 金标准解析：{ok[0].get('resolved_gold')}/{result.get('num_samples')} 条样本成功解析到分块"
        f"（解析失败样本不计入指标，各策略的样本集合完全相同）。"
    )
    out.append("")

    # -- 柱状图 -------------------------------------------------------------
    out.append("## 三、指标对比图（纯文本）")
    out.append("")
    out.append("```text")
    for key, title in ((hit_key, f"Hit@{k}（严格）"), (ndcg_key, f"NDCG@{k}（严格）")):
        out.append(title)
        for row in ok:
            out.append(
                f"  {str(row.get('strategy')):<9} "
                f"{bar_chart(float(row.get(key) or 0.0), 1.0)} {_fmt(row.get(key))}"
            )
        out.append("")
    out.append("分块数（块越多说明粒度越细）")
    max_chunks = max((int(r.get("num_chunks") or 0) for r in ok), default=0) or 1
    for row in ok:
        out.append(
            f"  {str(row.get('strategy')):<9} "
            f"{bar_chart(float(row.get('num_chunks') or 0), max_chunks)} {row.get('num_chunks')}"
        )
    out.append("")
    out.append("平均检索延迟（越短越快）")
    max_lat = max((float(r.get("avg_latency_ms") or 0.0) for r in ok), default=0.0) or 1.0
    for row in ok:
        out.append(
            f"  {str(row.get('strategy')):<9} "
            f"{bar_chart(float(row.get('avg_latency_ms') or 0.0), max_lat)} "
            f"{_fmt(row.get('avg_latency_ms'), 3)} ms"
        )
    out.append("```")
    out.append("")

    # -- 结论 ---------------------------------------------------------------
    out.append("## 四、结论")
    out.append("")
    best_ndcg = _best(ok, ndcg_key)
    best_hit = _best(ok, hit_key)
    fastest = _best(ok, "avg_latency_ms", largest=False)
    finest = _best(ok, "num_chunks")
    coarsest = _best(ok, "num_chunks", largest=False)

    if best_ndcg:
        out.append(
            f"- 综合排序质量（严格 NDCG@{k}）最优：**{best_ndcg.get('label')}**"
            f"（`{best_ndcg.get('strategy')}`，{_fmt(best_ndcg.get(ndcg_key))}），"
            f"该策略共产出 {best_ndcg.get('num_chunks')} 个分块、平均 "
            f"{_fmt(best_ndcg.get('avg_chunk_chars'), 1)} 字符/块。"
        )
    if best_hit and (not best_ndcg or best_hit.get("strategy") != best_ndcg.get("strategy")):
        out.append(
            f"- 召回底线上限（严格 Hit@{k}）最高：**{best_hit.get('label')}**"
            f"（{_fmt(best_hit.get(hit_key))}）—— 与 NDCG 最优策略不同，"
            f"说明「能召回」与「排得准」在本语料上并不完全一致。"
        )
    if finest and coarsest and finest.get("strategy") != coarsest.get("strategy"):
        out.append(
            f"- 粒度跨度：最细的是 {finest.get('label')}（{finest.get('num_chunks')} 块），"
            f"最粗的是 {coarsest.get('label')}（{coarsest.get('num_chunks')} 块），"
            f"相差 {int(finest.get('num_chunks') or 0) - int(coarsest.get('num_chunks') or 0)} 块。"
        )
    if fastest:
        out.append(
            f"- 检索延迟最低：**{fastest.get('label')}**"
            f"（平均 {_fmt(fastest.get('avg_latency_ms'), 3)} ms/题）。"
        )
    out.append("")
    out.append(
        "- 以上为**本语料、本评测集**上的实测结果：三篇英文论文 + 中英混合提问。"
        "换语料（尤其是中文论文、长文档）时结论可能改变，因此报告给出了完整参数与"
        "一键复现命令，便于换数据集重跑后再下结论。"
    )
    out.append("")

    # -- 策略说明 -----------------------------------------------------------
    out.append("## 五、各策略原理与代价")
    out.append("")
    out.append("| 策略 | 原理与代价 |")
    out.append("| --- | --- |")
    for row in rows:
        name = row.get("strategy")
        out.append(f"| {row.get('label')}（`{name}`） | {STRATEGY_NOTES.get(str(name), '—')} |")
    out.append("")

    out.append("## 六、复现方式")
    out.append("")
    out.append("```bash")
    out.append("python -m src.evaluation.chunking_experiment             # 完整评测集")
    out.append("python -m src.evaluation.chunking_experiment --quick     # 分层抽样 20 条")
    out.append("python -m src.evaluation.chunking_experiment --strategies fixed,recursive")
    out.append("```")
    out.append("")
    out.append(
        f"本次实验总耗时 **{result.get('wall_ms', 0) / 1000:.1f} 秒**"
        f"（含 {len(ok)} 种策略各自的建索引时间），"
        f"临时工作目录：`{result.get('workdir')}`。"
    )
    out.append("")
    out.append("---")
    out.append("")
    out.append("_本报告由 `python -m src.evaluation.chunking_experiment` 自动生成；"
               "所有指标均来自实际运行，未执行的项已显式标注「未运行」。_")
    out.append("")
    return "\n".join(out)


def save_markdown(text: str, path: Optional[str] = None) -> str:
    """把报告写到磁盘，返回实际路径（默认 ``reports/分块策略对比实验报告.md``）。"""
    if path is None:
        from ..utils.config import get_config

        report_dir = Path(get_config().get("paths.report_dir", "reports"))
        target = report_dir / "分块策略对比实验报告.md"
    else:
        target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target)


# ==============================================================================
# 三、命令行入口
# ==============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m src.evaluation.chunking_experiment [--quick] [--strategies ...]``。"""
    from ..utils.config import load_config
    from ..utils.console import enable_utf8_output

    enable_utf8_output()

    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.chunking_experiment",
        description="分块策略对比实验：四种策略各建索引，跑同一套严格口径指标。",
    )
    parser.add_argument("--quick", action="store_true", help=f"分层抽样 {QUICK_LIMIT} 条，快速跑完")
    parser.add_argument("--strategies", default="fixed,recursive,semantic,paper",
                       help="要对比的策略，逗号分隔")
    parser.add_argument("--mode", default="hybrid", help="固定使用的检索模式（默认 hybrid）")
    parser.add_argument("--top-k", type=int, default=5, help="计算 Hit@K 等指标的 k（默认 5）")
    parser.add_argument("--out", default=None, help="报告输出路径（默认 reports/分块策略对比实验报告.md）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    strategies = [s.strip() for s in str(args.strategies).split(",") if s.strip()]
    cfg = load_config()
    cfg.ensure_dirs()

    print("=" * 68)
    print(f"分块策略对比实验（{'快速模式' if args.quick else '完整模式'}，"
          f"检索模式={args.mode}，策略={'、'.join(strategies)}）…")
    print("=" * 68)

    result = run_experiment(
        config=cfg, quick=args.quick, strategies=strategies, mode=args.mode, top_k=args.top_k
    )
    path = save_markdown(render_markdown(result), args.out)

    print(f"样本数          : {result['num_samples']} 条")
    for row in result["rows"]:
        if row.get("error"):
            print(f"  {row.get('strategy'):<10}: 未运行（{row.get('error')}）")
            continue
        print(
            f"  {row.get('strategy'):<10}: {row.get('num_chunks'):>3} 块  "
            f"平均 {_fmt(row.get('avg_chunk_chars'), 0):>5} 字符  "
            f"Hit@{args.top_k}={_fmt(row.get('strict_hit@%d' % args.top_k))}  "
            f"NDCG@{args.top_k}={_fmt(row.get('strict_ndcg@%d' % args.top_k))}  "
            f"{_fmt(row.get('avg_latency_ms'), 3)} ms/题"
        )
    print(f"总耗时          : {result['wall_ms'] / 1000:.1f} 秒")
    print("-" * 68)
    print(f"报告            : {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工执行入口
    import sys

    raise SystemExit(main(sys.argv[1:]))
