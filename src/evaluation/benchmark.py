"""模块五 —— 系统级性能评测（课程要求 5.5.2）。

三档评测对应课程要求的三类指标
-----------------------------
1. :class:`RetrievalBenchmark` —— **检索层**：纯向量 → 混合检索（RRF）→
   混合 + 重排序，三档对比表（Hit@5 / MRR / Recall@5 / NDCG@5 / 平均延迟 / 相对提升）。
   引擎口径直接复用 :meth:`src.retrieval.engine.RetrievalEngine.evaluate`
   （``compare_modes`` 内部即逐档调用它），保证"评测口径"与线上检索完全一致
   （同一份代码、同一套参数）；严格口径使用本模块自己的实现。
2. :class:`AgentBenchmark` —— **决策层**：工具调用准确率、平均推理轮次、
   终止原因分布、平均延迟、Token 消耗。为控制耗时，默认只跑评测集的前 15 条。
3. :class:`EndToEndBenchmark` —— **全链路**：RAG pipeline 的延迟分解
   （检索 ms / 生成 ms / 总计）、缓存命中率、降级率。

关键设计
--------
- **绝不污染用户数据**：评测索引一律建在临时目录（``tempfile.mkdtemp``），
  向量库、索引清单、对话记忆都指向临时路径。评测跑完后用户的知识库毫无变化。
- **真实数据原则**：所有指标均来自**实际执行**。任何一步失败（依赖缺失、
  LLM 不可用）都会在结果里写入 ``error`` 字段并把该小节标记为"未运行"，
  而不是填一个编造的数字。
- **优雅降级**：``openpyxl`` 缺失时 :func:`export_excel` 只打印提示并返回
  ``None``，不抛异常 —— 评测报告不能因为一个可选的导出依赖而整体失败。

用法
----
>>> from src.evaluation.benchmark import run_full_benchmark
>>> result = run_full_benchmark(quick=True)          # 快速模式（约 1 分钟）
>>> result["retrieval"]["table"][0]["mode"]
'vector'
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import aggregate

__all__ = [
    "CorpusBundle",
    "prepare_corpus",
    "RetrievalBenchmark",
    "AgentBenchmark",
    "EndToEndBenchmark",
    "run_full_benchmark",
    "export_excel",
    "write_performance_json",
    "strict_metric_keys",
]

#: 三档检索模式（与 config.yaml / RetrievalEngine.retrieve 的 mode 取值一致）
RETRIEVAL_MODES = ("vector", "hybrid", "full")

#: 档位中文名（报告渲染用）
MODE_LABELS = {
    "vector": "纯向量检索",
    "hybrid": "混合检索（向量+BM25 RRF 融合）",
    "full": "混合检索 + 启发式重排序",
}


def strict_metric_keys(k: int) -> Dict[str, str]:
    """严格口径的指标名（由 ``top_k`` 生成，而不是把 ``@5`` 写死在键名里）。

    ``k`` 变化时键名随之变化（``strict_hit@10`` / ``strict_recall@10`` …），
    避免"改了 top_k 却仍输出 @5"的错位。
    """
    k = int(k)
    return {
        "hit": f"strict_hit@{k}",
        "mrr": "strict_mrr",
        "recall": f"strict_recall@{k}",
        "precision": f"strict_precision@{k}",
        "ndcg": f"strict_ndcg@{k}",
        "num_samples": "strict_num_samples",
    }


# ==============================================================================
# 语料准备
# ==============================================================================
class CorpusBundle:
    """一次评测所用的"临时知识库"：样例 PDF + 已索引的检索/生成/Agent 组件。

    Attributes
    ----------
    tmpdir: 临时根目录（评测结束可整体删除，绝不影响 ``data/``）。
    pdf_paths: 生成的样例论文 PDF 路径。
    documents: 加载后的 :class:`~src.utils.types.LoadedDocument` 列表。
    engine: 已索引好的 :class:`~src.retrieval.engine.RetrievalEngine`。
    config: 覆盖了临时路径的配置对象。
    """

    def __init__(
        self,
        tmpdir: Path,
        pdf_paths: List[Path],
        documents: List[Any],
        engine: Any,
        config: Any,
        index_report: Dict[str, Any],
    ) -> None:
        self.tmpdir = tmpdir
        self.pdf_paths = pdf_paths
        self.documents = documents
        self.engine = engine
        self.config = config
        self.index_report = index_report

    # -- 便捷构造 ---------------------------------------------------------
    def create_pipeline(self, **kwargs: Any) -> Any:
        """基于本次评测的检索引擎创建 RAG 流水线。"""
        from ..generation.rag_pipeline import create_rag_pipeline

        return create_rag_pipeline(config=self.config, retrieval_engine=self.engine, **kwargs)

    def create_agent(self, **kwargs: Any) -> Any:
        """基于本次评测的检索引擎创建 ReAct Agent（记忆隔离到临时目录）。"""
        from ..agent.react_loop import create_agent

        return create_agent(config=self.config, retrieval_engine=self.engine, **kwargs)

    def corpus_stats(self) -> Dict[str, Any]:
        """语料统计（写入报告"评测环境"小节）。"""
        chunks = list(self.engine.chunks)
        lengths = [len(c.text) for c in chunks] or [0]
        return {
            "num_documents": self.engine.num_documents,
            "num_chunks": self.engine.num_chunks,
            "avg_chunk_chars": round(sum(lengths) / len(lengths), 1),
            "min_chunk_chars": min(lengths),
            "max_chunk_chars": max(lengths),
            "documents": [
                {
                    "title": d.meta.title,
                    "pages": d.meta.num_pages,
                    "chars": len(d.text),
                    "loader": d.loader,
                }
                for d in self.documents
            ],
            "engine_stats": self.engine.stats(),
        }


def prepare_corpus(config: Any = None, keep_dir: bool = False) -> CorpusBundle:
    """生成样例论文 PDF 并在**临时目录**建立完整检索索引。

    Parameters
    ----------
    config:
        基础配置；None 表示读取项目根目录 ``config.yaml``。
    keep_dir:
        是否保留临时目录（调试用）。False 时目录仍在，只是不做额外清理 ——
        实际清理交由调用方（评测脚本通常希望保留产出以便复现）。

    Returns
    -------
    CorpusBundle

    Notes
    -----
    这里的配置覆盖是评测隔离的核心：``vector_store.persist_dir`` /
    ``paths.index_dir`` / ``generation.cache.path`` / ``memory.path``
    全部指向临时目录，因此反复运行评测不会互相污染，也不会动到用户知识库。
    """
    from ..data_loader import loader_from_config
    from ..retrieval import create_retrieval_engine
    from ..utils.config import load_config
    from tests.fixtures.make_sample_pdf import write_sample_pdfs

    tmpdir = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    overrides: Dict[str, Any] = {
        "vector_store": {"persist_dir": str(tmpdir / "vector_store")},
        "paths": {"index_dir": str(tmpdir / "index"), "report_dir": str(tmpdir / "reports")},
        "generation": {"cache": {"path": str(tmpdir / "cache" / "semantic_cache.db")}},
        "memory": {"path": str(tmpdir / "memory" / "chat.db")},
    }
    cfg = (config or load_config()).merged_with(overrides)
    cfg.ensure_dirs()

    pdf_paths = write_sample_pdfs(tmpdir / "papers")
    loader = loader_from_config(cfg)
    documents = [loader.load(str(path)) for path in pdf_paths]

    engine = create_retrieval_engine(cfg)
    index_report = engine.index_documents(documents)
    return CorpusBundle(tmpdir, list(pdf_paths), documents, engine, cfg, index_report)


# ==============================================================================
# 一、检索三档对比
# ==============================================================================
class RetrievalBenchmark:
    """检索层三档对比评测。

    Parameters
    ----------
    engine:
        :class:`~src.retrieval.engine.RetrievalEngine`（已索引）。
    eval_set:
        评测样本列表；使用 ``match="text"`` 时每条需要 ``gold_texts``。
    top_k:
        计算 Hit@K / Recall@K 的 k（默认 5，即课程要求的 Hit@5）。
    match:
        相关性判定方式，``text``（按金标准文本片段重合）或 ``chunk_id`` / ``doc_id``。
    """

    def __init__(
        self,
        engine: Any,
        eval_set: Sequence[Dict[str, Any]],
        top_k: int = 5,
        match: str = "text",
    ) -> None:
        self.engine = engine
        self.eval_set = list(eval_set or ())
        self.top_k = int(top_k)
        self.match = str(match)
        #: 按 ``(mode, top_k)`` 记忆化的检索结果（题目 key → chunk_id 排序），
        #: 保证同一档位、同一 k 只真实检索一遍（严格指标与逐条明细共用）。
        self._rank_cache: Dict[Tuple[str, int], Dict[str, List[str]]] = {}

    def run(self, modes: Sequence[str] = RETRIEVAL_MODES) -> Dict[str, Any]:
        """执行三档对比，返回结构化结果（含对比表与相对提升）。

        指标分两个口径
        --------------
        - **引擎口径**（``hit@k`` / ``mrr`` / …）：直接复用
          :meth:`src.retrieval.engine.RetrievalEngine.evaluate`
          （``compare_modes`` 内部即逐档调用它）。注意 ``match="text"`` 时引擎内部
          会先取 ``max(top_k, 10)`` 条再判定金标准文本是否被覆盖，
          因此该口径实际度量的是"Top-10 内是否命中"，比 Top-5 宽松。
        - **严格口径**（``strict_hit@k`` / ``strict_mrr`` / …）：本模块自行实现，
          只使用**真实返回的前 top_k 条**，且金标准由"金标准文本落在哪个 chunk"
          精确解析为 chunk_id（评测集的 gold_texts 均保证唯一归属）。
          该口径没有重合度阈值与候选池放大的模糊空间，是排序质量的硬指标。

        两个口径一并给出，是为了让读者看清"宽松口径"与"硬指标"的差距，
        而不是只报一个好看的数。每个档位只真实检索两遍：引擎口径一遍
        （``max(top_k, 10)`` 条），严格口径一遍（``top_k`` 条，按 ``(mode, top_k)``
        记忆化后被指标与逐条明细共用）。

        Returns
        -------
        dict
            ``{"top_k", "num_samples", "match", "table": [...], "per_mode_detail": {...}}``。
        """
        began = time.perf_counter()
        from .metrics import hit_at_k as _hit_at_k

        keys = strict_metric_keys(self.top_k)
        # 金标准 chunk_id 只解析一次：严格指标与逐条明细共用同一份映射
        gold_ids = self.resolve_gold_chunk_ids()
        table: List[Dict[str, Any]] = []
        per_mode_detail: Dict[str, List[Dict[str, Any]]] = {}
        for mode in modes:
            mode = str(mode)
            # 引擎口径：本档位只 evaluate 一次，指标与逐条明细共用同一次检索结果
            metrics = self.engine.evaluate(
                self.eval_set, mode=mode, top_k=self.top_k, match=self.match
            )
            row: Dict[str, Any] = {
                "mode": mode,
                "hit@k": metrics.get("hit_at_k", 0.0),
                "mrr": metrics.get("mrr", 0.0),
                "recall@k": metrics.get("recall_at_k", 0.0),
                "precision@k": metrics.get("precision_at_k", 0.0),
                "ndcg@k": metrics.get("ndcg_at_k", 0.0),
                "avg_latency_ms": metrics.get("avg_latency_ms", 0.0),
                "p95_latency_ms": metrics.get("p95_latency_ms", 0.0),
            }
            row["mode_label"] = MODE_LABELS.get(mode, mode)
            # 统一别名为 latency_p95_ms（与 p95_latency_ms 同值）
            row["latency_p95_ms"] = row.get("p95_latency_ms", 0.0)
            # 严格口径：复用按 (mode, top_k) 记忆化的那次检索
            row.update(self.strict_metrics(mode, gold_ids))
            table.append(row)

            # 逐条明细（用于 Bad Case 分析与 Excel 的"逐条明细"表）
            ranked_by_key = self._ranked_ids(mode, self.top_k)
            rows = [dict(r) for r in (metrics.get("details") or [])]
            for detail in rows:
                sample = next(
                    (
                        s
                        for s in self.eval_set
                        if str(s.get("question") or "") == str(detail.get("question") or "")
                    ),
                    None,
                )
                key = str((sample or {}).get("id") or detail.get("question") or "")
                gold_for_sample = gold_ids.get(key) or []
                if gold_for_sample:
                    # 用真实返回的 top_k 条重算 —— 避免引擎内部 max(top_k,10) 的宽松判定，
                    # 且与严格指标表共用同一份记忆化排序
                    top = list(ranked_by_key.get(key) or [])[: self.top_k]
                    detail["strict_hit"] = bool(_hit_at_k(top, gold_for_sample, self.top_k))
                    detail["gold_chunk_ids"] = gold_for_sample
                detail["strict"] = bool(detail.get("strict_hit", detail.get("hit", False)))
            per_mode_detail[mode] = rows

        baseline = next((r for r in table if r.get("mode") == "vector"), table[0] if table else {})
        for row in table:
            # 增益列与 RetrievalEngine.compare_modes 保持同名（兼容既有消费方）
            row["hit_gain_vs_vector"] = round(
                float(row.get("hit@k", 0.0) or 0.0) - float(baseline.get("hit@k", 0.0) or 0.0), 4
            )
            row["mrr_gain_vs_vector"] = round(
                float(row.get("mrr", 0.0) or 0.0) - float(baseline.get("mrr", 0.0) or 0.0), 4
            )
            if baseline.get("avg_latency_ms"):
                row["latency_ratio"] = round(
                    float(row.get("avg_latency_ms", 0.0) or 0.0)
                    / float(baseline["avg_latency_ms"]),
                    2,
                )
            row["hit_relative_gain"] = _relative_gain(row.get("hit@k", 0.0), baseline.get("hit@k", 0.0))
            row["mrr_relative_gain"] = _relative_gain(row.get("mrr", 0.0), baseline.get("mrr", 0.0))
            row["ndcg_relative_gain"] = _relative_gain(row.get("ndcg@k", 0.0), baseline.get("ndcg@k", 0.0))
            row["strict_hit_relative_gain"] = _relative_gain(
                row.get(keys["hit"], 0.0), baseline.get(keys["hit"], 0.0)
            )

        best_row = max(table, key=lambda r: float(r.get(keys["ndcg"], 0.0) or 0.0)) if table else {}
        # 汇总口径（供 Bad Case 报告直接渲染），以严格口径为主
        summary: Dict[str, Any] = {
            "mode": str(best_row.get("mode", "")) or "full",
            "top_k": self.top_k,
            "match": self.match,
            "num_samples": len(self.eval_set),
            "hit_at_k": best_row.get(keys["hit"], 0.0),
            "mrr": best_row.get(keys["mrr"], 0.0),
            "recall_at_k": best_row.get(keys["recall"], 0.0),
            "precision_at_k": best_row.get(keys["precision"], 0.0),
            "ndcg_at_k": best_row.get(keys["ndcg"], 0.0),
            "avg_latency_ms": best_row.get("avg_latency_ms", 0.0),
            "p95_latency_ms": best_row.get("p95_latency_ms", 0.0),
            "strict": True,
        }
        vector_row = next((r for r in table if r.get("mode") == "vector"), {})
        summary["engine_hit_at_k"] = vector_row.get("hit@k", 0.0)
        return {
            "top_k": self.top_k,
            "num_samples": len(self.eval_set),
            "match": self.match,
            "table": table,
            "per_mode_detail": per_mode_detail,
            "summary": summary,
            "num_gold_chunks_resolved": sum(1 for ids in gold_ids.values() if ids),
            "latency_ms": round((time.perf_counter() - began) * 1000, 2),
            "best_mode_by_ndcg": best_row.get("mode", ""),
            "error": "",
        }

    # -- 严格口径 ---------------------------------------------------------
    def _ranked_ids(self, mode: str, top_k: int) -> Dict[str, List[str]]:
        """按 ``(mode, top_k)`` 记忆化检索结果（题目 key → chunk_id 排序）。

        同一档位、同一 k 只真实检索一遍；严格指标与逐条明细的重复调用直接复用，
        避免每个档位被检索 2~3 遍。
        """
        cache_key = (str(mode), int(top_k))
        cached = self._rank_cache.get(cache_key)
        if cached is None:
            cached = {}
            for sample in self.eval_set:
                question = str(sample.get("question") or "")
                if not question:
                    continue
                key = str(sample.get("id") or question)
                cached[key] = [
                    hit.chunk.chunk_id
                    for hit in self.engine.retrieve(question, top_k=int(top_k), mode=str(mode))
                ]
            self._rank_cache[cache_key] = cached
        return cached

    def resolve_gold_chunk_ids(self) -> Dict[str, List[str]]:
        """把每条样本的 ``gold_texts`` 解析成语料中的 chunk_id 列表。

        判定方式：金标准文本**逐字出现**在某个 chunk 的正文中即视为该 chunk 相关。
        评测集的 ``gold_texts`` 均取自样例论文原文且保证唯一归属，
        因此这个映射是确定性的 —— 没有重合度阈值、没有候选池放大的模糊空间。
        """
        chunks = list(self.engine.chunks)
        resolved: Dict[str, List[str]] = {}
        for sample in self.eval_set:
            key = str(sample.get("id") or sample.get("question") or "")
            ids: List[str] = []
            targets = [str(t) for t in (sample.get("gold_texts") or []) if str(t).strip()]
            for text in targets:
                for chunk in chunks:
                    if text in chunk.text and chunk.chunk_id not in ids:
                        ids.append(chunk.chunk_id)
            resolved[key] = ids
        return resolved

    def strict_metrics(
        self,
        mode: str,
        gold_ids: Optional[Dict[str, List[str]]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """只用**真实返回的前 k 条**计算 Hit/MRR/Recall/Precision/NDCG。

        同一 ``(mode, k)`` 的检索结果会被记忆化（见 :meth:`_ranked_ids`）：
        重复调用不会重复检索，指标键名也随 k 变化（``strict_hit@`` + k）。
        """
        from .metrics import hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

        top_k = int(k or self.top_k)
        gold = gold_ids if gold_ids is not None else self.resolve_gold_chunk_ids()
        ranked_by_key = self._ranked_ids(mode, top_k)
        hits: List[float] = []
        mrrs: List[float] = []
        recalls: List[float] = []
        precisions: List[float] = []
        ndcgs: List[float] = []
        for sample in self.eval_set:
            key = str(sample.get("id") or sample.get("question") or "")
            gold_for_sample = gold.get(key) or []
            if not gold_for_sample:
                continue
            ranked = ranked_by_key.get(key) or []
            hits.append(hit_at_k(ranked, gold_for_sample, top_k))
            mrrs.append(mrr(ranked, gold_for_sample))
            recalls.append(recall_at_k(ranked, gold_for_sample, top_k))
            precisions.append(precision_at_k(ranked, gold_for_sample, top_k))
            ndcgs.append(ndcg_at_k(ranked, gold_for_sample, top_k))
        keys = strict_metric_keys(top_k)
        count = max(1, len(hits))
        return {
            keys["hit"]: round(sum(hits) / count, 4) if hits else 0.0,
            keys["mrr"]: round(sum(mrrs) / count, 4) if mrrs else 0.0,
            keys["recall"]: round(sum(recalls) / count, 4) if recalls else 0.0,
            keys["precision"]: round(sum(precisions) / count, 4) if precisions else 0.0,
            keys["ndcg"]: round(sum(ndcgs) / count, 4) if ndcgs else 0.0,
            keys["num_samples"]: len(hits),
        }


def _relative_gain(value: float, baseline: float) -> float:
    """相对提升（baseline 为 0 时返回 0.0，避免除零和"无穷大提升"的假象）。"""
    if not baseline:
        return 0.0
    return round((float(value) - float(baseline)) / float(baseline), 4)


def _stratified_sample(
    items: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """按问题类型分层抽样：每类均匀取样，避免"只测到最简单的那一段"。

    评测集按 factual → comparison → summary → reasoning 分段排列，
    而前段（factual/easy）基线接近满分。直接取前 N 条会让评测结果虚高，
    因此这里按类型分组后轮询取样，并保持原有顺序。
    """
    rows = list(items or ())
    if limit <= 0 or len(rows) <= limit:
        return rows
    from collections import OrderedDict

    buckets: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for item in rows:
        buckets.setdefault(str(item.get("type") or "unknown"), []).append(item)

    picked: List[Dict[str, Any]] = []
    cursors = {key: 0 for key in buckets}
    while len(picked) < limit:
        progressed = False
        for key, bucket in buckets.items():
            if len(picked) >= limit:
                break
            index = cursors[key]
            if index < len(bucket):
                picked.append(bucket[index])
                cursors[key] = index + 1
                progressed = True
        if not progressed:
            break
    # 保持评测集原有顺序，便于人工对照
    order = {id(item): position for position, item in enumerate(rows)}
    picked.sort(key=lambda item: order.get(id(item), 0))
    return picked


def _sources_of(retrieved: Sequence[Any], preview_chars: int = 400) -> List[Dict[str, Any]]:
    """从检索结果中提取来源信息（来源名 / 页码 / 正文预览）。

    预览正文会随评测明细一起落盘，用于 Bad Case 分析复核"引用是否真实存在"
    与"答案是否真有上下文依据"；截断到 400 字符以控制报告体积。
    """
    rows: List[Dict[str, Any]] = []
    for item in retrieved or ():
        chunk = getattr(item, "chunk", None)
        if chunk is not None:
            rows.append(
                {
                    "source": str(getattr(chunk, "source", "") or ""),
                    "page": int(getattr(chunk, "page", 0) or 0),
                    "preview": str(getattr(chunk, "text", "") or "")[:preview_chars],
                }
            )
        elif isinstance(item, dict):
            rows.append(
                {
                    "source": str(item.get("source") or ""),
                    "page": int(item.get("page", 0) or 0),
                    "preview": str(item.get("preview") or item.get("text") or "")[:preview_chars],
                }
            )
        else:
            rows.append(
                {
                    "source": str(getattr(item, "source", "") or ""),
                    "page": int(getattr(item, "page", 0) or 0),
                    "preview": str(getattr(item, "text", "") or "")[:preview_chars],
                }
            )
    return rows


# ==============================================================================
# 二、Agent 决策层评测
# ==============================================================================
#: 当评测样本缺少 ``expected_tools`` 时的启发式推断表
_HEURISTIC_TOOLS = {
    "comparison": ["paper_compare", "knowledge_search"],
    "summary": ["summarize_paper", "knowledge_search"],
    "reasoning": ["knowledge_search", "paper_compare"],
    "factual": ["knowledge_search", "paper_metadata"],
}


def infer_expected_tools(sample: Dict[str, Any]) -> List[str]:
    """推断某条样本"期望调用的工具"。

    优先使用样本自带的 ``expected_tools`` 字段；缺失时按问题类型启发式推断
    （comparison → paper_compare；summary → summarize_paper；其余 → knowledge_search），
    并把多文档问题额外视作可能需要 ``paper_compare``。

    启发式只是**兜底**：它会带来评测噪声（真实模型可能用别的合法工具得到同样好的答案），
    因此结果里会标注 ``expected_tools_source``，报告需如实说明口径。
    """
    explicit = sample.get("expected_tools")
    if isinstance(explicit, (list, tuple)) and explicit:
        return [str(t) for t in explicit]
    qtype = str(sample.get("type") or "factual")
    tools = list(_HEURISTIC_TOOLS.get(qtype, ["knowledge_search"]))
    if len(sample.get("gold_doc_titles") or []) > 1 and "paper_compare" not in tools:
        tools.insert(0, "paper_compare")
    return tools


class AgentBenchmark:
    """Agent 决策层评测：工具选择 / 推理轮次 / 终止原因 / 延迟 / Token。

    Parameters
    ----------
    agent:
        :class:`~src.agent.react_loop.ReActAgent`。
    eval_set:
        评测样本列表。
    limit:
        最多评测多少条（默认 15，用于控制耗时）。
    """

    def __init__(self, agent: Any, eval_set: Sequence[Dict[str, Any]], limit: int = 15) -> None:
        self.agent = agent
        self.eval_set = list(eval_set or ())
        self.limit = max(1, int(limit))

    @staticmethod
    def _first_tool(result: Any) -> str:
        """取 Agent 第一次实际调用的工具名（无调用则返回空字符串）。"""
        from ..utils.types import StepType

        for step in getattr(result, "steps", []) or []:
            if step.type == StepType.ACTION and step.tool_name:
                return str(step.tool_name)
        return ""

    @staticmethod
    def _called_tools(result: Any) -> List[str]:
        """取 Agent 实际调用过的全部工具名（保序去重）。"""
        from ..utils.types import StepType

        tools: List[str] = []
        for step in getattr(result, "steps", []) or []:
            if step.type == StepType.ACTION and step.tool_name and step.tool_name not in tools:
                tools.append(str(step.tool_name))
        return tools

    def run(self, session_prefix: str = "eval") -> Dict[str, Any]:
        """逐条运行 Agent 并汇总指标。

        Returns
        -------
        dict
            ``{"num_samples", "tool_accuracy", "tool_accuracy_strict",
            "avg_iterations", "terminated_reasons", "avg_latency_ms",
            "total_tokens", "details": [...]}``。
        """
        began = time.perf_counter()
        details: List[Dict[str, Any]] = []
        samples = self.eval_set[: self.limit]
        terminated: Counter = Counter()
        errors: List[str] = []

        for index, sample in enumerate(samples, start=1):
            question = str(sample.get("question") or "")
            if not question:
                continue
            expected = infer_expected_tools(sample)
            expected_source = "explicit" if sample.get("expected_tools") else "heuristic"
            started = time.perf_counter()
            try:
                result = self.agent.run(question, session_id=f"{session_prefix}_{index}")
                error = ""
            except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整轮评测
                errors.append(f"[{sample.get('id')}] {type(exc).__name__}: {exc}")
                details.append(
                    {
                        "id": sample.get("id", ""),
                        "type": sample.get("type", ""),
                        "question": question,
                        "expected_tools": expected,
                        "expected_tools_source": expected_source,
                        "actual_tools": [],
                        "first_tool": "",
                        "tool_correct": False,
                        "num_tools": 0,
                        "iterations": 0,
                        "terminated_reason": "exception",
                        "latency_ms": 0.0,
                        "total_tokens": 0,
                        "answer_chars": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                terminated["exception"] += 1
                continue

            actual_tools = self._called_tools(result)
            first_tool = self._first_tool(result)
            # 宽松口径：期望工具之一被调用过即算正确（多工具协作是合法策略）
            correct = any(tool in actual_tools for tool in expected) if expected else False
            # 严格口径：第一个调用的工具必须在期望集合内
            strict = first_tool in expected if expected else False
            terminated[str(result.terminated_reason)] += 1
            details.append(
                {
                    "id": sample.get("id", ""),
                    "type": sample.get("type", ""),
                    "question": question,
                    "expected_tools": expected,
                    "expected_tools_source": expected_source,
                    "actual_tools": actual_tools,
                    "first_tool": first_tool,
                    "tool_correct": bool(correct),
                    "tool_correct_strict": bool(strict),
                    "num_tools": int(getattr(result, "tool_calls", 0)),
                    "iterations": int(getattr(result, "iterations", 0)),
                    "terminated_reason": str(getattr(result, "terminated_reason", "")),
                    "latency_ms": round(float(getattr(result, "latency_ms", 0.0)), 2),
                    "total_tokens": int(getattr(getattr(result, "usage", None), "total_tokens", 0)),
                    "answer_chars": len(str(getattr(result, "answer", ""))),
                    "tool_success_rate": round(float(getattr(result, "tool_success_rate", 1.0)), 4),
                    "session_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": error,
                }
            )

        stats = aggregate(
            [
                {
                    "tool_correct": 1.0 if d["tool_correct"] else 0.0,
                    "tool_correct_strict": 1.0 if d.get("tool_correct_strict") else 0.0,
                    "iterations": float(d["iterations"]),
                    "latency_ms": float(d["latency_ms"]),
                    "total_tokens": float(d["total_tokens"]),
                    "answer_chars": float(d["answer_chars"]),
                }
                for d in details
            ]
        )
        return {
            "num_samples": len(details),
            "tool_accuracy": stats.get("tool_correct", {}).get("mean", 0.0),
            "tool_accuracy_strict": stats.get("tool_correct_strict", {}).get("mean", 0.0),
            "avg_iterations": stats.get("iterations", {}).get("mean", 0.0),
            "max_iterations": stats.get("iterations", {}).get("max", 0.0),
            "avg_latency_ms": stats.get("latency_ms", {}).get("mean", 0.0),
            "p95_latency_ms": stats.get("latency_ms", {}).get("p95", 0.0),
            "total_tokens": int(sum(d["total_tokens"] for d in details)),
            "avg_tokens": stats.get("total_tokens", {}).get("mean", 0.0),
            "terminated_reasons": dict(terminated),
            "details": details,
            "errors": errors,
            "wall_ms": round((time.perf_counter() - began) * 1000, 2),
            "agent_stats": self.agent.stats() if hasattr(self.agent, "stats") else {},
        }


# ==============================================================================
# 三、端到端全链路评测
# ==============================================================================
class EndToEndBenchmark:
    """RAG 全链路评测：延迟分解 + 缓存命中率 + 降级率。

    Parameters
    ----------
    pipeline:
        :class:`~src.generation.rag_pipeline.RAGPipeline`。
    eval_set:
        评测样本列表。
    limit:
        最多评测条数（默认 30）。
    repeat:
        是否**重复提问**以测量语义缓存命中率（默认 True：同一条问题问两遍，
        第二遍应命中缓存）。这会额外增加一倍耗时，但缓存命中率无法用单次问答测得。
    """

    def __init__(
        self,
        pipeline: Any,
        eval_set: Sequence[Dict[str, Any]],
        limit: int = 30,
        repeat: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.eval_set = list(eval_set or ())
        self.limit = max(1, int(limit))
        self.repeat = bool(repeat)

    def run(self) -> Dict[str, Any]:
        """逐条调用 ``pipeline.answer()`` 并汇总。

        Returns
        -------
        dict
            ``{"num_calls", "avg_latency_ms", "avg_retrieval_ms", "avg_generation_ms",
            "cache_hit_rate", "degrade_rate", "degrade_reasons", "total_tokens",
            "details": [...]}``。
        """
        from .metrics import citation_accuracy, citation_coverage, faithfulness

        began = time.perf_counter()
        samples = self.eval_set[: self.limit]
        details: List[Dict[str, Any]] = []
        degrade_reasons: Counter = Counter()
        errors: List[str] = []

        for index, sample in enumerate(samples, start=1):
            question = str(sample.get("question") or "")
            if not question:
                continue
            rounds = 2 if self.repeat else 1
            for round_no in range(1, rounds + 1):
                try:
                    answer = self.pipeline.answer(question)
                    error = ""
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"[{sample.get('id')}] {type(exc).__name__}: {exc}")
                    details.append(
                        {
                            "id": sample.get("id", ""), "round": round_no, "question": question,
                            "latency_ms": 0.0, "retrieval_ms": 0.0, "generation_ms": 0.0,
                            "cache_hit": False, "degraded": False, "degrade_reason": "",
                            "num_retrieved": 0, "answer_chars": 0, "total_tokens": 0,
                            "citation_accuracy": 0.0, "citation_coverage": 0.0,
                            "faithfulness": 0.0, "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                contexts = [h.text for h in getattr(answer, "retrieved", []) or []]
                accuracy = citation_accuracy(
                    answer.answer, answer.citations, getattr(answer, "retrieved", []) or []
                )
                coverage = citation_coverage(answer.answer)
                grounded = faithfulness(answer.answer, contexts)
                if answer.degraded:
                    degrade_reasons[str(answer.degrade_reason or "unknown")] += 1

                details.append(
                    {
                        "id": sample.get("id", ""),
                        "round": round_no,
                        "type": sample.get("type", ""),
                        "question": question,
                        "latency_ms": round(float(answer.latency_ms), 3),
                        "retrieval_ms": round(float(answer.retrieval_ms), 3),
                        "generation_ms": round(float(answer.generation_ms), 3),
                        "cache_hit": bool(answer.cache_hit),
                        "degraded": bool(answer.degraded),
                        "degrade_reason": str(answer.degrade_reason or ""),
                        "num_retrieved": len(getattr(answer, "retrieved", []) or []),
                        "top_confidence": round(float(answer.top_confidence), 4),
                        "answer_chars": len(answer.answer or ""),
                        "total_tokens": int(getattr(answer.usage, "total_tokens", 0)),
                        "citation_total": accuracy["total"],
                        "citation_invalid": accuracy["invalid"],
                        "citation_accuracy": accuracy["accuracy"],
                        "citation_coverage": coverage["coverage"],
                        "faithfulness": grounded["score"],
                        # 完整答案与来源清单：供 Bad Case 分析复核引用与忠实度
                        # （只在第一轮记录，避免重复提问时数据冗余）
                        "answer": (answer.answer or "") if round_no == 1 else "",
                        "retrieved_sources": _sources_of(getattr(answer, "retrieved", []) or [])
                        if round_no == 1
                        else [],
                        "error": error,
                    }
                )

        stats = aggregate(
            [
                {
                    "latency_ms": float(d["latency_ms"]),
                    "retrieval_ms": float(d["retrieval_ms"]),
                    "generation_ms": float(d["generation_ms"]),
                    "cache_hit": 1.0 if d["cache_hit"] else 0.0,
                    "degraded": 1.0 if d["degraded"] else 0.0,
                    "citation_accuracy": float(d["citation_accuracy"]),
                    "citation_coverage": float(d["citation_coverage"]),
                    "faithfulness": float(d["faithfulness"]),
                    "total_tokens": float(d["total_tokens"]),
                    "num_retrieved": float(d["num_retrieved"]),
                    "answer_chars": float(d["answer_chars"]),
                }
                for d in details
            ]
        )
        first_round = [d for d in details if d.get("round") == 1]
        return {
            "num_calls": len(details),
            "num_questions": len(first_round),
            "repeat": self.repeat,
            "avg_latency_ms": stats.get("latency_ms", {}).get("mean", 0.0),
            "p95_latency_ms": stats.get("latency_ms", {}).get("p95", 0.0),
            "avg_retrieval_ms": stats.get("retrieval_ms", {}).get("mean", 0.0),
            "avg_generation_ms": stats.get("generation_ms", {}).get("mean", 0.0),
            "cache_hit_rate": stats.get("cache_hit", {}).get("mean", 0.0),
            "degrade_rate": stats.get("degraded", {}).get("mean", 0.0),
            "degrade_reasons": dict(degrade_reasons),
            "avg_citation_accuracy": stats.get("citation_accuracy", {}).get("mean", 0.0),
            "avg_citation_coverage": stats.get("citation_coverage", {}).get("mean", 0.0),
            "avg_faithfulness": stats.get("faithfulness", {}).get("mean", 0.0),
            "avg_retrieved": stats.get("num_retrieved", {}).get("mean", 0.0),
            "avg_answer_chars": stats.get("answer_chars", {}).get("mean", 0.0),
            "total_tokens": int(sum(d["total_tokens"] for d in details)),
            "details": details,
            "errors": errors,
            "pipeline_stats": self.pipeline.stats(),
            "wall_ms": round((time.perf_counter() - began) * 1000, 2),
        }


# ==============================================================================
# 统一入口
# ==============================================================================
def run_full_benchmark(
    config: Any = None,
    quick: bool = False,
    run_agent: bool = True,
    run_end_to_end: bool = True,
    eval_set: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """统一评测入口：检索三档 + Agent + 端到端，并落盘 JSON。

    Parameters
    ----------
    config:
        基础配置；None 表示读 ``config.yaml``。
    quick:
        快速模式：检索用全部样本但 Agent/端到端只跑少量样本，
        端到端关闭"重复提问"（不测缓存命中率），整体耗时控制在 1 分钟内。
    run_agent / run_end_to_end:
        是否运行对应小节（False 时该小节写 ``{"skipped": True}``）。
    eval_set:
        自定义评测集；None 表示用 :func:`~src.evaluation.eval_set.build_eval_set`。

    Returns
    -------
    dict
        结构化评测结果，同时写入 ``reports/性能评测数据.json``。
        任何小节失败都会在 ``["<section>"]["error"]`` 里留下真实错误信息，
        绝不会用编造的数字填充。
    """
    from ..utils.config import load_config
    from .eval_set import build_eval_set, eval_set_path, load_eval_set, save_eval_set

    began = time.perf_counter()
    base_config = config or load_config()
    if eval_set is None:
        eval_set = load_eval_set()
        if not eval_set:
            eval_set = build_eval_set()
    items = list(eval_set)

    # 样本量说明（重要）
    # ------------------
    # 评测集的**前 25 条全部是 factual 类**（easy/medium），基线在这些问题上接近满分。
    # 如果 quick 模式只取前若干条，Bad Case 分析就会因为"分析的都是简单题"而报 0 条失败，
    # 得出误导性的乐观结论。因此下面的取样一律**按类型跨段抽样**，
    # 保证 factual / comparison / summary / reasoning 四类都被覆盖。
    if quick:
        retrieval_items = _stratified_sample(items, 40)
        agent_samples = _stratified_sample(items, 20)
        e2e_samples = _stratified_sample(items, 40)
    else:
        retrieval_items = items
        agent_samples = _stratified_sample(items, 30)
        e2e_samples = items
    agent_limit = len(agent_samples)
    e2e_limit = len(e2e_samples)

    result: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quick_mode": bool(quick),
        "eval_set_size": len(items),
        "manifest": {
            "retrieval_samples": len(retrieval_items),
            "agent_samples": len(agent_samples) if run_agent else 0,
            "e2e_questions": len(e2e_samples) if run_end_to_end else 0,
            "sampling": "按问题类型分层抽样（quick 模式）" if quick else "全部样本",
        },
        "corpus": {},
        "retrieval": {},
        "agent": {},
        "end_to_end": {},
        "errors": [],
        "wall_ms": 0.0,
    }

    # ---- 准备语料 --------------------------------------------------------
    try:
        bundle = prepare_corpus(base_config)
        result["corpus"] = bundle.corpus_stats()
        result["corpus"]["tmpdir"] = str(bundle.tmpdir)
        result["corpus"]["index_report"] = {
            "num_chunks": bundle.index_report.get("num_chunks"),
            "latency_ms": bundle.index_report.get("latency_ms"),
        }
        result["environment"] = _environment_info(bundle.config)
    except Exception as exc:  # noqa: BLE001
        message = f"评测语料准备失败：{type(exc).__name__}: {exc}"
        result["errors"].append(message)
        result["retrieval"] = {"error": message, "table": []}
        result["agent"] = {"error": message}
        result["end_to_end"] = {"error": message}
        result["wall_ms"] = round((time.perf_counter() - began) * 1000, 2)
        write_performance_json(result, base_config)
        return result

    # ---- [1] 检索三档 ---------------------------------------------------
    try:
        result["retrieval"] = RetrievalBenchmark(
            bundle.engine, retrieval_items, top_k=5, match="text"
        ).run()
    except Exception as exc:  # noqa: BLE001
        message = f"检索评测失败：{type(exc).__name__}: {exc}"
        result["errors"].append(message)
        result["retrieval"] = {"error": message, "table": []}

    # ---- [2] Agent ------------------------------------------------------
    if not run_agent:
        result["agent"] = {"skipped": True, "reason": "run_agent=False"}
    else:
        try:
            agent = bundle.create_agent()
            result["agent"] = AgentBenchmark(
                agent, agent_samples, limit=len(agent_samples)
            ).run()
        except Exception as exc:  # noqa: BLE001
            message = f"Agent 评测失败：{type(exc).__name__}: {exc}"
            result["errors"].append(message)
            result["agent"] = {"error": message}

    # ---- [3] 端到端 -----------------------------------------------------
    if not run_end_to_end:
        result["end_to_end"] = {"skipped": True, "reason": "run_end_to_end=False"}
    else:
        try:
            pipeline = bundle.create_pipeline()
            result["end_to_end"] = EndToEndBenchmark(
                pipeline, e2e_samples, limit=len(e2e_samples), repeat=not quick
            ).run()
        except Exception as exc:  # noqa: BLE001
            message = f"端到端评测失败：{type(exc).__name__}: {exc}"
            result["errors"].append(message)
            result["end_to_end"] = {"error": message}

    result["eval_set_path"] = str(eval_set_path(base_config))
    result["wall_ms"] = round((time.perf_counter() - began) * 1000, 2)
    write_performance_json(result, base_config)
    # 顺手把评测集也落盘一份，保证"报告里引用的评测集"与"实际使用的评测集"一致
    try:
        save_eval_set(items, eval_set_path(base_config))
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"评测集落盘失败：{exc}")
    return result


def _environment_info(config: Any) -> Dict[str, Any]:
    """采集运行环境信息（写入报告，保证评测结果可复现）。"""
    import platform
    import sys

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        info["numpy"] = "未安装"
    try:
        from ..llm import create_llm

        llm = create_llm(config)
        info["llm_provider"] = getattr(llm, "provider", "unknown")
        info["llm_model"] = getattr(llm, "model", "")
    except Exception as exc:  # noqa: BLE001
        info["llm_provider"] = f"创建失败：{exc}"
    try:
        from ..embeddings import create_embedder

        embedder = create_embedder(config)
        info["embedding_provider"] = getattr(embedder, "provider", "unknown")
        info["embedding_model"] = getattr(embedder, "model", "")
        info["embedding_dim"] = getattr(embedder, "dimension", 0)
    except Exception as exc:  # noqa: BLE001
        info["embedding_provider"] = f"创建失败：{exc}"
    for key in ("llm.provider", "retrieval.top_k", "retrieval.candidate_k",
                "retrieval.enable_rerank", "chunking.chunk_size", "chunking.strategy"):
        try:
            info[key] = config.get(key)
        except Exception:  # noqa: BLE001
            pass
    return info


def write_performance_json(result: Dict[str, Any], config: Any = None) -> Path:
    """把评测结果写入 ``reports/性能评测数据.json``（原子写，UTF-8）。"""
    from ..utils.config import load_config
    from ..utils.storage import write_json

    cfg = config or load_config()
    report_dir = cfg.resolve_path(str(cfg.paths.report_dir))
    target = Path(report_dir) / "性能评测数据.json"
    write_json(target, result)
    return target


# ==============================================================================
# Excel 导出（openpyxl，可选依赖）
# ==============================================================================
def export_excel(results: Dict[str, Any], path: Any = None, config: Any = None) -> Optional[Path]:
    """把评测指标导出为多 sheet 的 Excel（**openpyxl 缺失时优雅降级**）。

    Sheet 结构
    ----------
    1. ``检索三档对比`` —— Hit@5 / MRR / Recall@5 / Precision@5 / NDCG@5 /
       平均延迟 / P95 延迟 / 相对提升
    2. ``Agent 指标`` —— 汇总指标 + 逐条决策明细
    3. ``端到端`` —— 延迟分解、缓存命中率、降级率、引用质量
    4. ``逐条明细`` —— 检索（三档）+ 端到端的全部逐条记录

    Parameters
    ----------
    results: :func:`run_full_benchmark` 的返回值。
    path: 目标文件；None 表示 ``reports/性能评测数据.xlsx``。
    config: 配置对象（用于解析 ``paths.report_dir``）。

    Returns
    -------
    Path 或 None
        openpyxl 不可用时打印提示并返回 ``None``（不抛异常）。
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:  # pragma: no cover - 取决于环境是否装了 openpyxl
        print(
            f"[跳过] openpyxl 不可用（{exc}），未生成 Excel。"
            f"如需 Excel 报告请执行：pip install openpyxl"
        )
        return None

    from ..utils.config import load_config
    from ..utils.storage import ensure_dir

    cfg = config or load_config()
    if path is None:
        report_dir = cfg.resolve_path(str(cfg.paths.report_dir))
        target = Path(report_dir) / "性能评测数据.xlsx"
    else:
        target = Path(path)
    ensure_dir(target.parent)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    title_font = Font(bold=True, size=12)

    workbook = Workbook()
    workbook.remove(workbook.active)

    # ---- Sheet 1：检索三档对比 -----------------------------------------
    sheet = workbook.create_sheet("检索三档对比")
    sheet["A1"] = "检索层三档对比（纯向量 → 混合检索 → 混合+重排序）"
    sheet["A1"].font = title_font
    sheet.append([])
    columns = ["档位", "模式", "Hit@5", "MRR", "Recall@5", "Precision@5", "NDCG@5",
               "平均延迟(ms)", "P95延迟(ms)", "Hit相对提升", "MRR相对提升"]
    sheet.append(columns)
    for row in (results.get("retrieval") or {}).get("table") or []:
        sheet.append([
            row.get("mode_label", ""), row.get("mode", ""),
            row.get("hit@k", 0.0), row.get("mrr", 0.0), row.get("recall@k", 0.0),
            row.get("precision@k", 0.0), float(row.get("ndcg@k", 0.0) or 0.0),
            row.get("avg_latency_ms", 0.0), row.get("p95_latency_ms", 0.0),
            row.get("hit_relative_gain", 0.0), row.get("mrr_relative_gain", 0.0),
        ])
    _style_sheet(sheet, header_row=3, header_font=header_font, header_fill=header_fill)

    # ---- Sheet 2：Agent 指标 -------------------------------------------
    sheet = workbook.create_sheet("Agent 指标")
    sheet["A1"] = "Agent 决策层指标"
    sheet["A1"].font = title_font
    agent = results.get("agent") or {}
    summary_rows = [
        ("样本数", agent.get("num_samples", "未运行")),
        ("工具调用准确率（宽松：期望工具被调用过）", agent.get("tool_accuracy", "未运行")),
        ("工具调用准确率（严格：首个工具正确）", agent.get("tool_accuracy_strict", "未运行")),
        ("平均推理轮次", agent.get("avg_iterations", "未运行")),
        ("最大推理轮次", agent.get("max_iterations", "未运行")),
        ("平均延迟(ms)", agent.get("avg_latency_ms", "未运行")),
        ("P95延迟(ms)", agent.get("p95_latency_ms", "未运行")),
        ("Token 总消耗", agent.get("total_tokens", "未运行")),
        ("平均 Token/题", agent.get("avg_tokens", "未运行")),
    ]
    for label, value in summary_rows:
        sheet.append([label, value])
    sheet.append([])
    sheet.append(["终止原因", "次数"])
    for reason, count in (agent.get("terminated_reasons") or {}).items():
        sheet.append([reason, count])
    sheet.append([])
    agent_header_row = sheet.max_row + 1
    sheet.append(["题号", "类型", "问题", "期望工具", "期望来源", "实际工具",
                  "首个工具", "工具正确", "调用次数", "推理轮次", "终止原因",
                  "延迟(ms)", "Token", "答案长度"])
    for row in agent.get("details") or []:
        sheet.append([
            row.get("id", ""), row.get("type", ""), row.get("question", ""),
            "、".join(row.get("expected_tools") or []), row.get("expected_tools_source", ""),
            "、".join(row.get("actual_tools") or []), row.get("first_tool", ""),
            "是" if row.get("tool_correct") else "否",
            row.get("num_tools", 0), row.get("iterations", 0),
            row.get("terminated_reason", ""), row.get("latency_ms", 0.0),
            row.get("total_tokens", 0), row.get("answer_chars", 0),
        ])
    _style_sheet(sheet, header_row=agent_header_row, header_font=header_font, header_fill=header_fill)

    # ---- Sheet 3：端到端 ----------------------------------------------
    sheet = workbook.create_sheet("端到端")
    sheet["A1"] = "全链路（RAG Pipeline）指标"
    sheet["A1"].font = title_font
    e2e = results.get("end_to_end") or {}
    sheet.append([])
    for label, value in [
        ("调用次数", e2e.get("num_calls", "未运行")),
        ("问题数", e2e.get("num_questions", "未运行")),
        ("是否重复提问（测缓存）", "是" if e2e.get("repeat") else "否"),
        ("平均总延迟(ms)", e2e.get("avg_latency_ms", "未运行")),
        ("P95 总延迟(ms)", e2e.get("p95_latency_ms", "未运行")),
        ("平均检索耗时(ms)", e2e.get("avg_retrieval_ms", "未运行")),
        ("平均生成耗时(ms)", e2e.get("avg_generation_ms", "未运行")),
        ("缓存命中率", e2e.get("cache_hit_rate", "未运行")),
        ("降级率", e2e.get("degrade_rate", "未运行")),
        ("平均引用准确率", e2e.get("avg_citation_accuracy", "未运行")),
        ("平均引用覆盖率", e2e.get("avg_citation_coverage", "未运行")),
        ("平均忠实度", e2e.get("avg_faithfulness", "未运行")),
        ("平均召回片段数", e2e.get("avg_retrieved", "未运行")),
        ("Token 总消耗", e2e.get("total_tokens", "未运行")),
    ]:
        sheet.append([label, value])
    sheet.append([])
    sheet.append(["降级原因", "次数"])
    for reason, count in (e2e.get("degrade_reasons") or {}).items():
        sheet.append([reason, count])
    _style_sheet(sheet, header_row=1, header_font=header_font, header_fill=header_fill)

    # ---- Sheet 4：逐条明细 --------------------------------------------
    sheet = workbook.create_sheet("逐条明细")
    sheet.append(["来源", "题号", "阶段/模式", "问题", "命中", "MRR", "Recall",
                  "NDCG", "延迟(ms)", "缓存", "降级", "引用准确率", "忠实度"])
    retrieval = results.get("retrieval") or {}
    for mode, rows in (retrieval.get("per_mode_detail") or {}).items():
        for row in rows:
            sheet.append([
                "检索", row.get("question", "")[:60], mode, row.get("question", ""),
                "是" if row.get("hit") else "否", row.get("mrr", 0.0),
                row.get("recall", 0.0), row.get("ndcg", 0.0),
                row.get("latency_ms", 0.0), "", "", "", "",
            ])
    for row in (e2e.get("details") or []):
        sheet.append([
            "端到端", row.get("id", ""), f"第{row.get('round', 1)}轮",
            row.get("question", ""), "", "", "", "",
            row.get("latency_ms", 0.0),
            "是" if row.get("cache_hit") else "否",
            row.get("degrade_reason", "") or ("是" if row.get("degraded") else "否"),
            row.get("citation_accuracy", 0.0), row.get("faithfulness", 0.0),
        ])
    _style_sheet(sheet, header_row=1, header_font=header_font, header_fill=header_fill)

    workbook.save(target)
    print(f"[OK] Excel 已导出：{target}")
    return target


def _style_sheet(
    sheet: Any,
    header_row: int,
    header_font: Any,
    header_fill: Any,
    min_width: int = 10,
    max_width: int = 60,
) -> None:
    """表头加粗填色 + 列宽自适应（按显示宽度估算，中文按 2 个字符宽计）。"""
    from openpyxl.utils import get_column_letter

    for cell in sheet[header_row]:
        if cell.value is not None:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = _wrap_alignment()

    widths: Dict[int, int] = {}
    for row in sheet.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            width = _display_width(str(cell.value))
            column = cell.column
            widths[column] = max(widths.get(column, 0), width)
    for column, width in widths.items():
        sheet.column_dimensions[get_column_letter(column)].width = max(
            min_width, min(max_width, width + 2)
        )


def _wrap_alignment() -> Any:
    """表头单元格对齐方式（openpyxl 不可用时返回 None）。"""
    try:
        from openpyxl.styles import Alignment

        return Alignment(horizontal="center", vertical="center", wrap_text=True)
    except ImportError:  # pragma: no cover
        return None


def _display_width(text: str) -> int:
    """估算字符串显示宽度（CJK 字符按 2 计）。"""
    return sum(2 if "\u1100" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef" else 1 for ch in text)
