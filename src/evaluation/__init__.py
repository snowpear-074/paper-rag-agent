"""模块五 —— 评测与优化（Evaluation & Optimization）。

本包是 RAG+Agent 论文问答系统的**质量度量与迭代闭环**：
前四个模块负责"把系统做出来"，本模块负责"用数据证明它有多好、并指出下一步改哪里"。

四个子模块
----------
.. list-table::
   :widths: 18 82

   * - :mod:`~src.evaluation.metrics`
     - **纯函数指标库**：检索指标（Hit@K / MRR / Recall@K / Precision@K /
       NDCG@K / MAP@K）+ 生成指标（引用准确率 / 引用覆盖率 / 答案相关性 /
       忠实度 / Token-F1 / ROUGE-L / EM）+ 聚合工具。零第三方依赖。
   * - :mod:`~src.evaluation.eval_set`
     - **评测集构建**：基于 3 篇样例论文手写 60+ 条问答对，
       覆盖 factual / comparison / summary / reasoning 四类问题，中英混合提问。
   * - :mod:`~src.evaluation.benchmark`
     - **系统级性能评测**：检索三档对比（纯向量 → 混合 → 混合+重排序）、
       Agent 决策层指标、端到端延迟分解，并导出 Excel。
   * - :mod:`~src.evaluation.badcase`
     - **Bad Case 分析与优化**：四类失败自动分类、根因归纳、
       真实参数对照实验与 Markdown 报告。
   * - :mod:`~src.evaluation.report`
     - **报告汇总**：一次产出系统评测报告 / 性能数据 JSON / Excel / Bad Case 报告。

运行方式
--------
.. code-block:: bash

    python -m src.evaluation.eval_set        # 生成 reports/评测集.json
    python -m src.evaluation.badcase         # Bad Case 分析（含真实对照实验）
    python -m src.evaluation.report --quick  # 快速评测，产出全部报告
    python -m unittest tests.test_evaluation -v

设计约束
--------
- **单向依赖**：本包可以导入 ``src.retrieval`` / ``src.generation`` / ``src.agent``，
  反过来绝不可以 —— 保证评测代码不污染业务代码，业务代码也不需要感知评测的存在。
- **零编造数据**：所有指标均来自实际运行；未运行的小节在报告中显式标注「未运行」。
- **不修改用户知识库**：所有评测索引都建在 ``tempfile.mkdtemp()`` 的临时目录下。
"""

from __future__ import annotations

from .badcase import (
    CATEGORIES,
    CATEGORY_LABELS,
    BadCase,
    BadCaseReport,
    analyze,
    classify_badcase,
    optimization_experiment,
    save_markdown,
)
from .benchmark import (
    MODE_LABELS,
    RETRIEVAL_MODES,
    AgentBenchmark,
    CorpusBundle,
    EndToEndBenchmark,
    RetrievalBenchmark,
    export_excel,
    prepare_corpus,
    run_full_benchmark,
)
from .eval_set import (
    DIFFICULTIES,
    DOC_BERT,
    DOC_RESNET,
    DOC_TRANSFORMER,
    QUESTION_TYPES,
    build_eval_set,
    eval_set_path,
    eval_set_statistics,
    load_eval_set,
    save_eval_set,
    validate_eval_set,
)
from .metrics import (
    aggregate,
    answer_relevance,
    citation_accuracy,
    citation_coverage,
    exact_match,
    faithfulness,
    hit_at_k,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)
from .report import bar_chart, generate_reports, render_markdown

__all__ = [
    # 指标
    "hit_at_k", "mrr", "recall_at_k", "precision_at_k", "ndcg_at_k", "map_at_k",
    "citation_accuracy", "citation_coverage", "answer_relevance", "faithfulness",
    "token_f1", "exact_match", "rouge_l", "aggregate",
    # 评测集
    "build_eval_set", "save_eval_set", "load_eval_set", "validate_eval_set",
    "eval_set_statistics", "eval_set_path",
    "QUESTION_TYPES", "DIFFICULTIES",
    "DOC_TRANSFORMER", "DOC_RESNET", "DOC_BERT",
    # 基准测试
    "RetrievalBenchmark", "AgentBenchmark", "EndToEndBenchmark",
    "run_full_benchmark", "prepare_corpus", "export_excel",
    "CorpusBundle", "RETRIEVAL_MODES", "MODE_LABELS",
    # Bad Case
    "BadCase", "BadCaseReport", "classify_badcase", "analyze",
    "save_markdown", "optimization_experiment",
    "CATEGORIES", "CATEGORY_LABELS",
    # 报告
    "generate_reports", "render_markdown", "bar_chart",
]

__version__ = "1.0.0"
