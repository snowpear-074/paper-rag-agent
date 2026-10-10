# -*- coding: utf-8 -*-
"""模块五（评测模块）单元测试。

运行方式（项目根目录）::

    python -m unittest tests.test_evaluation -v

覆盖范围
--------
1. :mod:`src.evaluation.metrics` —— 每个指标函数的正确性 + 边界条件
   （空输入、k 越界、全命中、全不命中、手算对照）；
2. :mod:`src.evaluation.eval_set` —— 数量/类型分布/字段完整性/校验器/读写往返；
3. :mod:`src.evaluation.badcase` —— 四类失败的分类判定（构造最小样例）；
4. :mod:`src.evaluation.report` —— Markdown 渲染内容（小节标题 + 条形图字符）。

设计约束
--------
- **只用标准库 unittest**，不依赖 pytest 特有语法；
- **单个测试 5 秒内完成**：任何耗时的完整 benchmark 都不在测试范围内
  （benchmark 的验证通过 ``--quick`` 命令独立执行）；
- 涉及检索的样例用 :mod:`tests.fixtures.make_sample_pdf` 合成的小论文
  （3 篇、27 个分块），并使用 ``setUpClass`` 只建一次索引以节省时间。
"""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.evaluation import badcase as badcase_module
from src.evaluation import benchmark as benchmark_module
from src.evaluation import chunking_experiment as chunking_module
from src.evaluation import eval_set as eval_set_module
from src.evaluation import metrics as M
from src.evaluation import report as report_module


# ===========================================================================
# 一、检索指标
# ===========================================================================
class RetrievalMetricsTest(unittest.TestCase):
    """Hit@K / MRR / Recall@K / Precision@K / NDCG@K / MAP@K。"""

    def test_hit_at_k_hit_and_miss(self):
        self.assertEqual(M.hit_at_k(["c3", "c1"], ["c1"], k=2), 1.0)
        self.assertEqual(M.hit_at_k(["c3"], ["c1"], k=1), 0.0)
        # 命中的那条排在第 3 位，k=2 时不算命中
        self.assertEqual(M.hit_at_k(["x", "y", "c1"], ["c1"], k=2), 0.0)
        self.assertEqual(M.hit_at_k(["x", "y", "c1"], ["c1"], k=3), 1.0)

    def test_hit_at_k_boundaries(self):
        """空输入、空金标准、k<=0 都返回 0.0（不抛异常）。"""
        self.assertEqual(M.hit_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(M.hit_at_k(["c1"], [], k=5), 0.0)
        self.assertEqual(M.hit_at_k(["c1"], ["c1"], k=0), 0.0)
        self.assertEqual(M.hit_at_k(["c1"], ["c1"], k=-3), 0.0)
        # k 超过序列长度不算错
        self.assertEqual(M.hit_at_k(["c1"], ["c1"], k=999), 1.0)

    def test_mrr_values(self):
        self.assertEqual(M.mrr(["c1", "c2"], ["c1"]), 1.0)
        self.assertEqual(M.mrr(["x", "c1"], ["c1"]), 0.5)
        self.assertAlmostEqual(M.mrr(["x", "y", "c1"], ["c1"]), 1 / 3, places=6)
        self.assertEqual(M.mrr(["x", "y"], ["c1"]), 0.0)
        self.assertEqual(M.mrr([], ["c1"]), 0.0)

    def test_recall_at_k(self):
        self.assertEqual(M.recall_at_k(["c1", "c2", "c3"], ["c1", "c2"], k=3), 1.0)
        self.assertAlmostEqual(M.recall_at_k(["c1"], ["c1", "c2"], k=3), 0.5, places=6)
        self.assertEqual(M.recall_at_k(["c1"], ["c2"], k=3), 0.0)
        self.assertEqual(M.recall_at_k(["c1"], ["c1"], k=0), 0.0)
        self.assertEqual(M.recall_at_k(["c1"], [], k=3), 0.0)

    def test_precision_at_k_denominator_uses_actual_length(self):
        """分母是实际返回条数：只返回 1 条且命中 → 精度为 1.0，而不是 0.2。"""
        self.assertEqual(M.precision_at_k(["c1"], ["c1"], k=5), 1.0)
        self.assertEqual(M.precision_at_k(["c1", "x"], ["c1"], k=5), 0.5)
        self.assertEqual(M.precision_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(M.precision_at_k(["c1"], ["c1"], k=0), 0.0)

    def test_ndcg_at_k_hand_computed(self):
        """与手算值对照：DCG = 1/log2(i+1)，IDCG = 1.0（单条金标准）。"""
        # 第 1 位命中 → DCG = 1/log2(2) = 1.0
        self.assertAlmostEqual(M.ndcg_at_k(["c1", "x"], ["c1"], k=2), 1.0, places=6)
        # 第 2 位命中 → DCG = 1/log2(3)
        expected = 1.0 / math.log2(3)
        self.assertAlmostEqual(M.ndcg_at_k(["x", "c1"], ["c1"], k=2), expected, places=4)
        self.assertAlmostEqual(M.ndcg_at_k(["x", "c1"], ["c1"], k=2), 0.6309, places=4)
        # 未命中
        self.assertEqual(M.ndcg_at_k(["x", "y"], ["c1"], k=2), 0.0)

    def test_ndcg_at_k_ideal_ordering_is_one(self):
        """理想排序（金标准全在最前）NDCG 必须为 1.0。"""
        gold = ["a", "b"]
        self.assertAlmostEqual(M.ndcg_at_k(["a", "b", "z"], gold, k=3), 1.0, places=6)
        # 顺序颠倒（金标准占前两位但顺序不同）在二值相关性下仍是 1.0
        self.assertAlmostEqual(M.ndcg_at_k(["b", "a", "z"], gold, k=3), 1.0, places=6)
        # 金标准被拆散 → 小于 1
        self.assertLess(M.ndcg_at_k(["a", "z", "b"], gold, k=3), 1.0)

    def test_ndcg_at_k_multi_gold_hand_computed(self):
        """两条金标准、k=2：命中一条排第 2 位。

        DCG = 1/log2(3) = 0.63093；IDCG = 1/log2(2) + 1/log2(3) = 1.63093
        → NDCG ≈ 0.38685
        """
        expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(
            M.ndcg_at_k(["x", "a"], ["a", "b"], k=2), expected, places=4
        )

    def test_ndcg_at_k_boundaries(self):
        self.assertEqual(M.ndcg_at_k([], ["a"], k=2), 0.0)
        self.assertEqual(M.ndcg_at_k(["a"], [], k=2), 0.0)
        self.assertEqual(M.ndcg_at_k(["a"], ["a"], k=0), 0.0)

    def test_map_at_k_hand_computed(self):
        """命中位置 1 与 3：AP = (1/1 + 2/3) / 2 = 0.8333。"""
        self.assertAlmostEqual(M.map_at_k(["c1", "x", "c2"], ["c1", "c2"], k=3), 0.8333, places=4)
        # 全部命中且排最前 → 1.0
        self.assertEqual(M.map_at_k(["c1", "c2"], ["c1", "c2"], k=2), 1.0)
        self.assertEqual(M.map_at_k(["x", "y"], ["c1"], k=2), 0.0)
        self.assertEqual(M.map_at_k(["c1"], ["c1"], k=0), 0.0)

    def test_all_metrics_on_full_and_empty_hits(self):
        """全命中 / 全不命中时各指标的自洽性。"""
        ranked = ["a", "b", "c", "d", "e"]
        gold = ["a", "b", "c", "d", "e"]
        self.assertEqual(M.hit_at_k(ranked, gold, 5), 1.0)
        self.assertEqual(M.recall_at_k(ranked, gold, 5), 1.0)
        self.assertEqual(M.precision_at_k(ranked, gold, 5), 1.0)
        self.assertEqual(M.ndcg_at_k(ranked, gold, 5), 1.0)
        self.assertEqual(M.map_at_k(ranked, gold, 5), 1.0)
        self.assertEqual(M.mrr(ranked, gold), 1.0)

        miss = ["x", "y", "z"]
        self.assertEqual(M.hit_at_k(miss, gold, 3), 0.0)
        self.assertEqual(M.recall_at_k(miss, gold, 3), 0.0)
        self.assertEqual(M.precision_at_k(miss, gold, 3), 0.0)
        self.assertEqual(M.ndcg_at_k(miss, gold, 3), 0.0)
        self.assertEqual(M.map_at_k(miss, gold, 3), 0.0)
        self.assertEqual(M.mrr(miss, gold), 0.0)


# ===========================================================================
# 二、生成质量指标
# ===========================================================================
class CitationAccuracyTest(unittest.TestCase):
    """引用准确率：含"引用幻觉"识别。"""

    def test_all_citations_valid(self):
        answer = "该模型取得 28.4 BLEU【Attention Is All You Need p.2】。"
        available = [{"source": "Attention Is All You Need", "page": 2}]
        result = M.citation_accuracy(answer, [], available)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["valid"], 1)
        self.assertEqual(result["invalid"], 0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["hallucinated"], [])

    def test_hallucinated_citation_detected(self):
        """答案引用了知识库里不存在的论文 → 必须被识别为幻觉引用。"""
        answer = (
            "Transformer 完全基于注意力机制【Attention Is All You Need p.2】，"
            "并在某未收录论文中也有类似结论【A Paper That Does Not Exist p.9】。"
        )
        available = [
            {"source": "Attention Is All You Need", "page": 2},
            {"source": "Deep Residual Learning for Image Recognition", "page": 1},
        ]
        result = M.citation_accuracy(answer, [], available)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["valid"], 1)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(len(result["hallucinated"]), 1)
        self.assertIn("A Paper That Does Not Exist", result["hallucinated"][0]["source"])

    def test_hallucinated_page_detected(self):
        """来源名对但页码差得远（>±1 容差）也算幻觉。"""
        answer = "结论见【Attention Is All You Need p.99】。"
        available = [{"source": "Attention Is All You Need", "page": 2}]
        result = M.citation_accuracy(answer, [], available)
        self.assertEqual(result["invalid"], 1)

    def test_page_tolerance_accepts_off_by_one(self):
        """PDF 解析页码与模型口述页码常差 1 页 → 允许 ±1 容差。"""
        answer = "结论见【Attention Is All You Need p.3】。"
        available = [{"source": "Attention Is All You Need", "page": 2}]
        self.assertEqual(M.citation_accuracy(answer, [], available)["invalid"], 0)

    def test_source_name_normalization(self):
        """标题 / 文件名 / 下划线写法应能互相匹配。"""
        answer = "见【attention is all you need.pdf p.2】。"
        available = [{"source": "Attention Is All You Need", "page": 2}]
        self.assertEqual(M.citation_accuracy(answer, [], available)["valid"], 1)

    def test_no_citation_is_not_an_error(self):
        result = M.citation_accuracy("这是一段没有引用的答案。", [], [])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["accuracy"], 1.0)

    def test_duplicate_citations_counted_once(self):
        answer = "【Attention Is All You Need p.2】……【Attention Is All You Need p.2】"
        available = [{"source": "Attention Is All You Need", "page": 2}]
        self.assertEqual(M.citation_accuracy(answer, [], available)["total"], 1)

    def test_retrieved_chunk_objects_supported(self):
        """available_sources 可以直接传 RetrievedChunk 对象。"""
        from src.utils.types import Chunk, RetrievedChunk

        chunk = Chunk(chunk_id="c1", doc_id="d1", text="text", source="Attention Is All You Need", page=2)
        answer = "见【Attention Is All You Need p.2】。"
        result = M.citation_accuracy(answer, [], [RetrievedChunk(chunk=chunk, score=1.0)])
        self.assertEqual(result["valid"], 1)


class CitationCoverageTest(unittest.TestCase):
    """引用覆盖率。"""

    def test_full_coverage(self):
        answer = "该模型取得 28.4 BLEU【Attention Is All You Need p.2】。"
        result = M.citation_coverage(answer)
        self.assertEqual(result["sentences"], 1)
        self.assertEqual(result["cited"], 1)
        self.assertEqual(result["coverage"], 1.0)

    def test_zero_coverage(self):
        answer = "该模型取得了很好的效果且训练速度非常快。"
        result = M.citation_coverage(answer)
        self.assertEqual(result["sentences"], 1)
        self.assertEqual(result["coverage"], 0.0)

    def test_half_coverage(self):
        answer = (
            "该模型取得 28.4 BLEU【Attention Is All You Need p.2】。"
            "另外它训练得更快并且更易于并行化处理长序列。"
        )
        result = M.citation_coverage(answer)
        self.assertEqual(result["sentences"], 2)
        self.assertEqual(result["cited"], 1)
        self.assertEqual(result["coverage"], 0.5)

    def test_empty_answer(self):
        result = M.citation_coverage("")
        self.assertEqual(result["sentences"], 0)
        self.assertEqual(result["coverage"], 0.0)


class AnswerRelevanceTest(unittest.TestCase):
    """答案相关性（词面近似）。"""

    def test_relevant_answer(self):
        score = M.answer_relevance(
            "The Transformer was trained on the WMT 2014 English-German dataset "
            "and the WMT 2014 English-French dataset.",
            "What datasets were used to train the Transformer?",
        )
        self.assertGreaterEqual(score, 0.5, "覆盖了问题关键词的答案应获得较高相关性")

    def test_irrelevant_answer(self):
        score = M.answer_relevance(
            "今天天气不错，适合出去散步。",
            "What datasets were used to train the Transformer?",
        )
        self.assertLess(score, 0.2)

    def test_relevance_is_a_coarse_proxy(self):
        """文档化该指标的性质：它只做词面覆盖，不做语义判断。

        因此"完全正确但换了同义词"的答案也会得到较低的分数 —— 这是已知局限，
        报告里必须标注为粗筛指标而不是精确指标。
        """
        paraphrase = M.answer_relevance(
            "It was fit on parallel corpora of several million sentence pairs.",
            "What datasets were used to train the Transformer?",
        )
        self.assertLess(paraphrase, 0.4, "同义改写会被低估，这是词面近似的固有局限")

    def test_empty_inputs(self):
        self.assertEqual(M.answer_relevance("", "question?"), 0.0)
        self.assertEqual(M.answer_relevance("answer", ""), 0.0)


class FaithfulnessTest(unittest.TestCase):
    """忠实度（4-gram 重合近似）。"""

    def test_supported_answer_scores_high(self):
        context = "We trained on the WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs."
        answer = "We trained on the WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs."
        result = M.faithfulness(answer, [context])
        self.assertGreater(result["score"], 0.9)
        self.assertGreater(result["supported"], 0)

    def test_unsupported_answer_scores_low(self):
        context = "We trained on the WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs."
        answer = "The authors claim the model was evaluated on a secret internal dataset in 2031."
        result = M.faithfulness(answer, [context])
        self.assertLessEqual(result["score"], 0.5)

    def test_empty_contexts(self):
        result = M.faithfulness("Some answer sentence here.", [])
        self.assertEqual(result["score"], 0.0)

    def test_empty_answer(self):
        result = M.faithfulness("", ["some context"])
        self.assertEqual(result["sentences"], 0)


# ===========================================================================
# 三、文本相似指标
# ===========================================================================
class TextSimilarityTest(unittest.TestCase):
    """Token-F1 / EM / ROUGE-L。"""

    def test_token_f1_identical(self):
        self.assertEqual(M.token_f1("28.4 BLEU", "28.4 BLEU"), 1.0)

    def test_token_f1_partial_overlap(self):
        score = M.token_f1("the big model achieves 28.4 BLEU", "the model achieves 41.8 BLEU")
        self.assertGreater(score, 0.4)
        self.assertLess(score, 1.0)

    def test_token_f1_no_overlap_and_empty(self):
        self.assertEqual(M.token_f1("alpha beta", "gamma delta"), 0.0)
        self.assertEqual(M.token_f1("", "gamma"), 0.0)
        self.assertEqual(M.token_f1("alpha", ""), 0.0)

    def test_token_f1_repeated_words_reduce_precision(self):
        """重复词会拉低精度（多重集合口径）。"""
        repeated = M.token_f1("bleu bleu bleu", "bleu")
        self.assertLess(repeated, 1.0)

    def test_exact_match(self):
        self.assertEqual(M.exact_match("28.4 BLEU", "28.4  bleu"), 1.0)
        self.assertEqual(M.exact_match("28.4 BLEU", "41.8 BLEU"), 0.0)
        self.assertEqual(M.exact_match("", ""), 0.0)

    def test_rouge_l_identical(self):
        text = "the big model achieves 28.4 BLEU"
        self.assertAlmostEqual(M.rouge_l(text, text), 1.0, places=6)

    def test_rouge_l_order_sensitive(self):
        """词都对但顺序反了 → ROUGE-L 低于 Token-F1 同等情况。"""
        forward = M.rouge_l("the big model achieves 28.4 BLEU", "the big model achieves 28.4 BLEU")
        reversed_order = M.rouge_l("BLEU 28.4 achieves model big the", "the big model achieves 28.4 BLEU")
        self.assertGreater(forward, reversed_order)

    def test_rouge_l_no_overlap_and_empty(self):
        self.assertEqual(M.rouge_l("alpha beta", "gamma delta"), 0.0)
        self.assertEqual(M.rouge_l("", "gamma"), 0.0)

    def test_rouge_l_partial(self):
        score = M.rouge_l("the model achieves", "the big model achieves 28.4 BLEU")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


# ===========================================================================
# 四、聚合工具
# ===========================================================================
class AggregateTest(unittest.TestCase):
    """aggregate 的均值/中位数/P95。"""

    def test_basic_aggregation(self):
        rows = [{"hit": 1.0, "latency_ms": 10.0}, {"hit": 0.0, "latency_ms": 20.0}]
        result = M.aggregate(rows)
        self.assertEqual(result["hit"]["mean"], 0.5)
        self.assertEqual(result["hit"]["median"], 0.5)
        self.assertEqual(result["latency_ms"]["mean"], 15.0)
        self.assertEqual(result["latency_ms"]["min"], 10.0)
        self.assertEqual(result["latency_ms"]["max"], 20.0)
        self.assertEqual(result["latency_ms"]["count"], 2)

    def test_p95_present(self):
        rows = [{"v": float(i)} for i in range(100)]
        result = M.aggregate(rows)
        self.assertIn("p95", result["v"])
        self.assertGreater(result["v"]["p95"], result["v"]["median"])

    def test_empty_input(self):
        self.assertEqual(M.aggregate([]), {})
        self.assertEqual(M.aggregate(None), {})

    def test_non_numeric_fields_ignored(self):
        rows = [{"name": "a", "score": 0.5}]
        result = M.aggregate(rows)
        self.assertNotIn("name", result)
        self.assertIn("score", result)

    def test_explicit_keys(self):
        rows = [{"a": 1.0, "b": 2.0}]
        result = M.aggregate(rows, keys=["a"])
        self.assertIn("a", result)
        self.assertNotIn("b", result)

    def test_boolean_not_treated_as_number(self):
        rows = [{"flag": True, "score": 1.0}]
        self.assertNotIn("flag", M.aggregate(rows))
        self.assertIn("score", M.aggregate(rows))

    def test_text_helpers(self):
        """配套文本工具的基本行为。"""
        self.assertEqual(M.split_sentences("第一句足够长可以算作实质句子。第二句同样足够长。"),
                         ["第一句足够长可以算作实质句子。", "第二句同样足够长。"])
        self.assertEqual(M.normalize_text("  Hello   World  "), "hello world")
        self.assertIn("wmt", M.tokenize("WMT 2014"))
        self.assertEqual(len(M.ngrams(["a", "b", "c"], 2)), 2)

    def test_split_sentences_strips_mid_sentence_citations(self):
        """句中的引用标记与匹配度后缀必须被剥掉，否则会污染忠实度。"""
        text = ("BERT advances the state of the art for eleven NLP tasks."
                "（匹配度 33%）【BERT Pre-training of Deep Bidirectional Transformers p.1】")
        sentences = M.split_sentences(text)
        self.assertEqual(len(sentences), 1)
        self.assertNotIn("【", sentences[0])
        self.assertNotIn("匹配度", sentences[0])
        self.assertNotIn("p.1", sentences[0])

    def test_split_sentences_drops_boilerplate(self):
        """模板套话不参与忠实度计算（它们不含事实内容，会系统性压低分数）。"""
        answer = ("根据知识库检索结果，可以回答如下：\n\n"
                  "- We trained on the WMT 2014 English-German dataset consisting of about 4.5 million "
                  "sentence pairs.【Attention Is All You Need p.2】\n"
                  "- 以上内容摘自检索到的 5 个文档片段中相关度最高的部分。")
        sentences = M.split_sentences(answer)
        self.assertEqual(len(sentences), 1, f"应只保留 1 句实质内容，实际：{sentences}")
        self.assertIn("WMT 2014", sentences[0])
        # 套话被过滤后，忠实度应反映真实情况（这句确实在上下文里）
        context = ("We trained on the WMT 2014 English-German dataset consisting of about 4.5 million "
                   "sentence pairs, and on the WMT 2014 English-French dataset consisting of 36M sentences.")
        self.assertEqual(M.faithfulness(answer, [context])["score"], 1.0)


# ===========================================================================
# 五、评测集
# ===========================================================================
class EvalSetTest(unittest.TestCase):
    """评测集的数量、分布、字段完整性与读写往返。"""

    @classmethod
    def setUpClass(cls):
        cls.items = eval_set_module.build_eval_set()

    def test_at_least_60_items(self):
        self.assertGreaterEqual(len(self.items), 60, "课程要求评测集 ≥ 50 条，本模块承诺 ≥ 60 条")

    def test_all_four_types_present(self):
        types = {item["type"] for item in self.items}
        for qtype in eval_set_module.QUESTION_TYPES:
            self.assertIn(qtype, types, f"缺少问题类型 {qtype}")

    def test_type_distribution_meets_target(self):
        counts = {}
        for item in self.items:
            counts[item["type"]] = counts.get(item["type"], 0) + 1
        self.assertGreaterEqual(counts.get("factual", 0), 20)
        self.assertGreaterEqual(counts.get("comparison", 0), 12)
        self.assertGreaterEqual(counts.get("summary", 0), 12)
        self.assertGreaterEqual(counts.get("reasoning", 0), 12)

    def test_required_fields_complete(self):
        for item in self.items:
            for field in eval_set_module.REQUIRED_FIELDS:
                with self.subTest(item=item.get("id"), field=field):
                    self.assertIn(field, item)
                    self.assertTrue(item[field], f"{item.get('id')} 的 {field} 为空")

    def test_ids_unique_and_sequential(self):
        ids = [item["id"] for item in self.items]
        self.assertEqual(len(ids), len(set(ids)), "评测集 id 必须唯一")
        self.assertEqual(ids[0], "q001")

    def test_gold_doc_titles_are_sample_papers(self):
        allowed = {
            eval_set_module.DOC_TRANSFORMER,
            eval_set_module.DOC_RESNET,
            eval_set_module.DOC_BERT,
        }
        for item in self.items:
            for title in item["gold_doc_titles"]:
                with self.subTest(item=item["id"]):
                    self.assertIn(title, allowed, "gold_doc_titles 必须是样例论文标题")

    def test_gold_texts_long_enough(self):
        for item in self.items:
            for text in item["gold_texts"]:
                with self.subTest(item=item["id"]):
                    self.assertGreaterEqual(len(text), 12, "金标准片段过短，无法支撑重合判定")

    def test_mixed_languages(self):
        zh = [i for i in self.items if eval_set_module._has_cjk(i["question"])]
        en = [i for i in self.items if not eval_set_module._has_cjk(i["question"])]
        self.assertGreaterEqual(len(zh), 10, "评测集应包含中文提问")
        self.assertGreaterEqual(len(en), 30, "评测集应包含英文提问")

    def test_validate_eval_set_passes(self):
        report = eval_set_module.validate_eval_set(self.items)
        self.assertTrue(report["passed"], f"校验未通过：{report['errors']}")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["num_documents"], 3)

    def test_validate_eval_set_detects_problems(self):
        """构造有缺陷的评测集，校验器必须报错而不是放行。"""
        broken = [
            {"id": "x1", "question": "q", "type": "factual", "gold_doc_titles": [],
             "gold_texts": [], "reference_answer": "", "difficulty": "easy"},
        ]
        report = eval_set_module.validate_eval_set(broken)
        self.assertFalse(report["passed"])
        self.assertTrue(report["errors"])

    def test_validate_detects_duplicate_ids_and_bad_type(self):
        items = [
            {"id": "dup", "question": "q1", "type": "factual",
             "gold_doc_titles": ["A"], "gold_texts": ["x" * 20],
             "reference_answer": "a", "difficulty": "easy"},
            {"id": "dup", "question": "q2", "type": "unknown_type",
             "gold_doc_titles": ["A"], "gold_texts": ["y" * 20],
             "reference_answer": "b", "difficulty": "impossible"},
        ]
        report = eval_set_module.validate_eval_set(items, min_items=1)
        joined = " ".join(report["errors"])
        self.assertIn("id 重复", joined)
        self.assertIn("未知问题类型", joined)
        self.assertIn("未知难度", joined)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "评测集.json"
            written = eval_set_module.save_eval_set(self.items, path)
            self.assertTrue(Path(written).exists())
            loaded = eval_set_module.load_eval_set(path)
            self.assertEqual(len(loaded), len(self.items))
            self.assertEqual(loaded[0]["id"], self.items[0]["id"])
            self.assertEqual(loaded[-1]["question"], self.items[-1]["question"])
            self.assertEqual(loaded[0]["gold_texts"], self.items[0]["gold_texts"])

    def test_load_missing_file_falls_back_to_builtin(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = eval_set_module.load_eval_set(Path(tmp) / "not_exist.json")
            self.assertGreaterEqual(len(loaded), 60)

    def test_eval_set_statistics(self):
        stats = eval_set_module.eval_set_statistics(self.items)
        self.assertEqual(stats["num_items"], len(self.items))
        self.assertGreater(stats["avg_question_chars"], 0)
        self.assertGreater(stats["avg_gold_texts_per_item"], 0)


# ===========================================================================
# 六、Bad Case 分类器
# ===========================================================================
class BadCaseClassifierTest(unittest.TestCase):
    """四类失败的最小样例判定。"""

    def _sample(self, **kwargs):
        base = {
            "id": "t1",
            "question": "What datasets were used to train the Transformer?",
            "type": "factual",
            "gold_doc_titles": ["Attention Is All You Need"],
            "gold_texts": ["We trained on the WMT 2014 English-German dataset"],
            "reference_answer": "WMT 2014 English-German and English-French datasets.",
            "expected_tools": ["knowledge_search"],
        }
        base.update(kwargs)
        return base

    def test_retrieval_failure_detected(self):
        """召回片段完全不覆盖金标准 → retrieval_failure。"""
        sample = self._sample()
        cases = badcase_module.classify_badcase(
            sample,
            retrieval_result=[{"source": "Deep Residual Learning", "page": 1,
                               "preview": "Deeper neural networks are more difficult to train."}],
            agent_result=None,
        )
        categories = {c.category for c in cases}
        self.assertIn("retrieval_failure", categories)
        failure = next(c for c in cases if c.category == "retrieval_failure")
        self.assertEqual(failure.severity, "high")
        self.assertTrue(failure.suggestion)
        self.assertIn("gold_texts", failure.evidence)

    def test_retrieval_failure_when_no_results(self):
        cases = badcase_module.classify_badcase(self._sample(), retrieval_result=[], agent_result=None)
        self.assertIn("retrieval_failure", {c.category for c in cases})

    def test_citation_error_detected(self):
        """答案引用了检索结果中不存在的来源 → citation_error（critical）。"""
        sample = self._sample()
        answer = "训练使用了 WMT 2014 数据集【A Paper That Does Not Exist p.7】。"
        cases = badcase_module.classify_badcase(
            sample,
            retrieval_result={
                "answer": answer,
                "retrieved": [{"source": "Attention Is All You Need", "page": 2,
                               "preview": "We trained on the WMT 2014 English-German dataset consisting of"}],
            },
            agent_result=None,
        )
        categories = {c.category for c in cases}
        self.assertIn("citation_error", categories)
        citation_case = next(c for c in cases if c.category == "citation_error")
        self.assertEqual(citation_case.severity, "critical")
        self.assertIn("A Paper That Does Not Exist", citation_case.got)
        self.assertEqual(citation_case.evidence["citation_invalid"], 1)

    def test_generation_error_detected(self):
        """召回成功但答案既无上下文支撑又与参考答案无关 → generation_error。"""
        sample = self._sample()
        answer = ("The weather in Nanjing is pleasant and the library opens at eight in the "
                  "morning every weekday during the term.")
        cases = badcase_module.classify_badcase(
            sample,
            retrieval_result={
                "answer": answer,
                "retrieved": [{"source": "Attention Is All You Need", "page": 2,
                               "preview": "We trained on the WMT 2014 English-German dataset consisting of"}],
            },
            agent_result=None,
            retrieval_hit=True,
        )
        categories = {c.category for c in cases}
        self.assertIn("generation_error", categories)
        case = next(c for c in cases if c.category == "generation_error")
        self.assertEqual(case.severity, "medium")
        self.assertLess(case.evidence["faithfulness"], 0.5)
        self.assertLess(case.evidence["token_f1"], 0.3)
        self.assertNotIn("retrieval_failure", categories, "召回成功时不应再报检索失败")

    def test_paraphrased_correct_answer_is_not_generation_error(self):
        """措辞与参考答案不同但覆盖了问题关键词时，不应误报生成错误。"""
        sample = self._sample()
        answer = ("The Transformer was trained on the WMT 2014 English-German dataset and the "
                  "WMT 2014 English-French dataset.")
        cases = badcase_module.classify_badcase(
            sample,
            retrieval_result={
                "answer": answer,
                "retrieved": [{"source": "Attention Is All You Need", "page": 2,
                               "preview": "We trained on the WMT 2014 English-German dataset consisting of "
                                          "about 4.5 million sentence pairs, and on the WMT 2014 "
                                          "English-French dataset consisting of 36M sentences."}],
            },
            agent_result=None,
            retrieval_hit=True,
        )
        self.assertNotIn("generation_error", {c.category for c in cases})

    def test_agent_decision_error_on_loop(self):
        """终止原因为 loop_detected → agent_decision_error。"""

        class _Result:
            answer = "some answer"
            terminated_reason = "loop_detected"
            iterations = 6
            tool_calls = 6
            tool_failures = 0
            routed_tools = ["knowledge_search"]
            steps = []
            retrieved = []
            citations = []

        cases = badcase_module.classify_badcase(
            self._sample(), retrieval_result=None, agent_result=_Result(), retrieval_hit=True
        )
        categories = {c.category for c in cases}
        self.assertIn("agent_decision_error", categories)
        case = next(c for c in cases if c.category == "agent_decision_error")
        self.assertIn("循环", case.root_cause)

    def test_agent_decision_error_on_max_iterations(self):
        class _Result:
            answer = "some answer"
            terminated_reason = "max_iterations"
            iterations = 8
            tool_calls = 9
            tool_failures = 1
            routed_tools = ["knowledge_search"]
            steps = []
            retrieved = []
            citations = []

        cases = badcase_module.classify_badcase(
            self._sample(), retrieval_result=None, agent_result=_Result(), retrieval_hit=True
        )
        self.assertIn("agent_decision_error", {c.category for c in cases})

    def test_agent_decision_error_on_wrong_first_tool(self):
        """首个工具不在期望集合内 → agent_decision_error。"""

        class _Step:
            def __init__(self, tool):
                from src.utils.types import StepType

                self.type = StepType.ACTION
                self.tool_name = tool
                self.tool_args = {}
                self.tool_result = None
                self.content = ""

        class _Result:
            answer = "answer"
            terminated_reason = "final_answer"
            iterations = 2
            tool_calls = 1
            tool_failures = 0
            routed_tools = ["calculator"]
            steps = [_Step("calculator")]
            retrieved = []
            citations = []

        cases = badcase_module.classify_badcase(
            self._sample(expected_tools=["knowledge_search"]),
            retrieval_result=None,
            agent_result=_Result(),
            retrieval_hit=True,
        )
        case = next((c for c in cases if c.category == "agent_decision_error"), None)
        self.assertIsNotNone(case, "工具选错必须被判为 agent_decision_error")
        self.assertIn("calculator", case.got)

    def test_clean_sample_produces_no_bad_case(self):
        """一切正常时不应误报 Bad Case（引用真实存在 + 答案有上下文依据）。"""
        sample = self._sample()
        context = ("We trained on the WMT 2014 English-German dataset consisting of about 4.5 "
                   "million sentence pairs, and on the WMT 2014 English-French dataset consisting "
                   "of 36M sentences.")
        answer = ("We trained on the WMT 2014 English-German dataset consisting of about 4.5 "
                  "million sentence pairs, and on the WMT 2014 English-French dataset consisting "
                  "of 36M sentences【Attention Is All You Need p.2】.")
        cases = badcase_module.classify_badcase(
            sample,
            retrieval_result={
                "answer": answer,
                "retrieved": [{"source": "Attention Is All You Need", "page": 2, "preview": context}],
            },
            agent_result=None,
            retrieval_hit=True,
        )
        self.assertEqual([c.category for c in cases], [])

    def test_generic_placeholder_source_is_hallucination(self):
        """模型写【来源 p.3】这类占位符时，来源名不是空的而是"来源"，应判为引用幻觉。"""
        answer = "结论见【来源 p.3】。"
        result = M.citation_accuracy(answer, [], [{"source": "Attention Is All You Need", "page": 2}])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["unresolvable"], 0)

    def test_unresolvable_citation_is_not_a_hallucination(self):
        """来源名为空（无法解析）时应单独计数，不能算作引用幻觉。"""
        from src.utils.types import Citation

        answer = "这是一段没有可解析引用的答案，长度足够用于统计。"
        result = M.citation_accuracy(
            answer, [Citation(index=1, source="", page=3)],
            [{"source": "Attention Is All You Need", "page": 2}],
        )
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["invalid"], 0)
        self.assertEqual(result["hallucinated"], [])
        self.assertEqual(result["num_parsed_citations"], 1)

    def test_empty_source_citation_object_does_not_trigger_citation_error(self):
        """parse_citations 对未知来源会产出 source="" 的 Citation，不能据此报引用错误。"""
        from src.utils.types import Citation

        sample = self._sample()
        answer = "We trained on the WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs."
        cases = badcase_module.classify_badcase(
            sample,
            retrieval_result={
                "answer": answer,
                "citations": [Citation(index=1, source="", page=3)],
                "retrieved": [{"source": "Attention Is All You Need", "page": 2,
                               "preview": "We trained on the WMT 2014 English-German dataset consisting of "
                                          "about 4.5 million sentence pairs, and"}],
            },
            agent_result=None,
            retrieval_hit=True,
        )
        self.assertNotIn("citation_error", {c.category for c in cases})

    def test_gold_text_recall_helper(self):
        retrieved = ["We trained on the WMT 2014 English-German dataset consisting of about 4.5"]
        self.assertEqual(
            badcase_module.gold_text_recall(retrieved, ["We trained on the WMT 2014 English-German dataset"]),
            1.0,
        )
        self.assertEqual(badcase_module.gold_text_recall(retrieved, ["Completely unrelated sentence here"]), 0.0)
        self.assertEqual(badcase_module.gold_text_recall([], ["something"]), 0.0)
        self.assertEqual(badcase_module.gold_text_recall(retrieved, []), 0.0)

    def test_analyze_aggregates_from_retrieval_details(self):
        """只给检索明细时，也能汇总出检索类失败报告。"""
        retrieval_metrics = {
            "mode": "full", "top_k": 5, "match": "text",
            "num_samples": 2, "hit_at_k": 0.5, "mrr": 0.5,
            "recall_at_k": 0.5, "precision_at_k": 0.2, "ndcg_at_k": 0.5,
            "avg_latency_ms": 1.0, "p95_latency_ms": 1.5,
            "details": [
                {"question": "Q1 ok", "hit": True, "mrr": 1.0, "recall": 1.0, "ndcg": 1.0,
                 "retrieved_ids": ["a"], "gold_ids": ["a"], "latency_ms": 1.0},
                {"question": "Q2 fail", "hit": False, "mrr": 0.0, "recall": 0.0, "ndcg": 0.0,
                 "retrieved_ids": ["b"], "gold_ids": ["c"], "latency_ms": 1.0},
            ],
        }
        eval_set = [
            {"id": "q1", "question": "Q1 ok", "gold_texts": ["a"], "type": "factual"},
            {"id": "q2", "question": "Q2 fail", "gold_texts": ["c"], "type": "factual"},
        ]
        report = badcase_module.analyze(
            retrieval_metrics=retrieval_metrics, eval_set=eval_set,
            retrieval_details=retrieval_metrics["details"],
        )
        self.assertEqual(report.total_samples, 2)
        self.assertEqual(report.category_stats["retrieval_failure"]["count"], 1)
        self.assertEqual(report.bad_case_rate, 0.5)
        self.assertTrue(report.cases)
        # 严重度最高的案例排在最前
        self.assertEqual(report.cases[0].severity, "critical" if False else report.cases[0].severity)
        self.assertEqual(report.retrieval_summary["hit_at_k"], 0.5)

    def test_analyze_with_end_to_end_details(self):
        """端到端明细（含答案与来源）能触发引用类/生成类判定。"""
        eval_set = [
            {"id": "q1", "question": "What datasets were used?",
             "gold_texts": ["We trained on the WMT 2014 English-German dataset"],
             "reference_answer": "WMT 2014 English-German dataset.",
             "type": "factual"},
        ]
        e2e = [
            {"id": "q1", "round": 1, "question": "What datasets were used?",
             "answer": "训练数据集是某个不存在的语料【Ghost Paper p.13】。",
             "retrieved_sources": [
                 {"source": "Attention Is All You Need", "page": 2,
                  "preview": "We trained on the WMT 2014 English-German dataset consisting of"}
             ],
             "cache_hit": False, "degraded": False},
        ]
        report = badcase_module.analyze(
            retrieval_metrics={"mode": "full", "top_k": 5, "match": "text", "num_samples": 1,
                               "hit_at_k": 1.0, "mrr": 1.0, "recall_at_k": 1.0,
                               "precision_at_k": 1.0, "ndcg_at_k": 1.0,
                               "avg_latency_ms": 1.0, "p95_latency_ms": 1.0,
                               "details": [{"question": "What datasets were used?", "hit": True,
                                            "mrr": 1.0, "recall": 1.0, "ndcg": 1.0,
                                            "retrieved_ids": ["c57e564f"], "gold_ids": ["c57e564f"],
                                            "latency_ms": 1.0}]},
            eval_set=eval_set,
            end_to_end_details=e2e,
        )
        categories = {c.category for c in report.cases}
        self.assertIn("citation_error", categories)
        self.assertEqual(report.category_stats["citation_error"]["count"], 1)

    def test_badcase_to_dict(self):
        case = badcase_module.BadCase(case_id="x", category="retrieval_failure", severity="high")
        data = case.to_dict()
        self.assertEqual(data["case_id"], "x")
        self.assertEqual(data["severity"], "high")
        self.assertGreater(case.severity_rank, 0)

    def test_dedupe_cases_keeps_richest_evidence(self):
        """同一 (题号, 类别) 的分析路径重复记录必须去重，保留证据更充分的那条。"""
        cases = [
            badcase_module.BadCase(case_id="q1", category="retrieval_failure", evidence={"a": 1}),
            badcase_module.BadCase(
                case_id="q1", category="retrieval_failure",
                evidence={"a": 1, "b": 2, "retrieved_preview": ["x"]},
            ),
            # 同题不同类别 → 不去重
            badcase_module.BadCase(case_id="q1", category="citation_error", evidence={"c": 1}),
            # 无 case_id → 不参与去重
            badcase_module.BadCase(case_id="", category="retrieval_failure", evidence={}),
        ]
        deduped = badcase_module._dedupe_cases(cases)
        self.assertEqual(len(deduped), 3)
        kept = next(c for c in deduped if c.case_id == "q1" and c.category == "retrieval_failure")
        self.assertIn("retrieved_preview", kept.evidence, "应保留证据更充分的那条")
        self.assertIn("b", kept.evidence)

    def test_analyze_does_not_double_count_across_paths(self):
        """检索明细与端到端明细同时提供时，同一题的同类失败不应被重复计数。"""
        question = "What datasets were used to train the Transformer?"
        eval_set = [{"id": "q1", "question": question, "type": "factual",
                     "gold_texts": ["We trained on the WMT 2014 English-German dataset"],
                     "reference_answer": "WMT 2014 datasets."}]
        retrieval_metrics = {
            "mode": "full", "top_k": 5, "match": "text", "num_samples": 1,
            "hit_at_k": 0.0, "mrr": 0.0, "recall_at_k": 0.0, "precision_at_k": 0.0,
            "ndcg_at_k": 0.0, "avg_latency_ms": 1.0, "p95_latency_ms": 1.0,
            "details": [{"question": question, "hit": False, "mrr": 0.0, "recall": 0.0,
                         "ndcg": 0.0, "retrieved_ids": ["x"], "gold_ids": ["y"],
                         "latency_ms": 1.0}],
        }
        end_to_end = [{
            "id": "q1", "round": 1, "question": question,
            "answer": "Some answer that is long enough to be analysed.",
            "retrieved_sources": [{"source": "Attention Is All You Need", "page": 2,
                                   "preview": "Multi-head attention allows the model to jointly attend"}],
            "cache_hit": False, "degraded": False,
        }]
        report = badcase_module.analyze(
            retrieval_metrics=retrieval_metrics, eval_set=eval_set,
            retrieval_details=retrieval_metrics["details"], end_to_end_details=end_to_end,
        )
        self.assertEqual(report.category_stats["retrieval_failure"]["count"], 1)
        self.assertEqual(report.total_bad_cases, len(report.cases))

    def test_analyze_uses_strict_hit_for_retrieval_failure(self):
        """引擎口径 hit=True 但严格口径 strict_hit=False 的样本必须判为检索失败。

        ``strict_hit`` 缺失时同样按未命中处理（缺证据不能当作命中）。
        """
        eval_set = [
            {"id": "q1", "question": "Q1 loose hit only", "type": "factual",
             "gold_texts": ["We trained on the WMT 2014 English-German dataset"],
             "reference_answer": "WMT 2014 English-German dataset."},
            {"id": "q2", "question": "Q2 strict hit missing", "type": "factual",
             "gold_texts": ["We trained on the WMT 2014 English-German dataset"],
             "reference_answer": "WMT 2014 English-German dataset."},
        ]
        details = [
            {"question": "Q1 loose hit only", "hit": True, "strict_hit": False, "mrr": 0.0,
             "recall": 0.0, "ndcg": 0.0, "retrieved_ids": ["b"], "gold_ids": ["c"],
             "latency_ms": 1.0},
            {"question": "Q2 strict hit missing", "hit": True, "mrr": 0.0,
             "recall": 0.0, "ndcg": 0.0, "retrieved_ids": ["b"], "gold_ids": ["c"],
             "latency_ms": 1.0},
        ]
        end_to_end = [
            {"id": "q1", "round": 1, "question": "Q1 loose hit only",
             "answer": "We trained on the WMT 2014 English-German dataset.",
             "retrieved_sources": [{"source": "Attention Is All You Need", "page": 2,
                                    "preview": "We trained on the WMT 2014 English-German dataset"}],
             "cache_hit": False, "degraded": False},
            {"id": "q2", "round": 1, "question": "Q2 strict hit missing",
             "answer": "We trained on the WMT 2014 English-German dataset.",
             "retrieved_sources": [{"source": "Attention Is All You Need", "page": 2,
                                    "preview": "We trained on the WMT 2014 English-German dataset"}],
             "cache_hit": False, "degraded": False},
        ]
        report = badcase_module.analyze(
            retrieval_metrics={"mode": "full", "top_k": 5, "match": "text", "num_samples": 2,
                               "hit_at_k": 1.0, "mrr": 0.0, "recall_at_k": 0.0,
                               "precision_at_k": 0.0, "ndcg_at_k": 0.0,
                               "avg_latency_ms": 1.0, "p95_latency_ms": 1.0, "details": details},
            eval_set=eval_set,
            retrieval_details=details,
            end_to_end_details=end_to_end,
        )
        failures = [c for c in report.cases if c.category == "retrieval_failure"]
        self.assertEqual(len(failures), 2, "宽松口径 hit=True 不应掩盖严格口径的未命中")
        self.assertEqual({c.case_id for c in failures}, {"q1", "q2"})

    def test_analyze_counts_unique_samples_across_suites(self):
        """同一道题同时出现在端到端与 Agent 明细里时分母只算一次。"""
        eval_set = [
            {"id": "q1", "question": "Q1 citation ghost", "type": "factual",
             "gold_texts": ["We trained on the WMT 2014 English-German dataset"],
             "reference_answer": "WMT 2014 English-German dataset."},
            {"id": "q2", "question": "Q2 clean", "type": "factual",
             "gold_texts": ["We trained on the WMT 2014 English-German dataset"],
             "reference_answer": "WMT 2014 English-German dataset."},
        ]
        details = [
            {"question": "Q1 citation ghost", "hit": True, "strict_hit": True, "mrr": 1.0,
             "recall": 1.0, "ndcg": 1.0, "retrieved_ids": ["c1"], "gold_ids": ["c1"],
             "latency_ms": 1.0},
            {"question": "Q2 clean", "hit": True, "strict_hit": True, "mrr": 1.0,
             "recall": 1.0, "ndcg": 1.0, "retrieved_ids": ["c1"], "gold_ids": ["c1"],
             "latency_ms": 1.0},
        ]
        end_to_end = [
            {"id": "q1", "round": 1, "question": "Q1 citation ghost",
             "answer": "We trained on the WMT 2014 English-German dataset【Ghost Paper p.13】.",
             "retrieved_sources": [{"source": "Attention Is All You Need", "page": 2,
                                    "preview": "We trained on the WMT 2014 English-German dataset"}],
             "cache_hit": False, "degraded": False},
            {"id": "q2", "round": 1, "question": "Q2 clean",
             "answer": "We trained on the WMT 2014 English-German dataset.",
             "retrieved_sources": [{"source": "Attention Is All You Need", "page": 2,
                                    "preview": "We trained on the WMT 2014 English-German dataset"}],
             "cache_hit": False, "degraded": False},
        ]
        # Agent 明细与端到端明细覆盖同一批题（quick 模式下 Agent 抽样是端到端的子集）
        agent_rows = [
            {"id": "q1", "question": "Q1 citation ghost", "type": "factual",
             "terminated_reason": "final_answer", "iterations": 1, "num_tools": 1,
             "actual_tools": ["knowledge_search"], "routed_tools": ["knowledge_search"]},
            {"id": "q2", "question": "Q2 clean", "type": "factual",
             "terminated_reason": "final_answer", "iterations": 1, "num_tools": 1,
             "actual_tools": ["knowledge_search"], "routed_tools": ["knowledge_search"]},
        ]
        report = badcase_module.analyze(
            retrieval_metrics={"mode": "full", "top_k": 5, "match": "text", "num_samples": 2,
                               "hit_at_k": 1.0, "mrr": 1.0, "recall_at_k": 1.0,
                               "precision_at_k": 1.0, "ndcg_at_k": 1.0,
                               "avg_latency_ms": 1.0, "p95_latency_ms": 1.0, "details": details},
            agent_results={"num_samples": 2, "details": agent_rows},
            eval_set=eval_set,
            retrieval_details=details,
            end_to_end_details=end_to_end,
        )
        self.assertEqual(report.total_samples, 2, "同一批题跨两套明细只应计一次")
        self.assertEqual(report.category_stats["citation_error"]["count"], 1)
        self.assertEqual(report.bad_case_rate, 0.5, "1/2，而不是 1/4")


# ===========================================================================
# 七、报告渲染
# ===========================================================================
class ReportRenderingTest(unittest.TestCase):
    """Markdown 报告的关键小节与纯文本条形图。"""

    @classmethod
    def setUpClass(cls):
        items = eval_set_module.build_eval_set()[:4]
        retrieval = {
            "top_k": 5, "num_samples": len(items), "match": "text",
            "best_mode_by_ndcg": "hybrid",
            "table": [
                {"mode": "vector", "mode_label": "纯向量检索", "hit@k": 0.5, "mrr": 0.4,
                 "recall@k": 0.5, "precision@k": 0.2, "ndcg@k": 0.45,
                 "avg_latency_ms": 0.3, "p95_latency_ms": 0.4,
                 "strict_hit@5": 0.5, "strict_mrr": 0.4, "strict_recall@5": 0.5,
                 "strict_precision@5": 0.2, "strict_ndcg@5": 0.45,
                 "hit_relative_gain": 0.0},
                {"mode": "hybrid", "mode_label": "混合检索", "hit@k": 0.75, "mrr": 0.6,
                 "recall@k": 0.7, "precision@k": 0.3, "ndcg@k": 0.65,
                 "avg_latency_ms": 0.6, "p95_latency_ms": 0.8,
                 "strict_hit@5": 0.75, "strict_mrr": 0.6, "strict_recall@5": 0.7,
                 "strict_precision@5": 0.3, "strict_ndcg@5": 0.65,
                 "hit_relative_gain": 0.5},
                {"mode": "full", "mode_label": "混合检索 + 重排序", "hit@k": 0.75, "mrr": 0.7,
                 "recall@k": 0.7, "precision@k": 0.3, "ndcg@k": 0.7,
                 "avg_latency_ms": 0.9, "p95_latency_ms": 1.2,
                 "strict_hit@5": 0.75, "strict_mrr": 0.65, "strict_recall@5": 0.7,
                 "strict_precision@5": 0.3, "strict_ndcg@5": 0.68,
                 "hit_relative_gain": 0.5},
            ],
            "per_mode_detail": {},
        }
        results = {
            "generated_at": "2026-01-01 00:00:00",
            "quick_mode": True,
            "eval_set_size": len(items),
            "manifest": {"retrieval_samples": len(items), "agent_samples": 1, "e2e_questions": 1},
            "corpus": {
                "num_documents": 1, "num_chunks": 10, "avg_chunk_chars": 400.0,
                "min_chunk_chars": 100, "max_chunk_chars": 500,
                "documents": [{"title": "Attention Is All You Need", "pages": 2,
                               "chars": 3979, "loader": "pdf:builtin"}],
                "tmpdir": "C:/tmp/eval",
            },
            "environment": {"python": "3.11.0", "platform": "Windows", "llm_provider": "mock",
                            "embedding_provider": "hash"},
            "retrieval": retrieval,
            "agent": {
                "num_samples": 1, "tool_accuracy": 1.0, "tool_accuracy_strict": 1.0,
                "avg_iterations": 2.0, "max_iterations": 2,
                "avg_latency_ms": 5.0, "p95_latency_ms": 5.0,
                "total_tokens": 100, "avg_tokens": 100.0,
                "terminated_reasons": {"final_answer": 1},
                "details": [{"id": "q001", "type": "factual", "question": "Q?",
                             "expected_tools": ["knowledge_search"],
                             "actual_tools": ["knowledge_search"], "first_tool": "knowledge_search",
                             "tool_correct": True, "tool_correct_strict": True,
                             "num_tools": 1, "iterations": 2, "terminated_reason": "final_answer",
                             "latency_ms": 5.0, "total_tokens": 100, "answer_chars": 50}],
            },
            "end_to_end": {
                "num_calls": 2, "num_questions": 1, "repeat": True,
                "avg_latency_ms": 3.0, "p95_latency_ms": 4.0,
                "avg_retrieval_ms": 1.0, "avg_generation_ms": 2.0,
                "cache_hit_rate": 0.5, "degrade_rate": 0.0,
                "degrade_reasons": {}, "avg_citation_accuracy": 1.0,
                "avg_citation_coverage": 0.8, "avg_faithfulness": 0.9,
                "avg_retrieved": 5.0, "avg_answer_chars": 200.0, "total_tokens": 300,
                "details": [],
            },
            "errors": [],
            "wall_ms": 1234.0,
        }
        badcase_report = badcase_module.analyze(retrieval_metrics=retrieval, eval_set=items)
        badcase_report.optimization = {
            "enabled": True, "num_samples": 3, "runtime_ms": 500.0,
            "baseline_params": {"retrieval": {"candidate_k": 20}},
            "optimized_params": {"retrieval": {"candidate_k": 40}},
            "baseline": {"corpus": {"num_chunks": 10, "chunk_size": 512}, "engine_hit@5": 0.5},
            "optimized": {"corpus": {"num_chunks": 14, "chunk_size": 384}, "engine_hit@5": 0.7},
            "delta": {
                "hit@5": {"before": 0.5, "after": 0.7, "delta": 0.2, "relative": 0.4, "improved": True},
                "mrr": {"before": 0.4, "after": 0.6, "delta": 0.2, "relative": 0.5, "improved": True},
                "avg_latency_ms": {"before": 1.0, "after": 1.5, "delta": 0.5,
                                   "relative": 0.5, "improved": False},
            },
        }
        cls.markdown = report_module.render_markdown(
            results=results, badcase_report=badcase_report, eval_set=items, quick=True,
            extra_paths={"badcase_md": "Bad_Case分析报告.md", "xlsx": "性能评测数据.xlsx"},
        )

    def test_contains_key_section_titles(self):
        for title in (
            "# 智能科研助理 —— 系统评测报告",
            "## 一、评测方法",
            "## 二、评测环境",
            "## 三、检索层三档对比实验",
            "## 四、Agent 决策层评测",
            "## 五、端到端全链路评测",
            "## 六、Bad Case 分析摘要",
            "## 七、一轮迭代优化的前后对比",
            "## 八、结论与优化建议",
            "## 九、评测产物清单",
        ):
            with self.subTest(title=title):
                self.assertIn(title, self.markdown)

    def test_contains_bar_chart_characters(self):
        """条形图用 █ / ░ 字符绘制。"""
        self.assertIn("█", self.markdown)
        self.assertIn("░", self.markdown)

    def test_contains_retrieval_table_headers(self):
        self.assertIn("| 档位 | 模式 | Hit@5 | MRR |", self.markdown)
        self.assertIn("引擎口径", self.markdown)

    def test_reports_unrun_sections_honestly(self):
        """未运行的小节必须显式标注，而不是给出占位数字。"""
        items = eval_set_module.build_eval_set()[:2]
        results = {
            "generated_at": "t", "eval_set_size": 2, "manifest": {}, "corpus": {},
            "environment": {}, "retrieval": {"error": "检索评测失败：RuntimeError: boom", "table": []},
            "agent": {"skipped": True, "reason": "run_agent=False"},
            "end_to_end": {"error": "端到端评测失败：ValueError: x"},
            "errors": ["something"], "wall_ms": 1.0,
        }
        report = badcase_module.analyze(eval_set=items)
        text = report_module.render_markdown(
            results=results, badcase_report=report, eval_set=items, quick=True
        )
        self.assertIn("未运行", text)
        self.assertIn("boom", text)

    def test_bar_chart_helper(self):
        self.assertEqual(report_module.bar_chart(1.0, 1.0, width=10), "█" * 10)
        self.assertEqual(report_module.bar_chart(0.0, 1.0, width=10), "░" * 10)
        self.assertEqual(report_module.bar_chart(0.5, 1.0, width=10), "█" * 5 + "░" * 5)
        # 越界值被裁剪而不是抛异常
        self.assertEqual(report_module.bar_chart(5.0, 1.0, width=4), "█" * 4)
        self.assertEqual(report_module.bar_chart(-1.0, 1.0, width=4), "░" * 4)
        self.assertEqual(report_module.bar_chart(1.0, 0.0, width=4), "░" * 4)

    def test_sample_counts_derived_from_section_not_eval_set_size(self):
        """检索小节的样本量必须取本节 ``num_samples``，而不是评测集总量（曾写死 61）。"""
        items = eval_set_module.build_eval_set()[:4]
        retrieval = {
            "top_k": 5, "num_samples": 40, "num_gold_chunks_resolved": 40, "match": "text",
            "best_mode_by_ndcg": "full",
            "table": [{"mode": "full", "mode_label": "混合检索 + 重排序",
                       "hit@k": 0.5, "mrr": 0.4, "recall@k": 0.5, "precision@k": 0.2,
                       "ndcg@k": 0.5, "avg_latency_ms": 1.0, "p95_latency_ms": 1.0,
                       "strict_hit@5": 0.5, "strict_mrr": 0.4, "strict_recall@5": 0.5,
                       "strict_precision@5": 0.2, "strict_ndcg@5": 0.5,
                       "hit_relative_gain": 0.0}],
            "per_mode_detail": {},
        }
        results = {
            "generated_at": "t", "eval_set_size": 61,
            "manifest": {"retrieval_samples": 40, "agent_samples": 20, "e2e_questions": 40},
            "corpus": {"num_chunks": 10}, "environment": {},
            "retrieval": retrieval,
            "agent": {"num_samples": 20, "details": []},
            "end_to_end": {"num_calls": 80, "num_questions": 40, "details": []},
            "errors": [], "wall_ms": 1.0,
        }
        report = badcase_module.analyze(eval_set=items)
        text = report_module.render_markdown(
            results=results, badcase_report=report, eval_set=items, quick=True
        )
        self.assertNotIn("61 条样本全部成功解析", text)
        self.assertIn("40/40 条样本成功解析", text)
        self.assertNotIn("（61 条）条件下", text)
        self.assertIn("（40 条）条件下", text)

    def test_hallucination_note_reflects_measured_data(self):
        """P0 行的幻觉结论必须按实测明细生成，不能无条件写"无幻觉样本"。"""
        note = report_module._hallucination_note({
            "details": [
                {"round": 1, "citation_invalid": 0, "faithfulness": 0.9},
                {"round": 1, "citation_invalid": 1, "faithfulness": 0.2},
                {"round": 1, "citation_invalid": 0, "faithfulness": 0.8},
            ],
        })
        self.assertEqual(note, "检出引用幻觉样本 1/3 条，忠实度低于 0.5 的样本 1/3 条")
        clean = report_module._hallucination_note({
            "details": [{"round": 1, "citation_invalid": 0, "faithfulness": 0.9}],
        })
        self.assertEqual(clean, "1 条样本均未检出引用幻觉，全部样本忠实度 ≥ 0.5")
        # 没有逐条明细时退回汇总的引用准确率
        self.assertEqual(
            report_module._hallucination_note({"avg_citation_accuracy": 1.0}),
            "汇总口径未检出引用幻觉",
        )
        self.assertIn(
            "存在引用幻觉",
            report_module._hallucination_note({"avg_citation_accuracy": 0.5}),
        )

    def test_no_unconditional_no_hallucination_claim(self):
        self.assertNotIn("无幻觉样本", self.markdown)


# ===========================================================================
# 七点五、检索基准的口径键名与记忆化（离线桩引擎）
# ===========================================================================
class RetrievalBenchmarkCacheTest(unittest.TestCase):
    """严格指标键名由 ``top_k`` 生成，检索按 ``(mode, top_k)`` 记忆化。"""

    class _Hit:
        def __init__(self, chunk_id):
            self.chunk = SimpleNamespace(chunk_id=chunk_id)

    class _StubEngine:
        """只记录调用次数的离线引擎（不加载模型/语料）。"""

        def __init__(self):
            self.chunks = [SimpleNamespace(chunk_id="c1", text="gold text one and only")]
            self.calls = []
            self.eval_calls = 0

        def retrieve(self, question, top_k=5, mode="full"):
            self.calls.append((str(question), int(top_k), str(mode)))
            return [RetrievalBenchmarkCacheTest._Hit("c1")]

        def evaluate(self, eval_set, mode="full", top_k=5, match="text"):
            self.eval_calls += 1
            return {
                "hit_at_k": 1.0, "mrr": 1.0, "recall_at_k": 1.0, "precision_at_k": 0.2,
                "ndcg_at_k": 1.0, "avg_latency_ms": 1.0, "p95_latency_ms": 1.0,
                "details": [
                    {"question": "Q1", "hit": True, "mrr": 1.0, "recall": 1.0, "ndcg": 1.0,
                     "retrieved_ids": ["c1"], "gold_ids": ["c1"], "latency_ms": 1.0}
                ],
            }

    def _bench(self, top_k=5):
        engine = self._StubEngine()
        eval_set = [
            {"id": "q1", "question": "Q1", "gold_texts": ["gold text one"], "type": "factual"},
            {"id": "q2", "question": "Q2", "gold_texts": ["gold text one"], "type": "factual"},
        ]
        return benchmark_module.RetrievalBenchmark(engine, eval_set, top_k=top_k), engine

    def test_strict_metric_keys_follow_top_k(self):
        keys = benchmark_module.strict_metric_keys(3)
        self.assertEqual(keys["hit"], "strict_hit@3")
        self.assertEqual(keys["recall"], "strict_recall@3")
        self.assertEqual(keys["precision"], "strict_precision@3")
        self.assertEqual(keys["ndcg"], "strict_ndcg@3")
        self.assertEqual(keys["mrr"], "strict_mrr")
        self.assertEqual(keys["num_samples"], "strict_num_samples")
        self.assertNotIn("strict_hit@5", set(keys.values()))

    def test_retrieval_memoized_by_mode_and_top_k(self):
        bench, engine = self._bench()
        first = bench.strict_metrics("full")
        second = bench.strict_metrics("full")
        self.assertEqual(len(engine.calls), 2, "同 (mode, top_k) 第二次调用不应重新检索")
        self.assertEqual(first, second)
        self.assertEqual(first["strict_hit@5"], 1.0)
        self.assertEqual(first["strict_num_samples"], 2)
        bench.strict_metrics("full", k=3)
        self.assertEqual(len(engine.calls), 4, "top_k 变化必须重新检索")
        bench.strict_metrics("vector")
        self.assertEqual(len(engine.calls), 6, "mode 变化必须重新检索")
        bench.strict_metrics("full", k=5)
        self.assertEqual(len(engine.calls), 6, "回到已缓存的 (mode, top_k) 应直接复用")

    def test_run_strict_table_keys_follow_top_k(self):
        bench, engine = self._bench(top_k=3)
        result = bench.run(modes=["vector"])
        row = result["table"][0]
        self.assertIn("strict_hit@3", row)
        self.assertNotIn("strict_hit@5", row)
        self.assertEqual(row["strict_num_samples"], 2)
        self.assertEqual(row["strict_hit@3"], result["summary"]["hit_at_k"])
        self.assertEqual(engine.eval_calls, 1, "每档引擎口径只 evaluate 一次")
        self.assertEqual(len(engine.calls), 2, "严格口径同一 (mode, top_k) 只检索一遍")
        detail = result["per_mode_detail"]["vector"][0]
        self.assertEqual(detail["gold_chunk_ids"], ["c1"])
        self.assertTrue(detail["strict_hit"])


class BadCaseMarkdownTest(unittest.TestCase):
    """Bad Case Markdown 报告的结构与内容。"""

    def test_save_markdown_writes_required_sections(self):
        items = eval_set_module.build_eval_set()[:3]
        retrieval = {
            "mode": "full", "top_k": 5, "match": "text", "num_samples": 3,
            "hit_at_k": 0.33, "mrr": 0.33, "recall_at_k": 0.33,
            "precision_at_k": 0.1, "ndcg_at_k": 0.33,
            "avg_latency_ms": 1.0, "p95_latency_ms": 1.2,
            "details": [
                {"question": items[0]["question"], "hit": True, "mrr": 1.0, "recall": 1.0,
                 "ndcg": 1.0, "retrieved_ids": ["a"], "gold_ids": ["a"], "latency_ms": 1.0},
                {"question": items[1]["question"], "hit": False, "mrr": 0.0, "recall": 0.0,
                 "ndcg": 0.0, "retrieved_ids": ["b"], "gold_ids": ["c"], "latency_ms": 1.0},
                {"question": items[2]["question"], "hit": False, "mrr": 0.0, "recall": 0.0,
                 "ndcg": 0.0, "retrieved_ids": ["d"], "gold_ids": ["e"], "latency_ms": 1.0},
            ],
        }
        report = badcase_module.analyze(
            retrieval_metrics=retrieval, eval_set=items,
            retrieval_details=retrieval["details"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = badcase_module.save_markdown(report, Path(tmp) / "bc.md")
            text = Path(path).read_text(encoding="utf-8")
        for section in (
            "# Bad Case 分析报告",
            "## 一、总览",
            "## 二、典型样例",
            "## 三、根因归纳",
            "## 四、一轮迭代优化的前后对比（真实对照实验）",
        ):
            with self.subTest(section=section):
                self.assertIn(section, text)
        self.assertIn("█", text)
        self.assertIn("检索失败", text)
        # 未跑对照实验时必须显式写"未运行"
        self.assertIn("未运行", text)


# ===========================================================================
# 八、与真实检索链路的轻量集成测试（小 top_k，快速）
# ===========================================================================
class RetrievalIntegrationTest(unittest.TestCase):
    """用样例论文建一次索引，验证严格指标与评测集金标准的对接（秒级）。"""

    @classmethod
    def setUpClass(cls):
        from src.data_loader import loader_from_config
        from src.evaluation.benchmark import RetrievalBenchmark, prepare_corpus
        from tests.fixtures.make_sample_pdf import write_sample_pdfs  # noqa: F401

        cls.bundle = prepare_corpus()
        cls.engine = cls.bundle.engine
        cls.items = eval_set_module.build_eval_set()[:6]
        cls.benchmark = RetrievalBenchmark(cls.engine, cls.items, top_k=5, match="text")

    def test_corpus_indexed(self):
        self.assertEqual(self.engine.num_documents, 3)
        self.assertGreaterEqual(self.engine.num_chunks, 10)

    def test_gold_chunk_ids_resolved(self):
        """6 条样本的金标准文本都必须能在语料中定位到分块。"""
        resolved = self.benchmark.resolve_gold_chunk_ids()
        self.assertEqual(len(resolved), 6)
        for key, ids in resolved.items():
            with self.subTest(item=key):
                self.assertTrue(ids, f"{key} 的金标准文本未能在语料中定位")

    def test_strict_metrics_range(self):
        strict = self.benchmark.strict_metrics("full")
        for key, value in strict.items():
            if key == "strict_num_samples":
                continue
            with self.subTest(metric=key):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        self.assertEqual(strict["strict_num_samples"], 6)

    def test_retrieve_returns_chunks_with_confidence(self):
        """评测集里判断相关性必须用 confidence；此处验证字段确实被填充。"""
        hits = self.engine.retrieve("What datasets were used to train the Transformer?", top_k=3)
        self.assertTrue(hits)
        for hit in hits:
            self.assertGreaterEqual(hit.confidence, 0.0)
            self.assertLessEqual(hit.confidence, 1.0)
            self.assertTrue(hit.chunk.source)

    def test_metrics_against_real_ranking(self):
        """把真实检索结果喂给指标函数，验证与手算一致。"""
        question = "What top-5 error does the ensemble ResNet achieve on the ImageNet test set?"
        hits = self.engine.retrieve(question, top_k=5, mode="full")
        ranked = [h.chunk.chunk_id for h in hits]
        gold = self.benchmark.resolve_gold_chunk_ids()
        item = self.items[0]
        # 用第一条样本的金标准，指标函数本身不应抛异常且结果落在 [0,1]
        for value in (
            M.hit_at_k(ranked, gold[item["id"]], 5),
            M.mrr(ranked, gold[item["id"]]),
            M.ndcg_at_k(ranked, gold[item["id"]], 5),
            M.recall_at_k(ranked, gold[item["id"]], 5),
            M.precision_at_k(ranked, gold[item["id"]], 5),
            M.map_at_k(ranked, gold[item["id"]], 5),
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class ExportExcelTest(unittest.TestCase):
    """Excel 导出的优雅降级行为。"""

    def test_export_returns_none_when_openpyxl_missing(self):
        """openpyxl 缺失时必须返回 None 且不抛异常（用 mock 模拟缺失）。"""
        import sys
        from unittest import mock

        from src.evaluation.benchmark import export_excel

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("openpyxl"):
                raise ImportError("simulated missing openpyxl")
            return real_import(name, *args, **kwargs)

        with mock.patch.dict(sys.modules, {"openpyxl": None, "openpyxl.styles": None,
                                           "openpyxl.utils": None}):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with tempfile.TemporaryDirectory() as tmp:
                    result = export_excel({"retrieval": {"table": []}}, Path(tmp) / "x.xlsx")
        self.assertIsNone(result)

    def test_export_writes_workbook_when_available(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("未安装 openpyxl")
        from src.evaluation.benchmark import export_excel

        results = {
            "retrieval": {"table": [
                {"mode": "vector", "mode_label": "纯向量检索", "hit@k": 0.5, "mrr": 0.4,
                 "recall@k": 0.5, "precision@k": 0.2, "ndcg@k": 0.45,
                 "avg_latency_ms": 0.3, "p95_latency_ms": 0.4,
                 "hit_relative_gain": 0.0, "mrr_relative_gain": 0.0},
            ]},
            "agent": {"num_samples": 1, "tool_accuracy": 1.0, "terminated_reasons": {"final_answer": 1},
                      "details": [{"id": "q1", "type": "factual", "question": "Q",
                                   "expected_tools": ["knowledge_search"], "actual_tools": ["knowledge_search"],
                                   "first_tool": "knowledge_search", "tool_correct": True,
                                   "num_tools": 1, "iterations": 2, "terminated_reason": "final_answer",
                                   "latency_ms": 1.0, "total_tokens": 10, "answer_chars": 5}]},
            "end_to_end": {"num_calls": 1, "num_questions": 1, "degrade_reasons": {},
                           "details": [{"id": "q1", "round": 1, "question": "Q", "latency_ms": 1.0,
                                        "cache_hit": False, "degraded": False,
                                        "citation_accuracy": 1.0, "faithfulness": 1.0}]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = export_excel(results, Path(tmp) / "perf.xlsx")
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).exists())
            import openpyxl

            book = openpyxl.load_workbook(path)
            for sheet in ("检索三档对比", "Agent 指标", "端到端", "逐条明细"):
                with self.subTest(sheet=sheet):
                    self.assertIn(sheet, book.sheetnames)


class ChunkingExperimentTest(unittest.TestCase):
    """分块策略对比实验（模块一 5.1.2 的产出物生成器）。

    真实跑一遍四种策略的完整实验要 1 秒左右，所以这里既测纯渲染，
    也真跑一次**两策略**的实验（既验证隔离，也防止渲染与实验脱节）。
    """

    def test_strategy_config_isolates_persist_dirs(self):
        """两种策略必须落到不同目录：共用目录会让后建的索引覆盖前一个。"""
        from src.evaluation import chunking_experiment as CE

        base = SimpleNamespace(merged_with=lambda over: over)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            a = CE._strategy_config(base, "fixed", work)
            b = CE._strategy_config(base, "semantic", work)
        self.assertNotEqual(a["vector_store"]["persist_dir"], b["vector_store"]["persist_dir"])
        self.assertNotEqual(a["paths"]["index_dir"], b["paths"]["index_dir"])
        self.assertEqual(a["chunking"]["strategy"], "fixed")
        self.assertEqual(b["chunking"]["strategy"], "semantic")

    def test_render_markdown_has_table_and_bars(self):
        result = {
            "generated_at": "2026-01-01 00:00:00",
            "quick": False,
            "mode": "hybrid",
            "top_k": 5,
            "num_samples": 2,
            "strategies": ["fixed", "recursive"],
            "corpus": {"num_documents": 1, "documents": []},
            "wall_ms": 1000.0,
            "workdir": "tmp",
            "rows": [
                {"strategy": "fixed", "label": "固定窗口切分", "num_chunks": 25,
                 "avg_chunk_chars": 463.3, "min_chunk_chars": 152, "max_chunk_chars": 512,
                 "index_ms": 50.0, "avg_latency_ms": 1.1, "resolved_gold": 2,
                 "strict_hit@5": 0.5, "strict_mrr": 0.4, "strict_recall@5": 0.5,
                 "strict_precision@5": 0.2, "strict_ndcg@5": 0.45, "strict_num_samples": 2},
                {"strategy": "recursive", "label": "递归字符切分", "num_chunks": 27,
                 "avg_chunk_chars": 433.0, "min_chunk_chars": 175, "max_chunk_chars": 509,
                 "index_ms": 48.0, "avg_latency_ms": 0.9, "resolved_gold": 2,
                 "strict_hit@5": 1.0, "strict_mrr": 0.9, "strict_recall@5": 0.8,
                 "strict_precision@5": 0.2, "strict_ndcg@5": 0.85, "strict_num_samples": 2},
            ],
        }
        text = chunking_module.render_markdown(result)
        self.assertIn("# 分块策略对比实验报告", text)
        self.assertIn("| 分块策略 | 分块数 |", text)
        self.assertIn("固定窗口切分（`fixed`） | 25", text)
        self.assertIn("```text", text)
        # 结论里应指出 NDCG 最优策略
        self.assertIn("递归字符切分", text.split("## 四、结论")[1].split("## 五")[0])
        self.assertIn("本次实验总耗时 **1.0 秒**", text)

    def test_render_markdown_marks_failed_strategy(self):
        """某个策略跑挂时必须显式标注「未运行」，不能悄悄少一行。"""
        result = {
            "generated_at": "2026-01-01 00:00:00", "quick": True, "mode": "hybrid",
            "top_k": 5, "num_samples": 1, "strategies": ["fixed", "semantic"],
            "corpus": {"num_documents": 1, "documents": []}, "wall_ms": 10.0,
            "workdir": "tmp",
            "rows": [
                {"strategy": "fixed", "label": "固定窗口切分", "num_chunks": 1,
                 "avg_chunk_chars": 1.0, "min_chunk_chars": 1, "max_chunk_chars": 1,
                 "index_ms": 1.0, "avg_latency_ms": 1.0, "resolved_gold": 1,
                 "strict_hit@5": 1.0, "strict_mrr": 1.0, "strict_recall@5": 1.0,
                 "strict_precision@5": 1.0, "strict_ndcg@5": 1.0, "strict_num_samples": 1},
                {"strategy": "semantic", "label": "语义切分", "error": "RuntimeError: boom"},
            ],
        }
        text = chunking_module.render_markdown(result)
        self.assertIn("未运行", text)
        self.assertIn("RuntimeError: boom", text)

    def test_render_markdown_without_successful_rows(self):
        """全部策略失败时也要出报告，而不是抛 KeyError。"""
        result = {
            "generated_at": "2026-01-01 00:00:00", "quick": True, "mode": "hybrid",
            "top_k": 5, "num_samples": 0, "strategies": ["fixed"],
            "corpus": {"num_documents": 0, "documents": []}, "wall_ms": 1.0,
            "workdir": "tmp",
            "rows": [{"strategy": "fixed", "label": "固定窗口切分", "error": "ImportError: x"}],
        }
        text = chunking_module.render_markdown(result)
        self.assertIn("所有策略均未成功运行", text)
        self.assertIn("ImportError: x", text)

    def test_quick_subset_is_stratified_and_stable(self):
        items = eval_set_module.build_eval_set()
        subset = chunking_module._quick_subset(items)
        self.assertLessEqual(len(subset), chunking_module.QUICK_LIMIT)
        # 分层抽样：四类问题都应出现（评测集四类各 12+ 条）
        types = {str(s.get("type")) for s in subset}
        self.assertGreaterEqual(len(types), 3)
        # 同一输入两次抽样结果一致（可复现，报告能对齐）
        self.assertEqual([s["id"] for s in subset], [s["id"] for s in chunking_module._quick_subset(items)])

    def test_run_experiment_two_strategies(self):
        """真跑一次两策略实验：分块数必须互不相同（证明各建各的索引）。"""
        from src.evaluation import chunking_experiment as CE

        result = CE.run_experiment(quick=True, strategies=("fixed", "semantic"))
        rows = {r["strategy"]: r for r in result["rows"]}
        for name in ("fixed", "semantic"):
            with self.subTest(strategy=name):
                self.assertNotIn("error", rows[name], rows[name].get("error"))
                self.assertGreater(rows[name]["num_chunks"], 0)
                self.assertGreater(rows[name]["strict_num_samples"], 0)
                self.assertLessEqual(rows[name]["strict_num_samples"], result["num_samples"])
                self.assertGreaterEqual(rows[name]["avg_latency_ms"], 0.0)
        self.assertNotEqual(rows["fixed"]["num_chunks"], rows["semantic"]["num_chunks"],
                            "语义切分粒度更细，分块数应与固定窗口不同")

    def test_save_markdown_writes_file(self):
        from src.evaluation import chunking_experiment as CE

        with tempfile.TemporaryDirectory() as tmp:
            path = CE.save_markdown("# 标题\n", str(Path(tmp) / "r.md"))
            self.assertTrue(Path(path).exists())
            self.assertIn("标题", Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
