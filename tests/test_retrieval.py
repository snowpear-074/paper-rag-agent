# -*- coding: utf-8 -*-
"""模块一（分块 + 检索）测试：四种分块策略、向量库、BM25、RRF、重排序、检索引擎。

运行::

    python -m unittest tests.test_retrieval -v
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.common import TempWorkspace, build_test_text, ensure_sample_pdfs, sample_documents


# ===========================================================================
# 分块策略
# ===========================================================================
class FixedSizeChunkerTest(unittest.TestCase):
    """固定窗口切分：验证大小约束与重叠。"""

    def _chunker(self, size: int, overlap: int):
        from src.chunking import ChunkingConfig, FixedSizeChunker

        return FixedSizeChunker(
            ChunkingConfig(chunk_size=size, chunk_overlap=overlap, min_chunk_size=10)
        )

    def test_chunk_size_respected(self):
        from src.utils.types import DocumentMeta, LoadedDocument

        text = "A" * 1000
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        chunks = self._chunker(100, 20).split(doc)
        self.assertGreater(len(chunks), 5)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 100 + 20)

    def test_overlap_creates_context_continuity(self):
        from src.utils.types import DocumentMeta, LoadedDocument

        text = "".join(f"句子{i:03d}。" for i in range(60))
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        chunks = self._chunker(120, 40).split(doc)
        self.assertGreater(len(chunks), 1)
        # 相邻块应有重叠字符（前一块尾部出现在后一块头部）
        tail = chunks[0].text[-20:]
        self.assertIn(tail[:10], chunks[1].text)

    def test_different_sizes_produce_different_chunk_counts(self):
        from src.utils.types import DocumentMeta, LoadedDocument

        text = build_test_text() * 4
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        counts = [len(self._chunker(size, 0).split(doc)) for size in (256, 512, 1024)]
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])


class RecursiveChunkerTest(unittest.TestCase):
    """递归切分：优先在句子/段落边界断开。"""

    def test_respects_sentence_boundaries(self):
        from src.chunking import ChunkingConfig, RecursiveChunker
        from src.utils.types import DocumentMeta, LoadedDocument

        text = "第一句完整内容。第二句完整内容。第三句完整内容。第四句完整内容。" * 4
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        # 注意 chunk_size 下限为 64（见 RecursiveChunker._split_page），过小的值会被抬高
        chunker = RecursiveChunker(
            ChunkingConfig(chunk_size=64, chunk_overlap=0, min_chunk_size=1)
        )
        chunks = chunker.split(doc)
        self.assertGreater(len(chunks), 1)
        # 绝大多数块应以句末标点结束（说明没有从句中硬切）
        ended_cleanly = sum(1 for c in chunks[:-1] if c.text.rstrip().endswith(("。", "！", "？")))
        self.assertGreaterEqual(ended_cleanly, len(chunks) - 2)

    def test_chunk_ids_unique_and_indexed(self):
        from src.chunking import create_chunker

        doc = sample_documents()[0]
        chunks = create_chunker(None, strategy="recursive").split(doc)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        for chunk in chunks:
            self.assertEqual(chunk.doc_id, doc.doc_id)
            self.assertGreaterEqual(chunk.page, 1)
            self.assertEqual(chunk.strategy, "recursive")

    def test_offsets_within_page_bounds(self):
        from src.chunking import create_chunker

        doc = sample_documents()[1]
        chunks = create_chunker(None, strategy="recursive").split(doc)
        page_len = {i + 1: len(p) for i, p in enumerate(doc.pages)}
        for chunk in chunks:
            self.assertGreaterEqual(chunk.start_offset, 0)
            self.assertLessEqual(chunk.end_offset, max(1, page_len.get(chunk.page, 10**6)))


class SemanticChunkerTest(unittest.TestCase):
    """语义切分：话题切换处断开。"""

    def test_splits_at_topic_change(self):
        from src.chunking import ChunkingConfig, SemanticChunker
        from src.utils.types import DocumentMeta, LoadedDocument

        # 三个明显不同的话题，语义切分应在话题边界断开
        text = (
            "注意力机制通过查询与键的点积计算权重，用于序列建模。注意力可以捕捉长距离依赖关系。"
            "卷积神经网络在图像分类任务上表现优异，通过局部感受野提取空间特征。池化层可以降低特征图分辨率。"
            "强化学习通过与环境交互获得奖励信号，使用策略梯度方法优化决策。价值函数估计未来累积回报。"
        )
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        chunker = SemanticChunker(
            ChunkingConfig(chunk_size=400, min_chunk_size=10, semantic_threshold=0.35,
                           semantic_max_sentences=3)
        )
        chunks = chunker.split(doc)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertTrue(chunk.text.strip())

    def test_lexical_fallback_similarity(self):
        from src.chunking import SemanticChunker

        same = SemanticChunker._lexical_similarity("注意力机制很重要", "注意力机制真的很重要")
        different = SemanticChunker._lexical_similarity("注意力机制很重要", "今天天气非常好")
        self.assertGreater(same, different)
        self.assertEqual(SemanticChunker._lexical_similarity("", "abc"), 0.0)


class PaperChunkerTest(unittest.TestCase):
    """论文专用策略：参考文献剔除、标题吸附、跨页合并。"""

    def test_removes_reference_section(self):
        from src.chunking import ChunkingConfig, PaperChunker
        from src.utils.types import DocumentMeta, LoadedDocument

        text = (
            "## 1 方法\n本文提出一种新方法。\n\n"
            "## 2 实验\n实验结果表明有效。\n\n"
            "## References\n"
            "[1] Author A. Some Paper. 2019.\n"
            "[2] Author B. Another Paper. 2020.\n"
            "[3] Author C. Third Paper. 2021.\n"
        )
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        chunker = PaperChunker(
            ChunkingConfig(chunk_size=200, chunk_overlap=0, min_chunk_size=1, remove_references=True)
        )
        combined = "\n".join(c.text for c in chunker.split(doc))
        self.assertNotIn("Author C", combined)
        self.assertIn("本文提出一种新方法", combined)

    def test_keeps_references_when_disabled(self):
        from src.chunking import ChunkingConfig, PaperChunker
        from src.utils.types import DocumentMeta, LoadedDocument

        text = "## 方法\n内容。\n\n## References\n[1] Author A. Paper. 2019.\n[2] Author B. Paper. 2020.\n"
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        chunker = PaperChunker(
            ChunkingConfig(chunk_size=200, chunk_overlap=0, min_chunk_size=1, remove_references=False)
        )
        combined = "\n".join(c.text for c in chunker.split(doc))
        self.assertIn("Author A", combined)

    def test_formula_lines_are_not_broken(self):
        from src.chunking import ChunkingConfig, PaperChunker
        from src.utils.types import DocumentMeta, LoadedDocument

        formula = "L = 1/N * sum(y_i - y_hat_i)^2 + lambda * ||w||^2"
        text = f"前文说明。\n{formula}\n后文说明。" * 3
        doc = LoadedDocument(doc_id="d", text=text, pages=[text], meta=DocumentMeta(doc_id="d"))
        chunker = PaperChunker(
            ChunkingConfig(chunk_size=120, chunk_overlap=0, min_chunk_size=1,
                           keep_formula_lines=True, merge_cross_page=False)
        )
        combined = "\n".join(c.text for c in chunker.split(doc))
        self.assertIn(formula, combined)

    def test_chunk_document_facade(self):
        from src.chunking import chunk_document

        doc = sample_documents()[0]
        for strategy in ("fixed", "recursive", "semantic", "paper"):
            with self.subTest(strategy=strategy):
                chunks = chunk_document(doc, strategy=strategy)
                self.assertGreater(len(chunks), 0)
                self.assertTrue(all(c.strategy == strategy for c in chunks))


class ChunkerFactoryTest(unittest.TestCase):
    """工厂的签名与容错。"""

    def test_create_chunker_strategies(self):
        from src.chunking import CHUNKER_REGISTRY, create_chunker

        for name in CHUNKER_REGISTRY:
            chunker = create_chunker(None, strategy=name)
            self.assertEqual(chunker.name, name)

    def test_unknown_strategy_falls_back(self):
        from src.chunking import RecursiveChunker, create_chunker

        self.assertIsInstance(create_chunker(None, strategy="not_a_strategy"), RecursiveChunker)

    def test_positional_string_raises_helpful_error(self):
        """create_chunker("paper") 这种老式调用应给出明确指引，而不是静默出错。"""
        from src.chunking import create_chunker

        with self.assertRaises(TypeError) as ctx:
            create_chunker("paper")  # type: ignore[arg-type]
        self.assertIn("strategy=", str(ctx.exception))

    def test_overrides_apply(self):
        from src.chunking import create_chunker

        chunker = create_chunker(None, strategy="fixed", chunk_size=128, chunk_overlap=16)
        self.assertEqual(chunker.config.chunk_size, 128)
        self.assertEqual(chunker.config.chunk_overlap, 16)


# ===========================================================================
# 向量库
# ===========================================================================
class VectorStoreTest(unittest.TestCase):
    """NumPy 向量库：增删改查、持久化、增量更新、幂等。"""

    def _store(self, dimension: int = 8):
        from src.retrieval import NumpyVectorStore

        with TempWorkspace() as ws:
            store = NumpyVectorStore(dimension=dimension, persist_dir=ws.dir("vs"),
                                     collection="t", auto_load=False)
            return store, ws

    def test_add_and_search(self):
        from src.utils.types import Chunk

        store, ws = self._store()
        try:
            chunks = [Chunk(doc_id="d1", text=f"文本{i}", index=i) for i in range(3)]
            vectors = np.eye(3, 8, dtype=np.float32)
            self.assertEqual(store.add(chunks, vectors), 3)
            self.assertEqual(store.count(), 3)

            hits = store.search(vectors[1], top_k=1)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].chunk.text, "文本1")
            self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        finally:
            ws.cleanup()

    def test_add_is_idempotent_on_same_chunk_id(self):
        from src.utils.types import Chunk

        store, ws = self._store()
        try:
            chunk = Chunk(doc_id="d1", text="同样的内容", index=0)
            store.add([chunk], np.ones((1, 8), dtype=np.float32))
            store.add([chunk], np.ones((1, 8), dtype=np.float32))
            self.assertEqual(store.count(), 1, "相同 chunk_id 重复写入不应产生重复行")
        finally:
            ws.cleanup()

    def test_incremental_add_keeps_existing(self):
        """增量更新：新文档加入不应影响已有向量。"""
        from src.utils.types import Chunk

        store, ws = self._store()
        try:
            store.add([Chunk(doc_id="d1", text="第一批", index=0)], np.eye(1, 8, dtype=np.float32))
            store.add([Chunk(doc_id="d2", text="第二批", index=0)], np.eye(1, 8, dtype=np.float32))
            self.assertEqual(store.count(), 2)
            self.assertEqual(sorted(store.doc_ids()), ["d1", "d2"])
        finally:
            ws.cleanup()

    def test_delete_by_doc_id(self):
        from src.utils.types import Chunk

        store, ws = self._store()
        try:
            store.add(
                [Chunk(doc_id="d1", text="a", index=0), Chunk(doc_id="d1", text="b", index=1),
                 Chunk(doc_id="d2", text="c", index=0)],
                np.eye(3, 8, dtype=np.float32),
            )
            self.assertEqual(store.delete("d1"), 2)
            self.assertEqual(store.count(), 1)
            self.assertEqual(store.doc_ids(), ["d2"])
            self.assertEqual(store.delete("no_such_doc"), 0)
        finally:
            ws.cleanup()

    def test_persist_and_reload(self):
        from src.utils.types import Chunk

        with TempWorkspace() as ws:
            from src.retrieval import NumpyVectorStore

            directory = ws.dir("vs")
            store = NumpyVectorStore(dimension=8, persist_dir=directory, collection="t", auto_load=False)
            store.add(
                [Chunk(doc_id="d1", text="持久化测试", index=0)],
                np.eye(1, 8, dtype=np.float32),
            )
            store.persist()

            reloaded = NumpyVectorStore(dimension=8, persist_dir=directory, collection="t", auto_load=True)
            self.assertEqual(reloaded.count(), 1)
            self.assertEqual(reloaded.all_chunks()[0].text, "持久化测试")
            self.assertEqual(reloaded.search(np.eye(1, 8, dtype=np.float32)[0], top_k=1)[0].score, 1.0)

    def test_dimension_mismatch_raises(self):
        from src.utils.types import Chunk

        store, ws = self._store(dimension=8)
        try:
            with self.assertRaises(ValueError):
                store.add([Chunk(doc_id="d", text="t", index=0)], np.ones((1, 4), dtype=np.float32))
        finally:
            ws.cleanup()

    def test_stats_and_health(self):
        store, ws = self._store()
        try:
            stats = store.stats()
            self.assertEqual(stats["backend"], "numpy")
            self.assertEqual(stats["dimension"], 8)
            self.assertTrue(store.health_check()["ok"])
        finally:
            ws.cleanup()


# ===========================================================================
# BM25
# ===========================================================================
class BM25Test(unittest.TestCase):
    """BM25 打分：公式正确性、排序合理性、中文支持、可解释性。"""

    def _corpus(self):
        return [
            ["attention", "mechanism", "transformer"],
            ["residual", "learning", "image"],
            ["bert", "pretraining", "masked"],
            ["attention", "transformer", "multi", "head"],
        ]

    def test_scores_are_nonnegative_and_rank_relevant_higher(self):
        from src.retrieval import BM25Okapi

        bm25 = BM25Okapi(self._corpus())
        scores = bm25.get_scores(["attention", "transformer"])
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.all(scores >= 0))
        # 文档 0 与 3 都含两个查询词，应高于只含一个的文档 1/2
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[3], scores[2])

    def test_unknown_terms_yield_zero(self):
        from src.retrieval import BM25Okapi

        bm25 = BM25Okapi(self._corpus())
        scores = bm25.get_scores(["nonexistent"])
        self.assertTrue(np.allclose(scores, 0.0))

    def test_top_n_ordering_and_cutoff(self):
        from src.retrieval import BM25Okapi

        bm25 = BM25Okapi(self._corpus())
        top = bm25.get_top_n(["attention"], n=2)
        self.assertEqual(len(top), 2)
        self.assertGreaterEqual(top[0][1], top[1][1])
        # 只含 "attention" 的文档 0 排第一
        self.assertEqual(top[0][0], 0)

    def test_idf_penalises_common_terms(self):
        from src.retrieval import BM25Okapi

        corpus = [["common", f"u{i}"] for i in range(10)] + [["rare", "u0"]]
        bm25 = BM25Okapi(corpus)
        self.assertGreater(bm25.idf[bm25.vocab["rare"]], bm25.idf[bm25.vocab["common"]])

    def test_empty_query(self):
        from src.retrieval import BM25Okapi

        bm25 = BM25Okapi(self._corpus())
        self.assertTrue(np.allclose(bm25.get_scores([]), 0.0))
        self.assertEqual(bm25.get_top_n([], n=3), [])

    def test_retriever_with_chinese_text(self):
        from src.retrieval import BM25Retriever
        from src.utils.types import Chunk

        retriever = BM25Retriever()
        chunks = [
            Chunk(doc_id="d", text="本文提出一种注意力机制用于机器翻译任务", index=0),
            Chunk(doc_id="d", text="本文研究图像分类中的残差网络结构", index=1),
            Chunk(doc_id="d", text="强化学习通过奖励信号优化策略", index=2),
        ]
        retriever.index(chunks)
        hits = retriever.search("注意力机制 机器翻译", top_k=2)
        self.assertGreater(len(hits), 0)
        self.assertIn("注意力", hits[0].text)
        self.assertGreater(hits[0].bm25_score, 0)
        self.assertEqual(hits[0].source_route, "bm25")

    def test_retriever_incremental_and_remove(self):
        from src.retrieval import BM25Retriever
        from src.utils.types import Chunk

        retriever = BM25Retriever()
        retriever.index([Chunk(doc_id="d1", text="注意力机制", index=0)])
        self.assertEqual(retriever.size, 1)
        retriever.add([Chunk(doc_id="d2", text="卷积神经网络", index=0)])
        self.assertEqual(retriever.size, 2)
        self.assertEqual(retriever.remove("d1"), 1)
        self.assertEqual(retriever.size, 1)
        self.assertEqual(retriever.search("注意力机制", top_k=5), [])

    def test_term_contributions_explainable(self):
        from src.retrieval import BM25Retriever
        from src.utils.types import Chunk

        retriever = BM25Retriever()
        chunk = Chunk(doc_id="d", text="attention mechanism in transformer models", index=0)
        retriever.index([chunk, Chunk(doc_id="d2", text="residual learning", index=0)])
        contributions = retriever.explain("attention transformer", chunk.chunk_id)
        self.assertIn("attention", contributions)
        self.assertGreater(contributions["attention"], 0)

    def test_stats(self):
        from src.retrieval import BM25Retriever
        from src.utils.types import Chunk

        retriever = BM25Retriever()
        # 用真实词（而非单字符 "a b c" —— 单字符会被分词器过滤掉，词表为空）
        retriever.index([Chunk(doc_id="d", text="attention mechanism transformer", index=0)])
        stats = retriever.stats()
        self.assertEqual(stats["num_chunks"], 1)
        self.assertGreater(stats["vocab_size"], 0)
        self.assertIn(stats["backend"], ("scipy", "builtin-csr"))
        self.assertEqual(stats["k1"], 1.5)
        self.assertEqual(stats["b"], 0.75)


# ===========================================================================
# RRF 融合
# ===========================================================================
class RRFFusionTest(unittest.TestCase):
    """倒数排名融合：排名交换性、权重、多路合并。"""

    def test_rrf_prefers_items_ranked_high_in_both_lists(self):
        from src.retrieval import rrf_fuse

        # a 在两路都第一；b/c 各只在一路靠前
        vector = ["a", "b", "x"]
        bm25 = ["a", "c", "y"]
        fused = rrf_fuse([("vector", vector), ("bm25", bm25)], k=60)
        self.assertEqual(fused[0][0], "a")
        self.assertAlmostEqual(fused[0][1], 2 / 61, places=8)
        self.assertEqual(fused[0][2], {"vector": 1, "bm25": 1})

    def test_rrf_score_formula(self):
        from src.retrieval import rrf_fuse

        fused = rrf_fuse([("v", ["only"])], k=60)
        self.assertAlmostEqual(fused[0][1], 1 / 61, places=10)

    def test_rrf_weights(self):
        from src.retrieval import rrf_fuse

        fused = rrf_fuse([("v", ["a"]), ("b", ["z"])], k=60, weights=[1.0, 0.0])
        self.assertEqual([item[0] for item in fused], ["a"])

    def test_rrf_empty_inputs(self):
        from src.retrieval import rrf_fuse

        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_fuse([("v", [])]), [])

    def test_rrf_accepts_objects_with_chunk_id(self):
        from src.retrieval import rrf_fuse
        from src.utils.types import Chunk, RetrievedChunk

        hits = [RetrievedChunk(chunk=Chunk(doc_id="d", text="t", index=0))]
        fused = rrf_fuse([("v", hits)], k=60)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0][0], hits[0].chunk.chunk_id)

    def test_rrf_k_affects_relative_gap(self):
        from src.retrieval import rrf_fuse

        small_k = rrf_fuse([("v", ["first", "second"])], k=1)
        large_k = rrf_fuse([("v", ["first", "second"])], k=1000)
        gap_small = small_k[0][1] - small_k[1][1]
        gap_large = large_k[0][1] - large_k[1][1]
        self.assertGreater(gap_small, gap_large)


# ===========================================================================
# 重排序
# ===========================================================================
class RerankerTest(unittest.TestCase):
    """启发式重排序：特征、置信度门控、可解释性。"""

    def _candidates(self):
        from src.utils.types import Chunk, RetrievedChunk

        texts = [
            "The Transformer uses multi-head attention and trains on WMT 2014 dataset.",
            "今天天气不错，适合出门散步。",
            "Weather forecast says it will rain tomorrow.",
        ]
        return [
            RetrievedChunk(chunk=Chunk(doc_id=f"d{i}", text=text, index=i), score=1.0 - i * 0.1)
            for i, text in enumerate(texts)
        ]

    def test_relevant_document_scores_higher(self):
        from src.retrieval.reranker import HeuristicReranker

        reranker = HeuristicReranker(top_n=3)
        hits = reranker.rerank("What dataset does the Transformer train on?", self._candidates())
        self.assertIn("WMT 2014", hits[0].text)
        self.assertGreater(hits[0].confidence, hits[-1].confidence)

    def test_confidence_is_gated_to_zero_without_term_overlap(self):
        """无词项交集时必须给出接近 0 的置信度，否则降级判断会失效。"""
        from src.retrieval.reranker import HeuristicReranker
        from src.utils.types import Chunk, RetrievedChunk

        reranker = HeuristicReranker(top_n=1)
        hits = reranker.rerank(
            "lattice quantum chromodynamics",
            [RetrievedChunk(chunk=Chunk(doc_id="d", text="The cat sat on the mat peacefully.", index=0))],
        )
        self.assertLess(hits[0].confidence, 0.05)

    def test_score_and_confidence_are_separate(self):
        """rerank 后 score 是归一化相对分（Top-1=1），confidence 保留绝对量。"""
        from src.retrieval.reranker import HeuristicReranker

        reranker = HeuristicReranker(top_n=3)
        hits = reranker.rerank("attention dataset", self._candidates())
        self.assertEqual(hits[0].score, 1.0)
        self.assertLessEqual(hits[0].confidence, 1.0)
        self.assertGreater(hits[0].confidence, 0.0)

    def test_explain_returns_feature_breakdown(self):
        from src.retrieval.reranker import HeuristicReranker

        reranker = HeuristicReranker(top_n=3)
        rows = reranker.explain("attention dataset", self._candidates())
        self.assertEqual(len(rows), 3)
        self.assertIn("coverage", rows[0]["features"])
        self.assertIn("gate", rows[0])
        self.assertIn("confidence", rows[0])

    def test_fit_computes_idf(self):
        from src.retrieval.reranker import HeuristicReranker
        from src.utils.types import Chunk

        reranker = HeuristicReranker(top_n=1)
        reranker.fit([Chunk(doc_id="d", text="attention " * 5, index=0)])
        self.assertGreater(len(reranker._idf), 0)

    def test_reranker_factory_none_disables(self):
        from src.retrieval.reranker import create_reranker

        with TempWorkspace() as ws:
            cfg = ws.config()
            self.assertIsNone(create_reranker(cfg, provider="none"))

    def test_reranker_factory_default_is_heuristic(self):
        from src.retrieval.reranker import HeuristicReranker, create_reranker

        with TempWorkspace() as ws:
            reranker = create_reranker(ws.config())
            self.assertIsInstance(reranker, HeuristicReranker)

    def test_empty_candidates(self):
        from src.retrieval.reranker import HeuristicReranker

        self.assertEqual(HeuristicReranker().rerank("q", []), [])


# ===========================================================================
# 检索引擎（集成）
# ===========================================================================
class RetrievalEngineTest(unittest.TestCase):
    """引擎装配、索引、三档检索、评估、增量与删除。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = sample_documents()

    def _engine(self, ws, **chunking):
        from src.retrieval import create_retrieval_engine

        cfg = ws.config(chunking=chunking or {"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50})
        return create_retrieval_engine(cfg), cfg

    def test_index_and_retrieve_three_modes(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            report = engine.index_documents(self.documents)
            self.assertEqual(report["succeeded"], 3)
            self.assertGreater(report["num_chunks"], 5)
            self.assertEqual(engine.num_documents, 3)

            query = "How does residual learning address the degradation problem?"
            for mode in ("vector", "hybrid", "full"):
                with self.subTest(mode=mode):
                    hits = engine.retrieve(query, top_k=3, mode=mode)
                    self.assertEqual(len(hits), 3)
                    for hit in hits:
                        self.assertGreaterEqual(hit.score, 0.0)
                        self.assertGreaterEqual(hit.confidence, 0.0)
                    self.assertTrue(any("Residual" in h.chunk.source for h in hits))

    def test_top_k_is_honored_when_corpus_larger_than_k(self):
        """语料足够时，三档检索都必须返回**恰好** top_k 条。

        回归测试：早期实现的收尾过滤写成 ``r.score > 0``，而 score 是
        min-max 归一化的相对分（本批最差恒为 0.0），于是所有"并列最低分"的
        候选被整批删除 —— 20 个片段、top_k=20 时只返回 2 条，且并列项在
        启发式重排序下很常见。这会让 top_k 形同虚设，并连带污染 Hit@K/MRR/Recall。
        """
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws, strategy="fixed", chunk_size=120, chunk_overlap=0)
            engine.index_documents(self.documents)
            self.assertGreaterEqual(engine.num_chunks, 20, "语料需大于 top_k 才有意义")

            for mode in ("vector", "hybrid", "full"):
                with self.subTest(mode=mode):
                    hits = engine.retrieve("attention dataset training", top_k=20, mode=mode)
                    self.assertEqual(len(hits), 20)

    def test_min_score_filters_on_absolute_confidence(self):
        """min_score 作用在 confidence 上：设成 1.0 时不应有结果通过。"""
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            engine.retriever.min_score = 1.0
            hits = engine.retrieve("attention dataset training", top_k=5, mode="full")
            self.assertEqual(hits, [])
            # 复位后必须恢复 —— 确认过滤不是"把结果的 confidence 清零"这类副作用
            engine.retriever.min_score = 0.0
            self.assertEqual(len(engine.retrieve("attention dataset training", top_k=5, mode="full")), 5)

    def test_fusion_preserves_both_route_scores(self):
        """被两路同时召回的片段必须同时带上两路分数。

        回归：早期 ``_fuse`` 用 ``setdefault`` 只保留向量路的对象，于是
        "BM25 排第 1"的片段 bm25_score 仍是 0.0 —— 可解释性面板与 Bad Case
        归因据此会得出"这条是纯向量召回的"错误结论。
        """
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            hits = engine.retrieve("ImageNet top-5 error residual learning", top_k=10, mode="hybrid")
            both = [h for h in hits if h.vector_rank > 0 and h.bm25_rank > 0]
            self.assertTrue(both, "该查询应命中被两路共同召回的片段")
            for hit in both:
                self.assertGreater(hit.vector_score, 0.0)
                self.assertGreater(hit.bm25_score, 0.0)

    def test_hybrid_beats_vector_on_term_specific_query(self):
        """含专有名词的查询上，混合检索应优于纯向量（这是引入 BM25 的初衷）。"""
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            query = "What BLEU score did the Transformer achieve on WMT 2014?"
            vector_top = engine.retrieve(query, top_k=3, mode="vector")
            hybrid_top = engine.retrieve(query, top_k=3, mode="hybrid")
            # 混合检索的 Top-1 应来自 Transformer 论文
            self.assertIn("Attention", hybrid_top[0].chunk.source)
            # 且相关度不低于纯向量
            self.assertGreaterEqual(hybrid_top[0].confidence, vector_top[0].confidence - 0.05)

    def test_retrieval_trace_recorded(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            engine.retrieve("attention mechanism", top_k=3, mode="full")
            trace = engine.retriever.last_trace
            self.assertIsNotNone(trace)
            self.assertEqual(trace.num_returned, 3)
            self.assertGreater(trace.total_ms, 0)
            self.assertTrue(trace.retrieved_chunk_ids)

    def test_incremental_index_and_delete(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_document(self.documents[0])
            first_count = engine.num_chunks
            self.assertGreater(first_count, 0)

            engine.index_document(self.documents[1])
            self.assertGreater(engine.num_chunks, first_count)

            removed = engine.delete_document(self.documents[0].doc_id)
            self.assertGreater(removed, 0)
            self.assertEqual(engine.num_documents, 1)
            # 删除后不应再检索到被删文档
            hits = engine.retrieve("Attention Is All You Need", top_k=5, mode="full")
            self.assertTrue(all(h.chunk.doc_id != self.documents[0].doc_id for h in hits))

    def test_reindex_same_document_is_idempotent(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_document(self.documents[0])
            count_once = engine.num_chunks
            engine.index_document(self.documents[0])  # 重复索引同一篇
            self.assertEqual(engine.num_chunks, count_once, "重复索引不应产生重复分块")
            self.assertEqual(engine.num_documents, 1)

    def test_persist_and_reload_engine(self):
        from src.retrieval import create_retrieval_engine

        with TempWorkspace() as ws:
            cfg = ws.config(chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50})
            engine = create_retrieval_engine(cfg)
            engine.index_documents(self.documents)
            count = engine.num_chunks
            del engine

            reloaded = create_retrieval_engine(cfg)
            self.assertEqual(reloaded.num_chunks, count)
            self.assertEqual(reloaded.num_documents, 3)
            self.assertGreater(len(reloaded.retrieve("attention", top_k=3)), 0)

    def test_empty_knowledge_base_returns_no_hits(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            self.assertEqual(engine.retrieve("anything", top_k=3), [])
            self.assertEqual(engine.num_chunks, 0)

    def test_evaluate_with_text_match(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            eval_set = [
                {
                    "question": "What datasets were used to train the Transformer?",
                    "gold_texts": ["WMT 2014 English-German dataset"],
                },
                {
                    "question": "What is masked language model?",
                    "gold_texts": ["masked language model"],
                },
            ]
            metrics = engine.evaluate(eval_set, mode="full", top_k=5, match="text")
            self.assertEqual(metrics["num_samples"], 2)
            for key in ("hit_at_k", "mrr", "recall_at_k", "precision_at_k", "ndcg_at_k"):
                self.assertIn(key, metrics)
                self.assertGreaterEqual(metrics[key], 0.0)
                self.assertLessEqual(metrics[key], 1.0)
            self.assertEqual(len(metrics["details"]), 2)

    def test_compare_modes_returns_gain_columns(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            eval_set = [
                {"question": "What BLEU score did the Transformer achieve?",
                 "gold_texts": ["28.4 BLEU"]},
                {"question": "How many layers does ResNet-152 have?",
                 "gold_texts": ["152-layer"]},
            ]
            result = engine.compare_modes(eval_set, top_k=5, match="text")
            self.assertEqual(len(result["table"]), 3)
            modes = [row["mode"] for row in result["table"]]
            self.assertEqual(modes, ["vector", "hybrid", "full"])
            for row in result["table"]:
                self.assertIn("hit_gain_vs_vector", row)
                self.assertIn("latency_ratio", row)

    def test_retrieve_with_context_payload(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            payload = engine.retrieve_with_context("attention", top_k=3)
            self.assertEqual(payload["num"], 3)
            self.assertIn("trace", payload)
            for item in payload["results"]:
                for key in ("index", "chunk_id", "doc_id", "source", "page", "score",
                            "confidence", "route", "preview"):
                    self.assertIn(key, item)

    def test_engine_stats_and_health(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            stats = engine.stats()
            self.assertEqual(stats["num_documents"], 3)
            self.assertIn("vector_store", stats)
            self.assertIn("bm25", stats)
            self.assertTrue(engine.health_check()["ok"])

    def test_clear_removes_everything(self):
        with TempWorkspace() as ws:
            engine, _cfg = self._engine(ws)
            engine.index_documents(self.documents)
            engine.clear()
            self.assertEqual(engine.num_chunks, 0)
            self.assertEqual(engine.num_documents, 0)
            self.assertEqual(engine.retrieve("attention", top_k=3), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
