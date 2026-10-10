# -*- coding: utf-8 -*-
"""模块一（分块策略）测试：fixed / recursive / semantic / paper 四种切分策略。

覆盖重点：

1. 各策略的块数与大小约束、重叠、空输入/超短文本/超长文本等边界情况；
2. **偏移不变式**——每个块的 ``page_text[start_offset:end_offset]``
   必须等于 ``chunk.text``（纯切片块严格相等，同页合并块按空白归一化相等）。
   这是"引用溯源能定位到原文"的前提，旧实现用 ``" ".join(...)`` 重建文本后
   反查偏移，换行被空格替换导致 2/2、3/3 的块区间失配；
3. PaperChunker 的断裂修复规则：只合并**同页**相邻块（跨页一律不合并），
   页码与偏移必须自洽；全角右括号/右引号视为句末标点；
4. 确定性：同一输入两次切分结果完全一致。

全部用例离线、确定性：Embedding 用固定桩（不加载真实模型、不联网），
词面相似度回退走本地 jieba。

运行::

    python -m unittest tests.test_chunking -v
"""

from __future__ import annotations

import unittest
from typing import List, Optional, Sequence

import numpy as np

from tests.common import build_test_text


# ===========================================================================
# 测试桩与公共工具
# ===========================================================================
class _RaisingEmbedder:
    """编码即抛错：迫使语义分块走词面相似度回退（离线、零依赖）。"""

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        raise RuntimeError("测试环境禁用真实 Embedding 模型")


class _KeywordEmbedder:
    """按主题词计数构造确定性向量（不依赖模型/网络）。

    同一主题的句子向量同向、不同主题正交，因此"话题切换"位置明确可控，
    用于验证语义分块确实在话题边界断开。
    """

    TOPICS = (
        ("注意力", "查询", "键", "序列"),
        ("卷积", "图像", "特征"),
        ("强化", "奖励", "策略"),
    )

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.TOPICS)), dtype=np.float32)
        for row, text in enumerate(texts):
            for col, words in enumerate(self.TOPICS):
                matrix[row, col] = float(sum(text.count(word) for word in words))
            norm = float(np.linalg.norm(matrix[row]))
            if norm:
                matrix[row] /= norm
        return matrix


def _doc(*pages: str, doc_id: str = "doc1"):
    """用给定页文本构造 LoadedDocument（内联 fixture，避免读盘）。"""
    from src.utils.types import DocumentMeta, LoadedDocument

    return LoadedDocument(
        doc_id=doc_id, text="\n".join(pages), pages=list(pages), meta=DocumentMeta(doc_id=doc_id)
    )


def _chunker(strategy: str, embedder: Optional[object] = None, **overrides):
    """按策略名构造分块器；缺省参数保证测试快、离线。"""
    from src.chunking import (
        ChunkingConfig,
        FixedSizeChunker,
        PaperChunker,
        RecursiveChunker,
        SemanticChunker,
    )

    params = {"chunk_size": 200, "chunk_overlap": 0, "min_chunk_size": 20}
    params.update(overrides)
    config = ChunkingConfig(**params)
    if strategy == "fixed":
        return FixedSizeChunker(config)
    if strategy == "recursive":
        return RecursiveChunker(config)
    if strategy == "semantic":
        return SemanticChunker(config, embedder=embedder if embedder is not None else _RaisingEmbedder())
    if strategy == "paper":
        return PaperChunker(config)
    raise ValueError(f"未知策略: {strategy}")


def _assert_offsets(test: unittest.TestCase, chunks: Sequence[object], pages: Sequence[str]) -> None:
    """核心不变式：块的偏移必须指向该页原文的真实区间。

    - 普通块：``page_text[start:end] == chunk.text``（严格相等）；
    - 同页合并块（metadata 里有 ``merged_chunks``）：文本由两段原文去掉
      首尾空白后用空格拼接，偏移是覆盖两者的最小连续区间，
      因此按空白归一化后应当一致。
    """
    from src.utils.text import normalize_text

    for chunk in chunks:
        msg = f"chunk#{chunk.index} p{chunk.page} [{chunk.start_offset}:{chunk.end_offset}]"
        source = normalize_text(pages[chunk.page - 1])
        test.assertGreaterEqual(chunk.start_offset, 0, msg)
        test.assertLessEqual(chunk.start_offset, chunk.end_offset, msg)
        test.assertLessEqual(chunk.end_offset, len(source), msg)
        expected = source[chunk.start_offset : chunk.end_offset]
        if "merged_chunks" in chunk.metadata:
            test.assertEqual(" ".join(expected.split()), " ".join(chunk.text.split()), msg)
        else:
            test.assertEqual(expected, chunk.text, msg)


def _snapshot(chunks: Sequence[object]) -> List[tuple]:
    """把块序列压缩为可比较的快照（用于确定性断言）。"""
    return [(c.page, c.start_offset, c.end_offset, c.text) for c in chunks]


def _topic_text() -> str:
    """三个话题、每话题三句，话题之间相似度骤降，用于语义切分。"""
    return (
        "注意力机制通过查询与键的点积计算权重，用于序列建模任务。查询与键的匹配程度决定了序列中各位置的权重分配。\n"
        "自注意力把同一个序列同时当作查询、键和值，便于建模长距离依赖关系。注意力权重可以对序列中的每个位置进行加权。\n"
        "多头注意力把表示空间划分为多个子空间并行计算，最后拼接各个子空间的结果。每个查询都会与所有键进行匹配。\n"
        "卷积神经网络通过局部感受野提取图像的空间特征，适合处理二维图像数据。卷积核在图像上滑动计算特征图。\n"
        "池化层可以降低特征图的分辨率，从而减少图像特征的计算量。卷积核参数在整张图像上共享，参数量较小。\n"
        "残差连接让梯度直接回传到浅层，缓解了图像分类网络的训练困难。图像特征经过多层卷积后语义更加抽象。\n"
        "强化学习通过与环境交互获得奖励信号，使用策略梯度方法优化决策。奖励信号的期望值决定策略更新方向。\n"
        "价值函数估计未来累积奖励，用于降低策略梯度估计的方差。策略在探索与利用之间取得平衡。\n"
    )


def _attention_paragraph() -> str:
    """单一话题、多句、句间只有换行——最容易触发"拼接后偏移失配"的形态。"""
    return (
        "注意力机制通过查询与键的点积计算权重。\n"
        "查询与键的匹配程度决定了各个位置的权重分配。\n"
        "自注意力把同一个序列同时当作查询键和值。\n"
        "注意力权重可以对序列中的每个位置进行加权。\n"
        "多头注意力把表示空间划分为多个子空间并行计算。\n"
    )


def _sample_paper_pages() -> List[str]:
    """三页论文样例：含标题、公式行、全角标点与双栏式断行。"""
    page1 = (
        "# 基于自注意力的文本表示方法\n\n"
        "摘要：本文提出一种结合自注意力与位置编码的文本表示方法。\n"
        "方法在多个基准数据集上取得了一致的提升。\n\n"
        "## 1 引言\n\n"
        "近年来，基于 Transformer 的模型在自然语言处理任务中被广泛采用（Vaswani 等，2017）。\n"
        "当输入序列较长时，自注意力的计算开销会显著增加；许多工作因此尝试引入稀疏化或低秩近似。\n\n"
        "## 2 方法\n\n"
        "给定输入序列 X，我们首先把它映射为查询、键与值三个矩阵（见式 1）。\n"
        "L = softmax(QK^T / sqrt(d_k)) V\n\n"
        "其中 d_k 表示键向量的维度；除以 sqrt(d_k) 是为了稳定梯度。\n"
    )
    page2 = (
        "## 3 实验\n\n"
        "我们在 WMT 2014 英德翻译任务上进行了实验。\n"
        "训练使用 8 块 GPU，批大小为 256，共训练 10 万步。\n\n"
        "结果显示，本方法比基线提升 1.8 BLEU（p < 0.05）。\n"
        "值得注意的是，增益在长句上更为明显。\n"
    )
    page3 = (
        "## 4 结论\n\n"
        "本文提出的方法在长文本建模上表现稳定，\n"
        "后续工作将探索更高效的注意力变体（例如线性注意力）。\n"
        "我们相信，该方向对实际系统落地具有重要价值。\n"
    )
    return [page1, page2, page3]


# ===========================================================================
# 策略一：固定大小切分
# ===========================================================================
class FixedSizeChunkerTest(unittest.TestCase):
    """固定窗口：大小上限、重叠与边界输入。"""

    def test_chunk_size_is_respected(self):
        text = "A" * 1000
        chunks = _chunker("fixed", chunk_size=100).split(_doc(text))
        self.assertGreater(len(chunks), 5)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 100)

    def test_overlap_repeats_context(self):
        text = "".join(f"句子{i:03d}。" for i in range(80))
        chunks = _chunker("fixed", chunk_size=120, chunk_overlap=40).split(_doc(text))
        self.assertGreater(len(chunks), 1)
        # 相邻块共享一段上下文
        self.assertIn(chunks[0].text[-10:], chunks[1].text)
        # 偏移同样体现重叠：后一块起点落在前一块结束之前
        self.assertLess(chunks[1].start_offset, chunks[0].end_offset)

    def test_short_text_is_one_chunk(self):
        text = "这是一段远短于 chunk_size 的文本。"
        chunks = _chunker("fixed", chunk_size=500).split(_doc(text))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_empty_and_whitespace_pages_are_skipped(self):
        self.assertEqual(_chunker("fixed").split(_doc("", "   \n\t  ", "　　")), [])

    def test_offsets_are_exact_on_tricky_text(self):
        text = (
            "方法部分采用双栏排版，本行会在栏底被人为\n"
            "折断，换行并不代表句子结束。\n"
            "公式行：A = softmax(QK^T / sqrt(d_k))\n"
            "其中全角括号（）、引号“示例”、书名号《论文》都需原样保留；\n"
            "还有断词 inter-\nnational 与制表符\t混排。\n"
        )
        chunks = _chunker("fixed", chunk_size=48, min_chunk_size=8).split(_doc(text))
        self.assertGreater(len(chunks), 2)
        _assert_offsets(self, chunks, [text])

    def test_larger_size_produces_fewer_chunks(self):
        text = build_test_text() * 3
        counts = [len(_chunker("fixed", chunk_size=size).split(_doc(text))) for size in (128, 256, 512)]
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])

    def test_chunk_ids_unique_and_indices_continuous(self):
        chunks = _chunker("fixed", chunk_size=150).split(_doc(build_test_text() * 2))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(c.strategy == "fixed" for c in chunks))


# ===========================================================================
# 策略二：递归字符切分
# ===========================================================================
class RecursiveChunkerTest(unittest.TestCase):
    """递归切分：优先在句子/段落边界断开，且偏移保持精确。"""

    def test_most_chunks_end_at_sentence_boundary(self):
        text = "第一句完整内容。第二句完整内容。第三句完整内容。第四句完整内容。" * 6
        chunks = _chunker("recursive", chunk_size=64).split(_doc(text))
        self.assertGreater(len(chunks), 1)
        ended_cleanly = sum(1 for c in chunks[:-1] if c.text.rstrip().endswith(("。", "！", "？")))
        self.assertGreaterEqual(ended_cleanly, len(chunks) - 2)

    def test_offsets_are_exact_with_overlap(self):
        """带重叠的块以"上一块尾部 + 新内容"开头，最容易定位错位。"""
        text = build_test_text() * 2
        chunks = _chunker("recursive", chunk_size=128, chunk_overlap=32).split(_doc(text))
        self.assertGreater(len(chunks), 2)
        _assert_offsets(self, chunks, [text])

    def test_non_whitespace_chars_are_fully_covered(self):
        """切分只允许丢弃空白：每个非空白字符都必须落在某个块的区间内。"""
        from src.utils.text import normalize_text

        text = build_test_text() * 3
        chunks = _chunker("recursive", chunk_size=128).split(_doc(text))
        source = normalize_text(text)
        covered = bytearray(len(source))
        for chunk in chunks:
            for pos in range(chunk.start_offset, chunk.end_offset):
                covered[pos] = 1
        missing = [pos for pos, char in enumerate(source) if not char.isspace() and not covered[pos]]
        self.assertEqual(missing, [], f"{len(missing)} 个非空白字符未被任何块覆盖")

    def test_size_limit_with_documented_slack(self):
        text = "".join(f"这是第{i:03d}个用于压测的句子，内容长度适中。" for i in range(120))
        chunks = _chunker("recursive", chunk_size=128).split(_doc(text))
        self.assertGreater(len(chunks), 5)
        for chunk in chunks:
            # 1.5 倍是 _split_oversized 对"单句超长"情形的宽容上限
            self.assertLessEqual(len(chunk.text), int(128 * 1.5), chunk.text[:30])
        _assert_offsets(self, chunks, [text])

    def test_short_text_and_empty_input(self):
        self.assertEqual(_chunker("recursive").split(_doc("")), [])
        text = "很短的一句话。"
        chunks = _chunker("recursive", chunk_size=500).split(_doc(text))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)


# ===========================================================================
# 策略三：语义切分
# ===========================================================================
class SemanticChunkerTest(unittest.TestCase):
    """语义切分：话题边界断开、拼接块偏移精确、无重复末块。"""

    def _semantic(self, **overrides):
        params = {"chunk_size": 400, "semantic_threshold": 0.5, "semantic_max_sentences": 3}
        params.update(overrides)
        return _chunker("semantic", embedder=_KeywordEmbedder(), **params)

    def test_topic_change_produces_separate_chunks(self):
        text = _topic_text()
        chunks = self._semantic().split(_doc(text))
        self.assertGreaterEqual(len(chunks), 3)
        for keyword in ("注意力", "卷积", "强化"):
            self.assertTrue(any(keyword in c.text for c in chunks), f"话题 {keyword} 丢失")
        _assert_offsets(self, chunks, [text])

    def test_joined_chunks_offsets_are_exact(self):
        """多句拼接的块是旧实现的重灾区（" ".join 把换行换成空格后定位失败）。"""
        text = _attention_paragraph()
        chunks = self._semantic().split(_doc(text))
        self.assertGreaterEqual(len(chunks), 2)
        # 必须真的产生了"多句拼接"的块，否则本用例测不到目标路径
        self.assertTrue(any("\n" in c.text for c in chunks))
        _assert_offsets(self, chunks, [text])

    def test_last_chunk_is_not_duplicated(self):
        text = _attention_paragraph()
        chunks = self._semantic().split(_doc(text))
        texts = [c.text for c in chunks]
        self.assertEqual(len(set(texts)), len(texts), "末尾句被重复产出了一个块")

    def test_lexical_fallback_when_embedder_unavailable(self):
        text = _topic_text()
        chunks = _chunker("semantic", embedder=_RaisingEmbedder(), chunk_size=300,
                          semantic_max_sentences=3).split(_doc(text))
        self.assertGreater(len(chunks), 0)
        _assert_offsets(self, chunks, [text])

    def test_few_sentences_returns_whole_page(self):
        from src.utils.text import normalize_text

        text = "只有一句话，而且长度足够，单独构成一个语义块。"
        chunks = self._semantic().split(_doc(text))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, normalize_text(text))

    def test_empty_input_yields_no_chunks(self):
        self.assertEqual(self._semantic().split(_doc("   \n  ")), [])


# ===========================================================================
# 策略四：论文专用
# ===========================================================================
class PaperChunkerTest(unittest.TestCase):
    """论文策略：公式保护、断裂修复（仅同页）、偏移精确。"""

    def test_formula_line_is_kept_intact(self):
        formula = "L = softmax(QK^T / sqrt(d_k)) V"
        text = f"前文说明。\n{formula}\n后文说明。\n" * 4
        chunks = _chunker("paper", chunk_size=60, min_chunk_size=5, merge_cross_page=False).split(_doc(text))
        combined = "\n".join(c.text for c in chunks)
        self.assertIn(formula, combined)
        self.assertFalse(any("\x00" in c.text for c in chunks), "公式行掩码字符不应出现在输出中")
        _assert_offsets(self, chunks, [text])

    def test_offsets_are_exact_on_sample_paper(self):
        pages = _sample_paper_pages()
        chunks = _chunker("paper", chunk_size=180, chunk_overlap=20).split(_doc(*pages))
        self.assertGreater(len(chunks), len(pages))
        _assert_offsets(self, chunks, pages)

    def test_reference_section_is_removed_but_offsets_stay_valid(self):
        pages = [
            _sample_paper_pages()[0],
            "## 参考文献\n[1] 作者甲. 论文一. 2019.\n[2] 作者乙. 论文二. 2020.\n[3] 作者丙. 论文三. 2021.\n",
        ]
        chunks = _chunker("paper", chunk_size=200, remove_references=True).split(_doc(*pages))
        combined = "\n".join(c.text for c in chunks)
        self.assertNotIn("作者丙", combined)
        _assert_offsets(self, chunks, pages)

    def test_cross_page_break_is_not_merged(self):
        """上一页末尾句子未结束、下一页以小写/中文续写，也**不得**跨页合并。"""
        page1 = (
            "## 1 方法\n本文提出一种基于图神经网络的知识表示方法，通过对实体与关系的联合编码"
            "获得结构化的语义表示，并在大规模知识图谱上验证了其有效性。实验说明该表示"
        )
        page2 = "方法能够显著提升下游任务的准确率。具体而言，我们在链接预测与实体分类两个任务上进行了评测。\n"
        chunks = _chunker("paper", chunk_size=120, merge_cross_page=True).split(_doc(page1, page2))
        self.assertEqual([c.page for c in chunks], [1, 2])
        _assert_offsets(self, chunks, [page1, page2])

    def test_fix_cross_page_breaks_skips_other_pages(self):
        from src.utils.types import Chunk

        chunker = _chunker("paper")
        first = Chunk(doc_id="d", text="实验结果说明该方法", page=1, start_offset=0, end_offset=9,
                      index=0, metadata={})
        second = Chunk(doc_id="d", text="在长文本上依然有效。", page=2, start_offset=0, end_offset=10,
                       index=1, metadata={})
        merged = chunker._fix_cross_page_breaks([first, second])
        self.assertEqual(len(merged), 2)
        self.assertEqual([c.page for c in merged], [1, 2])

    def test_fix_cross_page_breaks_merges_within_page(self):
        from src.utils.types import Chunk

        chunker = _chunker("paper")
        first = Chunk(doc_id="d", text="实验结果说明该方法", page=1, start_offset=0, end_offset=9,
                      index=0, metadata={})
        second = Chunk(doc_id="d", text="在长文本上依然有效。", page=1, start_offset=10, end_offset=20,
                       index=1, metadata={})
        merged = chunker._fix_cross_page_breaks([first, second])
        self.assertEqual(len(merged), 1)
        # 页码与偏移必须同属一页、自洽（旧实现跨页合并后 end_offset 取自下一页）
        self.assertEqual(merged[0].page, 1)
        self.assertEqual((merged[0].start_offset, merged[0].end_offset), (0, 20))
        self.assertLessEqual(merged[0].end_offset, 20)
        self.assertIn("在长文本上依然有效。", merged[0].text)
        self.assertEqual(merged[0].metadata["merged_chunks"], 2)
        self.assertEqual([c.index for c in merged], [0])

    def test_full_width_closers_block_merge(self):
        """全角右括号/右引号说明句子已完整，不应被断裂修复吞并。"""
        from src.utils.types import Chunk

        chunker = _chunker("paper")
        for closer in ("）", "”", "」"):
            with self.subTest(closer=closer):
                first = Chunk(doc_id="d", text=f"实验结果说明该方法有效（见附录{closer}", page=1,
                              start_offset=0, end_offset=14, index=0, metadata={})
                second = Chunk(doc_id="d", text="在长文本上依然有效。", page=1, start_offset=15,
                               end_offset=25, index=1, metadata={})
                merged = chunker._fix_cross_page_breaks([first, second])
                self.assertEqual(len(merged), 2)

    def test_merge_respects_size_ceiling(self):
        from src.utils.types import Chunk

        chunker = _chunker("paper", chunk_size=512, min_chunk_size=20)
        first = Chunk(doc_id="d", text="甲" * 400, page=1, start_offset=0, end_offset=400,
                      index=0, metadata={})
        second = Chunk(doc_id="d", text="乙" * 400, page=1, start_offset=401, end_offset=801,
                       index=1, metadata={})
        merged = chunker._fix_cross_page_breaks([first, second])
        self.assertEqual(len(merged), 2, "合并不得突破 chunk_size 的 1.25 倍上限")

    def test_empty_input_yields_no_chunks(self):
        self.assertEqual(_chunker("paper").split(_doc(" \n \t")), [])


# ===========================================================================
# 偏移不变式（跨策略）
# ===========================================================================
class OffsetInvariantTest(unittest.TestCase):
    """任何策略产出的块都必须能由 ``page_text[start:end]`` 精确还原。"""

    def test_all_strategies_return_exact_slices_on_sample_paper(self):
        pages = _sample_paper_pages()
        for strategy in ("fixed", "recursive", "semantic", "paper"):
            with self.subTest(strategy=strategy):
                chunks = _chunker(strategy, chunk_size=160, chunk_overlap=24).split(
                    _doc(*pages, doc_id=strategy)
                )
                self.assertGreater(len(chunks), 0)
                _assert_offsets(self, chunks, pages)

    def test_tricky_multiline_text_stays_exact(self):
        text = (
            "方法部分采用双栏排版，本行会在栏底被人为\n"
            "折断，换行并不代表句子结束。\n"
            "公式（1）给出注意力权重的计算方式：A = softmax(QK^T / sqrt(d_k))\n"
            "其中全角括号（）、引号“示例”、书名号《论文》都需原样保留。\n"
            "柱状对比实验表明，本方法在长文本上更稳定；\n"
            "其在短文本上也未出现明显退化。\n"
        )
        for strategy in ("fixed", "recursive", "semantic", "paper"):
            with self.subTest(strategy=strategy):
                chunks = _chunker(strategy, chunk_size=96, chunk_overlap=16).split(
                    _doc(text, doc_id=strategy)
                )
                self.assertGreater(len(chunks), 0)
                _assert_offsets(self, chunks, [text])

    def test_pages_and_offsets_stay_within_bounds(self):
        from src.utils.text import normalize_text

        pages = _sample_paper_pages()
        for strategy in ("fixed", "recursive", "semantic", "paper"):
            with self.subTest(strategy=strategy):
                chunks = _chunker(strategy, chunk_size=140).split(_doc(*pages, doc_id=strategy))
                page_numbers = [c.page for c in chunks]
                self.assertEqual(page_numbers, sorted(page_numbers), "页码必须非递减")
                for chunk in chunks:
                    self.assertGreaterEqual(chunk.page, 1)
                    self.assertLessEqual(chunk.page, len(pages))
                    self.assertLessEqual(chunk.end_offset, len(normalize_text(pages[chunk.page - 1])))

    def test_very_long_text_is_handled_by_all_strategies(self):
        text = "".join(f"第{i:03d}句：本方法在长文本上表现稳定，检索效果良好。" for i in range(400))
        for strategy in ("fixed", "recursive", "semantic", "paper"):
            with self.subTest(strategy=strategy):
                chunks = _chunker(strategy, chunk_size=256).split(_doc(text, doc_id=strategy))
                self.assertGreater(len(chunks), 10)
                self.assertTrue(all(c.text.strip() for c in chunks))
                _assert_offsets(self, chunks, [text])

    def test_empty_inputs_yield_no_chunks_for_all_strategies(self):
        from src.chunking import STRATEGIES

        for strategy in STRATEGIES:
            for pages in ([], [""], ["   \n\t "], ["　　"]):
                with self.subTest(strategy=strategy, pages=pages):
                    self.assertEqual(_chunker(strategy).split(_doc(*pages)), [])


# ===========================================================================
# 确定性
# ===========================================================================
class ChunkDeterminismTest(unittest.TestCase):
    """同一输入两次切分必须得到完全一致的块（页码/偏移/文本/顺序）。"""

    def test_all_strategies_are_deterministic(self):
        pages = _sample_paper_pages()
        for strategy in ("fixed", "recursive", "semantic", "paper"):
            with self.subTest(strategy=strategy):
                doc = _doc(*pages, doc_id="det")
                first = _snapshot(_chunker(strategy, chunk_size=160, chunk_overlap=24).split(doc))
                second = _snapshot(_chunker(strategy, chunk_size=160, chunk_overlap=24).split(doc))
                self.assertGreater(len(first), 0)
                self.assertEqual(first, second)

    def test_semantic_matches_across_chunker_instances(self):
        """两个独立实例（各自解析 embedder）也必须给出一致结果。"""
        text = _topic_text()
        doc = _doc(text, doc_id="det")
        snapshots = [
            _snapshot(_chunker("semantic", embedder=_KeywordEmbedder(),
                               chunk_size=400, semantic_max_sentences=3).split(doc))
            for _ in range(2)
        ]
        self.assertEqual(snapshots[0], snapshots[1])
