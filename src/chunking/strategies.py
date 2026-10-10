"""文本分块策略（模块一 5.1.2）。

三种及以上策略，统一接口 :meth:`BaseChunker.split`：

=============  ==================================================================
策略            原理与适用场景
=============  ==================================================================
``fixed``      固定字符窗口 + 重叠。最简单，作为对照基线。
               用于对比 chunk_size=256/512/1024 的检索效果。
``recursive``  递归字符切分：按 段落 → 换行 → 中英句号 → 分号 → 逗号 → 空格
               的优先级逐层尝试，尽量在语义边界断开。**默认策略**，
               是效果/复杂度平衡最好的选择。
``semantic``   语义切分：先切句，再用 Embedding 计算相邻句相似度，
               在相似度骤降处（话题切换）断开。块内语义一致性最高，
               但需要 Embedding 模型，成本较高。
``paper``      学术论文专用：在 recursive 基础上增加双栏/跨页合并、
               公式行保护、章节标题吸附、参考文献段剔除等优化。
=============  ==================================================================

所有策略都保证：
- 每个 :class:`Chunk` 携带 ``page``（页码）与 ``start_offset/end_offset``
  （相对该页文本的字符偏移），使引用溯源能精确定位。
- 偏移是**页内区间的真实坐标**：分块全程用 :class:`_Piece` 记录来源区间，
  文本只从原文切出（不再用 ``" ".join(...)`` 之类重建后再回头查找），
  因此 ``page_text[start_offset:end_offset] == chunk.text`` 恒成立，
  绝不会凭空捏造区间；万一片段无法定位，则退化为哨兵值
  :data:`_OFFSET_UNKNOWN` 并记录警告。
- 不产生小于 ``min_chunk_size`` 的碎片（自动并入相邻块）。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.text import normalize_text, split_sentences
from ..utils.types import Chunk, LoadedDocument, stable_hash

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "PaperChunker",
    "CHUNKER_REGISTRY",
    "create_chunker",
    "chunk_document",
    "chunk_documents",
    "STRATEGIES",
]

_log = get_logger("chunking")

STRATEGIES: Tuple[str, ...] = ("fixed", "recursive", "semantic", "paper")

# 递归切分的分隔符优先级（从粗到细）
_SEPARATORS: Tuple[str, ...] = (
    "\n\n\n",       # 空行（章节）
    "\n\n",         # 段落
    "\n",           # 行
    "。", "！", "？", "；",   # 中文句末
    ". ", "! ", "? ", "; ",   # 英文句末
    "，", ", ",       # 从句
    "、", " ",        # 顿号 / 空格
    "",
)

# 章节标题行（用于"标题吸附"：把标题并入其后首个分块，避免孤立标题块）
_RE_SECTION_HEADING = re.compile(
    r"^(?:#{1,6}\s+\S|\d+(?:\.\d+)*\s*[\.、]?\s*\S{2,60}$"
    r"|(?:abstract|introduction|related work|method(?:s|ology)?|experiment(?:s)?|results?|"
    r"discussion|conclusion(?:s)?|references|acknowledg(?:e?ments?)|appendix)\b.*$"
    r"|(?:摘\s*要|引\s*言|相关工作|方\s*法|实\s*验|结\s*果|讨\s*论|结\s*论|参考文献|致\s*谢)\s*[:：]?$)",
    re.IGNORECASE,
)
# 公式行：以 = 为中心，或包含大量数学符号
_RE_FORMULA = re.compile(r"^[^。！？]{0,80}[=≈≤≥∑∏∫√±×÷∈∀∃][^。！？]{0,80}$")
_RE_REFERENCE_START = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?:references|bibliography|参\s*考\s*文\s*献|引\s*用\s*文\s*献)"
    r"\s*(?:\*\*)?\s*[:：]?\s*$",
    re.IGNORECASE,
)
_RE_REFERENCE_ITEM = re.compile(r"^\s*(?:\[\d{1,3}\]|\d{1,3}\.\s|\*\s*\[?\d)")


def _strip_markdown_marks(text: str) -> str:
    """去掉行首的 Markdown 标记，便于按纯文本规则匹配。"""
    return re.sub(r"^\s*(?:#{1,6}\s*|\*\*\s*|\*\s*)", "", text).strip()


# 偏移无法定位时的哨兵区间：调用方可据此判断"该块的溯源位置不可用"。
# 取 (0, 0) 而非 (-1, -1)，保证 ``start <= end <= len(page_text)`` 不变式恒成立；
# 空片段会被直接丢弃，因此非空块不可能自然产生空区间，该哨兵无歧义。
_OFFSET_UNKNOWN: Tuple[int, int] = (0, 0)

# 句末标点（含全角右括号/右引号）：出现在这里说明句子已完整，无需跨块拼接
_SENTENCE_ENDINGS: Tuple[str, ...] = (
    "。", "！", "？", ".", "!", "?",
    "：", ":", "；", ";",
    "】", ")", "）", "”", "“", "」", "』",
)

# 公式行掩码：掩码字符不是分隔符、也不是空白，切分时整行保持原子；
# 掩码长度与原文行等长，因此掩码文本上的偏移可直接平移到原文本
_FORMULA_MASK = "\x00"


@dataclass(frozen=True)
class _Piece:
    """页内片段：对页面文本某个**连续区间** ``[start, end)`` 的视图。

    旧实现先把文本切好再 ``page.find(text)`` 反查位置；而
    ``" ".join(...)`` / ``f"{a}\\n{b}"`` 这类重建会把原文换行替换成空格，
    导致 find 失败后只能"猜"一个区间（引用溯源随之错位）。

    本类从根上避免这个问题：``text`` 恒等于 ``source[start:end]``，
    合并/切分全部退化为区间运算，偏移天然精确、不可能虚构。
    """

    source: str
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.source[self.start : self.end]

    def stripped(self) -> "_Piece":
        """去掉首尾空白后的新片段（仍引用原文区间）。"""
        lo, hi = self.start, self.end
        while lo < hi and self.source[lo].isspace():
            lo += 1
        while hi > lo and self.source[hi - 1].isspace():
            hi -= 1
        return _Piece(self.source, lo, hi)

    def __len__(self) -> int:
        return self.end - self.start


def _join_span(pieces: Sequence["_Piece"]) -> "_Piece":
    """把按原文顺序排列的若干片段并为**一个连续区间**（含片段间的原文空白）。

    片段之间只可能夹着空白（切分时丢弃的空片段必为空白），因此合并后的
    文本依然是原文的连续切片，而不是用分隔符拼出来的近似串。
    """
    first, last = pieces[0], pieces[-1]
    return _Piece(first.source, first.start, last.end)


def _find_flexible(source: str, needle: str, start: int = 0) -> Optional[Tuple[int, int]]:
    """在 ``source[start:]`` 中定位 ``needle``，返回 ``(start, end)``；失败返回 None。

    先尝试精确匹配；失败后按"空白通配"对齐：``needle`` 的非空白字符必须
    按序出现，中间只允许跳过 ``source`` 的空白字符。
    ``split_sentences`` / ``" ".join`` 只会改写空白、不会增删实义字符，
    因此按该规则仍能找回原文区间（返回的是覆盖所有匹配字符的最小连续区间）。
    """
    pos = source.find(needle, start)
    if pos != -1:
        return pos, pos + len(needle)

    idx = 0
    cursor = start
    first = last = -1
    total = len(source)
    while idx < len(needle):
        char = needle[idx]
        if char.isspace():
            idx += 1
            continue
        while cursor < total and source[cursor].isspace():
            cursor += 1
        if cursor >= total or source[cursor] != char:
            return None
        if first < 0:
            first = cursor
        last = cursor
        idx += 1
        cursor += 1
    if first < 0:
        return None
    return first, last + 1


def _locate_sentences(text: str, sentences: Sequence[str]) -> Optional[List[_Piece]]:
    """把按序切自 ``text`` 的句子映射回页内区间；任一句定位失败返回 None。"""
    pieces: List[_Piece] = []
    cursor = 0
    for sentence in sentences:
        found = _find_flexible(text, sentence, cursor)
        if found is None:
            return None
        pieces.append(_Piece(text, found[0], found[1]))
        cursor = found[1]
    return pieces


# ==============================================================================
# 抽象基类
# ==============================================================================
@dataclass
class ChunkingConfig:
    """分块参数（可由 config.yaml 的 ``chunking`` 段构造）。"""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 80
    semantic_threshold: float = 0.62
    semantic_max_sentences: int = 12
    remove_references: bool = True
    merge_cross_page: bool = True
    keep_formula_lines: bool = True
    dehyphenate: bool = True

    @classmethod
    def from_config(cls, config: Any) -> "ChunkingConfig":
        """从全局配置对象构造。"""
        node = config.chunking
        semantic = node.semantic
        optimize = node.paper_optimize
        return cls(
            chunk_size=int(node.chunk_size),
            chunk_overlap=int(node.chunk_overlap),
            min_chunk_size=int(node.min_chunk_size),
            semantic_threshold=float(semantic.similarity_threshold),
            semantic_max_sentences=int(semantic.max_sentences),
            remove_references=bool(optimize.remove_references),
            merge_cross_page=bool(optimize.merge_cross_page),
            keep_formula_lines=bool(optimize.keep_formula_lines),
            dehyphenate=bool(optimize.dehyphenate),
        )


class BaseChunker(ABC):
    """分块器抽象基类。

    子类实现 :meth:`_split_page`：把**单页文本**切成若干 :class:`_Piece`
    （页内区间的视图，自带精确偏移）。
    基类负责统一的页码标注、ID 生成、碎片合并与元数据填充。
    """

    name: str = "base"

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    # -- 子类实现 ---------------------------------------------------------
    @abstractmethod
    def _split_page(self, text: str) -> List[_Piece]:
        """把一页文本切成若干片段（片段自带相对页首的字符区间）。"""
        raise NotImplementedError

    # -- 统一流程 ---------------------------------------------------------
    def split(self, document: LoadedDocument) -> List[Chunk]:
        """把文档切成 :class:`Chunk` 列表。"""
        pages = document.pages or [document.text]
        chunks: List[Chunk] = []
        index = 0

        for page_no, page_text in enumerate(pages, start=1):
            cleaned = normalize_text(page_text) if self.config.dehyphenate else page_text
            if not cleaned.strip():
                continue
            cleaned = self._preprocess_page(cleaned, page_no, len(pages))
            if not cleaned.strip():
                continue
            pieces = self._split_page(cleaned)
            pieces = self._merge_small(pieces)
            pieces = self._enforce_size_limit(pieces)
            for piece in pieces:
                # 片段即页内区间视图，偏移直接取自区间，无需（也不允许）反查。
                piece = piece.stripped()
                if not piece.text:
                    continue
                start, end = self._resolve_offsets(piece, cleaned)
                chunks.append(
                    self._make_chunk(
                        document=document,
                        text=piece.text,
                        index=index,
                        page=page_no,
                        start=start,
                        end=end,
                    )
                )
                index += 1

        if not chunks and document.text.strip():
            # 极端兜底：pages 为空或无有效内容时，对全文做一次切分
            fallback = normalize_text(document.text)
            for piece in self._merge_small(self._split_page(fallback)):
                piece = piece.stripped()
                if not piece.text:
                    continue
                start, end = self._resolve_offsets(piece, fallback)
                chunks.append(
                    self._make_chunk(
                        document=document,
                        text=piece.text,
                        index=index,
                        page=1,
                        start=start,
                        end=end,
                    )
                )
                index += 1
        return chunks

    def _resolve_offsets(self, piece: _Piece, page_text: str) -> Tuple[int, int]:
        """校验片段区间，返回可安全写入 :class:`Chunk` 的 ``(start, end)``。

        正常情况下片段是页内区间的视图，必然满足
        ``page_text[start:end] == piece.text``。若子类自定义切分逻辑破坏了
        该前提，则**绝不臆造**位置：记录警告并退化为哨兵值
        :data:`_OFFSET_UNKNOWN`（``(0, 0)``，调用方可据此跳过引用定位）。
        """
        if (
            0 <= piece.start <= piece.end <= len(page_text)
            and page_text[piece.start : piece.end] == piece.text
        ):
            return piece.start, piece.end
        _log.warning(
            "分块片段无法在页内定位，偏移退化为哨兵 %s: %r",
            _OFFSET_UNKNOWN,
            piece.text[:40],
        )
        return _OFFSET_UNKNOWN

    def _preprocess_page(self, text: str, page_no: int, total_pages: int) -> str:
        """页级预处理（默认直接返回；论文策略会覆写以剔除参考文献）。"""
        return text

    def _make_chunk(
        self,
        document: LoadedDocument,
        text: str,
        index: int,
        page: int,
        start: int,
        end: int,
    ) -> Chunk:
        """构造 Chunk 并填充元数据。"""
        meta = document.meta
        return Chunk(
            chunk_id=stable_hash(f"{document.doc_id}:{self.name}:{index}:{text[:120]}"),
            doc_id=document.doc_id,
            text=text,
            index=index,
            page=page,
            start_offset=start,
            end_offset=end,
            strategy=self.name,
            source=meta.title or meta.source_path or document.doc_id,
            metadata={
                "title": meta.title,
                "authors": meta.authors,
                "year": meta.year,
                "venue": meta.venue,
                "doi": meta.doi,
                "file_type": meta.file_type,
                "source_path": meta.source_path,
                "num_chars": len(text),
            },
        )

    def _merge_small(self, pieces: Sequence[_Piece]) -> List[_Piece]:
        """把过短的碎片并入相邻块，并施加**目标大小上限**。

        ⚠️ 这里有一个实测踩出来的关键点：
        早期实现只判断"当前片段是否过短"，合并时**完全不看结果大小**，
        于是 512 目标大小的分块器会产出 1500+ 字符的超大块
        （论文里大量短行如小标题、公式、图表说明会被连续吞并）。

        后果很隐蔽但很严重：一个 1504 字符的块里可能同时包含
        "多头注意力的复杂度分析" 与 "WMT 2014 数据集与训练细节"，
        而检索打分会**均值化**整块语义 —— 明明知识库里有 WMT 2014，
        问"训练用了什么数据集"却检索不到（该块被更靠前的注意力内容带偏）。

        因此合并规则改为两条硬约束：
        1. 短片段（< ``min_chunk_size``）→ 并入相邻块；
        2. 合并后的长度不得超过 ``chunk_size`` 的 1.25 倍（留出少量弹性）。

        合并即"区间取并"：新片段覆盖前一段到当前段的连续原文区间，
        因此合并后的文本仍是原文切片（旧实现用 ``\\n`` 拼接，会丢失原换行）。
        """
        minimum = max(1, self.config.min_chunk_size)
        ceiling = max(minimum * 2, int(self.config.chunk_size * 1.25))

        merged: List[_Piece] = []
        for piece in pieces:
            piece = piece.stripped()
            if not piece.text:
                continue
            if not merged:
                merged.append(piece)
                continue
            current = merged[-1]
            joined_len = piece.end - current.start
            # 短碎片且合并后不超上限 → 并入前一块
            if len(piece) < minimum and joined_len <= ceiling:
                merged[-1] = _join_span((current, piece))
            else:
                merged.append(piece)

        # 首块过短且与后一块合并后仍在上限内 → 向后合并
        if len(merged) > 1 and len(merged[0]) < minimum and merged[1].end - merged[0].start <= ceiling:
            merged[1] = _join_span((merged[0], merged[1]))
            merged.pop(0)
        return merged

    def _enforce_size_limit(self, pieces: Sequence[_Piece]) -> List[_Piece]:
        """对超过目标大小的块做二次切分（在句子边界断开）。

        为什么需要：递归切分本身受 ``chunk_size`` 约束，但
        :meth:`_merge_small` 的合并、以及语义/论文策略的特殊处理
        都可能产出偏大的块。这里做最后一道兜底，保证
        "声明了 chunk_size 就真的不会大幅超出"，避免块内语义被稀释。
        """
        ceiling = int(self.config.chunk_size * 1.25)
        result: List[_Piece] = []
        for piece in pieces:
            piece = piece.stripped()
            if not piece.text:
                continue
            if len(piece) <= ceiling:
                result.append(piece)
                continue
            result.extend(self._split_oversized(piece, ceiling))
        return result

    def _split_oversized(self, piece: _Piece, ceiling: int) -> List[_Piece]:
        """把超大片段按句子边界切成多个接近 ``ceiling`` 的片段。

        切出的片段直接取**原文区间**：旧实现用 ``" ".join(句子)`` 重建文本，
        换行被空格替换后已不是原文子串，偏移只能靠"猜"，引用定位随之失效。
        """
        sentences = split_sentences(piece.text, min_len=1)
        if len(sentences) <= 1:
            # 无法按句子切（例如整块是一张表格）→ 按字符硬切
            return self._hard_split(piece, ceiling)

        located = _locate_sentences(piece.text, sentences)
        if located is None:
            # 理论上不会发生（句子就切自本片段）；保守起见不猜位置
            _log.warning("超大块无法按句子定位，退化为按字符硬切: %r", piece.text[:40])
            return self._hard_split(piece, ceiling)

        blocks: List[_Piece] = []
        current: Optional[_Piece] = None
        for sentence in located:
            if current is None:
                current = sentence
                continue
            if sentence.end - current.start > ceiling:
                blocks.append(current)
                current = sentence
            else:
                current = _join_span((current, sentence))
        if current is not None:
            blocks.append(current)

        # 单句本身就超长时继续硬切
        final: List[_Piece] = []
        for block in blocks:
            if len(block) <= ceiling * 1.5:
                final.append(block)
            else:
                final.extend(self._hard_split(block, ceiling))
        return final

    def _hard_split(self, piece: _Piece, size: int) -> List[_Piece]:
        """按固定字符窗口硬切片段（保持原文视图）。"""
        result: List[_Piece] = []
        for start in range(piece.start, piece.end, size):
            sub = _Piece(piece.source, start, min(start + size, piece.end)).stripped()
            if sub.text:
                result.append(sub)
        return result

    def describe(self) -> Dict[str, Any]:
        """参数摘要（写入实验报告 / 前端展示）。"""
        return {
            "strategy": self.name,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
        }


# ==============================================================================
# 策略一：固定大小切分
# ==============================================================================
class FixedSizeChunker(BaseChunker):
    """固定字符窗口切分（带重叠）。

    完全按字符数硬切，不考虑语义边界 —— 作为对照组基线，
    用于量化"尊重语义边界"带来的检索收益。
    """

    name = "fixed"

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        if chunk_size is not None:
            self.config.chunk_size = int(chunk_size)
        if chunk_overlap is not None:
            self.config.chunk_overlap = int(chunk_overlap)

    def _split_page(self, text: str) -> List[_Piece]:
        size = max(16, self.config.chunk_size)
        overlap = max(0, min(self.config.chunk_overlap, size // 2))
        step = max(1, size - overlap)
        pieces: List[_Piece] = []
        for start in range(0, len(text), step):
            piece = _Piece(text, start, min(start + size, len(text))).stripped()
            if piece.text:
                pieces.append(piece)
            if start + size >= len(text):
                break
        return pieces


# ==============================================================================
# 策略二：递归字符切分
# ==============================================================================
class RecursiveChunker(BaseChunker):
    """递归字符切分（LangChain ``RecursiveCharacterTextSplitter`` 的等价手写实现）。

    算法
    ----
    1. 取当前优先级的最高分隔符切分文本；
    2. 若某个片段仍超过 ``chunk_size``，用下一级分隔符递归切分；
    3. 相邻片段在不超过 ``chunk_size`` 的前提下贪心合并；
    4. 相邻块之间保留 ``chunk_overlap`` 个字符的上下文重叠。

    相比固定切分，它能把句子/段落完整保留在同一个块里，
    显著提升"答案片段完整性"与人工可读性。
    """

    name = "recursive"

    def _split_page(self, text: str) -> List[_Piece]:
        size = max(64, self.config.chunk_size)
        raw_units = self._split_recursive(_Piece(text, 0, len(text)), 0, size)
        return self._pack(raw_units, size)

    def _split_recursive(self, piece: _Piece, level: int, size: int) -> List[_Piece]:
        """按分隔符优先级递归切分，直到所有片段都不超过 size。"""
        if len(piece) <= size:
            return [piece] if piece.text.strip() else []
        if level >= len(_SEPARATORS):
            # 无合适分隔符：硬切
            return self._hard_split(piece, size)

        separator = _SEPARATORS[level]
        if separator == "":
            return self._hard_split(piece, size)

        parts = self._split_keep_separator(piece, separator)
        units: List[_Piece] = []
        for part in parts:
            if len(part) <= size:
                if part.text.strip():
                    units.append(part)
            else:
                units.extend(self._split_recursive(part, level + 1, size))
        return units

    @staticmethod
    def _split_keep_separator(piece: _Piece, separator: str) -> List[_Piece]:
        """切分并**保留**分隔符在片段尾部（句号等标点不应丢失）。

        返回的仍是原文区间视图：分隔符并入前一段，字符一个不少。
        """
        text = piece.source
        if text.find(separator, piece.start, piece.end) == -1:
            return [piece]
        parts: List[_Piece] = []
        part_start = piece.start
        idx = piece.start
        while idx < piece.end:
            if text.startswith(separator, idx, piece.end):
                idx += len(separator)
                parts.append(_Piece(text, part_start, idx))
                part_start = idx
            else:
                idx += 1
        if part_start < piece.end:
            parts.append(_Piece(text, part_start, piece.end))
        return parts

    def _pack(self, units: Sequence[_Piece], size: int) -> List[_Piece]:
        """贪心合并相邻单元，并保留块间重叠。

        重叠会让块略微超过 ``size``，因此上限放宽到 ``size + overlap``。
        （此前用了 1.5 倍上限，实测会产出远超声明大小的块，
        导致块内混入多个主题、检索时语义被均值化。）

        合并/重叠都是区间运算：合并取两段之间的连续原文区间，
        重叠取上一块尾部 ``overlap`` 个字符所在的区间 —— 因此块文本
        始终是原文切片，带重叠时也不会出现"找不到该文本"的偏移。
        """
        overlap = max(0, min(self.config.chunk_overlap, size // 2))
        ceiling = size + overlap
        blocks: List[_Piece] = []
        current: Optional[_Piece] = None
        for unit in units:
            if current is None:
                current = unit
                continue
            if len(current) + len(unit) <= size:
                current = _join_span((current, unit))
            else:
                blocks.append(current)
                if overlap:
                    tail = _Piece(current.source, max(current.start, current.end - overlap), current.end)
                    current = _join_span((tail, unit))
                else:
                    current = unit
                # 保护性裁剪：重叠不应让块无限增长
                if len(current) > ceiling:
                    blocks.append(_Piece(current.source, current.start, current.start + size))
                    current = _Piece(current.source, current.start + size, current.end)
        if current is not None and current.text.strip():
            blocks.append(current)
        return blocks


# ==============================================================================
# 策略三：语义切分
# ==============================================================================
class SemanticChunker(BaseChunker):
    """基于句子嵌入相似度的语义切分。

    算法
    ----
    1. 把页面切成句子；
    2. 用 Embedding 模型编码每个句子；
    3. 计算相邻句子的余弦相似度，相似度低于 ``semantic_threshold``
       的位置视为"话题切换点"；
    4. 在切换点断开，并受 ``semantic_max_sentences`` 与 ``chunk_size`` 双重约束。

    优点：块内语义高度一致，检索时噪声更少。
    代价：需要 Embedding 模型（本模块对该依赖做惰性加载，缺省时自动回退到
    基于词汇重叠的相似度，保证策略始终可用）。
    """

    name = "semantic"

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embedder: Optional[Any] = None,
    ) -> None:
        super().__init__(config)
        self._embedder = embedder
        self._embedder_resolved = embedder is not None

    def _get_embedder(self) -> Optional[Any]:
        """惰性获取 Embedding 模型；失败则返回 None（触发词面回退）。"""
        if self._embedder_resolved:
            return self._embedder
        self._embedder_resolved = True
        try:
            from ..embeddings import create_embedder

            self._embedder = create_embedder(use_cache=True)
        except Exception as exc:  # noqa: BLE001
            _log.warning("语义分块未能加载 Embedding 模型，回退到词面相似度: %s", exc)
            self._embedder = None
        return self._embedder

    def _split_page(self, text: str) -> List[_Piece]:
        sentences = split_sentences(text, min_len=8)
        if len(sentences) <= 2:
            return [_Piece(text, 0, len(text))]

        located = _locate_sentences(text, sentences)
        if located is None:
            # 理论不可达（句子就切自本页文本）；保守起见不猜位置
            _log.warning("语义分块无法把句子映射回原文，退化为整页单块")
            return [_Piece(text, 0, len(text))]

        similarities = self._adjacent_similarities(sentences)
        threshold = self.config.semantic_threshold
        max_sentences = max(2, self.config.semantic_max_sentences)
        size_limit = max(64, self.config.chunk_size)

        # 按"区间取并"组合句子：块文本 = 首句到末句的连续原文区间，
        # 句间空白保持原样（旧实现用 " ".join 重建，换行丢失导致偏移反查失败）。
        blocks: List[_Piece] = []
        current: List[_Piece] = []
        for idx, sentence in enumerate(located):
            current.append(sentence)
            # similarities 只有 len(sentences)-1 项：末句之后没有话题切换点，
            # 统一交给循环外的收尾逻辑（旧实现在分支里既 append 又 break，
            # 末句因此被重复产出一个块）。
            if idx >= len(similarities):
                break
            boundary = similarities[idx] < threshold
            span_len = current[-1].end - current[0].start
            too_long = span_len >= size_limit or len(current) >= max_sentences
            if boundary or too_long:
                blocks.append(_join_span(current))
                current = []
        if current:
            blocks.append(_join_span(current))
        return blocks

    def _adjacent_similarities(self, sentences: Sequence[str]) -> List[float]:
        """计算相邻句子的相似度序列，长度为 len(sentences)-1。"""
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                matrix = embedder.encode_batch(list(sentences))
                sims: List[float] = []
                for i in range(len(sentences) - 1):
                    sims.append(float(np.dot(matrix[i], matrix[i + 1])))
                return sims
            except Exception as exc:  # noqa: BLE001
                _log.warning("语义相似度计算失败，回退词面相似度: %s", exc)
        return [self._lexical_similarity(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]

    @staticmethod
    def _lexical_similarity(left: str, right: str) -> float:
        """词面 Jaccard 相似度（无 Embedding 时的回退方案）。"""
        from ..utils.text import tokenize

        a, b = set(tokenize(left)), set(tokenize(right))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


# ==============================================================================
# 策略四：学术论文专用
# ==============================================================================
class PaperChunker(RecursiveChunker):
    """面向学术论文优化的分块策略（课程要求 5.1.2 的"针对学术论文特点优化"）。

    在 :class:`RecursiveChunker` 之上增加四项学术场景优化：

    1. **参考文献剔除**：检测 "References / 参考文献" 段落并截断，避免大量
       文献条目污染检索结果（这类块与问题语义相似度高但信息价值低）。
    2. **公式行保护**：把包含 ``=``、``∑``、``∫`` 等数学符号的行视为原子单元，
       不从中断开，保证公式与其上下文不被切断。
    3. **章节标题吸附**：把孤立的章节标题行并入其后首个块，避免出现
       "## 3 Method" 这种只有标题、没有信息的块。
    4. **断裂修复（仅限同页）**：若上一块以未结束的句子结尾（缺少句末标点）
       且下一块以中文/小写字母开头，说明段落被切分逻辑切断，则重新拼接。
       这里**不做跨页合并**：:class:`Chunk` 只能表示单页内的一个连续区间，
       跨页拼接会让 ``page`` 与 ``start_offset/end_offset`` 分属两套坐标系
       （旧实现即因此让引用指向错误的页与区间），故跨页断裂保持断开。
    """

    name = "paper"

    def split(self, document: LoadedDocument) -> List[Chunk]:
        chunks = super().split(document)
        if self.config.merge_cross_page:
            chunks = self._fix_cross_page_breaks(chunks)
        return chunks

    # -- 优化 1：参考文献剔除 --------------------------------------------
    def _preprocess_page(self, text: str, page_no: int, total_pages: int) -> str:
        if not self.config.remove_references:
            return text
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            if _RE_REFERENCE_START.match(line.strip()):
                # 确认后续确实是大片文献条目（避免误伤提到 "References" 的正文）
                tail = lines[idx + 1 : idx + 12]
                ref_like = sum(1 for t in tail if _RE_REFERENCE_ITEM.match(t) or re.search(r"\b(19|20)\d{2}\b", t))
                if ref_like >= max(2, len(tail) // 3):
                    return "\n".join(lines[:idx])
        return text

    # -- 优化 2：公式行保护 ----------------------------------------------
    def _split_page(self, text: str) -> List[_Piece]:
        if not self.config.keep_formula_lines:
            return super()._split_page(text)

        # 用**等长掩码**替换公式行：掩码不含任何分隔符，公式行在切分时
        # 保持原子；又因长度与原文行相同，掩码文本上的偏移可直接平移到
        # 原文本 —— 最终片段统一改回原文视图，掩码字符绝不会进入结果。
        lines = text.split("\n")
        masked: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and _RE_FORMULA.match(stripped) and len(stripped) < 200:
                masked.append(_FORMULA_MASK * len(line))
            else:
                masked.append(line)
        pieces = super()._split_page("\n".join(masked))
        return [_Piece(text, piece.start, piece.end) for piece in pieces]

    # -- 优化 3：章节标题吸附 --------------------------------------------
    def _merge_small(self, pieces: Sequence[_Piece]) -> List[_Piece]:
        """章节标题吸附 + 大小约束。

        在父类 :meth:`RecursiveChunker._merge_small`（已含尺寸上限）的基础上，
        额外处理"孤立小标题"：形如 ``## 3 Method`` 的块本身信息量极低，
        单独成块会污染检索结果，因此暂存它与下一个正文块合并。

        "吸附"同样是区间取并，合并后的文本仍是原文切片。

        注意：合并后的长度同样受上限约束，否则会把标题和整节内容揉成超大块，
        重新引入"块内语义被稀释"的问题。
        """
        merged = super()._merge_small(pieces)
        ceiling = max(
            self.config.min_chunk_size * 2, int(self.config.chunk_size * 1.25)
        )

        result: List[_Piece] = []
        pending: Optional[_Piece] = None
        for piece in merged:
            if not piece.text:
                continue
            # 孤立标题（内容基本就是标题本身）暂存，等下一块合并
            if len(piece) < self.config.min_chunk_size and _RE_SECTION_HEADING.match(piece.text):
                pending = _join_span((pending, piece)) if pending is not None else piece
                continue
            if pending is not None:
                candidate = _join_span((pending, piece))
                # 只有合并后不超上限才吸附，否则标题单独成块（可接受）
                if len(candidate) <= ceiling:
                    result.append(candidate)
                else:
                    result.append(pending)
                    result.append(piece)
                pending = None
            else:
                result.append(piece)

        if pending is not None:
            if result and pending.end - result[-1].start <= ceiling:
                result[-1] = _join_span((result[-1], pending))
            else:
                result.append(pending)
        return result

    # -- 优化 4：断裂修复（仅限同页） ------------------------------------
    def _fix_cross_page_breaks(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        """把"上一块以未结束的句子结尾 + 下一块同属一段"的块合并。

        判定条件（全部满足才合并）：
        1. 上一块与下一块**同页**；
        2. 上一块末尾不是句末标点（句子被截断；全角 ``）`` ``”`` ``」``
           等右括号/右引号同样视为句子完整）；
        3. 下一块起始是小写字母或中文（说明是同一句的延续，而不是新段落的开头）；
        4. 合并后长度不超过 ``chunk_size`` 的 1.25 倍。

        关于第 1 条：这里选择"跨页直接不合并"这一最简方案。单个
        :class:`Chunk` 只能表达**同一页内**的一个连续区间，而跨页拼接会让
        ``page`` 与 ``start_offset/end_offset`` 分属两套坐标系 —— 旧实现
        正是如此：合并块保留上一页的 ``page``，``end_offset`` 却取自下一页，
        于是 ``p.<page>`` 引用指向错误的页、区间切片也与文本对不上。
        只有同页合并，页码与偏移才保持自洽（始终在同一坐标系内）。

        第 4 条同样是**必需的**：这里原本写死了 1600 字符上限，结果把前面
        :meth:`_enforce_size_limit` 刚切好的 500 字块又合并回 1500 字，
        块内混入多个主题、检索时语义被均值化 ——
        表现为"知识库明明有 WMT 2014 数据集，问训练数据却检索不到"。
        修复断裂的收益远小于超长块的代价，因此宁可放弃合并也要守住尺寸。
        """
        if len(chunks) < 2:
            return list(chunks)

        ceiling = max(
            self.config.min_chunk_size * 2, int(self.config.chunk_size * 1.25)
        )
        merged: List[Chunk] = [chunks[0]]
        for chunk in chunks[1:]:
            previous = merged[-1]
            tail = previous.text.rstrip()
            head = chunk.text.lstrip()
            if (
                previous.page == chunk.page
                and tail
                and head
                and not tail.endswith(_SENTENCE_ENDINGS)
                and (head[0].islower() or "\u4e00" <= head[0] <= "\u9fff")
                and len(previous.text) + len(chunk.text) <= ceiling
            ):
                previous.text = f"{tail} {head}"
                # 同页才走到这里：chunk.end_offset 与 previous 的偏移同属
                # 该页坐标系，页码/区间因此保持自洽
                previous.end_offset = chunk.end_offset
                previous.metadata["merged_chunks"] = previous.metadata.get("merged_chunks", 1) + 1
                continue
            merged.append(chunk)
        # 重新编号，保证索引连续
        for order, chunk in enumerate(merged):
            chunk.index = order
        return merged


# ==============================================================================
# 注册表与工厂
# ==============================================================================
CHUNKER_REGISTRY: Dict[str, type] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "paper": PaperChunker,
}


def create_chunker(
    config: Any = None,
    *,
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedder: Optional[Any] = None,
) -> BaseChunker:
    """创建分块器。

    Parameters
    ----------
    config: 全局配置对象（**第一个位置参数**，与其它工厂保持一致）
    strategy: ``fixed`` / ``recursive`` / ``semantic`` / ``paper``
    chunk_size / chunk_overlap: 覆盖配置中的同名参数（用于对比实验）
    embedder: 语义分块可注入的 Embedding 模型（依赖注入，便于测试）

    除 ``config`` 外全部为 **keyword-only**：避免 ``create_chunker(cfg, "paper")``
    这类位置传参把配置对象误传给 ``strategy``。

    Examples
    --------
    >>> create_chunker(cfg, strategy="paper", chunk_size=512)
    >>> create_chunker(strategy="fixed", chunk_size=256)     # 用默认配置
    """
    from ..utils.config import get_config

    cfg = config or get_config()
    if isinstance(cfg, str):
        # 容错：有人可能按老习惯写 create_chunker("paper")，给出明确指引
        raise TypeError(
            f"create_chunker 的第一个参数应为配置对象，收到字符串 {cfg!r}。"
            f'请改用 create_chunker(strategy="{cfg}")。'
        )
    chunk_config = ChunkingConfig.from_config(cfg)
    if chunk_size is not None:
        chunk_config.chunk_size = int(chunk_size)
    if chunk_overlap is not None:
        chunk_config.chunk_overlap = int(chunk_overlap)

    key = str(strategy or cfg.chunking.strategy or "recursive").lower()
    cls = CHUNKER_REGISTRY.get(key)
    if cls is None:
        _log.warning("未知分块策略 %s，回退默认 recursive", strategy)
        cls = RecursiveChunker

    if cls is SemanticChunker:
        return cls(chunk_config, embedder=embedder)
    return cls(chunk_config)


def chunk_document(
    document: LoadedDocument,
    strategy: str = "recursive",
    config: Any = None,
    **kwargs: Any,
) -> List[Chunk]:
    """便捷函数：用指定策略切分单个文档。"""
    return create_chunker(config, strategy=strategy, **kwargs).split(document)


def chunk_documents(
    documents: Iterable[LoadedDocument],
    strategy: str = "recursive",
    config: Any = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    **kwargs: Any,
) -> List[Chunk]:
    """批量切分多个文档。"""
    chunker = create_chunker(config, strategy=strategy, **kwargs)
    items = list(documents)
    all_chunks: List[Chunk] = []
    for idx, document in enumerate(items, start=1):
        all_chunks.extend(chunker.split(document))
        if on_progress:
            on_progress(idx, len(items))
    return all_chunks
