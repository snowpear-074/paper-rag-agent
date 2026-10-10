"""中英文混合分词与停用词表。

优先使用 jieba（若已安装）以获得高质量中文分词；未安装时回退到
"中文单字 + 二元组 + 英文单词" 的混合切分策略 —— 该策略对 BM25 检索
依然有效（中文二元组近似 bigram 索引），且零依赖。

对外统一入口：:func:`tokenize`
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Sequence

__all__ = ["tokenize", "tokenize_query", "split_sentences", "extract_keywords", "STOPWORDS"]

# ------------------------------------------------------------------------------
# 正则
# ------------------------------------------------------------------------------
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
_RE_CJK_RUN = re.compile(rf"[{_CJK}]+")
_RE_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]*|\d+(?:\.\d+)?%?")
_RE_SENTENCE_SPLIT = re.compile(
    rf"(?<=[。！？；!?;])\s*|(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff])|\n{{2,}}"
)

# ------------------------------------------------------------------------------
# 停用词表（中英文常见虚词）
# ------------------------------------------------------------------------------
_ZH_STOP = """
的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
自己 这 那 这些 那些 之 与 及 以及 或 或者 而 而且 但 但是 如果 因为 所以 由于
对于 关于 通过 根据 按照 为了 以便 可以 能够 需要 应该 必须 可能 也许 已经 正在
被 把 让 使 从 向 往 对 于 以 为 其 该 此 本 等 等等 其中 上述 如下 以下 上述
我们 他们 她们 它们 你们 这个 那个 什么 怎么 怎样 如何 为什么 哪些 哪个
以 后 前 中 内 外 时 时候 时候 已经 还 再 又 更 最 太 非常 十分 特别 尤其
""".split()

_EN_STOP = """
a an the and or but if then else of to in on at by for with from as is are was were
be been being do does did have has had having this that these those it its it's
we our you your they their he she his her them us i me my mine
not no nor so such than too very can will just should now what which who whom whose
when where why how all any both each few more most other some only own same
about above after again against between into during before below under over
""".split()

STOPWORDS: frozenset[str] = frozenset(set(_ZH_STOP) | set(_EN_STOP))

# 中文停用字（单字层面的高频虚词，用于关键词提取时过滤）
_ZH_STOP_CHARS = frozenset("的了和是在我有就不人都一上也很到说要去你会着之与及或而但")


def _jieba_available() -> bool:
    try:  # pragma: no cover - 环境相关
        import jieba  # noqa: F401

        return True
    except Exception:
        return False


_HAS_JIEBA = _jieba_available()


@lru_cache(maxsize=20000)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    """带缓存的分词实现（LRU 上限 2 万条，避免重复分词开销）。"""
    if not text:
        return ()
    tokens: List[str] = []

    if _HAS_JIEBA:  # pragma: no cover - 环境相关
        import jieba

        for word in jieba.cut(text):
            # 必须先小写再比对停用词：停用词表是小写的，而英文句首/专有名词
            # （The / In / We / This…）经 jieba 切出来仍带大写，直接比对会漏掉。
            word = word.strip().lower()
            if not word or word in STOPWORDS:
                continue
            if _RE_CJK_RUN.fullmatch(word) and len(word) == 1 and word in _ZH_STOP_CHARS:
                continue
            tokens.append(word)
        if tokens:
            return tuple(tokens)

    # ---- 回退：中文 bigram + 英文单词 ----
    for run in _RE_CJK_RUN.findall(text):
        if len(run) == 1:
            if run not in _ZH_STOP_CHARS:
                tokens.append(run)
            continue
        # unigram（过滤停用字）+ bigram：兼顾精确匹配与语义召回
        for ch in run:
            if ch not in _ZH_STOP_CHARS:
                tokens.append(ch)
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram not in STOPWORDS:
                tokens.append(bigram)
    for word in _RE_LATIN.findall(text):
        lowered = word.lower().strip(".-")
        if lowered and lowered not in STOPWORDS and len(lowered) > 1:
            tokens.append(lowered)
    return tuple(tokens)


def tokenize(text: str) -> List[str]:
    """把文本切分为检索用 token 列表（已去停用词、已小写化）。"""
    if not text:
        return []
    return list(_tokenize_cached(text))


def tokenize_query(text: str) -> List[str]:
    """查询侧分词：保留更多信息（不过度过滤），提升短查询召回。"""
    if not text:
        return []
    tokens = tokenize(text)
    if tokens:
        return tokens
    # 极端情况（全是停用词）：退化为字符级，保证 BM25 不为空
    return [ch for ch in _RE_CJK_RUN.findall(text) for ch in ch]


def split_sentences(text: str, min_len: int = 6, max_len: int = 800) -> List[str]:
    """按中英文标点切句，并合并过短片段、硬切过长片段。

    用于：语义分块、引用片段抽取、关键词提取、摘要生成。
    """
    if not text:
        return []
    raw_parts = [p.strip() for p in _RE_SENTENCE_SPLIT.split(text)]
    sentences: List[str] = []
    buffer = ""
    for part in raw_parts:
        if not part:
            continue
        # 列表项 / 短标题单独成句
        if buffer:
            if len(buffer) < min_len:
                buffer = f"{buffer} {part}".strip()
            else:
                sentences.append(buffer)
                buffer = part
        else:
            buffer = part
        while len(buffer) > max_len:
            sentences.append(buffer[:max_len])
            buffer = buffer[max_len:]
    if buffer:
        sentences.append(buffer)

    # 合并仍然过短的尾巴
    merged: List[str] = []
    for sent in sentences:
        if merged and len(sent) < min_len and len(merged[-1]) + len(sent) < max_len:
            merged[-1] = f"{merged[-1]} {sent}"
        else:
            merged.append(sent)
    return [s for s in merged if s.strip()]


# 词性/结构先验：标题行、摘要行更可能是关键信息
_HINT_PATTERNS = (
    "摘要", "abstract", "关键词", "keywords", "方法", "method", "结论", "conclusion",
    "结果", "result", "实验", "experiment", "提出", "propose", "本文", "this paper",
)


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """基于词频 + 位置先验的关键词提取（零依赖 TF 变体）。

    评分 = 词频 * (1 + 长度权重) * (1 + 位置先验)
    """
    if not text:
        return []
    tokens = tokenize(text)
    if not tokens:
        return []
    freq: dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    first_pos: dict[str, int] = {}
    for pos, tok in enumerate(tokens):
        first_pos.setdefault(tok, pos)
    total = len(tokens)
    head_text = text[:300].lower()
    scored: List[tuple[float, str]] = []
    for tok, count in freq.items():
        length_bonus = 1.0 + 0.15 * min(len(tok), 8)
        pos_bonus = 1.0 + 0.5 * (1.0 - first_pos[tok] / total)
        hint_bonus = 1.3 if any(h in head_text for h in _HINT_PATTERNS) else 1.0
        # 过滤纯数字与单字中文（信息量过低）
        if tok.isdigit() or (len(tok) == 1 and _RE_CJK_RUN.fullmatch(tok)):
            continue
        scored.append((count * length_bonus * pos_bonus * hint_bonus, tok))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [tok for _, tok in scored[:top_k]]


def normalize_text(text: str) -> str:
    """文本归一化：统一换行、全角转半角、压缩空白、修复断词。

    用于分块前预处理与检索侧归一化，保证索引与查询口径一致。
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")   # 连字
    # 全角字母数字 → 半角
    out_chars: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out_chars.append(chr(code - 0xFEE0))
        else:
            out_chars.append(ch)
    text = "".join(out_chars)
    # 修复跨行断词：inter-\nnational -> international
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 合并行内多余空格，但保留段落换行
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文按 ~1.5 字符/token，英文按 ~4 字符/token）。

    无真实 tokenizer 时的通用做法，误差约 ±20%，足以支撑预算控制逻辑。
    """
    if not text:
        return 0
    cjk = len(_RE_CJK_RUN.findall(text))
    cjk_chars = sum(len(run) for run in _RE_CJK_RUN.findall(text))
    other = max(0, len(text) - cjk_chars)
    return int(cjk_chars / 1.5 + other / 4) + (1 if cjk else 0)


def join_with_limit(items: Sequence[str], limit_chars: int, sep: str = "\n") -> str:
    """按字符预算拼接字符串序列，超出预算即停止（用于上下文截断）。"""
    parts: List[str] = []
    used = 0
    for item in items:
        if used + len(item) > limit_chars and parts:
            break
        parts.append(item)
        used += len(item) + len(sep)
    return sep.join(parts)
