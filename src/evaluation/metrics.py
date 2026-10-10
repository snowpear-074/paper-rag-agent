"""模块五 —— 评测指标库（纯函数式，零第三方依赖，仅用标准库 + numpy）。

设计要点
--------
1. **纯函数**：所有指标都是无状态的纯函数 —— 输入两个序列/一段文本，输出分数。
   没有类、没有全局状态，因此可以随意在单元测试、基准脚本、报告生成器之间复用。
2. **零第三方依赖**：只用 ``re`` / ``math`` / ``collections`` + numpy。
   检索指标尤其注意只用标准库实现，避免评测脚本本身因缺包跑不起来。
3. **两类指标分开**：
   - **检索指标**（Hit@K / MRR / Recall@K / Precision@K / NDCG@K / MAP@K）：
     衡量"该找到的资料有没有被找到、排得够不够前"，输入是**排序列表 + 金标准集合**。
   - **生成指标**（引用准确率 / 引用覆盖率 / 答案相关性 / 忠实度 / Token-F1 /
     ROUGE-L / EM）：衡量"答案本身对不对、有没有依据、引用是否真实存在"。
4. **与已有系统解耦**：引用解析复用 :mod:`src.generation.prompt_template` 的
   ``CITATION_PATTERN`` 与 ``parse_citations``（只读导入），保证"评测口径"
   与"线上解析口径"完全一致 —— 否则评测会因为正则差异得出与真实运行不同的结论。

指标口径说明（避免常见误用）
--------------------------
- 检索相关性一律使用 :class:`~src.utils.types.RetrievedChunk` 的 ``confidence``
  （绝对相关度），**不要用** ``score``（min-max 归一化的相对展示分，Top-1 恒为 1.0）。
- ``hit_at_k`` / ``recall_at_k`` 等的 ``k`` 语义是"只看前 k 条"；``k <= 0`` 时
  一律返回 0.0（而不是抛异常），这样批量评测脚本不会因个别样本的脏数据整体崩掉。
- ``ndcg_at_k`` 使用**二值相关性**（命中记 1，未命中记 0）。课程场景的人工标注
  只区分"相关/不相关"，没有分级相关性标签，二值是最诚实的口径。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    # 检索指标
    "hit_at_k",
    "mrr",
    "recall_at_k",
    "precision_at_k",
    "ndcg_at_k",
    "map_at_k",
    # 生成质量指标
    "citation_accuracy",
    "citation_coverage",
    "answer_relevance",
    "faithfulness",
    # 文本相似指标
    "token_f1",
    "exact_match",
    "rouge_l",
    # 聚合工具
    "aggregate",
    "split_sentences",
    "ngrams",
    "normalize_text",
    "tokenize",
]


# ==============================================================================
# 通用文本工具
# ==============================================================================
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|\n+")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_.'][a-zA-Z0-9]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_CITATION_TAIL = re.compile(r"[【（(][^】）)]{0,60}[】）)]\s*$")

#: 模板化套话（不含事实内容）。它们会被 4-gram 忠实度判为"无上下文支撑"，
#: 从而系统性地拉低忠实度分数 —— 必须显式过滤，否则指标衡量的是"套话比例"
#: 而不是"幻觉比例"。这些句式来自内置的离线 mock 生成器与常见 LLM 开场白。
_BOILERPLATE_PATTERNS = (
    re.compile(r"^根据知识库检索结果"),
    re.compile(r"^以上内容摘自检索到的"),
    re.compile(r"^以下是就"),
    re.compile(r"^知识库中检索到的"),
    re.compile(r"^根据(?:提供的)?(?:参考)?资料"),
    re.compile(r"^以下是基于"),
    re.compile(r"^当前知识库中未找到"),
    re.compile(r"^（?离线模拟模式"),
)


def _is_boilerplate(sentence: str) -> bool:
    """判断句子是否为不含事实内容的模板套话。"""
    text = sentence.strip()
    if not text:
        return True
    return any(pattern.search(text) for pattern in _BOILERPLATE_PATTERNS)

# 英文停用词（用于答案相关性/关键信息覆盖率，刻意保持精简：
# 只去掉疑问词与虚词，避免把 "not"、"no" 这类影响语义的词滤掉）
_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "this", "that", "these", "those", "it", "its", "as", "at", "from", "into",
    "than", "then", "there", "their", "they", "we", "you", "i", "he", "she",
    "his", "her", "our", "us", "them", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "have", "has", "had", "about",
    # 中文疑问/虚词（按字处理，因此只放单字）
    "的", "了", "是", "在", "和", "与", "有", "为", "被", "把", "对", "从",
    "吗", "呢", "吧", "啊", "什么", "哪些", "怎么", "如何", "多少", "为什么",
    "请问", "介绍", "一下", "以及", "并且", "这个", "那个",
}


def normalize_text(text: str) -> str:
    """文本归一化：全角转半角、压空白、转小写。

    用于所有字符串比较类指标（EM / Token-F1 / ROUGE-L / 引用来源比对），
    保证"同一句话的不同排版"不会造成虚假的不匹配。
    """
    if not text:
        return ""
    # 全角字母数字 → 半角
    chars: List[str] = []
    for ch in str(text):
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            chars.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            chars.append(" ")
        else:
            chars.append(ch)
    cleaned = "".join(chars)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def tokenize(text: str) -> List[str]:
    """分词：英文按单词（保留连字符/小数点），中文按**单字 + 二元组**。

    中文用字级别 + bigram，是因为本系统语料以英文论文为主，
    中文只出现在"问题"一侧；不做分词器依赖的前提下，bigram 能提供
    足够的区分度（例如"数据/据集/集"覆盖 "数据集" 这个词）。
    """
    if not text:
        return []
    normalized = normalize_text(text)
    tokens: List[str] = _WORD_RE.findall(normalized)
    for run in _CJK_RE.findall(normalized):
        tokens.extend(list(run))
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def ngrams(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    """由 token 序列生成 n-gram 集合（用于重合度计算）。"""
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def split_sentences(text: str, keep_citations: bool = False) -> List[str]:
    """按中英文标点切句，并剥掉引用标记与附加元信息。

    处理三类噪声，否则它们会被当成"答案内容"参与忠实度计算，
    把本来正确的答案误判为幻觉：

    1. Markdown 列表符号与行首编号（``- `` / ``1. `` / ``> ``）；
    2. **任意位置**的引用标记 ``【来源 p.N】``（不只句末 —— 模型常把引用
       写在句中，例如 ``xxx【论文A p.2】。``）；
    3. 附加的匹配度后缀 ``（匹配度 45%）``。

    长度 < 8 的碎片、以及不含事实内容的模板套话会被丢弃。

    Parameters
    ----------
    keep_citations:
        是否保留引用标记。**默认 False（剥掉）**，因为忠实度需要比对的是
        "答案的事实内容是否来自上下文"，引用标记只是元信息。
        但 :func:`citation_coverage` 恰恰要判断"这句话有没有带引用"，
        因此它必须传 ``keep_citations=True``。

    适用场景：引用覆盖率、忠实度等"按句判定"的指标。
    """
    if not text:
        return []

    from ..generation.prompt_template import CITATION_PATTERN

    raw_text = str(text)
    # 先把 Markdown 列表符号与行首编号清掉，避免 "- " 与 "1. " 被当成句子
    cleaned = re.sub(r"(?m)^\s*(?:[-*+>]|\d+[.、)])\s*", "", raw_text)
    if not keep_citations:
        # 再剥掉全文任意位置的引用标记与匹配度后缀
        cleaned = CITATION_PATTERN.sub(" ", cleaned)
        cleaned = re.sub(r"[（(]匹配度[^）)]*[）)]", " ", cleaned)
    else:
        # 保留引用标记，但仍去掉匹配度后缀（它不是引用的一部分，会干扰正则）
        cleaned = re.sub(r"（匹配度[^）)]*）", "", cleaned)

    raw = _SENTENCE_SPLIT.split(cleaned)
    sentences: List[str] = []
    for piece in raw:
        sentence = re.sub(r"\s+", " ", (piece or "").strip())
        if not keep_citations:
            # 反复剥离句末残留的引用标记（兼容非标准写法，如"【来源 第3页】"）
            for _ in range(3):
                stripped = _CITATION_TAIL.sub("", sentence).strip()
                if stripped == sentence:
                    break
                sentence = stripped
        if len(sentence) >= 8 and not _is_boilerplate(sentence):
            sentences.append(sentence)
    return sentences


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """最长公共子序列长度（滚动数组实现，空间 O(min(len))）。"""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


# ==============================================================================
# 一、检索指标
# ==============================================================================
def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Hit@K：前 k 条结果中**是否至少有一条**命中金标准。

    算法：取 ``ranked[:k]`` 与 ``gold`` 求交集，非空记 1.0，否则 0.0。

    适用场景：单答案检索任务的"召回底线"指标。它只回答"找到了吗"，
    不关心找到几条、排在第几 —— 所以必须与 MRR/NDCG 一起看，
    否则"把唯一正确结果排在第 5 位"和"排在第 1 位"会被判为同样好。

    >>> hit_at_k(["c3", "c1"], ["c1"], k=2)
    1.0
    >>> hit_at_k(["c3"], ["c1"], k=1)
    0.0
    """
    if k <= 0:
        return 0.0
    gold_set = set(gold or ())
    if not gold_set:
        return 0.0
    return 1.0 if any(item in gold_set for item in list(ranked or ())[:k]) else 0.0


def mrr(ranked: Sequence[str], gold: Sequence[str]) -> float:
    """MRR（Mean Reciprocal Rank 的单样本形式）：第一个命中结果的倒数排名。

    算法：从头扫描 ``ranked``，第 i 个位置命中则返回 ``1/i``；全部未命中返回 0。

    适用场景：衡量"正确结果排得够不够前"，对答案唯一（如"某指标是多少"）
    的问答最敏感。MRR 只看第一个命中位置，所以它天然惩罚"靠后的正确结果"。

    >>> mrr(["c9", "c1", "c2"], ["c1"])
    0.5
    """
    gold_set = set(gold or ())
    if not gold_set:
        return 0.0
    for rank, item in enumerate(ranked or (), start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Recall@K：前 k 条覆盖了金标准集合的比例。

    算法：``|ranked[:k] ∩ gold| / |gold|``。

    适用场景：金标准有**多条**相关片段时（对比类/总结类问题通常需要多篇论文
    的证据），Recall@K 比 Hit@K 更能反映"资料够不够用"。

    >>> recall_at_k(["c1", "c2", "c3"], ["c1", "c2"], k=3)
    1.0
    """
    gold_set = set(gold or ())
    if k <= 0 or not gold_set:
        return 0.0
    top = set(list(ranked or ())[:k])
    return len(gold_set & top) / len(gold_set)


def precision_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Precision@K：前 k 条里有多少比例是相关的。

    算法：``|ranked[:k] ∩ gold| / min(k, len(ranked))``。
    注意分母用**实际返回条数**而不是 k —— 检索只返回 3 条却要求 top-5 时，
    若按 k=5 做分母会人为压低分数，掩盖"这次检索其实全对"的事实。

    适用场景：衡量"检索结果里噪声有多少"。上下文窗口有限时（要拼进 Prompt），
    高精度比高召回更重要。
    """
    if k <= 0:
        return 0.0
    ranked_list = list(ranked or ())
    if not ranked_list:
        return 0.0
    gold_set = set(gold or ())
    top = ranked_list[:k]
    return len(gold_set & set(top)) / min(k, len(top))


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """NDCG@K：归一化折损累计增益（二值相关性）。

    算法：
        1. ``DCG = Σ_{i=1..k} rel_i / log2(i + 1)``，``rel_i`` 为 1（命中）或 0；
        2. ``IDCG`` 为理想排序（所有相关项排在最前）的 DCG；
        3. ``NDCG = DCG / IDCG``。

    适用场景：同时考察"召回"与"排序质量"的综合指标，是三档检索对比实验的主指标。

    手算对照（gold 为单条 c1）::

        排序 ["c1", "x"] k=2 → DCG = 1/log2(2) = 1.0；IDCG = 1.0 → NDCG = 1.0
        排序 ["x", "c1"] k=2 → DCG = 1/log2(3) = 0.6309；IDCG = 1.0 → NDCG ≈ 0.6309
        排序 ["x", "y"] k=2 → 0.0

    >>> round(ndcg_at_k(["x", "c1"], ["c1"], k=2), 4)
    0.6309
    """
    if k <= 0:
        return 0.0
    gold_set = set(gold or ())
    if not gold_set:
        return 0.0
    dcg = 0.0
    for index, item in enumerate(list(ranked or ())[:k], start=1):
        if item in gold_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def map_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """MAP@K：平均精度均值（单样本形式，即 Average Precision@K）。

    算法：遍历前 k 个位置，每遇到一个命中就记录"此刻的精度"（命中数/已看条数），
    最后除以 ``min(|gold|, k)`` 做归一。

    与 NDCG@K 的区别：NDCG 用对数折损，MAP 用线性折损，
    因此 MAP 对"正确结果排在很后面"惩罚更重。

    >>> round(map_at_k(["c1", "x", "c2"], ["c1", "c2"], k=3), 4)
    0.8333
    """
    gold_set = set(gold or ())
    if k <= 0 or not gold_set:
        return 0.0
    hits = 0
    total = 0.0
    for index, item in enumerate(list(ranked or ())[:k], start=1):
        if item in gold_set:
            hits += 1
            total += hits / index
    denominator = min(len(gold_set), k)
    return total / denominator if denominator else 0.0


# ==============================================================================
# 二、生成质量指标
# ==============================================================================
def _normalize_source_name(source: str) -> str:
    """来源名归一化：小写、去扩展名、去空白与常见分隔符。

    系统里同一个来源可能写作 ``Attention Is All You Need``（论文标题）、
    ``sample_transformer.pdf``（文件名）或 ``attention_is_all_you_need``，
    归一化后三者可以互相匹配。
    """
    text = normalize_text(source)
    text = re.sub(r"\.(pdf|docx?|txt|md|markdown)$", "", text)
    text = re.sub(r"[\s_\-—–]+", "", text)
    return text


def citation_accuracy(
    answer: str,
    citations: Sequence[Any],
    available_sources: Sequence[Any],
) -> Dict[str, Any]:
    """引用准确率：答案里标注的 ``【来源 p.N】`` 是否真实存在于检索结果中。

    算法
    ----
    1. 用 :data:`src.generation.prompt_template.CITATION_PATTERN` 从答案正文中
       抽出全部 ``(来源, 页码)`` 标注（与线上前端解析使用**同一个正则**）；
    2. 每个标注与 ``available_sources`` 逐个比对：先精确匹配 ``(来源, 页码)``，
       再退化为**来源名归一化匹配**（容忍标题/文件名/大小写/页码缺失差异）；
    3. 两者都不中的标注计入 ``hallucinated`` —— 即"模型编造了一个知识库里
       根本没有的来源"，这是科研问答里最危险的错误类型。

    Parameters
    ----------
    answer:
        模型生成的答案原文。
    citations:
        系统解析出的引用对象列表（:class:`~src.utils.types.Citation` 或等价 dict）。
        仅用于统计条数，判定有效性以 ``available_sources`` 为准。
    available_sources:
        本次**实际检索到**的来源列表，元素为 ``Citation`` / ``RetrievedChunk`` /
        ``dict`` / ``str``。``RetrievedChunk`` 会自动取 ``chunk.source`` 与
        ``chunk.page``；字符串则视为 "来源名"（页码通配，只要名字对就算有效）。

    Returns
    -------
    dict
        ``{"total", "valid", "invalid", "accuracy", "hallucinated": [...]}``。
        ``total == 0`` 时 ``accuracy`` 记为 1.0（没引用就谈不上引用错误），
        但 ``total`` 为 0 本身通常在"引用覆盖率"指标里被扣分。

    适用场景：检测**引用幻觉**。离线 mock 模式与真实 LLM 模式都会出现
    "为了显得严谨而编造来源"的行为，这个指标是发现该问题的唯一自动化手段。
    """
    from ..generation.prompt_template import CITATION_PATTERN

    # -- 1) 归一化可用来源 --------------------------------------------------
    exact: Set[Tuple[str, int]] = set()
    by_name: Dict[str, Set[int]] = {}
    name_only: Set[str] = set()
    for item in available_sources or ():
        source, page = _extract_source_page(item)
        if not source:
            continue
        exact.add((source.strip().lower(), int(page)))
        key = _normalize_source_name(source)
        by_name.setdefault(key, set()).add(int(page))
        name_only.add(key)

    # -- 2) 逐条核对答案中的标注 -------------------------------------------
    total = 0
    valid = 0
    unresolvable = 0
    hallucinated: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int]] = set()
    for match in CITATION_PATTERN.finditer(answer or ""):
        source = match.group("source").strip()
        page = int(match.group("page"))
        dedup_key = (source.lower(), page)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # 来源名本身就是占位符（模型写了【来源 p.3】或答案解析出了空来源）：
        # 无法核对 → 既不计入有效，也不判为幻觉，而是单独计数并显式报告。
        # 这是刻意的保守选择：把"无法解析"当成"幻觉"会制造假警报，
        # 而假警报会让真正的引用幻觉淹没在噪声里。
        if not _normalize_source_name(source):
            unresolvable += 1
            continue

        total += 1
        if _citation_exists(source, page, exact, by_name):
            valid += 1
        else:
            hallucinated.append(
                {
                    "source": source,
                    "page": page,
                    "reason": "该来源未出现在本次检索结果中",
                    "raw": match.group(0),
                }
            )

    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "accuracy": round(valid / total, 4) if total else 1.0,
        "hallucinated": hallucinated,
        "unresolvable": unresolvable,
        "num_available_sources": len(name_only),
        "num_parsed_citations": len(list(citations or ())),
    }


def _citation_exists(
    source: str,
    page: int,
    exact: Set[Tuple[str, int]],
    by_name: Dict[str, Set[int]],
) -> bool:
    """判断某个引用标注是否真实存在（先精确后模糊）。"""
    if (source.strip().lower(), page) in exact:
        return True
    key = _normalize_source_name(source)
    pages = by_name.get(key)
    if pages is None:
        return False
    # 来源名对得上：页码允许 ±1 的容差
    # （PDF 解析出的页码与模型口述的页码常因封面/目录相差一页）
    return any(abs(page - actual) <= 1 for actual in pages)


def _extract_source_page(item: Any) -> Tuple[str, int]:
    """从异构的"来源"对象中抽出 ``(来源名, 页码)``。

    找不到来源名时返回空字符串 —— 调用方必须把它当作"未知来源"而不是
    "名称为空的来源"，否则会把无法解析的引用误判成幻觉引用。
    """
    if item is None:
        return "", 0
    if isinstance(item, str):
        return item, 0
    if isinstance(item, dict):
        source = item.get("source") or item.get("title") or item.get("doc_id") or ""
        page = item.get("page", 0) or 0
        return str(source), int(page or 0)
    # RetrievedChunk（优先于 Citation 判断：它没有 label() 但有 chunk 属性）
    chunk = getattr(item, "chunk", None)
    if chunk is not None:
        source = getattr(chunk, "source", "") or getattr(chunk, "doc_id", "") or ""
        return str(source), int(getattr(chunk, "page", 0) or 0)
    source = getattr(item, "source", "") or ""
    page = getattr(item, "page", 0)
    return str(source), int(page or 0)


def citation_coverage(answer: str, min_sentence_chars: int = 15) -> Dict[str, Any]:
    """引用覆盖率：有多少比例的**实质句子**带了引用标记。

    算法：按句切分 → 只保留长度 ≥ ``min_sentence_chars`` 的实质句（过滤标题、
    列表符号、过渡语）→ 统计其中含 ``【来源 p.N】`` 的比例。

    适用场景：衡量答案的"可核查程度"。覆盖率低意味着大量断言没有出处，
    即使内容正确，在科研场景下也不可接受。

    >>> citation_coverage("- 该模型取得 28.4 BLEU【Attention Is All You Need p.2】。")["coverage"]
    1.0
    """
    from ..generation.prompt_template import CITATION_PATTERN

    # 必须保留引用标记切句：本指标判定的正是"这句话有没有带引用"
    sentences = [
        s for s in split_sentences(answer, keep_citations=True)
        if len(s) >= min_sentence_chars
    ]
    cited = [s for s in sentences if CITATION_PATTERN.search(s)]
    return {
        "sentences": len(sentences),
        "cited": len(cited),
        "coverage": round(len(cited) / len(sentences), 4) if sentences else 0.0,
        "uncited_examples": [s[:120] for s in sentences if not CITATION_PATTERN.search(s)][:3],
    }


def answer_relevance(answer: str, question: str) -> float:
    """答案相关性：问题关键词/术语在答案中的覆盖率（词面近似）。

    算法
    ----
    1. 从问题中抽取关键词：英文单词（去停用词、长度 ≥ 2）+ 中文 2 字以上片段；
    2. 计算关键词在答案文本（归一化后）中的出现比例；
    3. 关键词为空时返回 0.0。

    为什么只是"近似"
    ----------------
    真正的答案相关性需要语义判断（LLM-as-Judge 或训练打分模型）。本指标是
    **零依赖的离线代理指标**：它能可靠识别"答非所问"（问题问数据集，答案在
    讲结论），但无法区分"用同义词正确回答"与"答偏了"。报告中必须标注这一点。

    适用场景：批量回归测试的**粗筛**——把明显跑题的答案挑出来人工复核。
    """
    keywords = _question_keywords(question)
    if not keywords:
        return 0.0
    haystack = normalize_text(answer)
    if not haystack:
        return 0.0
    hit = sum(1 for kw in keywords if kw in haystack)
    return round(hit / len(keywords), 4)


def _question_keywords(question: str, max_keywords: int = 20) -> List[str]:
    """抽取问题关键词（英文单词 + 中文 2 字以上片段），去停用词与疑问词。"""
    normalized = normalize_text(question)
    keywords: List[str] = []
    for word in _WORD_RE.findall(normalized):
        if len(word) < 2 or word in _STOPWORDS:
            continue
        keywords.append(word)
    for run in _CJK_RE.findall(normalized):
        # 中文：整段（若不太长）+ 每个 2 字组合
        if 2 <= len(run) <= 8:
            keywords.append(run)
        keywords.extend(run[i : i + 2] for i in range(len(run) - 1))
    # 去重保序
    unique: List[str] = []
    for kw in keywords:
        if kw not in unique and kw not in _STOPWORDS:
            unique.append(kw)
    return unique[:max_keywords]


def faithfulness(
    answer: str,
    contexts: Sequence[str],
    n: int = 4,
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """忠实度：答案句子能在检索上下文中找到支撑的比例（4-gram 重合近似）。

    算法
    ----
    1. 把 ``contexts`` 全量拼接并 token 化，构建 4-gram 索引；
    2. 逐句处理答案：句子的 4-gram 中有多大比例出现在上下文索引里；
    3. 比例 ≥ ``threshold`` 记为"有依据"；
    4. ``score = 有依据句数 / 总句数``。

    为什么用 4-gram 而不是更复杂的 NLI 模型
    --------------------------------------
    忠实度的本质是"答案是否由上下文蕴含"。无模型环境下，最可靠的近似是
    **字面重合**：如果一句话的 4-gram 几乎都能在上下文里找到，它大概率是
    上下文的重述而非编造。反之，编造的数字/结论几乎不可能 4-gram 重合。
    该近似的已知盲区是"同义改写"（会被低估）与"拼接多个片段得到的错误结论"
    （会被高估），报告中需如实说明。

    Parameters
    ----------
    answer: 模型答案。
    contexts: 检索到的上下文片段文本列表。
    n: n-gram 阶数，默认 4（英文约相当于 3~4 个词，足够区分表述来源）。
    threshold: 判定单句"有依据"的重合比例阈值，默认 0.6。

    适用场景：检测**生成幻觉**。与 :func:`citation_accuracy` 互补：
    前者查"来源是否真实存在"，本函数查"内容是否真有依据"。
    """
    sentences = split_sentences(answer)
    if not sentences:
        return {
            "score": 0.0, "sentences": 0, "supported": 0,
            "threshold": threshold, "unsupported_examples": [],
        }

    joined = "\n".join(str(c or "") for c in (contexts or ()))
    context_tokens = tokenize(joined)
    context_grams = ngrams(context_tokens, n)

    supported = 0
    unsupported: List[str] = []
    for sentence in sentences:
        gram_set = ngrams(tokenize(sentence), n)
        if not gram_set:
            # 句子太短（token 数 < n）：用词汇包含做退化判定
            ratio = 1.0 if normalize_text(sentence) in normalize_text(joined) else 0.0
        else:
            ratio = len(gram_set & context_grams) / len(gram_set)
        if ratio >= threshold:
            supported += 1
        else:
            unsupported.append(sentence[:120])

    return {
        "score": round(supported / len(sentences), 4),
        "sentences": len(sentences),
        "supported": supported,
        "threshold": threshold,
        "unsupported_examples": unsupported[:3],
    }


# ==============================================================================
# 三、答案与参考答案的文本相似指标
# ==============================================================================
def token_f1(prediction: str, reference: str) -> float:
    """Token 级 F1（SQuAD 风格）。

    算法：把两句都 token 化后按**多重集合**计算交集，得到
    ``precision = 交集/预测长度``、``recall = 交集/参考长度``，
    再取调和平均。重复词按出现次数计（多说的重复词会拉低精度）。

    适用场景：抽取式问答的标准指标，适合"答案是一个数字/短语"的
    factual 类问题。对同义改写不友好，因此需与 ROUGE-L、忠实度配合看。

    >>> token_f1("28.4 BLEU", "28.4 BLEU")
    1.0
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def exact_match(prediction: str, reference: str) -> float:
    """精确匹配（归一化后完全一致记 1.0，否则 0.0）。

    适用场景：指标数字、数据集名称、超参数这类**不允许改写**的答案。
    对长答案几乎必然为 0，因此只对短答案有意义。
    """
    left = normalize_text(prediction)
    right = normalize_text(reference)
    if not left or not right:
        return 0.0
    return 1.0 if left == right else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L：基于最长公共子序列的 F 值（纯 Python 实现）。

    算法
    ----
    1. 两句分别 token 化，求 LCS 长度 ``lcs``；
    2. ``R = lcs / len(ref)``、``P = lcs / len(pred)``；
    3. ``F = (1 + β²) · P · R / (R + β² · P)``，默认 β = 1（F1）。

    与 Token-F1 的区别：ROUGE-L 是**顺序敏感**的（子序列要求相对顺序一致），
    因此对"词都对但语序混乱"的答案给分更低，更接近人类的可读性直觉。

    适用场景：summary 类问题的参考答案比对。长答案上比 EM 有意义得多。

    >>> round(rouge_l("the big model achieves 28.4 BLEU", "the big model achieves 28.4 BLEU"), 4)
    1.0
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    recall = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)
    if precision + recall == 0:
        return 0.0
    beta_squared = 1.0
    score = (1 + beta_squared) * precision * recall / (recall + beta_squared * precision)
    return round(score, 4)


# ==============================================================================
# 四、聚合工具
# ==============================================================================
def aggregate(
    rows: Sequence[Dict[str, Any]],
    keys: Optional[Sequence[str]] = None,
    percentiles: Sequence[float] = (95.0,),
) -> Dict[str, Dict[str, float]]:
    """对一批字典记录做数值聚合：均值 / 中位数 / P95 / 最值 / 样本数。

    Parameters
    ----------
    rows:
        记录列表，例如 ``[{"hit": 1.0, "latency_ms": 12.3}, ...]``。
    keys:
        要聚合的字段；None 表示自动收集所有"值可转 float"的字段。
    percentiles:
        需要计算的百分位（默认 P95）。用 ``numpy.percentile`` 线性插值。

    Returns
    -------
    dict
        ``{字段名: {"count", "mean", "median", "p95", "min", "max"}}``。
        空输入返回 ``{}``（不抛异常，便于报告端直接渲染"未运行"）。

    适用场景：把逐条明细（每个问题的指标）汇总成一行整体指标。
    报告里的"平均延迟 / P95 延迟"就来自这里。
    """
    records = [r for r in (rows or ()) if isinstance(r, dict)]
    if not records:
        return {}

    if keys is None:
        collected: List[str] = []
        for record in records:
            for key, value in record.items():
                if key in collected:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                collected.append(key)
        keys = collected

    summary: Dict[str, Dict[str, float]] = {}
    for key in keys or ():
        values: List[float] = []
        for record in records:
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isnan(float(value)) or math.isinf(float(value)):
                continue
            values.append(float(value))
        if not values:
            continue
        stats: Dict[str, float] = {
            "count": len(values),
            "mean": round(float(np.mean(values)), 4),
            "median": round(float(np.median(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
        }
        for percentile in percentiles or ():
            tag = f"p{int(percentile)}" if float(percentile).is_integer() else f"p{percentile}"
            stats[tag] = round(float(np.percentile(values, percentile)), 4)
        summary[str(key)] = stats
    return summary
