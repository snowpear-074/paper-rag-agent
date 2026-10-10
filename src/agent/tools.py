"""Agent 工具集（模块三 5.3.2）—— 8 个已注册可调用的工具。

============================  ==================================================
工具名                         职责
============================  ==================================================
``knowledge_search``          知识库检索（封装模块二 RAG 引擎）—— **最核心工具**
``paper_metadata``            论文元信息提取（标题/作者/年份/摘要/DOI）
``paper_compare``             论文对比（方法/数据集/结果的多维对照）
``keyword_extract``           关键词提取
``summarize_paper``           论文结构化摘要（背景-方法-结果-结论）
``current_time``              返回系统当前时间
``calculator``                安全数学表达式求值
``web_search``                联网搜索（可选，默认关闭，需配置）
============================  ==================================================

实现要点
--------
- 所有工具都**不抛异常**：内部错误统一转成 ``ToolResult(ERROR)`` + 可读建议，
  这是 Agent 错误恢复机制能生效的前提。
- 依赖未就绪的工具（如未配置联网搜索）会被标记 ``enabled=False``，
  不会出现在给 LLM 的工具清单里，从而避免模型调用必然失败的工具。
- 工具描述写得比较"啰嗦"是刻意的：实测把"什么时候该用/不该用"写清楚，
  能显著提升 ReAct 的工具选择准确率（评测见模块五）。
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.logger import get_logger, get_metrics
from ..utils.text import extract_keywords, split_sentences
from ..utils.types import ToolCall, ToolResult
from .base_tool import BaseTool, ToolError, ToolParameter

__all__ = [
    "KnowledgeSearchTool",
    "PaperMetadataTool",
    "PaperCompareTool",
    "KeywordExtractTool",
    "SummarizePaperTool",
    "CurrentTimeTool",
    "CalculatorTool",
    "WebSearchTool",
    "build_default_registry",
]

_log = get_logger("agent.tools")


# ==============================================================================
# 1. 知识库检索（核心工具）
# ==============================================================================
class KnowledgeSearchTool(BaseTool):
    """封装模块二 RAG 引擎。Agent 回答论文相关问题时首先调用它。"""

    name = "knowledge_search"
    description = (
        "在用户已上传的论文知识库中做语义+关键词混合检索，返回最相关的原文片段及其来源页码。\n"
        "**适用场景**：任何与已上传论文内容有关的问题 —— 论文用了什么方法/数据集、实验结果数值、"
        "某个概念的原文定义、某篇论文的具体做法。\n"
        "**不适用**：纯数学计算（用 calculator）、问当前时间（用 current_time）、"
        "问知识库里肯定没有的常识（直接回答即可）。\n"
        "返回内容包含片段编号、来源文档名、页码与相关度，可直接用于组织带引用的答案。"
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="检索问题或关键词。用自然语言描述你想找什么，可以比原始问题更聚焦",
            required=True,
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description="返回片段数量，2~10 之间；默认 5",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="mode",
            type="string",
            description="检索档位：full=混合检索+重排序（默认，最准）；hybrid=只用混合检索；vector=只用向量检索",
            required=False,
            default="full",
            enum=["full", "hybrid", "vector"],
        ),
    ]
    examples = ['knowledge_search(query="Transformer 训练用了哪些数据集", top_k=5)']

    #: 默认置信度过滤阈值。
    #:
    #: 标定依据（3 篇样例论文 / 混合检索+重排序，实测数据）：
    #:   域内问题（英文提问）Top-1 置信度 0.29 ~ 0.94
    #:   域内问题（中文提问）Top-1 置信度 0.12 ~ 0.29  ← 中文查英文语料天然偏低
    #:   域外问题（任意语言）Top-1 置信度恒为 0.0103（重排序相关性门控的下限）
    #: 两组之间存在干净的分割间隔（0.010 vs 0.124），取 0.06 可稳定分开，
    #: 同时不会误杀中文提问。可用 `retrieval.min_confidence` 覆盖。
    DEFAULT_MIN_CONFIDENCE = 0.06

    def __init__(
        self,
        retrieval_engine: Any = None,
        min_confidence: Optional[float] = None,
        config: Any = None,
    ) -> None:
        super().__init__()
        self.retrieval = retrieval_engine
        if min_confidence is None:
            try:
                from ..utils.config import get_config

                cfg = config or get_config()
                min_confidence = float(
                    cfg.get("retrieval.min_confidence", self.DEFAULT_MIN_CONFIDENCE)
                )
            except Exception:  # noqa: BLE001
                min_confidence = self.DEFAULT_MIN_CONFIDENCE
        self.min_confidence = float(min_confidence)
        self.enabled = retrieval_engine is not None

    def execute(self, query: str, top_k: int = 5, mode: str = "full", **_: Any) -> Tuple[str, Any]:
        if self.retrieval is None:
            raise ToolError("检索模块未初始化")
        if not str(query).strip():
            raise ToolError("query 不能为空，请给出要检索的问题或关键词")

        top_k = max(1, min(int(top_k or 5), 10))
        mode = mode if mode in ("full", "hybrid", "vector") else "full"
        began = time.perf_counter()
        context = self.retrieval.retrieve_with_context(query, top_k=top_k, mode=mode)
        elapsed_ms = (time.perf_counter() - began) * 1000.0

        # 过滤掉"低置信度噪声"结果。
        #
        # 必要性：向量检索**总会**返回 top_k 条最相似的片段，即使全部毫不相关。
        # 若把这些片段原样交给模型，它会误以为"知识库里有这个内容"从而编造答案。
        # 用绝对相关度（confidence）过滤后，无关问题会得到明确的"没检索到"，
        # Agent 才会正确地拒答或改用其它工具。
        threshold = self.min_confidence
        retrieved = list(context.get("results", []))
        results = [
            item for item in retrieved
            if float(item.get("confidence", item.get("score", 0.0)) or 0.0) >= threshold
        ]
        context["results"] = results
        context["num_raw"] = len(retrieved)      # 过滤前条数
        context["num"] = len(results)            # 过滤后条数
        context["filtered"] = len(retrieved) - len(results)

        # 记一次"裸检索"指标（Agent 这条路径不经过 RAG 流水线）。
        # 观测失败绝不能影响检索本身，所以整段吞异常。
        try:
            top_confidence = float(results[0].get("confidence", 0.0) or 0.0) if results else 0.0
            get_metrics().record_retrieval(
                latency_ms=elapsed_ms, top_score=top_confidence, hit=bool(results)
            )
        except Exception:  # noqa: BLE001 - 见上
            pass

        if not results:
            return (
                f"知识库中没有检索到与「{query}」相关的内容"
                f"（已检索但相关度均低于阈值 {threshold:.2f}）。\n"
                f"可能原因：知识库中没有覆盖该主题的论文，或问题表述与论文原文差异较大。\n"
                f"建议：换用论文中出现过的术语重新提问；若确实需要外部信息，"
                f"可考虑使用 web_search（若已启用）。",
                {"query": query, "num": 0, "results": [], "filtered": True},
            )

        lines = [f"检索「{query}」共命中 {len(results)} 个片段（档位 {mode}）：", ""]
        for item in results:
            relevance = item.get("confidence", item.get("score", 0.0))
            lines.append(f"[{item['index']}] 来源：{item['source']} p.{item['page']} | 相关度 {relevance:.2f}")
            lines.append(item["preview"].strip())
            lines.append("")
        lines.append(
            "提示：请在答案中对来自上述片段的事实标注【文档名 p.页码】。"
            "若上述片段不足以回答问题，请如实说明。"
        )
        text = "\n".join(lines)
        return text, context

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["knowledge_base"] = (
            {
                "num_documents": self.retrieval.num_documents,
                "num_chunks": self.retrieval.num_chunks,
            }
            if self.retrieval is not None
            else None
        )
        return data


# ==============================================================================
# 2. 论文元信息提取
# ==============================================================================
class PaperMetadataTool(BaseTool):
    """从已索引文档中提取标题、作者、年份、摘要、DOI、关键词。"""

    name = "paper_metadata"
    description = (
        "查询知识库中论文的元信息：标题、作者、发表年份、会议/期刊、DOI、摘要、关键词。\n"
        "**适用场景**：「这篇论文是谁写的」「XX 是哪一年发表的」「有哪些论文和 Transformer 有关」"
        "「某篇论文的 DOI 是多少」。\n"
        "**不适用**：问论文的技术细节（用 knowledge_search）。\n"
        "返回结构化字段，未收录的字段会明确标注为『未提取到』。"
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="论文标题关键词、作者名或主题；留空则返回知识库中全部论文的元信息列表",
            required=False,
            default="",
        ),
        ToolParameter(
            name="doc_id",
            type="string",
            description="精确指定文档 ID（若已知）；提供后忽略 query",
            required=False,
            default="",
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="最多返回多少篇，默认 5",
            required=False,
            default=5,
        ),
    ]
    examples = [
        'paper_metadata(query="ResNet")',
        'paper_metadata()   # 列出全部论文',
    ]

    def __init__(self, retrieval_engine: Any = None) -> None:
        super().__init__()
        self.retrieval = retrieval_engine
        self.enabled = retrieval_engine is not None

    def execute(self, query: str = "", doc_id: str = "", limit: int = 5, **_: Any) -> Tuple[str, Any]:
        if self.retrieval is None:
            raise ToolError("检索模块未初始化")

        documents = self.retrieval.documents()
        if not documents:
            return (
                "知识库中还没有任何论文。请先在上传区导入 PDF/Word/TXT 论文后重试。",
                {"num": 0, "documents": []},
            )

        if doc_id:
            matched = [d for d in documents if d.get("doc_id") == doc_id]
            if not matched:
                raise ToolError(f"未找到 doc_id={doc_id} 的论文。可用 ID：{[d.get('doc_id') for d in documents][:10]}")
        elif query.strip():
            matched = self._match(documents, query)
        else:
            matched = documents

        limit = max(1, min(int(limit or 5), 50))
        matched = matched[:limit]

        if not matched:
            return (
                f"知识库中没有找到与「{query}」匹配的论文。"
                f"当前共有 {len(documents)} 篇："
                + "、".join(str(d.get("title", "?"))[:40] for d in documents[:8]),
                {"num": 0, "documents": [], "available": [d.get("title") for d in documents]},
            )

        lines = [f"共匹配到 {len(matched)} 篇论文：", ""]
        for idx, doc in enumerate(matched, start=1):
            lines.append(f"【论文 {idx}】")
            lines.append(f"  doc_id：{doc.get('doc_id', '未提取到')}")
            lines.append(f"  标题：{doc.get('title') or '未提取到'}")
            authors = doc.get("authors") or []
            lines.append(f"  作者：{'、'.join(authors) if authors else '未提取到'}")
            lines.append(f"  年份：{doc.get('year') or '未提取到'}")
            lines.append(f"  发表：{doc.get('venue') or '未提取到'}")
            lines.append(f"  DOI：{doc.get('doi') or '未提取到'}")
            keywords = doc.get("keywords") or []
            if keywords:
                lines.append(f"  关键词：{'、'.join(str(k) for k in keywords[:8])}")
            abstract = (doc.get("abstract") or "").strip()
            if abstract:
                lines.append(f"  摘要：{abstract[:300]}{'…' if len(abstract) > 300 else ''}")
            lines.append(f"  分块数：{doc.get('num_chunks', 0)}")
            lines.append("")
        return "\n".join(lines), {"num": len(matched), "documents": matched}

    @staticmethod
    def _match(documents: Sequence[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """按标题/作者/关键词/摘要做词面匹配，并按命中强度排序。"""
        tokens = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9\-]{1,}", query)]
        if not tokens:
            return list(documents)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for doc in documents:
            title = str(doc.get("title") or "").lower()
            haystack = " ".join(
                [
                    title,
                    " ".join(str(a) for a in (doc.get("authors") or [])),
                    " ".join(str(k) for k in (doc.get("keywords") or [])),
                    str(doc.get("abstract") or ""),
                ]
            ).lower()
            score = 0.0
            for token in tokens:
                low = token.lower()
                if low in title:
                    score += 3.0
                elif low in haystack:
                    score += 1.0
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored]


# ==============================================================================
# 3. 论文对比
# ==============================================================================
class PaperCompareTool(BaseTool):
    """对两篇（或多篇）论文做方法/数据集/结果的多维对照。"""

    name = "paper_compare"
    description = (
        "对比知识库中多篇论文，从『核心方法』『数据集』『评价指标/结果』『主要贡献』四个维度生成对照表。\n"
        "**适用场景**：「对比论文A和论文B的方法差异」「这两篇工作哪个在 ImageNet 上更好」"
        "「XX 和 YY 有什么区别」。\n"
        "**参数说明**：传入论文标识列表（标题关键词或 doc_id）。如果只传一个，"
        "会自动选取知识库中与它最相关的第二篇作为对比对象。\n"
        "**不适用**：只问单篇论文（用 knowledge_search 或 summarize_paper）。"
    )
    parameters = [
        ToolParameter(
            name="papers",
            type="array",
            description="待对比的论文标识列表，如 [\"Transformer\", \"BERT\"]；也接受逗号分隔字符串",
            required=True,
            items_type="string",
        ),
        ToolParameter(
            name="aspects",
            type="array",
            description="对比维度，默认 ['方法','数据集','结果']",
            required=False,
            default=["方法", "数据集", "结果"],
            items_type="string",
        ),
    ]
    examples = ['paper_compare(papers=["Attention Is All You Need", "BERT"])']

    #: 每个维度对应的检索探针（用于从原文自动抽取证据）
    _ASPECT_QUERIES: Dict[str, str] = {
        "方法": "method architecture model design approach",
        "数据集": "dataset benchmark training data corpus",
        "结果": "result performance accuracy BLEU experiment",
        "贡献": "contribution novel we propose main",
    }

    def __init__(self, retrieval_engine: Any = None, top_k_per_aspect: int = 2) -> None:
        super().__init__()
        self.retrieval = retrieval_engine
        self.top_k_per_aspect = top_k_per_aspect
        self.enabled = retrieval_engine is not None

    def execute(self, papers: Any, aspects: Any = None, **_: Any) -> Tuple[str, Any]:
        if self.retrieval is None:
            raise ToolError("检索模块未初始化")

        identifiers = self._normalize_list(papers)
        if len(identifiers) < 1:
            raise ToolError("请至少提供一篇论文的标识（标题关键词或 doc_id）")

        documents = self.retrieval.documents()
        if not documents:
            raise ToolError("知识库为空，无法对比论文")

        resolved = []
        for identifier in identifiers:
            doc = self._resolve_document(documents, identifier)
            if doc is not None and doc not in resolved:
                resolved.append(doc)

        if len(resolved) < 2:
            # 自动补一篇最相关的其它论文
            supplement = self._pick_supplement(documents, resolved, identifiers)
            if supplement is not None:
                resolved.append(supplement)

        if len(resolved) < 2:
            return (
                f"知识库中只找到 {len(resolved)} 篇匹配的论文，无法做对比。"
                f"当前共有 {len(documents)} 篇："
                + "、".join(str(d.get("title", "?"))[:35] for d in documents[:8]),
                {"num": len(resolved), "documents": resolved},
            )

        aspect_list = self._normalize_list(aspects) or ["方法", "数据集", "结果"]
        evidence: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for doc in resolved:
            key = str(doc.get("doc_id"))
            evidence[key] = {}
            for aspect in aspect_list:
                probe = self._ASPECT_QUERIES.get(aspect, aspect)
                evidence[key][aspect] = self._collect_evidence(doc, probe, aspect)

        # 生成 Markdown 对照表
        headers = ["对比维度"] + [self._short_title(d) for d in resolved]
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        for aspect in aspect_list:
            row = [aspect]
            for doc in resolved:
                items = evidence[str(doc.get("doc_id"))].get(aspect, [])
                if not items:
                    row.append("未检索到相关原文")
                else:
                    snippet = str(items[0].get("preview", "")).replace("|", "／").replace("\n", " ").strip()
                    row.append(
                        f"{snippet[:150]}【{items[0].get('source', '?')} p.{items[0].get('page', '?')}】"
                    )
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("逐篇证据明细：")
        for doc in resolved:
            lines.append(f"\n■ {doc.get('title', '未命名')}（{doc.get('year') or '年份未知'}）")
            for aspect in aspect_list:
                for item in evidence[str(doc.get("doc_id"))].get(aspect, []):
                    lines.append(
                        f"  - [{aspect}] {str(item.get('preview', ''))[:220].strip()}"
                        f"【{item.get('source', '?')} p.{item.get('page', '?')}】"
                    )

        data = {
            "num": len(resolved),
            "documents": [
                {"doc_id": d.get("doc_id"), "title": d.get("title"), "year": d.get("year")}
                for d in resolved
            ],
            "aspects": aspect_list,
            "evidence": evidence,
        }
        return "\n".join(lines), data

    # -- 内部工具 ---------------------------------------------------------
    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    import json

                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:  # noqa: BLE001
                    pass
            return [v.strip() for v in re.split(r"[,，、;；]| and | 与 ", text) if v.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value)]

    def _resolve_document(
        self, documents: Sequence[Dict[str, Any]], identifier: str
    ) -> Optional[Dict[str, Any]]:
        text = identifier.strip()
        # 先按 doc_id 精确匹配
        for doc in documents:
            if str(doc.get("doc_id")) == text:
                return doc
        # 再按标题模糊匹配
        low = text.lower()
        best: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        tokens = [t.lower() for t in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9\-]{1,}", text)]
        for doc in documents:
            title = str(doc.get("title") or "").lower()
            if not title:
                continue
            if low in title or title in low:
                score = 10.0
            else:
                score = sum(1.0 for t in tokens if t in title)
            if score > best[0]:
                best = (score, doc)
        return best[1] if best[0] > 0 else None

    def _pick_supplement(
        self,
        documents: Sequence[Dict[str, Any]],
        resolved: Sequence[Dict[str, Any]],
        identifiers: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        chosen = {str(d.get("doc_id")) for d in resolved}
        for identifier in identifiers:
            for doc in documents:
                if str(doc.get("doc_id")) in chosen:
                    continue
                if self._resolve_document([doc], identifier) is not None:
                    return doc
        for doc in documents:
            if str(doc.get("doc_id")) not in chosen:
                return doc
        return None

    def _collect_evidence(self, doc: Dict[str, Any], probe: str, aspect: str) -> List[Dict[str, Any]]:
        """在指定论文范围内检索该维度证据。

        实现方式：先用探针做全局检索，再按 doc_id 过滤，保留该论文内的片段。
        说明：:meth:`RetrievalEngine.retrieve_with_context` 返回的每条结果是
        精简字典（含 ``preview`` 而非完整 ``text``），这里统一按该结构消费。
        """
        try:
            context = self.retrieval.retrieve_with_context(
                f"{doc.get('title', '')} {probe}".strip(), top_k=12, mode="full"
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("对比证据检索失败: %s", exc)
            return []
        doc_id = str(doc.get("doc_id"))
        items = []
        for item in context.get("results", []):
            if str(item.get("doc_id")) != doc_id:
                continue
            items.append(item)
            if len(items) >= self.top_k_per_aspect:
                break
        if not items:
            # 该论文没有命中，退化为取全局 Top-1（并标注来源以便人工判断）
            items = list(context.get("results", []))[:1]
        return items

    @staticmethod
    def _short_title(doc: Dict[str, Any]) -> str:
        title = str(doc.get("title") or doc.get("doc_id") or "未命名")
        return title if len(title) <= 28 else title[:26] + "…"


# ==============================================================================
# 4. 关键词提取
# ==============================================================================
class KeywordExtractTool(BaseTool):
    """从问题或指定论文中提取核心关键词。"""

    name = "keyword_extract"
    description = (
        "提取文本的核心关键词（基于词频、位置与长度先验的 TF 变体算法）。\n"
        "**适用场景**：用户要求「提取关键词」「这篇论文的关键词是什么」；"
        "或你想先把复杂问题拆成检索关键词再决定怎么查。\n"
        "**不适用**：需要论文原文内容（用 knowledge_search）。"
    )
    parameters = [
        ToolParameter(name="text", type="string", description="要提取关键词的文本；为空则用 query 检索知识库后提取", required=False, default=""),
        ToolParameter(name="query", type="string", description="从知识库中检索哪些内容来提取关键词", required=False, default=""),
        ToolParameter(name="top_k", type="integer", description="返回关键词个数，默认 10", required=False, default=10),
    ]
    examples = ['keyword_extract(query="Transformer 的注意力机制", top_k=8)']

    def __init__(self, retrieval_engine: Any = None) -> None:
        super().__init__()
        self.retrieval = retrieval_engine

    def execute(self, text: str = "", query: str = "", top_k: int = 10, **_: Any) -> Tuple[str, Any]:
        top_k = max(1, min(int(top_k or 10), 30))
        source_note = ""
        corpus = str(text or "").strip()

        if not corpus and query.strip():
            if self.retrieval is None:
                raise ToolError("既没有提供 text，也无法检索知识库（检索模块未初始化）")
            context = self.retrieval.retrieve_with_context(query, top_k=5)
            results = context.get("results", [])
            if not results:
                return f"知识库中没有找到与「{query}」相关的内容，无法提取关键词。", {"keywords": []}
            corpus = "\n".join(item["preview"] for item in results)
            source_note = f"（来源：知识库检索「{query}」的 {len(results)} 个片段）"

        if not corpus:
            raise ToolError("请提供 text 或 query 之一")

        keywords = extract_keywords(corpus, top_k=top_k)
        if not keywords:
            return "未能从给定文本中提取出有效关键词（文本可能过短或全是停用词）。", {"keywords": []}

        lines = [f"提取到 {len(keywords)} 个关键词{source_note}：", "", "、".join(keywords)]
        lines.append("")
        lines.append("（按词频 × 位置先验 × 长度权重排序，已过滤停用词）")
        return "\n".join(lines), {"keywords": keywords, "source_chars": len(corpus)}


# ==============================================================================
# 5. 论文结构化摘要
# ==============================================================================
class SummarizePaperTool(BaseTool):
    """生成『背景-方法-结果-结论』四段式论文摘要。"""

    name = "summarize_paper"
    description = (
        "针对指定论文生成结构化摘要，固定包含四部分：研究背景与动机、方法与技术路线、实验结果、结论与局限。\n"
        "**适用场景**：「总结一下这篇论文」「这篇论文讲了什么」「给我 XX 论文的摘要」。\n"
        "**不适用**：针对某个具体细节提问（用 knowledge_search 更精准）。\n"
        "摘要内容全部来自知识库原文片段，每部分都带来源页码引用。"
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="论文标题关键词，或想总结的主题",
            required=True,
        ),
        ToolParameter(
            name="doc_id",
            type="string",
            description="精确指定文档 ID（可选，优先级高于 query）",
            required=False,
            default="",
        ),
        ToolParameter(
            name="sections",
            type="integer",
            description="每个部分最多引用几段原文，默认 2",
            required=False,
            default=2,
        ),
    ]
    examples = ['summarize_paper(query="Deep Residual Learning")']

    _SECTIONS: List[Tuple[str, str]] = [
        ("研究背景与动机", "problem motivation challenge difficult to train why important"),
        ("方法与技术路线", "method architecture model design framework propose approach"),
        ("实验结果", "result experiment accuracy BLEU benchmark outperform dataset"),
        ("结论与局限", "conclusion future work limitation we presented"),
    ]

    def __init__(self, retrieval_engine: Any = None) -> None:
        super().__init__()
        self.retrieval = retrieval_engine
        self.enabled = retrieval_engine is not None

    def execute(
        self, query: str, doc_id: str = "", sections: int = 2, **_: Any
    ) -> Tuple[str, Any]:
        if self.retrieval is None:
            raise ToolError("检索模块未初始化")
        if not str(query).strip() and not doc_id:
            raise ToolError("请提供 query（论文标题关键词）或 doc_id")

        target_title = ""
        target_doc_id = doc_id
        if not target_doc_id:
            documents = self.retrieval.documents()
            matches = PaperMetadataTool._match(documents, query) if documents else []
            if matches:
                target_title = str(matches[0].get("title") or "")
                target_doc_id = str(matches[0].get("doc_id") or "")
            elif documents:
                target_title = str(documents[0].get("title") or "")
                target_doc_id = str(documents[0].get("doc_id") or "")

        probe_prefix = target_title or query
        per_section = max(1, min(int(sections or 2), 5))
        blocks: List[str] = []
        collected: Dict[str, List[Dict[str, Any]]] = {}

        for heading, probe in self._SECTIONS:
            try:
                context = self.retrieval.retrieve_with_context(
                    f"{probe_prefix} {probe}".strip(), top_k=10, mode="full"
                )
            except Exception as exc:  # noqa: BLE001
                _log.debug("摘要检索失败(%s): %s", heading, exc)
                context = {"results": []}

            items = [
                item
                for item in context.get("results", [])
                if not target_doc_id or item.get("doc_id") == target_doc_id
            ][:per_section]
            if not items:
                items = list(context.get("results", []))[:1]
            collected[heading] = items

            blocks.append(f"### {heading}")
            if not items:
                blocks.append("（知识库中未检索到与该部分相关的内容）")
            else:
                for item in items:
                    snippet = item["preview"].replace("\n", " ").strip()
                    blocks.append(f"- {snippet[:260]}【{item['source']} p.{item['page']}】")
            blocks.append("")

        header = f"# {target_title or query} —— 结构化摘要\n"
        footer = (
            "\n说明：以上四部分均摘自知识库检索到的原文片段，方括号内为来源与页码，"
            "可据此回到原文核对。未检索到内容的部分已明确标注。"
        )
        text = header + "\n" + "\n".join(blocks) + footer
        return text, {
            "doc_id": target_doc_id,
            "title": target_title or query,
            "sections": {k: len(v) for k, v in collected.items()},
            "evidence": collected,
        }


# ==============================================================================
# 6. 当前时间
# ==============================================================================
class CurrentTimeTool(BaseTool):
    """返回系统当前时间（Agent 判断"最新论文"类问题时需要）。"""

    name = "current_time"
    description = (
        "返回系统当前日期与时间。\n"
        "**适用场景**：用户问「今天几号」「现在几点」；或需要判断某篇论文相对现在是几年前的工作。\n"
        "**不适用**：不要用它来假装知道训练数据之后发生的事。"
    )
    parameters = [
        ToolParameter(
            name="format",
            type="string",
            description="输出格式：full=完整时间（默认），date=仅日期，time=仅时间，timestamp=Unix 时间戳",
            required=False,
            default="full",
            enum=["full", "date", "time", "timestamp"],
        ),
        ToolParameter(
            name="timezone_offset",
            type="integer",
            description="时区偏移小时数，默认使用本机时区",
            required=False,
            default=None,
        ),
    ]
    examples = ["current_time()", 'current_time(format="date")']

    def execute(self, format: str = "full", timezone_offset: Any = None, **_: Any) -> Tuple[str, Any]:
        now = datetime.now()
        fmt = format if format in ("full", "date", "time", "timestamp") else "full"
        if fmt == "date":
            value = now.strftime("%Y-%m-%d")
        elif fmt == "time":
            value = now.strftime("%H:%M:%S")
        elif fmt == "timestamp":
            value = str(int(now.timestamp()))
        else:
            value = now.strftime("%Y-%m-%d %H:%M:%S")
        weekday = "一二三四五六日"[now.weekday()]
        text = f"当前时间：{value}（星期{weekday}）"
        return text, {
            "iso": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timestamp": int(now.timestamp()),
            "weekday": f"星期{weekday}",
            "year": now.year,
        }


# ==============================================================================
# 7. 计算器（安全求值）
# ==============================================================================
class CalculatorTool(BaseTool):
    """安全数学表达式求值。

    **不使用 eval**：用 ``ast`` 白名单解析表达式树，只允许算术运算与少量数学函数。
    这样即使 LLM 生成恶意表达式（如 ``__import__('os').system(...)``）也无法执行。
    """

    name = "calculator"
    description = (
        "计算数学表达式，支持 + - * / // % ** 与 sqrt/log/exp/sin/cos/abs/round/min/max 等函数。\n"
        "**适用场景**：纯数值计算，例如「3.14 乘以 2.56」「(28.4-26.4)/26.4 提升百分比」。\n"
        "**重要**：把自然语言先翻译成表达式再传入，例如「28.4 比 26.4 高多少百分比」"
        "→ expression=\"(28.4-26.4)/26.4*100\"。\n"
        "**不适用**：单位换算以外的领域知识问题。"
    )
    parameters = [
        ToolParameter(
            name="expression",
            type="string",
            description="数学表达式，如 '(28.4-26.4)/26.4*100'",
            required=True,
        ),
        ToolParameter(
            name="precision",
            type="integer",
            description="结果保留小数位数，默认 4",
            required=False,
            default=4,
        ),
    ]
    examples = ['calculator(expression="(28.4-26.4)/26.4*100", precision=2)']

    # AST 白名单
    _BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    _FUNCTIONS: Dict[str, Any] = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
        "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "abs": abs, "round": round, "min": min, "max": max,
        "floor": math.floor, "ceil": math.ceil, "pow": pow, "fabs": math.fabs,
        "degrees": math.degrees, "radians": math.radians,
    }
    _CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

    def execute(self, expression: str, precision: int = 4, **_: Any) -> Tuple[str, Any]:
        raw = str(expression or "").strip()
        if not raw:
            raise ToolError("expression 不能为空")
        # 容忍常见的 Unicode 运算符与全角字符
        normalized = (
            raw.replace("×", "*").replace("÷", "/").replace("−", "-")
            .replace("^", "**").replace("，", ",").replace("（", "(").replace("）", ")")
        )
        # 去掉末尾等号
        normalized = normalized.rstrip("= ").strip()

        try:
            result = self._eval(normalized)
        except ToolError:
            raise
        except ZeroDivisionError:
            raise ToolError("除数为零，无法计算") from None
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"表达式无法解析：{raw!r}（{exc}）") from exc

        precision = max(0, min(int(precision if precision is not None else 4), 10))
        if isinstance(result, float):
            if math.isnan(result) or math.isinf(result):
                raise ToolError(f"计算结果不是有效数值（{result}）")
            shown = round(result, precision)
            if shown == int(shown) and abs(shown) < 1e15:
                shown_text = str(int(shown))
            else:
                shown_text = f"{shown:.{precision}f}".rstrip("0").rstrip(".") if "." in f"{shown:.{precision}f}" else f"{shown:.{precision}f}"
        else:
            shown, shown_text = result, str(result)

        text = f"{raw} = {shown_text}"
        return text, {"expression": raw, "normalized": normalized, "result": shown, "text": shown_text}

    def _eval(self, expression: str) -> Any:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"表达式语法错误：{expression!r}（{exc.msg}）") from exc
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError(f"不支持的常量类型：{type(node.value).__name__}")
            return node.value
        if isinstance(node, ast.BinOp):
            op = self._BIN_OPS.get(type(node.op))
            if op is None:
                raise ToolError(f"不支持的运算符：{type(node.op).__name__}")
            left, right = self._eval_node(node.left), self._eval_node(node.right)
            # 限制指数规模，防止 9**9**9 卡死进程
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ToolError("指数过大（上限 1000），已拒绝计算以防资源耗尽")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op = self._UNARY_OPS.get(type(node.op))
            if op is None:
                raise ToolError(f"不支持的一元运算符：{type(node.op).__name__}")
            return op(self._eval_node(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ToolError("只支持直接调用内置数学函数")
            func = self._FUNCTIONS.get(node.func.id)
            if func is None:
                raise ToolError(
                    f"不支持的函数 '{node.func.id}'。可用：{sorted(self._FUNCTIONS)}"
                )
            if node.keywords:
                raise ToolError("函数不支持关键字参数")
            return func(*[self._eval_node(arg) for arg in node.args])
        if isinstance(node, ast.Name):
            if node.id in self._CONSTANTS:
                return self._CONSTANTS[node.id]
            if node.id in self._FUNCTIONS:
                return self._FUNCTIONS[node.id]
            raise ToolError(f"未知标识符 '{node.id}'（可用常量：{sorted(self._CONSTANTS)}）")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(e) for e in node.elts]
        raise ToolError(f"不支持的表达式结构：{type(node).__name__}（为安全起见已拒绝求值）")


# ==============================================================================
# 8. 联网搜索（国内多源 + 学术 API）
# ==============================================================================
class WebSearchTool(BaseTool):
    """联网搜索工具 —— 面向国内网络环境的多源自动降级。

    **网页搜索**（均无需 API Key，直接复用本机网络出口）：

    ================  =====================================================
    引擎               说明 / 实测
    ================  =====================================================
    ``so360``         **默认首选**。抓取 ``www.so.com``（360 搜索），
                      国内可直连，实测稳定返回 10 条带标题/链接/摘要的结果。
    ``bing``          抓取 ``cn.bing.com``，国内可直连，作为次选。
    ``sogou``         搜狗。可能返回反爬验证页，失败时自动跳过。
    ``baidu``         百度。可能返回"安全验证"页，失败时自动跳过。
    ``duckduckgo``    国外源，国内通常不可达，仅作兜底。
    ================  =====================================================

    **学术搜索**（返回结构化论文元数据，无 Key，国内可直连）：

    ================  =====================================================
    引擎               说明
    ================  =====================================================
    ``arxiv``         抓取 arXiv 官方 Atom API，返回标题/作者/摘要/发布日期。
                      预印本覆盖全，适合找最新工作。
    ``crossref``      抓取 Crossref REST API，返回 **DOI**/期刊/年份。
                      DOI 是论文的权威唯一标识，适合精确核对文献。
    ================  =====================================================

    ``provider: auto``（默认）按 ``so360 → bing → sogou → baidu`` 顺序尝试，
    任一成功即返回；``scope: academic`` 时改用 ``arxiv → crossref``。
    """

    name = "web_search"
    description = (
        "联网搜索互联网上的公开信息，用于补充知识库之外的内容。\n"
        "**适用场景**：问「最新的 XX 进展」「某模型什么时候发布的」"
        "「帮我找关于 XX 的论文」，或核实知识库之外的公开资料。\n"
        "**参数 scope**：`web`（默认）搜索网页；`academic` 搜索学术论文并返回"
        "标题/作者/DOI/摘要，适合找文献。\n"
        "**注意**：结果来自互联网，权威性低于知识库原文。引用时必须标注为"
        "「网络来源」并给出链接，**不要为网络内容编造页码**。"
        "若与知识库原文冲突，以知识库原文为准。\n"
        "**不适用**：知识库中已有明确答案的问题（优先用 knowledge_search）。"
    )
    parameters = [
        ToolParameter(name="query", type="string", description="搜索关键词", required=True),
        ToolParameter(
            name="max_results",
            type="integer",
            description="返回结果条数，默认 5",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="scope",
            type="string",
            description="搜索范围：web=网页（默认）；academic=学术论文（arXiv + Crossref）",
            required=False,
            default="web",
            enum=["web", "academic"],
        ),
        ToolParameter(
            name="provider",
            type="string",
            description="指定引擎；留空或 auto 表示按国内可用性自动选择",
            required=False,
            default="",
            enum=["", "auto", "so360", "bing", "sogou", "baidu", "duckduckgo",
                  "arxiv", "crossref", "tavily", "serper"],
        ),
    ]
    examples = [
        'web_search(query="DeepSeek 最新模型 发布时间", max_results=5)',
        'web_search(query="attention mechanism survey", scope="academic")',
    ]

    #: 网页搜索的尝试顺序。
    #:
    #: 排序依据是**实测稳定性**而非"是否国内站点"：
    #:   必应中国（cn.bing.com）—— 实测 3/3 次成功，返回高质量中文结果
    #:                             （知乎/CSDN/DataWhale 等），延迟约 260ms
    #:   360 搜索（so.com）     —— 国内站点，但无 Cookie 裸请求下约半数概率
    #:                             返回反爬验证页（6KB 左右），故列为次选
    #:   搜狗 / 百度            —— 实测均返回验证页，仅作后备
    #:   DuckDuckGo            —— 国外源，国内超时，最后兜底
    _WEB_ENGINES = ("bing", "so360", "sogou", "baidu", "duckduckgo")
    #: 学术搜索的尝试顺序
    _ACADEMIC_ENGINES = ("arxiv", "crossref")
    #: 需要商业 API Key 的引擎
    _KEYED_ENGINES = ("tavily", "serper")

    def __init__(self, config: Any = None) -> None:
        super().__init__()
        from ..utils.config import get_config

        cfg = config or get_config()
        self.provider = str(cfg.get("web_search.provider", "auto") or "auto").lower()
        self.api_key = str(cfg.get("web_search.api_key", "") or "")
        self.max_results = int(cfg.get("web_search.max_results", 5))
        self.timeout = float(cfg.get("web_search.timeout", 15))
        self.enabled = bool(cfg.get("web_search.enabled", False))
        self.user_agent = str(
            cfg.get("web_search.user_agent", "")
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        self._disabled_reason = "" if self.enabled else "web_search.enabled=false（默认关闭）"

    # -- 主入口 -----------------------------------------------------------
    def execute(
        self,
        query: str,
        max_results: int = 5,
        scope: str = "web",
        provider: str = "",
        **_: Any,
    ) -> Tuple[str, Any]:
        if not self.enabled:
            raise ToolError(f"联网搜索功能未启用（{self._disabled_reason}）")
        text = str(query or "").strip()
        if not text:
            raise ToolError("query 不能为空")
        count = max(1, min(int(max_results or self.max_results), 10))
        scope = scope if scope in ("web", "academic") else "web"

        candidates = self._candidate_engines(provider, scope)
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        used_engine = ""

        for engine in candidates:
            fetcher = self._fetcher(engine)
            if fetcher is None:
                errors.append(f"{engine}: 不支持的引擎")
                continue
            try:
                if engine in self._KEYED_ENGINES and not self.api_key:
                    raise ToolError(f"{engine} 需要配置 web_search.api_key")
                results = fetcher(text, count)
                if results:
                    used_engine = engine
                    break
                errors.append(f"{engine}: 无结果")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{engine}: {type(exc).__name__}: {str(exc)[:70]}")
                _log.debug("搜索引擎 %s 失败: %s", engine, exc)

        if not results:
            return (
                f"联网搜索「{text}」未能获取结果（范围：{scope}）。\n"
                f"尝试过的引擎：{'、'.join(candidates)}\n"
                f"失败详情：{'; '.join(errors[:5])}\n"
                f"建议：改用 knowledge_search 查询已上传论文；"
                f"或检查本机网络是否可访问 360 搜索 / 必应。",
                {"query": text, "num": 0, "results": [], "errors": errors, "scope": scope},
            )

        scope_label = "学术论文" if scope == "academic" else "网页"
        lines = [
            f"联网搜索{scope_label}「{text}」返回 {len(results)} 条结果（引擎 {used_engine}）：",
            "",
        ]
        for idx, item in enumerate(results, start=1):
            lines.append(f"[{idx}] {item.get('title') or '无标题'}")
            if item.get("authors"):
                lines.append(f"    作者：{item['authors']}")
            if item.get("published"):
                lines.append(f"    日期：{item['published']}")
            if item.get("doi"):
                lines.append(f"    DOI：{item['doi']}")
            if item.get("snippet"):
                lines.append(f"    摘要/简介：{item['snippet']}")
            lines.append(f"    链接：{item.get('url', '')}")
            lines.append("")
        lines.append(
            "注意：以上为互联网公开信息，引用时请标注为「网络来源」并给出链接"
            "（或 DOI），不要为网络内容编造页码。若与知识库原文冲突，以知识库原文为准。"
        )
        return "\n".join(lines), {
            "query": text,
            "num": len(results),
            "scope": scope,
            "engine": used_engine,
            "results": results,
            "errors": errors,
        }

    def _candidate_engines(self, provider: str = "", scope: str = "web") -> List[str]:
        """确定引擎尝试顺序（国内优先，保留后备）。"""
        default = list(self._ACADEMIC_ENGINES if scope == "academic" else self._WEB_ENGINES)

        chosen = str(provider or "").strip().lower()
        if chosen in ("", "auto"):
            chosen = self.provider
        if chosen in ("", "auto"):
            return default
        if chosen in self._KEYED_ENGINES:
            return [chosen]
        # 显式指定：排首位，但仍保留同范围的后备引擎
        return [chosen] + [e for e in default if e != chosen]

    def _fetcher(self, engine: str) -> Optional[Any]:
        method = getattr(self, f"_search_{engine}", None)
        return method if callable(method) else None

    # -- HTTP 基础 --------------------------------------------------------
    def _fetch(self, url: str, data: Optional[bytes] = None, timeout: Optional[float] = None) -> str:
        """发一次 HTTP 请求并返回解码后的内容（自动识别 GBK）。"""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST" if data is not None else "GET"
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            raw = response.read()
        head = raw[:2048].decode("latin-1", "ignore").lower()
        charset = "gbk" if ("charset=gbk" in head or "charset=gb2312" in head) else "utf-8"
        return raw.decode(charset, errors="ignore")

    @staticmethod
    def _check_anti_bot(html: str, engine: str, min_chars: int = 6000) -> None:
        """识别反爬/验证页并抛出可读错误（让上层自动切到下一个引擎）。

        百度与搜狗在无 Cookie 的裸请求下常返回几百字节的"安全验证"页，
        若不识别就会被当成"无结果"，掩盖真实原因。
        """
        markers = ("安全验证", "请输入验证码", "验证码", "反爬", "abnormal", "captcha")
        snippet = html[:3000]
        if len(html) < min_chars or any(m in snippet for m in markers):
            raise ToolError(f"{engine} 返回反爬验证页（{len(html)} 字符），已跳过")

    # -- 网页搜索引擎 -----------------------------------------------------
    def _search_so360(self, query: str, count: int) -> List[Dict[str, Any]]:
        """360 搜索（www.so.com）—— 国内可直连的首选引擎。"""
        html = self._fetch("https://www.so.com/s?" + urllib.parse.urlencode({"q": query}))
        self._check_anti_bot(html, "360搜索")

        results: List[Dict[str, Any]] = []
        for block in re.findall(r'<li class="res-list[^"]*".*?</li>', html, re.DOTALL):
            if len(results) >= count:
                break
            # 标题链接：h3 内的第一个 <a>
            link = re.search(
                r"<h3[^>]*>\s*(?:<[^>]+>\s*)*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
                block,
                re.DOTALL,
            ) or re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            if not link:
                continue
            title = _strip_tags(link.group(2))
            if not title:
                continue
            url = _unescape_html(link.group(1))
            # 优先用 data-mdurl / data-url 里的真实地址（跳过 360 自家跳转）
            real = re.search(r'data-(?:mdurl|url)="([^"]+)"', block)
            if real:
                candidate = _unescape_html(real.group(1))
                if candidate.startswith("http") and "/link?" not in candidate:
                    url = candidate
            snippet = ""
            for pattern in (
                r'class="res-desc"[^>]*>(.*?)</p>',
                r'class="res-rich[^"]*"[^>]*>(.*?)</div>',
                r'class="res-comm-con[^"]*"[^>]*>(.*?)</div>',
            ):
                found = re.search(pattern, block, re.DOTALL)
                if found:
                    snippet = _strip_tags(found.group(1))
                    break
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "engine": "so360",
                }
            )
        return results

    def _search_bing(self, query: str, count: int) -> List[Dict[str, Any]]:
        """必应中国（cn.bing.com）—— 次选，国内可直连。"""
        results: List[Dict[str, Any]] = []
        for host in ("https://cn.bing.com/search?", "https://www.bing.com/search?"):
            try:
                html = self._fetch(host + urllib.parse.urlencode({"q": query, "ensearch": "0"}))
            except Exception as exc:  # noqa: BLE001
                _log.debug("bing 站点 %s 失败: %s", host, exc)
                continue
            for block in re.findall(r'<li class="b_algo".*?</li>', html, re.DOTALL):
                if len(results) >= count:
                    break
                link = re.search(
                    r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL
                )
                if not link:
                    continue
                url = _unescape_html(link.group(1))
                if not url.startswith("http"):
                    continue
                snippet = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)
                results.append(
                    {
                        "title": _strip_tags(link.group(2)),
                        "url": url,
                        "snippet": _strip_tags(snippet.group(1)) if snippet else "",
                        "engine": "bing",
                    }
                )
            if results:
                return results
        return results

    def _search_sogou(self, query: str, count: int) -> List[Dict[str, Any]]:
        """搜狗（www.sogou.com）—— 可能返回验证页。"""
        html = self._fetch("https://www.sogou.com/web?" + urllib.parse.urlencode({"query": query}))
        self._check_anti_bot(html, "搜狗")

        results: List[Dict[str, Any]] = []
        for block in re.findall(r'<div class="vrwrap".*?</div>\s*</div>', html, re.DOTALL):
            if len(results) >= count:
                break
            link = re.search(
                r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL
            )
            if not link:
                continue
            title = _strip_tags(link.group(2))
            if not title:
                continue
            snippet = re.search(r'class="(?:str-text-info|space-txt|fz-mid)"[^>]*>(.*?)</div>',
                                block, re.DOTALL)
            url = _unescape_html(link.group(1))
            if url.startswith("/"):
                url = "https://www.sogou.com" + url
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": _strip_tags(snippet.group(1)) if snippet else "",
                    "engine": "sogou",
                }
            )
        return results

    def _search_baidu(self, query: str, count: int) -> List[Dict[str, Any]]:
        """百度（www.baidu.com）—— 可能返回"安全验证"页。"""
        html = self._fetch("https://www.baidu.com/s?" + urllib.parse.urlencode({"wd": query}))
        self._check_anti_bot(html, "百度")

        titles = re.findall(
            r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL
        )
        snippets = re.findall(
            r'<span[^>]+class="[^"]*(?:content-right|c-abstract)[^"]*"[^>]*>(.*?)</span>',
            html,
            re.DOTALL,
        )
        results: List[Dict[str, Any]] = []
        for idx, (href, title) in enumerate(titles):
            if len(results) >= count:
                break
            clean_title = _strip_tags(title)
            if not clean_title:
                continue
            results.append(
                {
                    "title": clean_title,
                    "url": _unescape_html(href),
                    "snippet": _strip_tags(snippets[idx]) if idx < len(snippets) else "",
                    "engine": "baidu",
                }
            )
        return results

    def _search_duckduckgo(self, query: str, count: int) -> List[Dict[str, Any]]:
        """DuckDuckGo（国外源，国内通常不可达，仅兜底）。"""
        html = self._fetch(
            "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        )
        results: List[Dict[str, Any]] = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
            r"(?:(?!result__a).)*?class=\"result__snippet\"[^>]*>(?P<snippet>.*?)</a>",
            re.DOTALL,
        )
        for match in pattern.finditer(html):
            if len(results) >= count:
                break
            results.append(
                {
                    "title": _strip_tags(match.group("title")),
                    "url": urllib.parse.unquote(_unescape_html(match.group("url"))),
                    "snippet": _strip_tags(match.group("snippet")),
                    "engine": "duckduckgo",
                }
            )
        return results

    # -- 学术搜索引擎 -----------------------------------------------------
    def _search_arxiv(self, query: str, count: int) -> List[Dict[str, Any]]:
        """arXiv 官方 Atom API —— 返回标题/作者/摘要/发布日期。"""
        import xml.etree.ElementTree as ET

        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "max_results": max(1, min(count, 10)),
                "sortBy": "relevance",
            }
        )
        content = self._fetch(url, timeout=max(self.timeout, 25))
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ToolError(f"arXiv 返回内容无法解析: {exc}") from exc

        ns = {"a": "http://www.w3.org/2005/Atom"}
        results: List[Dict[str, Any]] = []
        for entry in root.findall("a:entry", ns):
            title = _strip_tags((entry.findtext("a:title", "", ns) or "").replace("\n", " "))
            summary = _strip_tags((entry.findtext("a:summary", "", ns) or "").replace("\n", " "))
            authors = ", ".join(
                a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)[:3]
            )
            results.append(
                {
                    "title": title,
                    "url": entry.findtext("a:id", "", ns),
                    "snippet": summary[:400],
                    "authors": authors,
                    "published": (entry.findtext("a:published", "", ns) or "")[:10],
                    "engine": "arxiv",
                }
            )
        return results[:count]

    def _search_crossref(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Crossref REST API —— 返回 DOI/期刊/年份等权威元数据。"""
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
            {
                "query": query,
                "rows": max(1, min(count, 10)),
                "select": "title,author,issued,DOI,container-title,abstract,URL",
            }
        )
        content = self._fetch(url, timeout=max(self.timeout, 25))
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ToolError(f"Crossref 返回内容无法解析: {exc}") from exc

        results: List[Dict[str, Any]] = []
        for item in (payload.get("message") or {}).get("items", []):
            titles = item.get("title") or []
            title = _strip_tags(titles[0]) if titles else ""
            if not title:
                continue
            authors = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in (item.get("author") or [])[:3]
            )
            parts = (item.get("issued") or {}).get("date-parts") or [[None]]
            year = parts[0][0] if parts and parts[0] else None
            results.append(
                {
                    "title": title,
                    "url": item.get("URL") or (
                        f"https://doi.org/{item['DOI']}" if item.get("DOI") else ""
                    ),
                    "snippet": _strip_tags(item.get("abstract") or "")[:400],
                    "authors": authors,
                    "published": str(year) if year else "",
                    "doi": item.get("DOI", ""),
                    "venue": (item.get("container-title") or [""])[0],
                    "engine": "crossref",
                }
            )
        return results[:count]

    # -- 商业 API（可选）--------------------------------------------------
    def _search_tavily(self, query: str, count: int) -> List[Dict[str, Any]]:
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps({"api_key": self.api_key, "query": query, "max_results": count}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": self.user_agent},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
                "engine": "tavily",
            }
            for r in (body.get("results") or [])[:count]
        ]

    def _search_serper(self, query: str, count: int) -> List[Dict[str, Any]]:
        request = urllib.request.Request(
            "https://google.serper.dev/search",
            data=json.dumps({"q": query, "num": count}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ToolError(f"Serper HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ToolError(f"Serper 请求失败：{exc}") from exc
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", ""),
                "engine": "serper",
            }
            for r in (body.get("organic") or [])[:count]
        ]

    # -- 可用性探测 -------------------------------------------------------
    def probe(self, scope: str = "web") -> Dict[str, Any]:
        """探测各引擎在当前网络下的可达性（供 /api/health 与排障使用）。"""
        engines = self._ACADEMIC_ENGINES if scope == "academic" else self._WEB_ENGINES
        report: Dict[str, Any] = {
            "enabled": self.enabled,
            "provider": self.provider,
            "scope": scope,
            "engines": {},
        }
        if not self.enabled:
            report["note"] = "web_search.enabled=false"
            return report
        for engine in engines:
            fetcher = self._fetcher(engine)
            if fetcher is None:
                continue
            try:
                hits = fetcher("Transformer attention mechanism", 2)
                report["engines"][engine] = {"ok": bool(hits), "num": len(hits)}
            except Exception as exc:  # noqa: BLE001
                report["engines"][engine] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:60]}",
                }
        report["ok"] = any(v.get("ok") for v in report["engines"].values())
        return report


def _strip_tags(html: str) -> str:
    """去掉 HTML 标签并反转义实体。"""
    return _unescape_html(re.sub(r"<[^>]+>", "", html or "")).strip()


def _unescape_html(text: str) -> str:
    """反转义常见 HTML 实体并压缩空白。"""
    for entity, char in (
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
        ("&#x27;", "'"), ("&#39;", "'"), ("&nbsp;", " "), ("&ensp;", " "),
        ("&emsp;", " "), ("&#0183;", "·"), ("&#12290;", "。"),
    ):
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()






# ==============================================================================
# 工厂
# ==============================================================================
def build_default_registry(
    retrieval_engine: Any = None,
    config: Any = None,
    rag_pipeline: Any = None,
    llm: Any = None,
) -> Any:
    """构建包含全部 8 个工具的注册表。

    Parameters
    ----------
    retrieval_engine: 模块一检索引擎（knowledge_search / paper_metadata /
                      paper_compare / summarize_paper / keyword_extract 依赖它）
    config: 全局配置
    rag_pipeline: 模块二 RAG 流水线（保留参数，便于将来把 knowledge_search
                  切换到"检索+生成"一体化的实现）
    llm: LLM 客户端（保留参数，供将来需要 LLM 参与的工具使用）
    """
    from .base_tool import ToolRegistry

    registry = ToolRegistry()
    tools: List[BaseTool] = [
        KnowledgeSearchTool(retrieval_engine),
        PaperMetadataTool(retrieval_engine),
        PaperCompareTool(retrieval_engine),
        KeywordExtractTool(retrieval_engine),
        SummarizePaperTool(retrieval_engine),
        CurrentTimeTool(),
        CalculatorTool(),
        WebSearchTool(config),
    ]
    for tool in tools:
        registry.register(tool)
    _log.info(
        "工具集已构建：%d 个工具（%d 个可用）",
        len(registry),
        len(registry.enabled_tools()),
    )
    return registry
