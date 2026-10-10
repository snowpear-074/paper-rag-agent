"""工具选择路由（模块三 5.3.3）。

为什么需要路由
--------------
朴素 ReAct 的每一轮都要把**全部工具**的完整说明塞进提示词，代价有三：

1. **Token 开销**：8 个工具的详细说明约 1200 token，每轮都要重发；
2. **选择噪声**：候选工具越多，模型选错的概率越高（尤其"该不该检索"这类判断）；
3. **无谓轮次**：对"3.14 乘 2.56"这种问题，模型常会先检索一轮才发现该用计算器。

本模块用**一次廉价的意图分类**解决这三个问题：
先判断问题属于哪一类，把候选工具集从 8 个缩小到 1~3 个，
并对"无需工具"的问题直接跳过整个循环。

两种路由实现
------------
- :class:`RuleRouter`（默认）：**零成本关键词规则**。基于中文/英文关键词与
  正则模式打分，无需调用 LLM，耗时 < 1ms。对科研场景的常见问法准确率已足够高。
- :class:`LLMRouter`：调用一次 LLM 做分类，准确率更高但每次多花约 200 token
  与 0.5~2s 延迟。可通过 ``agent.router_provider=llm`` 启用。

两者都输出统一结构，便于在评测中对比（模块五的"路由决策准确率"指标）。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm import LLMError, extract_json
from ..utils.logger import get_logger
from ..utils.types import Message

__all__ = ["RuleRouter", "LLMRouter", "NullRouter", "create_router", "RouteDecision"]

_log = get_logger("agent.router")


@dataclass
class RouteDecision:
    """路由结果。"""

    routes: List[str] = field(default_factory=list)      # 命中的路径（工具名或 direct）
    tools: List[str] = field(default_factory=list)       # 建议的候选工具集
    reason: str = ""
    confidence: float = 0.0
    direct: bool = False                                 # 是否可跳过工具直接作答
    latency_ms: float = 0.0
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routes": self.routes,
            "tools": self.tools,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "direct": self.direct,
            "latency_ms": round(self.latency_ms, 2),
            "provider": self.provider,
        }


# ==============================================================================
# 规则路由
# ==============================================================================
class RuleRouter:
    """基于关键词与正则的零成本意图路由。

    评分机制：每条路径维护一组带权关键词/正则，命中即累加；
    取得分最高的 1~2 条路径作为候选。全部未命中时**回退到知识库检索**
    （科研问答的默认动作，宁可多检索一轮也不要漏答）。
    """

    provider = "rule"

    # 路径 -> [(模式, 权重), ...]
    _PATTERNS: Dict[str, List[Tuple[str, float]]] = {
        "calculator": [
            (r"\d+\s*[\+\-\*/×÷\^]\s*\d+", 4.0),
            (r"(计算|算一下|算算|求值|等于多少|结果是多少)", 3.0),
            (r"(相乘|相除|相加|相减|乘以|除以|加上|减去|的平方|开方|开根号)", 3.0),
            (r"(百分比|占比|提升了多少|高出多少|增长率)", 2.0),
            (r"\b(sqrt|log|exp|sin|cos)\s*\(", 2.5),
        ],
        "current_time": [
            (r"(现在几点|当前时间|现在时间|今天几号|今天是几号|今天日期|当前日期|今天是星期)", 5.0),
            (r"(现在是什么时候|今年是哪一年|今年是几几年)", 4.0),
        ],
        "paper_metadata": [
            (r"(谁写的|作者是谁|作者有哪|哪些作者|第一作者|通讯作者)", 4.0),
            (r"(哪一年发表|发表年份|什么时候发表|发表在|发表于|期刊|会议名|收录于)", 3.5),
            (r"\b(doi|DOI)\b", 4.0),
            (r"(论文标题|标题是什么|叫什么名字)", 3.0),
            (r"(有哪些论文|知识库里有什么|收录了哪些)", 3.0),
        ],
        "paper_compare": [
            (r"(对比|比较|相比|区别|差异|不同之处|哪个更好|哪个更优|优劣)", 4.0),
            (r"\b(compare|comparison|versus|vs\.?|difference between)\b", 3.5),
            (r"(和.*相比|与.*相比|A 和 B)", 2.0),
        ],
        "summarize_paper": [
            (r"(总结|概括|摘要|综述|讲了什么|主要讲|全文大意|核心内容)", 4.0),
            (r"\b(summarize|summary|overview|abstract of)\b", 3.5),
            (r"(这篇论文|这篇文章|该论文).{0,6}(内容|说了什么|贡献)", 3.0),
        ],
        "keyword_extract": [
            (r"(关键词|关键字|核心词|提取词|术语)", 4.5),
            (r"\b(keywords?|key terms?)\b", 3.5),
        ],
        "web_search": [
            # —— 时效性诉求 ——
            (r"(最新|近期|最近发表|今年新出|有什么新|前沿进展|最新进展|最新研究)", 3.5),
            (r"(20\d{2})\s*年?\s*.{0,12}(新|最新|进展|动态|论文|工作|研究|会议)", 3.5),
            (r"\b(latest|recent|state of the art in \d{4}|newly released)\b", 3.0),
            # —— "寻找/推荐外部文献"诉求 ——
            # 关键区分：问"知识库里的论文 A 说了什么" → knowledge_search；
            #           问"帮我找/推荐几篇关于 X 的论文" → web_search。
            # 早期版本缺少这组规则，导致后者被误判为知识库检索，
            # Agent 只在本地库里空查两轮、最后只能回答"知识库中没有"。
            # 注意动词与"论文/文献"之间可能有很长的定语，因此用非贪婪跨词匹配。
            (r"(找|检索|搜索|搜一下|查一下|查询|推荐|提供|列出|给我|搜集|调研).{0,40}?"
             r"(论文|文献|文章|资料|工作|研究|综述)", 4.5),
            # 注意：把"有没有"放在"有"前面，否则正则先匹配到单个"有"就停止了
            (r"(有没有|有哪些|有木有|是否存在|是否已有|有).{0,20}?(论文|文献|工作|研究|资料)", 3.5),
            (r"(相关文献|相关论文|参考文献|文献检索|文献调研|文献综述)", 4.0),
            (r"\b(find|search for|look up|recommend|papers about|related work on|literature review)\b", 3.0),
            (r"(网上|互联网|外部|公开渠道|外部资料)", 3.5),
        ],
        "knowledge_search": [
            (r"(方法|模型|架构|结构|算法|机制|原理|设计)", 2.0),
            (r"(数据集|实验|结果|指标|准确率|性能|效果|对比实验|消融)", 2.0),
            (r"(为什么|如何|怎么|怎样|是什么|什么是|解释|说明)", 1.5),
            (r"(论文|文献|原文|文中|作者提出|该文)", 2.5),
            (r"\b(method|model|architecture|dataset|experiment|result|why|how|what)\b", 1.5),
        ],
        # "direct" 是兜底路径，由下面的 _DIRECT_PATTERNS 单独判定
    }

    # 明确不需要工具的寒暄/闲聊
    _DIRECT_PATTERNS: List[Tuple[str, float]] = [
        (r"^(你好|您好|hi|hello|hey|在吗|早上好|下午好|晚上好)[\s!！。~]*$", 5.0),
        (r"^(谢谢|多谢|感谢|thanks|thank you|辛苦了)[\s!！。~]*$", 5.0),
        (r"^(你是谁|你叫什么|介绍一下你自己|你能做什么|你会什么|帮助|help)[\s?？]*$", 4.5),
        (r"^(再见|拜拜|bye|goodbye)[\s!！。~]*$", 4.5),
        (r"^(嗯|哦|好的|ok|OK|收到)[\s!！。~]*$", 4.0),
    ]

    # 各路径对应的候选工具集（路由命中后渲染给模型）
    _TOOL_MAP: Dict[str, List[str]] = {
        "knowledge_search": ["knowledge_search"],
        "paper_metadata": ["paper_metadata", "knowledge_search"],
        "paper_compare": ["paper_compare", "paper_metadata", "knowledge_search"],
        "summarize_paper": ["summarize_paper", "knowledge_search"],
        "keyword_extract": ["keyword_extract", "knowledge_search"],
        "calculator": ["calculator"],
        "current_time": ["current_time"],
        "web_search": ["web_search", "knowledge_search"],
        "direct": ["calculator", "current_time", "knowledge_search"],
    }

    def __init__(self, min_score: float = 2.0, max_routes: int = 2) -> None:
        self.min_score = float(min_score)
        self.max_routes = int(max_routes)
        self._compiled: Dict[str, List[Tuple[re.Pattern, float]]] = {
            route: [(re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in patterns]
            for route, patterns in self._PATTERNS.items()
        }
        self._direct_compiled = [
            (re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in self._DIRECT_PATTERNS
        ]
        self._stats = {"routed": 0, "direct": 0, "fallback": 0}
        self._last: Optional[RouteDecision] = None

    # -- 主接口 -----------------------------------------------------------
    def route(
        self, question: str, history: Optional[Sequence[Message]] = None
    ) -> Dict[str, Any]:
        """返回路由决策字典（供 Agent 直接消费）。

        注意返回的是 :meth:`RouteDecision.to_dict`，其中包含 ``direct`` 标记 ——
        Agent 依赖它来判断是否可以跳过整个 ReAct 循环直接作答。
        """
        began = time.perf_counter()
        return self._route(question, began).to_dict()

    def decide(
        self, question: str, history: Optional[Sequence[Message]] = None
    ) -> RouteDecision:
        """返回 :class:`RouteDecision` 对象（便于评测与单元测试）。"""
        return self._route(question, time.perf_counter())

    def _route(self, question: str, began: float) -> RouteDecision:
        self._stats["routed"] += 1
        text = (question or "").strip()
        decision = RouteDecision(provider=self.provider)

        if not text:
            decision.latency_ms = (time.perf_counter() - began) * 1000
            self._last = decision
            return decision

        # 1) 寒暄/闲聊 → 直接作答
        direct_score = sum(
            weight for pattern, weight in self._direct_compiled if pattern.search(text)
        )
        if direct_score >= 4.0:
            self._stats["direct"] += 1
            decision.routes = ["direct"]
            decision.tools = list(self._TOOL_MAP["direct"])
            decision.reason = "寒暄或自我介绍类问题，无需调用工具"
            decision.confidence = min(1.0, direct_score / 5.0)
            decision.direct = True
            decision.latency_ms = (time.perf_counter() - began) * 1000
            self._last = decision
            return decision

        # 2) 逐路径打分
        scores: Dict[str, float] = {}
        for route, patterns in self._compiled.items():
            total = sum(weight for pattern, weight in patterns if pattern.search(text))
            if total > 0:
                scores[route] = total

        # 3) 短问题里出现算式 → 强烈倾斜计算器
        if re.search(r"\d+\s*[\+\-\*/×÷\^]\s*\d+", text) and len(text) < 60:
            scores["calculator"] = scores.get("calculator", 0.0) + 3.0
            # 纯计算问题不应再触发检索
            if scores.get("calculator", 0) >= 4.0:
                scores.pop("knowledge_search", None)

        if not scores or max(scores.values()) < self.min_score:
            # 兜底：默认走知识库检索
            self._stats["fallback"] += 1
            decision.routes = ["knowledge_search"]
            decision.tools = list(self._TOOL_MAP["knowledge_search"])
            decision.reason = "未匹配到明确意图，按默认策略走知识库检索"
            decision.confidence = 0.4
            decision.latency_ms = (time.perf_counter() - began) * 1000
            self._last = decision
            return decision

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_routes = [route for route, _ in ranked[: self.max_routes]]
        top_score = ranked[0][1]

        # 5) 显式检索意图优先
        #
        # 当用户明确说"帮我找/推荐几篇论文"时，即便"论文"一词同时触发了
        # knowledge_search（分值可能更高），真实意图也是**找外部新文献**。
        # 若不在这里纠偏，Agent 会只在本地库里空查，然后回答"知识库中没有" ——
        # 这正是早期实测踩到的问题（用户明明要联网找文献，系统却去找已上传的）。
        explicit = self._compiled.get("web_search")
        if explicit and any(pattern.search(text) for pattern, _w in explicit):
            strong = any(
                pattern.search(text)
                for pattern, weight in explicit
                if weight >= 3.5
            )
            if strong and "web_search" not in top_routes:
                top_routes = ["web_search"] + [r for r in top_routes if r != "web_search"]
                top_routes = top_routes[: self.max_routes]
                scores["web_search"] = max(scores.get("web_search", 0.0), top_score + 0.5)
                top_score = scores["web_search"]
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])

        # 6) 候选工具集：取并集（保持顺序、去重）
        tools: List[str] = []
        for route in top_routes:
            for tool in self._TOOL_MAP.get(route, []):
                if tool not in tools:
                    tools.append(tool)

        decision.routes = top_routes
        decision.tools = tools
        decision.reason = "；".join(
            f"{route}({scores.get(route, 0.0):.1f})" for route in top_routes
        )
        # 置信度：Top-1 得分相对"该路径满分"的归一 + 领先第二名程度
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = (top_score - second) / top_score if top_score > 0 else 0.0
        decision.confidence = round(min(1.0, 0.55 + 0.45 * margin), 4)
        decision.latency_ms = (time.perf_counter() - began) * 1000
        self._last = decision
        return decision

    # -- 状态 -------------------------------------------------------------
    @property
    def last(self) -> Optional[RouteDecision]:
        return self._last

    def stats(self) -> Dict[str, Any]:
        total = max(1, self._stats["routed"])
        return {
            "provider": self.provider,
            **self._stats,
            "direct_rate": round(self._stats["direct"] / total, 4),
            "fallback_rate": round(self._stats["fallback"] / total, 4),
            "min_score": self.min_score,
            "max_routes": self.max_routes,
        }

    def health_check(self) -> Dict[str, Any]:
        data = self.stats()
        data["ok"] = True
        return data


# ==============================================================================
# LLM 路由
# ==============================================================================
class LLMRouter(RuleRouter):
    """调用一次 LLM 做意图分类（准确率更高，成本也更高）。

    继承 :class:`RuleRouter` —— 当 LLM 调用失败或返回不可解析结果时，
    自动回退到规则路由，保证路由环节**永不阻塞**主流程。
    """

    provider = "llm"

    def __init__(
        self,
        llm: Any,
        registry: Optional[Any] = None,
        min_score: float = 2.0,
        max_routes: int = 2,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(min_score=min_score, max_routes=max_routes)
        self.llm = llm
        self.registry = registry
        self.temperature = temperature
        self._llm_stats = {"calls": 0, "failures": 0, "fallbacks": 0}

    def _route(self, question: str, began: float) -> RouteDecision:
        text = (question or "").strip()
        if not text:
            return super()._route(question, began)

        from .prompts import build_route_prompt

        try:
            self._llm_stats["calls"] += 1
            reply = self.llm.chat(
                [Message.user(build_route_prompt(text))],
                temperature=self.temperature,
                max_tokens=200,
            )
            parsed = extract_json(reply) or {}
            routes = parsed.get("routes") or parsed.get("route") or []
            if isinstance(routes, str):
                routes = [routes]
            routes = [str(r).strip() for r in routes if str(r).strip()]
            if not routes:
                raise ValueError(f"路由结果缺少 routes 字段：{reply[:120]}")

            decision = RouteDecision(provider=self.provider)
            decision.routes = routes[: self.max_routes]
            decision.reason = str(parsed.get("reason") or "LLM 意图分类结果")[:200]
            try:
                decision.confidence = float(parsed.get("confidence", 0.8))
            except (TypeError, ValueError):
                decision.confidence = 0.8
            decision.confidence = max(0.0, min(1.0, decision.confidence))

            tools: List[str] = []
            for route in decision.routes:
                for tool in self._TOOL_MAP.get(route, []):
                    if tool not in tools:
                        tools.append(tool)
            decision.tools = tools or list(self._TOOL_MAP["knowledge_search"])
            decision.direct = "direct" in decision.routes and decision.confidence >= 0.75
            decision.latency_ms = (time.perf_counter() - began) * 1000
            self._last = decision
            return decision
        except (LLMError, ValueError, KeyError) as exc:
            self._llm_stats["failures"] += 1
            _log.warning("LLM 路由失败（回退规则路由）: %s", exc)
            decision = super()._route(question, began)
            decision.reason = f"[回退规则路由] {decision.reason}"
            self._llm_stats["fallbacks"] += 1
            return decision

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        total = max(1, self._llm_stats["calls"])
        data.update(self._llm_stats)
        data["llm_failure_rate"] = round(self._llm_stats["failures"] / total, 4)
        return data


# ==============================================================================
# 空路由（关闭路由）
# ==============================================================================
class NullRouter:
    """关闭路由时的占位路由器：不限制工具、不判定"直接作答"。

    为什么需要这个类
    ----------------
    早期实现用"把 ``RuleRouter.min_score`` 调到 1e9"来关闭路由。结果适得其反：
    ``_route`` 因为所有分数都低于阈值而走到**兜底分支**，返回
    ``tools=["knowledge_search", ...]``。Agent 拿到这个非空候选集后会把它当作
    **工具白名单**渲染进 System Prompt —— 想关掉路由，实际却把模型的可用工具
    砍到只剩检索一类，计算器、时间、论文对比全部消失，且没有任何报错。

    正确做法是返回**空候选集**：``ReActAgent`` 约定 ``candidate_tools`` 为空
    时渲染完整工具目录，模型可按自己的判断调用任何工具。
    """

    provider = "none"

    def __init__(self) -> None:
        self._stats = {"routed": 0, "direct": 0, "fallback": 0}
        self._last: Optional[RouteDecision] = None

    def route(
        self, question: str, history: Optional[Sequence[Message]] = None
    ) -> Dict[str, Any]:
        return self.decide(question, history).to_dict()

    def decide(
        self, question: str, history: Optional[Sequence[Message]] = None
    ) -> RouteDecision:
        self._stats["routed"] += 1
        decision = RouteDecision(
            provider=self.provider,
            routes=[],
            tools=[],  # 空 = 不限制，Agent 渲染完整工具目录
            reason="意图路由已关闭（agent.router_provider=none）",
            confidence=0.0,
            direct=False,
        )
        self._last = decision
        return decision

    @property
    def last(self) -> Optional[RouteDecision]:
        return self._last

    def stats(self) -> Dict[str, Any]:
        return {"provider": self.provider, **self._stats, "disabled": True}

    def health_check(self) -> Dict[str, Any]:
        data = self.stats()
        data["ok"] = True
        return data


# ==============================================================================
# 工厂
# ==============================================================================
def create_router(
    config: Any = None,
    llm: Any = None,
    registry: Optional[Any] = None,
) -> Any:
    """按配置创建路由器。

    ``agent.router_provider`` 可选值：

    - ``rule``（默认）：零成本规则路由
    - ``llm``：LLM 分类路由；``llm`` 为 None 时自动降级为规则路由
    - ``none``：关闭路由 —— 返回 :class:`NullRouter`，不给出候选工具，
      也不做"直接作答"判定，所有问题都交给完整的 ReAct 循环
    """
    from ..utils.config import get_config

    cfg = config or get_config()
    provider = str(cfg.get("agent.router_provider", "rule") or "rule").lower()

    if provider in ("none", "off", "disabled"):
        _log.info("意图路由已关闭（agent.router_provider=none）")
        return NullRouter()

    if provider == "llm" and llm is not None:
        _log.info("使用 LLM 意图路由")
        return LLMRouter(llm=llm, registry=registry)

    return RuleRouter()
