# -*- coding: utf-8 -*-
"""模块三（Agent 决策层）测试：工具框架、8 个工具、ReAct 循环、
意图路由、错误恢复、并行调用、死循环检测、对话记忆。

运行::

    python -m unittest tests.test_agent -v
"""

from __future__ import annotations

import time
import unittest

from tests.common import TempWorkspace, sample_documents


# ===========================================================================
# 工具框架
# ===========================================================================
class ToolFrameworkTest(unittest.TestCase):
    """参数校验、类型转换、异常封装、schema 生成。"""

    def _tool(self):
        from src.agent.base_tool import BaseTool, ToolParameter

        class Demo(BaseTool):
            name = "demo"
            description = "演示工具"
            parameters = [
                ToolParameter(name="text", type="string", required=True, description="文本"),
                ToolParameter(name="count", type="integer", required=False, default=3),
                ToolParameter(name="flag", type="boolean", required=False, default=False),
                ToolParameter(name="tags", type="array", required=False, default=None),
            ]

            def execute(self, text, count=3, flag=False, tags=None, **kwargs):
                if text == "boom":
                    raise RuntimeError("模拟内部错误")
                return f"text={text} count={count} flag={flag} tags={tags}"

        return Demo()

    def test_missing_required_parameter(self):
        from src.utils.types import ToolStatus

        result = self._tool().run()
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("text", result.content)
        self.assertIn("必填", result.content)

    def test_type_coercion(self):
        tool = self._tool()
        result = tool.run(text="hi", count="7", flag="true", tags="a,b,c")
        self.assertTrue(result.ok, result.content)
        self.assertIn("count=7", result.content)
        self.assertIn("flag=True", result.content)
        self.assertIn("tags=['a', 'b', 'c']", result.content)

    def test_array_from_json_string(self):
        result = self._tool().run(text="x", tags='["p", "q"]')
        self.assertTrue(result.ok)
        self.assertIn("['p', 'q']", result.content)

    def test_defaults_applied(self):
        result = self._tool().run(text="x")
        self.assertIn("count=3", result.content)
        self.assertIn("flag=False", result.content)

    def test_exception_becomes_error_result(self):
        """工具内部异常必须转成 ERROR 结果，绝不允许抛给 Agent。"""
        from src.utils.types import ToolStatus

        result = self._tool().run(text="boom")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("模拟内部错误", result.content)
        self.assertGreater(result.latency_ms, 0)

    def test_undeclared_parameters_ignored(self):
        result = self._tool().run(text="x", unknown_param="whatever")
        self.assertTrue(result.ok, "未声明参数应被静默忽略，而不是报错")

    def test_run_merges_call_args(self):
        """run(ToolCall) 必须把 call.args 展开为工具参数（回归测试）。

        真实踩坑：早期实现只取 kwargs，完全忽略了 call.args，
        导致所有经 ToolCall 发起的调用都报"缺少必填参数"。
        """
        from src.utils.types import ToolCall

        tool = self._tool()
        result = tool.run(ToolCall(call_id="c1", name="demo", args={"text": "来自 call", "count": 9}))
        self.assertTrue(result.ok, result.content)
        self.assertIn("来自 call", result.content)
        self.assertIn("count=9", result.content)

    def test_explicit_kwargs_override_call_args(self):
        from src.utils.types import ToolCall

        result = self._tool().run(ToolCall(call_id="c1", name="demo", args={"text": "a"}), text="b")
        self.assertIn("text=b", result.content)

    def test_to_schema(self):
        schema = self._tool().to_schema()
        self.assertEqual(schema["name"], "demo")
        self.assertEqual(schema["parameters"]["required"], ["text"])
        self.assertEqual(schema["parameters"]["properties"]["count"]["type"], "integer")

    def test_render_for_prompt(self):
        text = self._tool().render_for_prompt()
        self.assertIn("### demo", text)
        self.assertIn("必填", text)
        self.assertIn("可选", text)

    def test_stats(self):
        tool = self._tool()
        tool.run(text="ok")
        tool.run(text="boom")
        stats = tool.stats()
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["failures"], 1)
        self.assertAlmostEqual(stats["success_rate"], 0.5)


class ToolRegistryTest(unittest.TestCase):
    """注册、查找、执行与提示词渲染。"""

    def _registry(self):
        from src.agent.base_tool import ToolRegistry
        from src.agent.tools import CalculatorTool, CurrentTimeTool

        return ToolRegistry([CalculatorTool(), CurrentTimeTool()])

    def test_register_and_names(self):
        registry = self._registry()
        self.assertEqual(len(registry), 2)
        self.assertIn("calculator", registry.names)
        self.assertIn("calculator", registry)

    def test_duplicate_name_raises(self):
        from src.agent.tools import CalculatorTool

        registry = self._registry()
        with self.assertRaises(ValueError):
            registry.register(CalculatorTool())
        registry.register(CalculatorTool(), override=True)  # 显式覆盖则允许

    def test_execute_unknown_tool_returns_error(self):
        from src.utils.types import ToolCall, ToolStatus

        result = self._registry().execute(ToolCall(name="no_such_tool", args={}))
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("不存在", result.content)
        self.assertIn("calculator", result.content, "应列出可用工具帮助模型改道")

    def test_execute_dispatches(self):
        from src.utils.types import ToolCall

        result = self._registry().execute(
            ToolCall(name="calculator", args={"expression": "2+3"})
        )
        self.assertTrue(result.ok)
        self.assertIn("5", result.content)

    def test_disabled_tool_returns_actionable_error(self):
        """依赖未就绪的工具应返回带指引的 ERROR（而非空洞的 SKIPPED）。

        默认配置已启用联网搜索；这里显式用 ``enabled=False`` 的配置构造工具，
        以验证"关闭后仍给出可操作指引"这一行为。
        """
        from src.agent.base_tool import ToolRegistry
        from src.agent.tools import WebSearchTool
        from src.utils.types import ToolCall, ToolStatus

        with TempWorkspace() as ws:
            tool = WebSearchTool(ws.config(web_search={"enabled": False}))
            self.assertFalse(tool.enabled)
            registry = ToolRegistry([tool])
            result = registry.execute(ToolCall(name="web_search", args={"query": "x"}))
            self.assertEqual(result.status, ToolStatus.ERROR)
            self.assertIn("未启用", result.content)
            self.assertIn("web_search.enabled", result.content)

    def test_web_search_enabled_by_default(self):
        """默认配置下联网搜索应处于可用状态，且带多引擎降级链与学术源。"""
        from src.agent.tools import WebSearchTool

        tool = WebSearchTool()
        self.assertTrue(tool.enabled)
        self.assertEqual(tool.provider, "auto")
        # 网页搜索：必应中国为首选（实测最稳定），且必须有后备引擎
        self.assertIn("bing", tool._WEB_ENGINES)
        self.assertGreaterEqual(len(tool._WEB_ENGINES), 3, "至少要有三个网页引擎可互相兜底")
        # 学术搜索：arXiv + Crossref（返回 DOI 等权威元数据）
        self.assertEqual(set(tool._ACADEMIC_ENGINES), {"arxiv", "crossref"})

    def test_web_search_engine_selection_by_scope(self):
        """scope=academic 应只走学术源，scope=web 应走网页源。"""
        from src.agent.tools import WebSearchTool

        tool = WebSearchTool()
        self.assertEqual(tool._candidate_engines("", "academic"), ["arxiv", "crossref"])
        self.assertEqual(tool._candidate_engines("", "web"), list(tool._WEB_ENGINES))
        # 显式指定某引擎时排首位，但仍保留后备
        self.assertEqual(tool._candidate_engines("so360", "web")[0], "so360")
        self.assertIn("bing", tool._candidate_engines("so360", "web"))
        # 商业 API 只试它自己（避免降级到抓取式引擎造成行为不一致）
        self.assertEqual(tool._candidate_engines("tavily", "web"), ["tavily"])

    def test_web_search_detects_anti_bot_page(self):
        """反爬验证页必须被识别为失败，而不是当成"无结果"。"""
        from src.agent.tools import WebSearchTool
        from src.agent.base_tool import ToolError

        tool = WebSearchTool()
        with self.assertRaises(ToolError) as ctx:
            tool._check_anti_bot("<html>百度安全验证 网络不给力</html>", "百度")
        self.assertIn("反爬", str(ctx.exception))
        # 内容过短也判为验证页
        with self.assertRaises(ToolError):
            tool._check_anti_bot("<html>" + "x" * 100 + "</html>", "某引擎")

    def test_render_catalog_and_descriptions(self):
        registry = self._registry()
        catalog = registry.render_catalog()
        self.assertIn("### calculator", catalog)
        descriptions = registry.descriptions()
        self.assertEqual(len(descriptions), 2)
        self.assertIn("parameters", descriptions[0])

    def test_health_check(self):
        health = self._registry().health_check()
        self.assertTrue(health["ok"])
        self.assertEqual(health["num_tools"], 2)


# ===========================================================================
# 8 个工具
# ===========================================================================
class ToolsTest(unittest.TestCase):
    """逐个验证 8 个工具的行为与容错。"""

    @classmethod
    def setUpClass(cls) -> None:
        from src.retrieval import create_retrieval_engine

        cls._ws = TempWorkspace()
        cls.cfg = cls._ws.config(
            chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50}
        )
        cls.engine = create_retrieval_engine(cls.cfg)
        cls.engine.index_documents(sample_documents())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ws.cleanup()

    def test_registry_has_eight_tools(self):
        from src.agent.tools import build_default_registry

        registry = build_default_registry(retrieval_engine=self.engine, config=self.cfg)
        self.assertEqual(len(registry), 8)
        expected = {
            "knowledge_search", "paper_metadata", "paper_compare", "keyword_extract",
            "summarize_paper", "current_time", "calculator", "web_search",
        }
        self.assertEqual(set(registry.names), expected)
        # web_search 默认关闭，不应出现在可用清单中
        enabled = {t.name for t in registry.enabled_tools()}
        self.assertNotIn("web_search", enabled)
        self.assertEqual(len(enabled), 7)

    # -- 1. knowledge_search -------------------------------------------
    def test_knowledge_search(self):
        from src.agent.tools import KnowledgeSearchTool
        from src.utils.types import ToolCall

        tool = KnowledgeSearchTool(self.engine)
        result = tool.run(ToolCall(name="knowledge_search",
                                   args={"query": "residual learning degradation", "top_k": 3}))
        self.assertTrue(result.ok, result.content)
        self.assertIn("来源：", result.content)
        self.assertIn("相关度", result.content)
        self.assertEqual(len(result.data["results"]), 3)

    def test_knowledge_search_empty_query(self):
        from src.agent.tools import KnowledgeSearchTool
        from src.utils.types import ToolCall, ToolStatus

        tool = KnowledgeSearchTool(self.engine)
        result = tool.run(ToolCall(name="knowledge_search", args={"query": "  "}))
        self.assertEqual(result.status, ToolStatus.ERROR)

    def test_knowledge_search_no_match_still_succeeds(self):
        """完全无关的问题：工具必须返回"未检索到"而不是硬塞低相关片段。

        真实踩坑：向量检索**总会**返回 top_k 条最相似片段（哪怕全部无关），
        若原样交给模型，它会误以为知识库里真有这内容从而编造答案。
        因此工具用绝对相关度（confidence）做了过滤。
        """
        from src.agent.tools import KnowledgeSearchTool
        from src.utils.types import ToolCall

        tool = KnowledgeSearchTool(self.engine)
        result = tool.run(ToolCall(name="knowledge_search", args={"query": "zzzqqq nonexistent topic"}))
        self.assertTrue(result.ok, "无命中不是错误，应返回可读提示")
        self.assertIn("没有检索到", result.content)
        self.assertEqual(result.data["num"], 0)

    def test_knowledge_search_records_retrieval_metrics(self):
        """工具直连检索引擎，必须自己把检索指标记进 MetricsCollector。

        否则纯 Agent 部署下指标面板的检索类 KPI 永远是 0（见
        ``MetricsCollector.record_retrieval`` 的说明）。
        """
        from src.agent.tools import KnowledgeSearchTool
        from src.utils.logger import MetricsCollector
        from src.utils.types import ToolCall

        collector = MetricsCollector(enabled=False)
        import src.utils.logger as logger_module

        original = logger_module._metrics
        logger_module._metrics = collector
        try:
            tool = KnowledgeSearchTool(self.engine)
            tool.run(ToolCall(name="knowledge_search",
                              args={"query": "residual learning degradation", "top_k": 3}))
            tool.run(ToolCall(name="knowledge_search", args={"query": "zzzqqq nonexistent topic"}))
        finally:
            logger_module._metrics = original

        snapshot = collector.retrieval_snapshot()
        self.assertEqual(snapshot["requests"], 2, "两次检索都该被记账")
        self.assertAlmostEqual(snapshot["hit_rate"], 0.5, msg="一次命中、一次未命中")
        self.assertGreater(snapshot["avg_top_score"], 0.0)
        # Top 分数用绝对置信度（0~1），不是恒为 1.0 的归一化分
        self.assertLessEqual(snapshot["avg_top_score"], 1.0)

    # -- 2. paper_metadata ---------------------------------------------
    def test_paper_metadata_by_query(self):
        from src.agent.tools import PaperMetadataTool
        from src.utils.types import ToolCall

        tool = PaperMetadataTool(self.engine)
        result = tool.run(ToolCall(name="paper_metadata", args={"query": "Attention Is All You Need"}))
        self.assertTrue(result.ok, result.content)
        self.assertIn("Attention Is All You Need", result.content)
        self.assertIn("Ashish Vaswani", result.content)

    def test_paper_metadata_list_all(self):
        from src.agent.tools import PaperMetadataTool
        from src.utils.types import ToolCall

        tool = PaperMetadataTool(self.engine)
        result = tool.run(ToolCall(name="paper_metadata", args={}))
        self.assertTrue(result.ok)
        self.assertEqual(result.data["num"], 3)

    def test_paper_metadata_unknown_doc_id(self):
        from src.agent.tools import PaperMetadataTool
        from src.utils.types import ToolCall, ToolStatus

        tool = PaperMetadataTool(self.engine)
        result = tool.run(ToolCall(name="paper_metadata", args={"doc_id": "nope"}))
        self.assertEqual(result.status, ToolStatus.ERROR)

    # -- 3. paper_compare ----------------------------------------------
    def test_paper_compare_generates_table(self):
        from src.agent.tools import PaperCompareTool
        from src.utils.types import ToolCall

        tool = PaperCompareTool(self.engine)
        result = tool.run(ToolCall(name="paper_compare",
                                   args={"papers": ["Transformer", "ResNet"]}))
        self.assertTrue(result.ok, result.content)
        self.assertIn("| 对比维度 |", result.content)
        self.assertGreaterEqual(result.data["num"], 2)
        self.assertIn("evidence", result.data)

    def test_paper_compare_accepts_comma_string(self):
        from src.agent.tools import PaperCompareTool
        from src.utils.types import ToolCall

        tool = PaperCompareTool(self.engine)
        result = tool.run(ToolCall(name="paper_compare", args={"papers": "Transformer, BERT"}))
        self.assertTrue(result.ok, result.content)
        self.assertGreaterEqual(result.data["num"], 2)

    # -- 4. keyword_extract --------------------------------------------
    def test_keyword_extract_from_query(self):
        from src.agent.tools import KeywordExtractTool
        from src.utils.types import ToolCall

        tool = KeywordExtractTool(self.engine)
        result = tool.run(ToolCall(name="keyword_extract",
                                   args={"query": "attention mechanism transformer", "top_k": 5}))
        self.assertTrue(result.ok, result.content)
        self.assertLessEqual(len(result.data["keywords"]), 5)
        self.assertGreater(len(result.data["keywords"]), 0)

    def test_keyword_extract_requires_input(self):
        from src.agent.tools import KeywordExtractTool
        from src.utils.types import ToolCall, ToolStatus

        tool = KeywordExtractTool(self.engine)
        result = tool.run(ToolCall(name="keyword_extract", args={}))
        self.assertEqual(result.status, ToolStatus.ERROR)

    # -- 5. summarize_paper --------------------------------------------
    def test_summarize_paper_has_four_sections(self):
        from src.agent.tools import SummarizePaperTool
        from src.utils.types import ToolCall

        tool = SummarizePaperTool(self.engine)
        result = tool.run(ToolCall(name="summarize_paper", args={"query": "ResNet"}))
        self.assertTrue(result.ok, result.content)
        for heading in ("研究背景与动机", "方法与技术路线", "实验结果", "结论与局限"):
            self.assertIn(heading, result.content, f"缺少小节: {heading}")

    # -- 6. current_time -----------------------------------------------
    def test_current_time_formats(self):
        from datetime import datetime

        from src.agent.tools import CurrentTimeTool
        from src.utils.types import ToolCall

        tool = CurrentTimeTool()
        full = tool.run(ToolCall(name="current_time", args={}))
        self.assertTrue(full.ok)
        self.assertIn(str(datetime.now().year), full.content)

        date_only = tool.run(ToolCall(name="current_time", args={"format": "date"}))
        self.assertRegex(date_only.content, r"\d{4}-\d{2}-\d{2}")

        stamp = tool.run(ToolCall(name="current_time", args={"format": "timestamp"}))
        self.assertGreater(int(stamp.data["timestamp"]), 1_600_000_000)

    # -- 7. calculator --------------------------------------------------
    def test_calculator_arithmetic(self):
        from src.agent.tools import CalculatorTool
        from src.utils.types import ToolCall

        tool = CalculatorTool()
        cases = {
            "1+1": "2",
            "3.14*2.56": "8.0384",
            "(28.4-26.4)/26.4*100": "7.5758",
            "2**10": "1024",
            "sqrt(16)": "4",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = tool.run(ToolCall(name="calculator", args={"expression": expression}))
                self.assertTrue(result.ok, result.content)
                self.assertIn(expected, result.content)

    def test_calculator_chinese_operators(self):
        from src.agent.tools import CalculatorTool
        from src.utils.types import ToolCall

        tool = CalculatorTool()
        result = tool.run(ToolCall(name="calculator", args={"expression": "100 除以 4"}))
        # 工具本身只接受表达式；中文需由 LLM 翻译。此处验证不会崩溃
        self.assertIn(result.status.value, ("success", "error"))

    def test_calculator_blocks_code_injection(self):
        """计算器绝不允许执行任意代码 —— 这是 Agent 安全的关键一环。"""
        from src.agent.tools import CalculatorTool
        from src.utils.types import ToolCall, ToolStatus

        tool = CalculatorTool()
        attacks = [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "[].__class__.__bases__[0].__subclasses__()",
            "lambda: 1",
            "print('x')",
            "globals()",
            "eval('1+1')",
            "exec('x=1')",
        ]
        for attack in attacks:
            with self.subTest(attack=attack):
                result = tool.run(ToolCall(name="calculator", args={"expression": attack}))
                self.assertEqual(result.status, ToolStatus.ERROR, f"未拦截: {attack}")

    def test_calculator_blocks_resource_exhaustion(self):
        from src.agent.tools import CalculatorTool
        from src.utils.types import ToolCall, ToolStatus

        result = CalculatorTool().run(
            ToolCall(name="calculator", args={"expression": "9**9**9"})
        )
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("指数过大", result.content)

    def test_calculator_division_by_zero(self):
        from src.agent.tools import CalculatorTool
        from src.utils.types import ToolCall, ToolStatus

        result = CalculatorTool().run(ToolCall(name="calculator", args={"expression": "1/0"}))
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("零", result.content)

    # -- 8. web_search --------------------------------------------------
    def test_web_search_disabled_by_default(self):
        """显式关闭联网搜索时，工具应返回**带操作指引的错误**而非静默 SKIPPED。

        返回 ERROR 而非 SKIPPED 是刻意的：模型能从错误信息里读到
        "web_search.enabled=false（默认关闭）"，从而改用知识库检索；
        而一个空洞的 SKIPPED 对模型没有任何指导价值。
        """
        from src.agent.tools import WebSearchTool
        from src.utils.types import ToolCall, ToolStatus

        tool = WebSearchTool(self.cfg)
        self.assertFalse(tool.enabled)
        result = tool.run(ToolCall(name="web_search", args={"query": "test"}))
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("未启用", result.content)
        self.assertIn("web_search.enabled", result.content)


# ===========================================================================
# 意图路由
# ===========================================================================
class RouterTest(unittest.TestCase):
    """规则路由的意图判定与兜底。"""

    def _router(self):
        from src.agent import RuleRouter

        return RuleRouter()

    def test_calculator_routing(self):
        router = self._router()
        for question in ("3.14 乘以 2.56 等于多少", "计算 100/4", "1+1 等于几"):
            with self.subTest(q=question):
                decision = router.decide(question)
                self.assertIn("calculator", decision.routes)
                self.assertIn("calculator", decision.tools)

    def test_current_time_routing(self):
        decision = self._router().decide("现在几点了？")
        self.assertIn("current_time", decision.routes)

    def test_paper_metadata_routing(self):
        decision = self._router().decide("Attention Is All You Need 的作者是谁？")
        self.assertIn("paper_metadata", decision.routes)
        self.assertIn("paper_metadata", decision.tools)

    def test_compare_routing(self):
        decision = self._router().decide("对比 Transformer 和 ResNet 的方法差异")
        self.assertIn("paper_compare", decision.routes)
        self.assertEqual(decision.tools[0], "paper_compare")

    def test_summarize_routing(self):
        decision = self._router().decide("帮我总结一下 BERT 这篇论文")
        self.assertIn("summarize_paper", decision.routes)

    def test_keyword_routing(self):
        decision = self._router().decide("提取这段文本的关键词")
        self.assertIn("keyword_extract", decision.routes)

    def test_direct_routing_for_greetings(self):
        for question in ("你好", "谢谢", "你是谁", "hi", "再见"):
            with self.subTest(q=question):
                decision = self._router().decide(question)
                self.assertTrue(decision.direct, f"未判定为直接回答: {question}")
                self.assertEqual(decision.routes, ["direct"])

    def test_fallback_to_knowledge_search(self):
        """未匹配到明确意图时应默认走知识库检索，而不是什么都不做。"""
        decision = self._router().decide("嗯这个那个随便说点什么吧")
        self.assertTrue(decision.routes)
        self.assertIn("knowledge_search", decision.tools)

    def test_empty_question(self):
        decision = self._router().decide("")
        self.assertEqual(decision.routes, [])
        self.assertFalse(decision.direct)

    def test_confidence_in_range(self):
        for question in ("你好", "3+5", "对比 A 和 B", "随便一个问题"):
            decision = self._router().decide(question)
            self.assertGreaterEqual(decision.confidence, 0.0)
            self.assertLessEqual(decision.confidence, 1.0)

    def test_stats(self):
        router = self._router()
        router.decide("你好")
        router.decide("3+5")
        stats = router.stats()
        self.assertEqual(stats["routed"], 2)
        self.assertEqual(stats["direct"], 1)

    def test_factory_none_disables_routing(self):
        from src.agent import create_router

        with TempWorkspace() as ws:
            router = create_router(ws.config(agent={"router_provider": "none"}))
            decision = router.decide("你好")
            # 关闭路由后不应返回 direct（一切都交给完整循环处理）
            self.assertFalse(decision.direct)

    def test_factory_none_returns_empty_candidate_tools(self):
        """关闭路由时必须返回**空**候选工具集，而不是兜底的 knowledge_search。

        回归：早期实现靠把 ``min_score`` 调到 1e9 来"关闭"路由，结果每次都走
        兜底分支、返回 ``tools=["knowledge_search", ...]``。Agent 把它当作工具
        白名单渲染进提示词，等于把模型可用的工具砍到只剩检索一类 —— 想关路由，
        实际是悄悄禁用了计算器/时间/论文对比等工具，且没有任何报错。
        """
        from src.agent import NullRouter, create_router

        with TempWorkspace() as ws:
            router = create_router(ws.config(agent={"router_provider": "none"}))
            self.assertIsInstance(router, NullRouter)
            for question in ("你好", "帮我算一下 3.14 乘 2.56", "RAG 是什么", ""):
                self.assertEqual(router.decide(question).tools, [])

    def test_no_router_still_renders_full_tool_catalog(self):
        """空候选集 → 完整工具目录（两处约定必须一致）。"""
        from src.agent import AgentConfig, ReActAgent, create_router
        from src.agent.base_tool import ToolRegistry
        from src.llm import MockLLM

        with TempWorkspace() as ws:
            cfg = ws.config(agent={"router_provider": "none"})
            from src.agent.tools import build_default_registry

            registry = build_default_registry(config=cfg)
            agent = ReActAgent(
                registry=registry, llm=MockLLM(cfg), memory=None,
                router=create_router(cfg), config=cfg, agent_config=AgentConfig(),
            )
            prompt = agent._build_system_prompt([])
            for tool in registry.enabled_tools():
                self.assertIn(tool.name, prompt, f"工具 {tool.name} 未出现在完整目录中")


# ===========================================================================
# ReAct 循环
# ===========================================================================
class ReActAgentTest(unittest.TestCase):
    """Agent 主循环：工具调用、终止条件、错误恢复、死循环、并行。"""

    @classmethod
    def setUpClass(cls) -> None:
        from src.retrieval import create_retrieval_engine

        cls._ws = TempWorkspace()
        cls.cfg = cls._ws.config(
            chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50},
            agent={"max_iterations": 5, "tool_timeout": 8, "max_retries_per_tool": 1},
        )
        cls.engine = create_retrieval_engine(cls.cfg)
        cls.engine.index_documents(sample_documents())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ws.cleanup()

    def _agent(self, **overrides):
        from src.agent import create_agent

        cfg = self.engine.config if not overrides else self._ws.config(
            chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50},
            agent={"max_iterations": 5, "tool_timeout": 8, "max_retries_per_tool": 1, **overrides},
        )
        return create_agent(config=cfg, retrieval_engine=self.engine)

    # -- 基本循环 -------------------------------------------------------
    def test_react_loop_produces_thought_action_observation(self):
        agent = self._agent()
        result = agent.run("What datasets were used to train the Transformer?", session_id="t1")
        types = [step.type.value for step in result.steps]
        self.assertIn("thought", types)
        self.assertIn("action", types)
        self.assertIn("observation", types)
        self.assertIn("final", types)
        self.assertEqual(result.terminated_reason, "final_answer")
        self.assertTrue(result.answer.strip())

    def test_tool_selection_calculator(self):
        agent = self._agent()
        result = agent.run("3.14 乘以 2.56 等于多少", session_id="t2")
        tools = [s.tool_name for s in result.steps if s.type.value == "action"]
        self.assertIn("calculator", tools)
        self.assertEqual(result.tool_failures, 0)

    def test_tool_selection_metadata(self):
        agent = self._agent()
        result = agent.run("Attention Is All You Need 的作者是谁？", session_id="t3")
        tools = [s.tool_name for s in result.steps if s.type.value == "action"]
        self.assertIn("paper_metadata", tools)

    def test_direct_answer_skips_loop(self):
        agent = self._agent()
        result = agent.run("你好", session_id="t4")
        self.assertEqual(result.terminated_reason, "direct_answer")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.tool_calls, 0)

    def test_empty_question(self):
        result = self._agent().run("   ")
        self.assertEqual(result.terminated_reason, "error")
        self.assertTrue(result.answer)

    def test_routed_tools_recorded(self):
        result = self._agent().run("现在几点了", session_id="t5")
        self.assertTrue(result.routed_tools)
        self.assertIn("current_time", result.routed_tools)

    def test_on_step_callback_streams_trajectory(self):
        """on_step 回调必须实时产出每一步（前端 SSE 依赖它）。"""
        seen = []
        agent = self._agent()
        agent.run("What is attention?", session_id="t6", on_step=seen.append)
        self.assertGreater(len(seen), 2)
        kinds = {step.type.value for step in seen}
        self.assertIn("thought", kinds)

    def test_trajectory_serialisable(self):
        import json

        result = self._agent().run("What is self-attention?", session_id="t7")
        payload = json.dumps(result.trajectory(), ensure_ascii=False, default=str)
        self.assertIn("thought", payload)

    def test_citations_collected(self):
        result = self._agent().run(
            "How does residual learning work in ResNet?", session_id="t8"
        )
        self.assertGreater(len(result.citations), 0)
        for citation in result.citations:
            self.assertTrue(citation.source)

    def test_usage_and_latency_recorded(self):
        result = self._agent().run("What is BERT?", session_id="t9")
        self.assertGreater(result.usage.total_tokens, 0)
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.iterations, 0)

    # -- 错误恢复 -------------------------------------------------------
    def test_tool_timeout_retry_and_error_observation(self):
        from src.agent.base_tool import BaseTool, ToolParameter, ToolRegistry
        from src.agent import AgentConfig, ReActAgent
        from src.llm import MockLLM
        from src.utils.types import ToolStatus

        class SlowTool(BaseTool):
            name = "knowledge_search"  # 冒用检索工具名，让 mock 决策选中它
            description = "慢工具"
            parameters = [ToolParameter(name="query", type="string", required=False, default="")]
            parameters = [
                ToolParameter(name="query", type="string", required=False, default=""),
                ToolParameter(name="top_k", type="integer", required=False, default=3),
                ToolParameter(name="mode", type="string", required=False, default="full"),
            ]

            def execute(self, **kwargs):
                time.sleep(3)
                return "never reached"

        agent = ReActAgent(
            registry=ToolRegistry([SlowTool()]),
            llm=MockLLM(self.cfg),
            memory=None,
            router=None,
            config=self.cfg,
            agent_config=AgentConfig(max_iterations=2, tool_timeout=0.4, max_retries_per_tool=2),
        )
        result = agent.run("anything at all")
        timeouts = [
            s for s in result.steps
            if s.tool_result is not None and s.tool_result.status == ToolStatus.TIMEOUT
        ]
        self.assertTrue(timeouts, "未产生超时结果")
        self.assertGreaterEqual(timeouts[0].tool_result.retries, 1, "应至少重试一次")
        self.assertIn("错误恢复提示", timeouts[0].tool_result.content)

    def test_dead_loop_detection(self):
        """连续用相同参数调用同一工具时必须强制终止（回归测试）。

        真实踩坑：早期实现的死循环检测把"新签名与最近调用比较"写错了方向，
        导致 Agent 反复调用同一工具直到耗尽最大轮次，白白浪费 Token 与时间。
        """
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import BaseTool, ToolParameter, ToolRegistry
        from src.llm import MockLLM

        class LoopTool(BaseTool):
            name = "loop_tool"
            description = "总是返回固定结果"
            parameters = [ToolParameter(name="q", type="string", required=False, default="")]
            def execute(self, q="", **kwargs):
                return "固定观察结果"

        class LoopLLM(MockLLM):
            """始终选择同一个工具与参数，模拟模型陷入循环。"""

            def _simulate_decision(self, prompt, system):
                return {
                    "thought": "还需要更多信息。",
                    "action": "loop_tool",
                    "action_input": {"q": "same"},
                }

        agent = ReActAgent(
            registry=ToolRegistry([LoopTool()]),
            llm=LoopLLM(self.cfg),
            memory=None,
            router=None,
            config=self.cfg,
            agent_config=AgentConfig(max_iterations=10, loop_detect_window=2,
                                     force_final_on_max_iterations=False),
        )
        result = agent.run("循环测试")
        self.assertEqual(result.terminated_reason, "loop_detected")
        self.assertLess(result.iterations, 10, "应在达到最大轮次前就终止")
        self.assertTrue(result.degraded)
        self.assertTrue(result.answer.strip(), "强制终止后仍必须给出可读回复")

    def test_max_iterations_forces_final_answer(self):
        """达到最大轮次时必须基于已有信息收尾，而不是抛'推理超限'。"""
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import BaseTool, ToolParameter, ToolRegistry
        from src.llm import MockLLM

        class VariedTool(BaseTool):
            name = "varied_tool"
            description = "每次返回不同结果，避免触发死循环检测"
            parameters = [
                ToolParameter(name="q", type="string", required=False, default=""),
                ToolParameter(name="n", type="integer", required=False, default=0),
            ]
            counter = 0
            def execute(self, q="", n=0, **kwargs):
                VariedTool.counter += 1
                return f"这是第 {VariedTool.counter} 次工具返回的观察结果，包含可供作答的信息。"

        class NeverFinishLLM(MockLLM):
            def _simulate_decision(self, prompt, system):
                return {
                    "thought": "继续收集信息。",
                    "action": "varied_tool",
                    "action_input": {"q": "x", "n": VariedTool.counter},
                }

        agent = ReActAgent(
            registry=ToolRegistry([VariedTool()]),
            llm=NeverFinishLLM(self.cfg),
            memory=None,
            router=None,
            config=self.cfg,
            agent_config=AgentConfig(max_iterations=3, loop_detect_window=1,
                                     force_final_on_max_iterations=True),
        )
        result = agent.run("测试最大轮次收尾")
        self.assertIn(result.terminated_reason, ("max_iterations", "loop_detected"))
        self.assertTrue(result.answer.strip(), "必须有最终答案")
        self.assertNotIn("推理轮次超限", result.answer)
        self.assertGreater(result.iterations, 0)

    def test_llm_failure_is_handled(self):
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import ToolRegistry
        from src.agent.tools import CalculatorTool

        class BrokenLLM:
            model = "broken"

            def chat_with_usage(self, *args, **kwargs):
                raise RuntimeError("LLM 挂了")

            def chat(self, *args, **kwargs):
                raise RuntimeError("LLM 挂了")

            def health_check(self):
                return {"ok": False}

        agent = ReActAgent(
            registry=ToolRegistry([CalculatorTool()]),
            llm=BrokenLLM(),
            memory=None,
            router=None,
            config=self.cfg,
            agent_config=AgentConfig(max_iterations=2),
        )
        result = agent.run("1+1")
        self.assertTrue(result.answer.strip(), "LLM 失效也必须给出回复")
        self.assertIn(result.terminated_reason, ("parse_error", "max_iterations"))

    # -- 并行调用 -------------------------------------------------------
    def test_parallel_tool_execution(self):
        """一轮内多个独立工具应并发执行（总耗时显著小于串行之和）。"""
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import BaseTool, ToolParameter, ToolRegistry
        from src.llm import MockLLM

        class SleepTool(BaseTool):
            name = "sleep_tool"
            description = "模拟耗时工具"
            parameters = [ToolParameter(name="idx", type="integer", required=False, default=0)]
            def execute(self, idx=0, **kwargs):
                time.sleep(0.5)
                return f"工具 {idx} 完成"

        class ParallelLLM(MockLLM):
            def _simulate_decision(self, prompt, system):
                if "Observation" in prompt:
                    return {"thought": "信息足够。", "action": "final_answer",
                            "action_input": {"answer": "并行结果已收到。"}}
                return {
                    "thought": "需要并行调用三个独立工具。",
                    "actions": [
                        {"action": "sleep_tool", "action_input": {"idx": 0}},
                        {"action": "sleep_tool", "action_input": {"idx": 1}},
                        {"action": "sleep_tool", "action_input": {"idx": 2}},
                    ],
                }

        agent = ReActAgent(
            registry=ToolRegistry([SleepTool()]),
            llm=ParallelLLM(self.cfg),
            memory=None,
            router=None,
            config=self.cfg,
            agent_config=AgentConfig(max_iterations=3, enable_parallel=True,
                                     max_parallel_tools=3, tool_timeout=10),
        )
        started = time.perf_counter()
        result = agent.run("并行测试")
        elapsed = time.perf_counter() - started
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual(result.tool_failures, 0)
        self.assertLess(elapsed, 1.4, f"并行执行耗时 {elapsed:.2f}s，接近串行（1.5s）说明未真正并发")

    def test_parallel_disabled_runs_sequentially(self):
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import BaseTool, ToolParameter, ToolRegistry
        from src.llm import MockLLM

        class SleepTool(BaseTool):
            name = "sleep_tool"
            description = "模拟耗时工具"
            parameters = [ToolParameter(name="idx", type="integer", required=False, default=0)]
            def execute(self, idx=0, **kwargs):
                time.sleep(0.3)
                return f"工具 {idx} 完成"

        class ParallelLLM(MockLLM):
            def _simulate_decision(self, prompt, system):
                if "Observation" in prompt:
                    return {"thought": "足够。", "action": "final_answer",
                            "action_input": {"answer": "完成。"}}
                return {"thought": "并行。", "actions": [
                    {"action": "sleep_tool", "action_input": {"idx": 0}},
                    {"action": "sleep_tool", "action_input": {"idx": 1}},
                ]}

        agent = ReActAgent(
            registry=ToolRegistry([SleepTool()]),
            llm=ParallelLLM(self.cfg),
            memory=None, router=None, config=self.cfg,
            agent_config=AgentConfig(max_iterations=3, enable_parallel=False, tool_timeout=10),
        )
        started = time.perf_counter()
        result = agent.run("串行测试")
        elapsed = time.perf_counter() - started
        self.assertEqual(result.tool_calls, 2)
        self.assertGreater(elapsed, 0.55, "关闭并行后应为串行执行")

    # -- 状态 -----------------------------------------------------------
    def test_agent_stats_and_health(self):
        agent = self._agent()
        agent.run("What is attention?", session_id="stat")
        stats = agent.stats()
        self.assertGreaterEqual(stats["runs"], 1)
        self.assertIn("config", stats)
        health = agent.health_check()
        self.assertIn("tools", health)
        self.assertIn("memory", health)


# ===========================================================================
# 对话记忆
# ===========================================================================
class MemoryTest(unittest.TestCase):
    """会话隔离、滑动窗口、摘要压缩、反馈记录。"""

    def _memory(self, ws, **overrides):
        from src.agent import create_memory

        cfg = ws.config(memory=overrides)
        from src.llm import MockLLM

        return create_memory(cfg, llm=MockLLM(cfg))

    def test_session_isolation(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws)
            s1 = memory.create_session("会话一")
            s2 = memory.create_session("会话二")
            memory.add_turn(s1.session_id, "问题一", "回答一")
            memory.add_turn(s1.session_id, "问题二", "回答二")
            memory.add_turn(s2.session_id, "另一个问题", "另一个回答")

            self.assertEqual(memory.get_session(s1.session_id).num_turns, 2)
            self.assertEqual(memory.get_session(s2.session_id).num_turns, 1)
            ctx1 = memory.get_context(s1.session_id)
            ctx2 = memory.get_context(s2.session_id)
            self.assertEqual(len(ctx1["messages"]), 4)
            self.assertEqual(len(ctx2["messages"]), 2)

    def test_sliding_window_truncates(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws, max_tokens=120, max_turns=50, summary_threshold=1000)
            session = memory.create_session("长会话")
            for i in range(20):
                memory.add_turn(
                    session.session_id,
                    f"这是第 {i} 个询问内容，包含足够长度以消耗 token 预算",
                    f"这是第 {i} 个回答内容，同样包含足够长度以便触发窗口截断。" * 5,
                )
            ctx = memory.get_context(session.session_id)
            self.assertTrue(ctx["truncated"], "超预算历史应被截断")
            self.assertLessEqual(ctx["tokens"], 200)
            self.assertGreater(len(ctx["messages"]), 0)
            self.assertIn("省略", ctx["hint"])

    def test_summary_compression_triggered(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws, summary_threshold=4, keep_recent_after_summary=2)
            session = memory.create_session("摘要会话")
            for i in range(6):
                memory.add_turn(session.session_id, f"问题 {i}", f"回答 {i}")
            session = memory.get_session(session.session_id)
            self.assertTrue(session.summary, "超过阈值后应生成摘要")
            self.assertIn("用户曾询问", session.summary)

    def test_summary_persisted(self):
        with TempWorkspace() as ws:
            cfg = ws.config(memory={"summary_threshold": 3, "keep_recent_after_summary": 1})
            from src.agent import create_memory
            from src.llm import MockLLM

            memory = create_memory(cfg, llm=MockLLM(cfg))
            session = memory.create_session("持久化摘要")
            for i in range(5):
                memory.add_turn(session.session_id, f"问题 {i}", f"回答 {i}")

            reopened = create_memory(cfg, llm=MockLLM(cfg))
            restored = reopened.get_session(session.session_id)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.num_turns, 5)
            self.assertTrue(restored.summary)

    def test_compressed_turns_are_not_replayed_in_window(self):
        """已压缩进摘要的旧轮次不应再原样出现在滑动窗口里。

        回归：窗口曾直接取 ``turns[-max_turns:]``，把摘要和"摘要的来源"一起
        送进提示词：同一个话题被模型看到两次，Token 预算被白白吃掉。
        """
        with TempWorkspace() as ws:
            memory = self._memory(ws, summary_threshold=3, keep_recent_after_summary=2,
                                  max_turns=50, max_tokens=100000)
            session = memory.create_session("摘要窗口")
            for i in range(8):
                memory.add_turn(session.session_id, f"问题 {i}", f"回答 {i}")

            reloaded = memory.get_session(session.session_id)
            self.assertTrue(reloaded.summary)
            ctx = memory.get_context(session.session_id)
            questions = [m.content for m in ctx["messages"] if m.role.value == "user"]
            # 摘要覆盖了前 6 轮（8 - keep 2），窗口里只应剩最近 2 轮原文
            self.assertEqual(questions, ["问题 6", "问题 7"], questions)
            self.assertIn("summary_upto", reloaded.metadata)

    def test_compression_is_incremental(self):
        """二次压缩只处理新跨越边界的轮次，不把全部历史再喂一遍。

        回归：早期实现每次都把 ``turns[:-keep]`` 全量重摘，第 N 次压缩要读
        N 倍历史，长会话的摘要成本是 O(n²)，还让模型反复"总结摘要"。
        """
        with TempWorkspace() as ws:
            memory = self._memory(ws, summary_threshold=3, keep_recent_after_summary=2)
            session = memory.create_session("增量摘要")
            for i in range(5):
                memory.add_turn(session.session_id, f"问题 {i}", f"回答 {i}")
            first_upto = memory.get_session(session.session_id).metadata["summary_upto"]
            self.assertGreaterEqual(first_upto, 1, "首轮压缩未推进摘要边界")

            seen_texts = []
            original = memory._summarize

            def spy(history_text, previous=""):
                seen_texts.append(history_text)
                return original(history_text, previous=previous)

            memory._summarize = spy  # type: ignore[assignment]
            for i in range(5, 7):
                memory.add_turn(session.session_id, f"问题 {i}", f"回答 {i}")

            self.assertTrue(seen_texts, "未触发二次压缩")
            latest = seen_texts[-1]
            self.assertIn("本次新增对话", latest)
            new_section = latest.split("本次新增对话")[-1]
            # 新喂进摘要模型的只应是边界之后的那几轮，不含早已压缩的旧轮次
            self.assertIn("问题", new_section)
            self.assertNotIn("问题 0", new_section, "已压缩的旧轮次被重复喂给摘要模型")
            self.assertNotIn("问题 1", new_section, "已压缩的旧轮次被重复喂给摘要模型")
            self.assertGreater(
                memory.get_session(session.session_id).metadata["summary_upto"], first_upto
            )

    def test_rule_fallback_summary_keeps_previous_topics(self):
        """没有 LLM 时的规则兜底摘要必须与旧摘要合并，不能把旧要点删掉。

        回归：兜底摘要只抽取"本次新增对话"里的问题，不与 ``previous`` 合并，
        于是每压缩一次就把更早的要点覆盖掉，最终摘要里只剩最后一两轮。
        """
        from src.agent import create_memory

        with TempWorkspace() as ws:
            cfg = ws.config(memory={"summary_threshold": 2, "keep_recent_after_summary": 2})
            memory = create_memory(cfg, llm=None)  # 强制走规则兜底
            session = memory.create_session("兜底摘要")
            for i in range(5):
                memory.add_turn(session.session_id, f"问题 {i}", f"回答 {i}")

            summary = memory.get_session(session.session_id).summary
            self.assertIn("用户曾询问", summary)
            self.assertEqual(summary.count("历史对话要点"), 1, f"摘要标签重复：{summary}")
            # 最近 keep=2 轮按设计保留原文，不进摘要；其余 3 轮都必须在摘要里
            for i in range(3):
                self.assertIn(f"问题 {i}", summary, f"第 {i} 轮的要点在合并中丢失")

    def test_summary_tokens_counted_in_budget(self):
        """摘要本身要计入 Token 预算，否则 ``tokens`` 会系统性低报。"""
        with TempWorkspace() as ws:
            memory = self._memory(ws, summary_threshold=2, keep_recent_after_summary=1,
                                  max_turns=1, max_tokens=100000)
            session = memory.create_session("预算")
            for i in range(4):
                memory.add_turn(session.session_id, f"问题 {i}", f"回答 {i}")
            ctx = memory.get_context(session.session_id)
            body = sum(
                __import__("src.utils.text", fromlist=["estimate_tokens"]).estimate_tokens(m.content)
                for m in ctx["messages"]
            )
            expected = body + __import__(
                "src.utils.text", fromlist=["estimate_tokens"]
            ).estimate_tokens(ctx["summary"])
            self.assertEqual(ctx["tokens"], expected)

    def test_turn_history_roundtrip(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws)
            session = memory.create_session("往返")
            memory.add_turn(
                session.session_id,
                "原始问题",
                "原始回答",
                citations=[{"index": 1, "source": "论文X", "page": 2, "score": 0.9}],
                trajectory=[{"type": "thought", "content": "思考"}],
                latency_ms=123.0,
            )
            from src.agent import create_memory
            from src.llm import MockLLM

            reopened = create_memory(ws.config(), llm=MockLLM(ws.config()))
            turn = reopened.get_session(session.session_id).turns[0]
            self.assertEqual(turn.question, "原始问题")
            self.assertEqual(len(turn.citations), 1)
            self.assertEqual(turn.citations[0].source, "论文X")
            self.assertEqual(turn.trajectory[0]["type"], "thought")
            self.assertAlmostEqual(turn.latency_ms, 123.0)

    def test_rename_and_delete(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws)
            session = memory.create_session("原标题")
            self.assertTrue(memory.rename_session(session.session_id, "新标题"))
            self.assertEqual(memory.get_session(session.session_id).title, "新标题")
            self.assertTrue(memory.delete_session(session.session_id))
            self.assertIsNone(memory.get_session(session.session_id))

    def test_title_auto_from_first_question(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws)
            session = memory.create_session()
            memory.add_turn(session.session_id, "Transformer 的核心创新是什么", "回答")
            self.assertIn("Transformer", memory.get_session(session.session_id).title)

    def test_feedback_recorded(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws)
            session = memory.create_session("反馈")
            turn = memory.add_turn(session.session_id, "问题", "回答")
            self.assertTrue(memory.set_feedback(turn.turn_id, "up"))
            restored = memory.get_session(session.session_id).turns[0]
            self.assertEqual(restored.feedback, "up")

    def test_list_sessions_sorted_by_recency(self):
        with TempWorkspace() as ws:
            memory = self._memory(ws)
            first = memory.create_session("较早")
            memory.add_turn(first.session_id, "q1", "a1")
            time.sleep(0.01)
            second = memory.create_session("较晚")
            memory.add_turn(second.session_id, "q2", "a2")
            listed = memory.list_sessions()
            self.assertGreaterEqual(len(listed), 2)
            self.assertEqual(listed[0]["session_id"], second.session_id)

    def test_memory_disabled(self):
        from src.agent import NullMemory, create_memory

        with TempWorkspace() as ws:
            memory = create_memory(ws.config(memory={"backend": "none"}))
            self.assertIsInstance(memory, NullMemory)
            session = memory.create_session("x")
            memory.add_turn(session.session_id, "q", "a")
            self.assertEqual(memory.get_context(session.session_id)["messages"], [])

    def test_agent_uses_memory_across_turns(self):
        """同一会话内第二轮应能看到第一轮的上下文提示。"""
        from src.agent import create_agent
        from src.retrieval import create_retrieval_engine

        with TempWorkspace() as ws:
            cfg = ws.config(
                chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50},
                memory={"summary_threshold": 3, "keep_recent_after_summary": 1},
            )
            engine = create_retrieval_engine(cfg)
            engine.index_documents(sample_documents())
            agent = create_agent(config=cfg, retrieval_engine=engine)

            for question in ("What is attention?", "What is BERT?", "What is ResNet?",
                             "What else?", "And more?"):
                agent.run(question, session_id="multi")
            session = agent.memory.get_session("multi")
            self.assertEqual(session.num_turns, 5)
            self.assertTrue(session.summary)


class AgentRobustnessTest(unittest.TestCase):
    """ReAct 循环的健壮性回归测试（不依赖检索引擎，纯内存替身工具）。

    覆盖的都是"单线程跑一次看不出来、真实并发/超时场景下才暴露"的缺陷：

    1. ``tool_timeout`` 必须真正限住**墙钟时间**；
    2. 非临时性失败（参数错误）不做无意义重试；
    3. 循环状态必须按 ``run()`` 隔离，否则多会话并发时互相污染；
    4. ``max_parallel_tools`` 是并发度而非"接单上限"，超出的调用不能静默丢弃；
    5. 模型把 ``actions`` 写坏时，不能把 thought 当答案返回；
    6. 工具结果里的脏字段（``page="p.12"``）不能让汇总阶段崩掉。
    """

    def _config(self):
        from src.utils.config import get_config

        return get_config()

    def _tool(self, name: str, delay: float = 0.0, result: str = "ok"):
        from src.agent.base_tool import BaseTool, ToolParameter

        class _Stub(BaseTool):
            description = "测试替身"

            parameters = [
                ToolParameter(name="q", type="string", required=False, default=""),
                ToolParameter(name="idx", type="integer", required=False, default=0),
            ]

            def execute(self, q="", idx=0, **kwargs):
                if delay:
                    time.sleep(delay)
                return result

        _Stub.name = name
        return _Stub()

    def _llm(self, decision_factory):
        from src.llm import MockLLM

        class _ScriptedLLM(MockLLM):
            def _simulate_decision(self, prompt, system):
                return decision_factory(prompt)

        return _ScriptedLLM(None)

    def _agent(self, tools, decision_factory, **agent_kwargs):
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import ToolRegistry

        defaults = {"max_iterations": 3, "tool_timeout": 10, "max_retries_per_tool": 3}
        defaults.update(agent_kwargs)
        return ReActAgent(
            registry=ToolRegistry(list(tools)),
            llm=self._llm(decision_factory),
            memory=None,
            router=None,
            config=self._config(),
            agent_config=AgentConfig(**defaults),
        )

    # -- 1. 超时必须限住墙钟时间 ----------------------------------------
    def test_tool_timeout_bounds_wall_clock(self):
        """工具卡住时，Agent 必须在 tool_timeout 内返回，而不是等它跑完。

        回归：早期实现用 ``with ThreadPoolExecutor(...) as pool``，退出时的
        ``shutdown(wait=True)`` 会无视 timeout 一直等，实测 0.4s 的超时设置
        照样被 sleep(3s) 的工具拖满 3 秒。
        """
        from src.utils.types import ToolStatus

        tool = self._tool("slow_tool", delay=3.0)
        agent = self._agent(
            [tool],
            lambda _prompt: {"thought": "调用慢工具", "action": "slow_tool", "action_input": {"q": "x"}},
            max_iterations=1, tool_timeout=0.4, max_retries_per_tool=2,
        )
        started = time.perf_counter()
        result = agent.run("超时测试")
        elapsed = time.perf_counter() - started

        timeouts = [
            s for s in result.steps
            if s.tool_result is not None and s.tool_result.status == ToolStatus.TIMEOUT
        ]
        self.assertTrue(timeouts, "未产生超时结果")
        self.assertLess(elapsed, 2.0, f"墙钟耗时 {elapsed:.2f}s，超时并未真正生效")
        self.assertIn("错误恢复提示", timeouts[0].tool_result.content)

    # -- 2. 非临时性失败不重试 ------------------------------------------
    def test_non_retryable_failure_is_not_retried(self):
        """参数错误这类失败重试不可能成功，应当立即返回可操作的提示。"""
        tool = self._tool("bad_tool", result="never")
        agent = self._agent(
            [tool],
            lambda _prompt: {"thought": "调用", "action": "nonexistent_tool", "action_input": {}},
            max_iterations=1, tool_timeout=5, max_retries_per_tool=3,
        )
        started = time.perf_counter()
        result = agent.run("工具不存在的测试")
        elapsed = time.perf_counter() - started
        failures = [s for s in result.steps if s.tool_result is not None and not s.tool_result.ok]
        self.assertTrue(failures, "未知工具应当失败")
        self.assertEqual(failures[0].tool_result.retries, 0, "非临时性失败不应重试")
        self.assertLess(elapsed, 0.6, f"重试等待耗时 {elapsed:.2f}s，说明仍做了无意义重试")

    # -- 3. 循环状态按 run 隔离 -----------------------------------------
    def test_loop_state_is_isolated_between_concurrent_runs(self):
        """并发运行同一 Agent 实例时，重复检测状态不得互相污染。

        回归：状态曾挂在 ``self._tool_query_history`` 上并在 run() 开头重置，
        两个会话并发提问时，A 的重置会清空 B 的历史，B 的新查询又写进 A 的列表，
        导致正常提问被误判为"语义重复"而提前以 loop_detected 收尾。
        """
        import threading

        tool = self._tool("shared_tool", delay=0.05, result="固定结果")

        def decide(prompt):
            if "Observation" in prompt:
                return {"thought": "信息足够", "action": "final_answer",
                        "action_input": {"answer": "完成"}}
            return {"thought": "检索", "action": "shared_tool",
                    "action_input": {"q": "完全相同的查询词"}}

        agent = self._agent([tool], decide, max_iterations=3, loop_max_warnings=0)
        outcomes = {}

        def worker(tag):
            outcomes[tag] = agent.run(f"并发问题 {tag}", session_id=f"sess_{tag}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(outcomes), 4)
        for tag, result in outcomes.items():
            self.assertEqual(
                result.terminated_reason, "final_answer",
                f"会话 {tag} 被误判为死循环：{result.terminated_reason}",
            )

    # -- 4. 并行批次不静默丢弃 ------------------------------------------
    def test_parallel_batch_is_not_silently_truncated(self):
        """模型一轮给出多于 max_parallel_tools 的调用时，全部都要被执行。

        回归：``calls[:max_parallel_tools]`` 会静默丢掉多余的调用，
        模型等不到它们的 Observation，只能再问一遍，白白多烧一轮。
        """
        from src.utils.types import ToolStatus

        tool = self._tool("sleep_tool", delay=0.3)

        def decide(prompt):
            if "Observation" in prompt:
                return {"thought": "足矣", "action": "final_answer",
                        "action_input": {"answer": "并行结果"}}
            return {"thought": "并行 4 个", "actions": [
                {"action": "sleep_tool", "action_input": {"idx": i}} for i in range(4)
            ]}

        agent = self._agent([tool], decide, max_iterations=3, enable_parallel=True,
                            max_parallel_tools=2, tool_timeout=10)
        started = time.perf_counter()
        result = agent.run("并行批次测试")
        elapsed = time.perf_counter() - started

        self.assertEqual(result.tool_calls, 4, "部分并行调用被丢弃")
        self.assertEqual(result.tool_failures, 0)
        # 并发度 2 → 两批，总耗时约 0.6s；若并发度失效（4 线程）约 0.3s，若串行约 1.2s
        self.assertLess(elapsed, 1.6, f"并行执行耗时 {elapsed:.2f}s，未真正并发")

    # -- 5. thought 绝不作为答案返回 ------------------------------------
    def test_thought_is_never_returned_as_answer(self):
        """模型把 actions 写坏时，应回一条格式纠错观察，而不是把推理独白当答案。"""
        tool = self._tool("echo_tool", result="回显")

        def decide(prompt):
            return {
                "thought": "我应该先检索知识库里的相关论文，然后对比它们的方法差异。" * 3,
                "actions": "echo_tool",  # 类型错误：给了字符串而不是数组
            }

        agent = self._agent([tool], decide, max_iterations=3,
                            force_final_on_max_iterations=False)
        result = agent.run("格式错误测试")
        self.assertNotIn("我应该先检索", result.answer, "thought 泄漏成了用户可见答案")
        self.assertTrue(
            any("没有可识别的动作" in (s.content or "") for s in result.steps),
            "未产生格式纠错观察",
        )

    def test_mixed_batch_is_not_terminal(self):
        """同一批次里混用 final_answer 与真实工具调用时，不能只取 final_answer。"""
        tool = self._tool("echo_tool", result="回显结果")

        def decide(prompt):
            if "Observation" in prompt:
                return {"thought": "够了", "action": "final_answer",
                        "action_input": {"answer": "最终答案"}}
            return {"thought": "边查边答", "actions": [
                {"action": "echo_tool", "action_input": {"q": "x"}},
                {"action": "final_answer", "action_input": {"answer": "半句话"}},
            ]}

        agent = self._agent([tool], decide, max_iterations=3)
        result = agent.run("混合批次测试")
        self.assertEqual(result.tool_calls, 1, "混合批次里的工具调用被丢弃")
        self.assertEqual(result.answer, "最终答案")

    # -- 6. 重复调用：先提醒、再终止 ------------------------------------
    def test_loop_warns_before_terminating(self):
        """首次检测到重复应提醒模型换策略，只有提醒无效时才强制终止。"""
        tool = self._tool("loop_tool", result="固定观察结果")

        def decide(_prompt):
            return {"thought": "还需要更多信息", "action": "loop_tool",
                    "action_input": {"q": "same"}}

        agent = self._agent([tool], decide, max_iterations=10, loop_detect_window=2,
                            force_final_on_max_iterations=False)
        result = agent.run("循环测试")
        self.assertEqual(result.terminated_reason, "loop_detected")
        self.assertLess(result.iterations, 10, "应在达到最大轮次前终止")
        self.assertTrue(result.degraded)
        self.assertTrue(result.answer.strip(), "强制终止后仍必须给出可读回复")
        self.assertTrue(
            any("重复调用已经执行过" in (s.content or "") for s in result.steps),
            "未先提醒模型就终止了",
        )

    def test_semantic_duplicate_lookup_only_uses_recent_history(self):
        """语义重复检测只回看最近若干条，避免与很久以前的查询误撞。"""
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import ToolRegistry
        from src.agent.react_loop import _RunState

        agent = ReActAgent(
            registry=ToolRegistry([self._tool("search_tool")]),
            llm=self._llm(lambda _p: {}), memory=None, router=None,
            config=self._config(),
            agent_config=AgentConfig(loop_semantic_window=2),
        )
        state = _RunState()
        # 非空签名列表是进入语义检测级的前提（精确级为空说明还没调用过任何工具）
        state.seen_signatures = ['other_tool::{"query": "无关查询"}']
        state.tool_query_history["search_tool"] = [
            "transformer attention dataset",   # 很久以前
            "resnet residual learning",
            "bert pretraining corpus",
        ]
        # 与最早那条高度重合 → 已滚出窗口，不应判为重复
        self.assertFalse(
            agent._is_looping(state, ['search_tool::{"query": "transformer attention dataset"}'])
        )
        # 与最近那条高度重合 → 仍应判为重复
        self.assertTrue(
            agent._is_looping(state, ['search_tool::{"query": "bert pretraining corpus"}'])
        )

    # -- 7. 工具结果里的脏字段不致命 ------------------------------------
    def test_dirty_tool_payload_does_not_break_citations(self):
        """工具返回的 page 字段是 "p.12"/None/列表时，汇总引用不能抛异常。"""
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import ToolRegistry
        from src.llm import MockLLM
        from src.utils.types import AgentStep, StepType, ToolResult, ToolStatus

        agent = ReActAgent(
            registry=ToolRegistry([self._tool("knowledge_search")]),
            llm=MockLLM(None), memory=None, router=None, config=self._config(),
            agent_config=AgentConfig(),
        )
        step = AgentStep(
            index=0, type=StepType.OBSERVATION, content="x", tool_name="knowledge_search",
            tool_result=ToolResult(
                call_id="c", name="knowledge_search", status=ToolStatus.SUCCESS, content="x",
                data={"results": [
                    {"source": "a.pdf", "page": "p.12"},
                    {"source": "b.pdf", "page": None},
                    {"source": "c.pdf", "page": [3]},
                    {"source": "", "page": 1},          # 无来源名 → 跳过
                ]},
            ),
        )
        citations = agent._collect_citations([step], "见 【a.pdf p.12】")
        self.assertEqual(len(citations), 3)
        self.assertTrue(all(c.source for c in citations))
        self.assertIn(12, [c.page for c in citations])

    # -- 8. 观察结果总量有预算 ------------------------------------------
    def test_collected_observations_respect_budget(self):
        """兜底/强制收尾汇总的观察文本必须有总量上限，且优先保留最近结果。"""
        from src.agent import AgentConfig, ReActAgent
        from src.agent.base_tool import ToolRegistry
        from src.llm import MockLLM
        from src.utils.types import AgentStep, StepType

        agent = ReActAgent(
            registry=ToolRegistry([self._tool("t")]), llm=MockLLM(None), memory=None,
            router=None, config=self._config(),
            agent_config=AgentConfig(max_aggregate_chars=3000),
        )
        steps = [
            AgentStep(index=i, type=StepType.OBSERVATION, content="x" * 2000, tool_name=f"t{i}")
            for i in range(10)
        ]
        blob = agent._collect_observations(steps)
        self.assertLessEqual(len(blob), 3300)
        self.assertIn("t9", blob, "应优先保留最近的观察")
        self.assertNotIn("t0", blob, "过早的观察应被预算挤掉")


class TurnPersistenceTest(unittest.TestCase):
    """轮次落库的内容必须足以还原历史会话。

    回归：``_finalize`` 早期只写 question/answer/trajectory/usage/latency，
    引用与状态标记全部丢失 —— 重新打开历史会话时答案还在、**引用卡片全没了**，
    前端只能用"没有引用也没有轨迹"倒推缓存命中（于是降级回答被标成 ⚡命中缓存）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        from src.retrieval import create_retrieval_engine

        cls._ws = TempWorkspace()
        cls.cfg = cls._ws.config(
            chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50},
            agent={"max_iterations": 4, "tool_timeout": 8, "max_retries_per_tool": 1},
        )
        cls.engine = create_retrieval_engine(cls.cfg)
        cls.engine.index_documents(sample_documents())

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ws.cleanup()

    def test_citations_and_status_flags_are_persisted(self):
        from src.agent import create_agent

        agent = create_agent(config=self.cfg, retrieval_engine=self.engine)
        result = agent.run(
            "What datasets were used to train the Transformer?", session_id="persist"
        )
        self.assertTrue(result.citations, "前置条件：本轮应当产生引用")

        session = agent.memory.get_session("persist")
        self.assertIsNotNone(session)
        turn = session.turns[-1]

        self.assertEqual(
            len(turn.citations), len(result.citations),
            "落库的引用数与返回结果不一致 —— 历史会话恢复后引用会缺失",
        )
        self.assertEqual(turn.citations[0].source, result.citations[0].source)
        for key in ("cache_hit", "degraded", "terminated_reason"):
            self.assertIn(key, turn.metadata, f"轮次元数据缺少 {key}")
        self.assertEqual(turn.metadata["terminated_reason"], result.terminated_reason)

    def test_session_detail_exposes_turn_metadata(self):
        """会话详情接口要把 metadata 透出去，否则前端拿不到真实状态。"""
        from src.agent import create_agent
        from src.frontend.container import SystemContainer

        agent = create_agent(config=self.cfg, retrieval_engine=self.engine)
        agent.run("How does residual learning work in ResNet?", session_id="detail")

        container = SystemContainer(config=self.cfg)
        container.agent = agent
        container.memory = agent.memory
        detail = container.get_session("detail")
        self.assertIsNotNone(detail)
        self.assertIn("metadata", detail["turns"][-1])
        self.assertIn("cache_hit", detail["turns"][-1]["metadata"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
