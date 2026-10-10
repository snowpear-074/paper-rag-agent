# -*- coding: utf-8 -*-
"""模块二（RAG 生成层）测试：Prompt 工程、上下文构建、引用溯源、
流式输出、语义缓存、降级策略、可观测性。

运行::

    python -m unittest tests.test_generation -v
"""

from __future__ import annotations

import unittest

from tests.common import TempWorkspace, sample_documents


def _hits(texts, scores=None, confidences=None):
    """构造 RetrievedChunk 列表（便于测试上下文与引用逻辑）。

    默认让 confidence 与 score 同序（两者都反映相关性），
    这样按任一字段排序都应得到同一顺序。
    """
    from src.utils.types import Chunk, RetrievedChunk

    count = len(texts)
    score_list = scores if scores is not None else [1.0 - i * 0.1 for i in range(count)]
    confidence_list = confidences if confidences is not None else list(score_list)

    out = []
    for idx, text in enumerate(texts):
        out.append(
            RetrievedChunk(
                chunk=Chunk(doc_id=f"doc{idx % 2}", text=text, index=idx, page=idx + 1,
                            source=f"论文{idx % 2}", strategy="paper"),
                score=score_list[idx],
                confidence=confidence_list[idx],
                source_route="hybrid+rerank",
            )
        )
    return out


# ===========================================================================
# Prompt 工程
# ===========================================================================
class PromptTemplateTest(unittest.TestCase):
    """模板渲染、注册表、上下文拼接与引用解析。"""

    def test_registry_lookup_and_fallback(self):
        from src.generation import PROMPT_REGISTRY, get_prompt

        self.assertIn("rag_qa", PROMPT_REGISTRY)
        self.assertEqual(get_prompt("strict").name, "rag_strict")
        # 未知名称回退默认模板，而不是抛异常
        self.assertEqual(get_prompt("no_such_template").name, "rag_qa")

    def test_all_templates_render(self):
        from src.generation import PROMPT_REGISTRY

        for name, template in PROMPT_REGISTRY.items():
            with self.subTest(template=name):
                if name in ("no_context",):
                    rendered = template.render(question="Q", retrieval_note="0 条")
                elif name == "memory_summary":
                    rendered = template.render(history="用户：你好")
                else:
                    rendered = template.render(context="上下文内容", question="Q")
                self.assertIn("system", rendered)
                self.assertIn("user", rendered)
                self.assertTrue(rendered["system"].strip())

    def test_render_reports_missing_placeholder(self):
        from src.generation import PromptTemplate

        template = PromptTemplate(name="t", system="s", user="{a} and {b}")
        with self.assertRaises(KeyError) as ctx:
            template.render(a=1)
        self.assertIn("b", str(ctx.exception))

    def test_qa_prompt_contains_citation_instruction(self):
        from src.generation import RAG_QA_PROMPT

        self.assertIn("【文档名 p.页码】", RAG_QA_PROMPT.system)
        self.assertIn("未找到", RAG_QA_PROMPT.system, "缺少'资料不足时明说'的约束")

    def test_context_block_format(self):
        from src.generation import format_context_block

        hit = _hits(["正文内容"], scores=[0.8], confidences=[0.8])[0]
        block = format_context_block(hit, index=1, include_meta=True)
        self.assertIn("[1] 来源：", block)
        self.assertIn("p.1", block)
        self.assertIn("相关度 0.80", block)
        self.assertIn("正文内容", block)
        # 元信息里出现的是 confidence（绝对相关度），不是归一化后的 score
        self.assertNotIn("相关度 1.00", block)

    def test_context_block_truncates_at_sentence(self):
        from src.generation import format_context_block

        text = "第一句话内容完整。" * 40
        block = format_context_block(_hits([text])[0], index=1, max_chars=100)
        body = block.split("\n", 1)[1]
        self.assertLessEqual(len(body), 110)
        self.assertTrue(body.endswith("。"), f"未在句末截断: ...{body[-20:]!r}")

    def test_build_context_respects_budget(self):
        from src.generation import build_context

        hits = _hits([f"片段{i}内容。" * 50 for i in range(6)])
        context, selected = build_context(hits, max_chars=500, max_chunk_chars=200)
        self.assertLess(len(context), 1400)
        self.assertGreaterEqual(len(selected), 1)
        # 至少保留一个片段（即使超预算也不能返回空上下文）
        context, selected = build_context(hits, max_chars=10, max_chunk_chars=100)
        self.assertEqual(len(selected), 1)

    def test_build_context_orders_by_relevance(self):
        from src.generation import build_context

        hits = _hits(["低相关", "高相关", "中相关"], scores=[0.3, 0.9, 0.6])
        context, selected = build_context(hits, max_chars=10000)
        self.assertIn("高相关", context.split("\n\n")[0], "Top-1 应排在最前，利用首因效应")

    def test_build_citations_matches_chunk_order(self):
        from src.generation import build_citations

        hits = _hits(["a", "b", "c"])
        citations = build_citations(hits)
        self.assertEqual([c.index for c in citations], [1, 2, 3])
        self.assertEqual(citations[0].page, 1)
        self.assertEqual(citations[0].source, "论文0")
        self.assertGreater(citations[0].score, 0)

    def test_citation_pattern_variants(self):
        from src.generation import CITATION_PATTERN

        cases = {
            "【Attention Is All You Need p.3】": ("Attention Is All You Need", 3),
            "【论文A p.12】": ("论文A", 12),
            "【BERT 第5页】": ("BERT", 5),
            "【ResNet P.7】": ("ResNet", 7),
        }
        for text, (source, page) in cases.items():
            with self.subTest(text=text):
                match = CITATION_PATTERN.search(text)
                self.assertIsNotNone(match, f"未匹配: {text}")
                self.assertEqual(match.group("source").strip(), source)
                self.assertEqual(int(match.group("page")), page)

    def test_parse_citations_filters_to_referenced_only(self):
        from src.generation import build_citations, parse_citations

        hits = _hits(["内容一", "内容二", "内容三"])
        available = build_citations(hits)
        answer = "第一个结论【论文0 p.1】。第二个结论【论文1 p.2】。"
        used = parse_citations(answer, available)
        self.assertEqual(len(used), 2)
        self.assertEqual({c.source for c in used}, {"论文0", "论文1"})

    def test_parse_citations_detects_hallucinated_source(self):
        from src.generation import build_citations, parse_citations

        available = build_citations(_hits(["内容一"]))
        used = parse_citations("某结论【根本不存在的论文 p.99】。", available)
        self.assertEqual(len(used), 1)
        self.assertEqual(used[0].source, "根本不存在的论文")
        self.assertEqual(used[0].chunk_id, "", "幻觉引用不应关联到真实 chunk")

    def test_parse_citations_fuzzy_source_match(self):
        from src.generation import build_citations, parse_citations

        available = build_citations(_hits(["内容"]))  # source = "论文0"
        used = parse_citations("结论【论文0.pdf p.1】。", available)
        self.assertEqual(len(used), 1)
        self.assertEqual(used[0].chunk_id, available[0].chunk_id, "应模糊匹配到真实来源")

    def test_validate_citation_coverage(self):
        from src.generation import validate_citation_coverage

        good = "这是一句足够长的实质性陈述内容，用于测试引用覆盖率。【论文A p.1】另一句同样足够长的陈述内容。【论文A p.2】"
        result = validate_citation_coverage(good)
        self.assertGreaterEqual(result["coverage"], 0.5)
        self.assertTrue(result["passed"])

        bad = validate_citation_coverage("这是一句足够长的实质性陈述内容，但是完全没有引用标注哦。另一句也很长但没有引用。")
        self.assertFalse(bad["passed"])

    def test_validate_citation_coverage_empty(self):
        from src.generation import validate_citation_coverage

        result = validate_citation_coverage("")
        self.assertEqual(result["sentences"], 0)
        self.assertFalse(result["passed"])


# ===========================================================================
# 流式输出
# ===========================================================================
class StreamingTest(unittest.TestCase):
    """token 流 → 结构化事件（含跨 token 的引用识别）。"""

    def _events(self, pieces, citations=()):
        from src.generation import stream_events

        return list(stream_events(iter(pieces), citations=citations))

    def test_basic_token_flow(self):
        events = self._events(["你好", "，", "世界"])
        types = [e.type for e in events]
        self.assertEqual(types.count("token"), 3)
        self.assertEqual(types[-1], "done")

    def test_citation_recognised_across_token_boundaries(self):
        """引用标记被切开成多个 token 时也必须能识别（这是真实流式的常态）。"""
        from src.generation import build_citations

        hits = _hits(["内容"])
        citations = build_citations(hits)
        pieces = ["结论是", "【论文", "0 p.", "1", "】", "结束"]
        events = self._events(pieces, citations)
        citation_events = [e for e in events if e.type == "citation"]
        self.assertEqual(len(citation_events), 1, f"事件: {[e.type for e in events]}")
        self.assertEqual(citation_events[0].data["page"], 1)
        # 引用不应泄进普通 token 文本
        text = "".join(e.data["text"] for e in events if e.type == "token")
        self.assertNotIn("【论文", text)
        self.assertIn("结论是", text)
        self.assertIn("结束", text)

    def test_done_event_carries_full_answer(self):
        events = self._events(["完整", "答案"])
        done = events[-1]
        self.assertEqual(done.type, "done")
        self.assertEqual(done.data["answer"], "完整答案")

    def test_done_event_keeps_raw_answer_with_citation_markers(self):
        """DONE 必须给出**含引用标记**的原文，前端靠它渲染行内引用角标。

        回归：早期实现只保留"去标记"版本作为权威答案，于是流式回答的行内
        ``【论文 p.3】`` 全部消失（前端 markdown.js 正是靠它生成可点击角标），
        而且写进语义缓存的答案也没有标记 —— 与非流式 ``/api/ask`` 路径不一致。
        """
        from src.generation import build_citations

        citations = build_citations(_hits(["内容"]))
        events = self._events(["结论是", "【论文0 p.1】", "结束"], citations)
        done = events[-1]

        self.assertEqual(done.type, "done")
        # 去标记版（打字机文本）
        self.assertEqual(done.data["answer"], "结论是结束")
        # 原文版（权威）
        self.assertEqual(done.data["answer_raw"], "结论是【论文0 p.1】结束")

    def test_collect_stream_prefers_raw_answer(self):
        from src.generation import build_citations, collect_stream

        citations = build_citations(_hits(["内容"]))
        events = self._events(["结论是", "【论文0 p.1】", "结束"], citations)
        self.assertIn("【论文0 p.1】", collect_stream(iter(events))["answer"])

    def test_unclosed_bracket_flushed_at_end(self):
        """模型只输出了半个方括号时，内容不能被吞掉。"""
        events = self._events(["正文", "【未闭合"])
        text = "".join(e.data["text"] for e in events if e.type == "token")
        self.assertIn("未闭合", text)

    def test_duplicate_citation_emitted_once(self):
        from src.generation import build_citations

        citations = build_citations(_hits(["内容"]))
        events = self._events(["a【论文0 p.1】", "b【论文0 p.1】"], citations)
        self.assertEqual(len([e for e in events if e.type == "citation"]), 1)

    def test_retrieval_event_emitted_first(self):
        from src.generation import stream_events

        hits = _hits(["内容一", "内容二"])
        events = list(stream_events(iter(["x"]), citations=[], emit_retrieval=True, retrieved=hits))
        self.assertEqual(events[0].type, "retrieval")
        self.assertEqual(events[0].data["num"], 2)

    def test_sse_serialisation_format(self):
        from src.generation import StreamEvent

        payload = StreamEvent("token", {"text": "中文"}).to_sse()
        self.assertTrue(payload.startswith("event: token\ndata: "))
        self.assertTrue(payload.endswith("\n\n"))
        self.assertIn("中文", payload)

    def test_collect_stream_aggregates(self):
        from src.generation import collect_stream, stream_events

        events = stream_events(iter(["部分一", "部分二"]))
        result = collect_stream(events)
        self.assertEqual(result["answer"], "部分一 部分二".replace(" ", ""))
        self.assertIn("usage", result)

    def test_stream_error_produces_error_event(self):
        from src.generation import stream_events

        def broken():
            yield "开始"
            raise RuntimeError("模拟生成中断")

        events = list(stream_events(broken()))
        self.assertTrue(any(e.type == "error" for e in events))

    def test_stream_buffer_partial_flush(self):
        from src.generation import CITATION_PATTERN, StreamBuffer

        buffer = StreamBuffer(pattern=CITATION_PATTERN)
        output = buffer.feed("正常文本")
        self.assertEqual("".join(t for t, _ in output), "正常文本")
        self.assertEqual(buffer.pending, "")


# ===========================================================================
# 语义缓存
# ===========================================================================
class SemanticCacheTest(unittest.TestCase):
    """精确命中、语义命中、TTL、容量淘汰、参数隔离。"""

    def _cache(self, ws, **kwargs):
        from src.generation import create_cache

        return create_cache(ws.config(generation={"cache": kwargs}))

    def test_exact_hit(self):
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws)
            answer = RAGAnswer(question="Q", answer="A", latency_ms=1.0)
            self.assertTrue(cache.put("什么是注意力机制", answer, model="m", params={"k": 1}))
            hit = cache.get("什么是注意力机制", model="m", params={"k": 1})
            self.assertIsNotNone(hit)
            self.assertEqual(hit.answer, "A")

    def test_miss_on_different_question(self):
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws)
            cache.put("问题一", RAGAnswer(answer="A"), model="m", params={})
            self.assertIsNone(cache.get("完全不同的另一个问题", model="m", params={}))

    def test_semantic_hit_with_similar_vector(self):
        """向量高度相似时应命中缓存（语义缓存的本质）。"""
        import numpy as np

        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws, similarity_threshold=0.9)
            cache.put("Transformer 用了什么数据集", RAGAnswer(answer="WMT 2014"), model="m", params={})
            # 几乎相同的向量（余弦 ≈ 0.9999）
            query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            stored = np.array([1.0, 0.001, 0.0, 0.0], dtype=np.float32)
            cache._memory.clear()
            cache.put(
                "Transformer 用了什么数据集",
                RAGAnswer(answer="WMT 2014"),
                vector=stored,
                model="m",
                params={},
            )
            hit = cache.get("换个说法的同一问题", vector=query, model="m", params={})
            self.assertIsNotNone(hit)
            self.assertEqual(hit.answer, "WMT 2014")

    def test_model_and_params_partition_cache(self):
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws)
            cache.put("Q", RAGAnswer(answer="A1"), model="model-a", params={"k": 5})
            self.assertIsNotNone(cache.get("Q", model="model-a", params={"k": 5}))
            # 换模型或换参数都不应误用旧答案
            self.assertIsNone(cache.get("Q", model="model-b", params={"k": 5}))
            self.assertIsNone(cache.get("Q", model="model-a", params={"k": 10}))

    def test_degraded_answer_not_cached(self):
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws)
            degraded = RAGAnswer(answer="降级回答", degraded=True, degrade_reason="no_result")
            self.assertFalse(cache.put("Q", degraded, model="m", params={}))
            self.assertIsNone(cache.get("Q", model="m", params={}))

    def test_empty_answer_not_cached(self):
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws)
            self.assertFalse(cache.put("Q", RAGAnswer(answer=""), model="m", params={}))

    def test_disabled_cache(self):
        from src.generation import NullCache, create_cache
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = create_cache(ws.config(generation={"cache": {"enabled": False}}))
            self.assertIsInstance(cache, NullCache)
            self.assertFalse(cache.put("Q", RAGAnswer(answer="A")))
            self.assertIsNone(cache.get("Q"))

    def test_threshold_outside_range_is_clamped(self):
        """阈值写成 1.5 或 -1 属于配置错误，不能静默变成"永不命中/永远命中"。"""
        with TempWorkspace() as ws:
            self.assertEqual(self._cache(ws, similarity_threshold=1.5).threshold, 1.0)
        with TempWorkspace() as ws:
            self.assertEqual(self._cache(ws, similarity_threshold=-1).threshold, 0.0)
        with TempWorkspace() as ws:
            cache = self._cache(ws, similarity_threshold=0.9, memory_size=0, max_entries=0)
            self.assertGreaterEqual(cache.memory_size, 1)
            self.assertGreaterEqual(cache.max_entries, 1)

    def test_semantic_hit_from_memory_layer(self):
        """内存层也要支持向量近邻（不必等落盘再命中）。"""
        import numpy as np

        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws, similarity_threshold=0.9)
            cache.put(
                "Transformer 用了什么数据集",
                RAGAnswer(answer="WMT 2014"),
                vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                model="m",
                params={},
            )
            hit = cache.get(
                "换个说法的同一问题",
                vector=np.array([1.0, 0.001, 0.0, 0.0], dtype=np.float32),
                model="m",
                params={},
            )
            self.assertIsNotNone(hit)
            self.assertEqual(hit.answer, "WMT 2014")

    def test_database_hit_returns_complete_entry(self):
        """走数据库近邻路径时要按 key 把完整条目读回来（含引用与检索片段）。"""
        import numpy as np

        from src.utils.types import Citation, RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws, similarity_threshold=0.9)
            answer = RAGAnswer(
                question="Q",
                answer="答案 【a.pdf p.3】",
                citations=[Citation(index=1, source="a.pdf", page=3, snippet="片段")],
            )
            cache.put(
                "问题原文",
                answer,
                vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                model="m",
                params={"k": 5},
            )
            cache._memory.clear()  # 强制走 SQLite 路径
            hit = cache.get(
                "另一个措辞",
                vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                model="m",
                params={"k": 5},
            )
            self.assertIsNotNone(hit)
            self.assertEqual(hit.answer, "答案 【a.pdf p.3】")
            self.assertEqual(len(hit.citations), 1)
            self.assertEqual(hit.citations[0]["source"], "a.pdf")

    def test_vector_dimension_mismatch_does_not_crash(self):
        """换了 embedding 维度后，旧向量不能把检索/命中流程弄崩。"""
        import numpy as np

        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws, similarity_threshold=0.5)
            cache.put("旧问题", RAGAnswer(answer="A"), vector=np.zeros(8, dtype=np.float32), model="m", params={})
            self.assertIsNone(
                cache.get("新问题", vector=np.ones(4, dtype=np.float32), model="m", params={})
            )

    def test_clear_and_stats(self):
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws)
            for i in range(5):
                cache.put(f"问题{i}", RAGAnswer(answer=f"答案{i}"), model="m", params={})
            stats = cache.stats()
            self.assertEqual(stats["stores"], 5)
            self.assertGreaterEqual(stats["entries"], 5)
            self.assertGreaterEqual(cache.clear(), 5)
            self.assertIsNone(cache.get("问题0", model="m", params={}))

    def test_hit_rate_accounting(self):
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cache = self._cache(ws)
            cache.put("Q", RAGAnswer(answer="A"), model="m", params={})
            cache.get("Q", model="m", params={})
            cache.get("miss", model="m", params={})
            stats = cache.stats()
            self.assertEqual(stats["lookups"], 2)
            self.assertEqual(stats["hits"], 1)
            self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_persist_across_instances(self):
        from src.generation import create_cache
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            cfg = ws.config()
            cache = create_cache(cfg)
            cache.put("持久化问题", RAGAnswer(answer="持久化答案"), model="m", params={})

            reopened = create_cache(cfg)
            hit = reopened.get("持久化问题", model="m", params={})
            self.assertIsNotNone(hit)
            self.assertEqual(hit.answer, "持久化答案")


# ===========================================================================
# RAG 流水线（集成）
# ===========================================================================
class RAGPipelineTest(unittest.TestCase):
    """端到端：检索 → 上下文 → 生成 → 引用 → 缓存 → 降级 → 可观测。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = sample_documents()

    def _pipeline(self, ws, **overrides):
        from src.generation import create_rag_pipeline
        from src.retrieval import create_retrieval_engine

        cfg = ws.config(
            chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50},
            **overrides,
        )
        engine = create_retrieval_engine(cfg)
        engine.index_documents(self.documents)
        return create_rag_pipeline(config=cfg, retrieval_engine=engine), engine

    def test_answer_returns_citations_and_metadata(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            answer = pipeline.answer("What datasets were used to train the Transformer?")
            self.assertTrue(answer.answer.strip())
            self.assertGreater(len(answer.retrieved), 0)
            self.assertGreater(answer.latency_ms, 0.0)
            self.assertEqual(answer.model, "mock-model")
            self.assertGreater(answer.usage.total_tokens, 0)

    def test_in_domain_question_not_degraded(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            answer = pipeline.answer("How does residual learning work in ResNet?")
            self.assertFalse(answer.degraded, f"误判降级: {answer.degrade_reason}")
            self.assertGreater(answer.top_confidence, 0.25)

    def test_out_of_domain_question_triggers_degradation(self):
        """与知识库完全无关的问题必须触发低相关性降级。

        这是真实踩坑点：若用归一化后的相对分（Top-1 恒为 1.0）判断，
        这类问题永远不会降级，系统会拿无关片段去生成答案。
        """
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            answer = pipeline.answer("今天天气怎么样？适合去公园吗？")
            self.assertTrue(answer.degraded)
            self.assertEqual(answer.degrade_reason, "low_relevance")
            self.assertLess(answer.top_confidence, 0.25)

    def test_empty_knowledge_base_degrades(self):
        with TempWorkspace() as ws:
            from src.generation import create_rag_pipeline
            from src.retrieval import create_retrieval_engine

            cfg = ws.config()
            engine = create_retrieval_engine(cfg)
            pipeline = create_rag_pipeline(config=cfg, retrieval_engine=engine)
            answer = pipeline.answer("任何问题")
            self.assertTrue(answer.degraded)
            self.assertEqual(answer.degrade_reason, "no_result")
            self.assertTrue(answer.answer.strip(), "降级也必须给出可读回复")

    def test_empty_question_handled(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            answer = pipeline.answer("   ")
            self.assertTrue(answer.degraded)
            self.assertEqual(answer.degrade_reason, "empty_question")

    def test_semantic_cache_used_on_repeat(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            question = "What is multi-head attention?"
            first = pipeline.answer(question, use_cache=True)
            self.assertFalse(first.cache_hit)
            second = pipeline.answer(question, use_cache=True)
            self.assertTrue(second.cache_hit, "重复问题应命中语义缓存")
            self.assertLess(second.latency_ms, first.latency_ms + 1)

    def test_cache_disabled_flag(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            question = "What is masked language model?"
            pipeline.answer(question, use_cache=False)
            again = pipeline.answer(question, use_cache=False)
            self.assertFalse(again.cache_hit)

    def test_stream_produces_tokens_and_final(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            events = list(pipeline.answer_stream("What BLEU score did the Transformer achieve?"))
            types = [e.type for e in events]
            self.assertIn("retrieval", types)
            self.assertIn("token", types)
            self.assertIn("final", types)
            final = next(e for e in events if e.type == "final")
            self.assertTrue(final.data["answer"].strip())
            self.assertGreater(final.data["latency_ms"], 0)

    def test_stream_retrieval_event_precedes_tokens(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            events = list(pipeline.answer_stream("How many layers in ResNet-152?"))
            types = [e.type for e in events]
            self.assertLess(types.index("retrieval"), types.index("token"))

    def test_trace_and_metrics_recorded(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            pipeline.answer("What is self-attention?")
            stats = pipeline.stats()
            self.assertGreaterEqual(stats["total"], 1)
            self.assertIn("cache", stats)
            trace_file = ws.path("logs", "traces.jsonl")
            self.assertTrue(trace_file.exists(), "应写入 trace 日志")
            content = trace_file.read_text(encoding="utf-8")
            self.assertIn("What is self-attention?", content)

    def test_llm_error_is_degraded_not_crash(self):
        """LLM 调用失败应转成友好提示 + 降级标记，而不是抛异常。"""
        with TempWorkspace() as ws:
            from src.generation import RAGConfig, RAGPipeline
            from src.retrieval import create_retrieval_engine

            cfg = ws.config(chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50})
            engine = create_retrieval_engine(cfg)
            engine.index_documents(self.documents)

            class BrokenLLM:
                model = "broken"

                def chat_with_usage(self, *args, **kwargs):
                    raise RuntimeError("模拟 LLM 服务不可用")

                def chat(self, *args, **kwargs):
                    raise RuntimeError("模拟 LLM 服务不可用")

                def stream(self, *args, **kwargs):
                    raise RuntimeError("模拟 LLM 服务不可用")

                def health_check(self):
                    return {"ok": False, "error": "broken"}

            pipeline = RAGPipeline(engine, BrokenLLM(), cache=None, config=cfg,
                                   rag_config=RAGConfig.from_config(cfg))
            answer = pipeline.answer("What is attention?")
            self.assertTrue(answer.degraded)
            self.assertEqual(answer.degrade_reason, "llm_error")
            self.assertIn("错误", answer.answer)
            self.assertGreater(len(answer.retrieved), 0, "检索部分仍应成功")

    def test_prompt_template_switching(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            strict = pipeline.answer("What is BERT?", prompt_name="rag_strict", use_cache=False)
            self.assertTrue(strict.answer.strip())
            summary = pipeline.answer("BERT", prompt_name="rag_summary", use_cache=False)
            self.assertTrue(summary.answer.strip())

    def test_top_k_respected(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            answer = pipeline.answer("attention", top_k=2, use_cache=False)
            self.assertLessEqual(len(answer.retrieved), 2)

    def test_health_check(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            health = pipeline.health_check()
            self.assertIn("llm", health)
            self.assertIn("pipeline", health)

    def test_clear_cache(self):
        with TempWorkspace() as ws:
            pipeline, _engine = self._pipeline(ws)
            pipeline.answer("What is attention?", use_cache=True)
            removed = pipeline.clear_cache()
            self.assertGreaterEqual(removed, 1)


class CacheKeyTest(unittest.TestCase):
    """缓存键必须覆盖所有影响答案的输入，否则会串用错误答案。

    回归的三类错误命中：

    1. 调用方显式换提示词模板（``prompt_name="rag_strict"``），
       键里却写死配置默认模板 —— 两个模板的答案互相串用；
    2. 多轮追问（"它和 ResNet 相比如何？"）问题原文相同、上文不同 ——
       第二轮直接复用第一轮的答案；
    3. ``max_tokens`` 不同（截断长度不同）却共用同一条缓存。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = sample_documents()

    def _pipeline(self, ws, **overrides):
        from src.generation import create_rag_pipeline
        from src.retrieval import create_retrieval_engine

        cfg = ws.config(chunking={"strategy": "recursive", "chunk_size": 300}, **overrides)
        engine = create_retrieval_engine(cfg)
        engine.index_documents(self.documents)
        return create_rag_pipeline(config=cfg, retrieval_engine=engine)

    # -- 键的构成 ---------------------------------------------------------
    def test_prompt_template_participates_in_key(self):
        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws)
            default = pipeline._cache_params("full", 5)
            strict = pipeline._cache_params("full", 5, prompt_name="rag_strict")
            self.assertEqual(default["prompt"], "rag_qa")
            self.assertEqual(strict["prompt"], "rag_strict")
            self.assertNotEqual(default, strict)

    def test_no_context_template_is_reflected_in_key(self):
        """关闭纯模型回退时，零检索结果会切到 no_context 模板 —— 键必须跟着变。"""
        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws, generation={"fallback_to_llm": False})
            on_result = pipeline._cache_params("full", 5, degrade="")
            no_result = pipeline._cache_params("full", 5, degrade="no_result")
            self.assertEqual(on_result["prompt"], "rag_qa")
            self.assertEqual(no_result["prompt"], "no_context")
        with TempWorkspace() as ws:
            # 保留回退时不应切模板（仍用 rag_qa 让模型自行作答）
            pipeline = self._pipeline(ws, generation={"fallback_to_llm": True})
            self.assertEqual(
                pipeline._cache_params("full", 5, degrade="no_result")["prompt"], "rag_qa"
            )

    def test_max_tokens_participates_in_key(self):
        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws)
            before = pipeline._cache_params("full", 5)
            pipeline.rag.max_tokens = before["max_tokens"] + 512
            self.assertNotEqual(pipeline._cache_params("full", 5), before)

    def test_history_fingerprint_is_stable_and_discriminating(self):
        from src.utils.types import Message

        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws)
            history_a = [Message.user("第一篇论文讲了什么"), Message.assistant("讲了注意力机制")]
            history_b = [Message.user("第二篇论文讲了什么"), Message.assistant("讲了残差学习")]
            self.assertEqual(
                pipeline._history_fingerprint(history_a), pipeline._history_fingerprint(history_a)
            )
            self.assertNotEqual(
                pipeline._history_fingerprint(history_a), pipeline._history_fingerprint(history_b)
            )
            self.assertEqual(pipeline._history_fingerprint(None), "")
            self.assertEqual(pipeline._history_fingerprint([]), "")

    def test_history_fingerprint_only_covers_injected_window(self):
        """只有真正注入 LLM 的末尾若干条参与指纹（避免无关早期轮次降低命中率）。"""
        from src.utils.types import Message

        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws)
            tail = [
                message
                for i in range(5, 10)
                for message in (Message.user(f"问题{i}"), Message.assistant(f"回答{i}"))
            ]
            short = [Message.user("旧问题"), Message.assistant("旧回答")] + tail
            self.assertEqual(
                pipeline._history_fingerprint(short), pipeline._history_fingerprint(tail)
            )

    # -- 端到端：错误的命中必须真的被挡住 --------------------------------
    def test_same_question_with_different_history_is_not_a_hit(self):
        from src.utils.types import Message

        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws)
            question = "它和 ResNet 相比如何？"
            history_a = [Message.user("Transformer 是什么"), Message.assistant("一种序列建模架构")]
            history_b = [Message.user("BERT 是什么"), Message.assistant("一种预训练语言模型")]

            first = pipeline.answer(question, history=history_a, use_cache=True)
            self.assertFalse(first.cache_hit)
            second = pipeline.answer(question, history=history_b, use_cache=True)
            self.assertFalse(second.cache_hit, "上文不同的问题复用了缓存答案")

            third = pipeline.answer(question, history=history_a, use_cache=True)
            self.assertTrue(third.cache_hit, "上文相同的问题应能命中缓存")

    def test_switching_prompt_template_is_not_a_hit(self):
        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws)
            question = "What is multi-head attention?"
            pipeline.answer(question, prompt_name="rag_qa", use_cache=True)
            other = pipeline.answer(question, prompt_name="rag_strict", use_cache=True)
            self.assertFalse(other.cache_hit, "换了提示词模板却命中了另一个模板的答案")

    def test_stream_cache_key_matches_non_stream(self):
        """流式与非流式必须共用同一把键（否则同一问题会存两份、永远命不中）。"""
        with TempWorkspace() as ws:
            pipeline = self._pipeline(ws)
            question = "What is residual learning?"
            non_stream = pipeline.answer(question, use_cache=True)
            self.assertFalse(non_stream.cache_hit)
            streamed = list(pipeline.answer_stream(question, use_cache=True))
            finals = [e for e in streamed if e.type == "final"]
            self.assertTrue(finals)
            self.assertTrue(finals[0].data["cache_hit"], "流式路径没有复用非流式写入的缓存")


class RAGConfigTest(unittest.TestCase):
    """配置装载：生成参数不能悄悄退回 dataclass 默认值。"""

    def test_llm_section_drives_temperature_and_max_tokens(self):
        """回归：早期 from_config 漏读 llm.temperature / llm.max_tokens，
        config.yaml 里的 ``max_tokens: 2048`` 被默认值 1024 覆盖，长答案被截断。"""
        from src.generation import RAGConfig

        with TempWorkspace() as ws:
            cfg = ws.config(llm={"temperature": 0.7, "max_tokens": 2048})
            rag = RAGConfig.from_config(cfg)
            self.assertEqual(rag.max_tokens, 2048)
            self.assertAlmostEqual(rag.temperature, 0.7)

    def test_missing_llm_values_keep_defaults(self):
        from src.generation import RAGConfig
        from src.utils.config import Config

        # 用最小配置直接构造（不经过 config.yaml 的默认值合并），
        # 才能真的走到"键缺失 → 用默认值"这一分支
        cfg = Config(
            {
                "retrieval": {"top_k": 5, "low_relevance_threshold": 0.25},
                "generation": {
                    "max_context_chars": 6000,
                    "max_chunk_chars": 1500,
                    "fallback_to_llm": True,
                },
                "llm": {},
            }
        )
        rag = RAGConfig.from_config(cfg)
        self.assertEqual(rag.max_tokens, RAGConfig.max_tokens)
        self.assertAlmostEqual(rag.temperature, RAGConfig.temperature)


if __name__ == "__main__":
    unittest.main(verbosity=2)
