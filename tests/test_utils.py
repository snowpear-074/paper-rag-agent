# -*- coding: utf-8 -*-
"""基础设施测试：配置系统、文本处理、数据结构、存储、日志。

运行::

    python -m unittest tests.test_utils -v
"""

from __future__ import annotations

import json
import threading
import time
import unittest

from tests.common import PROJECT_ROOT, TempWorkspace


# ===========================================================================
# 配置系统
# ===========================================================================
class ConfigTest(unittest.TestCase):
    """配置读取、覆盖优先级与访问语义。"""

    def setUp(self) -> None:
        from src.utils.config import load_config

        self.cfg = load_config()

    def test_defaults_loaded(self):
        """默认配置应指向在线大模型（DeepSeek），且密钥为空（由密封库/环境变量注入）。"""
        self.assertEqual(str(self.cfg.llm.provider), "openai")
        self.assertEqual(str(self.cfg.llm.model), "deepseek-chat")
        self.assertIn("deepseek.com", str(self.cfg.llm.base_url))
        self.assertGreater(int(self.cfg.retrieval.top_k), 0)
        self.assertIn(str(self.cfg.chunking.strategy), ("fixed", "recursive", "semantic", "paper"))

    def test_secret_not_stored_in_config_file(self):
        """回归测试：config.yaml 中绝不能出现明文 API Key。

        密钥必须走密封凭据库（data/secrets/）或环境变量。
        """
        import re as _re

        from src.utils.config import DEFAULT_CONFIG_PATH

        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        for match in _re.finditer(r"^\s*api_key\s*:\s*(.+)$", text, _re.MULTILINE):
            value = match.group(1).split("#")[0].strip().strip("\"'")
            self.assertEqual(value, "", f"config.yaml 中出现非空 api_key：{value[:12]}...")
        self.assertNotIn("sk-", text, "config.yaml 中疑似存在明文密钥")

    def test_attribute_and_mapping_access_agree(self):
        self.assertEqual(self.cfg.retrieval.top_k, self.cfg["retrieval"]["top_k"])
        self.assertEqual(self.cfg.get("retrieval.top_k"), self.cfg.retrieval.top_k)

    def test_nested_get_with_default(self):
        self.assertEqual(self.cfg.get("no.such.path", "fallback"), "fallback")
        # 嵌套 section 上的 get 也应可用（关键：方法不能被配置键遮蔽）
        self.assertEqual(self.cfg.retrieval.get("no_such_key", 42), 42)

    def test_missing_attribute_raises(self):
        """缺失配置项必须快速失败，而不是返回空占位对象。

        这是真实踩坑：早期实现返回空 Config，导致错误被推迟到很远的地方才爆发
        （例如 str(cfg.x) 得到 "Config({})" 后再调用 .lower() 才报错），极难定位。
        """
        with self.assertRaises(AttributeError):
            _ = self.cfg.definitely_not_a_section

    def test_str_of_single_scalar_key(self):
        """被当作值使用的配置节点，str() 应得到标量本身。"""
        self.assertEqual(str(self.cfg.retrieval.rrf_k), "60")

    def test_str_of_section_is_empty_not_json(self):
        """回归测试：多键 section 的 str() 必须是空串，绝不能是 JSON。

        真实踩坑：``str(cfg.memory)`` 曾返回
        ``'{"backend": "sqlite", "path": ...}'``。当调用方本意是
        ``cfg.memory.backend`` 却误写成 ``str(cfg.memory)`` 时，这段 JSON
        会被当成合法的配置值一路传下去（缓存后端名变成一整段 JSON），
        错误延迟到很远才以难以理解的方式暴露。现在返回空串，
        配合 ``or 默认值`` 的写法会立即回退到安全默认值。
        """
        for section in ("memory", "retrieval", "llm", "generation"):
            with self.subTest(section=section):
                text = str(getattr(self.cfg, section))
                self.assertEqual(text, "", f"str(cfg.{section}) 应为空串，实际: {text[:60]!r}")
                self.assertNotIn("{", text)

    def test_yaml_inline_comment_after_quoted_scalar(self):
        """回归测试：``key: "value"   # 注释`` 必须解析出 value，而不是整行。

        真实踩坑：内置 YAML 回退解析器看到首字符是引号就跳过注释剥离，
        于是 ``memory.backend: "sqlite"  # sqlite | memory`` 被解析成
        字面量 ``'"sqlite"  # sqlite | memory'``，缓存/记忆后端因此静默退化成
        纯内存模式（表现为"重启后数据全丢"）。
        """
        from src.utils.config import _parse_scalar

        self.assertEqual(_parse_scalar('"sqlite"      # sqlite | memory'), "sqlite")
        self.assertEqual(_parse_scalar("'numpy'  # comment"), "numpy")
        self.assertEqual(_parse_scalar("sqlite   # comment"), "sqlite")
        self.assertEqual(_parse_scalar("60  # 默认值"), 60)
        # 引号内部的 # 必须保留
        self.assertEqual(_parse_scalar('"a # b"'), "a # b")

    def test_config_yaml_parses_cleanly_with_fallback(self):
        """config.yaml 经内置解析器读出的值必须是干净标量（无残留注释）。"""
        from src.utils.config import DEFAULT_CONFIG_PATH, _fallback_yaml_load

        parsed = _fallback_yaml_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(parsed["memory"]["backend"], "sqlite")
        self.assertEqual(parsed["vector_store"]["backend"], "numpy")
        self.assertEqual(parsed["embedding"]["provider"], "hash")
        for section in ("llm", "embedding", "memory", "generation", "agent", "server"):
            for key, value in parsed.get(section, {}).items():
                if isinstance(value, str):
                    self.assertNotIn("#", value, f"{section}.{key} 残留注释: {value!r}")

    def test_list_elements_are_wrapped(self):
        """列表元素也要被包装成 Config，支持链式属性访问。"""
        from src.utils.config import Config

        cfg = Config({"rows": [{"name": "a", "n": 1}, {"name": "b", "n": 2}]})
        self.assertEqual(cfg.rows[0].name, "a")
        self.assertEqual(cfg.rows[1].n, 2)

    def test_method_names_win_over_config_keys(self):
        """类方法优先于同名配置键 —— 用 `cfg["key"]` 读取被遮蔽的键。

        这是刻意的优先级设计：若让配置键获胜，一个名为 ``items``/``get`` 的键
        会把方法遮蔽成 Config 对象，进而在调用时抛出 "Config object is not
        callable" 这类难以定位的错误。
        """
        from src.utils.config import Config

        cfg = Config({"items": [{"name": "a"}], "get": "shadowed"})
        # 方法可用
        self.assertTrue(callable(cfg.get))
        self.assertEqual(cfg.get("nothing", "fallback"), "fallback")
        self.assertEqual(len(list(cfg.items())), 2)
        # 同名配置键通过下标访问
        self.assertEqual(cfg["items"][0].name, "a")
        self.assertEqual(cfg["get"], "shadowed")

    def test_overrides_deep_merge(self):
        from src.utils.config import load_config

        cfg = load_config(None, {"retrieval": {"top_k": 9}})
        self.assertEqual(int(cfg.retrieval.top_k), 9)
        # 未被覆盖的同级字段保持默认
        self.assertGreater(int(cfg.retrieval.candidate_k), 0)

    def test_merged_with_does_not_mutate(self):
        original = int(self.cfg.retrieval.top_k)
        clone = self.cfg.merged_with({"retrieval": {"top_k": original + 100}})
        self.assertEqual(int(clone.retrieval.top_k), original + 100)
        self.assertEqual(int(self.cfg.retrieval.top_k), original)

    def test_to_dict_is_plain(self):
        data = self.cfg.to_dict()
        self.assertIsInstance(data, dict)
        self.assertIsInstance(data["retrieval"], dict)
        json.dumps(data, ensure_ascii=False)  # 必须可序列化

    def test_resolve_path_is_absolute(self):
        resolved = self.cfg.resolve_path("./data/cache/x.db")
        self.assertTrue(resolved.is_absolute())
        self.assertTrue(str(resolved).startswith(str(PROJECT_ROOT)))

    def test_env_override(self):
        import os

        from src.utils.config import load_config

        os.environ["APP_RETRIEVAL__TOP_K"] = "17"
        try:
            cfg = load_config()
            self.assertEqual(int(cfg.retrieval.top_k), 17)
        finally:
            os.environ.pop("APP_RETRIEVAL__TOP_K", None)

    def test_yaml_fallback_parser(self):
        """内置 YAML 子集解析器必须能处理 config.yaml 的全部语法。"""
        from src.utils.config import DEFAULT_CONFIG_PATH, _fallback_yaml_load

        parsed = _fallback_yaml_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(parsed, dict)
        self.assertIn("llm", parsed)
        self.assertIn("retrieval", parsed)
        self.assertEqual(parsed["retrieval"]["top_k"], 5)
        self.assertIsInstance(parsed["server"]["cors_origins"], list)
        self.assertIsInstance(parsed["generation"]["cache"]["enabled"], bool)


# ===========================================================================
# 文本处理
# ===========================================================================
class TextTest(unittest.TestCase):
    """分词、切句、关键词、归一化、Token 估算。"""

    def test_tokenize_english(self):
        from src.utils.text import tokenize

        tokens = tokenize("The Transformer uses multi-head attention.")
        self.assertIn("transformer", tokens)
        self.assertIn("attention", tokens)
        # 停用词应被过滤
        self.assertNotIn("the", tokens)

    def test_tokenize_chinese(self):
        from src.utils.text import tokenize

        tokens = tokenize("注意力机制的计算复杂度")
        self.assertTrue(tokens)
        self.assertTrue(any("注意" in t or "注意" == t for t in tokens), tokens)

    def test_tokenize_empty(self):
        from src.utils.text import tokenize, tokenize_query

        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize_query(""), [])

    def test_split_sentences_mixed(self):
        from src.utils.text import split_sentences

        text = "第一句话。第二句话！Third sentence. 第四句？"
        sentences = split_sentences(text, min_len=2)
        self.assertGreaterEqual(len(sentences), 3)

    def test_extract_keywords(self):
        from src.utils.text import extract_keywords

        text = "注意力机制是深度学习的核心。注意力机制可以捕捉长距离依赖。" * 3
        keywords = extract_keywords(text, top_k=5)
        self.assertTrue(keywords)
        self.assertTrue(any("注意" in k for k in keywords), keywords)

    def test_normalize_text(self):
        from src.utils.text import normalize_text

        raw = "全角ＡＢＣ１２３\r\n\r\n\r\ninter-\nnational  多   空格"
        out = normalize_text(raw)
        self.assertIn("ABC123", out)
        self.assertIn("international", out)
        self.assertNotIn("\r", out)

    def test_estimate_tokens_monotonic(self):
        from src.utils.text import estimate_tokens

        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        self.assertGreater(long, short)
        self.assertEqual(estimate_tokens(""), 0)

    def test_join_with_limit(self):
        from src.utils.text import join_with_limit

        items = ["a" * 10, "b" * 10, "c" * 10]
        joined = join_with_limit(items, limit_chars=25)
        self.assertLessEqual(len(joined), 30)
        self.assertNotIn("c" * 10, joined)


# ===========================================================================
# 数据结构
# ===========================================================================
class TypesTest(unittest.TestCase):
    """数据契约的构造、序列化与派生属性。"""

    def test_chunk_id_is_stable(self):
        from src.utils.types import Chunk

        a = Chunk(doc_id="d1", text="same text", index=0)
        b = Chunk(doc_id="d1", text="same text", index=0)
        self.assertEqual(a.chunk_id, b.chunk_id)
        c = Chunk(doc_id="d1", text="same text", index=1)
        self.assertNotEqual(a.chunk_id, c.chunk_id)

    def test_retrieved_chunk_separates_score_and_confidence(self):
        """score 是相对展示分，confidence 是绝对相关度 —— 两者必须独立。"""
        from src.utils.types import Chunk, RetrievedChunk

        hit = RetrievedChunk(chunk=Chunk(doc_id="d", text="t"), score=1.0, confidence=0.01)
        self.assertEqual(hit.score, 1.0)
        self.assertEqual(hit.confidence, 0.01)

    def test_message_to_llm_dict(self):
        from src.utils.types import Message

        message = Message.tool("result", "calculator", "call_1")
        payload = message.to_llm_dict()
        self.assertEqual(payload["role"], "tool")
        self.assertEqual(payload["name"], "calculator")
        self.assertNotIn("timestamp", payload)

    def test_token_usage_addition(self):
        from src.utils.types import TokenUsage

        total = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15) + TokenUsage(
            prompt_tokens=1, completion_tokens=2, total_tokens=3
        )
        self.assertEqual(total.total_tokens, 18)
        self.assertEqual(total.prompt_tokens, 11)

    def test_agent_result_tool_success_rate(self):
        from src.utils.types import AgentResult

        result = AgentResult(tool_calls=4, tool_success=3, tool_failures=1)
        self.assertAlmostEqual(result.tool_success_rate, 0.75)
        self.assertEqual(AgentResult().tool_success_rate, 1.0)

    def test_rag_answer_top_confidence(self):
        from src.utils.types import Chunk, RAGAnswer, RetrievedChunk

        answer = RAGAnswer(
            retrieved=[RetrievedChunk(chunk=Chunk(doc_id="d", text="t"), score=1.0, confidence=0.42)]
        )
        self.assertEqual(answer.top_score, 1.0)
        self.assertAlmostEqual(answer.top_confidence, 0.42)

    def test_json_roundtrip(self):
        from src.utils.types import Chunk

        chunk = Chunk(doc_id="d1", text="内容", index=2, page=3, strategy="paper")
        restored = Chunk.from_dict(json.loads(chunk.to_json()))
        self.assertEqual(restored.doc_id, chunk.doc_id)
        self.assertEqual(restored.page, 3)
        self.assertEqual(restored.strategy, "paper")

    def test_to_jsonable_handles_enum_and_nested(self):
        from src.utils.types import Role, StepType, to_jsonable

        payload = to_jsonable({"role": Role.USER, "steps": [{"type": StepType.ACTION}]})
        self.assertEqual(payload["role"], "user")
        self.assertEqual(payload["steps"][0]["type"], "action")

    def test_loaded_document_page_of_offset(self):
        from src.utils.types import DocumentMeta, LoadedDocument

        doc = LoadedDocument(doc_id="d", text="aaa\nbbb\nccc", pages=["aaa", "bbb", "ccc"],
                             meta=DocumentMeta(doc_id="d"))
        self.assertEqual(doc.page_of_offset(0), 1)
        self.assertEqual(doc.page_of_offset(len("aaa\nbbb\n") + 1), 3)


# ===========================================================================
# 存储
# ===========================================================================
class StorageTest(unittest.TestCase):
    """原子 JSON 写入、NDJSON、SQLite 连接池、文件锁。"""

    def test_json_roundtrip_and_atomic_write(self):
        from src.utils.storage import read_json, write_json

        with TempWorkspace() as ws:
            path = ws.path("nested", "data.json")
            write_json(path, {"中文": "值", "n": 1})
            self.assertEqual(read_json(path)["中文"], "值")
            # 原子写：不应留下临时文件
            leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".tmp_")]
            self.assertEqual(leftovers, [])

    def test_read_json_missing_returns_default(self):
        from src.utils.storage import read_json

        with TempWorkspace() as ws:
            self.assertEqual(read_json(ws.path("nope.json"), default={"a": 1}), {"a": 1})

    def test_read_json_corrupt_returns_default(self):
        from src.utils.storage import read_json

        with TempWorkspace() as ws:
            path = ws.path("bad.json")
            path.write_text("{ not json", encoding="utf-8")
            self.assertIsNone(read_json(path))
            self.assertEqual(read_json(path, default=[]), [])

    def test_ndjson_append_and_tail(self):
        from src.utils.storage import append_ndjson, read_ndjson

        with TempWorkspace() as ws:
            path = ws.path("log.jsonl")
            append_ndjson(path, [{"i": i} for i in range(5)])
            append_ndjson(path, [{"i": 5}])
            self.assertEqual(len(read_ndjson(path)), 6)
            tail = read_ndjson(path, limit=2)
            self.assertEqual([r["i"] for r in tail], [4, 5])

    def test_sqlite_pool_concurrent_writes(self):
        from src.utils.storage import open_sqlite

        with TempWorkspace() as ws:
            pool = open_sqlite(ws.path("db", "test.db"))
            with pool.write() as conn:
                conn.execute("CREATE TABLE t (v INTEGER)")

            errors = []

            def writer(offset: int) -> None:
                try:
                    for i in range(20):
                        with pool.write() as conn:
                            conn.execute("INSERT INTO t (v) VALUES (?)", (offset + i,))
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=writer, args=(n * 100,)) for n in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [], f"并发写出现异常: {errors}")
            with pool.read() as conn:
                count = conn.execute("SELECT COUNT(*) AS c FROM t").fetchone()["c"]
            self.assertEqual(count, 80)

    def test_file_lock_mutual_exclusion(self):
        from src.utils.storage import FileLock

        with TempWorkspace() as ws:
            lock_path = ws.path("lock", "x.lock")
            first = FileLock(lock_path, timeout=1)
            second = FileLock(lock_path, timeout=0.2)
            self.assertTrue(first.acquire())
            try:
                # 同进程内的第二次获取应失败（超时返回 False，而不是抛异常）
                self.assertFalse(second.acquire())
            finally:
                first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_human_size(self):
        from src.utils.storage import human_size

        self.assertEqual(human_size(0), "0 B")
        self.assertIn("KB", human_size(2048))
        self.assertIn("MB", human_size(5 * 1024 * 1024))


# ===========================================================================
# 日志与可观测性
# ===========================================================================
class LoggerTest(unittest.TestCase):
    """Trace 日志、指标聚合、计时器。"""

    def test_timer(self):
        from src.utils.logger import Timer

        with Timer() as timer:
            time.sleep(0.01)
        self.assertGreaterEqual(timer.ms, 5.0)
        self.assertGreater(timer.seconds, 0)

    def test_trace_logger_writes_jsonl(self):
        from src.utils.logger import TraceLogger
        from src.utils.types import RAGAnswer

        with TempWorkspace() as ws:
            path = ws.path("traces.jsonl")
            logger = TraceLogger(path=path)
            logger.log_rag(RAGAnswer(question="Q?", answer="A!", latency_ms=1.0))
            logger.log_event("upload", filename="a.pdf")
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)
            records = json.loads(lines[0])
            self.assertEqual(records["kind"], "rag")
            self.assertEqual(records["question"], "Q?")
            self.assertEqual(logger.recent(5)[-1]["kind"], "upload")

    def test_metrics_collector(self):
        from src.utils.logger import MetricsCollector
        from src.utils.types import Chunk, RAGAnswer, RetrievedChunk

        collector = MetricsCollector(enabled=False)
        for confidence in (0.9, 0.5, 0.1):
            collector.record_rag(
                RAGAnswer(
                    question="q",
                    answer="a",
                    retrieved=[
                        RetrievedChunk(chunk=Chunk(doc_id="d", text="t"), score=1.0, confidence=confidence)
                    ],
                    latency_ms=100.0,
                )
            )
        snapshot = collector.rag_snapshot()
        self.assertEqual(snapshot["requests"], 3)
        self.assertEqual(snapshot["hit_rate"], 1.0)
        self.assertGreater(snapshot["p95_latency_ms"], 0)
        self.assertAlmostEqual(snapshot["avg_top_score"], 1.0)

    def test_metrics_tool_stats(self):
        from src.utils.logger import MetricsCollector

        collector = MetricsCollector(enabled=False)
        collector.record_tool("calculator", True, 1.0)
        collector.record_tool("calculator", False, 3.0)
        rows = collector.tool_snapshot()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["calls"], 2)
        self.assertAlmostEqual(rows[0]["success_rate"], 0.5)

    def test_metrics_reset(self):
        from src.utils.logger import MetricsCollector
        from src.utils.types import RAGAnswer

        collector = MetricsCollector(enabled=False)
        collector.record_rag(RAGAnswer(question="q", answer="a"))
        collector.reset()
        self.assertEqual(collector.rag_snapshot()["requests"], 0)

    def test_metrics_retrieval_snapshot(self):
        """裸检索（Agent 的 knowledge_search）必须单独记账。

        回归：Web 端提问全部走 Agent，Agent 的工具直连检索引擎、
        不构造 RAGAnswer，于是 rag_snapshot 恒为 0；指标面板的
        「检索命中率 / 检索 P50 / 平均 Top 分数」全是 0，看起来像面板坏了。
        """
        from src.utils.logger import MetricsCollector

        collector = MetricsCollector(enabled=False)
        collector.record_retrieval(latency_ms=10.0, top_score=0.42, hit=True)
        collector.record_retrieval(latency_ms=30.0, top_score=0.01, hit=False)

        snapshot = collector.retrieval_snapshot()
        self.assertEqual(snapshot["requests"], 2)
        self.assertAlmostEqual(snapshot["hit_rate"], 0.5)
        self.assertAlmostEqual(snapshot["avg_latency_ms"], 20.0)
        self.assertAlmostEqual(snapshot["avg_top_score"], 0.215)
        # 与 RAG 流水线口径互不污染
        self.assertEqual(collector.rag_snapshot()["requests"], 0)

        self.assertIn("retrieval", collector.full_snapshot())
        collector.reset()
        self.assertEqual(collector.retrieval_snapshot()["requests"], 0)

    def test_metrics_retrieval_empty_snapshot(self):
        from src.utils.logger import MetricsCollector

        snapshot = MetricsCollector(enabled=False).retrieval_snapshot()
        self.assertEqual(snapshot["requests"], 0)
        self.assertEqual(snapshot["hit_rate"], 0.0)
        self.assertEqual(snapshot["avg_top_score"], 0.0)

    def test_agent_snapshot(self):
        from src.utils.logger import MetricsCollector
        from src.utils.types import AgentResult, AgentStep, StepType, ToolResult

        collector = MetricsCollector(enabled=False)
        step = AgentStep(
            type=StepType.OBSERVATION,
            tool_name="knowledge_search",
            tool_result=ToolResult(name="knowledge_search", content="ok", latency_ms=12.0),
        )
        collector.record_agent(
            AgentResult(question="q", answer="a", steps=[step], iterations=2, tool_calls=1, tool_success=1)
        )
        snapshot = collector.agent_snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(snapshot["avg_iterations"], 2.0)
        self.assertEqual(snapshot["tool_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
