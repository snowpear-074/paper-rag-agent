# -*- coding: utf-8 -*-
"""模块四（系统集成与前端）测试：容器装配、HTTP API、SSE 流式问答、
文档上传、会话管理、可观测性与静态资源服务。

运行::

    python -m unittest tests.test_frontend -v
"""

from __future__ import annotations

import base64
import json
import os
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tests.common import PROJECT_ROOT, TempWorkspace, ensure_sample_pdfs


class _ApiClient:
    """极简 HTTP 客户端（标准库），封装 JSON 与 SSE 读取。"""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def request(self, method: str, path: str, body=None, timeout: float = 60.0, headers=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, {"raw": raw}

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, body=None, **kwargs):
        return self.request("POST", path, body, **kwargs)

    def put(self, path: str, body=None, **kwargs):
        return self.request("PUT", path, body, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)

    def raw_get(self, path: str, timeout: float = 30.0):
        """取原始响应（用于静态资源）。"""
        with urllib.request.urlopen(self.base + path, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get("Content-Type")

    def sse(self, path: str, body, timeout: float = 90.0):
        """读取 SSE 流，返回 [(event, data_dict), ...]。"""
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        events = []
        with urllib.request.urlopen(request, timeout=timeout) as response:
            buffer = b""
            for raw_line in response:
                buffer += raw_line
                if raw_line.strip() != b"":
                    continue
                text = buffer.decode("utf-8", "ignore")
                buffer = b""
                name, payload = None, None
                for line in text.splitlines():
                    if line.startswith("event: "):
                        name = line[7:].strip()
                    elif line.startswith("data: "):
                        payload = line[6:]
                if not name:
                    continue
                try:
                    events.append((name, json.loads(payload) if payload else {}))
                except json.JSONDecodeError:
                    events.append((name, {"raw": payload}))
        return events


class FrontendTestCase(unittest.TestCase):
    """公共前置：装配容器、建立索引、启动 HTTP 服务。"""

    @classmethod
    def setUpClass(cls) -> None:
        from src.frontend import SystemContainer
        from src.frontend.server import ApiServer

        cls._ws = TempWorkspace("frontend")
        cls.cfg = cls._ws.config(
            chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50},
            server={"host": "127.0.0.1", "port": 0, "max_upload_mb": 20},
            observability={"level": "ERROR", "trace_file": str(cls._ws.path("logs", "t.jsonl")),
                           "metrics_file": str(cls._ws.path("logs", "m.jsonl")),
                           "log_dir": str(cls._ws.dir("logs"))},
        )
        cls.container = SystemContainer(cls.cfg)
        cls.container.initialize()
        cls.container.index_files([str(p) for p in ensure_sample_pdfs()])

        cls.server = ApiServer(cls.container, host="127.0.0.1", port=0,
                               static_dir=PROJECT_ROOT / "src" / "frontend" / "static")
        cls.server.build()
        cls.server.start_background()
        host, port = cls.server.address
        cls.client = _ApiClient(f"http://{host}:{port}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()
        cls._ws.cleanup()


# ===========================================================================
# 容器装配
# ===========================================================================
class ContainerTest(unittest.TestCase):
    """系统容器：组件齐全、装配顺序、问答接口。"""

    @classmethod
    def setUpClass(cls) -> None:
        from src.frontend import SystemContainer

        cls._ws = TempWorkspace("container")
        cls.cfg = cls._ws.config(
            chunking={"strategy": "paper", "chunk_size": 400, "chunk_overlap": 50}
        )
        cls.container = SystemContainer(cls.cfg)
        cls.health = cls.container.initialize()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ws.cleanup()

    def test_all_components_assembled(self):
        for attribute in ("loader", "embedder", "vector_store", "retrieval", "llm",
                          "cache", "rag", "memory", "router", "registry", "agent"):
            with self.subTest(component=attribute):
                self.assertIsNotNone(getattr(self.container, attribute), f"{attribute} 未装配")

    def test_single_shared_dependencies(self):
        """组件间必须共享同一实例，否则会出现"索引写在一个库、检索查另一个库"。"""
        self.assertIs(self.container.retrieval.embedder, self.container.embedder)
        self.assertIs(self.container.retrieval.vector_store, self.container.vector_store)
        self.assertIs(self.container.agent.registry, self.container.registry)
        self.assertIs(self.container.agent.memory, self.container.memory)
        self.assertIs(self.container.rag.retrieval, self.container.retrieval)

    def test_health_payload(self):
        self.assertTrue(self.health["ok"])
        self.assertIn("llm", self.health)
        self.assertIn("embedding", self.health)
        self.assertIn("tools", self.health)
        self.assertEqual(self.health["init_errors"], "")

    def test_index_and_ask(self):
        report = self.container.index_files([str(p) for p in ensure_sample_pdfs()])
        self.assertGreaterEqual(report["num_documents"], 3)
        self.assertGreater(report["num_chunks"], 5)

        result = self.container.ask("What datasets were used to train the Transformer?", session_id="c1")
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "agent")
        self.assertTrue(result["answer"].strip())
        self.assertGreater(len(result["trajectory"]), 0)

    def test_ask_rag_mode(self):
        result = self.container.ask("What is attention?", use_agent=False)
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "rag")
        self.assertIn("retrieved", result)

    def test_ask_empty_question(self):
        result = self.container.ask("   ")
        self.assertFalse(result["ok"])
        self.assertIn("不能为空", result["error"])

    def test_session_lifecycle(self):
        session = self.container.create_session("容器测试会话")
        session_id = session["session_id"]
        self.container.ask("What is BERT?", session_id=session_id)

        detail = self.container.get_session(session_id)
        self.assertEqual(detail["num_turns"], 1)
        self.assertTrue(detail["turns"][0]["answer"])

        sessions = self.container.list_sessions()
        self.assertTrue(any(s["session_id"] == session_id for s in sessions))

        self.assertTrue(self.container.rename_session(session_id, "改名"))
        self.assertEqual(self.container.get_session(session_id)["title"], "改名")
        self.assertTrue(self.container.delete_session(session_id))
        self.assertIsNone(self.container.get_session(session_id))

    def test_stream_events_include_trajectory(self):
        events = list(self.container.ask_stream("What is multi-head attention?", session_id="cs1"))
        types = [e.type for e in events]
        self.assertIn("start", types)
        self.assertIn("thought", types)
        self.assertIn("action", types)
        self.assertIn("token", types)
        self.assertIn("final", types)
        self.assertIn("done", types)
        self.assertEqual(types[-1], "done")

    def test_stream_rag_mode(self):
        events = list(self.container.ask_stream("What is attention?", use_agent=False))
        types = [e.type for e in events]
        self.assertIn("retrieval", types)
        self.assertIn("token", types)
        self.assertIn("final", types)

    def test_index_bytes_and_delete(self):
        sample = ensure_sample_pdfs()[0].read_bytes()
        report = self.container.index_bytes("uploaded_sample.pdf", sample)
        self.assertGreaterEqual(report["num_documents"], 1)
        documents = self.container.documents()
        self.assertTrue(documents)
        doc_id = documents[0]["doc_id"]
        self.assertGreater(self.container.delete_document(doc_id), 0)

    def test_safe_filename_strips_path_traversal(self):
        from src.frontend.container import _safe_filename

        self.assertEqual(_safe_filename("../../etc/passwd"), "passwd")
        self.assertEqual(_safe_filename("a/b\\c.txt"), "c.txt")
        self.assertEqual(_safe_filename('bad<>:"|?*.txt'), "bad_______.txt")
        self.assertTrue(_safe_filename(""))

    def test_metrics_and_stats(self):
        self.container.ask("What is ResNet?", session_id="cm1")
        metrics = self.container.metrics()
        self.assertIn("rag", metrics)
        self.assertIn("agent", metrics)
        self.assertGreaterEqual(metrics["agent"]["requests"], 1)

        stats = self.container.stats()
        self.assertIn("config", stats)
        self.assertIn("retrieval", stats)
        self.assertIn("tools", stats)

    def test_tools_listing(self):
        tools = self.container.tools()
        self.assertEqual(len(tools), 8)
        names = {t["name"] for t in tools}
        self.assertIn("knowledge_search", names)
        web = next(t for t in tools if t["name"] == "web_search")
        self.assertFalse(web["enabled"])


# ===========================================================================
# HTTP API
# ===========================================================================
class ApiEndpointTest(FrontendTestCase):
    """REST 接口：状态码、响应结构、错误处理。"""

    def test_health(self):
        status, body = self.client.get("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertIn("llm", body)
        self.assertGreaterEqual(body["num_documents"], 3)

    def test_config_endpoint(self):
        status, body = self.client.get("/api/config")
        self.assertEqual(status, 200)
        config = body["config"]
        for section in ("llm", "embedding", "retrieval", "chunking", "agent"):
            self.assertIn(section, config)

    def test_stats_and_metrics_endpoints(self):
        self.assertEqual(self.client.get("/api/stats")[0], 200)
        status, body = self.client.get("/api/metrics")
        self.assertEqual(status, 200)
        self.assertIn("rag", body)
        self.assertIn("tools", body)

    def test_tools_endpoint(self):
        status, body = self.client.get("/api/tools")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["tools"]), 8)
        self.assertIn("parameters", body["tools"][0])

    def test_documents_listing(self):
        status, body = self.client.get("/api/documents")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(body["summary"]["num_documents"], 3)
        self.assertGreater(body["summary"]["num_chunks"], 0)

    def test_traces_endpoint(self):
        self.client.post("/api/chat", {"question": "What is attention?", "stream": False})
        status, body = self.client.get("/api/traces?n=5")
        self.assertEqual(status, 200)
        self.assertIsInstance(body["traces"], list)

    def test_session_crud_via_api(self):
        status, created = self.client.post("/api/sessions", {"title": "接口会话"})
        self.assertEqual(status, 201)
        session_id = created["session"]["session_id"]

        status, fetched = self.client.get(f"/api/sessions/{session_id}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched["session"]["num_turns"], 0)

        status, renamed = self.client.put(f"/api/sessions/{session_id}", {"title": "新名字"})
        self.assertEqual(status, 200)
        self.assertTrue(renamed["ok"])

        status, deleted = self.client.delete(f"/api/sessions/{session_id}")
        self.assertEqual(status, 200)
        self.assertTrue(deleted["ok"])

    def test_missing_session_returns_404(self):
        status, body = self.client.get("/api/sessions/does_not_exist")
        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])

    def test_chat_non_stream(self):
        status, body = self.client.post(
            "/api/chat",
            {"question": "Attention Is All You Need 的作者是谁？", "stream": False},
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertTrue(body["answer"].strip())
        self.assertIn("iterations", body)

    def test_chat_empty_question_returns_400(self):
        status, body = self.client.post("/api/chat", {"question": "", "stream": False})
        self.assertEqual(status, 400)
        self.assertFalse(body["ok"])

    def test_search_endpoint_three_modes(self):
        for mode in ("vector", "hybrid", "full"):
            with self.subTest(mode=mode):
                status, body = self.client.post(
                    "/api/search", {"query": "residual learning degradation", "mode": mode, "top_k": 3}
                )
                self.assertEqual(status, 200)
                self.assertTrue(body["ok"])
                self.assertEqual(body["num"], 3)
                for item in body["results"]:
                    self.assertIn("confidence", item)
                    self.assertIn("source", item)

    def test_retrieve_endpoint(self):
        status, body = self.client.post(
            "/api/retrieve", {"question": "What is attention?", "stream": False}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["mode"], "rag")

    def test_unknown_endpoint_404(self):
        status, body = self.client.get("/api/definitely_not_here")
        self.assertEqual(status, 404)
        self.assertFalse(body["ok"])

    def test_wrong_method_405(self):
        status, body = self.client.delete("/api/health")
        self.assertEqual(status, 405)
        self.assertIn("不被支持", body["error"])

    def test_invalid_json_returns_400(self):
        request = urllib.request.Request(
            self.client.base + "/api/chat",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=20)
        self.assertEqual(ctx.exception.code, 400)

    def test_cors_headers_present(self):
        """默认配置（空白名单）下不发送 CORS 头 —— 同源请求本就不需要。"""
        with urllib.request.urlopen(self.client.base + "/api/health", timeout=20) as response:
            self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_cors_origin_not_echoed_when_not_allowlisted(self):
        """未列入白名单的来源不得拿到 ACAO，否则任意网页都能读响应。"""
        request = urllib.request.Request(
            self.client.base + "/api/health",
            headers={"Origin": "https://evil.example"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))

    def test_options_preflight(self):
        request = urllib.request.Request(
            self.client.base + "/api/chat", method="OPTIONS",
            headers={"Origin": "http://localhost"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            self.assertEqual(response.status, 204)

    def test_document_delete_via_api(self):
        # 先上传一份再删除，避免影响其它用例的索引
        sample = ensure_sample_pdfs()[2].read_bytes()
        events = self.client.sse(
            "/api/documents/upload",
            {"filename": "to_delete.pdf", "content_base64": base64.b64encode(sample).decode()},
        )
        self.assertTrue(any(name == "indexed" for name, _ in events))
        documents = self.client.get("/api/documents")[1]["documents"]
        target = next(d for d in documents if "BERT" in str(d.get("title", "")))
        status, deleted = self.client.delete(f"/api/documents/{target['doc_id']}")
        self.assertEqual(status, 200)
        self.assertGreater(deleted["removed_chunks"], 0)

    def test_delete_unknown_document_returns_404(self):
        """删除不存在的文档必须如实报错，不能返回 ok=True。"""
        status, payload = self.client.delete("/api/documents/nonexistent-doc-id")
        self.assertEqual(status, 404)
        self.assertFalse(payload.get("ok"))


# ===========================================================================
# 安全：跨站防护与本机文件访问边界
# ===========================================================================
class SecurityTest(FrontendTestCase):
    """回归测试：CORS 通配、CSRF、按本地路径建索引的越权。"""

    def test_cross_site_post_is_rejected(self):
        """带外站 Origin 的状态变更请求必须 403（CSRF 防护）。"""
        status, payload = self.client.post(
            "/api/sessions", {"title": "csrf"}, headers={"Origin": "https://evil.example"}
        )
        self.assertEqual(status, 403)
        self.assertFalse(payload.get("ok"))

    def test_cross_site_delete_is_rejected(self):
        status, _ = self.client.delete(
            "/api/sessions/whatever", headers={"Origin": "https://evil.example"}
        )
        self.assertEqual(status, 403)

    def test_same_origin_post_is_allowed(self):
        """浏览器同源请求（Origin == Host）必须放行，否则前端会自己把自己拦死。"""
        host = self.client.base.split("//", 1)[1]
        status, payload = self.client.post(
            "/api/sessions", {"title": "same-origin"}, headers={"Origin": f"http://{host}"}
        )
        self.assertEqual(status, 201)
        self.assertTrue(payload.get("ok"))

    def test_documents_add_rejects_path_outside_roots(self):
        """按本地路径导入的接口不得读取允许目录之外的文件。"""
        outside = "C:/Windows/win.ini" if os.name == "nt" else "/etc/passwd"
        status, payload = self.client.post("/api/documents", {"paths": [outside]})
        self.assertEqual(status, 400)
        self.assertIn("允许导入", str(payload.get("error", "")))

    def test_documents_add_allows_configured_root(self):
        """落配置根目录内的路径仍然可用（批量导入依赖这条路径）。"""
        upload_dir = self.cfg.resolve_path(str(self.cfg.paths.upload_dir))
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / "imported_sample.pdf"
        target.write_bytes(ensure_sample_pdfs()[0].read_bytes())
        status, payload = self.client.post("/api/documents", {"paths": [str(target)]})
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"))


# ===========================================================================
# SSE 流式
# ===========================================================================
class SseStreamTest(FrontendTestCase):
    """SSE 协议正确性与事件内容。"""

    def test_chat_sse_event_sequence(self):
        events = self.client.sse(
            "/api/chat", {"question": "What is masked language model?", "stream": True}
        )
        names = [name for name, _ in events]
        self.assertEqual(names[0], "start")
        self.assertEqual(names[-1], "done")
        self.assertIn("thought", names)
        self.assertIn("action", names)
        self.assertIn("token", names)
        self.assertIn("final", names)

    def test_sse_final_payload_complete(self):
        events = self.client.sse(
            "/api/chat", {"question": "How does ResNet work?", "stream": True}
        )
        final = next(data for name, data in events if name == "final")
        for key in ("answer", "citations", "iterations", "tool_calls", "terminated_reason",
                    "usage", "latency_ms", "trajectory"):
            self.assertIn(key, final, f"final 事件缺少字段 {key}")
        self.assertTrue(final["answer"].strip())

    def test_sse_tokens_concatenate_to_answer(self):
        events = self.client.sse("/api/chat", {"question": "What is attention?", "stream": True})
        tokens = "".join(data.get("text", "") for name, data in events if name == "token")
        final = next(data for name, data in events if name == "final")
        self.assertEqual(tokens, final["answer"])

    def test_sse_retrieval_event_has_confidence(self):
        events = self.client.sse(
            "/api/chat", {"question": "What datasets for Transformer training?", "stream": True}
        )
        retrieval = [data for name, data in events if name == "retrieval"]
        self.assertTrue(retrieval, "应推送检索结果事件")
        for item in retrieval[0]["results"]:
            self.assertIn("confidence", item)
            self.assertIn("preview", item)

    def test_sse_content_type(self):
        request = urllib.request.Request(
            self.client.base + "/api/chat",
            data=json.dumps({"question": "hi", "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))

    def test_upload_sse_progress(self):
        sample = ensure_sample_pdfs()[1].read_bytes()
        events = self.client.sse(
            "/api/documents/upload",
            {"filename": "sse_upload.pdf", "content_base64": base64.b64encode(sample).decode()},
        )
        names = [name for name, _ in events]
        self.assertIn("progress", names)
        self.assertIn("indexed", names)
        self.assertEqual(names[-1], "done")
        indexed = next(data for name, data in events if name == "indexed")
        self.assertGreater(indexed["num_chunks"], 0)

    def test_upload_missing_fields_returns_error(self):
        status, body = self.client.post("/api/documents/upload", {"filename": "x.pdf"})
        self.assertEqual(status, 400)
        self.assertFalse(body["ok"])

    def test_upload_bad_base64_returns_error(self):
        status, body = self.client.post(
            "/api/documents/upload", {"filename": "x.pdf", "content_base64": "!!!not base64!!!"}
        )
        self.assertEqual(status, 400)


# ===========================================================================
# 静态资源
# ===========================================================================
class StaticAssetTest(FrontendTestCase):
    """前端文件服务：MIME、内容、回退。"""

    def test_index_served_at_root(self):
        status, content, content_type = self.client.raw_get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"<!DOCTYPE html>", content[:200] or b"<!DOCTYPE html>")
        self.assertGreater(len(content), 5000)

    def test_static_assets_have_correct_mime(self):
        cases = [
            ("/css/style.css", "text/css"),
            ("/js/app.js", "application/javascript"),
            ("/js/markdown.js", "application/javascript"),
        ]
        for path, expected in cases:
            with self.subTest(path=path):
                status, content, content_type = self.client.raw_get(path)
                self.assertEqual(status, 200)
                self.assertIn(expected, content_type)
                self.assertGreater(len(content), 500)

    def test_spa_fallback_for_unknown_path(self):
        status, content, content_type = self.client.raw_get("/some/deep/route")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)

    def test_path_traversal_blocked(self):
        for path in ("/../config.yaml", "/..%2fconfig.yaml", "/css/../../config.yaml"):
            with self.subTest(path=path):
                try:
                    status, content, _ = self.client.raw_get(path)
                except urllib.error.HTTPError as exc:
                    self.assertIn(exc.code, (400, 403, 404))
                    continue
                # SPA 回退返回的是 index.html，绝不能泄漏 config.yaml
                self.assertNotIn(b"vector_store", content)
                self.assertNotIn(b"llm:", content)


# ===========================================================================
# 命令行入口
# ===========================================================================
class CliEntryTest(unittest.TestCase):
    """命令行参数解析与无服务模式。"""

    def test_parser_has_expected_options(self):
        from src.frontend.app import build_parser

        parser = build_parser()
        options = {action.dest for action in parser._actions}
        for expected in ("host", "port", "config", "cli", "open", "list_tools", "health", "index"):
            self.assertIn(expected, options)

    def test_health_mode(self):
        from src.frontend.app import main

        with TempWorkspace() as ws:
            config_path = ws.path("config.yaml")
            config_path.write_text(
                "llm:\n  provider: mock\nembedding:\n  provider: hash\n"
                f"vector_store:\n  backend: numpy\n  persist_dir: {ws.dir('vs')}\n"
                f"paths:\n  index_dir: {ws.dir('idx')}\n  upload_dir: {ws.dir('up')}\n"
                f"observability:\n  level: ERROR\n  log_dir: {ws.dir('logs')}\n",
                encoding="utf-8",
            )
            exit_code = main(["--config", str(config_path), "--health", "--json"])
            self.assertIn(exit_code, (0, 1))

    def test_find_free_port_skips_busy_port(self):
        import socket

        from src.frontend.server import find_free_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            busy = sock.getsockname()[1]
            found = find_free_port("127.0.0.1", busy, attempts=20)
            self.assertNotEqual(found, busy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
