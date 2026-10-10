"""后端 HTTP 服务（模块四 5.4）—— 零依赖实现。

为什么用 ``http.server`` 而不是 FastAPI
---------------------------------------
课程要求"能在普通笔记本上一键跑起来"。FastAPI + uvicorn 需要额外安装
近 20 个包，在无网络教室环境里是部署门槛。本项目用标准库
:mod:`http.server` 的 :class:`ThreadingHTTPServer` 实现同等能力：

- REST API（JSON）
- **SSE 流式推送**（``text/event-stream``，逐事件 flush，前端打字机效果）
- 静态资源服务（含正确 MIME 与缓存头）
- 健康检查 / 指标 / 会话 / 文档管理

并发模型：每个请求一个线程（ThreadingHTTPServer）。系统组件内部都有锁保护，
SQLite 走 WAL 模式，足以支撑教学场景的并发量。

路由表见 :data:`ROUTES`；所有响应统一 JSON 结构
``{"ok": bool, ...}``，错误返回 ``{"ok": false, "error": "..."}`` + 合适的状态码。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import socket
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from ..utils.logger import get_logger
from .container import SystemContainer, get_container

__all__ = ["ApiServer", "serve", "ROUTES"]

_log = get_logger("frontend.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 路由表：(方法, 正则, 处理器名)
ROUTES: List[Tuple[str, str, str]] = [
    ("GET", r"^/api/health$", "handle_health"),
    ("GET", r"^/api/config$", "handle_config"),
    ("GET", r"^/api/stats$", "handle_stats"),
    ("GET", r"^/api/metrics$", "handle_metrics"),
    ("GET", r"^/api/tools$", "handle_tools"),
    ("GET", r"^/api/traces$", "handle_traces"),
    ("GET", r"^/api/documents$", "handle_documents_list"),
    ("POST", r"^/api/documents/upload$", "handle_documents_upload"),
    ("POST", r"^/api/documents/reindex$", "handle_documents_reindex"),
    ("DELETE", r"^/api/documents/(?P<doc_id>[^/]+)$", "handle_documents_delete"),
    ("POST", r"^/api/documents$", "handle_documents_add"),
    ("GET", r"^/api/sessions$", "handle_sessions_list"),
    ("POST", r"^/api/sessions$", "handle_sessions_create"),
    ("GET", r"^/api/sessions/(?P<session_id>[^/]+)$", "handle_session_get"),
    ("PUT", r"^/api/sessions/(?P<session_id>[^/]+)$", "handle_session_rename"),
    ("DELETE", r"^/api/sessions/(?P<session_id>[^/]+)$", "handle_session_delete"),
    ("POST", r"^/api/feedback$", "handle_feedback"),
    ("POST", r"^/api/chat$", "handle_chat"),
    ("POST", r"^/api/search$", "handle_search"),
    ("POST", r"^/api/retrieve$", "handle_retrieve"),
]


class ApiServer:
    """HTTP 服务封装。"""

    def __init__(
        self,
        container: Optional[SystemContainer] = None,
        host: str = "127.0.0.1",
        port: int = 8501,
        static_dir: Optional[Path] = None,
    ) -> None:
        self.container = container or get_container()
        self.host = host
        self.port = port
        self.static_dir = Path(static_dir) if static_dir else STATIC_DIR
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._compiled: List[Tuple[str, Pattern[str], str]] = [
            (method, re.compile(pattern), handler) for method, pattern, handler in ROUTES
        ]

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def build(self) -> ThreadingHTTPServer:
        """创建（但不启动）HTTP 服务器。"""
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        return self._httpd

    def start_background(self) -> "ApiServer":
        """在后台线程启动服务（测试与嵌入式使用）。"""
        if self._httpd is None:
            self.build()
        assert self._httpd is not None
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="api-server", daemon=True)
        self._thread.start()
        _log.info("HTTP 服务已在后台启动: http://%s:%d", self.host, self.port)
        return self

    def serve_forever(self) -> None:
        """前台阻塞运行（生产入口）。"""
        if self._httpd is None:
            self.build()
        assert self._httpd is not None
        actual_host, actual_port = self._httpd.server_address[:2]
        _log.info("=" * 68)
        _log.info("智能科研助理 已启动")
        _log.info("  界面地址: http://%s:%d", _display_host(actual_host), actual_port)
        _log.info("  健康检查: http://%s:%d/api/health", _display_host(actual_host), actual_port)
        _log.info("  按 Ctrl+C 停止服务")
        _log.info("=" * 68)
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            _log.info("收到中断信号，正在关闭…")
        finally:
            self.stop()

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
            self._httpd = None
        _log.info("HTTP 服务已停止")

    @property
    def address(self) -> Tuple[str, int]:
        if self._httpd is not None:
            host, port = self._httpd.server_address[:2]
            return str(host), int(port)
        return self.host, self.port

    # ------------------------------------------------------------------
    # 路由分发
    # ------------------------------------------------------------------
    def dispatch(self, method: str, path: str) -> Optional[str]:
        """返回匹配到的处理器方法名；未匹配返回 None。"""
        for route_method, pattern, handler in self._compiled:
            if route_method != method.upper():
                continue
            if pattern.match(path):
                return handler
        return None

    def match(self, method: str, path: str) -> Tuple[Optional[str], Dict[str, str]]:
        """返回 (处理器名, 路径参数)。"""
        for route_method, pattern, handler in self._compiled:
            if route_method != method.upper():
                continue
            found = pattern.match(path)
            if found:
                return handler, {k: urllib.parse.unquote(v) for k, v in found.groupdict().items()}
        return None, {}

    def allowed_methods(self, path: str) -> List[str]:
        return [m for m, pattern, _h in self._compiled if pattern.match(path)]

    # ------------------------------------------------------------------
    # 安全策略
    # ------------------------------------------------------------------
    def cors_origins(self) -> List[str]:
        """CORS 白名单（默认空 = 仅同源，不发送任何跨源头）。"""
        raw = self.container.config.get("server.cors_origins", []) or []
        if isinstance(raw, str):
            return [raw]
        return [str(item) for item in raw]

    def origin_allowed(self, origin: str) -> bool:
        """判断请求来源是否可信（空 Origin = 非浏览器客户端，放行）。

        默认白名单为**空**。早期默认值是 ``["*"]``，那意味着任意网页都能在用户
        浏览器里读写本机知识库 —— ``/api/documents/add`` 可按本地路径建索引、
        ``/api/chat`` 能把知识库内容读出来。要跨源访问必须显式配置白名单。
        """
        if not origin:
            return True
        origins = self.cors_origins()
        return "*" in origins or origin in origins

    def import_roots(self) -> List[Path]:
        """允许通过 HTTP 接口按本地路径导入文档的根目录白名单。

        约束的是 **HTTP 接口**；命令行（``run.py --index <任意路径>``）不受限制，
        因为那本来就是本机用户自己的显式操作。
        """
        config = self.container.config
        roots: List[Path] = []
        for key in ("upload_dir", "data_dir"):
            raw = config.paths.get(key)
            if not raw:
                continue
            try:
                roots.append(config.resolve_path(str(raw)).resolve())
            except Exception:  # noqa: BLE001 - 配置异常不应使接口不可用
                continue
        return roots

    def path_importable(self, raw: str) -> bool:
        """本地路径是否落在允许导入的目录内。"""
        if bool(self.container.config.get("server.allow_external_paths", False)):
            return True
        target = Path(str(raw)).expanduser()
        try:
            resolved = target.resolve()
        except OSError:
            return False
        for root in self.import_roots():
            if resolved == root or root in resolved.parents:
                return True
        return False


def _display_host(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "::") else host


def _secrets_status() -> Dict[str, Any]:
    """凭据状态（**只返回掩码与指纹，绝不含明文**）。"""
    try:
        from ..utils.secrets import get_vault

        info = get_vault().describe()
        return {
            "vault_exists": info["vault_exists"],
            "items": [
                {
                    "name": row["name"],
                    "configured": row["configured"],
                    "source": row["source"],
                    "masked": row["masked"],
                    "fingerprint": row["fingerprint"],
                }
                for row in info["secrets"]
                if row["configured"]
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:120]}


def _web_search_status(container: Any) -> Dict[str, Any]:
    """联网搜索工具状态。

    只报告**配置**与**可用引擎清单**，不发起真实搜索（避免健康检查变慢）。
    各引擎的真实可达性由 :meth:`WebSearchTool.probe` 按需触发：
    ``GET /api/health?probe_web=1``（网页源）或 ``?probe_web=academic``（学术源）。
    """
    try:
        tool = container.registry.get("web_search") if container.registry else None
        if tool is None:
            return {"available": False, "reason": "工具未注册"}
        return {
            "available": bool(getattr(tool, "enabled", False)),
            "provider": getattr(tool, "provider", ""),
            "web_engines": list(getattr(tool, "_WEB_ENGINES", ())),
            "academic_engines": list(getattr(tool, "_ACADEMIC_ENGINES", ())),
            "max_results": getattr(tool, "max_results", 0),
            "timeout": getattr(tool, "timeout", 0),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)[:120]}


# ==============================================================================
# 请求处理器
# ==============================================================================
def _make_handler(server: ApiServer) -> type:
    """构造绑定到具体 ApiServer 实例的请求处理器类。"""

    class _Handler(BaseHTTPRequestHandler):
        server_version = "ResearchAssistant/1.0"
        protocol_version = "HTTP/1.1"

        # -- 日志：交给统一 logger，避免 stderr 噪声 --------------------
        def log_message(self, fmt: str, *args: Any) -> None:
            _log.debug("%s - %s", self.client_address[0] if self.client_address else "-", fmt % args)

        # -- 通用工具 ---------------------------------------------------
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_error_json(self, message: str, status: int = 400, detail: str = "") -> None:
            self._send_json({"ok": False, "error": message, "detail": detail}, status=status)

        def _cors(self) -> None:
            """按白名单回显跨源头。

            只有**显式配置**了 ``server.cors_origins`` 才发送 CORS 头；默认
            （空列表）同源请求无需 CORS，跨源请求不发 ACAO —— 浏览器随即拒绝
            跨源读取响应。
            """
            origin = self.headers.get("Origin")
            if not origin or not server.origin_allowed(origin):
                return
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _same_origin(self, origin: str) -> bool:
            """Origin 是否等于本服务的 Host（同源）。"""
            host = self.headers.get("Host") or ""
            if not host:
                return False
            origin = origin.rstrip("/")
            return origin in (f"http://{host}", f"https://{host}")

        def _cross_site_blocked(self) -> bool:
            """跨站状态变更请求拦截（CSRF 防护）。

            浏览器对跨源请求**一定**会带 ``Origin`` 头，因此校验它就能阻断
            "用户访问了恶意网页 → 该网页借用户浏览器删本机知识库/按本地路径建索引"。
            非浏览器客户端（curl / 测试 / CLI）不带 Origin，放行。
            """
            origin = self.headers.get("Origin")
            if not origin:
                return False
            return not (server.origin_allowed(origin) or self._same_origin(origin))

        def _read_body(self, limit_mb: Optional[float] = None) -> bytes:
            limit = int((limit_mb or float(server.container.config.get("server.max_upload_mb", 100))) * 1024 * 1024)
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                # 非法 Content-Length：必须关连接，否则残留字节会被当成下一个请求解析
                self.close_connection = True
                raise ValueError("Content-Length 不是合法整数")
            if length > limit:
                # 超限时**不读取**请求体，若继续复用这条 keep-alive 连接，
                # 未读的 body 会被解析成下一个请求（HTTP 请求走私式错位）
                self.close_connection = True
                raise ValueError(f"请求体过大（{length / 1024 / 1024:.1f}MB），上限 {limit / 1024 / 1024:.0f}MB")
            if length <= 0:
                return b""
            return self.rfile.read(length)

        def _read_json(self) -> Dict[str, Any]:
            raw = self._read_body()
            if not raw:
                return {}
            try:
                data = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return data

        # -- SSE --------------------------------------------------------
        def _begin_sse(self) -> None:
            """开启 SSE 响应。

            这里显式设置 ``close_connection = True``，原因很关键：
            SSE 是**长度未知**的流式响应，而本服务用的是 HTTP/1.1 keep-alive，
            若不声明关闭，客户端无法判断流何时结束（读操作会一直阻塞到超时）。
            关闭连接后，服务端在处理器返回时关闭 socket，客户端读到 EOF 即正常收尾。
            """
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.end_headers()

        def _sse_write(self, payload: str) -> bool:
            """写一个 SSE 报文；客户端断开时返回 False。"""
            try:
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return False

        def _sse_error(self, message: str, detail: str = "") -> None:
            from ..generation.streaming import EventType, StreamEvent

            self._sse_write(
                StreamEvent(EventType.ERROR, {"message": message, "detail": detail}).to_sse()
            )

        # -- 静态文件 ---------------------------------------------------
        def _serve_static(self, path: str) -> None:
            relative = path.lstrip("/") or "index.html"
            # 阻止路径穿越
            candidate = (server.static_dir / relative).resolve()
            try:
                candidate.relative_to(server.static_dir.resolve())
            except ValueError:
                self._send_error_json("非法的资源路径", status=403)
                return
            if candidate.is_dir():
                candidate = candidate / "index.html"
            if not candidate.exists():
                # SPA 回退：未知路径统一返回 index.html
                index = server.static_dir / "index.html"
                if index.exists() and not path.startswith("/api/"):
                    candidate = index
                else:
                    self._send_error_json(f"资源不存在: {path}", status=404)
                    return

            content_type, _encoding = mimetypes.guess_type(str(candidate))
            if candidate.suffix == ".js":
                content_type = "application/javascript"
            content_type = content_type or "application/octet-stream"
            if content_type.startswith("text/") or content_type in (
                "application/javascript",
                "application/json",
            ):
                content_type += "; charset=utf-8"

            try:
                body = candidate.read_bytes()
            except OSError as exc:
                self._send_error_json(f"读取资源失败: {exc}", status=500)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        # -- HTTP 方法 --------------------------------------------------
        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle("HEAD")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def do_PUT(self) -> None:  # noqa: N802
            self._handle("PUT")

        def do_DELETE(self) -> None:  # noqa: N802
            self._handle("DELETE")

        def _handle(self, method: str) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)
            query = urllib.parse.parse_qs(parsed.query)

            # 跨站防护：状态变更请求必须来自同源或 CORS 白名单
            if method in ("POST", "PUT", "DELETE", "PATCH") and self._cross_site_blocked():
                self._send_error_json("跨站请求被拒绝（Origin 不在白名单）", status=403)
                return

            # API 路由（HEAD 按 GET 处理）
            lookup = "GET" if method == "HEAD" else method
            handler_name, params = server.match(lookup, path)
            if handler_name is None:
                if path.startswith("/api/"):
                    allowed = server.allowed_methods(path)
                    if allowed:
                        self._send_error_json(
                            f"方法 {method} 不被支持，可用: {', '.join(allowed)}",
                            status=405,
                        )
                    else:
                        self._send_error_json(f"接口不存在: {path}", status=404)
                    return
                self._serve_static(path)
                return

            handler: Callable[..., Any] = getattr(self, handler_name, None)
            if handler is None:
                self._send_error_json(f"处理器未实现: {handler_name}", status=500)
                return

            try:
                handler(params=params, query=query)
            except ValueError as exc:
                self._send_error_json(str(exc), status=400)
            except FileNotFoundError as exc:
                self._send_error_json(f"文件不存在: {exc}", status=404)
            except Exception as exc:  # noqa: BLE001
                _log.error("请求处理失败 %s %s: %s\n%s", method, path, exc, traceback.format_exc(limit=4))
                try:
                    self._send_error_json(
                        "服务器内部错误", status=500, detail=f"{type(exc).__name__}: {exc}"
                    )
                except Exception:  # noqa: BLE001
                    pass

        # ==============================================================
        # 各接口实现
        # ==============================================================
        def handle_health(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            payload = self._health_payload()
            # ?probe_web=1 探测网页引擎；?probe_web=academic 探测学术源
            # （会发起真实网络请求，默认关闭以免健康检查变慢）
            probe_scope = str((query.get("probe_web") or ["0"])[0]).lower()
            if probe_scope in ("1", "true", "yes", "web", "academic"):
                scope = "academic" if probe_scope == "academic" else "web"
                payload["web_search"]["probe"] = self._probe_web_search(scope)
            self._send_json(payload)

        def _probe_web_search(self, scope: str = "web") -> Dict[str, Any]:
            try:
                tool = server.container.registry.get("web_search") if server.container.registry else None
                if tool is None or not hasattr(tool, "probe"):
                    return {"ok": False, "reason": "工具未注册或不支持探测"}
                return tool.probe(scope=scope)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)[:150]}

        def _health_payload(self) -> Dict[str, Any]:
            container = server.container
            health = container.health()
            llm = health.get("llm") or {}
            embedding = health.get("embedding") or {}
            tools = health.get("tools") or {}
            return {
                "ok": bool(health.get("ok")),
                "version": health.get("version", "1.0.0"),
                "uptime_seconds": health.get("uptime_seconds", 0.0),
                "num_documents": health.get("num_documents", 0),
                "num_chunks": health.get("num_chunks", 0),
                "init_errors": health.get("init_errors", ""),
                "llm": {
                    "provider": llm.get("provider", "unknown"),
                    "model": llm.get("model", ""),
                    "ok": bool(llm.get("ok", False)),
                    "error": llm.get("error", ""),
                },
                "embedding": {
                    "provider": embedding.get("provider", "unknown"),
                    "model": embedding.get("model", ""),
                    "dimension": embedding.get("dimension", 0),
                    "ok": bool(embedding.get("ok", False)),
                },
                "vector_store": health.get("vector_store") or {},
                "web_search": _web_search_status(container),
                "memory": {
                    "backend": (health.get("memory") or {}).get("backend", ""),
                    "sessions": (health.get("memory") or {}).get("sessions", 0),
                },
                "cache": health.get("cache") or {},
                "secrets": _secrets_status(),
                "tools": {
                    "num": tools.get("num_tools", 0),
                    "enabled": tools.get("enabled", 0),
                    "names": tools.get("tools", []),
                },
            }

        def handle_config(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            container = server.container
            cfg = container.config
            payload = {
                "ok": True,
                "config": {
                    "llm": {
                        "provider": str(cfg.llm.provider),
                        "model": str(cfg.llm.model),
                        "temperature": float(cfg.llm.temperature),
                    },
                    "embedding": {
                        "provider": str(cfg.embedding.provider),
                        "model": str(cfg.embedding.model),
                        "dimension": int(cfg.embedding.dimension),
                    },
                    "vector_store": {
                        "backend": str(cfg.vector_store.backend),
                        "collection": str(cfg.vector_store.collection),
                    },
                    "retrieval": {
                        "top_k": int(cfg.retrieval.top_k),
                        "candidate_k": int(cfg.retrieval.candidate_k),
                        "enable_bm25": bool(cfg.retrieval.enable_bm25),
                        "enable_rerank": bool(cfg.retrieval.enable_rerank),
                        "rrf_k": int(cfg.retrieval.rrf_k),
                        "low_relevance_threshold": float(cfg.retrieval.low_relevance_threshold),
                    },
                    "chunking": {
                        "strategy": str(cfg.chunking.strategy),
                        "chunk_size": int(cfg.chunking.chunk_size),
                        "chunk_overlap": int(cfg.chunking.chunk_overlap),
                    },
                    "agent": {
                        "max_iterations": int(cfg.agent.max_iterations),
                        "enable_router": bool(cfg.agent.enable_router),
                        "enable_parallel": bool(cfg.agent.enable_parallel),
                        "tool_timeout": float(cfg.agent.tool_timeout),
                    },
                    "generation": {
                        "max_context_chars": int(cfg.generation.max_context_chars),
                        "cache_enabled": bool(cfg.get("generation.cache.enabled", True)),
                    },
                },
            }
            self._send_json(payload)

        def handle_stats(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            self._send_json({"ok": True, "stats": server.container.stats()})

        def handle_metrics(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            metrics = server.container.metrics()
            metrics["ok"] = True
            self._send_json(metrics)

        def handle_tools(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            self._send_json({"ok": True, "tools": server.container.tools()})

        def handle_traces(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            try:
                n = int((query.get("n") or ["20"])[0])
            except (TypeError, ValueError):
                n = 20
            kind = (query.get("kind") or [None])[0]
            self._send_json({"ok": True, "traces": server.container.traces(n=n, kind=kind)})

        def handle_documents_list(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            container = server.container
            documents = container.documents()
            retrieval = container.retrieval
            self._send_json(
                {
                    "ok": True,
                    "documents": documents,
                    "summary": {
                        "num_documents": len(documents),
                        "num_chunks": retrieval.num_chunks if retrieval else 0,
                        "num_vectors": retrieval.vector_store.count() if retrieval else 0,
                    },
                }
            )

        def handle_documents_upload(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            """上传文档并建立索引（SSE 进度流）。"""
            body = self._read_json()
            filename = str(body.get("filename") or "").strip()
            content_b64 = body.get("content_base64") or ""
            strategy = body.get("strategy") or None
            if not filename:
                raise ValueError("缺少 filename")
            if not content_b64:
                raise ValueError("缺少 content_base64")
            try:
                content = base64.b64decode(content_b64, validate=False)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"content_base64 解码失败: {exc}") from exc
            if not content:
                raise ValueError("上传内容为空")

            self._begin_sse()
            from ..generation.streaming import EventType, StreamEvent

            def progress(info: Dict[str, Any]) -> None:
                nonlocal alive
                if not alive:
                    return
                alive = self._sse_write(
                    StreamEvent(
                        EventType.PROGRESS,
                        {
                            "stage": info.get("stage", ""),
                            "percent": info.get("percent", 0.0),
                            "message": info.get("message", ""),
                            "filename": filename,
                            "current": info.get("current"),
                            "total": info.get("total"),
                        },
                    ).to_sse()
                )

            alive = True
            try:
                report = server.container.index_bytes(
                    filename, content, strategy=strategy, on_progress=progress
                )
                if not alive:
                    return
                if report.get("num_documents"):
                    self._sse_write(
                        StreamEvent(
                            EventType.INDEXED,
                            {
                                "filename": filename,
                                "num_documents": report.get("num_documents", 0),
                                "num_chunks": report.get("num_chunks", 0),
                                "documents": report.get("reports", []),
                            },
                        ).to_sse()
                    )
                else:
                    self._sse_error(
                        "文档未能建立索引",
                        "; ".join(str(f.get("error", "")) for f in report.get("failed", []))[:400],
                    )
                if report.get("failed"):
                    self._sse_write(
                        StreamEvent(
                            EventType.WARNING if hasattr(EventType, "WARNING") else EventType.ERROR,
                            {"message": "部分文件失败", "failed": report["failed"]},
                        ).to_sse()
                    )
            except Exception as exc:  # noqa: BLE001
                _log.exception("上传索引失败")
                self._sse_error("上传或索引失败", f"{type(exc).__name__}: {exc}")
            finally:
                if alive:
                    self._sse_write(StreamEvent(EventType.DONE, {"filename": filename}).to_sse())

        def handle_documents_add(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            """按服务器本地路径导入文档（便于 CLI 与批量测试）。

            ⚠️ 这是能读**服务器本地文件**的接口，因此路径必须落在允许的根目录内
            （``paths.upload_dir`` / ``paths.data_dir``）。否则任意网页都能借用户
            浏览器把本机文件建进知识库再读出来。确需导入外部路径时，显式设置
            ``server.allow_external_paths: true``。
            """
            body = self._read_json()
            paths = body.get("paths") or []
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                raise ValueError("缺少 paths（文件或目录路径列表）")
            outside = [str(p) for p in paths if not server.path_importable(str(p))]
            if outside:
                roots = ", ".join(str(r) for r in server.import_roots()) or "（无）"
                raise ValueError(
                    f"路径不在允许导入的目录内: {', '.join(outside[:3])}。"
                    f"允许的根目录: {roots}。如确需导入外部路径，"
                    f"请在 config.yaml 设置 server.allow_external_paths: true"
                )
            report = server.container.index_files(
                paths, strategy=body.get("strategy") or None
            )
            self._send_json({"ok": bool(report.get("num_documents")), **report})

        def handle_documents_reindex(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            """重建索引。

            ``doc_id`` 为空时对全部文档重建；否则只重建指定文档。
            """
            body = self._read_json() if (self.headers.get("Content-Length") or "0") != "0" else {}
            doc_id = str(body.get("doc_id") or "").strip()
            strategy = body.get("strategy") or None
            container = server.container
            if container.retrieval is None:
                raise ValueError("检索引擎未初始化")

            upload_dir = container.config.resolve_path(str(container.config.paths.upload_dir))
            targets: List[Dict[str, Any]] = container.documents()
            if doc_id:
                targets = [d for d in targets if str(d.get("doc_id")) == doc_id]
                if not targets:
                    raise FileNotFoundError(f"未找到 doc_id={doc_id} 的文档")

            rebuilt = 0
            chunks = 0
            failed: List[Dict[str, str]] = []
            from ..data_loader import BatchImporter

            importer = BatchImporter(container.loader)
            for document in targets:
                source = str(document.get("source_path") or "")
                candidates = [Path(source)] if source and Path(source).exists() else []
                if not candidates:
                    candidates = importer.collect_files(upload_dir)
                    stem = Path(source).stem if source else ""
                    if stem:
                        candidates = [p for p in candidates if p.stem == stem] or candidates
                if not candidates:
                    failed.append({"doc_id": str(document.get("doc_id")), "error": "找不到原始文件"})
                    continue
                try:
                    job, loaded = importer.import_documents([candidates[0]])
                    if not loaded:
                        failed.append(
                            {
                                "doc_id": str(document.get("doc_id")),
                                "error": job.errors[0] if job.errors else "加载失败",
                            }
                        )
                        continue
                    report = container.retrieval.index_document(loaded[0], strategy=strategy)
                    rebuilt += 1
                    chunks += int(report.get("num_chunks", 0))
                except Exception as exc:  # noqa: BLE001
                    failed.append({"doc_id": str(document.get("doc_id")), "error": str(exc)[:200]})

            self._send_json(
                {
                    "ok": rebuilt > 0 or not targets,
                    "num_rebuilt": rebuilt,
                    "num_chunks": chunks,
                    "failed": failed,
                }
            )

        def handle_documents_delete(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            doc_id = params.get("doc_id", "")
            if not doc_id:
                raise ValueError("缺少 doc_id")
            removed = server.container.delete_document(doc_id)
            if removed <= 0:
                # 没有命中任何文档：如实返回失败，别让前端以为删除成功了
                self._send_error_json(f"未找到 doc_id={doc_id} 对应的文档", status=404)
                return
            self._send_json({"ok": True, "deleted": doc_id, "removed_chunks": removed})

        def handle_sessions_list(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            try:
                limit = int((query.get("limit") or ["50"])[0])
            except (TypeError, ValueError):
                limit = 50
            self._send_json({"ok": True, "sessions": server.container.list_sessions(limit=limit)})

        def handle_sessions_create(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            body = self._read_json() if (self.headers.get("Content-Length") or "0") != "0" else {}
            session = server.container.create_session(title=str(body.get("title") or ""))
            self._send_json({"ok": True, "session": session}, status=201)

        def handle_session_get(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            session = server.container.get_session(params.get("session_id", ""))
            if session is None:
                self._send_error_json("会话不存在", status=404)
                return
            self._send_json({"ok": True, "session": session})

        def handle_session_rename(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            body = self._read_json()
            title = str(body.get("title") or "").strip()
            if not title:
                raise ValueError("缺少 title")
            ok = server.container.rename_session(params.get("session_id", ""), title)
            self._send_json({"ok": ok, "session_id": params.get("session_id", ""), "title": title})

        def handle_session_delete(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            session_id = params.get("session_id", "")
            ok = server.container.delete_session(session_id)
            self._send_json({"ok": ok, "deleted": session_id})

        def handle_feedback(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            body = self._read_json()
            turn_id = str(body.get("turn_id") or "").strip()
            feedback = str(body.get("feedback") or "").strip().lower()
            if not turn_id:
                raise ValueError("缺少 turn_id")
            if feedback not in ("up", "down", ""):
                raise ValueError("feedback 只能为 'up' 或 'down'")
            ok = server.container.set_feedback(turn_id, feedback)
            self._send_json({"ok": ok, "turn_id": turn_id, "feedback": feedback})

        # -- 核心：对话 ------------------------------------------------
        def handle_chat(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            """对话接口。``stream=true`` 返回 SSE，否则返回一次性 JSON。"""
            body = self._read_json()
            question = str(body.get("question") or "").strip()
            session_id = str(body.get("session_id") or "").strip()
            use_agent = bool(body.get("use_agent", True))
            stream = bool(body.get("stream", True))
            if not question:
                raise ValueError("question 不能为空")

            if not stream:
                result = server.container.ask(question, session_id=session_id, use_agent=use_agent)
                self._send_json(result)
                return

            self._begin_sse()
            alive = True
            try:
                for event in server.container.ask_stream(
                    question, session_id=session_id, use_agent=use_agent
                ):
                    if not self._sse_write(event.to_sse()):
                        alive = False
                        _log.info("客户端已断开，停止推送事件")
                        break
            except Exception as exc:  # noqa: BLE001
                _log.exception("流式问答失败")
                if alive:
                    self._sse_error("流式问答失败", f"{type(exc).__name__}: {exc}")

        def handle_search(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            """纯检索接口（不生成答案），用于调试与检索评测。"""
            body = self._read_json()
            text = str(body.get("query") or "").strip()
            if not text:
                raise ValueError("query 不能为空")
            top_k = int(body.get("top_k") or server.container.config.retrieval.top_k)
            mode = str(body.get("mode") or "full")
            engine = server.container.retrieval
            if engine is None:
                raise ValueError("检索引擎未初始化")
            self._send_json({"ok": True, **engine.retrieve_with_context(text, top_k=top_k, mode=mode)})

        def handle_retrieve(self, params: Dict[str, str], query: Dict[str, List[str]]) -> None:
            """RAG 问答接口（绕过 Agent，直接走检索+生成流水线）。"""
            body = self._read_json()
            question = str(body.get("question") or "").strip()
            if not question:
                raise ValueError("question 不能为空")
            if bool(body.get("stream", False)):
                self._begin_sse()
                alive = True
                try:
                    for event in server.container.ask_stream(
                        question, session_id=str(body.get("session_id") or ""), use_agent=False
                    ):
                        if not self._sse_write(event.to_sse()):
                            alive = False
                            break
                except Exception as exc:  # noqa: BLE001
                    if alive:
                        self._sse_error("RAG 问答失败", str(exc))
                return
            result = server.container.ask(question, use_agent=False)
            self._send_json(result)

    return _Handler


# ==============================================================================
# 入口
# ==============================================================================
def serve(
    host: Optional[str] = None,
    port: Optional[int] = None,
    config: Any = None,
    open_browser: bool = False,
) -> None:
    """启动 HTTP 服务（阻塞）。"""
    from ..utils.config import get_config

    cfg = config or get_config()
    bind_host = host or str(cfg.server.host)
    bind_port = int(port or cfg.server.port)

    container = get_container(cfg)
    container.initialize()

    api = ApiServer(container, host=bind_host, port=bind_port)
    if open_browser:
        threading.Timer(
            1.2,
            lambda: _open_browser(bind_host, bind_port),
        ).start()
    api.serve_forever()


def _open_browser(host: str, port: int) -> None:
    import webbrowser

    url = f"http://{_display_host(host)}:{port}"
    try:
        webbrowser.open(url)
        _log.info("已在浏览器中打开 %s", url)
    except Exception as exc:  # noqa: BLE001
        _log.debug("自动打开浏览器失败: %s", exc)


def find_free_port(host: str = "127.0.0.1", start: int = 8501, attempts: int = 20) -> int:
    """从 start 开始找一个可用端口（避免"端口被占用"直接启动失败）。"""
    for offset in range(attempts):
        candidate = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    return start
