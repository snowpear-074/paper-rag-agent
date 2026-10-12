#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抓取**真实运行系统**的界面截图（供交付文档 / 课程报告使用）。

设计原则
--------
截图必须是"真的跑出来的"，不做任何拼接或美化：

1. 启动（或复用）本地 Web 服务；
2. 用 Chrome / Edge 的无头模式打开 ``http://127.0.0.1:<port>``；
3. 通过 **DevTools Protocol（CDP）** 在页面里真实地输入问题、点击发送；
4. 等答案流式输出结束（打字机光标消失、指标条出现）后截图。

依赖
----
- Python 3.10+
- ``websocket-client``：仅本脚本需要（``pip install websocket-client``），
  **不属于系统运行依赖** —— 系统本身仍然零第三方依赖即可运行。
- 本机安装 Chrome 或 Edge（脚本会自动探测常见安装路径）。

用法::

    python run.py --index ./data/samples          # 可选：先把样例论文入库
    python scripts/capture_screenshots.py         # 默认输出 docs/images/screenshots/

常用参数::

    --port 8501            # 复用已在运行的服务端口
    --base-url URL         # 直接指定服务地址（跳过端口探测）
    --out DIR              # 输出目录
    --no-server            # 不启动服务，仅连接已在运行的实例
    --provider mock        # 截图时使用的 LLM 后端（默认沿用配置）
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIEWPORT = {"width": 1680, "height": 1050, "deviceScaleFactor": 1.5}

_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def find_browser(explicit: str = "") -> str:
    if explicit:
        return explicit
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in _CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "未找到 Chrome / Edge。请用 --browser 指定可执行文件路径，"
        "或先安装后重试。"
    )


def port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def http_json(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_http(url: str, timeout: float = 60.0, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            http_json(url)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------
class Server:
    """本地 Web 服务（复用已在运行的实例，或自己拉起一个）。"""

    def __init__(self, port: int, provider: str = "", log_path: Optional[Path] = None) -> None:
        self.port = port
        self.provider = provider
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.reused = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, allow_spawn: bool = True, timeout: float = 90.0) -> None:
        if port_open("127.0.0.1", self.port):
            print(f"[server] 复用已在运行的实例：{self.base_url}")
            self.reused = True
            return
        if not allow_spawn:
            raise SystemExit(f"端口 {self.port} 上没有服务，且指定了 --no-server。")

        env = dict(os.environ)
        env["APP_SERVER__PORT"] = str(self.port)
        env["APP_SERVER__HOST"] = "127.0.0.1"
        # 截图脚本自己就是用无头浏览器，别让服务再去弹一个桌面浏览器
        env["APP_NO_BROWSER"] = "1"
        if self.provider:
            env["APP_LLM__PROVIDER"] = self.provider
        stdout = open(self.log_path, "w", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        print(f"[server] 启动服务：python run.py --port {self.port}"
              + (f"（LLM provider={self.provider}）" if self.provider else ""))
        self.proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "run.py"), "--port", str(self.port)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
        )
        if not wait_http(f"{self.base_url}/api/health", timeout=timeout):
            self.stop()
            raise SystemExit(
                f"服务在 {timeout:.0f}s 内未就绪，请检查日志：{self.log_path}"
            )
        print(f"[server] 已就绪：{self.base_url}")

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


# ---------------------------------------------------------------------------
# CDP 客户端
# ---------------------------------------------------------------------------
class DevTools:
    """极简 Chrome DevTools Protocol 客户端（WebSocket + JSON-RPC）。"""

    def __init__(self, ws_url: str) -> None:
        import websocket  # 延迟导入：只有真正跑截图时才需要

        self._ws = websocket.create_connection(ws_url, timeout=30)
        self._id = 0

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._id += 1
        message_id = self._id
        self._ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            raw = self._ws.recv()
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("id") == message_id:
                if "error" in data:
                    raise RuntimeError(f"{method} 失败: {data['error']}")
                return data.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float = 60.0, interval: float = 0.4) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.evaluate(expression):
                    return True
            except Exception:  # noqa: BLE001 - 页面 reload 期间取值为空属正常
                pass
            time.sleep(interval)
        return False

    def screenshot(self, path: Path) -> Path:
        data = self.call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": False, "fromSurface": True},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data["data"]))
        print(f"[shot] {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size // 1024} KB)")
        return path

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass


def launch_browser(binary: str, port: int, user_data_dir: Path) -> subprocess.Popen:
    args = [
        binary,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        f"--remote-debugging-port={port}",
        # 新版 Chrome 默认校验 WebSocket 的 Origin 头，不带这个开关会握手 403
        "--remote-allow-origins=*",
        f"--user-data-dir={user_data_dir}",
        f"--window-size={VIEWPORT['width']},{VIEWPORT['height']}",
        "about:blank",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_http(f"http://127.0.0.1:{port}/json/version", timeout=40):
        proc.kill()
        raise SystemExit("浏览器调试端口未就绪（可能被杀软拦截或被已有实例占用）。")
    return proc


def connect_page(port: int) -> DevTools:
    version = http_json(f"http://127.0.0.1:{port}/json/version")
    print(f"[browser] {version.get('Browser', 'unknown')}")
    targets = http_json(f"http://127.0.0.1:{port}/json/list")
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        raise SystemExit("没有可用的页面目标。")
    return DevTools(pages[0]["webSocketDebuggerUrl"])


# ---------------------------------------------------------------------------
# 页面操作
# ---------------------------------------------------------------------------
JS_DOC_COUNT = "document.querySelectorAll('#doc-list .doc-item').length"
JS_BUSY = "document.querySelector('#btn-stop') && !document.querySelector('#btn-stop').hidden"


def ask(devtools: DevTools, question: str, timeout: float = 120.0) -> bool:
    """在输入框里真实输入问题并点击发送，等答案生成完毕。"""
    print(f"[ask ] {question}")
    devtools.evaluate(
        """
        (() => {
          const box = document.querySelector('#message-input');
          box.value = %s;
          box.dispatchEvent(new Event('input', { bubbles: true }));
          return true;
        })()
        """
        % json.dumps(question, ensure_ascii=False)
    )
    time.sleep(0.2)
    devtools.evaluate("document.querySelector('#btn-send').click(); true")

    if not devtools.wait_for(JS_BUSY, timeout=20):
        print("[warn] 未检测到「生成中」状态（可能是缓存命中，瞬间完成）")
    finished = devtools.wait_for(
        "(() => {"
        "  const nodes = document.querySelectorAll('#messages .msg.assistant');"
        "  if (!nodes.length) return false;"
        "  const last = nodes[nodes.length - 1];"
        "  const typing = last.querySelector('.typing');"
        "  const chips = last.querySelectorAll('.msg-meta .meta-chip').length;"
        "  return !typing && chips > 0;"
        "})()",
        timeout=timeout,
    )
    time.sleep(0.8)  # 让最后一段动画 / 引用卡片收起
    return finished


def open_modal(devtools: DevTools, button: str, modal: str, timeout: float = 25.0) -> bool:
    devtools.evaluate(f"document.querySelector('{button}').click(); true")
    ok = devtools.wait_for(f"!document.querySelector('{modal}').hidden", timeout=timeout)
    time.sleep(1.2)  # 等异步指标/工具数据渲染
    return ok


def close_modal(devtools: DevTools) -> None:
    devtools.evaluate(
        "document.querySelectorAll('.modal-mask').forEach(m => { m.hidden = true; }); true"
    )
    time.sleep(0.4)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def capture(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    shots: List[Path] = []

    server = Server(port=args.port, provider=args.provider,
                    log_path=Path(tempfile.gettempdir()) / "rag_agent_screenshot_server.log")
    server.start(allow_spawn=not args.no_server)

    user_data = Path(tempfile.mkdtemp(prefix="rag_shot_"))
    browser: Optional[subprocess.Popen] = None
    devtools: Optional[DevTools] = None

    try:
        browser = launch_browser(find_browser(args.browser), args.devtools_port, user_data)
        devtools = connect_page(args.devtools_port)
        devtools.call("Page.enable")
        devtools.call("Runtime.enable")
        devtools.call("Emulation.setDeviceMetricsOverride", dict(VIEWPORT, mobile=False))
        # 无头 Chrome 默认上报 prefers-color-scheme: dark，而前端默认跟随系统主题。
        # 报告里的截图统一用浅色（打印友好），深色主题由 --dark 单独抓一张。
        devtools.call("Emulation.setEmulatedMedia", {
            "features": [{"name": "prefers-color-scheme", "value": "light"}]
        })
        base = args.base_url or server.base_url
        devtools.call("Page.navigate", {"url": base})
        devtools.wait_for("document.readyState === 'complete'", timeout=40)
        devtools.wait_for("document.querySelector('#health-label') && "
                          "document.querySelector('#health-label').textContent !== '连接中…'",
                          timeout=40)
        # 等文档面板**渲染完成**，而不是死等"至少有一篇文档"：
        # 空知识库也要能出图（截图里展示空状态引导与降级提示）。
        # 只等"有渲染结果"—— 有文档项、或明确写出空/加载失败。
        devtools.wait_for(
            "(() => { const b = document.querySelector('#doc-list');"
            "  return !!b && (b.children.length > 0 || /为空|加载失败/.test(b.textContent)); })()",
            timeout=45,
        )
        time.sleep(1.5)

        # 前端把加载失败渲染成一个提示段落，而不是抛错 —— 这里显式断言，
        # 否则会安静地截出一张"知识库空白"的图（就是 Docs.load 调用错方法的症状）。
        broken = devtools.evaluate(
            "(() => { const b = document.querySelector('#doc-list');"
            "  return b && /加载失败/.test(b.textContent) ? b.textContent.trim() : ''; })()"
        )
        if broken:
            raise SystemExit(f"知识库列表渲染失败：{broken}")

        docs = devtools.evaluate(JS_DOC_COUNT) or 0
        print(f"[page] 知识库已加载 {docs} 篇文档")
        if docs == 0:
            print("[warn] 知识库为空：截图里文档列表为空，问答会走「检索无结果」的降级提示。"
                  "想抓带引用/轨迹的图，先执行：python run.py --index ./data/samples")

        # 开一个新会话：否则页面会自动打开"最近一次会话"，
        # 截图里会出现与本次演示无关的历史消息。
        # 但当前会话本来就是空的（首次启动 / 刚清过库）时不用再建一个 ——
        # 页面加载已经自动建好了，再点一次会让会话列表并排出现两个"新会话"。
        existing_msgs = devtools.evaluate("document.querySelectorAll('#messages .msg').length") or 0
        if existing_msgs:
            devtools.evaluate(
                "(() => { const b = document.querySelector('#btn-new-session');"
                "  if (b) { b.click(); return true; } return false; })()"
            )
            time.sleep(1.0)

        # 建会话会弹 Toast，正好盖在右侧轨迹面板上。等它自己淡出再截图 ——
        # 只等时间，不去动 DOM（脚本的底线是"截图必须是真跑出来的"）。
        devtools.wait_for("!document.querySelector('#toast-stack .toast')", timeout=12)
        time.sleep(0.5)

        # ① 主界面：知识库 + 空态对话区
        shots.append(devtools.screenshot(out_dir / "01_主界面_知识库与对话.png"))

        # ② 一次真实问答：散点问题 → 触发检索 + 引用
        if ask(devtools, args.question):
            shots.append(devtools.screenshot(out_dir / "02_问答_引用与Agent轨迹.png"))
        else:
            print("[warn] 第一个问题未在超时内完成")

        # ③ 多轮追问：验证上下文与记忆
        if ask(devtools, "它和 ResNet 有什么不同？"):
            devtools.evaluate(
                "document.querySelector('#messages').scrollTop = "
                "document.querySelector('#messages').scrollHeight; true"
            )
            time.sleep(0.6)
            shots.append(devtools.screenshot(out_dir / "03_多轮追问_上下文记忆.png"))

        # ④ 已注册工具清单
        if open_modal(devtools, "#btn-tools", "#tools-modal"):
            shots.append(devtools.screenshot(out_dir / "04_工具清单.png"))
        close_modal(devtools)

        # ⑤ 运行指标面板
        if open_modal(devtools, "#btn-metrics", "#metrics-modal"):
            shots.append(devtools.screenshot(out_dir / "05_运行指标面板.png"))
        close_modal(devtools)

        # ⑥ 深色主题（可选，用于展示可切换主题）
        if args.dark:
            devtools.evaluate("document.querySelector('#btn-theme').click(); true")
            time.sleep(1.0)
            shots.append(devtools.screenshot(out_dir / "06_深色主题.png"))
    finally:
        if devtools is not None:
            devtools.close()
        if browser is not None:
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill()
        shutil.rmtree(user_data, ignore_errors=True)
        server.stop()

    print(f"\n完成：{len(shots)} 张截图 → {out_dir}")
    for path in shots:
        print(f"  - {path.name}")
    return 0 if shots else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="抓取真实运行系统的界面截图（无头浏览器 + DevTools Protocol）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=8501, help="服务端口（默认 8501）")
    parser.add_argument("--base-url", default="", help="直接指定服务地址，跳过端口探测")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "images" / "screenshots"),
                        help="截图输出目录")
    parser.add_argument("--browser", default="", help="Chrome/Edge 可执行文件路径")
    parser.add_argument("--devtools-port", type=int, default=9333, help="浏览器调试端口")
    parser.add_argument("--provider", default="", help="截图时覆盖 LLM provider（如 mock）")
    parser.add_argument("--no-server", action="store_true", help="不启动服务，仅连接现有实例")
    parser.add_argument("--dark", action="store_true", help="额外抓一张深色主题截图")
    parser.add_argument("--question", default="Transformer 的核心创新是什么？",
                        help="用于截图的问题")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
