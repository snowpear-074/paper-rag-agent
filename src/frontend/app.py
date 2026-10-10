"""应用入口（模块四）。

三种启动方式
------------
1. **推荐**：``python run.py``        —— 零依赖 Web 服务 + 浏览器界面
2. **命令行**：``python -m src.frontend.app --cli``  —— 交互式问答
3. **编程**：``from src.frontend import get_container``

命令行参数
----------
``--host`` / ``--port``   绑定地址与端口（端口占用时自动顺延）
``--config``              指定配置文件路径
``--cli``                 启动命令行交互模式
``--open``                启动后自动打开浏览器
``--list-tools``          列出全部可用工具后退出
``--health``              打印健康检查结果后退出
``--index PATH``          导入文档/目录并建立索引后退出
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许以 `python src/frontend/app.py` 直接运行时也能找到包
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..utils.config import load_config
from ..utils.console import enable_utf8_output
from ..utils.logger import get_logger
from .container import SystemContainer, get_container
from .server import ApiServer, find_free_port

__all__ = ["main", "run_cli", "run_server", "build_parser"]

_log = get_logger("frontend.app")


# ==============================================================================
# 命令行参数
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="智能科研助理",
        description="基于 RAG + Agent 的论文知识库问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python run.py                          # 启动 Web 界面\n"
            "  python run.py --port 8600 --open       # 指定端口并自动打开浏览器\n"
            "  python run.py --cli                    # 命令行交互模式\n"
            "  python run.py --index ./data/samples   # 批量导入论文后退出\n"
            "  python run.py --health                 # 查看各组件健康状态\n"
        ),
    )
    parser.add_argument("--host", default=None, help="监听地址（默认取 config.yaml 的 server.host）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（默认 8501，被占用时自动顺延）")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument("--cli", action="store_true", help="启动命令行交互模式")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--list-tools", action="store_true", help="列出全部工具后退出")
    parser.add_argument("--health", action="store_true", help="打印健康检查结果后退出")
    parser.add_argument("--index", default=None, help="导入指定文件或目录后退出")
    parser.add_argument("--strategy", default=None,
                        help="分块策略：fixed / recursive / semantic / paper")
    parser.add_argument("--log-level", default=None, help="日志级别：DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（配合 --health）")
    return parser


# ==============================================================================
# 各模式实现
# ==============================================================================
def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    config: Any = None,
    open_browser: bool = False,
) -> None:
    """启动 Web 服务。"""
    cfg = config or load_config()
    bind_host = host or str(cfg.server.host)
    requested_port = int(port or cfg.server.port)

    container = get_container(cfg)
    health = container.initialize()
    _print_banner(health, cfg)

    # 端口占用时自动顺延，避免"启动即失败"
    actual_port = find_free_port(bind_host, requested_port)
    if actual_port != requested_port:
        _log.warning("端口 %d 已被占用，自动改用 %d", requested_port, actual_port)

    api = ApiServer(container, host=bind_host, port=actual_port)
    if open_browser:
        import threading
        import webbrowser

        display = "127.0.0.1" if bind_host in ("0.0.0.0", "::") else bind_host
        url = f"http://{display}:{actual_port}"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    api.serve_forever()


def run_cli(config: Any = None, strategy: Optional[str] = None) -> None:
    """命令行交互模式。"""
    cfg = config or load_config()
    container = get_container(cfg)
    health = container.initialize()
    _print_banner(health, cfg)

    session = container.create_session("命令行会话")
    session_id = session["session_id"]
    _print_cli_help()

    while True:
        try:
            question = input("\n\033[1;36m你 >\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question in ("/exit", "/quit", "exit", "quit"):
            print("再见！")
            break
        if question == "/help":
            _print_cli_help()
            continue
        if question == "/docs":
            docs = container.documents()
            if not docs:
                print("  知识库为空，请先用 /import <路径> 导入论文")
            for idx, doc in enumerate(docs, start=1):
                print(f"  {idx}. [{doc.get('doc_id')}] {doc.get('title')} "
                      f"({doc.get('year') or '?'}) · {doc.get('num_chunks')} 块")
            continue
        if question.startswith("/import "):
            path = question[len("/import "):].strip()
            print(f"  正在导入 {path} …")
            report = container.index_files([path], strategy=strategy,
                                           on_progress=lambda i: _cli_progress(i))
            print(f"\n  完成：{report.get('num_documents')} 篇文档，{report.get('num_chunks')} 个片段")
            continue
        if question == "/clear":
            container.clear_knowledge_base()
            print("  知识库已清空")
            continue

        started = time.perf_counter()
        result = container.ask(question, session_id=session_id)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"\n\033[1;32m助理 >\033[0m {result.get('answer', '')}")
        _print_cli_meta(result, elapsed)


def _print_cli_help() -> None:
    print(
        "\n\033[1m可用命令\033[0m\n"
        "  /import <路径>   导入论文文件或目录\n"
        "  /docs            查看知识库文档列表\n"
        "  /clear           清空知识库\n"
        "  /help            显示本帮助\n"
        "  /exit            退出\n"
    )


def _cli_progress(info: Dict[str, Any]) -> None:
    percent = float(info.get("percent", 0.0))
    message = str(info.get("message", ""))
    bar_width = 28
    filled = int(bar_width * percent / 100.0)
    bar = "█" * filled + "░" * (bar_width - filled)
    sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  {message[:52]:<52}")
    sys.stdout.flush()


def _print_cli_meta(result: Dict[str, Any], elapsed_ms: float) -> None:
    """打印答案的元信息与引用。"""
    citations = result.get("citations") or []
    if citations:
        print("\n\033[1m引用来源\033[0m")
        for citation in citations[:6]:
            print(
                f"  [{citation.get('index')}] {citation.get('source')} "
                f"p.{citation.get('page')}  (相关度 {citation.get('score', 0):.2f})"
            )
    trajectory = result.get("trajectory") or []
    if trajectory:
        print("\n\033[1m推理轨迹\033[0m")
        for step in trajectory:
            kind = step.get("type")
            if kind == "thought":
                print(f"  \033[36m[思考]\033[0m {str(step.get('content'))[:100]}")
            elif kind == "action":
                print(f"  \033[35m[行动]\033[0m {step.get('tool')}({json.dumps(step.get('args'), ensure_ascii=False)[:70]})")
            elif kind == "observation":
                print(f"  \033[32m[观察]\033[0m {str(step.get('content'))[:100]}")
    print(
        f"\n  \033[90m耗时 {elapsed_ms:.0f}ms · 轮次 {result.get('iterations', 0)} · "
        f"工具 {result.get('tool_calls', 0)} 次 · "
        f"Token {((result.get('usage') or {}).get('total_tokens', 0))} · "
        f"终止原因 {result.get('terminated_reason', '-')}\033[0m"
    )


def _print_banner(health: Dict[str, Any], config: Any) -> None:
    """启动横幅：把关键配置与健康状态一次打出来，便于排障。"""
    llm = health.get("llm") or {}
    embedding = health.get("embedding") or {}
    ok_mark = "\033[32m✔\033[0m"
    warn_mark = "\033[33m!\033[0m"

    llm_status = ok_mark if llm.get("ok") else f"{warn_mark} ({llm.get('error', '不可用')[:40]})"
    print("\n" + "═" * 70)
    print("  智能科研助理 —— 基于 RAG + Agent 的论文知识库问答系统  v1.0.0")
    print("═" * 70)
    print(f"  LLM 服务    : {llm.get('provider')}/{llm.get('model')} {llm_status}")
    print(f"  Embedding   : {embedding.get('provider')}/{embedding.get('model')} "
          f"(dim={embedding.get('dimension')})")
    print(f"  向量库      : {(health.get('vector_store') or {}).get('backend', '-')} · "
          f"{(health.get('vector_store') or {}).get('num_vectors', 0)} 向量")
    print(f"  知识库      : {health.get('num_documents', 0)} 篇文档 / {health.get('num_chunks', 0)} 个片段")
    print(f"  对话记忆    : {(health.get('memory') or {}).get('backend', '-')} · "
          f"{(health.get('memory') or {}).get('sessions', 0)} 个会话")
    print(f"  语义缓存    : {(health.get('cache') or {}).get('backend', '-')}")
    if health.get("init_errors"):
        print(f"  \033[33m降级组件    : {health['init_errors'][:100]}\033[0m")
    print("═" * 70)


# ==============================================================================
# 主入口
# ==============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    # 横幅与工具列表会打印 ░ / ✔ 等非 GBK 字符，先让输出流能承载它们
    enable_utf8_output()
    args = build_parser().parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.log_level:
        overrides.setdefault("observability", {})["level"] = args.log_level.upper()
    config = load_config(args.config, overrides or None)

    if args.list_tools:
        container = get_container(config)
        container.initialize()
        print("\n可用工具：\n")
        for tool in container.tools():
            flag = "✔" if tool.get("enabled") else "✘（依赖未就绪）"
            print(f"  {flag} \033[1m{tool['name']}\033[0m")
            print(f"      {tool['description'].splitlines()[0]}")
            params = (tool.get("parameters") or {}).get("properties") or {}
            if params:
                print(f"      参数: {', '.join(params.keys())}")
            print()
        return 0

    if args.health:
        container = get_container(config)
        health = container.initialize()
        if args.json:
            print(json.dumps(health, ensure_ascii=False, indent=2, default=str))
        else:
            _print_banner(health, config)
            print(f"  整体状态    : {'正常' if health.get('ok') else '异常'}")
            print(f"  运行时长    : {health.get('uptime_seconds', 0):.1f}s\n")
        return 0 if health.get("ok") else 1

    if args.index:
        container = get_container(config)
        container.initialize()
        print(f"正在导入：{args.index}")
        report = container.index_files([args.index], strategy=args.strategy, on_progress=_cli_progress)
        print()
        if report.get("failed"):
            print(f"\n\033[33m失败 {len(report['failed'])} 个文件：\033[0m")
            for failure in report["failed"][:10]:
                print(f"  - {Path(str(failure.get('path'))).name}: {failure.get('error')}")
        print(
            f"\n\033[32m导入完成\033[0m：{report.get('num_documents', 0)} 篇文档，"
            f"{report.get('num_chunks', 0)} 个片段"
        )
        return 0 if report.get("num_documents") else 1

    if args.cli:
        run_cli(config, strategy=args.strategy)
        return 0

    run_server(args.host, args.port, config, open_browser=args.open)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
