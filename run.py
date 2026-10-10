#!/usr/bin/env python
"""智能科研助理 —— 一键启动脚本。

用法::

    python run.py                      # 启动 Web 服务（默认 http://127.0.0.1:8501）
    python run.py --open               # 启动并自动打开浏览器
    python run.py --cli                # 命令行交互模式
    python run.py --index ./data/samples   # 批量导入论文后退出
    python run.py --health             # 查看各组件健康状态
    python run.py --list-tools         # 查看全部可用工具

查看全部参数：``python run.py --help``
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在模块搜索路径首位，使 `import src.xxx` 稳定可用
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check_python_version() -> None:
    if sys.version_info < (3, 10):
        sys.stderr.write(
            f"错误：需要 Python 3.10 或更高版本，当前为 {sys.version.split()[0]}\n"
            "提示：本项目使用了 PEP 604（`X | None`）等 3.10+ 语法。\n"
        )
        raise SystemExit(2)


def main() -> int:
    _check_python_version()
    # 必须在任何输出之前：中文 Windows 的 GBK 控制台无法编码 ░ / ✔ 等符号
    from src.utils.console import enable_utf8_output

    enable_utf8_output()
    try:
        from src.frontend.app import main as app_main
    except ImportError as exc:  # pragma: no cover
        sys.stderr.write(
            f"错误：无法导入项目模块（{exc}）。\n"
            f"请确认在项目根目录下运行本脚本：{PROJECT_ROOT}\n"
        )
        return 2
    return app_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
