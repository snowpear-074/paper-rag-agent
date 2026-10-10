"""控制台编码适配（中文 Windows 的 GBK 环境）。

问题
----
项目大量使用 ``░ █ ✔ ✘ ✅ ❌`` 等符号绘制条形图与状态标记。中文 Windows 上
``sys.stdout`` 的编码是 GBK（cp936），其中 ``░``(U+2591)、``✔``(U+2714) 等字符
**没有 GBK 编码**；而 stdout 的默认错误策略是 ``surrogateescape`` —— 一旦打印
就抛 :class:`UnicodeEncodeError`，并且是在 ``print`` 处中断整个进程。
受影响的有：启动横幅、``run.py --health``、``--list-tools``、
``python -m src.evaluation.*`` 的进度输出。

方案
----
两条腿走路，先试好的、再兜底：

1. **让控制台真正支持 UTF-8**：Windows 上把控制台输出代码页切到 65001（仅当
   stdout 是真实终端时才有意义），再把 ``sys.stdout`` / ``sys.stderr`` 重配为
   UTF-8。符号既能打印、也能正确显示。
2. **兜底**：第 1 步不适用时（重定向到文件、非 Windows、重配失败）仍尽力切到
   UTF-8；再失败则退化为 ``errors="replace"``，保证**只降级、不崩溃**。

用户若已显式设置 ``PYTHONIOENCODING``，或设置了 ``APP_NO_UTF8_PATCH=1``，
则完全尊重用户选择，不做任何改动。
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

__all__ = ["enable_utf8_output", "can_encode"]

#: 探测样本：覆盖项目中实际会用到的非 GBK 字符（块元素 / 勾叉 / 制表符）
_PROBE = "░█✔✘✅❌═│└→"
_ENV_FLAG = "APP_NO_UTF8_PATCH"

_applied = False


def can_encode(stream: Optional[TextIO], text: str = _PROBE) -> bool:
    """判断 *stream* 当前编码能否表示 *text*；无法判断时保守返回 ``True``。"""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _switch_windows_console_to_utf8() -> bool:
    """把 Windows 控制台输出代码页切到 UTF-8；成功返回 ``True``。"""
    if sys.platform != "win32":
        return False
    # 重定向到文件/管道时改代码页没有意义，也不该改
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.SetConsoleOutputCP(65001))
    except Exception:  # pragma: no cover - 非标准 Windows 环境
        return False


def _reconfigure(stream: Optional[TextIO], encoding: str, errors: str) -> bool:
    """对单个流调用 ``reconfigure``；不支持则返回 ``False``。"""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return False
    try:
        reconfigure(encoding=encoding, errors=errors)
        return True
    except (ValueError, OSError, AttributeError):  # pragma: no cover - 环境相关
        return False


def enable_utf8_output(force: bool = False) -> bool:
    """让 ``stdout`` / ``stderr`` 能安全输出项目用到的全部符号。

    Args:
        force: 为 ``True`` 时即使当前编码已经够用也重新配置。
            默认 ``False`` —— 只在确有字符无法编码时才动手，避免无谓副作用。

    Returns:
        是否对本进程的输出流做了改动。重复调用是幂等的。

    Note:
        只在**进程级入口**调用（``run.py``、``--cli``、``python -m src.evaluation.*``）。
        作为库被 import 时不会自动生效，以免影响宿主的输出行为。
    """
    global _applied
    if _applied and not force:
        return True
    if os.environ.get(_ENV_FLAG, "").strip() not in ("", "0", "false", "False"):
        return False
    # 用户显式指定过编码，尊重之
    if os.environ.get("PYTHONIOENCODING", "").strip():
        return False

    stdout_ok = can_encode(sys.stdout) and can_encode(sys.stderr)
    if stdout_ok and not force:
        _applied = True
        return False

    # 第一步：让控制台本身按 UTF-8 解释字节（失败不影响后续）
    _switch_windows_console_to_utf8()

    # 第二步：把流切到 UTF-8；errors="replace" 保证任何残余问题都不会抛异常
    for stream in (sys.stdout, sys.stderr):
        if stream is None:  # pythonw 等无控制台场景
            continue
        if not _reconfigure(stream, "utf-8", "replace"):
            if not can_encode(stream):
                # 最后兜底：保持原编码，但把无法编码的字符替换掉而不是崩溃
                _reconfigure(stream, getattr(stream, "encoding", "utf-8") or "utf-8", "replace")

    _applied = True
    return True
