"""API 凭据安全保管（模块：utils.secrets）。

设计目标
--------
1. **明文不落库、不入库**：密钥以**密封（sealed）**形式存放，文件内容被 XOR 流加密
   后再 Base64 编码。直接打开文件只看到密文，`git grep` 也搜不到明文。
2. **绝不回显**：提供 :func:`mask_secret` 用于日志/接口输出，任何地方只显示
   ``sk-7d28****901d`` 这种带掩码的形式。
3. **优先级明确**：``环境变量 > 密封文件 > config.yaml``。这样 CI/生产用环境变量
   注入，本地开发用密封文件，两者互不干扰，且配置文件里可以永久留空。

安全边界（务必理解）
--------------------
密封密钥用于**防止明文泄露（肩窥、误提交、日志抓取）**，**不是**用于防御
拥有本机读取权限的攻击者 —— 解封逻辑就在代码里，能读文件的人也能解密。
对于本项目的定位（个人科研助手、单机部署）这是恰当的权衡；
若需真正的密钥保护，应改用操作系统凭据管理器或 KMS。

用法
----
.. code-block:: bash

    # 写入（交互式，不回显）
    python -m src.utils.secrets set llm_api_key

    # 查看当前状态（只看掩码，不显示明文）
    python -m src.utils.secrets status

    # 清除
    python -m src.utils.secrets clear llm_api_key
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "SecretsVault",
    "get_vault",
    "get_secret",
    "set_secret",
    "delete_secret",
    "mask_secret",
    "inject_env",
    "SECRET_ENV_MAP",
]

# 项目根目录（src/utils/secrets.py -> src/utils -> src -> root）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEALED_PATH = PROJECT_ROOT / "data" / "secrets" / "credentials.sealed.json"

# 一个固定的"皮肤"值。它不是安全密钥（代码里可见），
# 作用是让密封文件不是可读明文，从而防住"误提交 / 顺手打开看一眼"这类泄露。
_VEIL = b"research-assistant::vault::v1"

# 逻辑密钥名 -> 环境变量名（供 inject_env 注入）
SECRET_ENV_MAP: Dict[str, str] = {
    "llm_api_key": "APP_LLM__API_KEY",
    "embedding_api_key": "APP_EMBEDDING__API_KEY",
    "web_search_api_key": "APP_WEB_SEARCH__API_KEY",
}


# ------------------------------------------------------------------------------
# 掩码
# ------------------------------------------------------------------------------
def mask_secret(value: str, head: int = 6, tail: int = 4) -> str:
    """把密钥转成可安全展示的掩码形式。

    例：``"sk-abc1234567890xyz"`` → ``"sk-abc****0xyz"``

    短于 ``head + tail + 4`` 的值整体打码，避免"掩码后反而暴露全貌"。
    """
    if not value:
        return "（未配置）"
    text = str(value)
    if len(text) <= head + tail + 4:
        return "*" * len(text)
    return f"{text[:head]}****{text[-tail:]}"


def secret_fingerprint(value: str) -> str:
    """密钥指纹（SHA-256 前 8 位）：用于校验"当前生效的是哪把钥匙"而不暴露它。"""
    if not value:
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8]


# ------------------------------------------------------------------------------
# 密封算法（XOR 流 + 机器相关盐）
# ------------------------------------------------------------------------------
def _keystream(length: int, salt: bytes) -> bytes:
    """由 VEIL + 盐派生 ``length`` 字节的伪随机流。

    用 SHA-256 计数器模式展开，避免引入任何加密库依赖。
    """
    blocks = bytearray()
    counter = 0
    seed = _VEIL + salt
    while len(blocks) < length:
        blocks += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(blocks[:length])


def _machine_salt() -> bytes:
    """本机盐：绑定当前用户与主机名。

    作用有二：(1) 让密封文件在不同机器间不可直接复用；(2) 提高"猜明文"的难度。
    注意它不是安全边界 —— 代码可读，盐可复现。
    """
    material = f"{platform.node()}|{os.environ.get('USERNAME') or os.environ.get('USER') or ''}"
    return hashlib.sha256(material.encode("utf-8")).digest()[:16]


def seal(plaintext: str, salt: Optional[bytes] = None) -> str:
    """把明文密封成 Base64 字符串。"""
    raw = plaintext.encode("utf-8")
    salt = salt if salt is not None else _machine_salt()
    stream = _keystream(len(raw), salt)
    return base64.urlsafe_b64encode(bytes(a ^ b for a, b in zip(raw, stream))).decode("ascii")


def unseal(ciphertext: str, salt: Optional[bytes] = None) -> str:
    """还原密封字符串；失败时返回空串（不抛异常，避免拖垮启动流程）。"""
    try:
        blob = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
    except Exception:  # noqa: BLE001
        return ""
    salt = salt if salt is not None else _machine_salt()
    stream = _keystream(len(blob), salt)
    try:
        return bytes(a ^ b for a, b in zip(blob, stream)).decode("utf-8")
    except UnicodeDecodeError:
        return ""


# ------------------------------------------------------------------------------
# 保管库
# ------------------------------------------------------------------------------
class SecretsVault:
    """密封凭据文件 + 进程缓存的组合。

    Parameters
    ----------
    path: 密封文件路径
    env_map: 逻辑名 -> 环境变量名 的映射
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        env_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.path = Path(path) if path else DEFAULT_SEALED_PATH
        self.env_map = dict(env_map or SECRET_ENV_MAP)
        self._cache: Dict[str, str] = {}
        self._loaded = False

    # -- 读写 -------------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for name, sealed in (payload.get("secrets") or {}).items():
            value = unseal(str(sealed))
            if value:
                self._cache[name] = value

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": "密封凭据文件。明文不落盘；请勿提交到版本库。",
            "version": 1,
            "secrets": {name: seal(value) for name, value in sorted(self._cache.items()) if value},
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # 收紧文件权限（Windows 上 chmod 语义有限，尽力而为）
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    # -- 公开 API ---------------------------------------------------------
    def get(self, name: str, default: str = "") -> str:
        """读取密钥。优先级：环境变量 > 密封文件。"""
        env_name = self.env_map.get(name)
        if env_name:
            from_env = os.environ.get(env_name, "").strip()
            if from_env:
                return from_env
        self._load()
        return self._cache.get(name, default)

    def set(self, name: str, value: str) -> None:
        """写入并立即落盘。"""
        self._load()
        self._cache[name] = str(value).strip()
        self._save()

    def delete(self, name: str) -> bool:
        self._load()
        existed = self._cache.pop(name, None) is not None
        if existed:
            self._save()
        return existed

    def clear(self) -> None:
        self._cache.clear()
        self._loaded = True
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass

    def names(self) -> List[str]:
        self._load()
        return sorted(self._cache.keys())

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    # -- 状态（可安全展示）------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """返回**不含明文**的状态报告，可直接写日志或返回给前端。"""
        self._load()
        rows: List[Dict[str, Any]] = []
        for name in sorted(set(self._cache) | set(self.env_map)):
            env_name = self.env_map.get(name, "")
            from_env = bool(env_name and os.environ.get(env_name, "").strip())
            value = self.get(name)
            rows.append(
                {
                    "name": name,
                    "configured": bool(value),
                    "source": "环境变量" if from_env else ("密封文件" if name in self._cache else "未配置"),
                    "env_var": env_name,
                    "masked": mask_secret(value),
                    "fingerprint": secret_fingerprint(value),
                    "length": len(value),
                }
            )
        return {
            "vault_path": str(self.path),
            "vault_exists": self.path.exists(),
            "secrets": rows,
        }


# ------------------------------------------------------------------------------
# 单例
# ------------------------------------------------------------------------------
_VAULT: Optional[SecretsVault] = None


def get_vault(path: Optional[Path] = None) -> SecretsVault:
    global _VAULT
    if _VAULT is None or path is not None:
        _VAULT = SecretsVault(path)
    return _VAULT


def get_secret(name: str, default: str = "") -> str:
    return get_vault().get(name, default)


def set_secret(name: str, value: str) -> None:
    get_vault().set(name, value)


def delete_secret(name: str) -> bool:
    return get_vault().delete(name)


def inject_env(vault: Optional[SecretsVault] = None, override: bool = False) -> List[str]:
    """把保管库中的密钥注入环境变量，使配置系统（``APP_*``）自动读取。

    Parameters
    ----------
    override: 为 False 时不覆盖已存在的环境变量（环境变量优先级最高）。

    Returns
    -------
    实际注入的环境变量名列表（**不含值**，可安全打印）。
    """
    vault = vault or get_vault()
    injected: List[str] = []
    for name, env_name in vault.env_map.items():
        if not override and os.environ.get(env_name, "").strip():
            continue
        value = vault.get(name)
        if value:
            os.environ[env_name] = value
            injected.append(env_name)
    return injected


# ------------------------------------------------------------------------------
# 命令行
# ------------------------------------------------------------------------------
def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    from .console import enable_utf8_output

    enable_utf8_output()

    parser = argparse.ArgumentParser(
        prog="python -m src.utils.secrets",
        description="API 密钥安全保管（密封存储，明文不落盘）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_set = sub.add_parser("set", help="写入密钥（交互式输入，不回显）")
    p_set.add_argument("name", help="逻辑名，如 llm_api_key")
    p_set.add_argument("--value", default=None, help="直接提供值（会留在命令历史里，不推荐）")

    p_del = sub.add_parser("clear", help="删除指定密钥")
    p_del.add_argument("name", nargs="?", default=None, help="逻辑名；留空则清空全部")

    sub.add_parser("status", help="查看配置状态（只显示掩码）")
    sub.add_parser("names", help="列出已知的逻辑名与环境变量名")

    args = parser.parse_args(argv)
    vault = get_vault()

    if args.command == "set":
        value = args.value
        if value is None:
            value = getpass.getpass(f"请输入 {args.name} 的值（输入不回显）: ").strip()
        if not value:
            print("✘ 值为空，已取消。")
            return 1
        vault.set(args.name, value)
        print(f"✔ 已密封保存 {args.name} → {vault.path}")
        print(f"  掩码: {mask_secret(value)}   指纹: {secret_fingerprint(value)}")
        print("  提示：请确认 data/secrets/ 已在 .gitignore 中（默认已包含）。")
        return 0

    if args.command == "clear":
        if args.name:
            ok = vault.delete(args.name)
            print(f"{'✔ 已删除' if ok else '– 未找到'} {args.name}")
        else:
            vault.clear()
            print("✔ 已清空全部密封凭据")
        return 0

    if args.command == "names":
        print("逻辑名 → 环境变量：")
        for name, env_name in sorted(vault.env_map.items()):
            print(f"  {name:24s} → {env_name}")
        return 0

    # status
    info = vault.describe()
    print("=" * 66)
    print("  API 凭据状态")
    print("=" * 66)
    print(f"  密封文件: {info['vault_path']}")
    print(f"  文件存在: {'是' if info['vault_exists'] else '否'}")
    print()
    for row in info["secrets"]:
        mark = "✔" if row["configured"] else "–"
        print(f"  {mark} {row['name']:24s} {row['masked']:20s} 来源: {row['source']}")
        if row["configured"]:
            print(f"     指纹 {row['fingerprint']} · 长度 {row['length']} · 环境变量 {row['env_var']}")
    print("=" * 66)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
