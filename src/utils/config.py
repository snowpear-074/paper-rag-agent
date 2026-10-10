"""全局配置系统。

设计目标
--------
1. **零依赖可运行**：优先使用 PyYAML；若未安装则回退到内置的极简 YAML 子集解析器，
   保证在没有第三方库的裸环境中也能读取 config.yaml。
2. **分层覆盖**：默认值 → config.yaml → 环境变量（``APP_LLM__PROVIDER`` 形式）→ 代码显式传参。
3. **属性访问**：``cfg.llm.provider``、``cfg.retrieval.top_k``，缺失时返回安全默认值而不是抛异常。

用法
----
>>> from src.utils.config import load_config
>>> cfg = load_config()                       # 读取项目根目录 config.yaml
>>> cfg.llm.provider
'mock'
>>> load_config("custom.yaml", overrides={"retrieval": {"top_k": 10}}).retrieval.top_k
10
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

__all__ = ["Config", "load_config", "get_config", "DEFAULTS", "PROJECT_ROOT"]

# 项目根目录：src/utils/config.py -> src/utils -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ------------------------------------------------------------------------------
# 内置默认值（当 config.yaml 缺失或字段缺失时兜底）
# ------------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "llm": {
        "provider": "openai",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 2048,
        "timeout": 120,
        "max_retries": 3,
        "stream": True,
    },
    "embedding": {
        "provider": "hash",
        "model": "bge-large-zh",
        "base_url": "http://127.0.0.1:11434",
        "api_key": "",
        "dimension": 512,
        "batch_size": 32,
        "normalize": True,
        "cache_dir": "./data/models",
    },
    "vector_store": {
        "backend": "numpy",
        "persist_dir": "./data/vector_store",
        "collection": "papers",
        "distance": "cosine",
    },
    "chunking": {
        "strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "min_chunk_size": 80,
        "semantic": {"similarity_threshold": 0.62, "max_sentences": 12},
        "paper_optimize": {
            "remove_references": True,
            "merge_cross_page": True,
            "keep_formula_lines": True,
            "dehyphenate": True,
        },
    },
    "retrieval": {
        "top_k": 5,
        "candidate_k": 20,
        "enable_bm25": True,
        "enable_rerank": True,
        "rrf_k": 60,
        "vector_weight": 1.0,
        "bm25_weight": 1.0,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "min_score": 0.0,
        "low_relevance_threshold": 0.25,
        # knowledge_search 工具的置信度过滤门槛（低于此值的片段不交给模型）。
        # 标定：域外问题置信度恒为 0.010，域内最低约 0.124，故取 0.06 分隔。
        "min_confidence": 0.06,
    },
    "data_loader": {
        "pdf_backend": "auto",
        "prefer_python_docx": False,
        "text_strategy": None,
    },
    "reranker": {
        "provider": "heuristic",
        "model": "bge-reranker-base",
        "base_url": "http://127.0.0.1:11434",
        "top_n": 5,
    },
    "generation": {
        "max_context_chars": 6000,
        "max_chunk_chars": 1500,
        "fallback_to_llm": True,
        "citation_style": "【{source} p.{page}】",
        "cache": {
            "enabled": True,
            "backend": "sqlite",
            "path": "./data/cache/semantic_cache.db",
            "similarity_threshold": 0.93,
            "ttl_seconds": 604800,
            "max_entries": 5000,
        },
    },
    "agent": {
        "max_iterations": 8,
        "max_parallel_tools": 3,
        "tool_timeout": 45,
        "max_retries_per_tool": 2,
        "enable_router": True,
        "enable_parallel": True,
        "loop_detect_window": 2,
        "max_observation_chars": 2000,
    },
    "memory": {
        "backend": "sqlite",
        "path": "./data/memory/chat.db",
        "max_turns": 20,
        "max_tokens": 3000,
        "summary_threshold": 12,
        "keep_recent_after_summary": 6,
    },
    "observability": {
        "log_dir": "./logs",
        "level": "INFO",
        "trace_file": "./logs/traces.jsonl",
        "metrics_file": "./logs/metrics.jsonl",
        "console_color": True,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8501,
        "reload": False,
        # 空 = 仅同源（默认最安全）。要允许跨源访问，显式列出来源，
        # 例如 ["http://localhost:5173"]；**不要**用 ["*"]，它会让任意网页
        # 借用户浏览器读写本机知识库。
        "cors_origins": [],
        "max_upload_mb": 100,
        # 是否允许 HTTP 接口按**任意**本地路径导入文档（默认关闭）
        "allow_external_paths": False,
    },
    "paths": {
        "data_dir": "./data",
        "upload_dir": "./data/uploads",
        "index_dir": "./data/index",
        "report_dir": "./reports",
    },
    "web_search": {
        "enabled": True,
        "provider": "auto",
        "api_key": "",
        "max_results": 5,
        "timeout": 20,
        "user_agent": "",
    },
}


# ------------------------------------------------------------------------------
# 极简 YAML 子集解析器（PyYAML 缺失时的回退方案）
# 支持：嵌套映射、行内简单标量、行内列表 [a, b]、块状列表 "- item"、
#       注释、引号字符串、bool/int/float/null 推断、块标量 "|" 与 ">"
# ------------------------------------------------------------------------------
_SCALAR_TRUE = {"true", "yes", "on"}
_SCALAR_FALSE = {"false", "no", "off"}
_SCALAR_NULL = {"null", "~", ""}


def _strip_inline_comment(text: str) -> str:
    """去掉行内注释，但保留引号内部的内容。

    必须**先定位闭合引号**再处理其后的注释 —— 早期实现看到首字符是引号就
    整段跳过注释剥离，导致 ``backend: "sqlite"   # sqlite | memory``
    被解析成字面量 ``'"sqlite"   # sqlite | memory'``，
    缓存后端名因此变成一长串垃圾字符串，缓存静默退化成纯内存模式
    （表现为"重启后缓存全丢"这种难以定位的现象）。
    """
    stripped = text.strip()
    if stripped[:1] in ("'", '"'):
        quote = stripped[0]
        end = stripped.find(quote, 1)
        if end != -1:
            value = stripped[: end + 1]
            tail = stripped[end + 1 :].strip()
            # 引号后只允许注释或空白，否则视为格式异常，仍取引号内内容
            if not tail or tail.startswith("#"):
                return value
            return value
        return stripped

    hash_index = stripped.find("#")
    if hash_index == -1:
        return stripped
    return stripped[:hash_index].strip()


def _parse_scalar(raw: str) -> Any:
    """把 YAML 标量字符串转成 Python 对象。"""
    text = _strip_inline_comment(raw)
    if not text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in _SCALAR_TRUE:
        return True
    if lowered in _SCALAR_FALSE:
        return False
    if lowered in _SCALAR_NULL:
        return None
    # 行内列表 [a, b, c]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in _split_inline_list(inner)]
    # 行内字典 {a: 1, b: 2}
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        result: Dict[str, Any] = {}
        for part in _split_inline_list(inner):
            if ":" in part:
                key, _, value = part.partition(":")
                result[key.strip().strip("\"'")] = _parse_scalar(value)
        return result
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_inline_list(inner: str) -> Iterable[str]:
    """按逗号切分，但忽略引号内的逗号。"""
    parts, buf, quote = [], [], ""
    for ch in inner:
        if quote:
            if ch == quote:
                quote = ""
            buf.append(ch)
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p != ""]


def _fallback_yaml_load(text: str) -> Dict[str, Any]:
    """缩进驱动的 YAML 子集解析器。

    只覆盖 config.yaml 中实际使用的语法，不追求 YAML 1.2 完全兼容。
    PyYAML 存在时不会被调用。
    """
    root: Dict[str, Any] = {}
    # 栈元素：(缩进层级, 容器对象, 该容器"当前等待子项的键名")
    stack: list[tuple[int, Any, Optional[str]]] = [(-1, root, None)]
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        i += 1
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = raw_line.strip()

        # 弹出缩进过深的层级
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        _, container, pending_key = stack[-1]

        # 块状列表项 "- xxx"
        if content.startswith("- "):
            item_text = content[2:].strip()
            target = container
            if isinstance(target, dict) and pending_key is not None:
                target = target.setdefault(pending_key, [])
            if not isinstance(target, list):
                continue
            if ":" in item_text and not item_text.startswith(("\"", "'")):
                # 形如 "- key: value" 的复合列表项
                key, _, value = item_text.partition(":")
                entry: Dict[str, Any] = {key.strip(): _parse_scalar(value)}
                target.append(entry)
                stack.append((indent, entry, None))
                # 后续同缩进的 "key: value" 归入该 entry
                stack[-1] = (indent - 1, entry, None)
            else:
                target.append(_parse_scalar(item_text))
            continue

        if ":" not in content:
            continue
        key, _, value = content.partition(":")
        key = key.strip().strip("\"'")
        value = value.strip()

        if not isinstance(container, dict):
            continue

        if value in ("|", ">"):
            # 块标量：收集后续更深缩进的行
            block_lines = []
            base_indent = indent
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    block_lines.append("")
                    i += 1
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip(" "))
                if nxt_indent <= base_indent:
                    break
                block_lines.append(nxt.strip() if value == ">" else nxt[(base_indent + 2):])
                i += 1
            container[key] = ("\n" if value == "|" else " ").join(block_lines).strip()
            continue

        if value == "":
            # 可能是嵌套映射，也可能是空值 —— 先探测下一行
            nxt_container: Any = {}
            container[key] = nxt_container
            stack.append((indent, nxt_container, None))
            continue

        container[key] = _parse_scalar(value)
    return root


def _yaml_load(path: Path) -> Dict[str, Any]:
    """加载 YAML，优先 PyYAML，失败则回退内置解析器。"""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:  # pragma: no cover - 取决于环境是否装了 PyYAML
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return _fallback_yaml_load(text)


# ------------------------------------------------------------------------------
# 深合并 & 环境变量覆盖
# ------------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并字典：override 覆盖 base，返回新字典（不修改入参）。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce_env_value(raw: str) -> Any:
    """环境变量是字符串，需要推断成合适的类型。"""
    stripped = raw.strip()
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return _parse_scalar(stripped)
    return _parse_scalar(stripped)


def _apply_env_overrides(data: Dict[str, Any], prefix: str = "APP_") -> Dict[str, Any]:
    """扫描 ``APP_<SECTION>__<KEY>`` 形式的环境变量并覆盖配置。"""
    result = copy.deepcopy(data)
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path = env_key[len(prefix):].lower().split("__")
        if not path or not path[0]:
            continue
        cursor = result
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce_env_value(env_value)
    return result


# ------------------------------------------------------------------------------
# Config 对象
# ------------------------------------------------------------------------------
# 需要掩码的键名（api_key / llm_api_key / embedding_api_key / token / secret ...）
_SENSITIVE_KEY_RE = re.compile(r"(?i)(api[_-]?key|apikey|token|secret|password|passwd)")


def _mask_sensitive(data: Any, depth: int = 0) -> Any:
    """递归掩码配置中的敏感字段（只用于 ``repr`` 等展示出口，不改变真实取值）。"""
    if depth > 8:
        return data
    if isinstance(data, dict):
        masked: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key) and value:
                masked[key] = "***" if not isinstance(value, str) else value[:4] + "***"
            else:
                masked[key] = _mask_sensitive(value, depth + 1)
        return masked
    if isinstance(data, list):
        return [_mask_sensitive(item, depth + 1) for item in data]
    return data


class Config:
    """支持属性链式访问的配置对象。

    ``cfg.retrieval.top_k`` 等价于 ``cfg["retrieval"]["top_k"]``。
    ``cfg.get("a.b.c", default)`` 支持点号路径查询。
    嵌套 dict 会被自动包装成 Config，因此可以无限层级链式访问。
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None) -> None:
        object.__setattr__(self, "_data", {})
        store = object.__getattribute__(self, "_data")
        for key, value in (data or {}).items():
            store[key] = Config._wrap(value)

    # -- 内部工具 ---------------------------------------------------------
    @staticmethod
    def _wrap(value: Any) -> Any:
        """把嵌套 dict 包装成 Config，使任意层级都能链式属性访问。

        列表也会逐元素包装（例如 ``paths.report_dirs: [{path: ..., enabled: true}]``），
        否则 ``cfg.xxx[0].path`` 会因元素仍是普通 dict 而失败。
        """
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    @staticmethod
    def _unwrap(value: Any) -> Any:
        if isinstance(value, Config):
            return value.to_dict()
        if isinstance(value, list):
            return [Config._unwrap(v) for v in value]
        return value

    def _store(self) -> Dict[str, Any]:
        """取内部字典。使用 object.__getattribute__ 绕过自定义 __getattribute__，
        避免 "_data" 这个查找本身再次进入自定义逻辑造成无限递归。"""
        return object.__getattribute__(self, "_data")

    # -- 访问接口 ---------------------------------------------------------
    def __getattribute__(self, item: str) -> Any:
        """属性访问优先级：**类成员（方法/属性）> 配置键**。

        这条优先级规则很重要：配置里完全可能出现 ``get``、``items``、
        ``keys`` 这类键名（例如 ``retrieval.get``），若让配置键获胜，
        会把 :meth:`get` 方法遮蔽成 Config 对象，进而在调用时抛出
        "Config object is not callable" 这种难以定位的错误。
        因此方法名一律优先；要读取同名的配置键请使用 ``cfg["get"]``。
        """
        # 双下划线魔术方法直接走默认路径（保证 __init__/__repr__ 等正常工作）
        if item.startswith("__") and item.endswith("__"):
            return object.__getattribute__(self, item)

        # 1) 类成员优先：数据描述符（property 等）直接由 object 处理；
        #    普通函数需要显式绑定 self；其余类属性直接返回。
        cls = type(self)
        for klass in cls.__mro__:
            if item in klass.__dict__:
                raw = klass.__dict__[item]
                if hasattr(raw, "__get__"):
                    return raw.__get__(self, cls)   # 绑定方法 / property
                return raw

        # 2) 再查配置键
        data = Config._store(self)
        if item in data:
            return data[item]

        # 3) 字段确实不存在 —— 抛 AttributeError，而不是返回空 Config。
        #
        # 这里刻意选择"快速失败"：早期版本返回空 Config 做占位，看似友好，
        # 实际会把错误推迟到很远的地方爆发（例如 str(cfg.x) 得到 "" 之后
        # 再调用 .lower() 才报 "Config object is not callable"），极难定位。
        # 需要"带默认值读取"的场景请统一使用 cfg.get("a.b.c", default)。
        raise AttributeError(
            f"配置项 '{item}' 不存在。可用键：{sorted(data.keys())}。"
            f"（如需带默认值读取，请用 cfg.get(\"{item}\", default)）"
        )

    def __getattr__(self, item: str) -> Any:
        """仅在常规查找（含 __getattribute__）失败后触发。

        正常情况下 :meth:`__getattribute__` 已经处理完所有分支，
        因此这里只会收到 ``_data`` 尚未初始化时的探测请求（如 pickle/copy），
        统一转为 AttributeError 以保持 Python 语义正确。
        """
        raise AttributeError(f"Config 对象没有属性 '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            object.__setattr__(self, key, value)
            return
        Config._store(self)[key] = Config._wrap(value)

    def __getitem__(self, key: str) -> Any:
        return Config._store(self)[key]

    def __setitem__(self, key: str, value: Any) -> None:
        Config._store(self)[key] = Config._wrap(value)

    def __delitem__(self, key: str) -> None:
        del Config._store(self)[key]

    def __contains__(self, key: str) -> bool:
        return key in Config._store(self)

    def __iter__(self):
        return iter(Config._store(self))

    def __len__(self) -> int:
        return len(Config._store(self))

    def __repr__(self) -> str:
        data = Config._store(self)
        if not data:
            return "Config(<empty>)"
        # 必须掩码后再序列化：repr 会出现在异常信息、HTTP 响应、print 等
        # **非 logging** 出口，logging 的密钥脱敏过滤器覆盖不到它们
        preview = json.dumps(_mask_sensitive(self.to_dict()), ensure_ascii=False, default=str)
        if len(preview) > 400:
            preview = preview[:400] + "..."
        return f"Config({preview})"

    def __str__(self) -> str:
        """字符串化规则。

        - **单标量键**的 Config（被当作值使用的叶子节点）→ 返回该标量。
          例：``str(cfg.retrieval.rrf_k)`` == ``"60"``。
        - **其它情况**（多键节点，或单键但值是嵌套结构）→ 返回 ``""``。

        为什么不是返回 JSON
            早期实现把多键节点序列化成 JSON 字符串，结果 ``str(cfg.memory)``
            得到 ``'{"backend": "sqlite", ...}'`` —— 当调用方本意是读
            ``cfg.memory.backend`` 却误写为 ``str(cfg.memory)`` 时，这个
            看似正常的字符串会被当成合法配置值一路传下去（例如缓存后端名
            变成一整段 JSON），错误被推迟到很远才以难以理解的方式暴露。
            现在返回 ``""``，配合 "``or 默认值``" 的写法会立刻回退到安全默认值，
            从而让这类误用**无害化**。

        注意：配置项**缺失**时不会走到这里 —— 属性访问会直接抛 AttributeError。
        """
        store = Config._store(self)
        if len(store) == 1:
            only = next(iter(store.values()))
            if not isinstance(only, (Config, list, dict)):
                return str(only)
        return ""

    def __bool__(self) -> bool:
        """空 Config 为 False，便于 ``if cfg.get("x", {})`` 这类写法。"""
        return bool(Config._store(self))

    def keys(self):
        return Config._store(self).keys()

    def items(self):
        return Config._store(self).items()

    def get(self, path: str, default: Any = None) -> Any:
        """点号路径查询：``cfg.get("llm.provider", "mock")``。"""
        cursor: Any = Config._store(self)
        for part in path.split("."):
            if isinstance(cursor, Config):
                cursor = Config._store(cursor)
            if not isinstance(cursor, Mapping) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def to_dict(self) -> Dict[str, Any]:
        """转为纯 Python 字典（递归解包）。"""
        return {k: Config._unwrap(v) for k, v in Config._store(self).items()}

    def merged_with(self, overrides: Optional[Mapping[str, Any]] = None) -> "Config":
        """返回叠加 overrides 后的新 Config（不修改自身）。"""
        return Config(_deep_merge(self.to_dict(), overrides or {}))

    def resolve_path(self, path: str) -> Path:
        """把配置中的相对路径解析为相对项目根目录的绝对路径。"""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (PROJECT_ROOT / candidate).resolve()

    def ensure_dirs(self) -> None:
        """创建配置里声明的各类数据目录。"""
        for section, keys in (
            ("paths", ("data_dir", "upload_dir", "index_dir", "report_dir")),
            ("vector_store", ("persist_dir",)),
            ("observability", ("log_dir",)),
        ):
            node = Config._store(self).get(section)
            if not isinstance(node, Config):
                continue
            for key in keys:
                value = Config._store(node).get(key)
                if isinstance(value, str) and value:
                    self.resolve_path(value).mkdir(parents=True, exist_ok=True)
        # 缓存 / 记忆数据库的父目录
        for dotted in ("generation.cache.path", "memory.path"):
            value = self.get(dotted)
            if isinstance(value, str) and value:
                self.resolve_path(value).parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 对外入口
# ------------------------------------------------------------------------------
_CACHED: Optional[Config] = None


def load_config(
    path: Optional[Any] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    use_env: bool = True,
) -> Config:
    """加载配置。

    Parameters
    ----------
    path:
        config.yaml 路径。None 表示使用项目根目录的 config.yaml。
    overrides:
        代码层覆盖，优先级最高。
    use_env:
        是否应用 ``APP_*`` 环境变量覆盖。
    """
    # 先把密封凭据库中的 API Key 注入环境变量，使下面的环境变量覆盖能读到它们。
    # 这样密钥**永远不需要写进 config.yaml**（避免明文入库）。
    # 已在环境里的值优先，不会被密封库覆盖。
    if use_env:
        try:
            from .secrets import inject_env

            inject_env()
        except Exception:  # noqa: BLE001 - 凭据库不可用不应阻断配置加载
            pass

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    file_data = _yaml_load(config_path)
    merged = _deep_merge(DEFAULTS, file_data)
    if use_env:
        merged = _apply_env_overrides(merged)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config(merged)


def get_config(reload: bool = False, **kwargs: Any) -> Config:
    """获取进程级单例配置（避免反复读盘）。"""
    global _CACHED
    if _CACHED is None or reload or kwargs:
        if kwargs:
            return load_config(kwargs.pop("path", None), kwargs or None)
        _CACHED = load_config()
    return _CACHED


def reset_config_cache() -> None:
    """清空单例缓存（测试用）。"""
    global _CACHED
    _CACHED = None
