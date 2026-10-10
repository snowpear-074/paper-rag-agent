"""Agent 工具框架（模块三 5.3.2）。

工具是 Agent 与外部世界交互的唯一通道。本模块定义统一的工具契约：

- :class:`ToolParameter` —— 参数描述（类型、是否必填、默认值、说明）
- :class:`BaseTool`      —— 工具基类：名称 / 描述 / 参数表 / ``run()``
- :class:`ToolRegistry`  —— 工具注册表：注册、查找、生成给 LLM 的工具清单

设计原则
--------
1. **声明式参数表**：工具自行声明参数结构，Agent 据此：
   (a) 渲染给 LLM 的工具说明；(b) 在调用前做类型转换与必填校验 —— 
   让"模型传错参数"在进入业务逻辑前就被拦截并给出可读错误，
   而不是抛一个难以理解的 TypeError。
2. **永不抛异常**：``run()`` 内部任何异常都转成 ``ToolResult(status=ERROR)``，
   由 Agent 决定重试或改用其它工具。这是"错误恢复"机制的基础。
3. **可观测**：每次调用记录耗时、成功与否，写入全局指标。
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type

from ..utils.logger import get_logger, get_metrics
from ..utils.types import ToolCall, ToolResult, ToolStatus, new_id

__all__ = [
    "ToolParameter",
    "BaseTool",
    "ToolRegistry",
    "ToolError",
    "tool_from_function",
    "PYTHON_TYPE_MAP",
]

_log = get_logger("agent.tools")


class ToolError(Exception):
    """工具执行失败（会被 :meth:`BaseTool.run` 捕获并转成错误结果）。"""


# JSON Schema 类型 ↔ Python 类型
PYTHON_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class ToolParameter:
    """单个工具参数的声明。"""

    name: str
    type: str = "string"                 # JSON Schema 类型
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None
    items_type: str = "string"           # type=array 时的元素类型

    def to_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        if self.type == "array":
            schema["items"] = {"type": self.items_type}
        if self.default is not None:
            schema["default"] = self.default
        return schema

    def coerce(self, value: Any) -> Any:
        """把传入值转换成声明类型（宽松转换，失败则抛 ToolError）。"""
        if value is None:
            return self.default
        target = PYTHON_TYPE_MAP.get(self.type)
        if target is None:
            return value
        # 已经是目标类型
        if isinstance(value, target) and not (target is float and isinstance(value, bool)):
            return value
        try:
            if target is str:
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) \
                    if isinstance(value, (dict, list)) else str(value)
            if target is bool:
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "y", "是", "真")
                return bool(value)
            if target is int:
                return int(float(value)) if isinstance(value, str) else int(value)
            if target is float:
                return float(value)
            if target is list:
                if isinstance(value, str):
                    # 允许模型传 JSON 数组字符串或逗号分隔串
                    text = value.strip()
                    if text.startswith("["):
                        parsed = json.loads(text)
                        return parsed if isinstance(parsed, list) else [parsed]
                    return [v.strip() for v in text.split(",") if v.strip()]
                return list(value) if isinstance(value, Iterable) else [value]
            if target is dict:
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ToolError(
                f"参数 '{self.name}' 无法转换为 {self.type}：{value!r}（{exc}）"
            ) from exc
        return value


class BaseTool(ABC):
    """工具基类。

    子类实现 :meth:`execute`，并使用 :meth:`__init__` 或类属性声明元信息。
    """

    #: 工具名（LLM 通过它来调用）—— 必须与 ReAct 提示词中列出的名字一致
    name: str = ""
    #: 一句话说明"这个工具能做什么"，直接影响 Agent 的路由准确率
    description: str = ""
    #: 参数表
    parameters: List[ToolParameter] = []
    #: 使用示例（写进提示词，实测能显著降低参数错误率）
    examples: List[str] = []
    #: 是否为主要工具（主要工具缺少依赖时会被标记为不可用）
    enabled: bool = True

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._calls = 0
        self._failures = 0
        self._total_ms = 0.0

    # -- 子类实现 ---------------------------------------------------------
    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """执行工具逻辑。

        返回值可以是：

        - ``str``：直接作为 Observation 文本
        - ``dict`` / ``list``：自动序列化，并在 ``ToolResult.data`` 中保留结构
        - ``tuple(text, data)``：分别指定文本与结构化结果
        """
        raise NotImplementedError

    # -- 统一调用入口 -----------------------------------------------------
    def run(self, call: Optional[ToolCall] = None, **kwargs: Any) -> ToolResult:
        """执行工具并返回 :class:`ToolResult`（**永不抛异常**）。

        参数来源有两种，可混用：

        - ``run(call)``：从 :class:`ToolCall` 中取 ``args`` 作为参数；
        - ``run(expression="1+1")``：直接以关键字传参。

        显式关键字优先于 ``call.args``（便于测试时覆盖单个参数）。
        """
        merged: Dict[str, Any] = {}
        call_id = ""
        if call is not None:
            call_id = call.call_id or new_id("call_")
            if isinstance(call.args, Mapping):
                merged.update(call.args)
        merged.update(kwargs)                       # 显式关键字覆盖 call.args
        merged.pop("__call_id", None)               # 清理历史占位参数
        call_id = call_id or new_id("call_")

        self._calls += 1
        started = time.perf_counter()

        try:
            validated = self.validate(merged)
            output = self.execute(**validated)
            if isinstance(output, tuple) and len(output) == 2:
                text, data = output
            else:
                text, data = self._stringify(output), output
            latency = (time.perf_counter() - started) * 1000
            self._total_ms += latency
            result = ToolResult(
                call_id=call_id,
                name=self.name,
                status=ToolStatus.SUCCESS,
                content=text if isinstance(text, str) else self._stringify(text),
                data=data,
                latency_ms=latency,
            )
        except ToolError as exc:
            self._failures += 1
            result = self._error(call_id, str(exc), started)
        except Exception as exc:  # noqa: BLE001
            self._failures += 1
            detail = traceback.format_exc(limit=2).splitlines()[-2:]
            _log.error("工具 %s 执行异常: %s | %s", self.name, exc, " / ".join(detail))
            result = self._error(call_id, f"{type(exc).__name__}: {exc}", started)

        try:
            get_metrics().record_tool(self.name, result.ok, result.latency_ms)
        except Exception:  # noqa: BLE001
            pass
        return result

    def _error(self, call_id: str, message: str, started: float) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            name=self.name,
            status=ToolStatus.ERROR,
            content=f"工具调用失败：{message}",
            error=message,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- 参数处理 ---------------------------------------------------------
    def validate(self, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        """校验并转换参数。

        - 丢弃未声明的参数（模型常会多传，静默忽略比报错更健壮）
        - 检查必填项
        - 按声明类型转换
        """
        declared = {p.name: p for p in self.parameters}
        validated: Dict[str, Any] = {}

        for name, param in declared.items():
            if name in kwargs and kwargs[name] is not None and kwargs[name] != "":
                validated[name] = param.coerce(kwargs[name])
            elif param.default is not None:
                validated[name] = param.default
            elif param.required:
                available = ", ".join(declared.keys()) or "（无参数）"
                raise ToolError(
                    f"缺少必填参数 '{name}'（{param.description}）。可用参数：{available}"
                )
            elif name in kwargs:
                validated[name] = kwargs[name]

        # 未声明的参数：保留下来但记录，便于排查模型行为
        extra = {k: v for k, v in kwargs.items() if k not in declared}
        if extra:
            _log.debug("工具 %s 收到未声明的参数，已忽略：%s", self.name, list(extra))
            validated.setdefault("__extra__", extra)
        return validated

    @staticmethod
    def _stringify(output: Any) -> str:
        if output is None:
            return "（无返回内容）"
        if isinstance(output, str):
            return output
        if isinstance(output, (dict, list)):
            try:
                return json.dumps(output, ensure_ascii=False, indent=None, default=str)
            except (TypeError, ValueError):
                return str(output)
        return str(output)

    # -- 元信息 -----------------------------------------------------------
    def to_schema(self) -> Dict[str, Any]:
        """转成 OpenAI Function Calling 风格的 JSON Schema。"""
        required = [p.name for p in self.parameters if p.required]
        schema: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {p.name: p.to_schema() for p in self.parameters},
            },
        }
        if required:
            schema["parameters"]["required"] = required
        return schema

    def render_for_prompt(self, max_examples: int = 1) -> str:
        """渲染成写入 ReAct System Prompt 的文本块。"""
        lines = [f"### {self.name}", self.description.strip()]
        if self.parameters:
            lines.append("参数：")
            for param in self.parameters:
                flag = "必填" if param.required else "可选"
                default = "" if param.default is None else f"，默认 {param.default!r}"
                enum = f"，取值 {param.enum}" if param.enum else ""
                lines.append(f"  - {param.name} ({param.type}, {flag}{default}{enum}): {param.description}")
        else:
            lines.append("参数：无")
        for example in self.examples[:max_examples]:
            lines.append(f"调用示例：{example}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "calls": self._calls,
            "failures": self._failures,
            "success_rate": round((self._calls - self._failures) / self._calls, 4) if self._calls else 0.0,
            "avg_latency_ms": round(self._total_ms / self._calls, 2) if self._calls else 0.0,
        }


# ==============================================================================
# 注册表
# ==============================================================================
class ToolRegistry:
    """工具注册表：集中管理 Agent 可调用的工具集合。"""

    def __init__(self, tools: Optional[Iterable[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    # -- 注册 -------------------------------------------------------------
    def register(self, tool: BaseTool, override: bool = False) -> None:
        """注册工具。名称冲突时默认抛错（避免静默覆盖导致行为不可预期）。"""
        if not tool.name:
            raise ValueError("工具必须有非空的 name")
        if tool.name in self._tools and not override:
            raise ValueError(f"工具名冲突：{tool.name} 已注册（如需替换请传 override=True）")
        self._tools[tool.name] = tool
        _log.debug("注册工具: %s", tool.name)

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    @property
    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def enabled_tools(self) -> List[BaseTool]:
        return [t for t in self._tools.values() if t.enabled]

    # -- 执行 -------------------------------------------------------------
    def execute(self, call: ToolCall) -> ToolResult:
        """按名称执行工具；工具不存在时返回明确的错误结果（而非抛异常）。

        对"依赖未就绪"的工具（如未配置的联网搜索）**不做拦截**，交给工具自身的
        ``execute`` 返回带操作指引的错误 —— 这样模型能看到"为什么不可用、
        怎么才能可用"，而不是一个空洞的 SKIPPED。
        """
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names) or "（无）"
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                status=ToolStatus.ERROR,
                content=f"工具 '{call.name}' 不存在。可用工具：{available}",
                error=f"unknown_tool:{call.name}",
            )
        return tool.run(call, **call.args)

    # -- 提示词与前端 -----------------------------------------------------
    def render_catalog(self) -> str:
        """渲染完整工具清单（写入 System Prompt）。"""
        return "\n\n".join(tool.render_for_prompt() for tool in self.enabled_tools())

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def descriptions(self) -> List[Dict[str, Any]]:
        """前端"可用工具"面板所需信息。"""
        return [
            {
                "name": tool.name,
                "description": tool.description.strip(),
                "enabled": tool.enabled,
                "parameters": tool.to_schema()["parameters"],
                "examples": tool.examples,
            }
            for tool in self._tools.values()
        ]

    def stats(self) -> List[Dict[str, Any]]:
        return [tool.stats() for tool in self._tools.values()]

    def health_check(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "num_tools": len(self._tools),
            "enabled": len(self.enabled_tools()),
            "tools": self.names,
        }


# ==============================================================================
# 函数式工具包装（便于快速扩展自定义工具）
# ==============================================================================
def tool_from_function(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    param_descriptions: Optional[Dict[str, str]] = None,
) -> BaseTool:
    """把普通函数包装成 :class:`BaseTool`（从类型注解推断参数表）。

    >>> def add(a: int, b: int) -> int:
    ...     '''两数相加'''
    ...     return a + b
    >>> registry.register(tool_from_function(add))
    """
    signature = inspect.signature(func)
    annotations = {
        "int": "integer", "float": "number", "str": "string",
        "bool": "boolean", "list": "array", "dict": "object",
    }
    params: List[ToolParameter] = []
    for param_name, param in signature.parameters.items():
        annotation = param.annotation
        type_name = getattr(annotation, "__name__", "string")
        json_type = annotations.get(type_name, "string")
        params.append(
            ToolParameter(
                name=param_name,
                type=json_type,
                description=(param_descriptions or {}).get(param_name, ""),
                required=param.default is inspect.Parameter.empty,
                default=None if param.default is inspect.Parameter.empty else param.default,
            )
        )

    class _FunctionTool(BaseTool):
        pass

    _FunctionTool.name = name or func.__name__
    _FunctionTool.description = description or (func.__doc__ or "").strip() or func.__name__
    _FunctionTool.parameters = params

    def execute(self: BaseTool, **kwargs: Any) -> Any:
        cleaned = {k: v for k, v in kwargs.items() if k in signature.parameters}
        return func(**cleaned)

    _FunctionTool.execute = execute  # type: ignore[assignment]
    return _FunctionTool()
