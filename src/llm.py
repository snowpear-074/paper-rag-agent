"""统一的 LLM 接入层。

支持四种后端，通过 ``config.llm.provider`` 切换，业务代码只依赖 :class:`LLMClient`
抽象接口，不感知具体厂商：

============  ==================================================================
provider      说明
============  ==================================================================
``ollama``    本地 Ollama（默认 http://127.0.0.1:11434），完全离线，隐私安全
``openai``    任何 OpenAI 兼容 ``/v1/chat/completions`` 接口
              （OpenAI / DeepSeek / vLLM / LM Studio / 硅基流动 / 通义兼容模式）
``dashscope`` 阿里云通义千问原生接口
``mock``      离线模拟后端 —— 不发起任何网络请求，用于开发自测、单元测试、
              以及无 GPU / 无 API Key 环境下的端到端联调
============  ==================================================================

统一能力
--------
- ``chat()``        同步对话
- ``stream()``      流式对话（生成器，逐 token 产出）
- ``chat_json()``   要求模型输出 JSON，并做容错解析（Agent 决策依赖此接口）
- 自动重试（指数退避）、超时控制、Token 用量估算与成本核算
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .utils.config import Config, get_config
from .utils.logger import get_logger
from .utils.text import estimate_tokens
from .utils.types import Message, Role, TokenUsage

__all__ = ["LLMClient", "LLMError", "create_llm", "MockLLM", "estimate_usage"]

_log = get_logger("llm")


class LLMError(RuntimeError):
    """LLM 调用异常（供 Agent 错误恢复机制捕获并降级）。"""


# ------------------------------------------------------------------------------
# 价格表（元 / 1K tokens），用于成本估算；本地模型记 0
# ------------------------------------------------------------------------------
_PRICE_TABLE: Dict[str, tuple[float, float]] = {
    "qwen2.5:7b": (0.0, 0.0),
    "qwen2.5:14b": (0.0, 0.0),
    "gpt-4o": (0.018, 0.072),
    "gpt-4o-mini": (0.001, 0.004),
    # DeepSeek 官方定价（元 / 1K tokens，缓存未命中口径，2025 年价目）
    "deepseek-chat": (0.002, 0.003),
    "deepseek-reasoner": (0.004, 0.016),
    "deepseek-flash": (0.001, 0.002),
    "deepseek-v4-pro": (0.004, 0.012),
    "qwen-plus": (0.0008, 0.002),
    "qwen-turbo": (0.0003, 0.0006),
}
# 本地推理的等效电费（元 / 1K tokens），用于"超低成本运营"的成本对比
_LOCAL_ELECTRICITY_PER_1K = 0.00004
# 明确按"本地免费"计价的模型（价格表里记 0 的 + Ollama 常见的 tag 形态）
_LOCAL_MODEL_HINTS = ("qwen2.5:", "llama", "mistral", "gemma", "bge", "phi")


def _is_local_model(model: str) -> bool:
    """判断模型是否属于"本地部署、无调用费用"（成本按等效电费计）。"""
    name = (model or "").strip().lower()
    if not name:
        return True  # 未标注模型名（如离线评测）按本地口径
    if name in _PRICE_TABLE:
        return _PRICE_TABLE[name] == (0.0, 0.0)
    return any(hint in name for hint in _LOCAL_MODEL_HINTS)


def estimate_usage(prompt: str, completion: str, model: str = "") -> TokenUsage:
    """估算一次调用的 Token 用量与成本。

    无真实 tokenizer 时按字符数经验公式估算（中文 ~1.5 字符/token，
    英文 ~4 字符/token），误差 ±20%，足以支撑成本对比与预算控制。

    关于"价格未知的模型"
    --------------------
    早期实现把**不在价格表里**的模型一律当成本地模型，按等效电费（0.00004 元/1K）
    计价。这会把任何未收录的云端模型（新发布的 gpt-*、claude-*、自建网关等）
    的真实成本低报几个数量级，而报告里的"成本对比"表格又直接引用该数字 ——
    属于**编造数据**。现在改为：只在模型确属本地时套用电费口径，
    否则 ``cost_yuan=0.0`` + ``pricing_known=False``，由上层显示"价格未收录"，
    绝不给出一个凭空的金额。
    """
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(completion)
    total = prompt_tokens + completion_tokens
    model = (model or "").strip()

    if _is_local_model(model):
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated=True,
            cost_yuan=round(total / 1000.0 * _LOCAL_ELECTRICITY_PER_1K, 6),
            pricing_known=True,
        )

    price = _PRICE_TABLE.get(model.lower())
    if price is None:
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated=True,
            cost_yuan=0.0,
            pricing_known=False,
        )

    price_in, price_out = price
    cost = prompt_tokens / 1000.0 * price_in + completion_tokens / 1000.0 * price_out
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
        estimated=True,
        cost_yuan=round(cost, 6),
        pricing_known=True,
    )


# ------------------------------------------------------------------------------
# 重试分类：只有"值得重试"的失败才退避重试
# ------------------------------------------------------------------------------
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_RETRYABLE_HINTS = (
    "timeout", "timed out", "超时", "connection", "连接", "reset", "refused",
    "temporarily", "rate limit", "too many requests", "限流", "overloaded",
    "bad gateway", "service unavailable", "gateway timeout", "broken pipe",
    "502", "503", "504", "529",
)


def _is_retryable(exc: BaseException) -> bool:
    """判断异常是否属于"重试可能有救"的瞬时故障。

    值得重试：超时、连接失败、限流(429)、服务端 5xx。
    **不值得重试**：401/403 鉴权失败、400/404 参数或模型名错误、额度耗尽 ——
    早期实现不分类别一律退避重试 3 次（约 5.6 秒纯等待），
    既拖慢失败响应，又把真实错误淹没在"已重试 N 次"里。
    """
    seen: set[int] = set()
    cursor: Optional[BaseException] = exc
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        if isinstance(cursor, urllib.error.HTTPError):
            if cursor.code in _RETRYABLE_STATUS:
                return True
        elif isinstance(cursor, (urllib.error.URLError, TimeoutError, ConnectionError)):
            return True
        status = getattr(cursor, "status_code", None) or getattr(cursor, "code", None)
        if isinstance(status, int) and status in _RETRYABLE_STATUS:
            return True
        cursor = cursor.__cause__ or cursor.__context__
    return any(hint in str(exc).lower() for hint in _RETRYABLE_HINTS)


def _http_error(message: str, code: int) -> LLMError:
    """构造携带 HTTP 状态码的 :class:`LLMError`（重试分类依赖 ``status_code``）。"""
    error = LLMError(message)
    error.status_code = code  # type: ignore[attr-defined]
    error.retryable = code in _RETRYABLE_STATUS  # type: ignore[attr-defined]
    return error


# ==============================================================================
# 抽象基类
# ==============================================================================
class LLMClient(ABC):
    """LLM 客户端抽象接口。"""

    provider: str = "base"
    model: str = ""

    def __init__(self, config: Optional[Config] = None) -> None:
        cfg = (config or get_config()).llm
        self.model = str(cfg.model)
        self.temperature = float(cfg.temperature)
        self.top_p = float(cfg.top_p)
        self.max_tokens = int(cfg.max_tokens)
        self.timeout = float(cfg.timeout)
        self.max_retries = int(cfg.max_retries)
        self._call_count = 0
        self._lock = threading.Lock()

    # -- 主接口 -----------------------------------------------------------
    @abstractmethod
    def _complete(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        stream: bool,
        **kwargs: Any,
    ) -> Iterator[str]:
        """底层实现：产出增量文本片段（非流式时一次产出完整文本）。"""
        raise NotImplementedError

    def chat(
        self,
        messages: Sequence[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """同步对话，返回完整回答文本。带重试。"""
        chunks = list(
            self._retrying_stream(
                messages,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                stream=False,
                **kwargs,
            )
        )
        return "".join(chunks).strip()

    def stream(
        self,
        messages: Sequence[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式对话，逐片段产出文本（首字延迟低，前端打字机效果）。"""
        yield from self._retrying_stream(
            messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            stream=True,
            **kwargs,
        )

    def chat_with_usage(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> tuple[str, TokenUsage]:
        """同步对话并返回 Token 用量。"""
        text = self.chat(messages, **kwargs)
        prompt_text = "\n".join(m.content for m in messages)
        return text, estimate_usage(prompt_text, text, self.model)

    def chat_json(
        self,
        messages: Sequence[Message],
        schema_hint: str = "",
        retries: int = 2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """要求模型输出 JSON 并解析。

        容错策略：
        1. 直接 ``json.loads``
        2. 剥离 ```json ``` 代码围栏后重试
        3. 正则抽取第一个平衡的 ``{...}`` 对象
        4. 全部失败则追加纠正提示重新请求（最多 ``retries`` 次）
        """
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt <= retries:
            text = self.chat(messages, **kwargs)
            parsed = extract_json(text)
            if parsed is not None:
                return parsed
            last_error = ValueError(f"模型未返回合法 JSON: {text[:200]}")
            _log.warning("JSON 解析失败（第 %d 次）: %s", attempt + 1, text[:160].replace("\n", " "))
            attempt += 1
            if attempt <= retries:
                hint = schema_hint or "一个合法的 JSON 对象"
                messages = list(messages) + [
                    Message.user(
                        f"你上一次的回复不是合法 JSON。请只输出{hint}，"
                        "不要包含任何解释文字、Markdown 代码围栏或多余内容。"
                    )
                ]
        raise LLMError(str(last_error) if last_error else "JSON 解析失败")

    # -- 重试包装 ---------------------------------------------------------
    def _retrying_stream(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        stream: bool,
        **kwargs: Any,
    ) -> Iterator[str]:
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.max_retries)):
            emitted = False
            try:
                with self._lock:
                    self._call_count += 1
                for piece in self._complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    **kwargs,
                ):
                    emitted = True
                    yield piece
                return
            except Exception as exc:  # noqa: BLE001 - 需要兜住所有网络/解析异常
                last_error = exc
                # 已经吐字了就不再重试（前端已收到片段，重来会出现重复内容）
                if emitted:
                    raise LLMError(f"LLM 流式调用失败: {exc}") from exc
                # 一个 token 都没产出时，流式与非流式的重试代价相同 ——
                # 早期实现写的是 ``if emitted or stream``，等于**流式永不重试**：
                # 网络在首字节前抖动一下就整轮失败，恰好是最该重试的场景。
                if not _is_retryable(exc):
                    raise LLMError(f"LLM 调用失败（不可重试）: {exc}") from exc
                backoff = min(8.0, 0.8 * (2**attempt))
                _log.warning(
                    "LLM 调用失败（第 %d/%d 次），%.1fs 后重试: %s",
                    attempt + 1,
                    self.max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
        raise LLMError(f"LLM 调用失败（已重试 {self.max_retries} 次）: {last_error}")

    # -- 工具 -------------------------------------------------------------
    @property
    def call_count(self) -> int:
        return self._call_count

    def health_check(self, deep: bool = False) -> Dict[str, Any]:
        """探测服务可用性（供 ``/api/health`` 使用）。

        ``deep=False``（默认）**只做配置自检，不发起网络请求**。
        早期实现每次都真的 ``chat("ping")``：健康检查接口被前端定时轮询时，
        这会产生持续的**计费调用**（且每次都要等一个完整的生成往返）。
        需要真实验证连通性时显式传 ``deep=True``。
        """
        if not deep:
            return {
                "ok": True,
                "provider": self.provider,
                "model": self.model,
                "deep": False,
                "note": "仅配置自检（未发起真实调用）；传 deep=True 可做连通性探测。",
            }

        start = time.perf_counter()
        try:
            reply = self.chat(
                [Message.user("ping")], temperature=0.0, max_tokens=8
            )
            return {
                "ok": True,
                "deep": True,
                "provider": self.provider,
                "model": self.model,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "sample": reply[:40],
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "deep": True,
                "provider": self.provider,
                "model": self.model,
                "error": str(exc)[:200],
            }

    def describe_pricing(self) -> Dict[str, Any]:
        """返回当前模型的价格口径（供成本展示/报告标注）。"""
        known = _is_local_model(self.model) or self.model.lower() in _PRICE_TABLE
        return {
            "model": self.model,
            "pricing_known": known,
            "local": _is_local_model(self.model),
            "unit": "元 / 1K tokens",
        }


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中鲁棒地抽取 JSON 对象。

    依次尝试：直接解析 → 去掉代码围栏 → 括号平衡扫描。
    """
    if not text:
        return None
    candidate = text.strip()

    for attempt in (candidate, _strip_code_fence(candidate)):
        if not attempt:
            continue
        try:
            data = json.loads(attempt)
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return {"actions": data}
        except json.JSONDecodeError:
            pass

    # 括号平衡扫描：找第一个完整的 {...}
    start = candidate.find("{")
    while start != -1:
        depth, in_str, escape = 0, False, False
        for idx in range(start, len(candidate)):
            ch = candidate[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = candidate[start : idx + 1]
                    try:
                        data = json.loads(snippet)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        break
        start = candidate.find("{", start + 1)
    return None


def _strip_code_fence(text: str) -> str:
    """剥离 Markdown 代码围栏。"""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text


# ==============================================================================
# Ollama 后端
# ==============================================================================
class OllamaLLM(LLMClient):
    """本地 Ollama 后端（``/api/chat``，支持 stream）。"""

    provider = "ollama"

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        cfg = (config or get_config()).llm
        self.base_url = str(cfg.base_url).rstrip("/")

    def _payload(
        self, messages: Sequence[Message], temperature: float, max_tokens: int, stream: bool
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.to_llm_dict() for m in messages],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens,
            },
        }

    def _complete(
        self, messages, temperature, max_tokens, stream, **kwargs: Any
    ) -> Iterator[str]:
        payload = self._payload(messages, temperature, max_tokens, stream)
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if not stream:
                    body = json.loads(response.read().decode("utf-8"))
                    yield (body.get("message") or {}).get("content", "")
                    return
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (event.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if event.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise LLMError(
                f"无法连接 Ollama({self.base_url})：{exc}。请确认已执行 `ollama serve` "
                f"且已 `ollama pull {self.model}`；或改用 provider=mock 离线模式。"
            ) from exc

    def health_check(self) -> Dict[str, Any]:
        """Ollama 健康检查：优先查 /api/tags，避免真的推理。"""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            return {
                "ok": True,
                "provider": self.provider,
                "model": self.model,
                "base_url": self.base_url,
                "available_models": models,
                "model_loaded": any(self.model.split(":")[0] in m for m in models),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "provider": self.provider, "model": self.model, "error": str(exc)[:200]}


# ==============================================================================
# OpenAI 兼容后端
# ==============================================================================
class OpenAICompatibleLLM(LLMClient):
    """任何 OpenAI 兼容的 ``/v1/chat/completions`` 接口。"""

    provider = "openai"

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        cfg = (config or get_config()).llm
        self.base_url = str(cfg.base_url or "").rstrip("/")
        if not self.base_url.endswith("/v1") and "openai.com" not in self.base_url:
            # 允许用户只填到根域名
            self.base_url = f"{self.base_url}/v1"
        # strip：配置文件里常见的尾随空格/换行会让 Authorization 头非法（401）
        self.api_key = str(cfg.api_key or "").strip() or "sk-no-key"

    def _complete(
        self, messages, temperature, max_tokens, stream, **kwargs: Any
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [m.to_llm_dict() for m in messages],
            "temperature": temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if not stream:
                    body = json.loads(response.read().decode("utf-8"))
                    choices = body.get("choices") or [{}]
                    yield (choices[0].get("message") or {}).get("content", "") or ""
                    return
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (event.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
                # 401/403 是最常见的配置错误（key 缺失/过期），给出可操作提示
                if exc.code in (401, 403):
                    detail += "（请检查 llm.api_key 是否有效，或环境变量 APP_LLM__API_KEY）"
            except Exception:  # noqa: BLE001
                pass
            raise _http_error(f"OpenAI 兼容接口 HTTP {exc.code}: {detail}", exc.code) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"无法连接 LLM 服务({self.base_url}): {exc}") from exc


# ==============================================================================
# DashScope 后端
# ==============================================================================
class DashScopeLLM(LLMClient):
    """阿里云通义千问 DashScope 原生接口。"""

    provider = "dashscope"

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        cfg = (config or get_config()).llm
        # strip 后再判空：只含空白字符的 key 会让下面的 ``if not self.api_key``
        # 误判为"已配置"，直到真的请求时才拿 401
        self.api_key = str(cfg.api_key or "").strip()
        self.base_url = str(cfg.base_url or "")
        if "dashscope" not in self.base_url:
            self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def _complete(
        self, messages, temperature, max_tokens, stream, **kwargs: Any
    ) -> Iterator[str]:
        if not self.api_key:
            raise LLMError("DashScope 需要配置 llm.api_key（或用环境变量 APP_LLM__API_KEY）")
        payload = {
            "model": self.model,
            "input": {"messages": [m.to_llm_dict() for m in messages]},
            "parameters": {
                "temperature": temperature,
                "top_p": self.top_p,
                "max_tokens": max_tokens,
                "result_format": "message",
                "incremental_output": bool(stream),
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if stream:
            headers["X-DashScope-SSE"] = "enable"
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if not stream:
                    body = json.loads(response.read().decode("utf-8"))
                    output = body.get("output") or {}
                    choices = output.get("choices") or []
                    if choices:
                        yield (choices[0].get("message") or {}).get("content", "")
                    else:
                        yield output.get("text", "")
                    return
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    choices = (event.get("output") or {}).get("choices") or []
                    if choices:
                        piece = (choices[0].get("message") or {}).get("content", "")
                        if piece:
                            yield piece
        except urllib.error.HTTPError as exc:
            # 早期实现不捕获：HTTPError 会以原始形态穿透到重试层，
            # 用户看到的是 "HTTP Error 401: Unauthorized" —— 没有任何可操作信息
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
                if exc.code in (401, 403):
                    detail += "（请检查 llm.api_key 是否有效，或环境变量 APP_LLM__API_KEY）"
            except Exception:  # noqa: BLE001
                pass
            raise _http_error(f"DashScope 接口 HTTP {exc.code}: {detail}", exc.code) from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"无法连接 DashScope 服务({self.base_url}): {exc}") from exc


# ==============================================================================
# Mock 后端（离线）
# ==============================================================================
def _clean_context_source(raw: str) -> str:
    """从上下文块的元信息行中还原干净的来源名。

    输入形如 ``Attention Is All You Need p.2 | 相关度 0.93 | 召回 hybrid+rerank``，
    输出 ``Attention Is All You Need p.2``。

    注意：论文标题本身可能含 ``|``（如 "Deep Learning | A Survey"），
    因此只把 **两侧带空格的 " | "** 视为字段分隔符。
    """
    text = raw.strip()
    parts = re.split(r"\s+\|\s+", text)
    return parts[0].strip() if parts else text


class MockLLM(LLMClient):
    """离线模拟后端 —— 零外部依赖，用于开发、单元测试与无 API Key 环境的端到端联调。

    它不是"随机输出"，而是实现了三类确定性启发式，使整条流水线（检索 → 引用 →
    Agent 决策 → 工具调用 → 评测）都能真实跑通并产生**可复现**的结果：

    1. **ReAct 决策模拟**：当 System Prompt 要求输出 JSON 决策时，解析提示词中
       声明的可用工具与 Observation 历史，按规则选择工具（计算 → 计算器；
       元信息 → 元信息工具；对比 → 对比工具；需检索 → 知识库检索），
       信息足够时输出 final_answer。
    2. **RAG 生成模拟**：从注入的上下文片段中按问题相关性抽取句子，
       保留 ``【来源 p.页码】`` 引用标记，产出结构化的"背景-方法-结果"式答案。
    3. **通用问答模拟**：无上下文时返回基于问题的模板化说明。

    这样做的工程价值：CI 与无模型环境下仍可自动化测试全链路，且评测指标可复现。
    """

    provider = "mock"

    # 触发规则路由的关键词表（工具名 -> 关键词）
    _TOOL_KEYWORDS: Dict[str, tuple[str, ...]] = {
        "calculator": ("计算", "算一下", "算算", "相乘", "乘以", "除以", "相除", "相加", "相减",
                       "加上", "减去", "等于", "多少", "+", "-", "*", "/", "×", "÷", "^"),
        "current_time": ("现在几点", "当前时间", "现在时间", "今天几号", "今天日期", "当前日期",
                         "今天是", "星期几", "今年是哪一年"),
        "paper_metadata": ("作者", "谁写", "谁做", "发表", "年份", "哪一年", "期刊", "会议",
                           "DOI", "doi", "标题", "题目"),
        "paper_compare": ("对比", "比较", "相比", "区别", "差异", "不同之处", "哪个更好", "优劣", "vs"),
        "keyword_extract": ("关键词", "关键字", "核心词", "提取词", "术语"),
        "summarize_paper": ("总结", "概括", "摘要", "综述", "讲了什么", "主要讲", "全文大意",
                            "核心内容", "介绍一下这", "介绍这", "主要贡献"),
        "knowledge_search": ("论文", "文献", "知识库", "方法", "实验", "数据集", "结果",
                             "什么是", "如何", "为什么", "怎么", "怎样", "解释", "原理", "机制"),
    }

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

    # -- 决策模拟 ---------------------------------------------------------
    def _available_tools(self, prompt: str, system: str) -> List[str]:
        """从提示词中解析出本轮**实际可用**的工具名单。

        ReAct 的 System Prompt 里每个工具都以 ``### 工具名`` 的形式出现
        （见 :meth:`BaseTool.render_for_prompt`）；而路由建议块会写明
        "本轮最可能需要这些工具"。两者取交集，使得 MockLLM 的行为与真实 LLM
        一样受路由影响 —— 否则离线模式下永远走 knowledge_search，无法验证路由是否生效。
        """
        # 注意：标题格式为 "### tool_name"（有空格），必须用 \s+ 而非直接拼接
        catalog = set(re.findall(r"^#{1,4}\s+([a-z_][a-z0-9_]*)\s*$", system, re.MULTILINE))
        catalog |= set(re.findall(r"^#{1,4}\s+([a-z_][a-z0-9_]*)\s*$", prompt, re.MULTILINE))
        if not catalog:
            catalog = set(self._TOOL_KEYWORDS.keys())

        # 路由建议：优先使用被显式建议的工具
        hint = re.search(
            r"最可能需要这些工具[：:]\s*([^\n。]+)", prompt + "\n" + system
        )
        if hint:
            suggested = [
                token.strip()
                for token in re.split(r"[、,，\s]+", hint.group(1))
                if token.strip() in catalog
            ]
            if suggested:
                return suggested
        return sorted(catalog)

    def _simulate_decision(self, prompt: str, system: str) -> Dict[str, Any]:
        """解析提示词中的工具清单与已有观察，决定下一步动作。"""
        tool_names = self._available_tools(prompt, system)

        observations = re.findall(r"Observation\s*[:：]\s*(.*)", prompt)
        question = self._extract_question(prompt)
        lowered = question.lower()
        has_observation = bool(observations) or "Observation" in prompt or "观察结果" in prompt

        # 已有观察：信息足够就收尾
        if has_observation:
            return {
                "thought": "已获得工具返回的信息，信息足以回答用户问题，可以给出最终答案。",
                "action": "final_answer",
                "action_input": {"answer": self._compose_from_observations(question, observations)},
            }

        # 按问题类型挑选工具，且只在**可用工具**范围内选择，否则退回知识库检索
        wants_compare = any(k in question for k in self._TOOL_KEYWORDS["paper_compare"])
        wants_meta = any(k in question for k in self._TOOL_KEYWORDS["paper_metadata"])
        wants_calc = bool(re.search(r"\d+\s*[\+\-\*/×÷\*]\s*\d+", question)) or any(
            k in question for k in self._TOOL_KEYWORDS["calculator"]
        )
        wants_time = any(k in question for k in self._TOOL_KEYWORDS["current_time"])
        wants_search = any(k in question for k in self._TOOL_KEYWORDS["knowledge_search"])

        def choose(preferred: str, fallback: str = "knowledge_search") -> str:
            if preferred in tool_names:
                return preferred
            return fallback if fallback in tool_names else (tool_names[0] if tool_names else "knowledge_search")

        if wants_compare and "paper_compare" in tool_names:
            return {
                "thought": "用户要求对比多篇论文，调用论文对比工具获取多维度证据。",
                "action": "paper_compare",
                "action_input": {"papers": self._guess_paper_refs(question), "aspects": ["方法", "数据集", "结果"]},
            }
        if wants_time and not wants_search:
            return {
                "thought": "用户询问当前时间，直接调用时间工具，无需检索知识库。",
                "action": choose("current_time"),
                "action_input": {},
            }
        if wants_calc and not wants_search:
            expression = self._extract_expression(question)
            return {
                "thought": "这是纯数学计算问题，使用计算器工具即可，无需检索知识库。",
                "action": choose("calculator"),
                "action_input": {"expression": expression},
            }
        if wants_meta and not wants_calc:
            return {
                "thought": "用户询问论文的作者/年份/DOI 等元信息，调用论文元信息提取工具。",
                "action": choose("paper_metadata"),
                "action_input": {"query": question},
            }
        if "summarize_paper" in tool_names and any(
            k in question for k in self._TOOL_KEYWORDS["summarize_paper"]
        ):
            return {
                "thought": "用户要求生成论文结构化摘要，调用摘要生成工具。",
                "action": "summarize_paper",
                "action_input": {"query": question},
            }
        if "keyword_extract" in tool_names and any(
            k in question for k in self._TOOL_KEYWORDS["keyword_extract"]
        ):
            return {
                "thought": "用户要求提取关键词，调用关键词提取工具。",
                "action": "keyword_extract",
                "action_input": {"query": question},
            }
        return {
            "thought": "该问题涉及知识库中的论文内容，需要先检索相关资料再回答。",
            "action": choose("knowledge_search"),
            "action_input": {"query": question, "top_k": 5},
        }

    @staticmethod
    def _extract_expression(question: str) -> str:
        """从中文问题里抽出数学表达式。

        支持"3.14 乘以 2.56"→"3.14*2.56"、"100 除以 4"→"100/4"、
        "(28.4-26.4)/26.4*100"（原样表达式）。
        """
        text = question
        # 已是表达式
        if re.fullmatch(r"[\d\.\s\+\-\*/%\(\)]+", text.strip()):
            return text.strip()
        # 中文运算词 → 运算符
        normalized = (
            text.replace("乘以", "*").replace("乘", "*")
            .replace("除以", "/").replace("除", "/")
            .replace("加上", "+").replace("加", "+")
            .replace("减去", "-").replace("减", "-")
            .replace("×", "*").replace("÷", "/")
        )
        match = re.search(r"[\d\.\s\+\-\*/%\(\)]{3,}", normalized)
        if match:
            expression = match.group(0).strip()
            expression = re.sub(r"\s+", "", expression).strip("+-*/")
            if re.search(r"\d", expression) and re.search(r"[\+\-\*/%]", expression):
                return expression
        return "1+1"

    @staticmethod
    def _guess_paper_refs(question: str) -> List[str]:
        """从对比类问题里猜出论文标识（英文标题片段 / 中文书名号内容）。"""
        refs: List[str] = []
        # 「论文A」「"XXX"」《XXX》
        for match in re.finditer(r"[《“\"]([^》”\"]{2,60})[》”\"]", question):
            refs.append(match.group(1).strip())
        # 英文大写开头的连续词（可能是论文名）
        for match in re.finditer(r"\b([A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,5})\b", question):
            candidate = match.group(1).strip()
            if len(candidate) > 3 and candidate.lower() not in ("the", "and", "for"):
                refs.append(candidate)
        # 中文"XX 和 YY"结构
        if not refs:
            for match in re.finditer(r"([\u4e00-\u9fff]{2,10})\s*(?:和|与|跟)\s*([\u4e00-\u9fff]{2,10})", question):
                refs.extend([match.group(1), match.group(2)])
        # 去重保序
        unique: List[str] = []
        for ref in refs:
            if ref not in unique:
                unique.append(ref)
        return unique[:4] or ["论文"]

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """从提示词中抽取用户问题原文。

        提示词里可能有多个候选行（系统指令、路由建议、"请输出 JSON"等），
        因此按以下优先级取值：

        1. 显式标记的问题行（``## 用户问题`` / ``用户问题：`` / ``Question:``）
        2. 形如 ``## xxx`` 的小节标题下的首行内容
        3. 兜底：最后一个"看起来像自然语言问题"的行
        """
        # 1) 显式标记
        patterns = [
            r"##\s*用户问题\s*\n\s*(.+)",
            r"用户问题\s*[:：]\s*(.+)",
            r"##\s*待核验问题\s*\n\s*(.+)",
            r"##\s*目标论文/问题\s*\n\s*(.+)",
            r"##\s*Question\s*\n\s*(.+)",
            r"Question\s*[:：]\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                candidate = match.group(1).strip().split("\n")[0].strip()
                if candidate:
                    return candidate[:300]

        # 2) 兜底：从后往前找第一个"不像指令"的行
        instruction_marks = (
            "请输出", "只输出", "JSON", "json", "要求", "格式", "工具", "字段说明",
            "决策规范", "停止条件", "输出格式", "##", "===", "```", "说明：",
        )
        for line in reversed([ln.strip() for ln in prompt.splitlines() if ln.strip()]):
            if len(line) < 4 or any(mark in line for mark in instruction_marks):
                continue
            return line[:300]
        return "请介绍一下你的能力"

    def _compose_from_observations(self, question: str, observations: List[str]) -> str:
        """把 Observation 内容整理成最终答案（保留引用标记）。"""
        body = observations[-1].strip() if observations else ""
        if not body:
            return self._compose_general(question)
        # Observation 已经是工具返回的结构化文本 → 直接作为答案主体
        if body.startswith(("检索「", "共匹配到", "当前时间", "知识库中", "以下", "根据", "联网搜索")):
            return body
        return f"根据工具返回的结果，回答如下：\n\n{body}"

    # -- 生成模拟 ---------------------------------------------------------
    def _parse_context_blocks(self, prompt: str) -> List[tuple[str, str, str]]:
        """解析提示词中的【参考资料】区块。

        上下文块格式（由 ``generation.prompt_template.format_context_block`` 产出）::

            [1] 来源：Attention Is All You Need p.4 | 相关度 0.87 | 召回 hybrid+rerank
            正文第一行
            正文第二行

        返回 ``[(index, source, body), ...]``。

        注意：必须只取"来源"行**之后**的正文 —— 元信息行里含有"相关度 0.87"
        这类数字与"引用"字样，若被当成正文，会污染抽取出的答案句子。
        """
        # 1) 截取参考资料区块
        start = -1
        for marker in ("## 参考资料", "参考资料"):
            position = prompt.find(marker)
            if position != -1:
                start = position + len(marker)
                break
        if start == -1:
            return []
        section = prompt[start:]
        next_heading = section.find("\n## ")
        if next_heading != -1:
            section = section[:next_heading]

        # 2) 按 [N] 分块，只取"来源："行之后的正文
        blocks: List[tuple[str, str, str]] = []
        pattern = re.compile(
            r"\[(\d+)\]\s*来源[：:]\s*(?P<source>[^\n]+?)\s*\n"
            r"(?P<body>.*?)(?=\n\[\d+\]\s*来源|\Z)",
            re.DOTALL,
        )
        for match in pattern.finditer(section):
            source = _clean_context_source(match.group("source"))
            body = match.group("body").strip()
            if body:
                blocks.append((match.group(1), source, body))

        # 3) 兼容形如 【来源 p.N】正文 的简写格式
        if not blocks:
            for idx, match in enumerate(
                re.finditer(
                    r"【(?P<source>[^】]{2,80}?)】\s*\n(?P<body>.*?)(?=\n【|\Z)",
                    section,
                    re.DOTALL,
                ),
                start=1,
            ):
                body = match.group("body").strip()
                if body:
                    blocks.append((str(idx), match.group("source").strip(), body))
        return blocks

    def _compose_from_context(self, question: str, prompt: str) -> str:
        """从注入的 RAG 上下文中抽取相关句子，产出带引用的答案。"""
        blocks = self._parse_context_blocks(prompt)
        if not blocks:
            return self._compose_general(question)

        # 问题关键词（中文词/英文字母串），过滤常见疑问词
        raw = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", question)
        stop = {
            "the", "and", "for", "are", "was", "were", "what", "which", "how", "does",
            "did", "use", "used", "with", "that", "this", "from", "its", "have", "has",
            "关于", "什么", "哪些", "怎么", "如何", "多少", "请问", "介绍", "一下", "论文",
        }
        keywords = [k.lower() for k in raw if k.lower() not in stop] or [k.lower() for k in raw]

        scored: List[tuple[float, str, str]] = []
        for _index, source, body in blocks:
            for sentence in re.split(r"(?<=[。！？；!?;])\s*|\n", body):
                sentence = sentence.strip()
                if len(sentence) < 12:
                    continue
                overlap = sum(1 for kw in keywords if kw in sentence.lower())
                if overlap:
                    scored.append((overlap / max(1, len(keywords)), source, sentence))
        scored.sort(key=lambda x: (-x[0], -len(x[2])))
        picked = scored[:5]

        if not picked:
            _index, source, body = blocks[0]
            head = re.sub(r"\s+", " ", body).strip()[:260]
            return (
                f"知识库中检索到的《{source}》相关内容如下：\n\n> {head}\n\n"
                f"（离线模拟模式：该结论未经大模型推理，仅用于验证检索与引用链路。）"
            )

        lines = ["根据知识库检索结果，可以回答如下：", ""]
        for score, source, sentence in picked:
            lines.append(f"- {sentence}（匹配度 {score:.0%}）【{source}】")
        lines.append("")
        lines.append(f"以上内容摘自检索到的 {len(blocks)} 个文档片段中相关度最高的部分。")
        return "\n".join(lines)

    def _compose_general(self, question: str) -> str:
        return (
            f"【离线模拟模式】当前 LLM provider 配置为 mock，未接入真实大模型。\n\n"
            f"收到的问题：{question}\n\n"
            "链路验证信息：检索层、Agent 决策层、工具调用与前端渲染均已正常执行。"
            "如需真实模型回答，请在 config.yaml 中把 llm.provider 改为 ollama 或 openai，"
            "或设置环境变量 APP_LLM__PROVIDER=ollama。"
        )

    # -- 摘要模拟 ---------------------------------------------------------
    def _compose_summary(self, prompt: str) -> str:
        """对话摘要压缩模拟：抽取对话中的关键问题与结论要点。"""
        questions = re.findall(r"用户\s*[:：]\s*(.+)", prompt)
        topics: List[str] = []
        for question in questions:
            cleaned = re.sub(r"[？?。，,、]", " ", question).strip()
            if cleaned:
                topics.append(cleaned[:40])
        if not topics:
            return "本轮对话尚未形成明确主题。"
        unique = list(dict.fromkeys(topics))[-6:]
        return "历史对话要点：" + "；".join(f"用户曾询问「{t}」" for t in unique) + "。"

    # -- 主入口 -----------------------------------------------------------
    def _complete(
        self, messages, temperature, max_tokens, stream, **kwargs: Any
    ) -> Iterator[str]:
        system = next((m.content for m in messages if m.role == Role.SYSTEM), "")
        prompt = "\n".join(m.content for m in messages if m.role != Role.SYSTEM)
        mode = kwargs.get("mock_mode", "")

        if mode == "summary" or "对话摘要" in system or "总结成一段" in prompt:
            text = self._compose_summary(prompt)
        elif mode == "react" or "action_input" in system or "ReAct" in system:
            decision = self._simulate_decision(prompt, system)
            text = json.dumps(decision, ensure_ascii=False)
        elif "上下文" in system or "【" in prompt or "[1] 来源" in prompt or "检索" in system:
            text = self._compose_from_context(self._extract_question(prompt), prompt)
        else:
            text = self._compose_general(self._extract_question(prompt))

        if stream:
            # 按标点切片，模拟真实流式的打字机节奏
            for piece in re.findall(r"[^。！？；\n]*[。！？；\n]?", text):
                if piece:
                    yield piece
        else:
            yield text

    def health_check(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "provider": "mock",
            "model": self.model,
            "mode": "offline-simulation",
            "note": "离线模拟后端：不调用任何外部服务，输出为确定性启发式结果。",
        }


# ==============================================================================
# 工厂
# ==============================================================================
_REGISTRY = {
    "ollama": OllamaLLM,
    "openai": OpenAICompatibleLLM,
    "dashscope": DashScopeLLM,
    "mock": MockLLM,
}


def create_llm(config: Optional[Config] = None, provider: Optional[str] = None) -> LLMClient:
    """按配置创建 LLM 客户端。

    Parameters
    ----------
    config: 全局配置对象（None 则使用单例）
    provider: 显式指定 provider，覆盖配置值（便于 A/B 对比实验）
    """
    cfg = config or get_config()
    name = (provider or str(cfg.llm.provider) or "mock").lower()
    client_cls = _REGISTRY.get(name)
    if client_cls is None:
        _log.warning("未知的 llm.provider=%s，回退到 mock 离线后端", name)
        client_cls = MockLLM
    client = client_cls(cfg)
    _log.info("LLM 客户端已创建: provider=%s model=%s", client.provider, client.model)
    return client
