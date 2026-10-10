# -*- coding: utf-8 -*-
"""LLM 接入层单测：重试分类、流式重试、健康检查、计量口径。

全部离线：用一个可控的桩后端（``_StubLLM``）模拟各种失败，
不发起任何真实网络请求。``time.sleep`` 被替换成记录器，
因此重试用例是"瞬时"的，也不会因为退避策略调整而变成慢测试。
"""

from __future__ import annotations

import unittest
import urllib.error
from typing import Any, Dict, List, Optional, Sequence

from tests.common import TempWorkspace

from src.llm import (
    LLMClient,
    LLMError,
    OpenAICompatibleLLM,
    DashScopeLLM,
    _is_retryable,
    estimate_usage,
)
from src.utils.types import Message, TokenUsage


# ---------------------------------------------------------------------------
# 桩后端
# ---------------------------------------------------------------------------
class _StubLLM(LLMClient):
    """按脚本产出或抛错的后端。

    ``script`` 里每一项的含义：
    ``Exception``            → 本次调用直接抛错；
    ``[str, ...]``           → 本次调用产出的增量片段；
    ``([str, ...], Exception)`` → 先产出片段，再抛错（模拟流中途断开）。
    """

    provider = "stub"

    def __init__(self, config: Any = None, script: Optional[Sequence[Any]] = None) -> None:
        super().__init__(config)
        self.script: List[Any] = list(script or [])
        self.calls: List[Dict[str, Any]] = []

    def _complete(  # type: ignore[override]
        self, messages, temperature, max_tokens, stream, **kwargs: Any
    ):
        self.calls.append({"stream": stream, "temperature": temperature, "max_tokens": max_tokens})
        step = self.script.pop(0) if self.script else ["ok"]
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, tuple):
            pieces, error = step
            for piece in pieces:
                yield piece
            raise error
        for piece in step:
            yield piece


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "err", None, None)


class _SleepRecorder:
    """替换 ``src.llm.time.sleep``，记录退避时长而不真的等待。"""

    def __init__(self) -> None:
        self.calls: List[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)

    def __enter__(self) -> "_SleepRecorder":
        import src.llm as module

        self._module = module
        self._original = module.time.sleep
        module.time.sleep = self  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._module.time.sleep = self._original  # type: ignore[assignment]
        return False


class RetryPolicyTest(unittest.TestCase):
    """重试分类：瞬时故障重试，配置类错误立即失败。"""

    def setUp(self) -> None:
        self.ws = TempWorkspace()
        self.cfg = self.ws.config(llm={"provider": "mock", "max_retries": 3})

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_transient_5xx_is_retried_then_succeeds(self) -> None:
        llm = _StubLLM(self.cfg, script=[_http_error(503), ["答案"]])
        with _SleepRecorder() as sleeper:
            text = llm.chat([Message.user("问题")])
        self.assertEqual(text, "答案")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(sleeper.calls), 1)  # 只退避一次

    def test_auth_failure_is_not_retried(self) -> None:
        """401 重试三次也没有意义 —— 应当立刻失败，且不做退避等待。"""
        llm = _StubLLM(self.cfg, script=[_http_error(401), ["不该被调用"]])
        with _SleepRecorder() as sleeper:
            with self.assertRaises(LLMError) as ctx:
                llm.chat([Message.user("问题")])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(sleeper.calls, [])
        self.assertIn("不可重试", str(ctx.exception))

    def test_bad_request_is_not_retried(self) -> None:
        llm = _StubLLM(self.cfg, script=[_http_error(400), ["不该被调用"]])
        with _SleepRecorder() as sleeper:
            with self.assertRaises(LLMError):
                llm.chat([Message.user("问题")])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(sleeper.calls, [])

    def test_status_code_attribute_is_honoured(self) -> None:
        """厂商 SDK 抛出的自定义异常靠 status_code 传递状态码。"""
        error = LLMError("服务端错误")
        error.status_code = 502  # type: ignore[attr-defined]
        self.assertTrue(_is_retryable(error))

        fatal = LLMError("模型不存在")
        fatal.status_code = 404  # type: ignore[attr-defined]
        self.assertFalse(_is_retryable(fatal))

    def test_wrapped_http_error_is_retried(self) -> None:
        """异常被包装后（__cause__ 链）仍要能识别出可重试。"""
        cause = _http_error(429)
        wrapped = LLMError("OpenAI 兼容接口 HTTP 429: rate limited")
        wrapped.__cause__ = cause
        self.assertTrue(_is_retryable(wrapped))

    def test_exhausted_retries_surface_last_error(self) -> None:
        llm = _StubLLM(self.cfg, script=[_http_error(503), _http_error(503), _http_error(503)])
        with _SleepRecorder() as sleeper:
            with self.assertRaises(LLMError) as ctx:
                llm.chat([Message.user("问题")])
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len(sleeper.calls), 3)
        self.assertIn("已重试", str(ctx.exception))

    def test_stream_retries_when_nothing_was_emitted(self) -> None:
        """一个 token 都没产出时，流式同样应当重试。

        回归：早期条件是 ``if emitted or stream``，等于"流式永不重试"，
        首字节前的网络抖动会直接让整轮问答失败。
        """
        llm = _StubLLM(self.cfg, script=[urllib.error.URLError("connection reset"), ["流", "式"]])
        with _SleepRecorder() as sleeper:
            chunks = list(llm.stream([Message.user("问题")]))
        self.assertEqual(chunks, ["流", "式"])
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(sleeper.calls), 1)

    def test_stream_does_not_retry_after_first_token(self) -> None:
        """已经吐字后绝不重试（否则前端会收到重复内容）。"""
        llm = _StubLLM(self.cfg, script=[(["半句话"], _http_error(503)), ["不该被调用"]])
        with _SleepRecorder() as sleeper:
            with self.assertRaises(LLMError):
                list(llm.stream([Message.user("问题")]))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(sleeper.calls, [])


class HealthCheckTest(unittest.TestCase):
    """健康检查默认不产生计费调用。"""

    def setUp(self) -> None:
        self.ws = TempWorkspace()
        self.cfg = self.ws.config(llm={"provider": "mock", "max_retries": 1})

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_default_health_check_does_not_call_the_model(self) -> None:
        llm = _StubLLM(self.cfg, script=[["ping"]] * 5)
        health = llm.health_check()
        self.assertTrue(health["ok"])
        self.assertFalse(health["deep"])
        self.assertEqual(llm.calls, [])  # 没有任何一次真实调用

    def test_deep_health_check_does_call_the_model(self) -> None:
        llm = _StubLLM(self.cfg, script=[["pong"]])
        health = llm.health_check(deep=True)
        self.assertTrue(health["ok"])
        self.assertTrue(health["deep"])
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("latency_ms", health)

    def test_deep_health_check_reports_failure(self) -> None:
        llm = _StubLLM(self.cfg, script=[_http_error(401)])
        health = llm.health_check(deep=True)
        self.assertFalse(health["ok"])
        self.assertIn("error", health)


class ApiKeyTest(unittest.TestCase):
    """API Key 的空白处理与错误提示。"""

    def setUp(self) -> None:
        self.ws = TempWorkspace()

    def tearDown(self) -> None:
        self.ws.cleanup()

    def test_openai_key_is_stripped(self) -> None:
        cfg = self.ws.config(llm={"provider": "openai", "api_key": "  sk-abc  "})
        self.assertEqual(OpenAICompatibleLLM(cfg).api_key, "sk-abc")

    def test_openai_blank_key_falls_back_to_placeholder(self) -> None:
        cfg = self.ws.config(llm={"provider": "openai", "api_key": "   \n"})
        self.assertEqual(OpenAICompatibleLLM(cfg).api_key, "sk-no-key")

    def test_dashscope_blank_key_is_rejected_with_actionable_message(self) -> None:
        """只含空白字符的 key 必须被当成"没配置"，而不是等到 401。"""
        cfg = self.ws.config(llm={"provider": "dashscope", "api_key": "  "})
        llm = DashScopeLLM(cfg)
        self.assertEqual(llm.api_key, "")
        with self.assertRaises(LLMError) as ctx:
            list(llm.stream([Message.user("问题")]))
        self.assertIn("api_key", str(ctx.exception))


class PricingTest(unittest.TestCase):
    """成本口径：未知模型绝不给出凭空金额。"""

    def test_known_cloud_model_is_priced(self) -> None:
        usage = estimate_usage("问" * 100, "答" * 100, "deepseek-chat")
        self.assertTrue(usage.pricing_known)
        self.assertGreater(usage.cost_yuan, 0.0)

    def test_local_model_is_priced_by_electricity(self) -> None:
        usage = estimate_usage("问" * 100, "答" * 100, "qwen2.5:7b")
        self.assertTrue(usage.pricing_known)
        self.assertGreater(usage.cost_yuan, 0.0)
        # 本地口径远低于云端：同样的 token 数，成本应至少低一个数量级
        cloud = estimate_usage("问" * 100, "答" * 100, "deepseek-chat")
        self.assertLess(usage.cost_yuan, cloud.cost_yuan / 10)

    def test_unknown_model_is_flagged_instead_of_charged_electricity(self) -> None:
        """回归：未收录的云端模型此前被按"本地电费"计价（低报数量级）。"""
        usage = estimate_usage("问" * 100, "答" * 100, "gpt-5-ultra-experimental")
        self.assertFalse(usage.pricing_known)
        self.assertEqual(usage.cost_yuan, 0.0)
        # token 数照常统计，只有金额被标记为"未收录"
        self.assertGreater(usage.total_tokens, 0)

    def test_blank_model_is_treated_as_local(self) -> None:
        """离线评测不传模型名时按本地口径（此时金额本就是等效电费）。"""
        usage = estimate_usage("问" * 100, "答" * 100, "")
        self.assertTrue(usage.pricing_known)

    def test_pricing_known_propagates_through_addition(self) -> None:
        unknown = estimate_usage("问" * 10, "答" * 10, "unknown-model-xyz")
        known = estimate_usage("问" * 10, "答" * 10, "deepseek-chat")
        self.assertFalse((unknown + known).pricing_known)
        total = TokenUsage(cost_yuan=1.0)
        total.add(unknown)
        self.assertFalse(total.pricing_known)


if __name__ == "__main__":
    unittest.main()
