"""
输出预算与截断可见性。

守的是一个很贵的教训：推理模型（DeepSeek v4.x / o-series / thinking 类）
把思维链也算进 completion 预算，预算耗尽时返回**空字符串**而非异常。
实测 deepseek-v4.1-flash 在 max_tokens=1200 下 reasoning 吃满 1200、
content 0 字节、finish_reason=length。下游只看到「JSON 解析失败」
或「提取到 0 个因子」，真因完全不可见。
"""

import os

import pytest

from autoalpha.llm import (
    OutputTruncatedError, _is_reasoning_model, max_tokens_for,
)


class TestReasoningDetection:
    def test_known_reasoning_models(self):
        for m in ["deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-pro",
                  "deepseek-reasoner", "moonshotai/kimi-k2-thinking",
                  "o3-mini", "qwq-32b"]:
            assert _is_reasoning_model(m), m

    def test_non_reasoning_models(self):
        for m in ["gpt-4o", "claude-sonnet-4", "z-ai/glm-5.3", "qwen3-max"]:
            assert not _is_reasoning_model(m), m


class TestBudget:
    def test_reasoning_gets_larger_budget(self):
        """这正是回归的根因：8192 够摘要，不够全文+思维链。"""
        assert (max_tokens_for("deepseek/deepseek-v4.1-flash")
                > max_tokens_for("gpt-4o"))

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
        assert max_tokens_for("deepseek/deepseek-v4.1-flash") == 4096
        assert max_tokens_for("gpt-4o") == 4096

    def test_invalid_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_TOKENS", "not-a-number")
        assert max_tokens_for("gpt-4o") == 8192

    def test_zero_override_ignored(self, monkeypatch):
        monkeypatch.setenv("LLM_MAX_TOKENS", "0")
        assert max_tokens_for("gpt-4o") == 8192


class TestTruncationVisible:
    """截断必须抛，不能把空串交给下游。"""

    def _client(self, finish_reason, content, reasoning_tokens=0):
        from autoalpha.llm import LLMClient

        class _Det:
            def __init__(self, rt): self.reasoning_tokens = rt

        class _U:
            prompt_tokens = 100
            completion_tokens = 1200
            def __init__(self, rt): self.completion_tokens_details = _Det(rt)

        class _Msg:
            def __init__(self, c): self.content = c

        class _Choice:
            def __init__(self, fr, c):
                self.finish_reason = fr
                self.message = _Msg(c)

        class _Resp:
            def __init__(self, fr, c, rt):
                self.choices = [_Choice(fr, c)]
                self.usage = _U(rt)

        class _Completions:
            def create(self, **kw):
                return _Resp(finish_reason, content, reasoning_tokens)

        class _Chat:
            completions = _Completions()

        class _Inner:
            chat = _Chat()

        c = LLMClient.__new__(LLMClient)
        c.provider = "openai"
        c.model = "deepseek/deepseek-v4.1-flash"
        c.client = _Inner()
        from autoalpha.llm import TokenMeter
        c.meter = TokenMeter()
        return c

    def test_truncation_raises_not_empty_string(self):
        c = self._client("length", "", reasoning_tokens=1200)
        with pytest.raises(OutputTruncatedError) as e:
            c.complete(user="x", purpose="extract")
        msg = str(e.value)
        assert "extract" in msg, "要说清哪个阶段被截断"
        assert "1200" in msg, "要报出思维链占用，否则看不出是推理模型的问题"
        assert "LLM_MAX_TOKENS" in msg, "要给出可操作的解法"

    def test_normal_completion_returns_text(self):
        c = self._client("stop", '[{"factor_name":"x"}]')
        assert c.complete(user="x") == '[{"factor_name":"x"}]'

    def test_partial_content_still_raises(self):
        """截断时即使有部分正文也要抛 —— 残缺 JSON 比空串更难排查。"""
        c = self._client("length", '[{"factor_name":"x"', reasoning_tokens=800)
        with pytest.raises(OutputTruncatedError):
            c.complete(user="x", purpose="extract")
