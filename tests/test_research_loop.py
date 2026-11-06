"""
研究循环：反思、轮次记账、状态重置。

守三条线：
  1. 下一轮方向必须从失败原因推出，不能是泛泛的「再试一次」；
  2. 记账无论反思成败都要留下 —— 选择性偏差不能静默累积；
  3. 新一轮要重置结果状态，但保留轮次与累计试验数，且 IS/OOS 切分不变。
"""

import pytest

from autoalpha.agents.research_loop_agent import (
    ResearchLoopAgent, _pattern_of,
)


class _Ev:
    def __init__(self, name, status, reason=""):
        self.factor_name = name
        self.status = status
        self.fail_reason = reason


class TestFailurePatterns:
    """失败模式分类是反思的输入，归错了下一轮方向就会错。"""

    def test_data_unavailable(self):
        e = _Ev("a", "SKIP_NO_DATA", "需要本数据集没有的数据：未采集宏观")
        assert _pattern_of(e.status, e.fail_reason) == "data_unavailable"

    def test_turnover(self):
        assert _pattern_of("FAIL_IS", "换手率=0.85 超过上限") == "turnover"

    def test_monotonicity(self):
        assert _pattern_of("FAIL_IS", "单调性=0.500 < 0.6") == "monotonicity"

    def test_sharpe(self):
        assert _pattern_of("FAIL_IS", "多空Sharpe=0.211 < 0.5") == "weak_sharpe"

    def test_implementation(self):
        assert _pattern_of("FAIL_IS", "实现失败(3次): ERROR: Traceback") == \
            "implementation"


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_user = ""

    def complete(self, system="", user="", purpose=""):
        self.last_user = user
        return self.payload


class TestReflect:
    def _evals(self):
        return [
            _Ev("ts_momentum", "SKIP_NO_DATA", "需要宏观序列，未采集"),
            _Ev("xs_momentum", "FAIL_IS", "多空Sharpe=0.211 < 0.5"),
        ]

    def test_returns_next_direction(self):
        llm = _FakeLLM(
            '{"learned":"动量在本池不显著","next_direction_zh":"低波动异象",'
            '"canonical_en":"low volatility anomaly equity",'
            '"avoid":["宏观数据"],"rationale":"动量失效需换机制"}')
        ag = ResearchLoopAgent(llm)
        out = ag.reflect(None, self._evals(), 1, 4)
        assert out["canonical_en"] == "low volatility anomaly equity"
        assert out["avoid"] == ["宏观数据"]

    def test_prompt_contains_failure_reasons(self):
        """下一轮方向只能从失败原因推出 —— prompt 必须真的带上它们。"""
        llm = _FakeLLM('{"canonical_en":"x y z"}')
        ag = ResearchLoopAgent(llm)
        ag.reflect(None, self._evals(), 1, 4)
        assert "Sharpe" in llm.last_user or "sharpe" in llm.last_user
        assert "data_unavailable" in llm.last_user
        assert "OHLCV" in llm.last_user, "必须告知数据边界，否则下轮又会要宏观数据"

    def test_parse_failure_is_visible(self):
        ag = ResearchLoopAgent(_FakeLLM("not json at all"))
        assert ag.reflect(None, self._evals(), 1, 4) is None
        assert ag.last_error, "失败必须留下可读原因，不能静默"

    def test_llm_failure_is_visible(self):
        class _Boom:
            def complete(self, **kw):
                raise RuntimeError("network down")
        ag = ResearchLoopAgent(_Boom())
        assert ag.reflect(None, self._evals(), 1, 4) is None
        assert "network" in ag.last_error

    def test_empty_evaluations(self):
        ag = ResearchLoopAgent(_FakeLLM('{"canonical_en":"x"}'))
        assert ag.reflect(None, [], 1, 4) is None

    def test_empty_query_rejected(self):
        ag = ResearchLoopAgent(_FakeLLM('{"learned":"x"}'))
        assert ag.reflect(None, self._evals(), 1, 4) is None
