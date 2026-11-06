"""
轮次记账与状态重置。

核心约束：选择性偏差必须可见。跑了 4 轮、试了 40 个因子之后，
「3 个通过门控」的含义和第一轮就跑出 3 个完全不同。所以：
  - 每一轮的试验次数都要留下，无论反思成功与否；
  - 新一轮清空结果但保留累计计数；
  - 达到轮次上限要明说，不能无限循环。
"""

import pytest

from autoalpha.flow import FlowSession


class _Ev:
    def __init__(self, name, status, reason=""):
        self.factor_name = name
        self.status = status
        self.fail_reason = reason


def _drain(gen):
    return list(gen)


class TestAccounting:
    def _orch(self, monkeypatch, proposal=None):
        from autoalpha.orchestrator import FlowOrchestrator as Orchestrator
        o = Orchestrator.__new__(Orchestrator)
        o.paper_search = type("PS", (), {"_results_cache": {}, "intent": None})()

        class _Loop:
            last_error = "mock 失败"
            def reflect(self, *a, **kw):
                return proposal
        o.research_loop = _Loop()
        return o

    def test_round_recorded_even_when_reflect_fails(self, monkeypatch):
        """反思失败也得记账，否则失败轮次的试验次数会凭空消失。"""
        o = self._orch(monkeypatch, proposal=None)
        sess = FlowSession(idea="x")
        sess.evaluations = [_Ev("a", "FAIL_IS", "Sharpe 低"),
                            _Ev("b", "PASS")]
        events = _drain(o.reflect_round(sess))
        assert len(sess.rounds) == 1
        assert sess.rounds[0]["n_evals"] == 2
        assert sess.rounds[0]["n_pass"] == 1
        assert sess.n_factor_trials == 2
        # 失败要可见
        assert any("反思失败" in (e.message or "") for e in events)

    def test_trials_accumulate_across_rounds(self, monkeypatch):
        o = self._orch(monkeypatch, proposal=None)
        sess = FlowSession(idea="x")
        for i in range(3):
            sess.round_no = i + 1
            sess.evaluations = [_Ev(f"f{i}", "FAIL_IS", "x")]
            _drain(o.reflect_round(sess))
        assert sess.n_factor_trials == 3
        assert len(sess.rounds) == 3

    def test_max_rounds_stops_loop(self, monkeypatch):
        o = self._orch(monkeypatch,
                       proposal={"canonical_en": "should not be used"})
        sess = FlowSession(idea="x")
        sess.round_no = o.MAX_ROUNDS
        sess.evaluations = [_Ev("a", "FAIL_IS", "x")]
        events = _drain(o.reflect_round(sess))
        assert sess.next_proposal is None
        joined = " ".join(e.message or "" for e in events)
        assert "轮次上限" in joined
        assert "选择性偏差" in joined, "到上限时必须提示选择性偏差"

    def test_proposal_offered_before_limit(self, monkeypatch):
        prop = {"canonical_en": "low volatility anomaly",
                "next_direction_zh": "低波动", "avoid": ["宏观"]}
        o = self._orch(monkeypatch, proposal=prop)
        sess = FlowSession(idea="x")
        sess.evaluations = [_Ev("a", "PASS")]
        events = _drain(o.reflect_round(sess))
        assert sess.next_proposal == prop
        last = events[-1]
        assert last.payload.get("next") == "next_round"
        assert last.payload.get("can_extend") is True
        assert last.payload.get("n_factor_trials") == 1


class TestNextRoundReset:
    def test_resets_results_but_keeps_accounting(self, monkeypatch):
        from autoalpha.orchestrator import FlowOrchestrator as Orchestrator
        from autoalpha.agents.intent_agent import SearchIntent

        o = Orchestrator.__new__(Orchestrator)
        o.paper_search = type(
            "PS", (), {"_results_cache": {"old": [1]}, "intent": None})()
        # 数据与切分复用，不应被重置
        o._engine = object()
        searched = []
        o.search_papers = lambda sess, idea: iter(())

        sess = FlowSession(idea="x")
        sess.intent = SearchIntent(canonical_en="old query",
                                   aliases=["old alias"], market="cn")
        sess.round_no = 1
        sess.n_factor_trials = 7
        sess.rounds = [{"round": 1, "n_evals": 7}]
        sess.papers = [{"title": "p"}]
        sess.specs = [object()]
        sess.evaluations = [_Ev("a", "PASS")]
        sess.next_proposal = {"canonical_en": "new query",
                              "next_direction_zh": "新方向",
                              "avoid": ["宏观数据"]}

        _drain(o.start_next_round(sess))

        assert sess.round_no == 2
        assert sess.n_factor_trials == 7, "累计试验数不能清零"
        assert sess.rounds == [{"round": 1, "n_evals": 7}], "历史轮次要保留"
        assert sess.papers == [] and sess.specs == [] and sess.evaluations == []
        assert sess.intent.canonical_en == "new query"
        assert sess.intent.market == "cn", "市场设定要继承"
        assert o._engine is not None, "IS/OOS 切分必须复用，否则跨轮不可比"
        assert o.paper_search._results_cache == {}, "检索缓存必须清，否则拿到旧结果"

    def test_no_proposal_is_noop(self):
        from autoalpha.orchestrator import FlowOrchestrator as Orchestrator
        o = Orchestrator.__new__(Orchestrator)
        sess = FlowSession(idea="x")
        sess.next_proposal = None
        events = _drain(o.start_next_round(sess))
        assert sess.round_no == 1, "没有方向时不应推进轮次"
        assert any("没有下一轮方向" in (e.message or "") for e in events)
