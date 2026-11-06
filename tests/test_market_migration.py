"""
跨市场迁移：美股研究应迁到 A股，而不是被判不可计算。

实测过两类误杀：
  1. LLM 遇到 VIX 就自己标 computable=false（已实现波动率本可替代）；
  2. 已经迁移好的因子（名字就叫 realized_vol_scaled_momentum），
     因 description 里「论文用 VIX」被字符串扫描拒掉。

约束有三条，缺一不可：
  - 声明了迁移的因子，引用原文市场数据不算数据依赖（豁免文本扫描）；
  - required_fields 的硬检查不受豁免 —— 声明迁移不能绕过真实的数据边界；
  - 未声明迁移却提到美股词的仍然拒绝，防止「给 VIX 编一个代理变量」。
"""

import pytest

from autoalpha.orchestrator import _check_data_available
from autoalpha.pipeline_state import FactorSpec

AVAIL = {"open", "high", "low", "close", "volume", "amount"}


def _spec(**kw):
    base = dict(factor_name="f", description="", hypothesis="h",
                formula="x", implementation_notes="",
                data_requirements={}, parameters={}, source_paper={})
    base.update(kw)
    return FactorSpec(**base)


class TestMigratedFactors:
    def test_vix_reference_is_not_a_dependency_when_migrated(self):
        """迁移版在描述里引用原文的 VIX 是正常的，不应被拒。"""
        s = _spec(factor_name="realized_vol_scaled_momentum",
                  description="论文用 VIX 缩放，本版用21日已实现波动率",
                  migration={"source_market": "us",
                             "changes": ["VIX → 21日年化已实现波动率"]})
        assert _check_data_available(s, AVAIL) is None

    def test_us_index_reference_allowed_after_migration(self):
        s = _spec(description="SPY / S&P 500 成分股 → A股ETF截面",
                  migration={"source_market": "us", "changes": ["SPY → ETF"]})
        assert _check_data_available(s, AVAIL) is None

    def test_required_fields_hard_check_survives_migration(self):
        """声明迁移不能绕过真实的数据边界：需要新闻文本就是做不了。"""
        s = _spec(required_fields=["news"],
                  migration={"source_market": "us",
                             "changes": ["VIX → 已实现波动率"]})
        verdict = _check_data_available(s, AVAIL)
        assert verdict is not None
        assert "news" in verdict

    def test_llm_self_declared_uncomputable_respected(self):
        s = _spec(computable=False,
                  migration={"source_market": "us", "changes": ["c"]})
        assert _check_data_available(s, AVAIL) is not None


class TestUnmigratedStillBlocked:
    def test_vix_without_migration_rejected(self):
        """防止「一边说可计算，一边给 VIX 编代理变量」—— 这个坑要守住。"""
        s = _spec(factor_name="vix_scaled_mom", description="uses VIX > 25")
        verdict = _check_data_available(s, AVAIL)
        assert verdict is not None
        assert "vix" in verdict

    def test_rejection_suggests_migration(self):
        """拒绝理由应提示「可以迁移」，而不是只丢下一句做不了。"""
        s = _spec(description="SPY momentum")
        verdict = _check_data_available(s, AVAIL)
        assert "迁移" in verdict

    def test_empty_migration_changes_not_exempted(self):
        """migration 字段存在但 changes 为空，不能算已声明迁移。"""
        s = _spec(description="uses VIX", migration={"source_market": "us"})
        assert _check_data_available(s, AVAIL) is not None


class TestExtractProducesMigration:
    def test_migration_field_parsed(self, monkeypatch):
        import json
        from autoalpha.agents.factor_extract_agent import FactorExtractAgent

        class _Fake:
            def complete(self, **kw):
                return json.dumps([{
                    "factor_name": "mom", "description": "d", "hypothesis": "h",
                    "formula": "f", "implementation_notes": "n",
                    "parameters": {}, "computable": True,
                    "required_fields": ["close"],
                    "migration": {"source_market": "us",
                                  "changes": ["VIX → 已实现波动率"]}}])
        ag = FactorExtractAgent(_Fake())
        specs = ag.extract({"title": "US study", "abstract": "a" * 200,
                            "pdf_url": "https://doi.org/10.1/x"})
        assert specs[0].migration["source_market"] == "us"
        assert "VIX" in specs[0].migration["changes"][0]
        # 声明了迁移后，即使描述里含美股词也应通过检查
        assert _check_data_available(specs[0], AVAIL) is None

    def test_missing_migration_field_defaults_empty(self):
        from autoalpha.pipeline_state import FactorSpec
        s = FactorSpec(factor_name="x", description="", hypothesis="",
                       formula="", implementation_notes="",
                       data_requirements={}, parameters={}, source_paper={})
        assert s.migration == {}
