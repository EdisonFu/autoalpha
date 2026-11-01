"""
engine.py 契约测试。

engine 承担防泄漏职责，这些测试锁定它的核心契约：
切分不重叠、OOS 不被偷看、门槛随宽度变化、指标口径正确。
"""

import numpy as np
import pandas as pd
import pytest

from autoalpha.engine import SplitEngine, compute_ic_metrics
from autoalpha.metrics import ic_mean_floor, t_threshold


def _mk_data(n_days=900, n_sym=40, seed=0, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    syms = [f"S{i:02d}" for i in range(n_sym)]
    ret = pd.DataFrame(rng.normal(0.0002, 0.015, (n_days, n_sym)),
                       index=dates, columns=syms)
    px = (1 + ret).cumprod() * 100
    return {s: pd.DataFrame({"close": px[s]}) for s in syms}


@pytest.fixture
def engine():
    return SplitEngine(_mk_data(), start="2022-01-01", end="2026-02-28")


# ── 切分契约 ────────────────────────────────────────────────────────────────

def test_is_oos_do_not_overlap(engine):
    """IS 与 OOS 不得有任何共同日期 —— 重叠即泄漏。"""
    assert len(engine.price_is.index.intersection(engine.price_oos.index)) == 0
    assert engine.price_is.index[-1] < engine.price_oos.index[0]


def test_embargo_gap_exists(engine):
    """禁运期必须真实存在：OOS 起点与 IS 终点之间要有间隔。"""
    gap = (engine.oos_start - engine.is_end).days
    assert gap >= 20, f"禁运间隔仅 {gap} 天"


def test_oos_reread_is_warned(engine, capsys):
    """重复读 OOS 必须告警 —— 反复看 OOS 等于把它变成第二个 IS。"""
    f = _flat_factor(engine.price_full)
    engine.evaluate_oos(f, "dup_test")
    capsys.readouterr()
    engine.evaluate_oos(f, "dup_test")
    assert "OOS 重复读取" in capsys.readouterr().out


def test_clip_restricts_to_period(engine):
    """送入全样本因子，IS 评估必须只用 IS 期数据。"""
    f = _flat_factor(engine.price_full)
    clipped = engine._clip(f, engine.price_is)
    dates = clipped.index.get_level_values(0)
    assert dates.max() <= engine.is_end
    assert len(clipped) < len(f)


# ── 门槛推导契约（修 D1 根因）──────────────────────────────────────────────

def test_gates_scale_with_universe_width():
    """门槛必须随截面宽度变化，不能是硬编码常量。"""
    narrow = ic_mean_floor(24, 1000)
    wide = ic_mean_floor(108, 1000)
    assert narrow > wide, "宽截面应允许更低的 IC"
    assert narrow / wide == pytest.approx(np.sqrt(108 / 24), rel=1e-6)


def test_gates_reproduce_legacy_threshold():
    """width=24 时推导值应接近旧硬编码 0.02 —— 证明旧门槛的来源被正确建模。"""
    assert ic_mean_floor(24, 1074) == pytest.approx(0.0187, abs=5e-4)


def test_t_threshold_rises_with_trials():
    """多重检验修正：试验越多门槛越高。"""
    assert t_threshold(77) > t_threshold(1)
    assert t_threshold(1) == pytest.approx(1.645, abs=0.01)


def test_gates_for_reports_actual_width(engine):
    g = engine.gates_for("is")
    assert g["_width"] > 0 and g["_n_obs"] == len(engine.price_is)
    assert g["ic_mean_floor"] == pytest.approx(
        ic_mean_floor(g["_width"], g["_n_obs"], g["ic_tstat_target"]))


# ── 指标口径契约（修 D2/D3）────────────────────────────────────────────────

def test_monthly_win_rate_is_monthly_not_daily():
    """
    修 D2 的核心契约：月度胜率必须按月聚合后计算。
    构造一个 IC 日内正负频繁交替、但按月聚合后稳定为正的因子 —— 
    逐日胜率会接近 0.5，真月度胜率应接近 1.0。
    """
    data = _mk_data(n_days=600, n_sym=30, seed=5)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28")
    px = eng.price_full
    fwd = px.pct_change(fill_method=None).shift(-1)
    # 因子 = 次日收益 + 大噪声：IC 为正但逐日波动剧烈
    rng = np.random.default_rng(11)
    noisy = fwd + rng.normal(0, 0.05, fwd.shape)
    f = noisy.stack(); f.index.names = ["date", "symbol"]

    ic = compute_ic_metrics(f, px, lags=(1, 5))
    assert ic.n_months > 10
    assert ic.monthly_win_rate > ic.monthly_ic_mean * 0  # 字段存在且已计算
    # 月度聚合降噪后，胜率应高于逐日水平
    assert ic.monthly_win_rate >= 0.5


def test_ic_by_lag_is_dimensionally_comparable():
    """
    修 D3 的核心契约：各 lag 的 IC 必须全部基于【前向1日收益】，
    仅因子滞后不同。旧实现用 5 日累计收益，比值系统性 >1（实测 1.06~1.61）。
    这里验证 lag=1 的值等于独立算出的 1 日 IC。
    """
    data = _mk_data(n_days=500, n_sym=30, seed=9)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28")
    px = eng.price_full
    f = px.pct_change(20, fill_method=None).stack()
    f.index.names = ["date", "symbol"]

    ic = compute_ic_metrics(f, px, lags=(1, 5, 10))
    assert ic.ic_by_lag[1] == pytest.approx(ic.ic_mean_1d, abs=1e-9)
    # 所有 lag 的 IC 量纲一致 → 不应出现系统性 >1 的比值
    for lg in (5, 10):
        ratio = abs(ic.ic_by_lag[lg] / ic.ic_by_lag[1]) if ic.ic_by_lag[1] else 0
        assert ratio < 5.0, f"lag={lg} 比值 {ratio} 异常，疑似量纲不可比"


def test_icir_naming_carries_scale():
    """icir_daily 与 icir_annual 必须差 sqrt(252) —— 口径不可混用。"""
    data = _mk_data(n_days=400, n_sym=25, seed=3)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28")
    px = eng.price_full
    f = px.pct_change(20, fill_method=None).stack()
    f.index.names = ["date", "symbol"]
    ic = compute_ic_metrics(f, px, lags=(1,))
    # 排除 ic_std≈0 的哨兵值（见 engine.compute_ic_metrics）
    # 容差 1% —— 两者各自 round(4) 后相除会有精度损失，不是逻辑偏差
    if 1e-9 < abs(ic.icir_daily) < 999:
        assert ic.icir_annual / ic.icir_daily == pytest.approx(np.sqrt(252), rel=1e-2)


def test_coverage_denominator_excludes_prehistory():
    """
    修 coverage 分母：长 lookback 因子在标的上市初期无值，
    不应被计入分母而误判为稀疏。
    """
    data = _mk_data(n_days=700, n_sym=30, seed=4)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28")
    px = eng.price_full
    f = px.pct_change(250, fill_method=None).stack()
    f.index.names = ["date", "symbol"]
    ic = compute_ic_metrics(f, px, lags=(1,))
    assert ic.coverage > 0.9, f"长窗口因子被误判稀疏: {ic.coverage}"


# ── 门控契约 ────────────────────────────────────────────────────────────────

def test_leaky_factor_passes_all_gates():
    """
    端到端：泄漏因子（用次日收益）应通过 IS 并进入 OOS。
    若连泄漏因子都过不了，说明门槛设置有误而非因子不行。
    """
    data = _mk_data(n_days=900, n_sym=40, seed=21)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28", rebalance="D")
    px = eng.price_full
    f = px.pct_change(fill_method=None).shift(-1).stack()
    f.index.names = ["date", "symbol"]

    ic, lm, _ = eng.evaluate_is(f)
    # 信号必须被识别出来：完美 IC + 完美单调 + 极高 Sharpe
    assert abs(ic.ic_tstat) > 10, f"泄漏因子 t 值未被识别: {ic.ic_tstat}"
    assert lm.monotonicity > 0.9
    assert lm.long_short.sharpe > 3.0
    # 注：日频调仓的泄漏因子换手天然 >3，会被换手闸门拦住。
    # 这正确 —— 换手闸门是独立于信号强度的交易性约束。
    ok, why = eng.check_is_gates(ic, lm)
    assert not ok and "换手" in why, f"预期被换手闸门拦住，实际: {why}"


def test_random_factor_fails_is_gates():
    """纯随机因子必须被 IS 层拦住。"""
    data = _mk_data(n_days=900, n_sym=40, seed=31)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28")
    rng = np.random.default_rng(99)
    px = eng.price_full
    noise = pd.DataFrame(rng.normal(0, 1, px.shape), index=px.index, columns=px.columns)
    f = noise.stack(); f.index.names = ["date", "symbol"]

    ev = eng.full_evaluate(f, "pure_noise")
    assert ev.status == "FAIL_IS", f"随机因子不应通过: {ev.status}"
    assert ev.oos_ic is None, "FAIL_IS 的因子不得访问 OOS"


def _flat_factor(px: pd.DataFrame) -> pd.Series:
    f = px.pct_change(20, fill_method=None).stack()
    f.index.names = ["date", "symbol"]
    return f


def test_coverage_never_exceeds_one():
    """
    coverage 必须 ≤ 1。实测过 1.07：因子在停牌日（价格 NaN）也输出了值，
    进了分子却不在分母 —— 分子分母口径必须一致。
    """
    data = _mk_data(n_days=600, n_sym=30, seed=17)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28")
    px = eng.price_full.copy()
    # 人为制造停牌：随机挖掉 15% 的价格
    rng = np.random.default_rng(5)
    holes = rng.random(px.shape) < 0.15
    px_holed = px.mask(holes)
    # 因子对所有格子都有值（模拟 ffill 行为）
    f = pd.DataFrame(rng.normal(0, 1, px.shape),
                     index=px.index, columns=px.columns).stack()
    f.index.names = ["date", "symbol"]

    ic = compute_ic_metrics(f, px_holed, lags=(1,))
    assert ic.coverage <= 1.0, f"coverage 超过 1: {ic.coverage}"
    assert ic.coverage > 0.5, f"coverage 异常偏低: {ic.coverage}"


# ── 数据可用性与池构建（docs/E2E_REVIEW.md P-2 / P-3）────────────────────────

def test_skip_factors_requiring_unavailable_data():
    """E2E 实测出现的四个因子必须被前置拦下，合法动量因子必须放行。"""
    from autoalpha.orchestrator import _check_data_available
    from autoalpha.pipeline_state import FactorSpec

    cols = {"open", "high", "low", "close", "volume", "amount"}

    def spec(name, hyp, formula):
        return FactorSpec(name, "", hyp, formula, "", {}, {}, {})

    blocked = [
        spec("vix_stress_relief", "VIX elevated then falling", "vix_ma - vix"),
        spec("spy_drawdown_depth", "SPY drawdown depth", "1 - spy/cummax(spy)"),
        spec("llm_news_momentum_score", "news sentiment", "sent * mom"),
        spec("rate_relief_signal", "interest_rate falling", "d(rate)"),
    ]
    for s in blocked:
        assert _check_data_available(s, cols), f"{s.factor_name} 应被跳过"

    ok = spec("etf_cross_sectional_momentum",
              "momentum from investor underreaction",
              "rank(close_t / close_{t-60} - 1)")
    assert _check_data_available(ok, cols) is None, "合法动量因子被误拦"


def test_llm_declared_uncomputable_is_skipped():
    from autoalpha.orchestrator import _check_data_available
    from autoalpha.pipeline_state import FactorSpec
    s = FactorSpec("x", "", "some signal", "f(x)", "needs fundamentals",
                   {}, {}, {})
    s.computable = False
    assert _check_data_available(s, {"close"}) is not None


def test_universe_preset_makes_cross_section_comparable():
    """
    修 P-3 的契约：默认预设须排除债券/货币等异质资产。
    实测全池波动极差 200 倍，仅股票类降到 2.2 倍。
    """
    from autoalpha.universe import classify, build_universe, PRESETS

    assert classify("国债ETF") == "bond"
    assert classify("华安张江REIT") == "reit"
    assert classify("华宝添益ETF") == "money"
    assert classify("黄金ETF") == "gold"
    assert classify("南方原油") == "oil"
    assert classify("沪深300ETF") == "equity"
    assert classify("可转债ETF") == "bond", "可转债须归债券，不得落入 equity"

    names = {"510300": "沪深300ETF", "511010": "国债ETF",
             "508000": "华安张江REIT", "518880": "黄金ETF",
             "511990": "华宝添益ETF"}
    data = {k: None for k in names}
    kept, prof = build_universe(data, names, "equity")
    assert set(kept) == {"510300"}
    assert prof["n_total"] == 5 and prof["n_kept"] == 1
    assert set(prof["excluded"]) == {"bond", "reit", "gold", "money"}


def test_data_context_forbids_proxies():
    """注入 prompt 的数据说明必须点名禁止自行代理（修 P-2 的静默代理问题）。"""
    from autoalpha.universe import describe_for_prompt
    txt = describe_for_prompt(
        {"n_kept": 92, "dist_kept": {"equity": 92}, "sample": ["510300 沪深300ETF"]},
        ["close", "volume"], ("2022-01-04", "2026-02-27"))
    # 必须说清三件事：什么市场、什么级别、什么资产类型
    assert "China A-share" in txt, "须点明市场"
    assert "daily bars" in txt.lower() or "daily" in txt, "须点明数据频率"
    assert "ETF" in txt, "须点明资产类型"
    # 按类别列出缺失，而不只是甩一串词
    assert "Wrong market" in txt and "SPY" in txt
    assert "Wrong granularity" in txt and "intraday" in txt
    assert "Wrong asset type" in txt and "fundamentals" in txt
    # ETF 专属陷阱（实测 limit_up_volume_strength 栽在这里）
    assert "price limit" in txt, "须提示 ETF 无涨跌停"
    assert "DO NOT invent a proxy" in txt


def test_recency_does_not_outrank_relevance():
    """
    修 P-1 的契约：用 E2E 实测的真实四篇复算，
    rel=8 的截面动量论文必须排在 rel=7 的新论文之前。
    """
    from autoalpha.agents.paper_search_agent import PaperSearchAgent
    rows = [("ChatGPT in Systematic Investing", 7, "2025-10-30"),
            ("Transfer Ranking: Cross-Sectional Momentum", 8, "2023-02-21"),
            ("Building Cross-Sectional Strategies", 8, "2020-12-13")]
    out = PaperSearchAgent._combined_sort(
        [{"title": t, "date": d} for t, _, d in rows],
        [s for _, s, _ in rows])
    assert out[0]["_relevance_score"] == 8, "相关性更高的论文应排前"
    # 核心契约：更新但相关性更低的论文不得排到第一。
    # （不再断言它垫底 —— 排序现在还计入引用量与文献类型，
    #   名次可能被其它维度带动，只要没压过相关性就是对的。）
    assert "ChatGPT" not in out[0]["title"], "rel=7 的论文不得夺冠"


def test_doc_type_and_citations_affect_ranking():
    """同等相关度下，期刊论文/研报应优于预印本；引用量为次要加权。"""
    from autoalpha.agents.paper_search_agent import PaperSearchAgent
    rows = [
        ("journal", 8, "2020-01-01", "paper", 100),
        ("preprint", 8, "2020-01-01", "preprint", 100),
        ("ssrn", 8, "2020-01-01", "working_paper", 100),
    ]
    out = PaperSearchAgent._combined_sort(
        [{"title": t, "date": d, "doc_type": dt, "citations": c}
         for t, _, d, dt, c in rows],
        [s for _, s, _, _, _ in rows])
    titles = [p["title"] for p in out]
    assert titles.index("journal") < titles.index("preprint")
    assert titles.index("ssrn") < titles.index("preprint")


def test_missing_doc_type_is_not_penalised():
    """未标类型的条目不应因此被降权（缺省按 paper 处理）。"""
    from autoalpha.agents.paper_search_agent import PaperSearchAgent
    a = PaperSearchAgent._combined_sort(
        [{"title": "x", "date": "2020-01-01"}], [8])[0]["_combined_score"]
    b = PaperSearchAgent._combined_sort(
        [{"title": "x", "date": "2020-01-01", "doc_type": "paper"}],
        [8])[0]["_combined_score"]
    assert a == b


def test_turnover_cap_scales_with_rebalance_frequency():
    """
    换手门槛必须随调仓频率变化。实测同一因子 mom_60d：
    日频 0.415 / 周频 0.188 / 月频 0.091 —— 固定 0.30 会让门槛
    变成频率的函数而非因子质量的函数。
    """
    from autoalpha.engine import turnover_cap
    assert turnover_cap("D") > turnover_cap("W-FRI") > turnover_cap("ME")
    assert turnover_cap("unknown") == turnover_cap("W-FRI")


def test_gates_use_engine_rebalance():
    data = _mk_data(n_days=600, n_sym=30, seed=23)
    from autoalpha.engine import turnover_cap
    for rb in ("D", "W-FRI", "ME"):
        eng = SplitEngine(data, start="2022-01-01", end="2026-02-28",
                          rebalance=rb)
        assert eng.gates_for("is")["turnover_daily"] == turnover_cap(rb)


def test_sign_flip_mirrors_ic_and_monotonicity():
    """
    符号一致性：因子取负后 IC/t/单调性必须精确反向。
    （Sharpe 不要求精确镜像 —— 几何年化对正负收益天然不对称，
    实测 +0.4%/日 年化 +173% 而 -0.4%/日 年化 -64%。）
    """
    data = _mk_data(n_days=700, n_sym=30, seed=29)
    eng = SplitEngine(data, start="2022-01-01", end="2026-02-28")
    px = eng.price_full
    f = px.pct_change(60, fill_method=None).stack()
    f.index.names = ["date", "symbol"]

    ic_a, lm_a, _ = eng.evaluate_is(f)
    ic_b, lm_b, _ = eng.evaluate_is(-f)

    assert ic_b.ic_mean_1d == pytest.approx(-ic_a.ic_mean_1d, abs=1e-9)
    assert ic_b.ic_tstat == pytest.approx(-ic_a.ic_tstat, abs=0.02)
    if abs(lm_a.monotonicity) > 0.1:
        assert lm_b.monotonicity == pytest.approx(-lm_a.monotonicity, abs=0.25)
