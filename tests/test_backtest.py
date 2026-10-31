"""
backtest.py 验收测试（docs/P0_BACKTEST_DESIGN.md §5）。

三条核心契约：
  1. 手算净值能对上
  2. 泄漏因子应被抓出极高 Sharpe（证明回测确实在抓信号）
  3. shift 与不 shift 结果必须显著不同（证明时序纪律生效）
"""

import numpy as np
import pandas as pd
import pytest

from autoalpha.backtest import (
    build_quantile_weights, compute_nav, evaluate_portfolio,
    evaluate_layers, nav_correlation, DEFAULT_COST_BPS,
)


@pytest.fixture
def toy_price():
    """3 标的 × 6 日，收益可手算。"""
    dates = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame({
        "A": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051],   # 每日 +10%
        "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],      # 不动
        "C": [100.0,  90.0,  81.0,  72.9,  65.61,  59.049],   # 每日 -10%
    }, index=dates)


def test_nav_matches_hand_calculation(toy_price):
    """全仓 A、零成本 → 净值必须等于 A 的累计收益（含 shift 导致的首日空仓）。"""
    w = pd.DataFrame(0.0, index=toy_price.index, columns=toy_price.columns)
    w["A"] = 1.0
    g, n, turnover = compute_nav(w, toy_price, cost_bps=0.0)

    # shift(1) → 首日空仓，之后每日 +10%，共 5 个收益日
    assert g.iloc[0] == pytest.approx(1.0)
    assert g.iloc[-1] == pytest.approx(1.1 ** 5, rel=1e-9)
    assert n.iloc[-1] == pytest.approx(g.iloc[-1], rel=1e-9)   # 零成本时两者相同
    assert turnover == pytest.approx(1.0 / len(toy_price))     # 仅首日建仓


def test_cost_reduces_nav(toy_price):
    """成本必须真的扣减净值。"""
    w = pd.DataFrame(0.0, index=toy_price.index, columns=toy_price.columns)
    w["A"] = 1.0
    g, n, _ = compute_nav(w, toy_price, cost_bps=100.0)        # 1% 单边
    assert n.iloc[-1] < g.iloc[-1]


def test_shift_discipline_changes_result():
    """
    时序纪律生效证明：把权重人为前移一天（模拟漏掉 shift），结果必须不同。
    若相同，说明 compute_nav 没有正确滞后权重。

    注意：此测试必须用【非均匀】收益序列。若标的每日收益恒定，
    "哪一天持有" 不影响结果，测试就无法区分 shift 与不 shift。
    """
    dates = pd.bdate_range("2024-01-01", periods=6)
    px = pd.DataFrame({
        "A": [100.0, 105.0, 103.0, 130.0, 128.0, 131.0],   # 日收益差异显著
        "B": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    }, index=dates)

    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.iloc[2, w.columns.get_loc("A")] = 1.0                # 只在第3天定权重

    g_correct, _, _ = compute_nav(w, px, cost_bps=0.0)
    g_leaky, _, _ = compute_nav(w.shift(-1).fillna(0.0), px, cost_bps=0.0)

    # 正确：t=2 定权重 → 吃 t=3 收益 (103→130, +26.2%)
    assert g_correct.iloc[-1] == pytest.approx(130.0 / 103.0, rel=1e-9)
    # 漏 shift：吃 t=2 收益 (105→103, -1.9%)
    assert g_leaky.iloc[-1] == pytest.approx(103.0 / 105.0, rel=1e-9)
    assert not np.isclose(g_correct.iloc[-1], g_leaky.iloc[-1])


def test_leaky_factor_is_detected():
    """
    泄漏因子（用未来收益当因子）必须被回测抓出极高 Sharpe。
    这验证回测有能力识别信号 —— 若此测试失败，说明回测根本没在算收益。
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-01", periods=300)
    syms = [f"S{i:02d}" for i in range(20)]
    ret = pd.DataFrame(rng.normal(0, 0.015, (len(dates), len(syms))),
                       index=dates, columns=syms)
    price = (1 + ret).cumprod() * 100

    # 作弊：因子 = 次日收益
    fwd = ret.shift(-1)
    factor = fwd.stack()
    factor.index.names = ["date", "symbol"]

    lm, _ = evaluate_layers(factor, price, n_quantiles=5,
                            rebalance="D", cost_bps=0.0)
    assert lm.long_short.sharpe > 3.0, f"泄漏因子未被抓出: {lm.long_short.sharpe}"
    assert lm.monotonicity > 0.8, f"泄漏因子应完美单调: {lm.monotonicity}"


def test_random_factor_has_no_edge():
    """纯随机因子的多空 Sharpe 应接近 0 —— 证明回测不会无中生有。"""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-01", periods=400)
    syms = [f"S{i:02d}" for i in range(20)]
    ret = pd.DataFrame(rng.normal(0, 0.015, (len(dates), len(syms))),
                       index=dates, columns=syms)
    price = (1 + ret).cumprod() * 100

    noise = pd.DataFrame(rng.normal(0, 1, (len(dates), len(syms))),
                         index=dates, columns=syms)
    factor = noise.stack()
    factor.index.names = ["date", "symbol"]

    lm, _ = evaluate_layers(factor, price, n_quantiles=5, cost_bps=0.0)
    assert abs(lm.long_short.sharpe) < 1.5, f"随机因子不应有边际: {lm.long_short.sharpe}"


def test_quantile_weights_sum_to_one(toy_price):
    """每层权重在调仓后应和为 1（层内等权）。"""
    dates = pd.bdate_range("2024-01-01", periods=40)
    syms = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.normal(0, 1, (len(dates), len(syms))),
                     index=dates, columns=syms).stack()
    f.index.names = ["date", "symbol"]

    layers = build_quantile_weights(f, n_quantiles=5, rebalance="W-FRI")
    for q, df in layers.items():
        s = df.sum(axis=1)
        nonzero = s[s > 1e-9]
        assert len(nonzero) > 0, f"层 {q} 全空"
        assert np.allclose(nonzero.values, 1.0, atol=1e-9)


def test_only_available_symbols_used():
    """分位切分只能用当日非 NaN 的标的（早期年份宽度小，不得把 NaN 计入分层）。"""
    dates = pd.bdate_range("2024-01-01", periods=20)
    syms = [f"S{i}" for i in range(10)]
    wide = pd.DataFrame(1.0, index=dates, columns=syms)
    wide.iloc[:, 5:] = np.nan                                  # 后 5 个标的全缺
    wide.iloc[:, :5] = np.arange(5, dtype=float)
    f = wide.stack(); f.index.names = ["date", "symbol"]

    layers = build_quantile_weights(f, n_quantiles=2, rebalance="D")
    for q, df in layers.items():
        assert df[syms[5:]].abs().to_numpy().sum() == 0.0, "缺失标的被计入分层"


def test_nav_correlation_identifies_duplicates():
    """同族因子的净值相关必须高 —— 这是修 D4 的核心契约。"""
    dates = pd.bdate_range("2022-01-01", periods=200)
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    nav_a = (1 + base).cumprod()
    nav_b = (1 + base * 1.02 + rng.normal(0, 0.0004, len(dates))).cumprod()
    nav_c = (1 + pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)).cumprod()

    assert nav_correlation(nav_a, nav_b) > 0.9, "同族因子未被识别"
    assert abs(nav_correlation(nav_a, nav_c)) < 0.3, "无关因子被误判为同族"
