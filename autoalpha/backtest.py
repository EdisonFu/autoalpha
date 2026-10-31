"""
日频分层回测层。

设计依据 docs/P0_BACKTEST_DESIGN.md §2.3。核心时序纪律：
    t 日收盘算因子 → t 日收盘后定权重 → t+1 日收益才归属该权重
即 ret_contrib[t+1] = weights[t] · ret[t+1]，权重必须 shift(1)。
违反这一条会产生前瞻偏差，而前瞻偏差扛得住任何统计修正
（[2608.27734] 实测：泄漏 oracle 打出 Sharpe 35，完整通过 DSR 与 PBO）。
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import PortfolioMetrics, LayerMetrics, TRADING_DAYS

DEFAULT_COST_BPS = 5.0          # ETF 单边 5bp
DEFAULT_N_QUANTILES = 5
DEFAULT_REBALANCE = "W-FRI"     # 周频；"D" 换手过高不推荐


# ── 因子 → 分层权重 ──────────────────────────────────────────────────────────

def build_quantile_weights(
    factor: pd.Series,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    rebalance: str = DEFAULT_REBALANCE,
) -> dict:
    """
    按日对因子做截面分位切分，返回 {layer_idx: 权重 DataFrame(date×symbol)}。

    关键细节（实测踩过的坑）：
      1. 分位切分只用【当日实际可用】的标的 —— 否则早期年份（宽度 1~20）分层退化
      2. 调仓日之间权重 ffill，非调仓日不交易 —— 这才产生真实换手
      3. 层内等权。IC 加权留到 P1（它会把 IC 噪声带进权重）
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise ValueError("factor 需要 MultiIndex (date, symbol)")

    wide = factor.unstack(level=-1).sort_index()          # date × symbol
    all_dates = wide.index

    # 调仓日：每个 rebalance 周期内的最后一个交易日
    if rebalance.upper() == "D":
        rb_dates = all_dates
    else:
        grp = pd.Series(all_dates, index=all_dates).groupby(
            pd.Grouper(freq=rebalance)
        ).max().dropna()
        rb_dates = pd.DatetimeIndex(grp.values)
        rb_dates = rb_dates.intersection(all_dates)

    layers = {q: pd.DataFrame(0.0, index=all_dates, columns=wide.columns)
              for q in range(n_quantiles)}

    for dt in rb_dates:
        row = wide.loc[dt].dropna()                       # 只用当日可用标的
        if len(row) < n_quantiles * 2:                    # 每层至少 2 个标的
            continue
        # rank → 均匀分位；ties 用 'first' 避免并列把某层撑爆
        ranks = row.rank(method="first", ascending=True)
        bins = np.ceil(ranks / len(row) * n_quantiles).astype(int) - 1
        bins = bins.clip(0, n_quantiles - 1)
        for q in range(n_quantiles):
            members = bins[bins == q].index
            if len(members):
                layers[q].loc[dt, members] = 1.0 / len(members)

    # 调仓日之间持仓不变
    for q in range(n_quantiles):
        df = layers[q]
        mask = df.index.isin(rb_dates)
        df[~mask] = np.nan
        layers[q] = df.ffill().fillna(0.0)

    return layers


# ── 权重 → 净值 ──────────────────────────────────────────────────────────────

def compute_nav(
    weights: pd.DataFrame,
    price: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Tuple[pd.Series, pd.Series, float]:
    """
    由权重和价格算净值。返回 (nav_gross, nav_net, turnover_daily)。

    时序纪律：weights.shift(1) —— t 日定的权重吃 t+1 日收益。
    成本：换仓日扣 cost_bps * 1e-4 * Σ|Δw|
    """
    px = price.reindex(index=weights.index, columns=weights.columns).sort_index()
    ret = px.pct_change(fill_method=None)

    w_eff = weights.shift(1).fillna(0.0)                  # ← 防前瞻的关键
    # 停牌/缺价的标的收益视为 0，不把 NaN 传染给组合
    port_ret = (w_eff * ret.fillna(0.0)).sum(axis=1)

    dw = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    turnover_daily = float(dw.mean())
    cost = dw.shift(1).fillna(0.0) * cost_bps * 1e-4      # 成本与权重同步滞后

    nav_gross = (1.0 + port_ret).cumprod()
    nav_net = (1.0 + port_ret - cost).cumprod()
    return nav_gross, nav_net, turnover_daily


def evaluate_portfolio(
    nav_gross: pd.Series,
    nav_net: pd.Series,
    turnover_daily: float,
    cost_bps: float = DEFAULT_COST_BPS,
) -> PortfolioMetrics:
    """年化/Sharpe/MDD/Calmar。自用场景不减无风险利率（做相对比较）。"""
    def _ann(nav: pd.Series) -> Tuple[float, float, float]:
        r = nav.pct_change(fill_method=None).dropna()
        if len(r) < 2 or r.std() < 1e-12:
            return 0.0, 0.0, 0.0
        ann_ret = float(nav.iloc[-1] ** (TRADING_DAYS / len(r)) - 1.0) \
            if nav.iloc[-1] > 0 else -1.0
        ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
        return ann_ret, ann_vol, sharpe

    ann_ret, ann_vol, sharpe = _ann(nav_gross)
    ann_ret_net, _, sharpe_net = _ann(nav_net)

    dd = nav_net / nav_net.cummax() - 1.0
    mdd = float(dd.min()) if len(dd) else 0.0
    calmar = ann_ret_net / abs(mdd) if abs(mdd) > 1e-9 else 0.0

    return PortfolioMetrics(
        ann_return=round(ann_ret, 5), ann_vol=round(ann_vol, 5),
        sharpe=round(sharpe, 4), max_drawdown=round(mdd, 5),
        calmar=round(calmar, 4), turnover_daily=round(turnover_daily, 5),
        ann_return_net=round(ann_ret_net, 5), sharpe_net=round(sharpe_net, 4),
        cost_bps=cost_bps, n_days=len(nav_net),
    )


# ── 完整分层评估 ─────────────────────────────────────────────────────────────

def evaluate_layers(
    factor: pd.Series,
    price: pd.DataFrame,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    rebalance: str = DEFAULT_REBALANCE,
    cost_bps: float = DEFAULT_COST_BPS,
) -> Tuple[LayerMetrics, pd.Series]:
    """
    返回 (LayerMetrics, 多空组合净值)。多空净值用于跨因子去重（修 D4）。

    monotonicity = spearman(层序号, 各层年化收益)
    这比 IC 难伪造：IC 可被少数极端样本撑起，但要让各层收益单调排列，
    因子必须真的有序。
    """
    from scipy.stats import spearmanr

    layers = build_quantile_weights(factor, n_quantiles, rebalance)

    layer_navs, layer_ann = {}, []
    for q in range(n_quantiles):
        g, n, t = compute_nav(layers[q], price, cost_bps)
        m = evaluate_portfolio(g, n, t, cost_bps)
        layer_navs[q] = n
        layer_ann.append(m.ann_return_net)

    mono = 0.0
    if len(layer_ann) >= 3 and np.std(layer_ann) > 1e-12:
        rho, _ = spearmanr(range(n_quantiles), layer_ann)
        mono = float(rho) if not np.isnan(rho) else 0.0

    # 多空：最高分位 - 最低分位（天然市场中性 → 满足 §3.3 超额收益要求）
    w_ls = layers[n_quantiles - 1] - layers[0]
    ls_g, ls_n, ls_t = compute_nav(w_ls, price, cost_bps)
    ls_metrics = evaluate_portfolio(ls_g, ls_n, ls_t, cost_bps)

    lo_g, lo_n, lo_t = compute_nav(layers[n_quantiles - 1], price, cost_bps)
    lo_metrics = evaluate_portfolio(lo_g, lo_n, lo_t, cost_bps)

    return LayerMetrics(
        n_quantiles=n_quantiles,
        layer_ann_returns=[round(x, 5) for x in layer_ann],
        monotonicity=round(mono, 4),
        long_short=ls_metrics, long_only=lo_metrics,
    ), ls_n


def nav_correlation(nav_a: pd.Series, nav_b: pd.Series) -> float:
    """
    组合净值日收益相关 —— 取代失效的 IC 序列相关（修 D4）。

    IC 序列相关衡量『两因子好坏时点是否同步』；
    净值相关衡量『持仓是否雷同』。后者才是去重要的东西。
    """
    ra = nav_a.pct_change(fill_method=None).dropna()
    rb = nav_b.pct_change(fill_method=None).dropna()
    idx = ra.index.intersection(rb.index)
    if len(idx) < 20:
        return 0.0
    c = ra.loc[idx].corr(rb.loc[idx])
    return 0.0 if pd.isna(c) else float(c)
