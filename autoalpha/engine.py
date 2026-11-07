"""
IS/OOS 切分引擎 + 门槛推导。

设计依据 docs/P0_BACKTEST_DESIGN.md §2.4 / §3。

为什么切分要由引擎持有而不是调用方自觉：
  autopaper/CLAUDE.md 第一句就是「所有指标基于 OOS 期计算，不得仅用 IS 期结果入库」，
  本项目剥离时把这个设计丢了（实测 grep train/test/oos/split 命中 0）。
  靠约定不行 —— [2608.27734] 实测：泄漏 oracle 打出 Sharpe 35，
  完整通过 Deflated Sharpe 与 PBO。防泄漏必须结构性。
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    ICMetrics, LayerMetrics, FactorEvaluation,
    ic_mean_floor, t_threshold, TRADING_DAYS,
)
from .backtest import (
    evaluate_layers, nav_correlation,
    DEFAULT_COST_BPS, DEFAULT_N_QUANTILES, DEFAULT_REBALANCE,
)

# 实测自 factor-signal/data/klines/etf-daily（见设计文档 §1）
#
# 起点取「宽度停止增长」而非「宽度达标」—— 这是第一版设计的错误。
# 实测季度 QoQ 增幅：2021Q3 +17.7% / 2021Q4 +4.3% / 2022Q1 +3.1% 后稳定在 <3%。
# 若从 2020-07 起，IS 期会跨越宽度 61→116 的整个增长过程，而 OOS 期宽度已稳定，
# 使 IS 系统性难于 OOS（实测 mom_250d: IS Sharpe 0.148 vs OOS 0.899，差 6 倍），
# 违背 IS/OOS 样本构成可比的基本假设。
DATA_START = "2022-01-01"
# 终点取宽度骤降前。实测逐月：2026-02 宽度 117，2026-03 月内掉到 84，2026-07 掉到 68
# （49/117 标的陆续停更，是抓取不完整而非退市）。
DATA_END   = "2026-02-28"
IS_FRAC      = 0.75
EMBARGO_DAYS = 21           # 防月度级 lookback 窗口跨切分点造成重叠泄漏


# ── IS 期门控（便宜，先跑）──────────────────────────────────────────────────
IS_GATES = {
    "ic_tstat_target":   3.0,    # 配合 ic_mean_floor 推导
    "monotonicity":      0.6,    # 核心判据：各层收益须单调
    "ls_sharpe":         0.5,    # 多空组合（天然市场中性）
    "coverage":          0.7,
}

# 换手门槛按调仓频率校准。实测同一因子（mom_60d）在不同频率下：
#   日频 0.415 / 周频 0.188 / 月频 0.091
# 固定 0.30 会让日频因子必然超标、月频因子轻松通过，门槛变成频率的函数
# 而非因子质量的函数。实测 E2E 四个周频因子全挤在 0.32-0.33，说明
# 对周频而言 0.30 偏紧。按频率给出各自的合理上限：
_TURNOVER_CAP = {
    "D":      0.60,   # 日频调仓，换手天然高
    "W-FRI":  0.40,
    "W":      0.40,
    "ME":     0.15,   # 月频，换手应很低
    "M":      0.15,
}
_TURNOVER_CAP_DEFAULT = 0.40


def turnover_cap(rebalance: str) -> float:
    return _TURNOVER_CAP.get(str(rebalance).upper(), _TURNOVER_CAP_DEFAULT)

# ── OOS 认证门控（仅 IS 通过者，只测一次）────────────────────────────────────
#
# 关于 IS/OOS Sharpe 差距：实测本数据集 IS 期（2022-01~2025-02）等权基准年化
# -0.26% / Sharpe -0.02（含 2022 熊市），OOS 期（2025-03~2026-02）年化 +26.06%
# / Sharpe +1.80（单边上行）。动量因子在趋势市天然更强，故 OOS > IS 是常态。
# 这是市场环境差异，任何切分点都会落在不同市场状态上，无法靠调整切分消除。
# 因此过拟合判据只看【衰减方向】：OOS 远低于 IS 才危险，OOS 高于 IS 不惩罚。
OOS_GATES = {
    "ls_sharpe_net":     0.3,    # 扣费后仍须为正
    "sharpe_decay":     -0.5,    # 衰减超 50% 视为过拟合
    "monotonicity":      0.4,
    "max_nav_corr":      0.7,    # 净值相关去重
}


class SplitEngine:
    """
    持有 IS/OOS 切分。搜索/修复循环只能看 IS；OOS 仅在认证时解锁。
    """

    def __init__(self, data: Dict[str, pd.DataFrame],
                 start: str = DATA_START, end: str = DATA_END,
                 is_frac: float = IS_FRAC, embargo_days: int = EMBARGO_DAYS,
                 cost_bps: float = DEFAULT_COST_BPS,
                 n_quantiles: int = DEFAULT_N_QUANTILES,
                 rebalance: str = DEFAULT_REBALANCE):
        px = pd.DataFrame({k: v["close"] for k, v in data.items()
                           if "close" in v.columns}).sort_index()
        if px.empty:
            raise ValueError("price matrix 为空")
        px = px.loc[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
        if len(px) < 250:
            raise ValueError(f"有效交易日太少: {len(px)}")

        self.price_full = px
        self.cost_bps = cost_bps
        self.n_quantiles = n_quantiles
        self.rebalance = rebalance

        idx = px.index
        cut = int(len(idx) * is_frac)
        self.is_end = idx[cut - 1]
        emb_pos = min(cut - 1 + embargo_days, len(idx) - 1)
        self.oos_start = idx[emb_pos]

        self.price_is = px.loc[:self.is_end]
        self.price_oos = px.loc[self.oos_start:]
        self._oos_reads: Dict[str, int] = {}

        print(f"[Engine] 数据 {px.index[0].date()}~{px.index[-1].date()} "
              f"({len(px)}日, {px.shape[1]}标的)")
        print(f"[Engine] IS  {self.price_is.index[0].date()}~{self.is_end.date()} "
              f"({len(self.price_is)}日, 宽度中位 {int(self.price_is.notna().sum(axis=1).median())})")
        print(f"[Engine] 禁运 {embargo_days}日 → OOS {self.oos_start.date()}~"
              f"{px.index[-1].date()} ({len(self.price_oos)}日)")

    # ── 宽度与门槛 ─────────────────────────────────────────────────────────

    def width(self, period: str = "is") -> float:
        px = self.price_is if period == "is" else self.price_oos
        return float(px.notna().sum(axis=1).median())

    def gates_for(self, period: str = "is", n_trials: int = 1) -> dict:
        """
        按该期【实际截面宽度】推导门槛，不读硬编码常量。

        这是修 D1 根因的地方：旧代码 `ic_mean: 0.02` 是为 24 个标的算的
        （mine_momentum.py:119 注释写明），换成 117 标的后过严 2.3 倍，
        但代码里没有任何地方记录这个耦合关系。
        """
        px = self.price_is if period == "is" else self.price_oos
        w = self.width(period)
        n_obs = len(px)
        g = dict(IS_GATES)
        g["turnover_daily"] = turnover_cap(self.rebalance)
        g["ic_mean_floor"] = ic_mean_floor(w, n_obs, IS_GATES["ic_tstat_target"])
        g["t_threshold"] = t_threshold(n_trials)
        g["_width"] = w
        g["_n_obs"] = n_obs
        return g

    # ── 评估 ───────────────────────────────────────────────────────────────

    def evaluate_is(self, factor: pd.Series) -> Tuple[ICMetrics, LayerMetrics, pd.Series]:
        """搜索/修复循环只能调这个。"""
        # 截面不足时必须说人话。compute_ic_metrics 要求每天 ≥10 只有效标的
        # 才算 IC，否则逐日 continue、base 为空、返回全 0 —— 用户看到的是
        # 「所有因子都失败」，而真因是「标的太少，IC 根本算不出来」。
        # 静默返回 0 违反「失败必须可见」。
        w = self.width("is")
        if w < 10 and not getattr(self, "_narrow_warned", False):
            self._narrow_warned = True
            print(f"[Engine] ⚠ 截面宽度中位仅 {w:.0f} 只（<10）："
                  f"每日有效标的不足 10 个时 IC 无法计算，会全部返回 0。"
                  f"这不是「因子都无效」，而是样本太窄。"
                  f"请扩充到 ≥10 只（建议 ≥20），或换更宽的数据集。")
        f = self._clip(factor, self.price_is)
        ic = compute_ic_metrics(f, self.price_is)
        lm, ls_nav = evaluate_layers(f, self.price_is, self.n_quantiles,
                                     self.rebalance, self.cost_bps)
        return ic, lm, ls_nav

    def evaluate_oos(self, factor: pd.Series,
                     factor_name: str = "?") -> Tuple[ICMetrics, LayerMetrics, pd.Series]:
        """
        仅对通过 IS 闸门的因子调用，一个因子只应读一次。
        重复读取会打警告 —— 反复看 OOS 等于把它变成第二个 IS。
        """
        self._oos_reads[factor_name] = self._oos_reads.get(factor_name, 0) + 1
        if self._oos_reads[factor_name] > 1:
            print(f"[Engine] ⚠ OOS 重复读取 {factor_name} "
                  f"(第{self._oos_reads[factor_name]}次) — 结果已不再是样本外")
        f = self._clip(factor, self.price_oos)
        ic = compute_ic_metrics(f, self.price_oos)
        lm, ls_nav = evaluate_layers(f, self.price_oos, self.n_quantiles,
                                     self.rebalance, self.cost_bps)
        return ic, lm, ls_nav

    @staticmethod
    def _clip(factor: pd.Series, px: pd.DataFrame) -> pd.Series:
        lo, hi = px.index[0], px.index[-1]
        dates = factor.index.get_level_values(0)
        return factor[(dates >= lo) & (dates <= hi)]

    # ── 双层门控 ───────────────────────────────────────────────────────────

    def check_is_gates(self, ic: ICMetrics, lm: LayerMetrics,
                       n_trials: int = 1) -> Tuple[bool, str]:
        g = self.gates_for("is", n_trials)
        fails = []
        if abs(ic.ic_tstat) < g["t_threshold"]:
            fails.append(f"|t|={abs(ic.ic_tstat):.2f} < {g['t_threshold']:.2f}")
        if lm.monotonicity < g["monotonicity"]:
            fails.append(f"单调性={lm.monotonicity:.3f} < {g['monotonicity']}")
        if lm.long_short and lm.long_short.sharpe < g["ls_sharpe"]:
            fails.append(f"多空Sharpe={lm.long_short.sharpe:.3f} < {g['ls_sharpe']}")
        if ic.coverage < g["coverage"]:
            fails.append(f"覆盖率={ic.coverage:.3f} < {g['coverage']}")
        if lm.long_short and lm.long_short.turnover_daily > g["turnover_daily"]:
            fails.append(f"换手={lm.long_short.turnover_daily:.3f} > {g['turnover_daily']}")
        return (not fails), "; ".join(fails)

    def check_oos_gates(self, is_lm: LayerMetrics, oos_ic: ICMetrics,
                        oos_lm: LayerMetrics, max_nav_corr: float
                        ) -> Tuple[bool, str, float]:
        fails = []
        decay = None
        if oos_lm.long_short and oos_lm.long_short.sharpe_net < OOS_GATES["ls_sharpe_net"]:
            fails.append(f"OOS扣费Sharpe={oos_lm.long_short.sharpe_net:.3f} "
                         f"< {OOS_GATES['ls_sharpe_net']}")
        if is_lm.long_short and oos_lm.long_short:
            base = abs(is_lm.long_short.sharpe)
            if base > 1e-9:
                decay = (oos_lm.long_short.sharpe - is_lm.long_short.sharpe) / base
                if decay < OOS_GATES["sharpe_decay"]:
                    fails.append(f"Sharpe衰减={decay:+.1%} < {OOS_GATES['sharpe_decay']:.0%}")
        if oos_lm.monotonicity < OOS_GATES["monotonicity"]:
            fails.append(f"OOS单调性={oos_lm.monotonicity:.3f} < {OOS_GATES['monotonicity']}")
        if max_nav_corr >= OOS_GATES["max_nav_corr"]:
            fails.append(f"净值相关={max_nav_corr:.3f} >= {OOS_GATES['max_nav_corr']}")
        return (not fails), "; ".join(fails), (decay if decay is not None else float("nan"))

    def full_evaluate(self, factor: pd.Series, factor_name: str,
                      existing_navs: Optional[list] = None,
                      n_trials: int = 1) -> FactorEvaluation:
        """IS → (通过则) OOS 的完整评估。"""
        ev = FactorEvaluation(factor_name=factor_name,
                              universe_width=self.width("is"))
        try:
            ic, lm, is_ls_nav = self.evaluate_is(factor)
        except Exception as e:
            ev.status, ev.fail_reason = "FAIL_IS", f"IS 评估异常: {e}"
            return ev
        ev.is_ic, ev.is_layers = ic, lm
        ev.nav_net = is_ls_nav          # 失败因子也保留 IS 净值，供前端展示诊断

        ok, why = self.check_is_gates(ic, lm, n_trials)
        if not ok:
            ev.status, ev.fail_reason = "FAIL_IS", why
            return ev

        try:
            oic, olm, ols_nav = self.evaluate_oos(factor, factor_name)
        except Exception as e:
            ev.status, ev.fail_reason = "FAIL_OOS", f"OOS 评估异常: {e}"
            return ev
        ev.oos_ic, ev.oos_layers, ev.nav_net = oic, olm, ols_nav

        corr = 0.0
        for prev in (existing_navs or []):
            corr = max(corr, abs(nav_correlation(ols_nav, prev)))
        ev.max_nav_corr = round(corr, 4)

        ok, why, decay = self.check_oos_gates(lm, oic, olm, corr)
        ev.sharpe_decay = None if np.isnan(decay) else round(decay, 4)
        ev.status, ev.fail_reason = ("PASS", "") if ok else ("FAIL_OOS", why)
        return ev


# ── IC 指标计算（口径显式化，修 D2/D3）──────────────────────────────────────

def compute_ic_metrics(factor: pd.Series, price: pd.DataFrame,
                       lags: tuple = (1, 5, 10, 20)) -> ICMetrics:
    """
    计算 ICMetrics。三处与旧实现不同：
      D2: monthly_win_rate 先按月聚合 IC 再算正负占比（旧实现是逐日胜率）
      D3: ic_by_lag 全部用【前向1日收益】、仅改因子滞后天数 → 量纲可比
          （旧实现 ic_5d/ic_1d 用 5 日累计收益，分子分母量纲不同，实测比值系统性>1）
      coverage: 按日(非空/当日可用)再平均，非除以 len(dates)*len(all_symbols)
    """
    from scipy.stats import spearmanr

    wide = factor.unstack(level=-1).reindex(index=price.index).sort_index()
    fwd1 = price.pct_change(fill_method=None).shift(-1)

    def _ic_for_lag(lag: int) -> list:
        fl = wide.shift(lag - 1)       # lag=1 即不滞后
        out = []
        for dt in price.index:
            if dt not in fl.index:
                continue
            a, b = fl.loc[dt].dropna(), fwd1.loc[dt].dropna()
            i = a.index.intersection(b.index)
            if len(i) < 10:
                continue
            r, _ = spearmanr(a.loc[i].to_numpy(), b.loc[i].to_numpy())
            if not np.isnan(r):
                out.append((dt, float(r)))
        return out

    base = _ic_for_lag(1)
    if len(base) < 20:
        return ICMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, {}, float("nan"), 0)

    s = pd.Series(dict(base)).sort_index()
    ic_mean, ic_std = float(s.mean()), float(s.std())
    # ic_std≈0 分两种情况，不能一律返回 0：
    #   mean≈0 → 确实无信号 → icir=0
    #   mean≠0 → IC 恒定不变（完美信号）→ 极度显著，不是无信号
    # 旧写法 `if ic_std > 1e-12 else 0.0` 把后者误判为前者，
    # 导致泄漏因子 t=0.00（实测 ic_mean=1.0 / ic_std=0.0 → t 归零）。
    if ic_std > 1e-12:
        icir_d = ic_mean / ic_std
    elif abs(ic_mean) > 1e-12:
        icir_d = float(np.sign(ic_mean) * 1e3)      # 有界哨兵，避免 inf 污染下游
    else:
        icir_d = 0.0

    m = s.groupby(pd.Grouper(freq="ME")).mean().dropna()     # D2: 真月度

    by_lag = {1: round(ic_mean, 5)}
    for lg in lags[1:]:
        r = _ic_for_lag(lg)
        by_lag[lg] = round(float(np.mean([v for _, v in r])), 5) if r else 0.0

    hl = float("nan")
    if by_lag[1] > 1e-6:
        ratios = [(lg, by_lag[lg] / by_lag[1]) for lg in lags if by_lag.get(lg, 0) > 0]
        decayed = [(lg, rt) for lg, rt in ratios if rt < 0.5]
        hl = float(decayed[0][0]) if decayed else float(max(lags))

    # coverage 分母 = 当日「有因子值资格」的标的数，而非当日所有可用标的。
    # 长 lookback 因子在标的上市初期天然无值（实测 mom_250d 按年 coverage：
    # 2021=0.324 → 2025=0.991），用全部可用标的作分母会系统性惩罚长窗口因子。
    # 资格 = 该标的在当日之前已有足够历史；用因子自身的首个非空日期界定。
    first_valid = wide.apply(lambda c: c.first_valid_index())
    eligible = pd.DataFrame(
        {c: (wide.index >= fv) if fv is not None else False
         for c, fv in first_valid.items()},
        index=wide.index,
    )
    # 同时要求该标的当日确有价格（停牌不算分母）
    has_px = price.reindex(index=wide.index, columns=wide.columns).notna()
    eligible = eligible & has_px
    # 分子必须与分母同口径：只数「当日有价格」的因子值。
    # 否则因子在停牌日 ffill 出的值会进分子而不进分母，实测 coverage 达 1.07。
    numer = (wide.notna() & has_px).sum(axis=1)
    denom = eligible.sum(axis=1).replace(0, np.nan)
    cov = float((numer / denom).clip(upper=1.0).mean())

    return ICMetrics(
        ic_mean_1d=round(ic_mean, 5), ic_std_1d=round(ic_std, 5),
        icir_daily=round(icir_d, 4),
        icir_annual=round(icir_d * np.sqrt(TRADING_DAYS), 4),
        ic_tstat=round(icir_d * np.sqrt(len(s)), 3), n_obs=len(s),
        monthly_ic_mean=round(float(m.mean()), 5) if len(m) else 0.0,
        monthly_win_rate=round(float((m > 0).mean()), 4) if len(m) else 0.0,
        n_months=len(m),
        ic_by_lag=by_lag, ic_halflife=hl, coverage=round(cov, 4),
    )
