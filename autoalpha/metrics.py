"""
指标容器：所有口径显式化。

设计原则（来自 docs/P0_BACKTEST_DESIGN.md §2.2）：
  禁止出现裸 "icir" —— 必须带 _daily / _annual 后缀。
  旧代码在 README 写 icir>0.5（月频口径）、实际算日频（实测 0.075~0.106），
  同一个名字在三处指不同的东西。口径写进名字是唯一可靠的修法。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class ICMetrics:
    """截面预测力。"""

    ic_mean_1d: float            # 日频 IC 均值（前向 1 日收益）
    ic_std_1d: float
    icir_daily: float            # ic_mean_1d / ic_std_1d
    icir_annual: float           # icir_daily * sqrt(252)
    ic_tstat: float              # icir_daily * sqrt(n_obs) —— 显著性只看这个
    n_obs: int

    # 真月度（修 D2：旧实现 np.mean(ic_arr > 0) 是逐日胜率）
    monthly_ic_mean: float
    monthly_win_rate: float
    n_months: int

    # 同量纲衰减（修 D3：废弃 ic_5d/ic_1d，分子分母量纲不可比）
    ic_by_lag: dict = field(default_factory=dict)
    ic_halflife: float = float("nan")

    coverage: float = 0.0        # 按日(非空/当日可用)再平均，非除以全池格子数

    @property
    def is_significant(self) -> bool:
        return abs(self.ic_tstat) > 1.96


@dataclass(frozen=True)
class PortfolioMetrics:
    """组合绩效 —— 『提取有效因子』的真正判据。"""

    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    turnover_daily: float
    ann_return_net: float
    sharpe_net: float
    cost_bps: float
    n_days: int = 0


@dataclass(frozen=True)
class LayerMetrics:
    """分层单调性 —— 比 IC 更难伪造：IC 可被少数极端样本撑起，
    但要让各层收益单调排列，因子必须真的有序。"""

    n_quantiles: int
    layer_ann_returns: list
    monotonicity: float          # spearman(层序号, 层年化收益)
    long_short: Optional[PortfolioMetrics] = None
    long_only: Optional[PortfolioMetrics] = None


@dataclass
class FactorEvaluation:
    """一个因子的完整评估：IS + OOS 双期。"""

    factor_name: str
    is_ic: Optional[ICMetrics] = None
    is_layers: Optional[LayerMetrics] = None
    oos_ic: Optional[ICMetrics] = None        # 仅 IS 通过后解锁
    oos_layers: Optional[LayerMetrics] = None
    sharpe_decay: Optional[float] = None      # (oos-is)/|is|
    max_nav_corr: float = 0.0                 # 净值相关去重（修 D4）
    universe_width: float = 0.0               # 门槛推导依据
    status: str = ""                           # PASS / FAIL_IS / FAIL_OOS
    fail_reason: str = ""
    # 因子在做什么。结果卡片只有名字时，用户得回头翻假设列表才知道
    # 这是什么因子，所以把描述一路带到结果里。
    description: str = ""
    zh_what: str = ""
    code_path: str = ""
    nav_net: Optional[object] = None           # pd.Series，用于后续去重


# ── 门槛推导（修 D1 根因：门槛不再是与样本量耦合的魔数）──────────────────────

def t_threshold(n_trials: int = 1, alpha: float = 0.05) -> float:
    """显著性门槛。n_trials>1 时做 Bonferroni 修正（P2 的搜索记账会用到）。"""
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / max(n_trials, 1)))


def ic_mean_floor(width: float, n_obs: int, t_target: float = 3.0) -> float:
    """
    由目标 t 值反推 IC 均值下限。

    截面 spearman 噪声底 ic_std ≈ 1/sqrt(width)，
    t = (ic_mean/ic_std)*sqrt(n_obs)  →  floor = t_target/sqrt(n_obs)/sqrt(width)

    实测对照（n_obs=1074）：
      width=24  → 0.0187  （正好解释旧代码的 0.02 从何而来）
      width=108 → 0.0088  （当前数据，旧门槛过严 2.3 倍）
    """
    if width <= 0 or n_obs <= 0:
        return float("inf")
    return float(t_target / np.sqrt(n_obs) / np.sqrt(width))
