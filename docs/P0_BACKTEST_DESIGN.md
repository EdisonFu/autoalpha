# P0 实施设计：日频回测层

> 决策：方案 (a) —— 先只做日频回测，日内因子暂仍只出 IC
> 目标：让「想法 → 论文 → 因子 → **回测** → 提取有效因子」这条链真正闭环
> 前置：本文档所有参数均从 `factor-signal/data/klines/etf-daily`（120 文件 / 117 标的）实测得出

---

## 0. 为什么 P0 是「补回测」而不是「修指标」

上一版方案把"评估诚实性"排 P0，那个判断的对象错了。对一个**自用因子生产**项目：

| | 性质 | 后果 |
|---|---|---|
| 缺回测 | **功能缺口** | 系统压根不产出你要的东西 |
| 指标口径错 | 质量缺陷 | 产出了但不可信 |

实测全仓库：`backtest` / `sharpe` / `quantile` / `calmar` / `long_short` / `分层` / `分位` 命中数**全为 0**。
`nav` / `spread` / `drawdown` 的少数命中全是无关项（数据源列名映射、prompt 示例词、因子子类名）。

**IC 不是回测。** IC 只说"因子排序与次日收益的相关性"，不说：分层组合挣多少、换手多高、
回撤多深、扣费后还剩什么。而"提取有效因子"的判据恰是后者。

而且两者有依赖：回测一旦建起来，D3/D4 会被更强的信号自然覆盖（见 §4）。

---

## 1. 数据层实测结论（决定一切参数）

### 1.1 `load_klines()` 零适配即可复用

`data_loader.py` 的 docstring 声明的目录结构与 CSV 列名，与 factor-signal 那份数据**完全一致**，
连文件名正则 `[a-z]+etf_(\d+)_` 都匹配 `shetf_510300_沪深300ETF.csv`。

```
trade_date,open,high,low,close,volume,amount,open_interest
2015-01-05 00:00:00,2.6253,2.7453,2.6116,2.7022,3075898438,11300578265.0,0.0
```

→ **复用不需要写任何适配代码**，只改默认路径。

### 1.2 ⚠️ 当前默认数据路径是空的

`mine_momentum.py:426` 默认 `--klines /Users/fuzhendong/Desktop/cc论文/data/klines`。
目录存在，但 `etf-daily/` 下 **0 个文件**。现在直接跑 `mine_momentum.py` 拿不到数据。

### 1.3 截面宽度随时间剧变 —— 这是硬编码门槛的根因

实测逐年可用标的数与 IC 噪声底 `1/√N`：

| 年份 | 中位宽度 | 噪声底 1/√N |
|---|---|---|
| 2013 | 4 | 0.500 |
| 2015 | 21 | 0.218 |
| 2017 | 30 | 0.183 |
| 2019 | 44 | 0.151 |
| 2020 | 62 | 0.127 |
| 2021 | 93 | 0.104 |
| 2023 | 109 | 0.096 |
| 2025 | 117 | **0.092** |

`mine_momentum.py:119` 的注释写「ETF universe has only 24 assets — IC noise floor ≈ 1/sqrt(24) ≈ 0.20」，
整套放宽门槛（`icir: 0.06`、`ic_mean: 0.02`）都是为 **24 个标的**算的。

现在是 117 个 → 噪声底从 0.20 降到 0.092，**那套放宽的理由不再成立**。

> **这揭示了 D1 的真正性质**：问题不是"门槛该取多少"，
> 而是门槛被硬编码成一个与样本量耦合的魔数，却没记录这个耦合。换数据源就全错。
> → 正确做法：**门槛由截面宽度推导**（§3.2）。

### 1.4 ⚠️ 数据尾部不完整，必须截断

实测各标的最后有效日期：

```
2026-03 停更: 33 个
2026-07 停更: 16 个
2026-08 仍有: 68 个
→ 49/117 在 2026-08 前停更
```

宽度在 2026-07 后恒定为 68（不是节假日波动），说明是**抓取不完整**而非退市。
若不截断，2026-06 之后的截面会从 117 骤降到 68，造成**样本构成突变**——
这是比幸存者偏差更隐蔽的陷阱：因子在"宽度 117 的截面"和"宽度 68 的截面"上不可比。

### 1.5 最终数据窗口

```python
DATA_START = "2020-07-01"   # 滚动60日中位宽度首次稳定 ≥60（实测 2020-05-26，取整到季度）
DATA_END   = "2026-05-31"   # 截掉尾部不完整段
# 实测该窗口: 1433 交易日, 宽度 median=108 / min=61
```

IS/OOS 切分（实测两个候选）：

| 比例 | IS 末 | 禁运至 | n_IS | n_OOS |
|---|---|---|---|---|
| 70/30 | 2024-08-15 | 2024-09-13 | 1003 | 409 |
| **75/25** | **2024-12-03** | **2025-01-02** | **1074** | **338** |

**选 75/25**：IS 末落在自然年末，便于人工核对；OOS 338 日（约 16 个月）足够覆盖多个市场状态。
禁运 21 日，防止月度级 lookback 窗口跨越切分点造成重叠泄漏。

---

## 2. 模块设计

### 2.1 新增文件

```
autoalpha/
  backtest.py      ← 新增：分层组合构建 + 绩效指标
  metrics.py       ← 新增：口径显式化的指标容器
  engine.py        ← 新增：IS/OOS 切分强制执行
```

### 2.2 `metrics.py` — 口径写进名字

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ICMetrics:
    """截面预测力。所有字段名自带口径，禁止出现裸 'icir'。"""
    ic_mean_1d:       float   # 日频 IC 均值（前向1日收益）
    ic_std_1d:        float
    icir_daily:       float   # ic_mean_1d / ic_std_1d
    icir_annual:      float   # icir_daily * sqrt(252)
    ic_tstat:         float   # icir_daily * sqrt(n_obs)  ← 显著性只看这个
    n_obs:            int     # 有效 IC 观测数

    # 真月度（修 D2：原实现是逐日胜率）
    monthly_ic_mean:  float   # 先按月聚合 IC，再取均值
    monthly_win_rate: float   # 按月聚合后 IC>0 的月份占比
    n_months:         int

    # 同量纲衰减（修 D3：废弃 ic_5d/ic_1d 比值）
    ic_by_lag:        dict    # {1: ic, 5: ic, 10: ic, 20: ic}
                              # 全部用前向1日收益，仅因子滞后天数不同
    ic_halflife:      float   # 由 ic_by_lag 指数拟合，单位=交易日

    # 截面覆盖
    coverage:         float   # 因子非空 / 当日可用标的数（按日算再平均）
                              # 注意：不是除以 len(dates)*len(all_symbols)

@dataclass(frozen=True)
class PortfolioMetrics:
    """组合绩效。这是『提取有效因子』的真正判据。"""
    ann_return:       float
    ann_vol:          float
    sharpe:           float
    max_drawdown:     float
    calmar:           float
    turnover_daily:   float   # 单边日均换手
    # 扣费后
    ann_return_net:   float
    sharpe_net:       float
    cost_bps:         float   # 记录用的是哪个成本假设

@dataclass(frozen=True)
class LayerMetrics:
    """分层单调性 —— 判断因子是否真有序，而非靠极端组侥幸。"""
    n_quantiles:      int
    layer_ann_returns: list   # 各分位年化收益，Q1→QN
    monotonicity:     float   # spearman(层序号, 层收益)，1.0=完美单调
    long_short:       PortfolioMetrics   # 多空组合（QN - Q1）
    long_only:        PortfolioMetrics   # 只做多最高分位

@dataclass(frozen=True)
class FactorEvaluation:
    """一个因子的完整评估：IS + OOS 双期。"""
    factor_name:      str
    is_ic:            ICMetrics
    is_layers:        LayerMetrics
    oos_ic:           ICMetrics | None   # 仅通过 IS 闸门后才解锁
    oos_layers:       LayerMetrics | None
    sharpe_decay:     float | None       # (oos_sharpe - is_sharpe) / |is_sharpe|
    max_nav_corr:     float              # 与库内因子的净值相关（修 D4）
    universe_width:   float              # 评估期平均截面宽度（门槛推导依据）
    status:           str                # PASS / FAIL_IS / FAIL_OOS
    fail_reason:      str = ""
```

### 2.3 `backtest.py` — 核心接口

```python
import pandas as pd
from .metrics import PortfolioMetrics, LayerMetrics

DEFAULT_COST_BPS = 5.0        # 单边 5bp，ETF 市场合理假设
DEFAULT_N_QUANTILES = 5       # 宽度>=60 时 5 层，每层约 12-23 个标的


def build_quantile_portfolios(
    factor: pd.Series,            # MultiIndex (date, symbol)
    n_quantiles: int = DEFAULT_N_QUANTILES,
    rebalance: str = "W-FRI",     # 周频调仓；"D"=日频（换手过高，不推荐）
) -> pd.DataFrame:
    """
    按日对因子值做截面分位切分，返回各层权重矩阵。

    关键细节（都是实测踩过的坑）：
      1. 分位切分只用【当日实际可用】的标的，不是全池 —— 否则早期年份分层退化
      2. 调仓日之间权重 forward-fill，非调仓日不交易（这才产生真实换手）
      3. 层内等权。IC 加权留到 P1，因为它会把 IC 的噪声带进权重

    Returns: DataFrame, index=(date, symbol), columns=[q1..qN]
    """


def compute_nav(
    weights: pd.DataFrame,
    price: pd.DataFrame,          # wide: index=date, columns=symbol
    cost_bps: float = DEFAULT_COST_BPS,
) -> tuple[pd.Series, pd.Series, float]:
    """
    由权重和价格算净值。

    时序纪律（防前瞻的核心）：
      - t 日收盘算因子 → t 日收盘后定权重 → **t+1 日收益才归属该权重**
      - 即 ret_contrib[t+1] = weights[t] · ret[t+1]，权重必须 shift(1)
      - 成本在换仓日扣：cost = cost_bps * 1e-4 * Σ|w[t] - w[t-1]|

    Returns: (nav_gross, nav_net, turnover_daily)
    """


def evaluate_portfolio(nav: pd.Series, ...) -> PortfolioMetrics:
    """年化/Sharpe/MDD/Calmar。年化用 252，Sharpe 不减无风险利率（自用，相对比较）。"""


def evaluate_layers(
    factor: pd.Series, price: pd.DataFrame, **kw
) -> LayerMetrics:
    """
    完整分层评估：各层净值 + 单调性 + 多空 + 纯多头。

    monotonicity = spearman(层序号, 各层年化收益)
    这是比 IC 更难伪造的指标：IC 可以靠少数极端样本撑起来，
    但要让 5 层收益单调排列，因子必须真的有序。
    """
```

### 2.4 `engine.py` — 切分强制执行

```python
class SplitEngine:
    """
    IS/OOS 切分由引擎持有。搜索/修复循环只能看 IS，OOS 仅在认证时解锁一次。
    设计依据：autopaper/CLAUDE.md 已有的 test_periods 设计（本项目剥离时丢失了）
    """

    def __init__(self, data: dict,
                 start=DATA_START, end=DATA_END,
                 is_frac=0.75, embargo_days=21):
        # 自动截断到 [start, end]，计算切分点，记录各期截面宽度
        ...

    @property
    def is_price(self) -> pd.DataFrame: ...
    @property
    def oos_price(self) -> pd.DataFrame:
        """访问时打日志。一个因子的 OOS 只应被读一次。"""

    def gates_for(self, period: str) -> dict:
        """按该期实际截面宽度推导门槛（见 §3.2），不读硬编码常量。"""
```

---

## 3. 质量门控重设计

### 3.1 双层门控

```
Layer 1  IS 闸门（便宜，先跑）
  ├─ ic_tstat        > t_threshold(width)      ← 推导，非硬编码
  ├─ monotonicity    > 0.6                      ← 新增，核心判据
  ├─ ls_sharpe       > 0.5                      ← 新增（多空组合）
  ├─ coverage        > 0.7
  └─ turnover_daily  < 0.3                      ← 新增，防高换手伪因子

Layer 2  OOS 认证（仅 IS 通过者，只测一次）
  ├─ oos_ls_sharpe_net > 0.3                    ← 扣费后仍须为正
  ├─ sharpe_decay      > -0.5                   ← 衰减超50%视为过拟合
  ├─ oos_monotonicity  > 0.4
  └─ max_nav_corr      < 0.7                    ← 净值相关去重（修 D4）
```

**废弃**：`ic_decay_ratio`（量纲不可比，实测 13/13 全过，不筛任何东西）。
由 `ic_halflife` 替代，但仅作诊断展示，不设闸门 —— 半衰期的合理区间依赖调仓频率。

### 3.2 门槛由截面宽度推导（修 D1 根因）

```python
import numpy as np

def t_threshold(n_trials: int = 1, alpha: float = 0.05) -> float:
    """显著性门槛只看 t 统计量，与截面宽度解耦。"""
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / max(n_trials, 1)))


def ic_mean_floor(width: float, n_obs: int, t_target: float = 3.0) -> float:
    """
    由目标 t 值反推 IC 均值下限。

    ic_std ≈ 1/sqrt(width)（截面 spearman 的噪声底）
    t = (ic_mean / ic_std) * sqrt(n_obs)
      → ic_mean_floor = t_target / sqrt(n_obs) / sqrt(width)

    实测对照（n_obs=1074 IS 期）：
      width= 24 → floor=0.0187   （解释了旧代码为何取 0.02）
      width=108 → floor=0.0088   （当前数据，旧门槛 0.02 过严 2.3 倍）
    """
    return t_target / np.sqrt(n_obs) / np.sqrt(width)
```

> 这一步把「24 个标的」这个隐含假设**从魔数变成了显式输入**。
> 换数据源、换时间窗口，门槛自动跟着变，不会再静默失效。

### 3.3 必须在超额收益上计算

按 [2610.07701] 的实测警告：**在绝对收益上算的准则会静默拒绝全部候选，包括真信号**。

所以所有 Sharpe / 年化 / 单调性，**对多空组合（天然市场中性）或相对等权基准的超额**计算，
绝不在绝对收益上算。等权基准 = 当日所有可用标的等权持有。

---

## 4. 回测层如何自然覆盖 D3 / D4

这是把回测排在"修指标"之前的真正理由：

**D4（去重用错口径）** — 原实现用 IC 时间序列的 pearson 相关。实测 `26w_` 与 `52w_drawdown_recovery`
（同公式只改窗口）相关仅 **-0.01**，轻松过 0.7 门槛，双双入库。

有了回测，改用**组合净值相关性**：

```python
def nav_correlation(nav_a: pd.Series, nav_b: pd.Series) -> float:
    """对日收益率算 pearson。两因子若选出同一批标的，净值走势必然同步。"""
    ra, rb = nav_a.pct_change().dropna(), nav_b.pct_change().dropna()
    idx = ra.index.intersection(rb.index)
    return float(ra[idx].corr(rb[idx]))
```

IC 序列相关衡量的是"两因子**好坏时点**是否同步"；净值相关衡量"**持仓**是否雷同"。
后者才是去重要的东西，而且比截面 rank 相关更贴近"这两个因子能不能都留"这个决策。

**验收**：`26w_` vs `52w_drawdown_recovery` 的净值相关必须显著 >0.7 并被判重复。

**D3（衰减闸门失效）** — 原 `ic_5d/ic_1d` 因分子分母量纲不可比而系统性 >1
（实测分布 1.06~1.61，门槛 0.5 形同不存在）。
有了分层净值，"信号衰减多快"直接体现为**不同调仓频率下的 Sharpe 差异**，比任何比值都直观。

---

## 5. 实施顺序与验收

```
Step 1  数据接入            默认路径 → factor-signal；窗口截断到 2020-07..2026-05
Step 2  metrics.py          口径显式化（顺带修 D1 命名 / D2 真月度 / 废弃 D3）
Step 3  backtest.py         分层 → 净值 → 换手 → 扣费（含 shift(1) 时序纪律）
Step 4  engine.py           IS/OOS 切分 + 门槛推导
Step 5  接回 pipeline       FactorEvalAgent 返回 FactorEvaluation；报告加净值曲线
Step 6  重跑 13 个因子       输出新旧对照表
```

### 验收标准

```
□ Step 3 必须有 pytest：
    - 已知权重+已知价格 → 手算净值能对上
    - 构造一个「用未来收益当因子」的泄漏因子 → Sharpe 应极高（证明回测能抓信号）
    - 同一因子 shift(1) 与不 shift → 结果必须显著不同（证明时序纪律生效）
□ 26w_ vs 52w_drawdown_recovery 净值相关 > 0.7 且被判重复
□ 13 个旧因子中，能通过新双层门控的数量预期显著下降
    —— PASS 变少是正确结果，不是退步
□ 报告须同时显示 IS 与 OOS 两栏，以及 sharpe_decay
□ 任何输出里不再出现裸 "ICIR"，必须带 _daily / _annual 后缀
```

### 预期结果（诚实预告）

13 个旧 PASS 因子里，**大概率只有少数能过新门控**。原因：

- 它们当初过的是 `ic_mean>0.02` + `icir_daily>0.06`，而新增了单调性、多空 Sharpe、
  换手、扣费后收益、OOS 衰减五道闸门
- `t_stat` 这一项它们其实扛得住（实测最佳 5.03）
- 但分层单调性和扣费后多空 Sharpe 是**全新的、更难的**考验

按 [2610.07701] 的实测：弱信号区加闸门会把采纳率压到 1–7%。
这是闸门在正常工作，不是出了问题。

---

## 6. 不在 P0 范围内（明确划界）

| 项 | 推迟到 | 理由 |
|---|---|---|
| 日内因子回测 | P1 后评估 | 需独立成本模型；扣费后日内因子常直接归零，先验证日频闭环 |
| 因子池边际贡献 | P1 | 依赖 P0 的净值基础设施 |
| 搜索记账 + deflation | P2 | 自用场景是加分项；但 77 次试验选 13 个的选择性偏差真实存在 |
| AlphaSchema / AST 记忆 / 谱系预算 | P2 | 搜索质量，不是产出可信度 |
| 模型路由 / GP 内层 / 合并重复代码 | P3 | 工程优化 |

---

## 附录 A：实施过程中发现的问题（实测记录）

本节记录 P0 实施时被真实运行暴露的问题。全部已修，列出以免重犯。

### A1　设计错误：IS/OOS 切分点选在宽度增长期

第一版取 `DATA_START=2020-07-01`（依据"宽度首次≥60"）。实测后发现 IS 期跨越了
截面宽度 61→116 的整个增长过程，而 OOS 期宽度已稳定，造成 **IS 系统性难于 OOS**：

```
            FULL     IS      OOS
mom_250d   0.676   0.148   0.899      ← IS 仅为 OOS 的 1/6
```

修正：起点改用「宽度**停止增长**」判据（季度 QoQ < 3%），即 `2022-01-01`。
终点从 `2026-05-31` 改为 `2026-02-28` —— 实测宽度在 2026-03 月内就掉到 84。

教训：选切分点要看宽度的**导数**，不只看水平值。

### A2　市场环境差异无法靠切分消除

修正切分后 IS/OOS 差距缩小但未消失（250d: 0.244 vs 0.852）。实测等权基准：

```
IS  2022-01~2025-02: 年化 -0.26%  Sharpe -0.02   （含 2022 熊市）
OOS 2025-03~2026-02: 年化 +26.06% Sharpe +1.80   （单边上行）
```

动量因子在趋势市天然更强。任何切分点都会落在不同市场状态上。
→ 过拟合判据只看**衰减方向**：`OOS << IS` 才危险，`OOS > IS` 不惩罚。
现有 `sharpe_decay > -0.5` 的门槛方向是对的。

### A3　真 bug：`ic_std≈0` 保护逻辑把完美信号判成无信号

```python
icir_d = ic_mean / ic_std if ic_std > 1e-12 else 0.0   # ← 错
```
泄漏因子（用次日收益）的 `ic_mean=1.0, ic_std=0.0` → `t` 被判为 **0.00**，
而同一因子的多空 Sharpe 高达 301615。IC 说无信号、回测说完美信号，自相矛盾。

修正：分三种情况 —— `std≈0 且 mean≠0` 是极度显著，不是无信号。

### A4　LLM 代码生成的三类系统性失败

端到端跑通后，首轮实测 **0/3 成功、每个因子耗尽 3 次重试**。逐一定位：

| 问题 | 现象 | 修法 |
|---|---|---|
| **pandas 3.x API** | `stack(dropna=)` 已移除，首次尝试必失败 | prompt 加环境约束清单 |
| **缺 main 块** | `rc=0` 无输出，错误信息是空的 `ERROR: ` | 补 NO_MAIN 分类 + 具体提示 |
| **自行月末重采样** | coverage 塌到 5%，分层年化失真到 ±290% | prompt 明确「每个交易日都要输出」+ 稀疏检测 |

其中**空错误信息**最隐蔽：LLM 在修复轮次里看不到任何线索，3 次重试全部浪费。
修复后代码生成成功率 **0% → 100%**（3/3 一次通过，无重试）。

教训：失败路径的错误信息质量，直接决定修复回路是否有效。
`returncode != 0 or marker not in stdout` 这种判定必须对**每条分支**给出可操作的描述。

### A5　两道因子形状前置检查（省掉无效回测）

实测发现两类"代码能跑、CSV 正常、但因子不可用"的情况，加检测拦在回测之前：

- `_sparse_dates`：因子只覆盖 <50% 交易日 → 疑似自行重采样
- `_cross_section_degenerate`：同一天所有标的取值几乎相同（典型成因 `np.tile`
  广播市场级标量）→ IC 恒为 0

### A6　NaN 不是合法 JSON

`ic_halflife` 的 `NaN` 让 `/api/session` 直接 500。SSE 路径因为用了
`json.dumps(default=)` 没暴露，Web API 用 FastAPI 默认编码器就炸了。
→ `flow._safe()` 统一清洗 NaN/Inf → `None`。

### A7　改代码后必须重启服务

一次排查中补丁已写入文件，但服务进程是补丁前启动的，仍跑旧代码，
导致误判"修复无效"。开发期改 `autoalpha/` 下任何文件后都要重启。
