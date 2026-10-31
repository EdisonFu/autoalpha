# AutoAlpha 优化方案（2026-Q4）

> 基线：`800fa19 v2`（2026-03），停更约 7 个月
> 资料截止：2026-10-30，arXiv q-fin + GitHub 实测扫描
> 本文档的全部诊断结论均来自对本仓库 `momentum_factors.db`（13 条 PASS 记录）与源码的实测，非纸面推断

---

## 0. 结论先行

项目的**骨架（假设→代码→评估→修复→经验记忆）是对的**，并且恰好站在 2026 下半年文献共识的方向上。

但**评估层存在 5 个可被实测证伪的缺陷**，它们使"13 个因子通过质量门控"这一结论当前**不可采信**。
2026 年 4 月后最重要的文献转向，恰好就是"评估诚实性"（见 §1.3），这不是巧合——
社区在同一时期发现：LLM 挖因子的瓶颈已经不是生成能力，而是**评估能力**。

因此本方案的优先级安排是：**先修评估，再谈搜索**。
在 P0 完成前，增加任何生成/搜索能力只会更快地生产不可信的因子。

---

## 1. 外部资料扫描结果

### 1.1 规模：半年新增 20+ 篇直接相关论文

项目 `research/` 库最新为 `2603.16365`（FactorEngine, 2026-03）。此后的新增（按时间倒序，实测 arXiv API）：

| 日期 | arXiv | 论文 | 对本项目的可吸纳点 |
|---|---|---|---|
| 2026-10-26 | [2610.34936](https://arxiv.org/abs/2610.34936) | **Det-MASO** | log-det 多样性惩罚，代价随资产数**线性**增长 |
| 2026-10-21 | [2610.29663](https://arxiv.org/abs/2610.29663) | **LineageEvo** | **谱系（lineage）**作为搜索控制单元；按谱系分配预算防分支垄断 |
| 2026-10-21 | [2610.29500](https://arxiv.org/abs/2610.29500) | **Auditing LLM Alpha Mining** | ⚠️ 直接针对本类系统的审计：指标口径混用、队列不完整 |
| 2026-10-16 | [2610.23744](https://arxiv.org/abs/2610.23744) | AlphaLedger | 成本调整后的**边际组合贡献**作为监督信号 |
| 2026-10-12 | [2610.17130](https://arxiv.org/abs/2610.17130) | **EvoQuantLab** | 内层 GP + 外层 LLM 诊断：**token 效率**关键设计 |
| 2026-10-11 | [2610.15488](https://arxiv.org/abs/2610.15488) | AlphaPilot | 多轮工具调用 + 保留因子池跨轮更新 |
| 2026-10-10 | [2610.14959](https://arxiv.org/abs/2610.14959) | **AlphaRLM** | **文献驱动**的算子装配（LEGO）——与本项目路线最接近 |
| 2026-10-04 | [2610.04959](https://arxiv.org/abs/2610.04959) | AlphaPADI | **池感知**生成：因子价值 = 对池的增量贡献 |
| 2026-09-28 | [2609.34188](https://arxiv.org/abs/2609.34188) | AlphaPareto | 多目标向量奖励（预测力/稳定性/鲁棒性/多样性）Pareto 正则 |
| 2026-09-24 | [2609.29014](https://arxiv.org/abs/2609.29014) | AlphaDiverse | 研究路径坍塌（research path collapse）问题的命名与缓解 |
| 2026-09-08 | [2609.08719](https://arxiv.org/abs/2609.08719) | **GoAnt** | 质量-多样性搜索；**map-independent effective-yield** 统一标尺 |
| 2026-08-27 | [2608.27734](https://arxiv.org/abs/2608.27734) | **What survives honest evaluation?** | ⚠️ 最重要：结构性防泄漏 + 按试验次数 deflate |
| 2026-08-24 | [2608.23808](https://arxiv.org/abs/2608.23808) | MinervaScore | DSR+PBO+SPA+MinTRL 五闸门robustness评分 |
| 2026-08-04 | [2608.11250](https://arxiv.org/abs/2608.11250) | AgonAlpha | **fresh-context 对抗评审员**（带重执行与否决权）+ 预算分配 |
| 2026-08-03 | [2608.01789](https://arxiv.org/abs/2608.01789) | **EC 视角综述** | 六组件分析框架 + 八维自治度评估框架（**本方案的评测骨架**） |
| 2026-07-29 | [2607.26642](https://arxiv.org/abs/2607.26642) | **AlphaSchema** | 显式语义探索空间（Event/Context/Qualities/Direction/Output）+ 代理模型 |
| 2026-07-09 | [2607.08332](https://arxiv.org/abs/2607.08332) | **X-Alpha** | 持久研究记忆 + 记忆驱动路由定义每轮搜索空间 |
| 2026-05-26 | [2606.20625](https://arxiv.org/abs/2606.20625) | **AlphaMemo** | **AST-diff 编辑母题**记忆 + 置信门控 + 非对称否决（[开源](https://github.com/jarrettyu/sspm)） |
| 2026-05-14 | [2605.15412](https://arxiv.org/abs/2605.15412) | QuantEvolver | 命名了**上下文爆炸/反馈漂移**问题 |
| 2026-04-29 | [2604.26747](https://arxiv.org/abs/2604.26747) | **Constrained LLM Agents** | 确定性引擎强制固定切分/选择闸门/成本；**append-only 实验轨迹** |
| 2026-03-26 | [2603.26797](https://arxiv.org/abs/2603.26797) | MemGuard-Alpha | LLM 记忆污染；**负面结果**：MIA 过滤无效 |
| 2026-01-29 | [2601.22119](https://arxiv.org/abs/2601.22119) | AlphaCFG | 上下文无关语法约束搜索空间 |

### 1.2 GitHub 生态观察

实测 `github.com/search` + API（按更新时间）：

- **microsoft/qlib** — 49.8k★，2026-10-30 仍活跃；已官方集成 RD-Agent
- **microsoft/RD-Agent** — 15.1k★，2026-10-21 活跃
- 新增 15+ 个 LLM alpha mining 个人repo（多为 2026-04 之后创建），**绝大多数 ★<10**
  - 共同特征：都在做"LLM 生成因子 + 回测"，**都没有做评估诚实性**
  - 值得注意：`chester1uo/Assay`（为 LLM agent 挖因子专门造的高性能回测引擎）
  - `jarrettyu/sspm`（AlphaMemo 官方开源，但仓库当前不可访问）

**生态判断**：生成侧已高度同质化、门槛极低；**评估侧是当前唯一的差异化空间**。
这与 §1.3 的文献转向互相印证。

### 1.3 最重要的转向：从"生成能力"到"评估诚实性"

2026-08 之后出现了一批**专门质疑本类系统**的论文。这是本方案 P0 的全部依据。

**[2608.27734] What survives honest evaluation?**
在 453 支美股 + 39 支 ETF 上，用两个前沿模型、搜索预算最多 100 个候选、重复 5 次：

> *"honest evaluation certifies passive benchmarks, **rejects every LLM-discovered strategy**"*

两个结构性论点：
1. **防泄漏必须是结构性的，不能靠统计修正**。一个故意泄漏的 oracle 打出 Sharpe 35，
   **完整通过了** Deflated Sharpe 和 PBO 检验。统计修正抓不住前瞻偏差。
2. **必须按搜索强度 deflate**。记录每一次评估，用试验次数修正报告业绩——
   最佳样本内 Sharpe 随试验数上升，但 deflation 门槛**上升得更快**。

**[2610.29500] Auditing LLM Alpha-Factor Mining**
审计一个真实的 LLM 挖因子项目，发现的问题与本项目**高度重合**：

> *"a mean over kept candidates is compared with a mean of per-round maxima"*（指标口径混用）
> *"excludes non-positive OOS outcomes, making its zero-reversal rate non-informative"*（队列不完整）

结论建议：**explicit metric grains, immutable cohort accounting, provenance-aware interpretation**。

**[2610.07701] On the Boundary of Admission Gates**（重要的反向平衡）
闸门不是免费的，这篇给出代价：
- 弱信号区：闸门能消除假发现，但**采纳率降到 1–7%**
- 强信号区：闸门**什么也没增加**
- ⚠️ **在绝对收益（而非超额收益）上计算的准则会静默地拒绝所有候选，包括真信号**

→ 所以 P0 不是"无脑加闸门"，而是**先把口径修对，再加闸门，且必须在超额收益上算**。

---

## 2. 本项目实测诊断

以下 5 项全部可在本仓库复现。

### 🔴 D1：ICIR 口径错位 —— "通过 0.5 门槛"的因子实际 ICIR 约 0.1

README 与 `factor_eval_agent.py:GATES` 写 `icir: 0.5`。实测 `momentum_factors.db`：

```
52w_high_drawdown_recovery_momentum   daily_ICIR=0.1060
price_to_52w_high_proximity           daily_ICIR=0.1011
classic_relative_momentum             daily_ICIR=0.0834
→ 13 条 PASS 记录的 ICIR 全部落在 0.075–0.106
```

根因：`mine_momentum.py:126` 把门槛改写为 `icir: 0.06`，但 README 与默认 `GATES` 仍宣称 0.5。
**这正是 [2610.29500] 批评的"指标口径混用"**——同一个"ICIR>门槛"的说法，在两处指不同的东西。

需要澄清的是：**0.5 和 0.1 不是同一个量**。
行业惯例的 ICIR≈0.5 指**月频/年化**口径；本项目算的是**日频** IC 的均值/标准差。
换算后 `0.106 × √252 ≈ 1.68` 年化，`t = 0.106 × √2248 ≈ 5.03`——
**单看显著性其实站得住**。问题不在数字本身，而在：

- 同一名字（ICIR）在 README、默认 GATES、实际运行中指三个不同口径
- 没有任何地方记录"这个 ICIR 是日频的"

→ 修法不是调数字，而是**把口径写进名字**（`icir_daily` / `icir_ann` / `ic_tstat`）。

### 🔴 D2：`monthly_win_rate` 根本不是月度

`factor_eval_agent.py:76`：
```python
monthly_win_rate = float(np.mean(ic_arr > 0))   # ic_arr 是逐日 IC
```
实测验证：DB 存 `0.5418`，重算 `np.mean(ic>0)` = `0.5418` —— 完全一致，**它是逐日胜率**。

后果：日频胜率天然贴近 0.5，`0.55` 门槛变成一个**几乎与信号质量无关**的随机筛子。
月度胜率的信息量（按月聚合 IC 再看正负）被完全丢失。

### 🔴 D3：`ic_decay_ratio` 门槛失效 —— 13/13 轻松通过

门槛 `ic_decay_ratio > 0.5`，实测分布：
```
min=1.06  median=1.40  max=1.61   → 全部远超门槛
```
根因：`ic_decay_ratio = ic_mean_5d / ic_mean_1d`，而 5 日**累计**收益的 IC 天然大于 1 日。
分子分母量纲不可比，比值系统性 >1。这个闸门目前**不筛任何东西**。

正确做法：比较**同量纲**的衰减（1d IC vs 5d IC 都用前向 1 日收益、滞后不同天数），
或直接拟合 IC 半衰期。

### 🔴 D4：用 IC 序列相关做去重 —— 几乎完全失效

`factor_eval_agent.py:_max_correlation_with_existing` 用**IC 时间序列**的 pearson 相关去重。
实测 13 个因子的 pairwise 相关矩阵：

```
                                  p52w  52wdd  26wdd  52wlog  52wpv  52wbm
price_to_52w_high_proximity       1.00   0.90   0.02   -0.01   0.02  -0.01
52w_high_drawdown_recovery        0.90   1.00  -0.01   -0.01   0.03   0.00
26w_high_drawdown_recovery        0.02  -0.01   1.00    0.01   0.03  -0.00
```

`26w_high_drawdown_recovery` 和 `52w_high_drawdown_recovery` —— **同一公式只改窗口长度**，
IC 序列相关只有 **-0.01**，轻松通过 0.7 门槛，双双入库。

根因：IC 序列相关衡量的是"两因子的**好坏时点**是否同步"，
而不是"两因子**选出的股票**是否相同"。后者才是去重要的东西。
正确做法：按日计算**因子值截面 rank 相关**，再对时间取均值。

另一个 bug：`max_correlation` **对入库顺序敏感**。先入库的因子永远记 `0.0`
（`price_to_52w_high_proximity` 记 0.0，而它与下一个的真实相关是 0.90）。
→ 这正是 [2610.29500] 说的 **immutable cohort accounting** 缺失。

### 🔴 D5：无样本外切分、无成本、无中性化、无多重检验修正

实测 grep 全仓库：

| 关键词 | 命中 | 实际情况 |
|---|---|---|
| `train/test/oos/split` | 0 | **全样本内评估**，无 OOS |
| `cost/slippage/commission/bps` | 0 | **无交易成本** |
| `turnover` | 1（仅注释） | **无换手率计算** |
| `market_cap` | 0 | **无市值中性化** |
| `neutral` | 1（仅 prompt 提示词） | 中性化只是"建议 LLM 这么做"，无强制 |

同时，搜索强度完全未被记录：
```
output_momentum/factors/*.py      = 47 个（即 47 次试验）
output_momentum_hf/factors/*.py   = 30 个
入库 PASS                          = 13
```
按 [2608.27734]，**77 次试验选出 13 个**必须按 N=77 deflate。实测 Bonferroni 门槛：
```
N= 30 → t 门槛 2.94
N= 60 → t 门槛 3.14
N=120 → t 门槛 3.34
```
本项目最佳因子 `t=5.03`，**在这一项上扛得住**（这是个好消息）。
但这纯属侥幸——系统既不记录 N，也不做修正，下一次就可能不扛得住。

另外 `mine_momentum.py:305` 有一处值得注意的设计退化：
```python
# Eval succeeded but gates failed: don't waste credits on repair calls.
return best_result
```
为省 API 成本，**修复回路在高频脚本里被整个关掉了**。`pipeline.py` 的修复逻辑在此路径完全不生效。
这是真实的成本压力，不是疏忽——[2610.17130] EvoQuantLab 的"内层 GP + 外层 LLM"正是针对它的解（见 P2）。

### 🟡 D6：架构与成本

- **LLM 模型过期**：`llm.py:38` 默认 `<中等模型>`、`<中等模型>`。当前应为 Claude 5 系
  （`<大模型>` / `<中等模型>` / `<小模型>`）
- **双份代码**：`mine_momentum.py`（630 行）与 `pipeline.py` 的核心循环大面积重复且已分叉
- **数据加载串行**：`pipeline.py:_get_price_data` 逐股 `ak.stock_zh_a_daily`，50 股约数分钟，无落盘缓存
- **每因子一次 pickle 全量数据**：`factor_implement_agent.py` 每次 `implement` 都 dump 一遍完整 price_data
- **上下文爆炸**：经验记忆直接拼进 prompt，未做预算控制（[2605.15412] 命名的 feedback drift）

---

## 3. 优化方案

### 设计原则

1. **评估诚实性优先于生成能力**。P0 完成前不扩展生成。
2. **结构性保证 > 统计修正**（[2608.27734]）。防泄漏靠 API 设计，不靠事后检验。
3. **闸门要付代价，所以口径必须先对**（[2610.07701]）。且必须在超额收益上算。
4. **用确定性引擎约束 LLM**（[2604.26747]）。切分、成本、闸门由引擎强制，LLM 不可协商。
5. **省钱靠架构，不靠砍掉正确性**（[2610.17130]）。

---

### P0：评估层重建（阻塞项，1–2 周）

> 这一阶段不增加任何新能力，只让现有结论变得可信。
> 完成后预期：**当前 13 个 PASS 会大幅减少**——这是正确的结果，不是退步。

#### P0-1　指标口径显式化（修 D1/D2/D3）

新建 `autoalpha/metrics.py`，所有指标**名字自带口径**：

```python
@dataclass(frozen=True)
class FactorMetrics:
    # ── 预测力（口径写进名字）──
    ic_mean_1d:      float   # 日频 IC 均值
    icir_daily:      float   # 日频 IC 均值/标准差
    icir_annual:     float   # icir_daily * sqrt(252)
    ic_tstat:        float   # icir_daily * sqrt(n_obs)   ← 显著性看这个
    n_obs:           int     # 有效 IC 观测数（队列记账）

    # ── 稳定性（真月度）──
    monthly_ic_mean: float   # 按月聚合 IC 的均值
    monthly_win_rate:float   # 按月聚合后 IC>0 的月份占比   ← 修 D2
    n_months:        int

    # ── 衰减（同量纲）──
    ic_by_lag:       dict    # {1: ic, 5: ic, 10: ic, 20: ic}，全部用前向1日收益+不同滞后
    ic_halflife:     float   # 由 ic_by_lag 拟合                ← 修 D3

    # ── 交易性 ──
    turnover_daily:  float   # 日均换手
    ic_after_cost:   float   # 扣 bps 成本后的净 IC

    # ── 去重（截面口径）──
    max_rank_corr:   float   # 与库内因子的截面 rank 相关（时间均值）← 修 D4

    # ── 搜索记账 ──
    trial_index:     int     # 这是第几次试验
    cohort_id:       str     # 不可变队列 ID
```

**废弃** `ic_decay_ratio = ic_5d/ic_1d`（量纲不可比）。

#### P0-2　样本外切分 + 确定性引擎（修 D5，最高价值单项）

参照 [2604.26747]，新建 `autoalpha/engine.py`。切分由引擎持有，**LLM 侧完全看不到 OOS 数据**：

```python
class DeterministicEngine:
    """切分、成本、闸门由引擎强制，不可被 LLM 或 prompt 协商。"""

    def __init__(self, data, is_end="2023-06-30", embargo_days=21,
                 cost_bps=5.0, cohort_id=None):
        # IS: 开始 → is_end        ← 唯一可见于搜索/修复的区间
        # embargo: is_end → +21d   ← 禁运期，防重叠窗口泄漏
        # OOS: embargo 后 → 结束   ← 仅在最终认证时解锁，一个因子只测一次
        ...

    def evaluate_is(self, factor_fn) -> FactorMetrics:
        """搜索循环只能调这个。"""

    def certify_oos(self, factor_fn, n_trials: int) -> Certification:
        """仅对通过 IS 闸门的因子调用一次。按 n_trials deflate。"""
```

三条硬规则：
- 修复回路、经验记忆、extension 生成**只能看 IS 指标**
- OOS **每个因子只测一次**，结果写入不可变记录
- `n_trials` 由引擎自己累加，调用方无法伪报

#### P0-3　截面 rank 相关去重（修 D4）

```python
def cross_sectional_rank_corr(f1: pd.Series, f2: pd.Series) -> float:
    """按日算截面 spearman，再对时间取均值。
    这才衡量'两因子是否选出同一批股票'。"""
    df = pd.concat([f1.rename("a"), f2.rename("b")], axis=1).dropna()
    daily = df.groupby(level="date").apply(
        lambda g: g["a"].corr(g["b"], method="spearman") if len(g) >= 10 else np.nan)
    return float(daily.mean())
```

配套修 `max_correlation` 的顺序敏感性：**入库后重算全库对称相关矩阵**，
而不是只记"与入库时已存在的因子"的相关。

验收：`26w_high_drawdown_recovery` vs `52w_high_drawdown_recovery` 的相关
必须从 `-0.01` 变成一个**高值**（预期 >0.7，应被判重复）。

#### P0-4　搜索强度记账 + deflation（[2608.27734] 核心）

```python
class SearchLedger:
    """append-only，不可删改。每次 evaluate 都记一笔。"""
    def log(self, factor_name, metrics, outcome): ...     # 含失败、含负 IC
    def n_trials(self) -> int: ...
    def deflated_tstat_threshold(self, alpha=0.05) -> float:
        return norm.ppf(1 - alpha / max(self.n_trials(), 1))
```

两条要求，直接对应 [2610.29500] 的批评：
- **记全队列**：失败的、负 IC 的、JUDGE 拒的，全部入账。不允许只记成功的。
- **报告必须同时给出** `n_trials` 与 deflate 后的门槛。

同时按 [2610.07701] 的警告：**所有准则在超额收益（相对基准）上计算**，
绝不在绝对收益上算——否则会静默拒绝全部候选。

#### P0-5　成本与换手（修 D5）

```python
turnover_t = 0.5 * Σ_i |w_{i,t} - w_{i,t-1}|      # 单边换手
ic_after_cost = ic_gross - cost_bps * 1e-4 * turnover_daily * scaling
```
按 [2610.23744]，因子的真实价值是**成本调整后对现有池的边际贡献**，
不是它自己的 IC。P1-2 会把这个作为核心闸门。

#### P0 验收标准

```
□ 重跑 momentum_factors.db 的 13 个因子，输出新旧指标对照表
□ 预期：PASS 数显著下降（这是对的）
□ 预期：26w/52w drawdown_recovery 被判为重复
□ 报告必须显示 n_trials 与 deflated 门槛
□ 任何一处 ICIR 都带口径后缀，README 与代码一致
```

---

### P1：搜索质量（2–4 周）

#### P1-1　AlphaSchema 显式语义空间（[2607.26642]）

当前 `factor_extract_agent` 从摘要自由生成，探索空间隐式、不可控。
改为先生成**结构化 schema plan**，再翻译成代码：

```python
@dataclass
class SchemaPlan:
    event:     str   # 触发信号：gap / breakout / reversal / volume_spike
    context:   str   # 条件：high_vol_regime / post_earnings / low_liquidity
    qualities: list  # 度量：persistence / magnitude / consistency / asymmetry
    direction: str   # long_high / long_low
    output:    str   # rank / zscore / binary
```

两个直接好处：
- 探索空间可枚举、可覆盖率统计（替代 BrainAlpha 的网格）
- [2607.26642] 实测：**同一 schema 由不同 LLM 实现，预测质量相当**
  → 可以把贵模型用在 schema 规划、便宜模型用在代码翻译（见 P2-1）

#### P1-2　池感知评估（[2610.04959] / [2610.23744]）

当前因子**独立**过闸门。改为评估**对现有池的边际贡献**：

```python
def marginal_contribution(new_factor, pool) -> float:
    ic_before = pool_ic(pool)
    ic_after  = pool_ic(pool + [new_factor])
    return ic_after - ic_before          # 扣成本后
```

这一改动会让 D4 那种"同族因子"自然被淘汰——它们边际贡献接近 0。

#### P1-3　AST-diff 编辑母题记忆（[2606.20625] AlphaMemo）

当前经验记忆存的是自然语言 `key_insight`，检索靠 `LIKE '%keyword%'`（`db.py:168`）—— 太粗。
改为按 [2606.20625]：

- 从**父子因子的 AST 差异**抽取编辑母题（如"加 ewm 平滑"、"改为行业相对"）
- 记录每个母题在**特定父因子上下文**下的成功/失败
- **置信门控**：样本不足的母题不参与建议
- **非对称否决**：高置信失败母题直接禁用（失败信号比成功信号更可信）

#### P1-4　谱系级搜索控制（[2610.29663] LineageEvo）

当前 extension 对每个 PASS 因子独立生成 3 个变体，分支会垄断预算。
加入谱系记账：
- 每个因子带 `lineage_id`，extension 继承父谱系
- 预算按**谱系**分配，而非按因子——防止一个成功分支吃掉全部预算
- 这直接缓解 [2609.29014] 命名的 **research path collapse**

#### P1-5　fresh-context 对抗评审（[2608.11250] AgonAlpha）

当前 `factor_judge_agent` 的两个问题：
- 与生成共享上下文，容易"自己批准自己"
- `judge()` 异常时**默认 APPROVE**（`factor_judge_agent.py:84`）—— 失败是静默放行的

改为：
- **全新上下文**，评审员只看代码+指标，不看生成过程与假设动机
- 有**重执行权**：独立跑一遍代码，核对指标是否可复现
- 有**否决权**，且异常时默认 **REJECT**（fail-closed）

---

### P2：成本与工程（并行推进）

#### P2-1　分层模型路由（省钱的正确做法）

D5 暴露的"为省钱关掉修复回路"要用架构解决，而不是砍掉正确性：

| 环节 | 模型 | 理由 |
|---|---|---|
| schema 规划 / 评审 | `<大模型>` | 需要深度推理 |
| 代码翻译 | `<小模型>` | [2607.26642] 证明此环节对模型不敏感 |
| 变体生成 | `<中等模型>` | 平衡 |

同时修 `llm.py:38` 的过期默认模型，并加 **prompt caching**（price_data schema、算子库描述是跨调用不变的）。

#### P2-2　内层 GP + 外层 LLM（[2610.17130] EvoQuantLab）

这是最大的 token 效率杠杆：
- **内层**：参数/结构的局部搜索交给**遗传编程**，零 LLM 调用
- **外层**：LLM 只做诊断与协议级改进（"这一族都失败了，换个信号族"）

预期：同等搜索覆盖下 LLM 调用量降一个数量级，**修复回路可以重新打开**。

#### P2-3　工程债

```
□ 合并 mine_momentum.py 与 pipeline.py 的核心循环（已分叉，630 行重复）
□ 数据层：parquet 落盘缓存 + 并发拉取（当前串行逐股，50股数分钟）
□ 因子数据传递：改用一次性共享内存/parquet，替代每因子 pickle 全量
□ 经验记忆加 token 预算上限（修 D6 上下文爆炸）
□ 补 pytest：metrics.py 与 engine.py 必须有单测（这是正确性的地基）
```

---

### P3：长期（选做）

- **八维自治度自评**（[2608.01789]）：搜索效率/适应度可靠性/残差alpha质量/经济多样性/
  可交易性/进化自治性/非稳健性鲁棒/可复现性 —— 作为项目的体检表
- **MinervaScore 式综合评分**（[2608.23808]）：DSR+PBO+SPA+MinTRL 五闸门
  ⚠️ 注意原文自己的保留："作为审计与报告层，而非未来可盈利性的证据"
- **跨市场迁移验证**（[2602.07085] QuantaAlpha 的 CSI300→CSI500/S&P500）
- **多目标 Pareto 而非加权标量**（[2609.34188]）

---

## 4. 执行顺序

```
P0（阻塞，1-2周）  指标口径 → OOS引擎 → 截面去重 → 搜索记账 → 成本换手
        ↓          验收：重跑13因子，PASS数下降，同族被判重复
P1（2-4周）        AlphaSchema → 池感知 → AST记忆 → 谱系预算 → 对抗评审
        ↓
P2（并行）         模型路由 → GP内层 → 工程债
        ↓
P3（选做）         自治度自评 / MinervaScore / 跨市场
```

**不要跳过 P0。** 在评估不可信时增强生成，只会更快地生产不可信的因子——
这正是 [2608.27734] 用 100 个候选 × 5 次重复 × 2 个前沿模型实测出的结论。

---

## 5. 一句话总结

> 项目的架构直觉是对的，而且恰好与 2026 下半年的文献共识同向。
> 但 5 个可实测的评估缺陷（ICIR 口径、胜率非月度、衰减闸门失效、
> 去重用错口径、无 OOS/成本/多重检验）使当前"13 个因子通过门控"的结论不可采信。
>
> 2026-08 之后文献的集体转向说明了原因：**LLM 挖因子的瓶颈早已不是生成，而是评估。**
> 先修评估层，再谈搜索。
