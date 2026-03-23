# R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization

**ArXiv**: https://arxiv.org/abs/2505.15155
**作者**: Yuante Li, Xu Yang, Xiao Yang, Minrui Xu et al.（微软研究院）
**年份**: 2025

---

## 核心贡献

R&D-Agent-Quant 将整个量化研究流程分解为**五个 LLM 驱动的专业单元**，在严格的闭环反馈中协同工作。核心创新是**赌徒调度器（Bandit Scheduler）**：将因子探索视为多臂赌博机问题，动态分配迭代资源给最有潜力的方向。

---

## 系统架构

### 五单元闭环
```
┌─────────────────────────────────────────────┐
│  Specification Unit（规范单元）               │
│  将研究目标转化为结构化技术规范                  │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Synthesis Unit（综合单元）                   │
│  整合先验知识生成因子假设                       │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Implementation Unit（实现单元）               │
│  LLM 生成因子计算代码，沙箱执行                  │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Validation Unit（验证单元）                   │
│  回测 + 多维度指标评估                          │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Analysis Unit（分析单元）                    │
│  分析失败原因，生成改进建议，反馈给 Synthesis      │
└──────────────────┬──────────────────────────┘
                   ↓
              Bandit Scheduler
              （决定下一轮重点探索哪个方向）
                   ↓
             回到 Synthesis Unit
```

---

## 核心创新

### 1. 赌徒调度器（Bandit Scheduler）
将不同研究方向视为多臂赌博机的各条"臂"：
- 每条臂代表一类假设（动量/反转/波动/流动性等）
- 每次成功因子发现 → 更新该臂的置信上界（UCB）
- 调度器优先分配计算资源给 UCB 最高的方向
- **平衡探索（新方向）与利用（高产方向）**

```python
# UCB 公式
ucb(arm) = mean_reward(arm) + sqrt(2 * ln(total_trials) / trials(arm))
```

### 2. 数据驱动 + 模型联合优化
不只优化单个因子，而是同时优化因子集合的整体预测能力：
- 单因子评估合格后，加入候选池
- 定期对候选池进行联合优化（最大化因子间低相关性）

### 3. 分析单元的结构化反馈
分析单元不只输出"失败"，而是输出结构化诊断：
```json
{
  "failure_type": "low_coverage",
  "root_cause": "数据稀疏导致因子值缺失超过30%",
  "recommended_fix": "使用更宽松的 dropna 条件，或用行业均值填充",
  "alternative_hypothesis": "考虑换用成交量而非换手率作为流动性代理"
}
```

---

## 实验结果

| 指标 | 数值 |
|---|---|
| 年化收益 | 约 2× 于基准因子库 |
| 有效因子数量 | 减少 70%（更精炼） |
| 计算预算效率 | 在小计算预算下超越深度时序模型 |

---

## 对 AutoAlpha 的启发

### 1. 赌徒调度器（中期引入，高价值）
当前我们对每篇论文的因子方向平等对待。引入 UCB 调度后，系统会自动学习"哪类假设更容易出好因子"：

```python
class BanditScheduler:
    def __init__(self, arms: List[str]):  # arms = 假设类型列表
        self.counts = {a: 0 for a in arms}
        self.rewards = {a: 0.0 for a in arms}

    def select(self) -> str:
        total = sum(self.counts.values()) + 1
        ucb_scores = {
            a: (self.rewards[a] / max(self.counts[a], 1))
               + sqrt(2 * log(total) / max(self.counts[a], 1))
            for a in self.counts
        }
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        self.rewards[arm] += reward  # reward = ic_mean if PASS else 0
```

### 2. 分析单元结构化反馈（直接可做）
将现有的错误信息从自由文本改为结构化诊断，供下次生成时使用。

### 3. 因子联合优化（长期目标）
当因子库积累到 50+ 个因子后，定期运行联合优化：选出低相关、高IC的最优子集。

### 4. 规范单元的使用
当前我们直接从论文摘要提取 `FactorSpec`，可以增加规范验证步骤，确保 spec 中的字段（数据需求、参数范围等）在我们的数据中是可用的。
