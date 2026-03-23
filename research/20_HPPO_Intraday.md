# Mining Intraday Risk Factor Collections via Hierarchical Reinforcement Learning

**ArXiv**: https://arxiv.org/abs/2501.07274
**作者**: Wenyan Xu, Jiayu Chen, Dawei Xiang et al.
**年份**: 2025

---

## 核心贡献

HPPO（Hierarchical PPO）将**分层强化学习**与**迁移学习**结合，专门用于挖掘高频（日内）风险因子。核心创新：用两层 PPO 模型分别学习"选择哪些特征"和"如何组合特征"，捕捉传统方法难以发现的非线性关系。

---

## 系统架构

### 分层 RL 框架
```
第一层 PPO（高层策略）：特征权重分配
  - 状态：当前市场快照（N 个特征）
  - 动作：为每个特征分配权重向量
  - 目标：识别当前最重要的特征子集
    ↓
第二层 PPO（低层策略）：非线性关系发现
  - 状态：加权后的特征向量
  - 动作：选择特征间的组合算子
  - 目标：找到预测收益的最优非线性变换
    ↓
因子输出：非线性组合的风险因子
    ↓
优化信号：皮尔逊相关性（实际收益 vs 预测收益）
```

### 迁移学习机制
```
在源市场（CSI 300）训练的策略
    ↓
迁移到目标市场（Nifty 100, S&P 500）
    ↓
少量微调（Fine-tune）
    ↓
适应新市场，无需从头训练
```

---

## 核心创新

### 1. 分层决策（Hierarchical Decision）
第一层解决"用什么特征"（特征选择），第二层解决"怎么用特征"（特征组合）。两层分工明确，避免了单层 RL 同时处理两个问题时的维度爆炸。

### 2. 隐性非线性关系挖掘
传统因子挖掘主要基于线性/单调关系。HPPO 可以发现：
- **条件关系**：只在高波动时有效的特征
- **交互效应**：两个特征的乘积比各自单独更有预测力
- **阈值效应**：某特征超过阈值才开始有效

### 3. 跨市场迁移
训练好的因子挖掘策略可以迁移到新市场，减少数据需求。这对新兴市场（数据不足）尤其有价值。

### 4. 优化信号：皮尔逊相关性
用实际收益与预测收益的皮尔逊相关系数作为 RL 奖励，与 IC 直接挂钩，优化目标与评估指标完全一致。

---

## 实验结果

| 市场 | 超额收益 |
|---|---|
| CSI 300 | 25%（高频交易） |
| CSI 800 | 显著正超额 |
| Nifty 100 | 迁移后有效 |
| S&P 500 | 迁移后有效 |

---

## 对 AutoAlpha 的启发

### 1. 非线性因子探索提示（代码生成层面）
在我们的代码生成 prompt 中，鼓励 LLM 探索非线性组合：

```python
_NONLINEAR_HINTS = """
除了简单的线性因子，也可以考虑：
1. 条件因子：if condition: use factor_A else use factor_B
   示例：高波动环境使用长期动量，低波动使用短期反转
2. 交互因子：factor_A × factor_B（两个独立信号的协同效应）
   示例：动量 × (1/换手率) = 低换手动量
3. 分位数条件：只在某个指标处于极端分位时激活信号
   示例：只在成交量 > 80 分位时使用价格反转
"""
```

### 2. 跨市场泛化性评估
在因子评估中增加简单的市场泛化性检验：
- 主要在 A 股数据上评估
- 可选：在港股/美股（如果有数据）上验证
- 泛化性好的因子在报告中标注

### 3. 皮尔逊 IC 与 Spearman IC 双报告
我们当前使用 Spearman 秩相关（ICIR）。HPPO 用皮尔逊相关。两者各有侧重：
- Spearman：对异常值鲁棒，更保守
- Pearson：对线性关系更敏感，可发现 Spearman 遗漏的线性预测力

```python
def _compute_ic_dual(self, factor_vals, forward_returns):
    from scipy.stats import spearmanr, pearsonr
    spearman_ic, _ = spearmanr(factor_vals, forward_returns)
    pearson_ic, _ = pearsonr(factor_vals, forward_returns)
    return {'spearman_ic': spearman_ic, 'pearson_ic': pearson_ic}
```

### 4. 分层评估视角
虽然不实现 RL，但可以借鉴分层思维评估因子：
- 第一层：因子选择（IC 绝对值 > 0.03？）
- 第二层：因子组合（与已有因子相关性 < 0.7？能增加组合 IC？）
两层都通过，才是真正有价值的新因子。
