# AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors

**ArXiv**: https://arxiv.org/abs/2406.18394
**作者**: Hao Shi, Weili Song, Xinting Zhang et al.
**年份**: 2024，AAAI 2025

---

## 核心贡献

AlphaForge 解决了一个重要问题：**单个因子的有效性会随市场状态变化，但当前大多数系统都使用固定权重组合多个因子**。AlphaForge 引入了**动态因子组合**机制：根据当前市场状态实时调整各因子的权重。

---

## 系统架构

### 两阶段框架
```
阶段一：因子挖掘（生成-预测神经网络）
  - 生成模块：探索因子表达式空间
  - 预测模块：预测每个因子在未来的IC
  - 两者协同训练，生成"高预期IC"的因子

阶段二：动态因子组合
  - 输入：因子库 + 当前市场状态特征
  - 时序模型：学习市场状态 → 因子有效性的映射
  - 输出：各因子的动态权重
  - 实时调整组合权重
```

### 动态权重模型
```
市场状态特征（波动率、趋势强度、换手率...）
    ↓
LSTM / Transformer 时序模型
    ↓
各因子权重向量 w(t)
    ↓
组合信号 = Σ w_i(t) × factor_i
```

---

## 核心创新

### 1. 因子有效性预测
不只评估因子的历史 IC，还训练模型预测因子**未来的 IC**：
- 特征：市场状态、因子历史 IC 序列
- 标签：未来 N 期的 IC
- 用于：在组合时对"当前最可能有效的因子"加权

### 2. 生成-预测联合训练
生成模块（探索新因子）和预测模块（评估潜力）联合优化：
- 生成模块的目标：生成预测模块打分高的因子
- 预测模块的目标：准确预测因子未来 IC
- 形成良性循环：越来越精准地生成高IC因子

### 3. 时序感知的动态组合
识别市场状态切换：
- 趋势市 → 动量因子权重上升
- 震荡市 → 反转因子权重上升
- 高波动 → 低换手稳健因子权重上升

---

## 实验结果

- 超越传统固定权重因子组合
- AAAI 2025 接收，业界认可度高
- 在 A 股和美股均有验证

---

## 对 AutoAlpha 的启发

### 1. 因子动态权重（中期引入）
当我们的因子库积累一定数量后，不用等权平均，而是基于市场状态动态调权：

```python
class DynamicFactorCombiner:
    def get_weights(self, factors: List[str], market_state: dict) -> dict:
        """
        根据市场状态返回各因子的当前建议权重
        market_state: {'volatility': 'high', 'trend': 'bull', 'turnover': 'high'}
        """
        # 简化版：基于规则的动态权重
        weights = {f: 1.0 for f in factors}
        if market_state['trend'] == 'bull':
            # 牛市加大动量因子权重
            for f in factors:
                if 'momentum' in f.lower():
                    weights[f] *= 1.5
        if market_state['volatility'] == 'high':
            # 高波动降低短期因子权重
            for f in factors:
                if '_1d' in f or '_3d' in f:
                    weights[f] *= 0.6
        # 归一化
        total = sum(weights.values())
        return {f: w/total for f, w in weights.items()}
```

### 2. 因子 IC 预测
在评估 IC 的历史均值之外，增加 IC 趋势分析：
- 近 6 个月 IC 是否在下降？（衰减信号）
- IC 在不同市场状态下的分布？（鲁棒性）
- 这些信息写入 `FactorResult` 供报告展示

### 3. 组合层输出
我们目前只输出单因子报告。长期目标：增加"最优因子组合"推荐，选出低相关、高 IC 的因子子集作为组合建议。
