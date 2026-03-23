# Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning

**ArXiv**: https://arxiv.org/abs/2512.23515
**作者**: Zuoyou Jiang, Li Zhao, Rui Sun, Ruohan Sun et al.（微软）
**年份**: 2025

---

## 核心贡献

Alpha-R1 训练了一个 **80 亿参数的推理型 LLM**（通过 RL 微调），专门用于**动态评估和激活/停用因子**。核心洞察：一个因子是否有效，不只取决于因子本身，还取决于**当前市场状态**——市场环境在变，因子权重也应该动态变化。

---

## 系统架构

```
市场数据 + 新闻/事件
    ↓
Alpha-R1 推理模型（8B 参数，RL 训练）
  - 输入：市场状态特征 + 候选因子 + 实时事件
  - 输出：每个因子的当前"激活权重"
    ↓
动态因子组合
  - 高权重因子：正常使用
  - 低权重因子：降权或停用
    ↓
投资组合构建
```

### RL 训练机制
```
初始：预训练 LLM（理解金融知识）
    ↓
RL 微调：
  - 奖励：基于动态组合的回测收益
  - 动作：为每个因子分配权重
  - 状态：市场状态 + 因子历史表现
    ↓
学习到：什么时候用什么因子
```

---

## 核心创新

### 1. 推理驱动的因子激活
不是简单的规则（"牛市用动量"），而是让推理型 LLM 综合分析：
- 当前宏观环境（利率、通胀趋势）
- 近期重大事件（政策变化、行业冲击）
- 因子近期表现（IC 是否下降？）
- 因子间相关性变化

```
推理过程示例：
"当前市场处于政策宽松期，流动性充裕，动量因子历史上在这种环境下
IC均值为0.042，高于均值。同时，上周央行降准，短期内成长风格可能
受益。建议：动量因子权重×1.3，价值因子权重×0.8。"
```

### 2. 跨资产池验证
在多个资产池验证：CSI 300、CSI 500、CSI 1000，以及不同市场状态（牛市、熊市、震荡）。

### 3. 实时新闻融合
将实时新闻事件作为因子激活决策的输入，使系统能快速响应市场突发事件。

---

## 实验结果

- 跨多个资产池、多市场状态，稳定优于传统策略
- 在市场状态切换时（牛转熊/熊转牛）表现尤为突出
- 8B 参数推理模型在实际部署中的延迟可接受

---

## 对 AutoAlpha 的启发

### 1. 因子健康度监控（直接可做）
定期重新评估因子库中已通过的因子，检测是否出现衰减：

```python
class FactorHealthMonitor:
    def check_decay(self, factor_name: str, recent_window: int = 60) -> dict:
        """检查因子近期 IC 是否相比历史显著下降"""
        recent_ic = self._compute_recent_ic(factor_name, recent_window)
        historical_ic = self.db.get_factor_ic(factor_name)
        decay_ratio = recent_ic / max(historical_ic, 1e-6)
        return {
            'factor': factor_name,
            'historical_ic': historical_ic,
            'recent_ic': recent_ic,
            'decay_ratio': decay_ratio,
            'status': 'DECAYING' if decay_ratio < 0.7 else 'HEALTHY'
        }
```

### 2. 市场状态感知的因子报告
在 HTML 报告中增加"当前市场适用性"分析：
```
当前市场状态：高波动（VIX > 25）
  - momentum_20d：历史上高波动期 IC=0.012，低于均值，建议降权
  - reversal_5d：历史上高波动期 IC=0.041，高于均值，建议升权
```

### 3. 长期：RL 驱动的因子调度
当因子库达到 30+ 后，可以训练一个轻量级的因子调度策略：
- 状态：市场特征向量（波动率、趋势、换手率等）
- 动作：各因子权重
- 奖励：组合 IC

暂时可以用规则代替，积累数据后再上 RL。
