# Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment

**ArXiv**: https://arxiv.org/abs/2402.09746
**作者**: Hang Yuan, Saizhuo Wang, Jian Guo
**年份**: 2024

---

## 核心贡献

Alpha-GPT 2.0 在第一代基础上，将人机协作从"单因子交互"扩展到**整个量化投资流程**：从市场分析、因子挖掘、组合构建到风险管理，每个环节都支持人机协同。核心突破是**迭代式交互研究（Iterative Interactive Research）**范式。

---

## 系统架构

### 全流程覆盖
```
市场分析层
  - LLM 分析宏观/行业趋势
  - 人类确认/修正分析结论
    ↓
因子挖掘层（继承1.0的AlphaBot）
  - 扩展：支持多因子同时探索
  - 扩展：因子间相关性可视化
    ↓
组合构建层
  - LLM 推荐因子权重分配
  - 人类调整约束条件
    ↓
风险管理层
  - LLM 识别组合风险敞口
  - 压力测试报告生成
    ↓
绩效归因层
  - LLM 解释收益来源
  - 自动生成投资报告
```

### 迭代式交互研究
```
人类：提出研究方向/约束
  ↓
AI：执行研究，返回结果+建议
  ↓
人类：接受/修改/扩展
  ↓
AI：基于反馈迭代
  ↓
（循环直到满意）
```

---

## 核心创新

### 1. 协作式发现（Collaborative Discovery）
不只是"AI 帮人执行"，而是真正的协作：AI 会主动提出人类可能没想到的探索方向，人类会修正 AI 的方向偏差。

### 2. 研究员专长融合
系统可以捕获特定研究员的偏好和专长：
- 研究员A：擅长基本面因子 → AI 在基本面方向加权
- 研究员B：擅长技术因子 → AI 在技术方向加权

### 3. 增强的 Alpha 诊断
引入"因子健康度"评分，涵盖：
- 时间稳定性（不同年份的 IC 方差）
- 市场环境适应性（牛市/熊市 IC 对比）
- 尾部风险（IC 最差 5% 的表现）

---

## 对 AutoAlpha 的启发

### 1. 因子健康度评分（直接扩展）
在我们的 `FactorResult` 中增加：

```python
@dataclass
class FactorResult:
    # 现有字段...
    ic_stability: float = 0.0      # IC 跨年稳定性（1 - CV）
    bull_ic: float = 0.0           # 牛市环境下的 IC
    bear_ic: float = 0.0           # 熊市环境下的 IC
    worst_5pct_ic: float = 0.0     # 最差 5% IC（尾部风险）
    health_score: float = 0.0      # 综合健康度评分
```

健康度计算：
```python
health_score = (
    0.4 * ic_mean_score +        # IC 绝对水平
    0.3 * ic_stability_score +   # 稳定性
    0.2 * icir_score +           # 信息比率
    0.1 * tail_risk_score        # 尾部风险
)
```

### 2. 研究方向记忆
记录用户过去成功的研究方向，新会话时优先推荐：
```
过去成功：动量类因子（平均IC=0.038）
建议：继续探索动量类因子的变体
```

### 3. 压力测试集成
在因子评估中增加市场状态分层分析：
- 按年度/季度分组计算 IC
- 识别因子在哪些市场状态下失效
- 在报告中展示稳定性分布
