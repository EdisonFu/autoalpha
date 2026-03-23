# Automate Strategy Finding with LLM in Quant Investment

**ArXiv**: https://arxiv.org/abs/2409.06289
**作者**: Zhizhuo Kou, Holam Yu, Junyu Luo et al.（香港科技大学）
**年份**: 2024（最新修订 2025年11月）

---

## 核心贡献

将 Alpha 挖掘扩展到**多模态数据源**：不只从价格/量数据挖掘因子，还从**研究报告文本、K线图像**中提取投资信号。同时引入**动态权重优化**，根据市场环境调整策略权重。

---

## 系统架构

### 三阶段框架
```
阶段一：多模态信号生成
  ├── 数值模块：从 OHLCV 数据生成技术因子
  ├── 文本模块：从研报/新闻中提取情绪/事件因子
  └── 视觉模块：分析 K 线图识别形态信号

阶段二：多市场状态评估
  ├── 分市场状态（牛/熊/震荡）分别回测
  ├── 评估因子在各状态下的有效性
  └── 标记每个因子的"最适市场状态"

阶段三：动态权重优化
  ├── 识别当前市场状态
  ├── 激活适合当前状态的因子
  └── 实时调整因子组合权重
```

### 多模态 LLM 处理
```
数值数据 → 标准技术指标 → LLM 因子描述
研究报告 → LLM 文本理解 → 情绪/主题因子
K线图像 → 视觉 LLM → 图形形态因子
      ↓ 融合
  多模态因子池
```

---

## 核心创新

### 1. 多模态因子融合
三种模态捕捉不同层次的市场信息：
- **数值**：价格动量、波动率等定量特征
- **文本**：市场情绪、政策预期等定性特征
- **视觉**：头肩顶、双底等图形模式

### 2. 市场状态分类
用机器学习方法将历史行情分类为若干状态：
- 趋势牛市（高涨幅、低回撤）
- 趋势熊市（高跌幅、高回撤）
- 高波动震荡
- 低波动震荡

每种状态下哪类因子最有效，事先标定。

### 3. 增强风险管理
识别因子在极端市场条件下的行为：
- 最大单日损失
- 尾部风险（5% 分位 IC）
- 危机期间（2015年股灾/2020年疫情）的表现

---

## 实验结果

| 指标 | 数值 |
|---|---|
| 累计收益（SSE50，2023年） | 53.17% |
| 相比基准超额 | 显著 |
| 下行保护 | 改善（最大回撤降低） |

---

## 对 AutoAlpha 的启发

### 1. 文本模态扩展（中期引入）
当前我们从论文中提取因子，可以扩展到财经新闻和研报：

```python
class TextFactorAgent:
    """从文本中提取情绪/事件类因子"""
    def extract_text_signals(self, news_text: str) -> FactorSpec:
        # 识别：利好/利空事件 → 事件因子
        # 识别：行业景气度描述 → 行业情绪因子
        # 识别：管理层言论 → 基本面预期因子
        pass
```

### 2. 市场状态分层评估（直接扩展）
在 `factor_eval_agent` 中，增加按市场状态分层的 IC 分析：

```python
def _compute_conditional_ic(self, factor_series, returns, market_states):
    """按市场状态分别计算 IC"""
    results = {}
    for state in ['bull', 'bear', 'volatile', 'calm']:
        dates = market_states[market_states == state].index
        if len(dates) > 20:
            ic_state = self._compute_ic_for_dates(
                factor_series, returns, dates
            )
            results[f'ic_{state}'] = ic_state
    return results
```

### 3. 因子适用市场标记
在 `FactorResult` 中增加字段，标记该因子在哪种市场状态下最有效：
```python
best_market_state: str = ""  # "bull" / "bear" / "volatile" / "all"
conditional_ics: dict = field(default_factory=dict)
```
在报告中展示给用户，帮助择时使用因子。
