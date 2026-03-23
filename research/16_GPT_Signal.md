# GPT-Signal: Generative AI for Semi-automated Feature Engineering in Alpha Research

**ArXiv**: https://arxiv.org/abs/2410.18448
**作者**: Yining Wang, Jinman Zhao, Yuri Lawryshyn
**年份**: 2024，FINNLP 2024

---

## 核心贡献

GPT-Signal 是一个**轻量级半自动特征工程**框架，使用 GPT-4 从原始金融数据中自动构造候选特征（因子），显著降低量化研究员在特征工程阶段的工作量。强调"半自动"：人类定义数据集和目标变量，AI 自动探索特征组合。

---

## 系统架构

```
研究员输入
  - 数据集（OHLCV + 基本面数据）
  - 预测目标（未来N日收益率）
  - 约束条件（不使用未来数据）
    ↓
GPT-4 特征工程引擎
  Step 1：分析数据结构，理解每列的金融含义
  Step 2：提出假设性特征（"过去20天的价格动量可能预测未来收益"）
  Step 3：生成特征计算代码
  Step 4：执行代码，计算特征值
    ↓
特征评估模块
  - 相关性过滤（去除与已有特征高度相关的）
  - 预测能力筛选（IC 阈值）
  - 稳定性检验
    ↓
最终特征集
```

---

## 核心创新

### 1. 数据结构感知的特征生成
GPT-4 不只是随机组合算子，而是先**理解每个数据列的金融含义**：
```
"close" → 收盘价，可用于计算收益率、动量
"volume" → 成交量，可用于流动性、换手率
"pe_ratio" → 市盈率，可用于估值因子
```
这种理解使生成的特征更有经济意义。

### 2. 隐式因子标准化
自动处理常见的数据质量问题：
- **前瞻偏差检测**：确保特征计算不使用未来数据
- **极端值处理**：自动加入 winsorize/clip
- **截面归一化**：自动加入 rank/zscore

### 3. 迭代式特征精化
初始生成的特征经评估后，将评估结果返回给 GPT-4：
- IC 低：提示 GPT-4 尝试不同的时间窗口
- 与现有特征高度相关：提示生成更有差异的特征
- 数据缺失率高：提示改变特征计算方式

---

## 实验结果

- FINNLP 2024 接收
- 在多个数据集上验证了自动特征工程的有效性
- 与人工特征工程相比，减少了 60% 的研究员时间投入

---

## 对 AutoAlpha 的启发

### 1. 数据结构感知的代码生成（直接可做）
在 `factor_implement_agent` 的 prompt 中，加入数据列的金融含义解释：

```python
_DATA_COLUMN_MEANINGS = {
    'close': '收盘价（用于计算收益率、动量、均值回归）',
    'volume': '成交量（用于流动性、资金流入/出信号）',
    'open': '开盘价（用于隔夜跳空、开盘效应）',
    'high': '最高价（用于波动率、支撑压力位）',
    'low': '最低价（用于下行风险、支撑位）',
    'turnover': '换手率（用于流动性、情绪指标）',
}

# 在 prompt 中注入：
# "各数据列的金融含义：close=收盘价（动量/均值回归）..."
```

### 2. 前瞻偏差自动检测
在代码执行后，增加前瞻偏差检测：

```python
def check_lookahead_bias(self, code: str) -> bool:
    """检查是否存在常见的前瞻偏差模式"""
    dangerous_patterns = [
        'shift(-',          # 向前移位（使用未来数据）
        'rolling().last()', # 末尾值可能包含当日
        'expanding().max()',# 扩展窗口可能引入未来
    ]
    return any(p in code for p in dangerous_patterns)
```

### 3. 自动数据质量处理模板
在代码生成 prompt 中，明确要求处理常见数据问题：
```
必须处理：
1. 前瞻偏差：所有时序计算使用 .shift(1) 或 closed='left'
2. 极端值：使用 clip(lower=-3std, upper=3std) 或 winsorize
3. 截面归一化：最终因子值使用 rank/count 归一化到 [0,1]
4. 缺失值：明确指定 min_periods 参数
```
