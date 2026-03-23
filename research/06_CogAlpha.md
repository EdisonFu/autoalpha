# CogAlpha: Cognitive Alpha Mining via LLM-Driven Code-Based Evolution

**ArXiv**: https://arxiv.org/abs/2511.18850
**作者**: Fengyuan Liu, Huang Yi, Sichun Luo, Yuqi Wang et al.
**年份**: 2025

---

## 核心贡献

CogAlpha 将 LLM 定位为**认知智能体**，融合了三种能力：领域知识推理（LLM）、系统性搜索（进化算法）、以及精确表达（代码表示）。核心洞察：纯进化搜索缺乏经济直觉，纯 LLM 缺乏系统性探索能力，两者结合才能在解空间中既"聪明"又"全面"。

---

## 系统架构

```
初始因子种群
（来自文献、经验或随机生成）
    ↓
┌───────────────────────────────┐
│  LLM 认知进化循环              │
│                               │
│  1. 理解（Comprehension）      │
│     分析当前因子的经济含义       │
│     ↓                         │
│  2. 变异（Mutation）           │
│     LLM 提出有经济意义的变异     │
│     ↓                         │
│  3. 交叉（Crossover）          │
│     LLM 融合两个父代因子的优点   │
│     ↓                         │
│  4. 代码化（Codification）     │
│     变异/交叉结果转为精确代码     │
│     ↓                         │
│  5. 适应度评估                 │
│     IC/ICIR/覆盖率             │
└───────────────────────────────┘
    ↓
精英保留（Top-K 因子存活到下一代）
    ↓
收敛 → 输出因子库
```

---

## 核心创新

### 1. 经济意义引导的变异（Semantic Mutation）
普通进化算法：随机替换算子（SMA→EMA）
CogAlpha：让 LLM 提出**有经济逻辑的**变异方向：

```
原因子：5 日收益率动量
LLM 变异提议："考虑加入成交量加权，因为动量在高成交量时更可靠"
变异结果：5 日成交量加权收益率
```

### 2. 概念驱动的交叉（Concept-Driven Crossover）
不是简单地拼接两个因子的代码，而是让 LLM 理解两个因子的经济含义后，提出有意义的融合方式：

```
父代A：短期反转因子（1日）
父代B：波动率因子（20日）
LLM 交叉提议："在高波动环境下强化反转信号"
子代：条件反转因子（高波动时用1日反转，低波动时用5日动量）
```

### 3. 代码级别表示（Code-based Representation）
以往进化算法多用表达式树（Expression Tree）。CogAlpha 直接以 Python 代码作为基因型，允许更复杂的逻辑结构（条件、循环、多步计算）。

### 4. 多阶段提示策略
- 理解阶段：T=0.3（精准解析已有因子）
- 变异阶段：T=0.7（创造性提出新方向）
- 代码化阶段：T=0.2（精确实现）

---

## 实验结果

- A 股权益数据，2018-2023
- 预测精度、收益稳定性：优于纯神经网络方法和纯符号搜索方法
- 生成因子具有明确经济解释，监管友好

---

## 对 AutoAlpha 的启发

### 1. 进化式因子扩展（替代当前 extension_agent）
当前 `factor_extension_agent` 让 LLM 自由提出 3-5 个变体，缺乏系统性。改为进化框架：

```python
class EvolutionaryExtensionAgent:
    def mutate(self, spec: FactorSpec, result: FactorResult) -> List[FactorSpec]:
        """有经济意义的变异"""
        prompt = f"""
        原因子: {spec.factor_name}
        经济含义: {spec.hypothesis}
        IC: {result.ic_mean:.4f}

        提出 3 个有经济逻辑支撑的变异方向（不只是换参数）：
        每个变异必须解释为什么它比原因子更能捕捉这个市场现象
        """
        ...

    def crossover(self, spec_a: FactorSpec, spec_b: FactorSpec) -> FactorSpec:
        """概念驱动的双因子交叉"""
        prompt = f"""
        因子A: {spec_a.factor_name} - {spec_a.hypothesis}
        因子B: {spec_b.factor_name} - {spec_b.hypothesis}

        这两个因子在什么市场条件下可以互补？
        请设计一个融合两者优点的新因子。
        """
        ...
```

### 2. 多阶段温度控制
在我们的 LLM 调用中区分生成阶段和修复阶段的温度：
- 假设生成：T=0.7（需要创造性）
- 代码生成：T=0.5（需要准确但允许多样）
- 错误修复：T=0.2（需要精确手术）

### 3. 精英因子保留策略
不是每次都从论文重新开始，维护一个"精英因子池"：
- 每次挖掘后保留 IC > 0.04 的因子作为精英
- 新一轮挖掘时，精英因子可以作为进化的起点
