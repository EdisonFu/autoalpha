# AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery

**ArXiv**: https://arxiv.org/abs/2506.13131
**作者**: Alexander Novikov, Ngân Vũ et al.（Google DeepMind）
**年份**: 2025

---

## 核心贡献

AlphaEvolve 是 Google DeepMind 开发的进化式代码智能体，通过**多个 LLM 协作进化**来发现科学和算法领域的突破性解法。虽然不专注于量化因子，但其**代码级进化框架**和**评估器驱动的进化策略**对 AutoAlpha 有极高参考价值。

---

## 系统架构

### 多 LLM 协作进化
```
算法种群（初始代码库）
    ↓
┌──────────────────────────────────────┐
│  进化循环                              │
│                                      │
│  1. 选择（Selection）                  │
│     锦标赛选择：从种群中选 K 个父代     │
│     ↓                                │
│  2. 变异 LLM（Variation LLM）         │
│     创造性强，负责提出新思路（T=0.8）   │
│     ↓                                │
│  3. 细化 LLM（Refinement LLM）       │
│     精确性强，负责完善代码（T=0.2）     │
│     ↓                                │
│  4. 评估器（Evaluator）               │
│     运行代码，返回客观分数              │
│     ↓                                │
│  5. 存档（Archive）                   │
│     维护 Pareto 前沿（性能×多样性）     │
└──────────────────────────────────────┘
    ↓
收敛 → 最优解
```

### 核心组件
- **变异 LLM**：专门负责创造性变异（高温度）
- **细化 LLM**：专门负责代码完善（低温度）
- **客观评估器**：确定性打分，不受 LLM 幻觉影响
- **多样性档案（MAP-Elites）**：保留多样化的优秀个体

---

## 核心创新

### 1. 角色分离的多 LLM 协作
不同的 LLM（或同一 LLM 的不同配置）专注不同角色：
- **探索 LLM**（高温度）：冒险提出激进新想法
- **利用 LLM**（低温度）：精细化打磨现有思路

这种分工避免了单个 LLM 在探索和利用之间的矛盾。

### 2. Pareto 档案（MAP-Elites）
不只保留"最高分"的个体，而是维护一个多维度的 Pareto 前沿：
- 维度1：性能（IC 值）
- 维度2：多样性（与其他因子的差异）
- 维度3：稳健性（IC 的时序稳定性）

每个维度空间的最优个体都被保留，确保种群多样性。

### 3. 客观评估器的重要性
LLM 不负责评判自己生成代码的质量（会产生自我偏见），所有评估由确定性的客观评估器完成。这与 BrainAlpha 的"P1 关注点绝对分离"原则一致。

### 4. 突破性成果
- 找到了比 Strassen 算法更快的 4×4 矩阵乘法（56年来首次改进）
- 在多个 NP-hard 问题上超越人类已知最优解

---

## 对 AutoAlpha 的启发

### 1. 双 LLM 角色分离（中期引入）
将我们的因子生成和修复分配给不同角色的 LLM 调用：

```python
class DualLLMFactorAgent:
    def generate(self, spec: FactorSpec, data: dict) -> str:
        """探索 LLM：高温度，鼓励创造性"""
        return self.llm.complete(
            system=_CODEGEN_SYSTEM,
            user=...,
            temperature=0.7  # 高温度：探索
        )

    def refine(self, code: str, feedback: str) -> str:
        """细化 LLM：低温度，精确修复"""
        return self.llm.complete(
            system=_REFINE_SYSTEM,
            user=f"代码：{code}\n改进反馈：{feedback}",
            temperature=0.2  # 低温度：利用
        )
```

### 2. Pareto 因子档案
当前我们只记录"通过"的因子。改为维护 Pareto 档案：
- 每个因子按（IC值、多样性、稳定性）三维归档
- 报告中展示 Pareto 前沿的因子集合
- 延伸时优先从 Pareto 前沿选择起点

### 3. 进化式评估不依赖 LLM
评估永远由确定性代码执行（IC 计算、覆盖率等），不用 LLM 评判因子质量的数值。LLM 只负责生成和解读，不负责量化评估。这是我们已经在做的，要坚持这个原则。

### 4. 多目标优化框架
将我们的质量门控从"全部通过才算 PASS"改为多目标评分：
```python
def compute_quality_score(result: FactorResult) -> float:
    """Pareto 综合评分"""
    scores = {
        'ic': min(result.ic_mean / 0.06, 1.0),      # 归一化到1
        'stability': min(result.icir / 1.0, 1.0),
        'coverage': result.coverage,
        'novelty': 1 - result.max_correlation,
    }
    return sum(scores.values()) / len(scores)
```
