# AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay

**ArXiv**: https://arxiv.org/abs/2502.16789
**作者**: Ziyi Tang, Zechuan Chen, Jiarui Yang et al.（中山大学、新南威尔士大学、南洋理工、香港中文大学深圳）
**年份/会议**: 2025，KDD '25

---

## 核心贡献

AlphaAgent 是首个系统性解决 **Alpha 衰减** 问题的 LLM 驱动因子挖掘框架。核心观点：Alpha 衰减的根本原因是新因子与已有因子在结构上高度同质，以及过于复杂的表达式导致过拟合。通过三重正则化约束从源头遏制衰减。

---

## 系统架构

三智能体闭环流水线：

```
Idea Agent（假设生成）
    ↓  Chain-of-Thought 推理
Factor Agent（因子构造）
    ↓  标准算子库 + 三重正则化
Eval Agent（因子评估）
    ↓  回测 + IC/ICIR 指标
    ↓  失败 → 反馈 → 重新生成
```

---

## 三大正则化机制（核心创新）

### 1. 原创性约束（Originality）
- 将新因子代码解析为 **抽象语法树（AST）**
- 计算与已有因子库中所有因子的 AST 结构相似度
- 相似度超过阈值 → 直接丢弃，重新生成
- **目的**：防止 LLM 换个参数就当新因子，从结构上保证多样性

### 2. 假设对齐（Hypothesis Alignment）
- 生成代码后，用 LLM 验证：**"这段代码是否真的实现了原始假设？"**
- 对齐得分低 → 代码逻辑偏离假设 → 重新生成
- **目的**：解决 LLM 生成语法正确但经济含义跑偏的问题

### 3. 复杂度控制（Complexity）
- 对因子表达式的 AST 节点深度/数量设置上限
- 过于复杂的嵌套表达式 → 过拟合风险高 → 衰减快
- **目的**：奥卡姆剃刀原则，简单因子更稳健

---

## 实验结果

| 指标 | 结果 |
|---|---|
| 有效因子生成率 | 提升 81%（vs 基准方法） |
| Token 消耗 | 减少 30% |
| IC 值 | 约 0.02，跨 2021-2024 年稳定 |
| CSI 500 年化超额 | 11.0%（IR=1.5） |
| S&P 500 年化超额 | 8.74%（IR=1.05） |

传统因子 IC 在 2-3 年内衰减至零，而 AlphaAgent 生成的因子 IC 保持稳定。

---

## 对 AutoAlpha 的启发

### 直接可引入的设计

**1. AST 原创性检查（高优先级）**
```python
import ast

def compute_ast_similarity(code1: str, code2: str) -> float:
    """比较两段代码的 AST 结构相似度"""
    # 提取 AST 节点类型序列
    nodes1 = set(type(n).__name__ for n in ast.walk(ast.parse(code1)))
    nodes2 = set(type(n).__name__ for n in ast.walk(ast.parse(code2)))
    # Jaccard 相似度
    return len(nodes1 & nodes2) / len(nodes1 | nodes2)

# 在 factor_eval_agent 入库前检查：
# if max(compute_ast_similarity(new_code, existing) for existing in library) > 0.75:
#     return "DUPLICATE"
```

**2. 假设对齐验证（高优先级）**
在 `factor_implement_agent` 生成代码后、沙箱执行前，增加验证步骤：
```
System: 你是代码审查员，判断代码是否实现了指定假设
User: 假设: {spec.hypothesis}
      代码: {generated_code}
      请回答：1-10 分，这段代码实现了假设的核心逻辑吗？
```
分数低于 6 → 直接重新生成，节省沙箱执行时间。

**3. 复杂度限制**
生成代码后计算 AST 节点数，超过 50 个节点的因子表达式设警告，超过 100 个节点强制简化。

### 量化收益预期
- 减少无效沙箱执行（假设对齐过滤）
- 减少重复因子入库（AST 原创性过滤）
- 提高因子库长期稳定性（复杂度控制）
