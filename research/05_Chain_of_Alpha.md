# Chain-of-Alpha: Unleashing the Power of LLMs for Alpha Mining in Quantitative Trading

**ArXiv**: https://arxiv.org/abs/2508.06312
**作者**: Lang Cao
**年份**: 2025

---

## 核心贡献

Chain-of-Alpha 将 Chain-of-Thought（思维链）推理范式引入 Alpha 挖掘，设计了**双链协同架构**：生成链（Factor Generation Chain）和优化链（Factor Optimization Chain）交替工作，形成迭代改进的闭环。

---

## 系统架构

### 双链架构
```
┌────────────────────────────────────┐
│     Factor Generation Chain        │
│                                    │
│  Step 1: 市场观察（Market Insight）  │
│     ↓                              │
│  Step 2: 假设构建（Hypothesis）      │
│     ↓                              │
│  Step 3: 数学表达（Formalization）   │
│     ↓                              │
│  Step 4: 代码生成（Implementation）  │
└──────────────┬─────────────────────┘
               ↓
           回测评估
               ↓
    ┌──────────┴──────────┐
    │ PASS                │ FAIL
    ↓                     ↓
  入库               ┌────────────────────────────────────┐
                     │     Factor Optimization Chain       │
                     │                                    │
                     │  Step 1: 失败诊断（Diagnosis）       │
                     │     ↓                              │
                     │  Step 2: 改进方向（Direction）       │
                     │     ↓                              │
                     │  Step 3: 代码修订（Revision）        │
                     │     ↓                              │
                     │  Step 4: 验证（Verification）        │
                     └──────────────┬─────────────────────┘
                                    ↓
                               回测评估（循环）
```

---

## 核心创新

### 1. 生成链的分步推理
不是直接生成代码，而是强制 LLM 经过四个明确的推理步骤：
1. **市场观察**：描述该类市场现象的经济学基础
2. **假设构建**：将观察转化为可验证的量化假设
3. **数学表达**：写出严格的数学公式
4. **代码生成**：仅在前三步确认后才生成代码

这种分步结构减少了 LLM 直接跳到代码导致的逻辑漂移。

### 2. 优化链的诊断-方向-修订三步
失败后不是简单重试，而是：
1. **诊断**：分析失败的根本原因（低IC？高换手？）
2. **方向**：确定改进方向（加中性化？调参数？换公式？）
3. **修订**：只修改相关部分，保留有效部分

### 3. 双链协同
生成链和优化链共享同一个 `FactorSpec` 对象，优化链可以修改任何层次（假设/公式/代码），而不局限于代码层。

---

## 对 AutoAlpha 的启发

### 1. 生成链分步推理（直接可做）
将 `factor_implement_agent` 的代码生成提示词从单步改为四步：

```python
_CODEGEN_SYSTEM = """
You are a quantitative researcher. Generate a factor step by step:

Step 1 - Market Insight: What market phenomenon does this factor capture?
Step 2 - Hypothesis: State the precise testable hypothesis
Step 3 - Formula: Write the mathematical formula
Step 4 - Code: Implement the formula in Python

Only proceed to Step 4 after completing Steps 1-3.
"""
```

### 2. 优化链替代当前修复逻辑
当前我们把错误信息直接塞回 LLM 让其修复。改为三步诊断流程：

```python
def _repair_with_optimization_chain(self, spec, code, error_msg, eval_result):
    # Step 1: 诊断
    diagnosis = self.llm.complete(
        system="分析因子失败的根本原因",
        user=f"因子：{spec.factor_name}\n失败：{error_msg}\n评估：{eval_result}"
    )
    # Step 2: 方向
    direction = self.llm.complete(
        system="基于诊断，确定改进方向",
        user=f"诊断：{diagnosis}"
    )
    # Step 3: 修订（低温度）
    new_code = self.llm.complete(
        system="精确修订代码（temperature=0.3）",
        user=f"原代码：{code}\n改进方向：{direction}"
    )
    return new_code
```

### 3. 假设层修订
不只修改代码，有时需要修改假设本身。当同一假设连续 3 次修复失败时，触发假设层修订（重新提取更简单的假设）。
