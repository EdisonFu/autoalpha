# FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework

**ArXiv**: https://arxiv.org/abs/2603.16365
**作者**: Qinhong Lin, Ruitao Feng, Yinglun Feng et al.
**年份**: 2026

---

## 核心贡献

FactorEngine 的核心设计哲学是**三重职责分离**，并引入**程序级知识注入**——从非结构化金融研报中提取可执行因子逻辑，构建结构化知识库，再用知识库指导因子挖掘。

---

## 系统架构

### 三重分离原则
```
分离1：逻辑修订 vs 参数优化
  - 逻辑修订：由 LLM 负责（需要推理能力）
  - 参数优化：由贝叶斯优化负责（需要精确搜索）
  - 不混用，各司其职

分离2：LLM 生成 vs 规则验证
  - LLM 生成候选
  - 确定性规则验证（无需 LLM）
  - 验证通过后才进入下一步

分离3：LLM 推理 vs 本地计算
  - LLM 只做推理（不做数值计算）
  - 数值计算全部在本地执行
```

### 整体流程
```
非结构化金融研报/论文
    ↓
知识引导的 Bootstrap 模块
（提取-验证-代码化 三步提取）
    ↓
结构化知识库（因子逻辑 + 参数空间）
    ↓
多智能体提取-验证-生成循环
    ↓
轨迹感知的精细化
（利用历史轨迹指导当前优化）
    ↓
因子评估 + 入库
```

---

## 核心创新

### 1. 程序级知识提取（Knowledge-Infused Bootstrap）
将非结构化文本（研报、论文）转化为**可执行因子程序**，分三步：

**提取**：LLM 从文本中识别投资逻辑
```
输入："高换手率往往意味着散户追涨，机构出逃，后续收益下降"
提取：{类型: 流动性反向因子, 数据: 换手率, 方向: 负相关}
```

**验证**：确定性规则检查提取结果的合法性
```
- 数据字段是否存在？
- 因子方向是否与假设一致？
- 参数范围是否合理？
```

**代码化**：将验证后的知识转为 Python 函数

### 2. 轨迹感知的精细化（Trajectory-Aware Refinement）
不是从头重试，而是利用历史轨迹：
```
历史轨迹：假设A → 代码A1(失败,低IC) → 代码A2(失败,高换手) → 代码A3(通过)
当前：假设B（类似A）→ 直接参考 A1/A2 的失败原因 → 生成代码B1（跳过已知坑）
```

### 3. 贝叶斯参数优化
逻辑确定后，用贝叶斯优化（Bayesian Optimization）精细化参数：
- 回溯期 N：在 [5, 120] 范围内搜索最优值
- 阈值 k：在历史数据上用高斯过程建模 IC(N, k)
- 比网格搜索更高效，比人工调参更系统

---

## 实验结果

- IC/ICIR：显著高于基准方法
- 夏普比率：投资组合夏普提升
- 稳定性：跨不同市场状态更稳健

---

## 对 AutoAlpha 的启发

### 1. 三重分离思想（架构层面）
我们当前的 `factor_implement_agent` 让 LLM 同时负责逻辑设计和参数选择。可以拆分：
- **LLM**：只负责因子逻辑结构（用什么数据、怎么计算）
- **规则引擎**：验证参数合法性
- **参数搜索**：在合法范围内搜索最优参数（暂用简单网格搜索）

### 2. 轨迹感知的错误修复（直接可做）
在错误修复时，不只传入当前错误，还传入**同类假设的历史失败经验**：

```python
def _get_similar_failure_history(self, spec: FactorSpec) -> str:
    """检索相似假设的历史失败经验"""
    similar = self.db.search_similar_failures(spec.hypothesis_type)
    if not similar:
        return ""
    lines = [f"类似假设的历史失败："]
    for s in similar[:3]:
        lines.append(f"  - {s['failure_pattern']}：{s['fix_applied']} → {s['fix_outcome']}")
    return "\n".join(lines)
```

### 3. 研报知识提取
将我们的 `paper_search_agent` 扩展：不只提取"有哪些因子假设"，还提取"具体的量化逻辑"（数据字段、计算方式、参数范围），构建更丰富的 `FactorSpec`。
