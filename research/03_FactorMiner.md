# FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery

**ArXiv**: https://arxiv.org/abs/2602.14670
**作者**: Yanlong Wang, Jian Xu, Hongkang Zhang, Shao-Lun Huang et al.
**年份**: 2026

---

## 核心贡献

FactorMiner 的核心理念：**智能体应该像人类研究员一样，从每次挖掘经历中学习并成长**。引入了两个关键机制：模块化技能库（Skills）和挖掘经验记忆（Experience Memory），让系统随着运行次数增加而越来越聪明。

---

## 系统架构

### RALPH 循环（核心范式）
```
Retrieve（检索相关经验和技能）
    ↓
Act（执行：生成假设 → 构造因子 → 评估）
    ↓
Learn（从结果中学习：成功/失败原因）
    ↓
Prune（剪枝：移除低质量技能和经验）
    ↓
Harvest（收获：将优质因子入库）
    ↓
→ 回到 Retrieve，开始下一轮
```

### 模块组成
```
技能库（Skill Library）
  - 封装金融领域操作为可复用工具
  - 例：行业中性化技能、IC计算技能、因子标准化技能
  ↕
经验记忆（Experience Memory）
  - 存储历史挖掘轨迹的结构化摘要
  - 记录：假设类型 → 成功率 → 失败原因 → 改进建议
  ↕
RALPH 执行循环
```

---

## 核心创新

### 1. 经验记忆（Experience Memory）
不是简单地存储因子结果，而是存储**结构化的挖掘经验**：
```json
{
  "hypothesis_type": "momentum",
  "operator_family": "ts_rank",
  "lookback": 20,
  "outcome": "PASS",
  "ic_mean": 0.035,
  "key_insight": "行业中性化后IC从0.015提升至0.035",
  "failure_pattern": null
}
```

失败案例同样被记录：
```json
{
  "hypothesis_type": "reversal",
  "operator_family": "diff",
  "lookback": 1,
  "outcome": "FAIL",
  "failure_pattern": "过度换手",
  "fix_applied": "decay_linear(x, 5)",
  "fix_outcome": "PASS"
}
```

### 2. 技能封装（Skill Encapsulation）
金融评估操作被封装为可调用工具，LLM 通过工具调用而非自由代码生成来执行评估。降低幻觉风险，提高可靠性。

### 3. 蒸馏机制（Distillation）
每 N 轮后，对经验记忆进行蒸馏：
- 合并相似经验
- 提炼高频成功/失败模式为通用规则
- 避免记忆库无限膨胀

### 4. 相关性红海约束（Correlation Red Sea）
新因子与已有因子库的最大相关性 < 0.7，否则不入库（与我们的评估门控一致）。

---

## 实验结果

- 提供 110 个 A 股权益因子，均有显式公式表达
- 因子间相关性低，多因子组合效果好
- 与深度学习黑盒方法相比，可解释性更强

---

## 对 AutoAlpha 的启发

### 1. 经验记忆系统（高优先级，中等难度）
在现有 SQLite `factor_library` 表之外，增加 `mining_experience` 表：

```sql
CREATE TABLE mining_experience (
    id INTEGER PRIMARY KEY,
    hypothesis_type TEXT,      -- 假设类型（动量/反转/波动等）
    operator_family TEXT,       -- 算子族
    lookback_range TEXT,        -- 回溯期范围
    neutralization TEXT,        -- 中性化方式
    outcome TEXT,               -- PASS/FAIL
    ic_mean REAL,
    icir REAL,
    failure_pattern TEXT,       -- 失败模式（低IC/高换手等）
    fix_applied TEXT,           -- 应用的修复措施
    fix_outcome TEXT,           -- 修复后结果
    key_insight TEXT,           -- 本次挖掘的关键洞察
    created_at TIMESTAMP
);
```

### 2. 检索增强生成（RAG on Experience）
在 `factor_implement_agent` 生成代码前，先检索类似假设的历史经验：
```
"类似的动量因子，行业中性化后IC提升明显，建议默认加上"
"回溯期 5 天的反转因子换手率过高，建议使用 decay_linear 平滑"
```
这些经验作为上下文注入 prompt，让 LLM 少走弯路。

### 3. RALPH 循环结构（对标我们的 pipeline）
当前我们的流水线是线性的，可以演化为循环：
```
当前：论文 → 提取 → 实现 → 评估 → 延伸（一次性）
目标：[检索经验] → 实现 → 评估 → [记录经验] → 延伸 → [检索经验] → ...
```

### 4. 技能封装思路
将我们的 `factor_eval_agent` 的各项评估函数（IC计算、覆盖率计算等）封装为独立工具，LLM 通过工具调用获取评估结果，而不是在代码中手动解析。
