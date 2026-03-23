# BrainAlpha: End-to-End Multi-Agent Alpha Discovery on WorldQuant Brain

**来源**: 知乎专栏文章
**年份**: 2025
**评估环境**: WorldQuant Brain WebSim（生产级基础设施）

---

## 核心贡献

BrainAlpha 是目前**工程化程度最高**的全自动 Alpha 挖掘系统，直接对接 WorldQuant Brain 生产级仿真环境。核心创新：将评估失败从"黑盒二元结果"升级为"强类型诊断信号"，驱动结构化修复。

---

## 系统架构

### 六模块 LangGraph 流水线
```
M0：假设生成（优先级加权探索网格）
    ↓
M1：RAG 规范化（20,000 字段向量索引）
    ↓
M2：表达式生成（Jaccard 多样性门控）
    ↓
M3：Brain 预言机评估（生产级仿真）
  ├── PASS → 输出日志
  ├── 常规失败 → M4 修复引擎
  ├── 系统错误 → M3.5 错误恢复
  └── 不可挽救 → 丢弃，M0 新生成
M4：修复引擎（三阶段，收敛控制）
SAO：信号 Alpha 对象（全局共享状态）
```

---

## 核心创新

### 1. 结构化探索网格（660 个离散单元）
```
11（信号族） × 5（算子模式） × 3（时间跨度） × 4（数据处理）= 660
```
每个单元格状态：{空置 → 已分配 → 探索中 → 已耗尽}

优先级公式：
```
Priority(c) = 0.5 × Novelty(c) + 0.3 × ExpectedYield(c) + 0.2 × AgentAlignment(c)
```
其中 ExpectedYield 每发现一个有效因子 +0.10（在线学习）。

### 2. 故障模式分类体系（FM-1 至 FM-6）
| 代码 | 判断条件 | 首选修复 |
|---|---|---|
| FM-1 低信号质量 | 夏普低/PnL 平坦 | 行业中性化 + 延长回溯期 |
| FM-2 过度换手 | 换手超上限 | decay_linear 平滑 + 加长回溯 |
| FM-3 高回撤 | 最大回撤超限 | Winsorization + 波动率条件化 |
| FM-4 信号反转 | 收益显著为负 | 乘以 -1 或倒数反转 |
| FM-5 低覆盖率 | 持仓数未达最低要求 | pasteurize() + 放宽过滤 |
| FM-6 低换手 | 换手低于活跃度要求 | 缩减回溯期 + 加大排序变化率 |

### 3. M4 修复引擎（三阶段）
**阶段一：可恢复性评估（硬停止）**
- I1：自相关性审查失败（结构性重复）
- I2：复杂信号（AST > 20节点）+ 信号反转
- I3：失败检查项 ≥ 4（多脏器衰竭）

**阶段二：确定性策略选择**（不用 LLM）
按失败类型查表，执行有序修复列表。

**阶段三：低温度 LLM 重写**（T=0.3）
- 携带完整历史（原始假设 + 失败记录 + 缺口百分比）
- 五道物理熔断：通过/达到最大迭代/停滞/恶化/策略耗尽

### 4. RAG 字段锚定
20,000 个有效字段名向量化，每次生成前 KNN 检索最相关字段，消灭"字段幻觉"。

---

## 实验结果

- 单实例产能匹敌一支小型量化研究团队
- 每额外产出一个有效信号的边际成本趋近于零

---

## 对 AutoAlpha 的启发（最直接可用）

### 1. 故障模式分类 → 定向修复（高优先级）
将我们的评估失败结果映射到因子层面的故障分类：

```python
def classify_factor_failure(result: FactorResult) -> str:
    if result.ic_mean < 0:
        return "FM-4_REVERSAL"        # 信号反转
    if result.coverage < 0.5:
        return "FM-5_LOW_COVERAGE"    # 低覆盖率
    if result.ic_mean < GATES['ic_mean']:
        return "FM-1_LOW_QUALITY"     # 低信号质量
    if result.icir < GATES['icir']:
        return "FM-1_LOW_STABILITY"   # 低稳定性
    return "FM-UNKNOWN"
```

### 2. 修复硬停止（立即可做）
在进入 LLM 修复前，先做可恢复性评估：
```python
def is_irrecoverable(code: str, result: FactorResult) -> Tuple[bool, str]:
    ast_nodes = len(list(ast.walk(ast.parse(code))))
    if result.ic_mean < -0.05 and ast_nodes > 30:
        return True, "复杂信号严重反转，无法通过修复解决"
    # ... 其他硬停止条件
```

### 3. 收敛控制（立即可做）
```python
class RepairLoop:
    MAX_ITER = 5
    STAGNATION_DELTA = 0.005  # IC 提升小于此值视为停滞

    def should_stop(self, ic_history: List[float]) -> Tuple[bool, str]:
        if len(ic_history) >= self.MAX_ITER:
            return True, "达到最大迭代次数"
        if len(ic_history) >= 3:
            recent_improvement = ic_history[-1] - ic_history[-3]
            if recent_improvement < self.STAGNATION_DELTA:
                return True, "连续迭代无显著改善（停滞）"
        if len(ic_history) >= 2 and ic_history[-1] < ic_history[-2]:
            return True, "指标恶化"
        return False, ""
```
