# Alpha²: Discovering Logical Formulaic Alphas using Deep Reinforcement Learning

**ArXiv**: https://arxiv.org/abs/2406.16505
**作者**: Feng Xu, Yan Yin, Xinyu Zhang, Tianyuan Liu, Shengyi Jiang, Zongzhang Zhang
**年份**: 2024

---

## 核心贡献

Alpha² 用**深度强化学习（DRL）**代替遗传规划（GP）来搜索因子表达式空间，同时引入**维度分析（Dimensional Analysis）**确保因子的数学/逻辑有效性。解决了 GP 对初始解敏感、容易陷入局部最优的问题。

---

## 系统架构

### RL 搜索框架
```
状态（State）：当前部分构建的因子表达式树
    ↓
策略网络（Policy Network）
  - 决定下一步选择什么算子/操作数
  - 基于 Transformer 对表达式树编码
    ↓
动作（Action）：选择一个节点扩展
  - 算子选择（ts_rank, diff, divide...）
  - 操作数选择（close, volume, open...）
    ↓
奖励（Reward）：回测 IC 值
  - 正奖励：IC > 阈值
  - 额外奖励：多样性（与现有因子不重复）
  - 惩罚：维度不合法 / 过于复杂
    ↓
策略梯度更新
```

### 维度分析（Dimensional Analysis）
每个算子都有明确的输入/输出维度约束：
```
rank(x) → [0,1]       输出范围受限
diff(x) → 差分序列    保持时序维度
divide(x, y) → 比率   x 和 y 需要同维度
```
不合法的维度组合 → 立即给负奖励，强制生成逻辑自洽的因子。

---

## 核心创新

### 1. DRL 取代 GP 的优势
| 方面 | 遗传规划(GP) | Deep RL |
|---|---|---|
| 初始解敏感性 | 高 | 低 |
| 局部最优 | 容易陷入 | 策略梯度可逃出 |
| 长程规划 | 无 | 有（未来奖励折扣） |
| 可迁移性 | 无 | 策略可迁移到新问题 |
| 计算速度 | 慢 | GPU 加速 |

### 2. 多目标优化
同时优化两个目标：
1. **预测能力**（IC 最大化）
2. **多样性**（与已有因子不相似）

在奖励函数中加入多样性项：
```
reward = IC + λ × diversity_bonus
diversity_bonus = max(0, 1 - max_similarity_to_library)
```

### 3. 经验回放（Experience Replay）
将历史探索的成功/失败因子存入经验池，训练时重放，提高样本利用率。

---

## 实验结果

- 超越 GP 方法（同等计算量下找到更多有效因子）
- 超越随机搜索方法
- 维度约束大幅减少无效表达式比例（从 60% 降至 20%）
- 代码已开源

---

## 对 AutoAlpha 的启发

### 1. 维度/逻辑有效性检查（直接可做）
在代码生成后，增加基于规则的逻辑合法性验证：

```python
def validate_factor_logic(self, code: str) -> Tuple[bool, str]:
    """检查因子代码的逻辑合法性"""
    checks = []

    # 检查1：输出必须是 pd.Series 且有 MultiIndex
    if 'MultiIndex' not in code and 'groupby' not in code:
        checks.append("缺少截面处理（MultiIndex/groupby）")

    # 检查2：必须有归一化步骤（rank 或 zscore）
    if 'rank' not in code and 'zscore' not in code:
        checks.append("缺少截面归一化（建议加 rank 或 zscore）")

    # 检查3：除法操作需要防止除以零
    if '/' in code and 'replace(0' not in code and 'clip' not in code:
        checks.append("除法操作缺少零值保护")

    # 检查4：回溯期不能超过数据范围
    # ...

    return len(checks) == 0, "; ".join(checks)
```

### 2. 多样性奖励在因子延伸中的应用
当 `factor_extension_agent` 评估延伸因子时，除了 IC 之外也考虑多样性：
```
综合评分 = 0.7 × IC得分 + 0.3 × (1 - 与库最大相关)
```
优先推荐既有预测力又有多样性的延伸因子。

### 3. RL 的长期方向
当因子库达到 100+ 后，可以考虑用 RL 来学习"什么样的假设类型在什么市场环境下更容易出好因子"，取代当前基于规则的调度。
