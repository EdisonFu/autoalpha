# AlphaPROBE: Alpha Mining via Principled Retrieval and On-graph Biased Evolution

**ArXiv**: https://arxiv.org/abs/2602.11917
**作者**: Taian Guo, Haiyang Shen, Junyu Luo, Binqi Chen et al.
**年份**: 2026

---

## 核心贡献

AlphaPROBE 将因子之间的演化关系建模为**有向无环图（DAG）**，每个节点是一个因子，有向边表示"从A变异得到B"或"A和B交叉得到C"的关系。通过贝叶斯检索找到最有潜力的"种子节点"，再沿图进行偏置进化。

---

## 系统架构

```
因子 DAG
（节点=因子，边=演化关系）
       ↓
贝叶斯因子检索器（Bayesian Factor Retriever）
  - 输入：研究目标
  - 输出：最有潜力的种子因子（Top-K 节点）
  - 依据：节点历史产出率 + 后代质量
       ↓
DAG 感知因子生成器（DAG-aware Factor Generator）
  - 以种子因子为起点
  - 利用完整的祖先链信息生成下一代因子
  - 避免与任何祖先高度重复
       ↓
上下文感知优化
  - 根据当前市场状态调整因子参数
  - （牛市/熊市/高波动/低波动 → 不同优化方向）
       ↓
评估 → 写入 DAG 新节点
```

---

## 核心创新

### 1. 因子 DAG（Factor Directed Acyclic Graph）
```
momentum_5d
    ├── momentum_10d（参数变异）
    ├── vol_adj_momentum_5d（与波动率交叉）
    │       └── vol_adj_momentum_10d
    └── industry_neutral_momentum_5d（中性化变异）
            └── ...
```

DAG 的价值：
- **完整血缘追踪**：知道任何因子的全部"祖先"，避免重复进化
- **产出率统计**：某个分支的后代质量高 → 优先从该分支继续进化
- **非冗余约束**：新因子不能与任何直系祖先相似

### 2. 贝叶斯种子检索
不是随机选择起点，而是用贝叶斯模型评估每个 DAG 节点的"后代质量期望"：
```
P(好后代 | 节点X) ∝ 历史后代中通过率 × 平均IC × 多样性
```
选择期望最高的 Top-K 节点作为种子，指数级提高探索效率。

### 3. 上下文感知优化
结合当前市场状态进行参数优化：
- 高波动期 → 偏好长期因子（减少噪声）
- 趋势市 → 偏好动量因子
- 震荡市 → 偏好反转因子

---

## 实验结果

- 预测精度：优于基于 GP 和基于 Transformer 的方法
- 训练效率：在相同计算预算下，找到更多有效因子
- 因子多样性：DAG 约束确保低冗余
- 开源实现：已发布 GitHub

---

## 对 AutoAlpha 的启发

### 1. 因子血缘记录（直接可做，低成本）
在 `factor_library` 表中增加字段：

```sql
ALTER TABLE factor_library ADD COLUMN parent_factor TEXT;
ALTER TABLE factor_library ADD COLUMN evolution_type TEXT;  -- mutation/crossover/extension
```

这样可以追踪每个因子是从哪个因子演化来的，防止近亲繁殖。

### 2. 简化版 DAG 检索
不需要完整 DAG，但可以在选择延伸起点时，优先选择**后代成功率高**的基础因子：

```python
def select_seed_for_extension(self) -> FactorSpec:
    """选择最有潜力作为进化起点的因子"""
    factors = self.db.get_passed_factors()
    # 计算每个因子的后代成功率
    for f in factors:
        descendants = self.db.get_descendants(f['factor_name'])
        f['descendant_pass_rate'] = (
            len([d for d in descendants if d['status'] == 'PASS'])
            / max(len(descendants), 1)
        )
    # 优先选择后代成功率高的因子
    return max(factors, key=lambda x: x['ic_mean'] * (1 + x['descendant_pass_rate']))
```

### 3. 祖先相似度约束
延伸生成的新因子，不只检查与整个因子库的相似度，还要特别检查与其直接"父代"和"祖父代"的相似度，相似度 > 0.8 则强制要求更大差异。
