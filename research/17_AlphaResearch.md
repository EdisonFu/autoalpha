# AlphaResearch: Accelerating New Algorithm Discovery with Language Models

**ArXiv**: https://arxiv.org/abs/2511.08522
**作者**: Zhaojian Yu, Kaiyue Feng, Yilun Zhao, Shilin He, Xiao-Ping Zhang, Arman Cohan
**年份**: 2025

---

## 核心贡献

AlphaResearch 提出了**双研究环境架构**：可执行验证环境（判断"代码是否正确运行"）+ 模拟同行评审环境（判断"方法是否有创新价值"）。两个环境协同过滤，确保生成的算法既可以执行，又有实际价值。

---

## 系统架构

### 双环境验证
```
算法生成智能体
    ↓
┌─────────────────────────────────────────┐
│  环境一：可执行验证（Executable Verify）    │
│  - 代码能否运行？                          │
│  - 结果是否满足形式约束？                   │
│  - 计算复杂度是否可接受？                   │
└────────────────┬────────────────────────┘
                 ↓ 通过
┌─────────────────────────────────────────┐
│  环境二：模拟同行评审（Peer Review Sim）    │
│  - 与现有方法相比是否有改进？               │
│  - 改进是否有理论支撑？                    │
│  - 是否可能在更大规模上泛化？              │
└────────────────┬────────────────────────┘
                 ↓ 通过
             算法接受
```

### 迭代流程
```
提出算法 → 双环境验证
    ↓         ↓
  通过      失败
    ↓         ↓
  入库   生成改进建议
           ↓
       修订算法（最多N次）
           ↓
      再次双环境验证
```

---

## 核心创新

### 1. 可执行性与创新性的解耦验证
以往系统只关注"能否运行"。AlphaResearch 在此之上增加"是否有价值"的评估维度。在量化因子场景下：
- 可执行性：代码能跑，输出合法的 MultiIndex Series
- 创新性：与已有因子相比，是否捕捉了不同的市场信息？

### 2. 模拟同行评审（LLM as Reviewer）
用 LLM 扮演同行评审员，从专业角度评价：
```
System: 你是顶级量化研究期刊的匿名审稿人
User: 请评审以下因子的方法论：
  [因子描述 + 公式 + 回测结果]
  与以下现有因子对比：[已有因子列表]

评审角度：
1. 经济含义是否清晰？
2. 与现有方法相比是否有实质改进？
3. 回测是否可能存在过拟合？
4. 综合评分（1-10）
```

### 3. AlphaResearchComp 基准
构建了包含 8 个开放式算法挑战的基准，评估 AI 系统的算法发现能力。

---

## 实验结果

- 2/8 挑战题超越人类研究员（特定问题）
- 在圆形打包问题（circle-packing）上达到人类已知最优
- 双环境验证显著提高了最终算法质量

---

## 对 AutoAlpha 的启发

### 1. LLM 同行评审（直接可做）
在因子入库前，增加一个 LLM 评审步骤：

```python
def peer_review_factor(self, spec: FactorSpec, result: FactorResult,
                       existing_factors: List[dict]) -> Tuple[float, str]:
    """LLM 扮演同行评审员评估因子"""
    existing_summary = "\n".join(
        f"- {f['factor_name']}: {f['description'][:100]}"
        for f in existing_factors[:10]
    )
    review = self.llm.complete(
        system="""你是顶级量化研究期刊审稿人。从三个角度评审因子：
        1. 经济含义（0-10）：假设是否有合理的经济基础？
        2. 差异化（0-10）：与现有因子相比是否有实质不同？
        3. 稳健性（0-10）：IC稳定性和衰减特征是否令人满意？
        输出 JSON：{"economic": N, "novelty": N, "robustness": N, "comment": "..."}""",
        user=f"""新因子：
        名称：{spec.factor_name}
        假设：{spec.hypothesis}
        IC均值：{result.ic_mean:.4f}，ICIR：{result.icir:.4f}

        现有因子库：
        {existing_summary}"""
    )
    scores = json.loads(strip_code_fence(review))
    overall = (scores['economic'] + scores['novelty'] + scores['robustness']) / 3
    return overall, scores.get('comment', '')
```

### 2. 双重验证门槛
将"通过评估"从只看 IC/ICIR 数值，扩展到需要同时通过：
- 定量门槛：IC > 0.03 等
- 定性评审：LLM 评审综合得分 > 6

### 3. 反馈改进建议
评审不通过时，LLM 审稿人给出改进建议，而不只是拒绝：
```
"该因子经济含义良好（8分），但与 momentum_20d 高度相似（差异化3分）。
建议：尝试加入行业相对动量或与波动率条件化组合，以提高差异化。"
```
