# Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment

**ArXiv**: https://arxiv.org/abs/2308.00016
**作者**: Saizhuo Wang, Hang Yuan, Leon Zhou, Lionel M. Ni, Heung-Yeung Shum, Jian Guo
**年份**: 2023（该领域开山之作）

---

## 核心贡献

Alpha-GPT 是**该领域最早将 LLM 用于 Alpha 挖掘的系统性框架**，提出了"人机交互式 Alpha 挖掘"范式：人类提供投资直觉，LLM 负责规范化、代码化和优化，两者形成对话式迭代。

---

## 系统架构

### AlphaBot 四模块
```
用户输入（交易想法 / 自然语言）
    ↓
┌─────────────────────────────────┐
│  Knowledge Compiler（知识编译器） │
│  将用户想法解析为结构化规范         │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  LLM APIs（语言模型接口）          │
│  生成候选因子表达式                │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  Thought Decompiler（思维解析器） │
│  将 LLM 输出解析为可执行代码        │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│  Knowledge Library（知识库）      │
│  存储历史 Alpha + 表现摘要          │
└──────────────┬──────────────────┘
               ↓
     回测评估 → 结果摘要给用户
               ↓
         用户修改 / 接受
```

### 交互式迭代
```
用户："帮我做一个基于动量的因子"
AlphaBot：生成因子A，回测 IC=0.02
用户："IC 太低了，试试加上成交量过滤"
AlphaBot：修改为因子B，回测 IC=0.035
用户："好多了，再试试行业中性化版本"
...
```

---

## 核心创新

### 1. 自然语言→量化规范的编译
将投资者的直觉描述（"强势股继续强"）翻译为精确的量化规范（20日收益率排名 top30%，持有5日）。

### 2. Alpha 摘要（Alpha Summary）
系统不只执行回测，还用 LLM 将复杂的回测结果转化为人类可读的自然语言摘要：
```
"该因子在牛市环境下表现优异（月均IC=0.045），但在2022年熊市中几乎失效（IC=-0.003）。
建议：在高市场波动时降低权重。"
```

### 3. 知识库积累
每个成功/失败的 Alpha 都写入知识库，供未来迭代参考。

---

## 历史意义

Alpha-GPT 证明了以下可能性：
1. LLM 具备将模糊投资想法转化为精确代码的能力
2. 人机对话可以有效迭代改善因子质量
3. LLM 可以生成比人工更易理解的回测报告

其局限性推动了后续全自动系统（AlphaAgent、QuantaAlpha 等）的发展。

---

## 对 AutoAlpha 的启发

### 1. 自然语言摘要（直接可做）
我们的 `report_agent` 生成 HTML 报告，可以增加 LLM 生成的**自然语言因子摘要**：

```python
def generate_factor_narrative(self, result: FactorResult, spec: FactorSpec) -> str:
    """生成因子的自然语言摘要"""
    return self.llm.complete(
        system="你是量化研究员，用简洁的语言总结因子表现",
        user=f"""
        因子：{spec.factor_name}
        假设：{spec.hypothesis}
        IC均值：{result.ic_mean:.4f}，ICIR：{result.icir:.4f}
        月胜率：{result.monthly_win_rate:.1%}
        状态：{result.status}

        请用2-3句话总结：该因子捕捉了什么市场现象，表现如何，适用什么市场环境
        """
    )
```

### 2. 交互式修改支持
在我们的 CLI 中，当因子评估失败时，可以允许用户输入修改建议：
```
因子 momentum_5d 评估失败（IC=0.018 < 0.03）
> 是否尝试修改？输入建议或回车跳过：_
```

### 3. 知识库的 Alpha 摘要字段
在 `factor_library` 中增加 `narrative` 字段，存储 LLM 生成的自然语言描述，让非技术用户也能理解因子库内容。
