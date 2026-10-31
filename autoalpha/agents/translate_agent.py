"""
中文概述生成。

成本设计：**批量单次调用**，不是每篇/每个因子调一次。
10 篇论文 1 次调用，而非 10 次 —— 这是 token 开销上的关键差别
（见 docs/TOKEN_COST.md）。失败时优雅降级：返回空概述，不阻断流水线。
"""

import json
from typing import Dict, List

from ..llm import LLMClient
from ..json_repair import loads_lenient, JSONRepairError
from ..utils import strip_code_fence

_PAPER_SYSTEM = """\
你是量化金融研究助理。给定若干英文论文的标题与摘要，为每篇生成中文概述。

每篇输出：
- summary: 2-3 句话，说明这篇论文做了什么、核心方法是什么
- relevance: 1 句话，说明它对「量化因子挖掘」可能有什么用（没有明显用处就直说）

要求：
- 用中文，术语保留英文原词并加中文解释，如「动量(momentum)」
- 不要空话套话，不要复述标题
- 若摘要信息不足，summary 写「摘要信息不足，需查看全文」

严格输出 JSON 数组，顺序与输入一致，每元素 {"summary": "...", "relevance": "..."}。

格式要求（违反会导致解析失败）：
- 只输出 JSON，前后不要加任何解释文字
- 字段值内部**不要使用英文双引号**，需要引用时用「」或圆括号
- 字段值内部不要换行，写成一段连续文字
"""

_FACTOR_SYSTEM = """\
你是量化研究员。给定若干因子假设（英文），为每个生成中文说明。

每个输出：
- what: 1-2 句，这个因子衡量什么
- why: 1-2 句，为什么它可能预测收益（经济/行为逻辑）
- how: 1 句，计算要点（窗口、归一化方式等）

要求：
- 用中文，专业术语保留英文原词，如「截面排序(cross-sectional rank)」
- 讲清经济直觉，不要只翻译公式
- how 必须忠于给定的「实现要点」字段，不要自行增删处理步骤；
  若实现要点提到去极值/标准化等操作，必须体现，不得写成「无需处理」

严格输出 JSON 数组，顺序与输入一致，每元素 {"what":"...","why":"...","how":"..."}。

格式要求（违反会导致解析失败）：
- 只输出 JSON，前后不要加任何解释文字
- 字段值内部**不要使用英文双引号**，需要引用时用「」或圆括号
- 字段值内部不要换行，写成一段连续文字
"""

MAX_ABSTRACT = 600          # 摘要截断，控制输入 token
BATCH_SIZE = 10             # 单次调用处理的条数上限


class TranslateAgent:
    """为论文和因子生成中文概述。批量调用以控制成本。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ── 论文 ───────────────────────────────────────────────────────────────

    def describe_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        返回与 papers 等长的 [{"summary","relevance"}]。
        失败时返回空字典列表 —— 概述是增强信息，不应阻断流水线。
        """
        if not papers:
            return []
        out: List[Dict] = []
        for i in range(0, len(papers), BATCH_SIZE):
            batch = papers[i:i + BATCH_SIZE]
            payload = "\n\n".join(
                f"[{j}] 标题: {p.get('title','')}\n摘要: "
                f"{(p.get('abstract') or '')[:MAX_ABSTRACT]}"
                for j, p in enumerate(batch))
            out.extend(self._call(_PAPER_SYSTEM, payload, len(batch),
                                  ("summary", "relevance"), "translate_paper"))
        return out

    # ── 因子 ───────────────────────────────────────────────────────────────

    def describe_factors(self, specs: List) -> List[Dict]:
        """返回与 specs 等长的 [{"what","why","how"}]。"""
        if not specs:
            return []
        out: List[Dict] = []
        for i in range(0, len(specs), BATCH_SIZE):
            batch = specs[i:i + BATCH_SIZE]
            payload = "\n\n".join(
                f"[{j}] 名称: {s.factor_name}\n描述: {s.description}\n"
                f"假设: {s.hypothesis}\n公式: {s.formula}\n"
                f"实现要点: {s.implementation_notes}"
                for j, s in enumerate(batch))
            out.extend(self._call(_FACTOR_SYSTEM, payload, len(batch),
                                  ("what", "why", "how"), "translate_factor"))
        return out

    # ── 内部 ───────────────────────────────────────────────────────────────

    FAIL_NOTE = "（中文说明生成失败，请查看运行日志）"

    def _call(self, system: str, payload: str, n: int,
              keys: tuple, purpose: str) -> List[Dict]:
        # 失败时填提示文字而非空串：空串在界面上表现为「中文没了」，
        # 用户无法区分「没生成」和「生成失败」（实测这正是困惑来源）。
        def _failed(reason: str) -> List[Dict]:
            print(f"[Translate] {purpose} 失败: {reason}")
            first = keys[0]
            return [{k: (self.FAIL_NOTE if k == first else "") for k in keys}
                    for _ in range(n)]

        try:
            raw = self.llm.complete(system=system, user=payload,
                                    temperature=0.3, purpose=purpose)
        except Exception as e:
            return _failed(f"LLM 调用异常: {e}")

        try:
            items = loads_lenient(raw, expect=list)
        except JSONRepairError as e:
            return _failed(f"JSON 解析失败: {str(e)[:200]}")

        # 对齐长度：LLM 可能少给或多给
        result = []
        for j in range(n):
            it = items[j] if j < len(items) and isinstance(items[j], dict) else {}
            result.append({k: str(it.get(k, "") or "") for k in keys})
        return result
