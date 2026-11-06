"""
研究循环的反思环节：把本轮结果变成下一轮该查什么。

**为什么需要**：原来的延伸探索是条死胡同 —— 只围绕已通过因子做参数/
公式/组合变体，走完就结束。那不是 auto research，是参数微调。

真正的 auto research 是 loop：本轮的结果本身就是下一轮该查什么的线索。
实测一轮 4 个因子全败，失败原因分别是「需要宏观数据（没有）」、
「多空 Sharpe 0.211 不足」、「单调性 0.50 不达标」、「代码结构误解」。
这四条失败原因指向的下一步完全不同，而旧流程对它们一视同仁地结束。

**必须守住的两条线**：
  1. 不能只是重跑。要把失败模式作为约束传下去，否则就是无意义循环。
  2. 轮次必须记账。每多跑一轮就是在同一批数据上多搜一次，
     选择性偏差累积，门控数字会越来越好看而越来越不可信。
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from ..json_repair import JSONRepairError, loads_lenient

_REFLECT_SYSTEM = """\
You are a quantitative research director reviewing one completed research round.

You are given the research intent, and every factor tried with its outcome and
the concrete reason it failed. Decide what the NEXT round should investigate.

Output a JSON object:
{
  "learned": "2-4 sentences: what this round actually established, in Chinese",
  "next_direction_zh": "one concrete research direction for the next round, in Chinese",
  "canonical_en": "a precise English academic search query for that direction",
  "canonical_zh": "对应的中文检索式，空格分隔的中文关键词，如 \"低波动 动量 择时\"",
  "avoid": ["specific thing to avoid next round", "..."],
  "rationale": "why this direction follows from the failures, 1-2 sentences"
}

Rules:
- Derive the next direction FROM THE FAILURE REASONS, not from generic quant knowledge.
- If factors failed on data availability, the next direction must use ONLY
  price/volume fields — do not propose fundamentals, macro series, news, or VIX.
- If factors failed on turnover, propose lower-frequency or smoother signals.
- If factors failed on monotonicity or weak IC, change the economic mechanism,
  do not just retune the window.
- If in-sample passed but out-of-sample decayed, prioritise robustness and
  parameter-insensitivity over a stronger in-sample fit.
- If some factors PASSED, the next direction should deepen their economic logic
  or find a low-correlation complement, not repeat them.
- Never propose the identical query that was just used.
- canonical_en must be a real bibliographic query: 3-8 words, standard academic
  terminology, no Chinese.
- canonical_zh uses the terminology Chinese analysts actually use, so the next
  round also searches broker research reports and quant community posts.
  It must follow the same direction as canonical_en and respect the same
  avoid-list (e.g. if only price/volume is available, do not write 宏观 or 基本面).
"""


class ResearchLoopAgent:
    """读本轮结果，产出下一轮研究方向。"""

    def __init__(self, llm):
        self.llm = llm
        self.last_error = ""

    def reflect(self, intent, evaluations: List, round_no: int,
                max_rounds: int) -> Optional[Dict]:
        """
        返回 {"learned", "next_direction_zh", "canonical_en", "avoid", "rationale"}，
        失败返回 None（调用方必须如实呈现，不能静默退化成「再跑一遍」）。
        """
        if not evaluations:
            self.last_error = "本轮没有任何评估结果，无法反思"
            return None

        summary = self._summarize(evaluations)
        user = self._build_user(intent, summary, round_no, max_rounds)
        try:
            raw = self.llm.complete(system=_REFLECT_SYSTEM, user=user,
                                    purpose="reflect")
        except Exception as e:
            self.last_error = f"反思调用失败: {e}"
            return None

        try:
            obj = loads_lenient(raw, expect=dict)
        except JSONRepairError as e:
            self.last_error = f"反思结果解析失败: {str(e)[:200]}"
            return None

        if not isinstance(obj, dict) or not obj.get("canonical_en"):
            self.last_error = "反思未给出下一轮检索式"
            return None

        obj["avoid"] = [str(x) for x in (obj.get("avoid") or [])][:6]
        self.last_error = ""
        return obj

    @staticmethod
    def _summarize(evaluations: List) -> Dict:
        """把评估结果压成反思需要的结构，并统计失败模式。"""
        rows, passes, fails = [], [], []
        patterns: Dict[str, int] = {}
        for e in evaluations:
            status = getattr(e, "status", "") or ""
            reason = (getattr(e, "fail_reason", "") or "")[:300]
            name = getattr(e, "factor_name", "?")
            rows.append({"factor": name, "status": status, "reason": reason})
            if status == "PASS":
                passes.append(name)
            else:
                fails.append(name)
                patterns[_pattern_of(status, reason)] = \
                    patterns.get(_pattern_of(status, reason), 0) + 1
        return {"rows": rows, "passed": passes, "failed": fails,
                "patterns": patterns}

    @staticmethod
    def _build_user(intent, summary: Dict, round_no: int,
                    max_rounds: int) -> str:
        lines = [
            f"Round {round_no} of {max_rounds}.",
            f"Original research idea: {getattr(intent, 'raw_input', '')}",
            f"Query used this round: {getattr(intent, 'canonical_en', '')}",
            f"Market: {getattr(intent, 'market', 'unknown')}",
            f"Available data: daily OHLCV for Chinese A-share ETFs ONLY. "
            f"No fundamentals, no macro, no news, no VIX.",
            "",
            f"PASSED ({len(summary['passed'])}): "
            + (", ".join(summary["passed"]) or "none"),
            f"FAILED ({len(summary['failed'])}): "
            + (", ".join(summary["failed"]) or "none"),
            "",
            "Failure pattern counts: "
            + json.dumps(summary["patterns"], ensure_ascii=False),
            "",
            "Per-factor outcomes:",
        ]
        for r in summary["rows"][:12]:
            lines.append(f"- {r['factor']}: {r['status']} — {r['reason']}")
        return "\n".join(lines)


def _pattern_of(status: str, reason: str) -> str:
    """把失败原因归成可操作的类别，供反思时识别主导失败模式。"""
    r = (reason or "").lower()
    if status == "SKIP_NO_DATA" or "没有的数据" in reason or "未采集" in reason:
        return "data_unavailable"
    if "换手" in reason or "turnover" in r:
        return "turnover"
    if "单调" in reason or "monoton" in r:
        return "monotonicity"
    if "sharpe" in r:
        return "weak_sharpe"
    if "衰减" in reason or "decay" in r:
        return "oos_decay"
    if "实现失败" in reason or "traceback" in r or "error" in r:
        return "implementation"
    if status.startswith("FAIL_OOS"):
        return "oos_gate"
    return "other"
