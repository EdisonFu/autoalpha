import json
from ..llm import LLMClient
from ..pipeline_state import FactorSpec, FactorResult
from ..utils import strip_code_fence

_JUDGE_SYSTEM = """\
You are a senior quantitative researcher conducting peer review of an alpha factor.

Evaluate the factor on four dimensions (score each 0-10):
1. ECONOMIC_SOUNDNESS: Is the hypothesis clear, logical, and grounded in finance theory?
   - Does the signal capture a real market inefficiency or risk premium?
   - Is the causal mechanism plausible and non-circular?
2. IMPLEMENTATION_FIDELITY: Does the code faithfully implement the stated formula?
   - Are there signs of lookahead bias (using future data)?
   - Does normalization match the hypothesis direction?
3. NOVELTY: Is this genuinely different from well-known commodity factors?
   - Common factors to check against: price momentum, mean-reversion, P/B value, size (market cap),
     profitability (ROE/ROA), low volatility, short-term reversal, volume/liquidity, market beta.
   - Deduct for trivial parameter variations of the above without new economic rationale.
4. OVERFITTING_RISK (higher score = lower risk):
   - Is the lookback reasonable relative to signal frequency?
   - Are there too many free parameters relative to the degrees of freedom?

Output ONLY valid JSON with this exact structure:
{
  "economic_soundness": <0-10>,
  "implementation_fidelity": <0-10>,
  "novelty": <0-10>,
  "overfitting_risk": <0-10>,
  "overall_score": <weighted average: 0.35*econ + 0.30*fidelity + 0.20*novelty + 0.15*overfit>,
  "verdict": "APPROVE" or "REJECT",
  "rejection_reason": "<empty if APPROVE, brief reason if REJECT>",
  "improvement_suggestions": "<1-2 concrete actionable suggestions>"
}

Rejection rules (apply ALL):
- REJECT if economic_soundness < 4
- REJECT if implementation_fidelity < 5
- REJECT if overall_score < 5.0
"""


class FactorJudgeAgent:
    """LLM peer-review layer: semantic quality gate after quantitative evaluation.

    Based on Multi-Agent Evolve (Proposer-Solver-Judge) design. Catches factors that
    pass quantitative IC gates by luck but lack economic or implementation soundness.
    Runs at low temperature (0.2) for consistent, reproducible verdicts.
    """

    APPROVE_THRESHOLD = 5.0

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def judge(self, spec: FactorSpec, result: FactorResult,
              code_path: str = "") -> dict:
        """
        Returns dict with verdict ("APPROVE"|"REJECT"), overall_score, and dimension scores.
        On LLM failure, defaults to APPROVE to avoid false rejections.
        """
        user_msg = self._build_prompt(spec, result, code_path)

        try:
            raw = strip_code_fence(self.llm.complete(
                system=_JUDGE_SYSTEM, user=user_msg, temperature=0.2,
                purpose="judge"
            ))
            verdict = json.loads(raw)
        except Exception as e:
            print(f"[Judge] 评审解析失败: {e}，默认 APPROVE")
            return {
                "verdict": "APPROVE", "overall_score": 7.0,
                "rejection_reason": "", "improvement_suggestions": "",
                "economic_soundness": 7.0, "implementation_fidelity": 7.0,
                "novelty": 7.0, "overfitting_risk": 7.0,
            }

        # Enforce verdict rules (LLM may not always follow them exactly)
        score = float(verdict.get("overall_score", 5.0))
        econ = float(verdict.get("economic_soundness", 5.0))
        fidelity = float(verdict.get("implementation_fidelity", 5.0))

        if econ < 4 or fidelity < 5 or score < self.APPROVE_THRESHOLD:
            verdict["verdict"] = "REJECT"
            if not verdict.get("rejection_reason"):
                parts = []
                if econ < 4:
                    parts.append(f"weak economic reasoning ({econ:.1f}/10)")
                if fidelity < 5:
                    parts.append(f"implementation mismatch ({fidelity:.1f}/10)")
                if score < self.APPROVE_THRESHOLD:
                    parts.append(f"low overall score ({score:.1f}/10)")
                verdict["rejection_reason"] = "; ".join(parts)
        else:
            verdict["verdict"] = "APPROVE"

        print(
            f"[Judge] {spec.factor_name}: "
            f"econ={econ:.1f} fidelity={fidelity:.1f} "
            f"novelty={verdict.get('novelty', 0):.1f} "
            f"score={score:.1f} → {verdict['verdict']}"
        )
        return verdict

    def _build_prompt(self, spec: FactorSpec, result: FactorResult,
                      code_path: str) -> str:
        parts = [
            f"Factor name: {spec.factor_name}",
            f"Hypothesis: {spec.hypothesis}",
            f"Formula: {spec.formula}",
            f"Description: {spec.description}",
            f"Parameters: {json.dumps(spec.parameters)}",
            "",
            "Quantitative evaluation results:",
            f"  IC mean: {result.ic_mean:.4f}",
            f"  ICIR: {result.icir:.4f}",
            f"  IC decay ratio (5d/1d): {result.ic_decay_ratio:.4f}",
            f"  Coverage: {result.coverage:.3f}",
            f"  Monthly win rate: {result.monthly_win_rate:.3f}",
        ]

        if code_path:
            try:
                with open(code_path) as f:
                    lines = f.readlines()[:60]
                code_preview = "".join(lines)
                parts += ["", f"Code implementation (first {len(lines)} lines):",
                          "```python", code_preview, "```"]
            except Exception:
                pass

        return "\n".join(parts)
