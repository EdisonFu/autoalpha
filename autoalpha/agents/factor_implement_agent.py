import re
import json
import sys
import os
import subprocess
import tempfile
from typing import Optional, Tuple
from ..llm import LLMClient
from ..utils import strip_code_fence
from ..pipeline_state import FactorSpec


# ── 错误分类 ──────────────────────────────────────────────────────────────────

def classify_error(error: str) -> str:
    """将实现错误分类为可操作的类型，驱动定向修复。"""
    e = error.lower()
    if "timeout" in e:
        return "TIMEOUT"
    if ("numpy" in e and ("compiled" in e or "abi" in e)) or "numpy 1.x" in e or "numpy 2." in e:
        return "ENV"
    if "syntaxerror" in e or "indentationerror" in e or "invalid syntax" in e:
        return "SYNTAX"
    if "nameerror" in e or "attributeerror" in e or "importerror" in e or "modulenotfounderror" in e:
        return "RUNTIME"
    if "keyerror" in e or "valueerror" in e or "indexerror" in e or "typeerror" in e:
        return "DATA"
    return "LOGIC"


_REPAIR_HINTS = {
    "SYNTAX":  "Fix syntax errors: check indentation, parentheses, colons, and string quotes carefully.",
    "RUNTIME": "Fix name/import errors: ensure all variables are defined before use, all imports are present, "
               "and column names exactly match the available data columns listed above.",
    "DATA":    "Fix data errors: use .dropna() before operations, check column existence with "
               "'if col in df.columns', specify min_periods in rolling calls, handle empty groups.",
    "LOGIC":   "Read the full traceback carefully and fix the root cause of the error.",
}


# ── 代码生成 Prompt（Chain-of-Thought，来自 Chain-of-Alpha 论文）────────────────

_CODEGEN_SYSTEM = """\
You are a quantitative developer. Generate Python code for a cross-sectional factor.

Think step by step INSIDE the code as comments before writing the logic:
  # STEP 1 - SIGNAL: What phenomenon does this factor capture? Why does it predict returns?
  # STEP 2 - DATA: Which columns map to the formula? What lookback is appropriate?
  # STEP 3 - PITFALLS: What mistakes to avoid (lookahead bias, NaN traps, index alignment)?
  # STEP 4 - IMPLEMENTATION: Write the logic below.

Requirements:
1. Accept `data: dict[str, pd.DataFrame]` — key=stock code, value=DataFrame with DatetimeIndex
   and columns: open, high, low, close, volume (and possibly more)
2. Return pd.Series with MultiIndex (date, stock_code) named "factor_value"
3. Use only pandas, numpy, scipy — no external data fetching
4. NO LOOKAHEAD BIAS: only use data available up to each date; never use future prices in the factor
5. Handle NaN: specify min_periods, use .dropna() before correlations, fill with group median if needed
6. Normalize cross-sectionally each date: rank / count → [0, 1] range
7. If the hypothesis implies NEGATIVE relationship with returns, multiply by -1 before normalization

Function signature:
```python
import pandas as pd
import numpy as np

def compute_factor(data: dict) -> pd.Series:
    # STEP 1 - SIGNAL: ...
    # STEP 2 - DATA: ...
    # STEP 3 - PITFALLS: ...
    # STEP 4 - IMPLEMENTATION:
    ...
    return factor_series  # MultiIndex (date, stock_code), name="factor_value"
```

After the function, add exactly this main block:
```python
import pickle, sys
if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f)
    result = compute_factor(data)
    result.to_csv(sys.argv[2])
    print("FACTOR_OK")
```

Output ONLY valid Python code, no markdown fences, no explanation after the code.
"""

# 修复 Prompt（低温度，外科手术式）
_REPAIR_SYSTEM = """\
You are a quantitative developer fixing a broken factor implementation.

Make the MINIMUM changes needed to fix the specific problem described.
- Do NOT rewrite logic that is not broken
- Do NOT change the factor's economic meaning or formula structure
- Fix ONLY what the error message or evaluation feedback indicates

Output ONLY valid Python code, no markdown fences, no explanation.
"""


class FactorImplementAgent:
    """为 FactorSpec 生成因子计算代码，在沙箱中执行，输出因子值 CSV"""

    TIMEOUT = 120

    def __init__(self, llm: LLMClient, output_dir: str = "./factors"):
        self.llm = llm
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def implement(self, spec: FactorSpec, data: dict,
                  error_context: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (factor_csv_path, error_msg).
        factor_csv_path is None on failure.
        error_context: previous error or eval failure — triggers targeted repair.
        """
        code = self._generate_code(spec, data, error_context)

        # Lookahead bias warning (non-blocking, informational)
        warnings = self._check_lookahead_bias(code)
        if warnings:
            print(f"[FactorImpl] ⚠ 可能的前瞻偏差: {'; '.join(warnings)}")

        factor_name_safe = re.sub(r"[^\w]", "_", spec.factor_name)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_path = os.path.join(self.output_dir, f"{factor_name_safe}_{ts}.py")
        csv_path  = os.path.join(self.output_dir, f"{factor_name_safe}_{ts}.csv")

        with open(code_path, "w") as f:
            f.write(code)

        import pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(data, f)
            data_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, code_path, data_path, csv_path],
                capture_output=True, text=True, timeout=self.TIMEOUT
            )
        except subprocess.TimeoutExpired:
            return None, f"TIMEOUT: exceeded {self.TIMEOUT}s"
        finally:
            os.unlink(data_path)

        if proc.returncode != 0 or "FACTOR_OK" not in proc.stdout:
            err = (proc.stderr or proc.stdout)[-2000:]
            return None, f"ERROR: {err}"

        return csv_path, None

    # ── internals ──────────────────────────────────────────────────────────────

    def _generate_code(self, spec: FactorSpec, data: dict, error_context: str = None) -> str:
        available_cols = set()
        for df in list(data.values())[:3]:
            available_cols.update(df.columns.tolist())

        base_parts = [
            f"Factor name: {spec.factor_name}",
            f"Description: {spec.description}",
            f"Hypothesis: {spec.hypothesis}",
            f"Formula: {spec.formula}",
            f"Implementation notes: {spec.implementation_notes}",
            f"Parameters: {json.dumps(spec.parameters)}",
            f"Available data columns: {sorted(available_cols)}",
            f"Number of stocks in data: {len(data)}",
        ]

        if error_context:
            # Repair mode: targeted prompt + low temperature（来自 BrainAlpha T=0.3 设计）
            err_type = classify_error(error_context)
            hint = _REPAIR_HINTS.get(err_type, _REPAIR_HINTS["LOGIC"])
            repair_parts = base_parts + [
                f"\nPrevious attempt failed. Error type: [{err_type}]",
                f"Repair guidance: {hint}",
                f"\nFull error / feedback:\n{error_context[:2000]}",
            ]
            return strip_code_fence(
                self.llm.complete(system=_REPAIR_SYSTEM, user="\n".join(repair_parts),
                                  temperature=0.3)
            )
        else:
            # Generation mode: Chain-of-Thought prompt（来自 Chain-of-Alpha 论文）
            return strip_code_fence(
                self.llm.complete(system=_CODEGEN_SYSTEM, user="\n".join(base_parts),
                                  temperature=0.7)
            )

    @staticmethod
    def _check_lookahead_bias(code: str) -> list:
        """Detect common lookahead bias patterns. Returns list of warning strings."""
        warnings = []
        if re.search(r'shift\s*\(\s*-[1-9]', code):
            warnings.append("shift(-N) 可能引入前瞻偏差（除非用于计算前向收益）")
        if re.search(r'pct_change\s*\([^)]*\)\s*(?!.*shift)', code):
            # pct_change without subsequent shift — might be used directly as feature
            pass  # Too noisy, skip
        return warnings
