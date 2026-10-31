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
    if "缺少" in error and "__main__" in error:
        return "NO_MAIN"
    if "未产生输出" in error:
        return "NO_OUTPUT"
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
    "NO_MAIN": "The file is MISSING the required main block. Append EXACTLY this at the end:\n"
               "```python\nimport pickle, sys\n"
               "if __name__ == \"__main__\":\n"
               "    with open(sys.argv[1], \"rb\") as f:\n"
               "        data = pickle.load(f)\n"
               "    result = compute_factor(data)\n"
               "    result.to_csv(sys.argv[2])\n"
               "    print(\"FACTOR_OK\")\n```\n"
               "Keep compute_factor() unchanged.",
    "NO_OUTPUT": "The script ran but produced no output. Verify the main block runs to completion, "
                 "that compute_factor() returns a non-empty Series, and that nothing exits early.",
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
8. OUTPUT A VALUE FOR EVERY TRADING DAY — never subsample to month-end/week-end.
   Rebalancing frequency is decided by the BACKTEST framework, not by this factor.
   If the hypothesis is monthly, express that through a LONGER LOOKBACK WINDOW
   (e.g. 252-day momentum), still emitting one value per stock per trading day.
   Subsampling to month-ends collapses coverage and breaks the backtest alignment.
9. The factor must VARY ACROSS STOCKS on each date. A value broadcast identically
   to all stocks (e.g. np.tile of a market-level scalar) has zero cross-sectional
   information and will be rejected.

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

ENVIRONMENT (%(PD_VER)s) — these APIs were REMOVED or CHANGED:
  - `df.stack(dropna=...)`     → REMOVED. Use `df.stack()` only.
  - `df.pct_change()`          → must pass `fill_method=None` explicitly
  - `df.fillna(method=...)`    → REMOVED. Use `df.ffill()` / `df.bfill()`
  - `df.append(...)`           → REMOVED. Use `pd.concat([...])`
  - `df.iteritems()`           → REMOVED. Use `df.items()`
  - `pd.Series.swapaxes`, `.mad()`, `inplace=` on most methods → avoid entirely
Writing any removed API guarantees a runtime failure on the first attempt.

Output ONLY valid Python code, no markdown fences, no explanation after the code.
"""

# 修复 Prompt（低温度，外科手术式）
_REPAIR_SYSTEM = """\
You are a quantitative developer fixing a broken factor implementation.

Make the MINIMUM changes needed to fix the specific problem described.
- Do NOT rewrite logic that is not broken
- Do NOT change the factor's economic meaning or formula structure
- Fix ONLY what the error message or evaluation feedback indicates
- PRESERVE the `if __name__ == "__main__":` block verbatim if present;
  if it is absent, ADD it back (the script does nothing without it)

ENVIRONMENT (%(PD_VER)s) — removed APIs:
  `stack(dropna=)` gone; `pct_change()` needs `fill_method=None`;
  `fillna(method=)` → `ffill()/bfill()`; `append()` → `pd.concat()`.

Output ONLY valid Python code, no markdown fences, no explanation.
"""


def _env_line() -> str:
    """实际运行环境，注入 prompt。硬编码版本号会在环境变化时静默失效。"""
    try:
        import pandas as _pd, numpy as _np
        return f"pandas {_pd.__version__} / numpy {_np.__version__}"
    except Exception:
        return "pandas 3.x / numpy 2.x"


class FactorImplementAgent:
    """为 FactorSpec 生成因子计算代码，在沙箱中执行，输出因子值 CSV"""

    TIMEOUT = 120

    def __init__(self, llm: LLMClient, output_dir: str = "./factors"):
        self.llm = llm
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def implement(self, spec: FactorSpec, data: dict,
                  error_context: str = None,
                  experience_context: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (factor_csv_path, error_msg).
        factor_csv_path is None on failure.
        error_context: previous error or eval failure — triggers targeted repair.
        """
        code = self._generate_code(spec, data, error_context, experience_context)

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
            err = (proc.stderr or proc.stdout or "").strip()[-2000:]
            if not err:
                # rc=0 但无输出 = 脚本只定义了函数就退出。
                # 实测最常见成因：LLM 漏写 `if __name__ == "__main__"` 主块。
                # 旧实现这里会返回空的 "ERROR: "，LLM 在修复轮次里看不到任何
                # 线索，导致连续多轮无效重试（实测 3 次全废）。
                has_main = "__main__" in code
                err = (
                    "脚本执行完毕但未产生输出（未打印 FACTOR_OK，也未写出 CSV）。"
                    + ("主块存在但未执行到结尾，检查是否有提前 return/exit，"
                       "或 compute_factor 内部静默失败。"
                       if has_main else
                       "文件中缺少 `if __name__ == \"__main__\":` 主块 —— "
                       "必须原样附加 prompt 中要求的 main 块，否则脚本只定义函数不执行。")
                )
            return None, f"ERROR: {err}"

        if not os.path.exists(csv_path):
            return None, ("ERROR: 打印了 FACTOR_OK 但 CSV 文件不存在，"
                          f"检查是否写到了 sys.argv[2] 指定的路径: {csv_path}")

        return csv_path, None

    # ── internals ──────────────────────────────────────────────────────────────

    def _generate_code(self, spec: FactorSpec, data: dict,
                       error_context: str = None,
                       experience_context: str = None) -> str:
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
                self.llm.complete(
                    system=_REPAIR_SYSTEM % {"PD_VER": _env_line()},
                    user="\n".join(repair_parts), temperature=0.3,
                    purpose="repair")
            )
        else:
            # Generation mode: Chain-of-Thought prompt（来自 Chain-of-Alpha 论文）
            if experience_context:
                base_parts = base_parts + [
                    "\n--- Historical mining experience (RALPH memory) ---",
                    experience_context,
                    "--- End of experience ---",
                ]
            return strip_code_fence(
                self.llm.complete(
                    system=_CODEGEN_SYSTEM % {"PD_VER": _env_line()},
                    user="\n".join(base_parts), temperature=0.7,
                    purpose="codegen")
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
