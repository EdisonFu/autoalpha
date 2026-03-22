import json
import sys
import os
import subprocess
import tempfile
from typing import Optional, Tuple
from ..llm import LLMClient
from ..utils import strip_code_fence
from ..pipeline_state import FactorSpec

_CODEGEN_SYSTEM = """\
You are a quantitative developer. Generate Python code that computes a cross-sectional factor.

The function must:
1. Accept `data: dict[str, pd.DataFrame]` — each key is a stock code, value is a DataFrame with DatetimeIndex and columns: open, high, low, close, volume (and more if available)
2. Return a pd.Series with MultiIndex (date, stock_code) named "factor_value"
3. Use only pandas, numpy, scipy (no external data fetching)
4. Handle NaN gracefully (dropna before rank correlation)
5. Normalize factor cross-sectionally each date (rank / count → 0~1 range) to improve IC stability

Signature:
```python
import pandas as pd
import numpy as np

def compute_factor(data: dict) -> pd.Series:
    # ... your implementation ...
    return factor_series  # MultiIndex (date, stock_code), name="factor_value"
```

After the function, add a main block:
```python
import pickle, sys, json
if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f)
    result = compute_factor(data)
    result.to_csv(sys.argv[2])
    print("FACTOR_OK")
```

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
        """
        code = self._generate_code(spec, data, error_context)
        factor_name_safe = re.sub(r"[^\w]", "_", spec.factor_name)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_path = os.path.join(self.output_dir, f"{factor_name_safe}_{ts}.py")
        csv_path  = os.path.join(self.output_dir, f"{factor_name_safe}_{ts}.csv")

        with open(code_path, "w") as f:
            f.write(code)

        # Serialize data
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

    def _generate_code(self, spec: FactorSpec, data: dict, error_context: str = None) -> str:
        available_cols = set()
        for df in list(data.values())[:3]:
            available_cols.update(df.columns.tolist())

        user_parts = [
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
            user_parts.append(f"\nPrevious error to fix:\n{error_context[:1500]}")

        return strip_code_fence(self.llm.complete(system=_CODEGEN_SYSTEM, user="\n".join(user_parts)))
