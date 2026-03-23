import numpy as np
import pandas as pd
from typing import Optional
from ..pipeline_state import FactorSpec, FactorResult

# Quality gates (from CLAUDE.md)
GATES = {
    "ic_mean":          0.03,
    "icir":             0.5,
    "ic_decay_ratio":   0.5,
    "coverage":         0.8,
    "monthly_win_rate": 0.55,
    "max_correlation":  0.7,   # must be BELOW this
}


class FactorEvalAgent:
    """
    读取因子值 CSV，结合价格数据计算 IC/ICIR/decay/coverage/月胜率，
    对比已有因子库的相关性，输出 FactorResult。
    """

    def __init__(self, existing_ic_series: list = None):
        """existing_ic_series: list of IC value lists from already-passing factors"""
        self.existing_ic_series = existing_ic_series or []

    def evaluate(self, spec: FactorSpec, factor_csv: str, price_data: dict) -> FactorResult:
        print(f"[FactorEval] 评估因子: {spec.factor_name}")
        try:
            factor_series = self._load_factor(factor_csv)
        except Exception as e:
            return self._fail(spec.factor_name, factor_csv, f"加载因子文件失败: {e}")

        try:
            price_matrix = self._build_price_matrix(price_data)
        except Exception as e:
            return self._fail(spec.factor_name, factor_csv, f"构建价格矩阵失败: {e}")

        # Coverage
        coverage = self._compute_coverage(factor_series, price_matrix)

        # IC series (1-day forward return)
        ic_series_1d = self._compute_ic_series(factor_series, price_matrix, fwd=1)
        ic_series_5d = self._compute_ic_series(factor_series, price_matrix, fwd=5)

        if len(ic_series_1d) < 10:
            return self._fail(spec.factor_name, factor_csv, f"有效 IC 观测太少: {len(ic_series_1d)}")

        ic_arr = np.array(ic_series_1d)
        ic_mean  = float(np.nanmean(ic_arr))
        ic_std   = float(np.nanstd(ic_arr))
        icir     = ic_mean / ic_std if ic_std > 1e-9 else 0.0

        ic_arr_5d = np.array(ic_series_5d)
        ic_mean_5d = float(np.nanmean(ic_arr_5d)) if len(ic_series_5d) >= 5 else 0.0
        ic_decay_ratio = (ic_mean_5d / ic_mean) if abs(ic_mean) > 1e-9 else 0.0

        monthly_win_rate = float(np.mean(ic_arr > 0))

        # Max correlation with existing factors
        max_corr = self._max_correlation_with_existing(ic_series_1d)

        result = FactorResult(
            factor_name=spec.factor_name,
            ic_mean=round(ic_mean, 5),
            icir=round(icir, 4),
            ic_decay_ratio=round(ic_decay_ratio, 4),
            coverage=round(coverage, 4),
            monthly_win_rate=round(monthly_win_rate, 4),
            max_correlation=round(max_corr, 4),
            status="",
            code_path=factor_csv,
            ic_series=ic_series_1d,
        )

        result.status, result.fail_reason = self._check_gates(result)
        return result

    # ── internals ─────────────────────────────────────────────────────────────

    def _load_factor(self, csv_path: str) -> pd.Series:
        df = pd.read_csv(csv_path, index_col=[0, 1], parse_dates=[0])
        df.index.names = ["date", "stock"]
        return df.iloc[:, 0].rename("factor_value")

    def _build_price_matrix(self, price_data: dict) -> pd.DataFrame:
        close_dict = {}
        for sym, df in price_data.items():
            if "close" in df.columns:
                close_dict[sym] = df["close"]
        if not close_dict:
            raise ValueError("price_data 中无 close 列")
        return pd.DataFrame(close_dict).sort_index()

    def _compute_coverage(self, factor_series: pd.Series, price_matrix: pd.DataFrame) -> float:
        total_cells = len(price_matrix) * len(price_matrix.columns)
        if total_cells == 0:
            return 0.0
        return float(factor_series.notna().sum() / total_cells)

    def _compute_ic_series(self, factor_series: pd.Series, price_matrix: pd.DataFrame, fwd: int = 1) -> list:
        from scipy.stats import spearmanr
        fwd_ret = price_matrix.pct_change(fwd).shift(-fwd)
        fwd_ret_long = fwd_ret.stack().rename("fwd_ret")
        fwd_ret_long.index.names = ["date", "stock"]

        combined = pd.concat([factor_series.rename("factor"), fwd_ret_long], axis=1).dropna()
        ic_list = []
        for date, grp in combined.groupby(level="date"):
            if len(grp) < 10:
                continue
            ic, _ = spearmanr(grp["factor"].values, grp["fwd_ret"].values)
            if not np.isnan(ic):
                ic_list.append(float(ic))
        return ic_list

    def _max_correlation_with_existing(self, ic_series: list) -> float:
        if not self.existing_ic_series or not ic_series:
            return 0.0
        from scipy.stats import pearsonr
        max_corr = 0.0
        for existing in self.existing_ic_series:
            min_len = min(len(ic_series), len(existing))
            if min_len < 10:
                continue
            try:
                corr, _ = pearsonr(ic_series[:min_len], existing[:min_len])
                max_corr = max(max_corr, abs(corr))
            except Exception:
                pass
        return max_corr

    def _check_gates(self, r: FactorResult):
        fails = []
        if r.ic_mean < GATES["ic_mean"]:
            fails.append(f"IC均值={r.ic_mean:.4f} < {GATES['ic_mean']}")
        if r.icir < GATES["icir"]:
            fails.append(f"ICIR={r.icir:.4f} < {GATES['icir']}")
        if r.ic_decay_ratio < GATES["ic_decay_ratio"]:
            fails.append(f"IC衰减比={r.ic_decay_ratio:.4f} < {GATES['ic_decay_ratio']}")
        if r.coverage < GATES["coverage"]:
            fails.append(f"覆盖率={r.coverage:.4f} < {GATES['coverage']}")
        if r.monthly_win_rate < GATES["monthly_win_rate"]:
            fails.append(f"月胜率={r.monthly_win_rate:.4f} < {GATES['monthly_win_rate']}")
        if r.max_correlation >= GATES["max_correlation"]:
            fails.append(f"与已有因子相关={r.max_correlation:.4f} >= {GATES['max_correlation']}")
        if fails:
            return "FAIL", "; ".join(fails)
        return "PASS", ""

    def get_repair_hints(self, result: "FactorResult") -> str:
        """
        基于评估失败的具体指标，返回定向修复建议。
        供 pipeline 在回传给 LLM 修复时使用（来自 BrainAlpha FM-1~FM-6 分类思想）。
        """
        hints = []
        if result.ic_mean < 0:
            hints.append(
                "Signal is reversed — multiply factor values by -1 before cross-sectional normalization."
            )
        elif result.ic_mean < GATES["ic_mean"]:
            hints.append(
                "IC too low. Try: (1) subtract group/industry mean before ranking to add neutralization, "
                "(2) increase the lookback window to reduce noise, "
                "(3) ensure rank normalization is applied cross-sectionally each date."
            )
        if result.icir < GATES["icir"]:
            hints.append(
                "ICIR too low (signal unstable). Try: (1) apply exponential decay weighting "
                "(e.g., ewm(halflife=N)) to smooth the factor, "
                "(2) winsorize at 3-sigma to reduce outlier impact on ranking."
            )
        if result.coverage < GATES["coverage"]:
            hints.append(
                "Coverage too low. Try: (1) reduce min_periods in rolling operations, "
                "(2) fill missing values with group median before ranking, "
                "(3) ensure you are iterating over all stocks not just non-NaN ones."
            )
        if result.ic_decay_ratio < GATES["ic_decay_ratio"]:
            hints.append(
                "Signal decays too fast (IC at 5d << IC at 1d). "
                "Try increasing lookback window or applying exponential decay weighting."
            )
        if result.monthly_win_rate < GATES["monthly_win_rate"]:
            hints.append(
                "Monthly win rate low. Verify the factor direction matches the hypothesis. "
                "Try cross-sectional z-score normalization to stabilize signal direction across time."
            )
        if result.max_correlation >= GATES["max_correlation"]:
            hints.append(
                "Factor too correlated with existing factors. "
                "Try: (1) compute industry-relative version (subtract sector mean), "
                "(2) orthogonalize against a common factor like market beta."
            )
        return " ".join(hints) if hints else "Review all quality gates and the factor formula."

    def _fail(self, name, code_path, reason):
        return FactorResult(
            factor_name=name, ic_mean=0, icir=0, ic_decay_ratio=0,
            coverage=0, monthly_win_rate=0, max_correlation=0,
            status="FAIL", fail_reason=reason, code_path=code_path,
        )
