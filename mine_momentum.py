#!/usr/bin/env python3
"""
Continuous closed-loop momentum factor mining using local klines data.

Usage:
    python mine_momentum.py [--freq etf-daily|min60|min5|min1]
                            [--target 100]
                            [--klines /path/to/klines]
                            [--output ./output_momentum]

Mines high/low-frequency momentum factors until `--target` pass the quality gates.
Each iteration:
  1. LLM generates a batch of diverse momentum factor hypotheses
  2. Each factor is implemented, sandboxed, evaluated (IC/ICIR/coverage/decay)
  3. LLM Judge does semantic peer review
  4. PASS factors are saved to DB + report; experience memory feeds next iteration
  5. Loop continues, biasing toward unexplored momentum sub-types
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

from autoalpha.llm import LLMClient, CreditExhaustedError
from autoalpha.db import FactorDB
from autoalpha.pipeline_state import FactorSpec, FactorResult
from autoalpha.agents.factor_implement_agent import FactorImplementAgent
from autoalpha.agents.factor_eval_agent import FactorEvalAgent
from autoalpha.agents.factor_judge_agent import FactorJudgeAgent
from autoalpha.agents.factor_extension_agent import FactorExtensionAgent
from autoalpha.agents.report_agent import ReportAgent
from autoalpha.pipeline_state import FactorReport
from autoalpha.pipeline import AutoAlphaPipeline
from autoalpha.data_loader import load_klines
from autoalpha.utils import strip_code_fence

# ── Momentum sub-types to explore (ensures diversity across 100 factors) ──────
# Each iteration we pick under-explored sub-types to prevent clustering
# freq tag: "daily" = daily-bar factors only; "intraday" = intraday factors only; "both" = any
_MOMENTUM_SUBTYPES = [
    # --- Low-frequency (daily) ---
    ("price_momentum",      "Price momentum: past N-day return predicts future return",                          "daily"),
    ("risk_adj_momentum",   "Risk-adjusted momentum: return divided by realized volatility",                     "daily"),
    ("volume_momentum",     "Volume-weighted momentum: price change scaled by volume surge",                     "both"),
    ("relative_momentum",   "Relative strength: asset return minus cross-sectional median return",               "daily"),
    ("residual_momentum",   "Residual momentum: idiosyncratic return after market beta removal",                 "daily"),
    ("52w_breakout",        "52-week high proximity: distance from 252-day high captures anchoring bias",        "daily"),
    ("drawdown_recovery",   "Drawdown recovery: momentum in assets recovering from maximum drawdown",            "daily"),
    ("dispersion_momentum", "Dispersion momentum: high-low range normalized by ATR as trend strength",           "both"),
    ("vol_trend",           "Volatility trend: rising/falling realized volatility as momentum signal",           "daily"),
    ("skew_momentum",       "Skewness momentum: past return skewness predicts future cross-sectional return",    "daily"),
    ("kurtosis_signal",     "Kurtosis signal: excess kurtosis as tail-risk adjusted momentum predictor",         "daily"),
    ("cross_asset_spread",  "Cross-asset momentum spread: relative momentum between sector/country ETFs",        "daily"),
    ("trend_consistency",   "Trend consistency: fraction of up-days in lookback window as momentum quality",     "daily"),
    ("turnover_adjusted",   "Turnover-adjusted momentum: price momentum scaled by volume turnover ratio",        "both"),
    ("return_acceleration", "Return acceleration: second-derivative of price trend, capturing trend strengthening","daily"),
    # --- High-frequency (intraday) ---
    ("intraday_reversal",   "Intraday reversal: opening gap fade and overnight return mean-reversion",           "intraday"),
    ("vwap_deviation",      "VWAP deviation: price distance from VWAP proxy drives intraday mean reversion",    "intraday"),
    ("open_drive",          "Opening drive momentum: first-bar direction persistence within session",            "intraday"),
    ("volume_clock",        "Volume clock: time-of-day volume pattern captures intraday momentum regime",        "intraday"),
    ("range_breakout",      "Range breakout: intraday high/low range expansion as momentum signal",              "intraday"),
    ("microstructure_mom",  "Microstructure momentum: bid-ask spread proxy from high-low range",                "intraday"),
    ("overnight_gap",       "Overnight gap momentum: gap direction persistence across sessions",                  "intraday"),
    ("session_split",       "Session split: morning vs afternoon return differential as persistence signal",     "intraday"),
]

_HYPOTHESIS_SYSTEM = """\
You are a quantitative researcher generating diverse alpha factor hypotheses.

Given a momentum sub-type and its description, generate 3-5 concrete, distinct factor hypotheses.
Each hypothesis must:
1. Have a unique economic rationale (not just parameter variation)
2. Be implementable using only OHLCV data (open, high, low, close, volume, amount)
3. Include a precise mathematical formula
4. Be appropriate for the data frequency specified

Output JSON array, each element:
{
  "factor_name": "snake_case_name",
  "description": "one sentence describing what the factor measures",
  "hypothesis": "why this predicts future returns (economic mechanism)",
  "formula": "mathematical formula using OHLCV notation",
  "implementation_notes": "key implementation pitfalls to avoid",
  "parameters": {"lookback": N, ...},
  "data_freq": "daily or intraday"
}

Output ONLY valid JSON array, no explanation.
"""


class MomentumMiner:
    """
    Closed-loop momentum factor miner.
    Continuously generates, implements, evaluates, and learns until target is reached.
    """

    def __init__(self, klines_dir: str, freq: str = "etf-daily",
                 target: int = 100, output_dir: str = "./output_momentum",
                 db_path: str = "momentum_factors.db"):
        self.klines_dir = klines_dir
        self.freq = freq
        self.target = target
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.llm = LLMClient()
        self.db = FactorDB(db_path)
        self.implement_agent = FactorImplementAgent(
            self.llm, output_dir=os.path.join(output_dir, "factors")
        )
        # ETF universe has only 24 assets — IC noise floor ≈ 1/sqrt(24) ≈ 0.20
        # Relax ICIR gate: typical achievable ICIR for IC=0.03 is ~0.15 with 24 assets
        # Keep IC_mean gate but relax ICIR and win_rate accordingly
        # 24 ETFs across diverse asset classes: IC noise ≈ 1/sqrt(24) ≈ 0.20 per date.
        # Observed IC_std ≈ 0.35-0.40 due to cross-asset heterogeneity.
        # ICIR = IC_mean/IC_std: for IC_mean=0.03, IC_std=0.40 → ICIR≈0.075.
        # t-stat = ICIR * sqrt(T) = 0.06 * sqrt(2357) ≈ 2.9 (still significant at p<0.004).
        etf_gates = {
            "ic_mean":          0.02,    # cross-asset momentum threshold
            "icir":             0.06,    # ICIR*sqrt(2357)=2.9σ → significant at 99% level
            "ic_decay_ratio":   0.25,    # cross-asset trend can be slow-decaying
            "coverage":         0.60,    # some ETFs have shorter histories
            "monthly_win_rate": 0.51,    # small consistent edge
            "max_correlation":  0.90,    # higher threshold for small homogeneous-ish set
        }
        self.eval_agent = FactorEvalAgent(gates=etf_gates)
        self.judge_agent = FactorJudgeAgent(self.llm)
        self.ext_agent = FactorExtensionAgent(self.llm)
        self.reporter = ReportAgent(output_dir=os.path.join(output_dir, "reports"))

        # Pipeline instance for experience memory helpers
        self._pipeline = _PipelineHelper(self.llm, self.db)

        # Filter subtypes by frequency
        _freq_class = "daily" if freq == "etf-daily" else "intraday"
        self._active_subtypes = [
            (k, d) for k, d, f in _MOMENTUM_SUBTYPES if f in (_freq_class, "both")
        ]
        # Priority subtypes: start at count=0 so they're explored first.
        # Others start at count=1 so they follow in random order.
        _PRIORITY_KEYS = {"price_momentum", "risk_adj_momentum", "52w_breakout",
                          "relative_momentum", "trend_consistency"}
        self._subtype_counts: Dict[str, int] = {
            k: (0 if k in _PRIORITY_KEYS else 1)
            for k, _ in self._active_subtypes
        }

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Momentum Factor Miner")
        print(f"  Target: {self.target} passing factors")
        print(f"  Frequency: {self.freq}")
        print(f"  Data: {self.klines_dir}")
        print(f"{'='*60}\n")

        # Load data once
        data = load_klines(self.klines_dir, self.freq)
        if not data:
            print("[Miner] ✗ 无法加载数据，退出")
            return

        self.eval_agent.existing_ic_series = self.db.get_all_ic_series()

        passed_results: List[FactorResult] = []
        failed_results: List[FactorResult] = []
        iteration = 0

        while len(passed_results) < self.target:
            iteration += 1
            n_passed = len(passed_results)
            remaining = self.target - n_passed
            print(f"\n[Miner] ── 第 {iteration} 轮 | 已通过: {n_passed}/{self.target} "
                  f"| 还需: {remaining} ──")

            # Pick least-explored sub-types (diversity control)
            subtype_key, subtype_desc = self._pick_subtype()
            self._subtype_counts[subtype_key] += 1

            print(f"[Miner] 探索子类型: {subtype_key} — {subtype_desc}")

            # Generate hypotheses for this sub-type
            try:
                specs = self._generate_hypotheses(subtype_key, subtype_desc, data)
            except CreditExhaustedError:
                print("[Miner] ⚠ API额度不足，等待60秒...")
                time.sleep(60)
                specs = []
            if not specs:
                print(f"[Miner] 生成失败，跳过此轮")
                continue

            print(f"[Miner] 生成 {len(specs)} 个因子假设")

            for spec in specs:
                if len(passed_results) >= self.target:
                    break

                print(f"\n[Miner] 实现因子: {spec.factor_name}")
                try:
                    result = self._implement_and_eval(spec, data)
                except CreditExhaustedError:
                    print("[Miner] ⚠ API额度不足，等待60秒后重试...")
                    time.sleep(60)
                    try:
                        result = self._implement_and_eval(spec, data)
                    except CreditExhaustedError:
                        print("[Miner] ✗ API额度持续不足，跳过此因子")
                        continue

                if result.status == "PASS":
                    passed_results.append(result)
                    self.db.save_factor(result, spec)
                    self.eval_agent.existing_ic_series.append(result.ic_series)
                    print(f"[Miner] ✓ PASS [{len(passed_results)}/{self.target}] "
                          f"IC={result.ic_mean:.4f} ICIR={result.icir:.4f} "
                          f"Judge={result.judge_score:.1f}")

                    # Extend passing factors (free factor amplification)
                    if len(passed_results) < self.target:
                        self._run_extensions(spec, result, data, passed_results)
                else:
                    failed_results.append(result)
                    print(f"[Miner] ✗ FAIL: {result.fail_reason[:100]}")

                self._pipeline._save_experience(spec, result)

            # Periodic progress report
            if iteration % 5 == 0 or len(passed_results) >= self.target:
                self._save_report(passed_results, failed_results, iteration)

        print(f"\n{'='*60}")
        print(f"  挖掘完成！通过因子: {len(passed_results)}")
        print(f"  总尝试: {len(passed_results) + len(failed_results)}")
        print(f"  成功率: {len(passed_results)/(len(passed_results)+len(failed_results)+1e-9)*100:.1f}%")
        print(f"{'='*60}")

        self._save_report(passed_results, failed_results, iteration, final=True)

    # ── Core factor loop ────────────────────────────────────────────────────────

    def _implement_and_eval(self, spec: FactorSpec, data: dict,
                             max_retries: int = 2) -> FactorResult:
        # max_retries=2: up to 3 attempts if implementation crashes (SYNTAX/RUNTIME errors).
        # We skip post-eval repair (IC quality improvement) because it rarely helps for
        # the ETF universe and wastes API credits on syntax-error-prone repair calls.
        from autoalpha.agents.factor_implement_agent import classify_error
        error_ctx = None
        ic_history = []
        exp_ctx = self._pipeline._get_experience_context(spec)
        best_result = None   # track best eval result so far (never discard working code)

        for attempt in range(1, max_retries + 2):
            csv_path, impl_error = self.implement_agent.implement(
                spec, data, error_ctx,
                experience_context=exp_ctx if not error_ctx else None,
            )

            if impl_error:
                err_type = classify_error(impl_error)
                print(f"  [实现] 第{attempt}次失败 [{err_type}]: {impl_error[:80]}")
                if err_type in ("TIMEOUT", "ENV"):
                    return FactorResult.failed(spec.factor_name, "",
                                               f"[{err_type}] 不可恢复: {impl_error[:150]}")
                if attempt > max_retries:
                    # Return best eval result if we have one, else hard fail
                    return best_result or FactorResult.failed(
                        spec.factor_name, "",
                        f"实现失败({max_retries+1}次): {impl_error[:150]}"
                    )
                error_ctx = impl_error
                continue

            result = self.eval_agent.evaluate(spec, csv_path, data)
            print(f"  [评估] IC={result.ic_mean:.4f} ICIR={result.icir:.4f} "
                  f"cov={result.coverage:.3f} → {result.status}")

            # Keep best result (highest IC) across all attempts
            if best_result is None or result.ic_mean > best_result.ic_mean:
                best_result = result

            if result.status == "PASS":
                # code_path in result holds the CSV path; derive the .py path for judge
                py_path = (csv_path.replace(".csv", ".py")
                           if csv_path and csv_path.endswith(".csv") else "")
                verdict = self.judge_agent.judge(spec, result, py_path)
                result.judge_score = float(verdict.get("overall_score", 0.0))
                result.judge_comment = (verdict.get("rejection_reason") or
                                        verdict.get("improvement_suggestions", ""))
                if verdict.get("verdict") == "REJECT":
                    result.status = "FAIL"
                    result.fail_reason = f"[JUDGE] {verdict.get('rejection_reason','')}"
                    print(f"  [Judge] REJECT: {result.fail_reason[:80]}")
                    best_result = result
                else:
                    return result

            # Eval succeeded but gates failed: don't waste credits on repair calls.
            # Just log the result and move on to the next factor hypothesis.
            return best_result

        return best_result

    def _run_extensions(self, base_spec: FactorSpec, base_result: FactorResult,
                        data: dict, passed_results: List[FactorResult],
                        max_ext: int = 3) -> None:
        try:
            ext_specs = self.ext_agent.propose_extensions(base_spec, base_result)
        except Exception as e:
            print(f"  [Extension] 生成失败: {e}")
            return

        for ext_spec in ext_specs[:max_ext]:
            if len(passed_results) >= self.target:
                break
            print(f"\n[Miner] 延伸因子: {ext_spec.factor_name}")
            ext_result = self._implement_and_eval(ext_spec, data)
            if ext_result.status == "PASS":
                passed_results.append(ext_result)
                self.db.save_factor(ext_result, ext_spec)
                self.eval_agent.existing_ic_series.append(ext_result.ic_series)
                print(f"[Miner] ✓ 延伸PASS [{len(passed_results)}/{self.target}] "
                      f"IC={ext_result.ic_mean:.4f}")
            self._pipeline._save_experience(ext_spec, ext_result)

    # ── Hypothesis generation ───────────────────────────────────────────────────

    def _generate_hypotheses(self, subtype_key: str, subtype_desc: str,
                              data: dict) -> List[FactorSpec]:
        available_cols = set()
        for df in list(data.values())[:3]:
            available_cols.update(df.columns.tolist())

        freq_label = "intraday" if self.freq != "etf-daily" else "daily"
        user_msg = (
            f"Momentum sub-type: {subtype_key}\n"
            f"Description: {subtype_desc}\n"
            f"Data frequency: {freq_label} ({self.freq})\n"
            f"Available columns: {sorted(available_cols)}\n"
            f"Number of assets: {len(data)}\n\n"
            f"Generate 3-5 diverse, concrete momentum factor hypotheses for this sub-type. "
            f"Each must have a distinct economic mechanism (not just different lookback values)."
        )

        try:
            raw = strip_code_fence(self.llm.complete(
                system=_HYPOTHESIS_SYSTEM, user=user_msg, temperature=0.9
            ))
            items = json.loads(raw)
            if not isinstance(items, list):
                items = [items]
        except Exception as e:
            print(f"[Miner] 假设生成失败: {e}")
            return []

        specs = []
        for item in items:
            try:
                specs.append(FactorSpec(
                    factor_name=item.get("factor_name", f"{subtype_key}_{len(specs)}"),
                    description=item.get("description", ""),
                    hypothesis=item.get("hypothesis", subtype_desc),
                    formula=item.get("formula", ""),
                    implementation_notes=item.get("implementation_notes", ""),
                    data_requirements={"freq": freq_label, "fields": list(available_cols)},
                    parameters=item.get("parameters", {}),
                    source_paper={"title": f"AutoMomentum/{subtype_key}", "url": "", "date": ""},
                ))
            except Exception as e:
                print(f"[Miner] FactorSpec 构建失败: {e}")
        return specs

    # ── Diversity control ───────────────────────────────────────────────────────

    def _pick_subtype(self) -> tuple:
        """Pick the least-explored sub-type; break ties randomly for variety."""
        import random
        min_count = min(self._subtype_counts.values())
        candidates = [(k, d) for k, d in self._active_subtypes
                      if self._subtype_counts[k] == min_count]
        return random.choice(candidates)

    # ── Reporting ───────────────────────────────────────────────────────────────

    def _save_report(self, passed: List[FactorResult], failed: List[FactorResult],
                     iteration: int, final: bool = False) -> None:
        label = "最终" if final else f"第{iteration}轮"
        report = FactorReport(
            idea=f"动量因子挖掘 ({self.freq}) — {label}",
            papers_used=[{"title": "AutoMomentum", "pdf_url": "", "date": ""}],
            total_extracted=len(passed) + len(failed),
            total_tested=len(passed) + len(failed),
            passed=passed,
            failed=[],          # keep report focused on passed factors
            extensions_passed=[],
        )
        path = self.reporter.generate(report)
        print(f"[Miner] 报告已保存: {path}")


class _PipelineHelper(AutoAlphaPipeline):
    """Thin subclass to reuse experience memory helpers without loading akshare."""

    def __init__(self, llm, db):
        # Bypass parent __init__ — only need the experience memory methods
        self.llm = llm
        self.db = db


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Momentum Factor Miner")
    parser.add_argument("--freq", default="etf-daily",
                        choices=["etf-daily", "min60", "min5", "min1"],
                        help="K线频率")
    parser.add_argument("--target", type=int, default=100,
                        help="目标通过因子数量")
    parser.add_argument("--klines", default="/Users/fuzhendong/Desktop/cc论文/data/klines",
                        help="klines数据根目录")
    parser.add_argument("--output", default="./output_momentum",
                        help="输出目录")
    parser.add_argument("--db", default="momentum_factors.db",
                        help="数据库文件路径")
    args = parser.parse_args()

    miner = MomentumMiner(
        klines_dir=args.klines,
        freq=args.freq,
        target=args.target,
        output_dir=args.output,
        db_path=args.db,
    )
    miner.run()


if __name__ == "__main__":
    main()
