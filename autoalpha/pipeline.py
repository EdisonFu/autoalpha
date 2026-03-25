import os
import pandas as pd
from typing import List, Dict, Optional

from .llm import LLMClient
from .db import FactorDB
from .pipeline_state import FactorSpec, FactorResult, FactorReport
from .agents.input_classifier import classify_input
from .agents.paper_search_agent import PaperSearchAgent
from .agents.factor_extract_agent import FactorExtractAgent
from .agents.factor_implement_agent import FactorImplementAgent, classify_error
from .agents.factor_eval_agent import FactorEvalAgent
from .agents.factor_extension_agent import FactorExtensionAgent
from .agents.factor_judge_agent import FactorJudgeAgent
from .agents.report_agent import ReportAgent


_DEFAULT_UNIVERSE = "000300"   # CSI300
_DEFAULT_START    = "2021-01-01"
_DEFAULT_END      = "2024-12-31"
_MAX_IMPLEMENT_RETRIES = 3     # 最多尝试次数（含初始生成）
_MAX_EXTENSIONS_PER_FACTOR = 3

# IC 变化小于此阈值视为停滞
_STAGNATION_DELTA = 0.002
# IC 下降超过此阈值视为恶化
_DETERIORATION_DELTA = -0.005


class AutoAlphaPipeline:
    """
    AutoAlpha 因子挖掘流水线

    流程：
      输入 → 论文检索 → 用户选择论文 → 因子提取 → 因子实现 → 因子评估 → 延伸 → 报告

    改进（v2）：
      - 故障分类：TIMEOUT/ENV 立即硬停止，不浪费重试次数
      - 评估失败回路：代码成功但指标不达标时，将定向修复建议反馈给 LLM 重新生成
      - 收敛控制：IC 停滞或恶化时提前终止，避免无效重试
    """

    def __init__(self, llm_config: dict = None,
                 db_path: str = "autoalpha.db",
                 output_dir: str = "./output",
                 universe: str = _DEFAULT_UNIVERSE,
                 start_date: str = _DEFAULT_START,
                 end_date: str = _DEFAULT_END,
                 universe_size: int = 50):
        self.llm   = LLMClient(llm_config)
        self.db    = FactorDB(db_path)
        self.universe_code = universe
        self.start_date    = start_date
        self.end_date      = end_date
        self.universe_size = universe_size

        self.paper_search     = PaperSearchAgent(self.llm)
        self.factor_extract   = FactorExtractAgent(self.llm)
        self.factor_implement = FactorImplementAgent(self.llm, output_dir=os.path.join(output_dir, "factors"))
        self.factor_eval      = FactorEvalAgent()
        self.factor_judge     = FactorJudgeAgent(self.llm)
        self.factor_ext       = FactorExtensionAgent(self.llm)
        self.reporter         = ReportAgent(output_dir=os.path.join(output_dir, "reports"))

        self._price_cache: Optional[Dict] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def search_papers_paged(self, idea: str, page: int = 1, page_size: int = 10):
        return self.paper_search.search_paged(idea, page, page_size)

    def run(self, idea: str, selected_papers: List[Dict] = None) -> FactorReport:
        """Run full pipeline. selected_papers can be pre-chosen by user."""
        input_type, confidence = classify_input(idea)
        print(f"[Pipeline] 输入类型={input_type}, 搜索置信度={confidence:.2f}")

        # Step 1: Paper search
        papers = selected_papers or []
        if not papers and input_type == "idea" and confidence >= 0.6:
            papers_all, _ = self.paper_search.search(idea, max_results=5)
            papers = papers_all

        if not papers:
            print("[Pipeline] 无参考论文，将直接基于研究方向生成因子假设")
            papers = [{"title": idea, "abstract": idea, "pdf_url": "", "date": ""}]

        # Step 2: Extract factor specs
        all_specs: List[FactorSpec] = []
        for paper in papers:
            specs = self.factor_extract.extract(paper)
            all_specs.extend(specs)

        print(f"[Pipeline] 共提取 {len(all_specs)} 个因子假设")

        if not all_specs:
            print("[Pipeline] 未提取到因子，退出")
            return FactorReport(idea=idea, papers_used=papers, total_extracted=0,
                                total_tested=0, passed=[], failed=[], extensions_passed=[])

        # Step 3: Load price data (cached)
        price_data = self._get_price_data()
        if not price_data:
            print("[Pipeline] 无法获取价格数据，退出")
            return FactorReport(idea=idea, papers_used=papers, total_extracted=len(all_specs),
                                total_tested=0, passed=[], failed=[], extensions_passed=[])

        self.factor_eval.existing_ic_series = self.db.get_all_ic_series()

        # Step 4: Implement and evaluate
        passed: List[FactorResult] = []
        failed: List[FactorResult] = []

        for i, spec in enumerate(all_specs, 1):
            print(f"[Pipeline] 处理因子 {i}/{len(all_specs)}: {spec.factor_name}")
            exp_ctx = self._get_experience_context(spec)
            result = self._implement_and_eval(spec, price_data, experience_context=exp_ctx)
            if result.status == "PASS":
                passed.append(result)
                self.db.save_factor(result, spec)
                self.factor_eval.existing_ic_series.append(result.ic_series)
            else:
                failed.append(result)
            self._save_experience(spec, result)

        # Step 5: Extensions for passing factors
        extensions_passed: List[FactorResult] = []
        for base_result in passed:
            base_spec = next((s for s in all_specs if s.factor_name == base_result.factor_name), None)
            if not base_spec:
                continue
            ext_specs = self.factor_ext.propose_extensions(base_spec, base_result)
            for ext_spec in ext_specs[:_MAX_EXTENSIONS_PER_FACTOR]:
                exp_ctx = self._get_experience_context(ext_spec)
                ext_result = self._implement_and_eval(ext_spec, price_data, experience_context=exp_ctx)
                if ext_result.status == "PASS":
                    extensions_passed.append(ext_result)
                    self.db.save_factor(ext_result, ext_spec)
                    self.factor_eval.existing_ic_series.append(ext_result.ic_series)
                self._save_experience(ext_spec, ext_result)

        # Step 6: Report
        report = FactorReport(
            idea=idea,
            papers_used=papers,
            total_extracted=len(all_specs),
            total_tested=len(all_specs),
            passed=passed,
            failed=failed,
            extensions_passed=extensions_passed,
        )
        report_path = self.reporter.generate(report)
        self.db.save_session(idea, len(papers), len(all_specs),
                             len(passed) + len(extensions_passed), report_path)
        return report

    # ── Core loop ──────────────────────────────────────────────────────────────

    def _implement_and_eval(self, spec: FactorSpec, price_data: dict,
                             retries: int = _MAX_IMPLEMENT_RETRIES,
                             experience_context: str = None) -> FactorResult:
        """
        因子实现 + 评估的核心循环，包含：
        1. 错误分类 + 硬停止（TIMEOUT/ENV）
        2. 评估失败 → 定向修复建议反馈给下轮生成
        3. IC 停滞/恶化收敛控制
        """
        error_ctx: Optional[str] = None
        ic_history: List[float] = []
        best_result: Optional[FactorResult] = None   # best eval result so far

        for attempt in range(1, retries + 2):  # 最多 retries+1 次尝试
            # ── 实现阶段 ──────────────────────────────────────────────────────
            csv_path, impl_error = self.factor_implement.implement(
                spec, price_data, error_ctx,
                experience_context=experience_context if not error_ctx else None,
            )

            if impl_error:
                err_type = classify_error(impl_error)
                print(f"[Pipeline] 实现失败（第{attempt}次）[{err_type}]: {impl_error[:120]}")

                # 硬停止：环境问题和超时无法通过修代码解决
                if err_type in ("TIMEOUT", "ENV"):
                    return FactorResult.failed(
                        spec.factor_name, "",
                        f"[{err_type}] 不可恢复错误，停止重试: {impl_error[:200]}"
                    )

                if attempt > retries:
                    return best_result or FactorResult.failed(
                        spec.factor_name, "",
                        f"实现失败（{retries+1}次尝试后放弃）: {impl_error[:200]}"
                    )
                error_ctx = impl_error
                continue

            # ── 评估阶段 ──────────────────────────────────────────────────────
            result = self.factor_eval.evaluate(spec, csv_path, price_data)
            print(f"[Pipeline] {spec.factor_name}: "
                  f"IC={result.ic_mean:.4f} ICIR={result.icir:.4f} → {result.status}")
            if best_result is None or result.ic_mean > best_result.ic_mean:
                best_result = result

            if result.status == "PASS":
                # ── LLM Judge: semantic peer review ───────────────────────────
                # result.code_path holds the CSV; derive the .py file for code review
                py_path = (csv_path.replace(".csv", ".py")
                           if csv_path and csv_path.endswith(".csv") else "")
                verdict = self.factor_judge.judge(spec, result, py_path)
                result.judge_score = float(verdict.get("overall_score", 0.0))
                result.judge_comment = (
                    verdict.get("rejection_reason") or
                    verdict.get("improvement_suggestions", "")
                )
                if verdict.get("verdict") == "REJECT":
                    result.status = "FAIL"
                    result.fail_reason = f"[JUDGE] {verdict.get('rejection_reason', 'semantic review failed')}"
                    print(f"[Pipeline] Judge 拒绝 {spec.factor_name}: {result.fail_reason}")
                else:
                    return result

            # ── 收敛控制 ──────────────────────────────────────────────────────
            ic_history.append(result.ic_mean)
            if len(ic_history) >= 2:
                delta = ic_history[-1] - ic_history[-2]
                if abs(delta) < _STAGNATION_DELTA:
                    print(f"[Pipeline] 停滞检测：连续迭代 IC 无改善（Δ={delta:.4f}），停止修复")
                    return best_result
                if delta < _DETERIORATION_DELTA:
                    print(f"[Pipeline] 恶化检测：IC 下降（Δ={delta:.4f}），停止修复")
                    return best_result

            if attempt > retries:
                return best_result

            # ── 评估失败 → 定向修复建议反馈 ──────────────────────────────────
            repair_hint = self.factor_eval.get_repair_hints(result)
            judge_suggestions = getattr(result, "judge_comment", "")
            extra = (f"\n\nLLM judge suggestions:\n{judge_suggestions}"
                     if judge_suggestions and "[JUDGE]" in result.fail_reason else "")
            error_ctx = (
                f"[EVAL_FAIL] Code ran without errors but failed quality gates.\n"
                f"Failed metrics: {result.fail_reason}\n\n"
                f"Targeted repair suggestions:\n{repair_hint}{extra}"
            )
            print(f"[Pipeline] 评估未达标，尝试定向修复（第{attempt}次）: "
                  f"{result.fail_reason[:100]}")

        return best_result  # unreachable but satisfies type checker

    # ── Experience memory (RALPH loop) ─────────────────────────────────────────

    def _save_experience(self, spec: FactorSpec, result: FactorResult) -> None:
        """Save mining experience for future retrieval (FactorMiner RALPH design)."""
        keyword = " ".join(spec.hypothesis.lower().split()[:5])
        failure_mode = ""
        if result.status == "FAIL":
            r = result.fail_reason.upper()
            if "TIMEOUT" in r:       failure_mode = "TIMEOUT"
            elif "ENV" in r:         failure_mode = "ENV"
            elif "[JUDGE]" in r:     failure_mode = "JUDGE_REJECT"
            elif "IC均值" in r or "IC_MEAN" in r.upper(): failure_mode = "LOW_IC"
            elif "ICIR" in r:        failure_mode = "LOW_ICIR"
            elif "覆盖率" in r or "COVERAGE" in r.upper(): failure_mode = "LOW_COVERAGE"
            elif "相关" in r or "CORR" in r.upper(): failure_mode = "HIGH_CORR"
            else:                    failure_mode = "OTHER"
        key_insight = getattr(result, "judge_comment", "") or result.fail_reason[:200]
        try:
            self.db.save_experience(
                factor_name=spec.factor_name,
                hypothesis_type=keyword,
                outcome=result.status,
                failure_mode=failure_mode,
                ic_mean=result.ic_mean,
                key_insight=key_insight,
            )
        except Exception as e:
            print(f"[Pipeline] 经验保存失败 (非致命): {e}")

    def _get_experience_context(self, spec: FactorSpec) -> str:
        """Fetch relevant mining experience to guide code generation (RALPH Retrieve step)."""
        keyword = " ".join(spec.hypothesis.lower().split()[:3])
        try:
            failures = self.db.get_similar_failures(keyword, limit=3)
            successes = self.db.get_successful_patterns(limit=2)
        except Exception:
            return ""
        if not failures and not successes:
            return ""
        lines = []
        if failures:
            lines.append("Avoid these previously-observed failure patterns for similar factors:")
            for f in failures:
                lines.append(f"  [{f['failure_mode']}] {f['key_insight']}")
        if successes:
            lines.append("Successful patterns to draw inspiration from:")
            for s in successes:
                lines.append(f"  [{s['hypothesis_type']}] IC={s['ic_mean']:.4f}: {s['key_insight']}")
        return "\n".join(lines)

    # ── Data loading ───────────────────────────────────────────────────────────

    def _get_price_data(self) -> Optional[Dict]:
        if self._price_cache is not None:
            return self._price_cache
        print(f"[Pipeline] 获取股票池数据（{self.universe_code}, {self.start_date}~{self.end_date}）...")
        try:
            import akshare as ak
        except ImportError as e:
            print(f"[Pipeline] ⚠ 未安装 akshare: {e}")
            return None
        try:
            cons_df = ak.index_stock_cons_weight_csindex(symbol=self.universe_code)
            stocks = cons_df["成分券代码"].tolist()[:self.universe_size]
        except Exception as e:
            print(f"[Pipeline] 获取成分股失败: {e}，使用预设列表")
            stocks = ["600519","000858","601318","600036","000333",
                      "600900","601899","600276","300750","000568",
                      "601988","600887","600309","601166","000002",
                      "002594","600030","601398","600104","000001"]

        price_data = {}
        sd = self.start_date.replace("-","")
        ed = self.end_date.replace("-","")
        for code in stocks:
            try:
                sym = f"sh{code}" if code[0] in "690" else f"sz{code}"
                df = ak.stock_zh_a_daily(symbol=sym, start_date=sd, end_date=ed, adjust="qfq")
                if df is not None and not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                    for col in ["open","high","low","close","volume"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    price_data[code] = df
            except Exception as e:
                print(f"[Pipeline] 获取 {code} 数据失败: {e}")
        print(f"[Pipeline] 成功获取 {len(price_data)} 只股票数据")
        if price_data:
            self._price_cache = price_data
        return price_data or None

    def add_data_adapter(self, adapter):
        print(f"[Pipeline] 数据适配器注册（pipeline 使用内置 AKShare 获取股票池数据）: {adapter}")

    def set_local_data(self, paths):
        print(f"[Pipeline] 本地数据路径: {paths}")

    def list_data_adapters(self) -> List[Dict]:
        return []
