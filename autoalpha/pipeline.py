import os
import pandas as pd
from typing import List, Dict, Optional

from .llm import LLMClient
from .db import FactorDB
from .pipeline_state import FactorSpec, FactorResult, FactorReport
from .agents.input_classifier import classify_input
from .agents.paper_search_agent import PaperSearchAgent
from .agents.factor_extract_agent import FactorExtractAgent
from .agents.factor_implement_agent import FactorImplementAgent
from .agents.factor_eval_agent import FactorEvalAgent
from .agents.factor_extension_agent import FactorExtensionAgent
from .agents.report_agent import ReportAgent


_DEFAULT_UNIVERSE = "000300"   # CSI300
_DEFAULT_START    = "2021-01-01"
_DEFAULT_END      = "2024-12-31"
_MAX_IMPLEMENT_RETRIES = 2
_MAX_EXTENSIONS_PER_FACTOR = 3


class AutoAlphaPipeline:
    """
    AutoAlpha 因子挖掘流水线

    流程：
      输入 → 论文检索 → 用户选择论文 → 因子提取 → 因子实现 → 因子评估 → 延伸 → 报告
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
        self.factor_ext       = FactorExtensionAgent(self.llm)
        self.reporter         = ReportAgent(output_dir=os.path.join(output_dir, "reports"))

        self._price_cache: Optional[Dict] = None  # cached price data

    # ── Public API ────────────────────────────────────────────────────────────

    def search_papers_paged(self, idea: str, page: int = 1, page_size: int = 10):
        return self.paper_search.search_paged(idea, page, page_size)

    def run(self, idea: str, selected_papers: List[Dict] = None) -> FactorReport:
        """
        Run full pipeline. selected_papers can be pre-chosen by user.
        """
        input_type, confidence = classify_input(idea)
        print(f"[Pipeline] 输入类型={input_type}, 搜索置信度={confidence:.2f}")

        # Step 1: Paper search (if idea)
        papers = selected_papers or []
        if not papers and input_type == "idea" and confidence >= 0.6:
            papers_all, _ = self.paper_search.search(idea, max_results=5)
            papers = papers_all

        if not papers:
            print("[Pipeline] 无参考论文，将直接基于研究方向生成因子假设")
            # Fallback: generate factor specs directly from the idea
            papers = [{"title": idea, "abstract": idea, "pdf_url": "", "date": ""}]

        # Step 2: Extract factor specs from papers
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

        # Update eval agent with existing factors
        self.factor_eval.existing_ic_series = self.db.get_all_ic_series()

        # Step 4: Implement and evaluate each factor
        passed: List[FactorResult] = []
        failed: List[FactorResult] = []

        for i, spec in enumerate(all_specs, 1):
            print(f"[Pipeline] 处理因子 {i}/{len(all_specs)}: {spec.factor_name}")
            result = self._implement_and_eval(spec, price_data)
            if result.status == "PASS":
                passed.append(result)
                self.db.save_factor(result, spec)
                self.factor_eval.existing_ic_series.append(result.ic_series)
            else:
                failed.append(result)

        # Step 5: Extensions for passing factors
        extensions_passed: List[FactorResult] = []
        for base_result in passed:
            base_spec = next((s for s in all_specs if s.factor_name == base_result.factor_name), None)
            if not base_spec:
                continue
            ext_specs = self.factor_ext.propose_extensions(base_spec, base_result)
            for ext_spec in ext_specs[:_MAX_EXTENSIONS_PER_FACTOR]:
                ext_result = self._implement_and_eval(ext_spec, price_data)
                if ext_result.status == "PASS":
                    extensions_passed.append(ext_result)
                    self.db.save_factor(ext_result, ext_spec)
                    self.factor_eval.existing_ic_series.append(ext_result.ic_series)

        # Step 6: Generate report
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _implement_and_eval(self, spec: FactorSpec, price_data: dict,
                             retries: int = _MAX_IMPLEMENT_RETRIES) -> FactorResult:
        error_ctx = None
        for attempt in range(1, retries + 2):
            csv_path, error = self.factor_implement.implement(spec, price_data, error_ctx)
            if error:
                print(f"[Pipeline] 因子实现失败（第{attempt}次）: {error[:150]}")
                error_ctx = error
                if attempt > retries:
                    return FactorResult.failed(spec.factor_name, "", f"实现失败: {error[:200]}")
                continue
            result = self.factor_eval.evaluate(spec, csv_path, price_data)
            print(f"[Pipeline] {spec.factor_name}: IC={result.ic_mean:.4f} ICIR={result.icir:.4f} → {result.status}")
            return result

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
        """返回已注册适配器列表（兼容 /source 命令）。"""
        return []
