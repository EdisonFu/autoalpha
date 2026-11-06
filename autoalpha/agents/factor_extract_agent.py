from typing import List
from ..llm import LLMClient
from ..pipeline_state import FactorSpec
from ..json_repair import loads_lenient, JSONRepairError

_EXTRACT_SYSTEM = """\
You are a quantitative finance researcher. Given a paper's title and its text
(either the full methodology sections or only the abstract — the user message
says which), extract 2-4 concrete, implementable factor hypotheses.

Each factor must be:
- Cross-sectional (ranks stocks at each time point)
- Computable from standard price/volume/financial data
- Based on a clear economic or behavioral hypothesis

Output a JSON array. Each element:
{
  "factor_name": "short_snake_case_name",
  "description": "what this factor measures (1-2 sentences)",
  "hypothesis": "why this factor predicts future returns (economic/behavioral logic)",
  "formula": "mathematical formula or pseudo-code",
  "implementation_notes": "key implementation details: lookback window, normalization, pitfalls",
  "data_requirements": {
    "freq": "daily",
    "fields": ["close", "volume"],
    "lookback_days": 252
  },
  "parameters": {"lookback": 20},
  "param_sources": {"lookback": "paper"},
  "computable": true,
  "required_fields": ["close", "volume"],
  "migration": {
    "source_market": "us",
    "changes": ["VIX → 20日已实现波动率(年化)", "SPY → A股ETF截面自身"]
  }
}

Rules:
- Only extract factors that are EXPLICITLY described or implied in the text
- Translate Chinese abstracts to English terminology
- If the paper describes a strategy (not a factor), extract the underlying signal as a factor
- MARKET MIGRATION IS THE JOB, NOT AN OBSTACLE. The paper may study US markets
  (SPY, S&P 500, VIX) — your target is Chinese A-share ETFs with daily OHLCV.
  The economic mechanism (momentum, reversal, low-volatility, turnover effects)
  is market-agnostic. Standard substitutions, and these are well-established
  practice, not hacks:
    VIX / implied volatility  → realized volatility from daily closes (annualized)
    SPY / S&P 500 index       → the A-share ETF cross-section itself
    size / market cap         → skip this leg (ETF universe has no cap data)
  Record every substitution in "migration": {"source_market": "...",
  "changes": ["原数据 → 替代方案", ...]}, in Chinese so users can read it.
  NEVER set computable=false merely because the paper used US data — that is
  what migration is for. computable=false is ONLY for signals that genuinely
  cannot be expressed with daily price/volume at all: news text, order-book
  microstructure, analyst estimates, quarterly fundamentals. When in doubt,
  build the migrated version and say so in migration.changes.
- NEVER invent a proxy and present it as the paper's signal: every substitution
  must be visible in migration.changes, and implementation_notes must state the
  original calibration (e.g. "论文阈值 VIX>25，迁移版用已实现波动率>35% 对应").
- Add a "computable" boolean and a "required_fields" list to each factor.
- "param_sources": for EVERY key in "parameters", say where the value came from:
  "paper" if the text states it explicitly, "inferred" if the text implies it
  without a number, "default" if you chose it yourself. Be strict — an abstract
  almost never states lookback windows or quantile counts, so those are usually
  "default". Users rely on this to know what is actually from the paper.
- Output ONLY valid JSON array, no explanation
"""


class FactorExtractAgent:
    """从论文摘要中提取因子假设列表"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.last_error = ""      # 供 orchestrator 把失败原因呈现给用户

    def extract(self, paper: dict, data_context: str = "") -> List[FactorSpec]:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        url = paper.get("pdf_url", "")
        date = paper.get("date", "")

        # 读全文（仅 arXiv 可得），拿不到就回落摘要。provenance 一路带到
        # FactorSpec —— 用户要能分辨参数是读自论文还是模型自己定的。
        from ..sources.fulltext import paper_context, detail_hits
        ctx = paper_context(paper)
        body, provenance = ctx["text"] or abstract, ctx["provenance"]
        hits = detail_hits(body)

        print(f"[FactorExtract] 解析论文: {title[:60]}")
        print(f"[FactorExtract]   {ctx['note']}；复现细节命中 "
              f"{len(hits)}/7 {hits}")

        if provenance == "fulltext":
            head = (f"Paper title: {title}\n\n"
                    f"Full-text methodology sections (not just the abstract):\n"
                    f"{body}")
        else:
            head = (f"Paper title: {title}\n\n"
                    f"ONLY THE ABSTRACT IS AVAILABLE ({ctx['note']}). "
                    f"Numeric choices are almost certainly not stated here — "
                    f"mark them \"default\" in param_sources.\n\n"
                    f"Abstract:\n{body}")
        user_msg = f"{data_context}\n\n{head}" if data_context else head
        try:
            raw = self.llm.complete(system=_EXTRACT_SYSTEM, user=user_msg,
                                    purpose="extract")
        except Exception as e:
            print(f"[FactorExtract] LLM 调用失败: {e}")
            self.last_error = f"LLM 调用失败: {e}"
            return []

        try:
            items = loads_lenient(raw, expect=list)
        except JSONRepairError as e:
            # 这里失败等于「提取到 0 个因子」，整轮白跑，必须让调用方知道
            print(f"[FactorExtract] JSON 解析失败: {str(e)[:300]}")
            self.last_error = f"因子假设解析失败: {str(e)[:200]}"
            return []
        if not isinstance(items, list):
            items = [items]
        self.last_error = ""

        specs = []
        for item in items:
            try:
                spec = FactorSpec(
                    factor_name=item.get("factor_name", "unknown_factor"),
                    description=item.get("description", ""),
                    hypothesis=item.get("hypothesis", ""),
                    formula=item.get("formula", ""),
                    implementation_notes=item.get("implementation_notes", ""),
                    data_requirements=item.get("data_requirements", {"freq": "daily", "fields": ["close"]}),
                    parameters=item.get("parameters", {}),
                    source_paper={"title": title, "url": url, "date": date,
                                  "abstract": abstract[:300],
                                  "provenance": provenance,
                                  "provenance_note": ctx["note"],
                                  "detail_hits": hits},
                )
                # LLM 自报的可计算性，供 orchestrator 做前置过滤
                spec.computable = bool(item.get("computable", True))
                spec.required_fields = list(item.get("required_fields", []) or [])
                spec.provenance = provenance
                # 只读到摘要时，LLM 常把自己定的窗口标成 "paper"。
                # 摘要里几乎不会写这些数字，所以统一降级为 default，
                # 宁可少标一个「来自论文」，也不能把编的说成读的。
                srcs = item.get("param_sources") or {}
                if provenance != "fulltext":
                    srcs = {k: "default" for k in (item.get("parameters") or {})}
                spec.param_sources = {
                    k: (v if v in ("paper", "inferred", "default") else "default")
                    for k, v in srcs.items()}
                # 市场迁移声明。检查器据此豁免文本扫描：迁移版因子在描述
                # 里提到原文的美股数据是正常的（「论文用 VIX，本版用已实现
                # 波动率」），不代表本因子需要那些数据。
                mig = item.get("migration") or {}
                if isinstance(mig, dict):
                    spec.migration = {
                        "source_market": str(mig.get("source_market") or ""),
                        "changes": [str(c) for c in
                                    (mig.get("changes") or [])][:8],
                    }
                specs.append(spec)
            except Exception as e:
                print(f"[FactorExtract] FactorSpec 构建失败: {e}")
        print(f"[FactorExtract] 提取到 {len(specs)} 个因子假设")
        return specs
