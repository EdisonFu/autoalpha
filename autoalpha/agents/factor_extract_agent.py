import json
import re
from typing import List
from ..llm import LLMClient
from ..pipeline_state import FactorSpec
from ..utils import strip_code_fence

_EXTRACT_SYSTEM = """\
You are a quantitative finance researcher. Given a paper title and abstract, extract 2-4 concrete, implementable factor hypotheses.

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
  "computable": true,
  "required_fields": ["close", "volume"]
}

Rules:
- Only extract factors that are EXPLICITLY described or implied in the abstract
- Translate Chinese abstracts to English terminology
- If the paper describes a strategy (not a factor), extract the underlying signal as a factor
- EVERY factor must be computable from the available data listed in the user message.
  Papers often use data we do not have (US tickers, VIX, news, fundamentals).
  When that happens, either restate the signal using only available fields, or
  set "computable": false and explain in implementation_notes. Never invent a proxy
  and present it as the paper's signal.
- Add a "computable" boolean and a "required_fields" list to each factor.
- Output ONLY valid JSON array, no explanation
"""


class FactorExtractAgent:
    """从论文摘要中提取因子假设列表"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, paper: dict, data_context: str = "") -> List[FactorSpec]:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        url = paper.get("pdf_url", "")
        date = paper.get("date", "")

        print(f"[FactorExtract] 解析论文: {title[:60]}")

        user_msg = f"Paper title: {title}\n\nAbstract:\n{abstract}"
        if data_context:
            user_msg = f"{data_context}\n\n{user_msg}"
        try:
            raw = strip_code_fence(self.llm.complete(system=_EXTRACT_SYSTEM, user=user_msg,
                                                 purpose="extract"))
            items = json.loads(raw)
            if not isinstance(items, list):
                items = [items]
        except Exception as e:
            print(f"[FactorExtract] 解析失败: {e}")
            return []

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
                    source_paper={"title": title, "url": url, "date": date, "abstract": abstract[:300]},
                )
                # LLM 自报的可计算性，供 orchestrator 做前置过滤
                spec.computable = bool(item.get("computable", True))
                spec.required_fields = list(item.get("required_fields", []) or [])
                specs.append(spec)
            except Exception as e:
                print(f"[FactorExtract] FactorSpec 构建失败: {e}")
        print(f"[FactorExtract] 提取到 {len(specs)} 个因子假设")
        return specs
