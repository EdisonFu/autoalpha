"""
Step 1：意图识别。

把用户的自然语言输入归一成结构化检索意图，供 Step 2 的多源检索使用。

为什么需要这一步（实测依据）：
    "ETF轮动"                                       →     18 篇
    "ETF rotation momentum"                         →    811 篇
    "cross-sectional momentum exchange-traded fund" → 14,494 篇

同一个意图，不同表述的召回量差 800 倍。用户不会用学术术语提问，
而检索源只认学术术语 —— 这个落差就是这一步要填的。

它还决定去哪些源找：想要实务经验就该去 SSRN 工作论文和研报，
想要方法论就该去 arXiv。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from ..llm import LLMClient
from ..json_repair import loads_lenient, JSONRepairError

_SYSTEM = """\
You are a quantitative finance research librarian. Turn a user's research
direction (often informal, often Chinese) into a structured search intent.

Output ONLY this JSON object:
{
  "canonical_en": "the single best English academic phrasing of this topic",
  "aliases": ["3-5 alternative phrasings used in the literature"],
  "market": "china_a|hk|us|japan|europe|global|unknown",
  "asset_class": "equity|etf|bond|commodity|crypto|fx|multi|unknown",
  "factor_family": "momentum|reversal|volatility|value|quality|size|liquidity|ml|macro|other",
  "horizon": "intraday|daily|weekly|monthly|quarterly|unknown",
  "doc_preference": "academic|practitioner|both",
  "restated_zh": "一句中文，复述你理解的研究目标",
  "confidence": 0.0-1.0
}

Rules:
- canonical_en must use the terminology the literature actually uses.
  "ETF轮动" is "cross-sectional momentum in exchange-traded funds",
  not a literal translation.
- MARKET MATTERS. Markets differ in microstructure, retail share, price
  limits, short-selling rules and settlement, so a finding in one market
  often does not carry to another. Infer the market from wording
  (A股/沪深/中证/创业板 → china_a; 港股/恒生 → hk; 美股/标普/纳指 → us),
  from ticker shape, or from the user naming it outright.
  When the market is identified, say so in canonical_en — e.g.
  "cross-sectional momentum in Chinese A-share ETFs" — and add at least
  one alias naming the market, so retrieval favours studies on it.
  Use "unknown" only when nothing in the input suggests a market.
- aliases should be genuinely different phrasings (synonyms, related
  methodology names), not trivial word reorderings. They widen recall.
- doc_preference: choose "practitioner" when the user signals wanting
  real-world/industry work (实务, 业界, 研报, 能落地, 实盘);
  "academic" for method/theory questions; "both" when unclear.
- confidence below 0.5 if the input is too vague to pin down a topic.
- restated_zh lets the user catch a misread before any search runs.
Output ONLY the JSON object, no explanation.
"""


# 市场 → 检索时附加的限定词。不同市场的制度与微观结构差异大，
# 在一个市场成立的结论常常不能平移到另一个。
MARKET_TERMS = {
    "china_a": ["Chinese A-share", "China stock market", "Shanghai Shenzhen"],
    "hk":      ["Hong Kong stock market", "Hang Seng"],
    "us":      ["US equity market", "S&P 500"],
    "japan":   ["Japanese stock market", "TOPIX"],
    "europe":  ["European equity market", "STOXX"],
    "global":  ["international markets", "cross-country"],
}

MARKET_LABEL = {
    "china_a": "A股", "hk": "港股", "us": "美股",
    "japan": "日股", "europe": "欧股", "global": "全球",
    "unknown": "未指定",
}


@dataclass
class SearchIntent:
    canonical_en: str = ""
    aliases: List[str] = field(default_factory=list)
    market: str = "unknown"
    asset_class: str = "unknown"
    factor_family: str = "other"
    horizon: str = "unknown"
    doc_preference: str = "both"
    restated_zh: str = ""
    confidence: float = 0.0
    raw_input: str = ""

    def query_terms(self, limit: int = 4) -> List[str]:
        """
        供各检索源使用的查询词，规范表述优先。

        已识别市场时补一条带市场限定的查询 —— 否则「ETF 动量」会把
        美股研究排到前面，而数据是 A 股 ETF，导出的因子未必适用。
        """
        terms = [self.canonical_en] if self.canonical_en else []
        for a in self.aliases:
            if a and a.lower() not in {t.lower() for t in terms}:
                terms.append(a)

        mt = MARKET_TERMS.get(self.market)
        if mt and self.canonical_en:
            # 市场词要【领头且短】。实测 Crossref 在长查询上排序很差：
            #   "cross-sectional momentum in exchange-traded funds Chinese A-share"
            #     → Exchange-traded notes / Sustainable Investing …
            #   "Chinese A-share ETF momentum"
            #     → ETF Momentum / How to Improve ETF Sector Momentum
            # 所以不是把市场词缀在长句尾部，而是单独组一条短查询。
            if mt[0].lower() not in self.canonical_en.lower():
                terms.insert(1, f"{mt[0]} {self._core_terms()}".strip())
        return terms[:limit] or [self.raw_input]

    _STOP = {"in", "of", "the", "a", "an", "for", "with", "and", "or", "on",
             "to", "using", "based", "via", "from", "by", "its", "their"}

    def _core_terms(self, n: int = 3) -> str:
        """从规范表述里取几个实词，用于拼短查询。"""
        words = [w for w in self.canonical_en.replace("-", " ").split()
                 if w.lower() not in self._STOP]
        return " ".join(words[:n])

    @property
    def market_label(self) -> str:
        return MARKET_LABEL.get(self.market, self.market)

    @property
    def wants_reports(self) -> bool:
        return self.doc_preference in ("practitioner", "both")

    def to_dict(self) -> dict:
        return asdict(self)


class IntentAgent:
    """单次 LLM 调用完成意图归一。失败时降级为原文直查，不阻断流程。"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._cache: dict = {}

    def recognize(self, raw_input: str) -> SearchIntent:
        key = raw_input.strip()
        if key in self._cache:
            return self._cache[key]

        intent = self._call(key)
        self._cache[key] = intent
        return intent

    def _call(self, raw: str) -> SearchIntent:
        fallback = SearchIntent(canonical_en=raw, raw_input=raw,
                                restated_zh=raw, confidence=0.3)
        try:
            out = self.llm.complete(system=_SYSTEM,
                                    user=f"Research direction: {raw}",
                                    temperature=0.2, purpose="intent")
        except Exception as e:
            print(f"[Intent] LLM 调用失败，降级为原文直查: {e}")
            return fallback

        try:
            d = loads_lenient(out, expect=dict)
        except JSONRepairError as e:
            print(f"[Intent] 解析失败，降级为原文直查: {str(e)[:160]}")
            return fallback

        aliases = d.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [str(aliases)]

        try:
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0

        return SearchIntent(
            canonical_en=str(d.get("canonical_en") or raw).strip(),
            aliases=[str(a).strip() for a in aliases if str(a).strip()][:6],
            market=str(d.get("market") or "unknown"),
            asset_class=str(d.get("asset_class") or "unknown"),
            factor_family=str(d.get("factor_family") or "other"),
            horizon=str(d.get("horizon") or "unknown"),
            doc_preference=str(d.get("doc_preference") or "both"),
            restated_zh=str(d.get("restated_zh") or "").strip(),
            confidence=max(0.0, min(1.0, conf)),
            raw_input=raw,
        )
