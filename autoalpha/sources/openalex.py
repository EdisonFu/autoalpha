"""
OpenAlex：全类型学术检索，是研报的主要来源。

实测（查询 "momentum factor investing"）：
    report 10,555 / preprint 8,980 / article 112,451 / dissertation 42,687
研报实际来自 NBER、Econstor 等机构，内容对口（《Factor Momentum and the
Momentum Factor》《Momentum, Reversals, and Investor Clientele》）。
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Dict, List

from .base import PaperSource

API = "https://api.openalex.org/works"


class OpenAlexSource(PaperSource):
    name = "openalex"

    def __init__(self, types: tuple = ("article", "preprint", "report")):
        self.types = types

    def search(self, intent, limit: int = 20) -> List[Dict]:
        out: List[Dict] = []
        terms = intent.query_terms(limit=2)
        # 想要实务内容时优先捞研报；纯学术时研报配额给少些
        want_report = getattr(intent, "wants_reports", True)
        per_type = {
            "article": max(4, limit // 2),
            "preprint": max(3, limit // 4),
            "report": max(4, limit // 3) if want_report else 2,
        }
        for t in self.types:
            for term in terms:
                out.extend(self._query(term, t, per_type.get(t, 5)))
                if len(out) >= limit * 2:
                    return out[:limit * 2]
        return out

    def _query(self, term: str, doc_type: str, n: int) -> List[Dict]:
        q = urllib.parse.urlencode({
            "search": term,
            "filter": f"type:{doc_type}",
            "per-page": max(1, min(n, 25)),
            "select": "title,publication_year,publication_date,doi,type,"
                      "cited_by_count,primary_location,abstract_inverted_index",
        })
        data = json.loads(self._get(f"{API}?{q}").decode("utf-8", "replace"))
        res = []
        for w in data.get("results", []):
            loc = (w.get("primary_location") or {}).get("source") or {}
            url = ((w.get("primary_location") or {}).get("landing_page_url")
                   or (f"https://doi.org/{w['doi'].replace('https://doi.org/', '')}"
                       if w.get("doi") else ""))
            res.append(self._mk(
                title=w.get("title") or "",
                abstract=_rebuild_abstract(w.get("abstract_inverted_index")),
                url=url,
                date=w.get("publication_date")
                     or (f"{w['publication_year']}-01-01"
                         if w.get("publication_year") else ""),
                source=self.name,
                venue=loc.get("display_name") or "",
                doc_type=_map_type(w.get("type") or doc_type),
                citations=w.get("cited_by_count") or 0,
                doi=(w.get("doi") or "").replace("https://doi.org/", ""),
            ))
        return res


def _map_type(t: str) -> str:
    return {"report": "report", "preprint": "preprint",
            "dissertation": "thesis", "book-chapter": "book"}.get(t, "paper")


def _rebuild_abstract(inv: Dict) -> str:
    """
    OpenAlex 出于版权把摘要存成倒排索引 {词: [位置,...]}，需还原成文本。
    """
    if not inv or not isinstance(inv, dict):
        return ""
    slots: List = []
    for word, positions in inv.items():
        for p in positions:
            if isinstance(p, int):
                while len(slots) <= p:
                    slots.append("")
                slots[p] = word
    return " ".join(w for w in slots if w)[:1200]
