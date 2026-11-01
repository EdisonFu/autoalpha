"""
Crossref：期刊论文 + SSRN 工作论文。

SSRN 官网在本环境不可达（papers.ssrn.com 超时），但 SSRN 的 DOI 前缀是
10.2139，通过 Crossref 可完整检索 —— 实测 "momentum" 命中 1,591 篇，
内容是实务向的工作论文（《Refining ETF Asset Momentum Strategy》
《How to Improve ETF Sector Momentum》），正是研报类需求的落点。
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Dict, List

from .base import PaperSource

API = "https://api.crossref.org/works"
SSRN_PREFIX = "10.2139"


class CrossrefSource(PaperSource):
    name = "crossref"

    def __init__(self, include_ssrn: bool = True):
        self.include_ssrn = include_ssrn

    def search(self, intent, limit: int = 20) -> List[Dict]:
        out: List[Dict] = []
        terms = intent.query_terms(limit=2)
        want_practice = getattr(intent, "wants_reports", True)

        for term in terms:
            out.extend(self._query(term, limit // 2))
            # 偏实务时单独走一次 SSRN 前缀过滤，确保工作论文有配额
            if self.include_ssrn and want_practice:
                out.extend(self._query(term, max(4, limit // 3),
                                       extra_filter=f"prefix:{SSRN_PREFIX}"))
        return out

    def _query(self, term: str, n: int, extra_filter: str = "") -> List[Dict]:
        params = {
            "query": term,
            "rows": max(1, min(n, 20)),
            "select": "title,abstract,issued,DOI,container-title,publisher,"
                      "type,is-referenced-by-count,URL",
        }
        if extra_filter:
            params["filter"] = extra_filter
        url = f"{API}?{urllib.parse.urlencode(params)}"
        data = json.loads(self._get(url).decode("utf-8", "replace"))

        res = []
        for it in data.get("message", {}).get("items", []):
            doi = (it.get("DOI") or "").lower()
            res.append(self._mk(
                title=(it.get("title") or [""])[0],
                abstract=_clean_jats(it.get("abstract") or ""),
                url=it.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                date=_issued_date(it.get("issued")),
                source=self.name,
                venue=(it.get("container-title") or [""])[0]
                      or (it.get("publisher") or ""),
                doc_type=_map_type(it.get("type", ""), doi),
                citations=it.get("is-referenced-by-count") or 0,
                doi=doi,
            ))
        return res


def _map_type(t: str, doi: str) -> str:
    if doi.startswith(SSRN_PREFIX):
        return "working_paper"      # SSRN：实务向工作论文
    return {
        "journal-article": "paper",
        "posted-content": "preprint",
        "report": "report",
        "book-chapter": "book",
        "proceedings-article": "paper",
    }.get(t, "paper")


def _issued_date(issued: Dict) -> str:
    parts = (issued or {}).get("date-parts") or [[]]
    p = parts[0] if parts else []
    if not p or p[0] is None:
        return ""
    y = p[0]
    m = p[1] if len(p) > 1 and p[1] else 1
    d = p[2] if len(p) > 2 and p[2] else 1
    return f"{y:04d}-{m:02d}-{d:02d}"


def _clean_jats(s: str) -> str:
    """Crossref 摘要常带 JATS XML 标签。"""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:1200]
