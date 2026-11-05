"""
arXiv：q-fin 预印本。沿用既有的宽召回 + 字段并联查询策略。
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List

from .base import PaperSource

API = "https://export.arxiv.org/api/query"
CATEGORIES = ["q-fin.PM", "q-fin.TR", "q-fin.ST", "q-fin.MF", "q-fin.CP"]
NS = {"a": "http://www.w3.org/2005/Atom"}


class ArxivSource(PaperSource):
    name = "arxiv"

    def search(self, intent, limit: int = 60) -> List[Dict]:
        query = self._build_query(intent)
        params = urllib.parse.urlencode({
            "search_query": query,
            "start": 0,
            "max_results": max(1, min(limit, 200)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        raw = self._get(f"{API}?{params}")
        return self._parse(raw)

    # 意图识别产出的 canonical_en 是完整学术表述（如 "cross-sectional
    # momentum in exchange-traded funds"）。arXiv 把带引号的长短语当精确
    # 匹配，整句几乎命中不了（实测 0 条），所以要先拆成可检索的词组。
    _STOP = {"in", "of", "the", "a", "an", "for", "with", "and", "or", "on",
             "to", "using", "based", "via", "from", "by", "its", "their"}

    def _phrases(self, intent) -> list:
        """把意图拆成 2-3 词的检索短语，长句降级为关键词组合。"""
        out, seen = [], set()
        for term in intent.query_terms(limit=4):
            words = [w for w in term.replace("-", " ").split() if w]
            # 短表述直接用；长句抽取非停用词再两两组合
            if len(words) <= 3:
                cand = [" ".join(words)]
            else:
                core = [w for w in words if w.lower() not in self._STOP]
                cand = [" ".join(core[i:i + 2]) for i in range(0, len(core), 2)]
                cand = [c for c in cand if len(c.split()) == 2] or [" ".join(core[:2])]
            for c in cand:
                k = c.lower()
                if c and k not in seen:
                    seen.add(k)
                    out.append(c)
        return out[:6]

    def _build_query(self, intent) -> str:
        terms = self._phrases(intent)
        def _q(t: str) -> str:
            t = t.strip()
            return f'"{t}"' if " " in t else t
        parts = [f"ti:{_q(t)}" for t in terms] + [f"abs:{_q(t)}" for t in terms]
        cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
        kw = " OR ".join(parts) if parts else "quantitative finance"
        return f"({kw}) AND ({cats})"

    def _parse(self, raw: bytes) -> List[Dict]:
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return []
        out = []
        for e in root.findall("a:entry", NS):
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            summary = " ".join((e.findtext("a:summary", "", NS) or "").split())
            pub = (e.findtext("a:published", "", NS) or "")[:10]
            url = e.findtext("a:id", "", NS) or ""
            pdf = url.replace("/abs/", "/pdf/") if "/abs/" in url else url
            # API 返回的是 http://arxiv.org/...，而明文 http 在部分网络下
            # 直接超时（实测 http 超时 / https 2s 拿到 1.4MB）。全文解析
            # 依赖这个链接，所以在入库前就升到 https。
            if pdf.startswith("http://"):
                pdf = "https://" + pdf[len("http://"):]
            out.append(self._mk(
                title=title, abstract=summary, url=pdf, date=pub,
                source=self.name, venue="arXiv", doc_type="preprint",
            ))
        return out
