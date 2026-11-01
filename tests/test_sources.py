"""
多源检索与意图识别测试。不打网络：用构造数据验证纯逻辑。
"""

import pytest

from autoalpha.agents.intent_agent import SearchIntent
from autoalpha.sources.base import dedupe, _richness
from autoalpha.sources.arxiv import ArxivSource
from autoalpha.sources.openalex import _rebuild_abstract, _map_type
from autoalpha.sources.crossref import _clean_jats, _issued_date
from autoalpha.sources.crossref import _map_type as cr_type


# ── 意图 ────────────────────────────────────────────────────────────────────

def test_query_terms_prefer_canonical():
    i = SearchIntent(canonical_en="cross-sectional momentum in ETFs",
                     aliases=["ETF rotation", "cross-sectional momentum in ETFs"])
    t = i.query_terms()
    assert t[0] == "cross-sectional momentum in ETFs"
    assert len(t) == 2, "重复表述应去掉"


def test_query_terms_fall_back_to_raw():
    i = SearchIntent(raw_input="ETF轮动")
    assert i.query_terms() == ["ETF轮动"]


def test_wants_reports_follows_preference():
    assert SearchIntent(doc_preference="practitioner").wants_reports
    assert SearchIntent(doc_preference="both").wants_reports
    assert not SearchIntent(doc_preference="academic").wants_reports


# ── arXiv 短语拆分 ─────────────────────────────────────────────────────────

def test_arxiv_splits_long_canonical_phrase():
    """
    canonical_en 是完整学术句，arXiv 当精确短语匹配会 0 命中（实测）。
    必须拆成 2 词短语。
    """
    i = SearchIntent(canonical_en="cross-sectional momentum in exchange-traded funds")
    ph = ArxivSource()._phrases(i)
    assert all(len(p.split()) <= 3 for p in ph), f"短语过长: {ph}"
    assert len(ph) >= 2
    assert not any(p.lower().startswith("in ") for p in ph), "停用词不应领头"


def test_arxiv_keeps_short_terms_intact():
    i = SearchIntent(canonical_en="ETF rotation", aliases=["factor momentum"])
    ph = ArxivSource()._phrases(i)
    assert "ETF rotation" in ph and "factor momentum" in ph


def test_arxiv_query_has_category_filter():
    i = SearchIntent(canonical_en="ETF momentum")
    q = ArxivSource()._build_query(i)
    assert "cat:q-fin" in q and "ti:" in q and "abs:" in q


# ── OpenAlex ───────────────────────────────────────────────────────────────

def test_rebuild_abstract_from_inverted_index():
    """OpenAlex 出于版权把摘要存成倒排索引。"""
    inv = {"Momentum": [0], "is": [1], "an": [2], "anomaly": [3]}
    assert _rebuild_abstract(inv) == "Momentum is an anomaly"
    assert _rebuild_abstract(None) == ""
    assert _rebuild_abstract({}) == ""


def test_openalex_type_mapping():
    assert _map_type("report") == "report"
    assert _map_type("dissertation") == "thesis"
    assert _map_type("journal-article") == "paper"


# ── Crossref ───────────────────────────────────────────────────────────────

def test_ssrn_doi_prefix_marks_working_paper():
    """SSRN 官网不可达，但其 DOI 前缀 10.2139 能在 Crossref 检索到。"""
    assert cr_type("posted-content", "10.2139/ssrn.123456") == "working_paper"
    assert cr_type("journal-article", "10.1016/j.jfe.2020.01") == "paper"


def test_clean_jats_strips_xml():
    raw = "<jats:p>Price <jats:italic>momentum</jats:italic> is real.</jats:p>"
    assert _clean_jats(raw) == "Price momentum is real."


def test_issued_date_handles_partial_dates():
    assert _issued_date({"date-parts": [[2020, 5, 12]]}) == "2020-05-12"
    assert _issued_date({"date-parts": [[2020]]}) == "2020-01-01"
    assert _issued_date({}) == ""
    assert _issued_date({"date-parts": [[None]]}) == ""


# ── 跨源去重 ───────────────────────────────────────────────────────────────

def _p(title, source, doi="", abstract="", venue="", cites=0):
    return {"title": title, "source": source, "doi": doi,
            "abstract": abstract, "venue": venue, "citations": cites,
            "doc_type": "paper", "date": "2020-01-01", "pdf_url": ""}


def test_dedupe_by_doi():
    out = dedupe([_p("A", "arxiv", "10.1/x"), _p("A different title", "crossref", "10.1/x")])
    assert len(out) == 1


def test_dedupe_by_normalized_title():
    """同一篇在 arXiv 与期刊的标题大小写/标点常有差异。"""
    out = dedupe([_p("ETF Momentum: A Study", "arxiv"),
                  _p("etf momentum - a study", "openalex")])
    assert len(out) == 1


def test_dedupe_keeps_richer_record():
    poor = _p("X", "arxiv")
    rich = _p("X", "crossref", doi="10.1/x", abstract="a" * 400, venue="JF", cites=50)
    out = dedupe([poor, rich])
    assert len(out) == 1 and out[0]["source"] == "crossref"
    assert _richness(rich) > _richness(poor)


def test_dedupe_records_other_sources():
    """同一篇多源命中时，应记录它还出现在哪，供界面展示。"""
    out = dedupe([_p("X", "crossref", doi="10.1/x", abstract="a" * 400),
                  _p("X", "arxiv")])
    assert len(out) == 1
    assert "arxiv" in out[0].get("also_in", [])


def test_dedupe_preserves_order_and_drops_empty_titles():
    out = dedupe([_p("A", "arxiv"), _p("B", "arxiv"), _p("", "arxiv")])
    assert [p["title"] for p in out] == ["A", "B"]
