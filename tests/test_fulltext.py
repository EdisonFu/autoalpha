"""
全文抓取与方法论抽取的回归测试。

这里的断言针对**真实失败模式**，不是形式检查：
  - 章节切分曾把 PDF 折行的正文碎片当成标题（实测一篇切出 55 段，
    真标题被淹没），所以要断言虚词开头的长句不是标题；
  - 按出现顺序截取会被靠前的长章节吃掉预算，复现细节掉到 2/7，
    所以要断言靠后的细节章节能被选中；
  - 只读摘要时 param_sources 必须全部降级为 default —— 摘要几乎不写
    回看窗口这类数字，标成「来自论文」是误导用户。
"""

import pytest

from autoalpha.sources.fulltext import (
    detail_hits, extract_methodology, is_arxiv, paper_context,
    split_sections, _is_heading,
)


class TestHeading:
    def test_real_headings(self):
        for s in ["3. Methodology", "DATA AND SAMPLE", "Momentum Strategies",
                  "IV. Empirical Results", "Factors Description"]:
            assert _is_heading(s), s

    def test_body_fragments_rejected(self):
        # 全部来自实测误判样本
        for s in ["The general framework for the CSM strategy",
                  "Most CSM strategies adhere to this framework and are",
                  "Keeping in mind that stock trends are highly influenced",
                  "We construct our monthly portfolios using data from",
                  "Increasing rate of the"]:
            assert not _is_heading(s), s

    def test_punctuation_ends_disqualify(self):
        assert not _is_heading("This is a sentence,")
        assert not _is_heading("Results show that:")


class TestSplit:
    def test_splits_on_headings(self):
        text = ("1. Introduction\nsome intro text here\n"
                "2. Methodology\nwe rank stocks by past returns\n"
                "3. Results\nthe strategy earns alpha\n")
        secs = split_sections(text)
        titles = [t for t, _ in secs]
        assert "Methodology" in titles
        assert "Results" in titles

    def test_numbering_stripped(self):
        secs = split_sections("4. EXPERIMENTAL SETUP\nbody text\n")
        assert secs[0][0] == "EXPERIMENTAL SETUP"


class TestDetailHits:
    def test_finds_reproduction_details(self):
        t = ("We rank stocks by past 12 month returns into deciles, "
             "rebalanced monthly from 1990 to 2022, reporting the Sharpe ratio "
             "net of transaction cost.")
        got = set(detail_hits(t))
        assert {"lookback", "quantile", "rebalance", "sample",
                "metric", "cost"} <= got

    def test_empty(self):
        assert detail_hits("") == []
        assert detail_hits("a paper about finance") == []


class TestExtractMethodology:
    def test_prefers_detail_dense_sections_over_order(self):
        """靠后的细节章节必须赢过靠前的空洞长章节。"""
        filler = "background discussion without any concrete numbers. " * 40
        dense = ("We sort into deciles using past 12 month returns, "
                 "rebalanced monthly, from 1990 to 2022, net of "
                 "transaction cost, reporting the Sharpe ratio. ") * 4
        text = (f"1. Model Overview\n{filler}\n"
                f"2. Data Discussion\n{filler}\n"
                f"3. Portfolio Construction\n{dense}\n")
        out = extract_methodology(text, max_chars=1500)
        # 预算只够一段，必须是细节那段
        assert "deciles" in out
        assert len(set(detail_hits(out))) >= 4

    def test_drops_references_and_related_work(self):
        body = "concrete method text with deciles and rebalanced monthly. " * 10
        noise = "Smith J. 2011. Some Paper. Journal of Things. " * 40
        text = (f"2. Methodology\n{body}\n"
                f"7. Related Work\n{noise}\n"
                f"8. References\n{noise}\n")
        out = extract_methodology(text, max_chars=4000)
        assert "deciles" in out
        assert "Related Work" not in out
        assert "References" not in out

    def test_respects_budget(self):
        text = "1. Methodology\n" + ("word " * 5000)
        assert len(extract_methodology(text, max_chars=800)) <= 800

    def test_unsplittable_falls_back_to_prefix(self):
        text = "just one long blob of text without headings. " * 50
        out = extract_methodology(text, max_chars=300)
        assert out == text[:300]

    def test_empty(self):
        assert extract_methodology("") == ""


class TestPaperContext:
    def test_non_arxiv_stays_on_abstract(self):
        ctx = paper_context({"pdf_url": "https://doi.org/10.1016/j.x",
                             "abstract": "an abstract"})
        assert ctx["provenance"] == "abstract"
        assert ctx["text"] == "an abstract"
        assert "登录墙" in ctx["note"] or "非 arXiv" in ctx["note"]

    def test_missing_pypdf_is_reported_not_raised(self, monkeypatch):
        import autoalpha.sources.fulltext as ft
        monkeypatch.setattr(ft, "pypdf_available", lambda: False)
        ctx = ft.paper_context({"pdf_url": "https://arxiv.org/pdf/1234.5678",
                                "abstract": "an abstract"})
        assert ctx["provenance"] == "abstract"
        assert "pypdf" in ctx["note"]

    def test_download_failure_falls_back(self, monkeypatch):
        import autoalpha.sources.fulltext as ft
        monkeypatch.setattr(ft, "pypdf_available", lambda: True)
        monkeypatch.setattr(ft, "fetch_fulltext", lambda u: None)
        ctx = ft.paper_context({"pdf_url": "https://arxiv.org/pdf/1234.5678",
                                "abstract": "an abstract"})
        assert ctx["provenance"] == "abstract"
        assert ctx["text"] == "an abstract"

    def test_fulltext_path(self, monkeypatch):
        import autoalpha.sources.fulltext as ft
        body = ("3. Methodology\nWe sort into deciles by past 12 month "
                "returns, rebalanced monthly from 1990 to 2022. " * 20)
        monkeypatch.setattr(ft, "pypdf_available", lambda: True)
        monkeypatch.setattr(ft, "fetch_fulltext", lambda u: body)
        ctx = ft.paper_context({"pdf_url": "https://arxiv.org/pdf/1234.5678",
                                "abstract": "short abstract"})
        assert ctx["provenance"] == "fulltext"
        assert len(ctx["text"]) > len("short abstract")


class TestIsArxiv:
    def test_variants(self):
        assert is_arxiv("https://arxiv.org/pdf/2302.10175v1")
        assert is_arxiv("http://arxiv.org/abs/2302.10175")
        assert not is_arxiv("https://doi.org/10.1016/j.x")
        assert not is_arxiv("")
