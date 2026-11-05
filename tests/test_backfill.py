"""
摘要回填与非研究条目过滤的回归测试。

断言针对实测到的真实问题：
  - 回填必须就地改写 papers，并标记 abstract_backfilled 供界面区分；
  - 阈值要与下游判「缺摘要」的口径一致，否则会出现「回填成功」
    但打分阶段仍当它是空的；
  - 过滤要认得 OECD 图表（`Figure 7.18. ...`）和书籍章节这类真实噪声，
    但不能误伤标题里含 "table"/"figure" 的正经论文。
"""

import pytest

from autoalpha.sources.backfill import (
    MIN_ABSTRACT, backfill_abstracts, drop_non_research, is_research_item,
    _norm_doi,
)


class TestNormDoi:
    def test_strips_prefix_and_cases(self):
        assert _norm_doi("https://doi.org/10.1016/J.X") == "10.1016/j.x"
        assert _norm_doi("  10.2139/SSRN.123  ") == "10.2139/ssrn.123"
        assert _norm_doi("") == ""
        assert _norm_doi(None) == ""


class TestIsResearchItem:
    def test_drops_oecd_figures(self):
        # 实测召回里混进来的真实样本
        for t in ["Figure 7.18. Global exchange-traded fund assets",
                  "Figure 3.13. Synthetic exchange traded Fund (ETF) example",
                  "Table 2.1 Summary statistics",
                  "Box 1. Policy considerations"]:
            assert not is_research_item({"title": t, "doc_type": "paper"}), t

    def test_drops_books(self):
        assert not is_research_item(
            {"title": "Exchange-traded notes", "doc_type": "book"})

    def test_keeps_real_papers(self):
        for t in ["How to Improve ETF Sector Momentum",
                  "Refining ETF Asset Momentum Strategy",
                  "Cross-Sectional Momentum in Cryptocurrency",
                  # 标题里含 figure/table 但不是图表题注，不能误伤
                  "Figures of Merit in Portfolio Construction",
                  "Roundtable Discussion on Factor Investing"]:
            assert is_research_item({"title": t, "doc_type": "paper"}), t

    def test_keeps_working_papers_and_reports(self):
        for dt in ("working_paper", "report", "preprint", "paper"):
            assert is_research_item({"title": "ETF Momentum", "doc_type": dt})

    def test_empty_title_dropped(self):
        assert not is_research_item({"title": "", "doc_type": "paper"})
        assert not is_research_item({"title": "   ", "doc_type": "paper"})


class TestDropNonResearch:
    def test_filters_and_preserves_order(self):
        ps = [{"title": "Momentum A", "doc_type": "paper"},
              {"title": "Figure 1. Chart", "doc_type": "paper"},
              {"title": "Momentum B", "doc_type": "working_paper"}]
        out = drop_non_research(ps, verbose=False)
        assert [p["title"] for p in out] == ["Momentum A", "Momentum B"]

    def test_empty(self):
        assert drop_non_research([], verbose=False) == []


class TestBackfill:
    def test_no_doi_is_skipped(self, monkeypatch):
        import autoalpha.sources.backfill as bf
        called = []
        monkeypatch.setattr(bf, "_from_openalex",
                            lambda d: called.append(d) or {})
        ps = [{"title": "x", "abstract": "", "doi": ""}]
        assert backfill_abstracts(ps, verbose=False) == 0
        assert not called          # 没有 DOI 就不该发请求

    def test_papers_with_abstracts_untouched(self, monkeypatch):
        import autoalpha.sources.backfill as bf
        monkeypatch.setattr(bf, "_from_openalex", lambda d: {})
        monkeypatch.setattr(bf, "_from_crossref", lambda d: {})
        full = "a" * (MIN_ABSTRACT + 50)
        ps = [{"title": "x", "abstract": full, "doi": "10.1/x"}]
        assert backfill_abstracts(ps, verbose=False) == 0
        assert ps[0]["abstract"] == full

    def test_fills_in_place_and_marks(self, monkeypatch):
        import autoalpha.sources.backfill as bf
        text = "b" * (MIN_ABSTRACT + 100)
        monkeypatch.setattr(bf, "_from_openalex", lambda d: {"10.1/x": text})
        monkeypatch.setattr(bf, "_from_crossref", lambda d: {})
        ps = [{"title": "x", "abstract": "", "doi": "https://doi.org/10.1/X"}]
        assert backfill_abstracts(ps, verbose=False) == 1
        assert ps[0]["abstract"] == text
        assert ps[0]["abstract_backfilled"] is True

    def test_crossref_only_queried_for_remainder(self, monkeypatch):
        """两家零重叠，第二家只该收到第一家没覆盖的 DOI。"""
        import autoalpha.sources.backfill as bf
        text = "c" * (MIN_ABSTRACT + 10)
        seen = {}

        def fake_oa(dois):
            seen["oa"] = list(dois)
            return {"10.1/a": text}

        def fake_cr(dois):
            seen["cr"] = list(dois)
            return {}

        monkeypatch.setattr(bf, "_from_openalex", fake_oa)
        monkeypatch.setattr(bf, "_from_crossref", fake_cr)
        ps = [{"title": "a", "abstract": "", "doi": "10.1/a"},
              {"title": "b", "abstract": "", "doi": "10.1/b"}]
        backfill_abstracts(ps, verbose=False)
        assert set(seen["oa"]) == {"10.1/a", "10.1/b"}
        assert seen["cr"] == ["10.1/b"]      # a 已被 OpenAlex 补上

    def test_short_result_rejected(self, monkeypatch):
        """回填来的摘要若仍短于阈值，不能算补上 —— 下游口径要一致。"""
        import autoalpha.sources.backfill as bf
        monkeypatch.setattr(bf, "_from_openalex", lambda d: {})
        monkeypatch.setattr(bf, "_from_crossref", lambda d: {})
        ps = [{"title": "x", "abstract": "too short", "doi": "10.1/x"}]
        assert backfill_abstracts(ps, verbose=False) == 0

    def test_same_doi_multiple_entries(self, monkeypatch):
        import autoalpha.sources.backfill as bf
        text = "d" * (MIN_ABSTRACT + 10)
        monkeypatch.setattr(bf, "_from_openalex", lambda d: {"10.1/x": text})
        monkeypatch.setattr(bf, "_from_crossref", lambda d: {})
        ps = [{"title": "x1", "abstract": "", "doi": "10.1/x"},
              {"title": "x2", "abstract": "", "doi": "10.1/X"}]
        assert backfill_abstracts(ps, verbose=False) == 2

    def test_network_failure_does_not_abort(self, monkeypatch):
        """
        一个批次网络失败时，整体不应抛 —— 与 _safe_search 同样的约定：
        单源故障降级为「少补几条」，不能让整轮检索失败。
        """
        import autoalpha.sources.backfill as bf

        def boom(url):
            raise OSError("network down")
        # 打在真实的 HTTP 层上，走 _from_openalex/_from_crossref 自己的
        # 异常处理，而不是替换掉整个函数（那样测的是桩，不是代码）
        monkeypatch.setattr(bf._http, "_get", boom)
        ps = [{"title": "x", "abstract": "", "doi": "10.1/x"}]
        assert backfill_abstracts(ps, verbose=False) == 0
        assert ps[0]["abstract"] == ""       # 没补上，但也没炸
