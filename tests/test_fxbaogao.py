"""发现报告中文研报源。"""

import pytest

from autoalpha.sources.fxbaogao import (
    FxBaoGaoSource, _clean_paragraphs, _org_from_title, _strip_em,
)


class _Intent:
    def __init__(self, raw="", family="other", asset="unknown"):
        self.raw_input = raw
        self.restated_zh = ""
        self.factor_family = family
        self.asset_class = asset


class TestCleaning:
    def test_strips_highlight_tags(self):
        """接口给关键词加 <em> 高亮，留着会进标题与摘要。"""
        assert _strip_em("基于<em>多因子</em>的指数增强") == "基于多因子的指数增强"
        assert _strip_em(None) == ""

    def test_paragraphs_joined(self):
        objs = [{"content": "第一段"}, {"content": "第二段"}, {"content": ""}]
        out = _clean_paragraphs(objs)
        assert "第一段" in out and "第二段" in out

    def test_paragraphs_capped(self):
        out = _clean_paragraphs([{"content": "字" * 3000}])
        assert len(out) <= 1500

    def test_empty_paragraphs(self):
        assert _clean_paragraphs(None) == ""
        assert _clean_paragraphs([]) == ""


class TestOrg:
    def test_extracts_broker_from_title(self):
        assert "申万宏源" in _org_from_title("申万宏源金工：因子分域研究", "")
        assert "广发" in _org_from_title("广发金工机器学习选股手册", "")

    def test_unknown_falls_back(self):
        assert _org_from_title("随便一个标题", "某机构") == "某机构"
        assert _org_from_title("随便一个标题", "") == "券商研报"

    def test_broker_already_known_is_kept(self):
        out = _org_from_title("某策略", "华泰证券")
        assert out == "华泰证券"


class TestSearch:
    def test_no_chinese_keywords_skips_api(self, monkeypatch):
        """没有中文词就不该发请求 —— 否则返回的是无关研报。"""
        called = []
        monkeypatch.setattr(FxBaoGaoSource, "_query",
                            lambda self, k, n: called.append(k) or [])
        assert FxBaoGaoSource().search(_Intent()) == []
        assert not called

    def test_queries_use_chinese_terms(self, monkeypatch):
        seen = []
        monkeypatch.setattr(FxBaoGaoSource, "_query",
                            lambda self, k, n: seen.append(k) or [])
        FxBaoGaoSource().search(_Intent("指数增强策略", "momentum", "etf"))
        assert any("指数增强" in k for k in seen)

    def test_network_failure_is_caught(self, monkeypatch):
        """单源失败不应拖垮整体检索（_safe_search 的约定）。"""
        def boom(*a, **kw):
            raise OSError("timeout")
        monkeypatch.setattr("urllib.request.urlopen", boom)
        src = FxBaoGaoSource()
        out = src._safe_search(_Intent("动量", "momentum"), 5)
        assert out == []
