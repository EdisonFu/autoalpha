"""
聚宽中文源。

守的主要是「不要返回噪声」：
  - 没有中文关键词时必须返回空，不能把整库当相关；
  - 列表接口的 keyword 参数被服务端忽略（实测），所以只能本地筛；
  - 摘要要剥掉 Markdown 图片/链接，否则检索与中文概述都被 URL 淹没。
"""

import pytest

from autoalpha.sources.joinquant import (
    JoinQuantSource, JoinQuantStore, _clean_md, normalize,
)
from autoalpha.sources.zh_terms import zh_keywords


class _Intent:
    def __init__(self, raw="", family="other", asset="unknown"):
        self.raw_input = raw
        self.restated_zh = ""
        self.factor_family = family
        self.asset_class = asset


class TestZhKeywords:
    def test_extracts_domain_terms(self):
        kws = zh_keywords(_Intent("A股ETF截面动量轮动", "momentum", "etf"))
        assert "动量" in kws
        assert "轮动" in kws

    def test_longer_terms_win(self):
        """「指数增强」不该被拆成「指数」——长词信息量更大。"""
        kws = zh_keywords(_Intent("指数增强策略研究"))
        assert "指数增强" in kws
        assert kws.index("指数增强") == 0

    def test_family_fills_missing(self):
        kws = zh_keywords(_Intent("", "volatility", "equity"))
        assert "低波动" in kws

    def test_empty_intent_gives_empty(self):
        assert zh_keywords(_Intent()) == []


class TestCleanMd:
    def test_strips_images(self):
        assert "http" not in _clean_md("![Img](https://x.com/a.png) 正文")

    def test_keeps_link_text(self):
        assert _clean_md("[策略说明](https://x.com)") == "策略说明"

    def test_strips_html_and_marks(self):
        out = _clean_md("## 标题\n**加粗** <div>x</div>")
        assert "<div>" not in out
        assert "##" not in out


class TestNormalize:
    def test_schema_matches_other_sources(self):
        raw = {"title": "T", "content": "正文", "postId": "p1",
               "uniqueKey": "u1", "addTime": "2026-01-01 10:00:00",
               "isBest": "1", "viewCount": "100", "collectionCount": "5"}
        p = normalize(raw)
        # 与 arXiv/Crossref 同一套键，才能进跨源去重与排序
        for k in ("title", "abstract", "pdf_url", "date", "source",
                  "doc_type", "citations", "doi"):
            assert k in p, k
        assert p["source"] == "joinquant"
        assert p["doc_type"] == "cn_post"
        assert p["is_best"] is True
        assert p["collections"] == 5
        assert p["date"] == "2026-01-01"
        assert "u1" in p["pdf_url"]

    def test_abstract_is_capped(self):
        p = normalize({"title": "T", "postId": "p", "uniqueKey": "u"},
                      content="字" * 9000)
        assert len(p["abstract"]) <= 2000


class TestSource:
    def test_no_keywords_returns_nothing(self, tmp_path):
        """没有中文关键词时返回空 —— 否则整库都会被当成相关。"""
        src = JoinQuantSource(db_path=str(tmp_path / "t.db"), auto_crawl=False)
        assert src.search(_Intent()) == []

    def test_empty_store_without_autocrawl(self, tmp_path):
        src = JoinQuantSource(db_path=str(tmp_path / "t.db"), auto_crawl=False)
        assert src.search(_Intent("动量", "momentum")) == []

    def test_searches_local_store(self, tmp_path):
        db = str(tmp_path / "t.db")
        store = JoinQuantStore(db)
        store.upsert([normalize(
            {"title": "ETF动量轮动策略", "postId": "p1", "uniqueKey": "u1",
             "collectionCount": "9"}, content="动量策略正文"),
            normalize({"title": "今天天气不错", "postId": "p2",
                       "uniqueKey": "u2"}, content="闲聊")])
        src = JoinQuantSource(db_path=db, auto_crawl=False)
        out = src.search(_Intent("动量", "momentum"), limit=10)
        assert len(out) == 1
        assert "动量" in out[0]["title"]

    def test_best_posts_rank_first(self, tmp_path):
        db = str(tmp_path / "t.db")
        store = JoinQuantStore(db)
        store.upsert([
            normalize({"title": "动量普通帖", "postId": "a", "uniqueKey": "a",
                       "isBest": "0", "collectionCount": "1"}, content="动量"),
            normalize({"title": "动量精华帖", "postId": "b", "uniqueKey": "b",
                       "isBest": "1", "collectionCount": "0"}, content="动量"),
        ])
        out = JoinQuantStore(db).search(["动量"])
        assert out[0]["title"] == "动量精华帖"


class TestPrescoreFairness:
    """
    截断不能按源顺序。

    实测 bug：dedupe 保留源顺序（arxiv→openalex→crossref→joinquant），
    而 papers[:60] 直接切，于是聚宽召回 25 条最终展示前 10 篇里
    一条中文都没有 —— 后接入的源被系统性歧视。
    """

    def test_chinese_posts_survive_truncation(self):
        from autoalpha.agents.paper_search_agent import PaperSearchAgent as P
        eng = [{"title": f"e{i}", "doc_type": "preprint",
                "pdf_url": "https://arxiv.org/pdf/x",
                "abstract": "deciles past 12 month", "citations": 3}
               for i in range(45)]
        cn = [{"title": f"c{i}", "doc_type": "cn_post",
               "abstract": "def initialize(context): import pandas",
               "is_best": True, "collections": 40} for i in range(25)]
        out = P._prescore_sort(eng + cn)
        kept = out[:60]
        n_cn = sum(1 for p in kept if p["doc_type"] == "cn_post")
        assert n_cn > 0, "中文帖不能在预排序后全军覆没"

    def test_low_quality_does_not_beat_detailed_paper(self):
        """预排序也不能让闲聊帖压过有复现细节的论文。"""
        from autoalpha.agents.paper_search_agent import PaperSearchAgent as P
        papers = [
            {"title": "闲聊", "doc_type": "cn_post", "abstract": "今天行情"},
            {"title": "实证", "doc_type": "preprint",
             "pdf_url": "https://arxiv.org/pdf/x",
             "abstract": "deciles past 12 month rebalanced monthly "
                         "from 1990 to 2022 Sharpe transaction cost"},
        ]
        out = P._prescore_sort(papers)
        assert out[0]["title"] == "实证"

    def test_collections_substitute_for_citations(self):
        """社区帖没有被引数，收藏数应充当同类的同行认可信号。"""
        from autoalpha.agents.paper_search_agent import PaperSearchAgent as P
        base = {"doc_type": "cn_post", "abstract": "def initialize(context):"}
        low = P._prescore_sort([dict(base, title="low", collections=0),
                                dict(base, title="high", collections=300)])
        assert low[0]["title"] == "high"
