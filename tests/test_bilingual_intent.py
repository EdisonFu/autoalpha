"""
双语检索式：canonical_zh 与 canonical_en 并行。

动机：中文源原本靠静态词典匹配（zh_terms.CN_TERMS）找关键词，
词典收不到的表述（「动量崩溃」「因子配置」「北向资金流」）会整块漏掉，
而这些恰恰可能是用户研究的方向。让意图识别直接产出中文检索式，
不需要额外 LLM 调用，且能覆盖任意新表述。
"""

import pytest

from autoalpha.agents.intent_agent import SearchIntent
from autoalpha.sources.zh_terms import zh_keywords


class TestCanonicalZh:
    def test_field_exists_and_defaults_empty(self):
        it = SearchIntent()
        assert it.canonical_zh == ""

    def test_out_of_vocabulary_terms_survive(self):
        """词典外的术语必须能检索 —— 这正是引入 canonical_zh 的原因。"""
        it = SearchIntent(canonical_zh="动量崩溃 因子有效性 因子配置",
                          factor_family="momentum")
        kws = zh_keywords(it)
        assert "动量崩溃" in kws
        assert "因子有效性" in kws

    def test_phrase_kept_for_phrase_capable_sources(self):
        """整串短语要保留，券商研报接口支持空格分隔的多词查询。"""
        it = SearchIntent(canonical_zh="ETF 截面动量 轮动")
        kws = zh_keywords(it)
        assert kws[0] == "ETF 截面动量 轮动", "整串应排最前，信息量最大"

    def test_dictionary_still_fills_when_absent(self):
        """老会话或 LLM 未产出 canonical_zh 时，词典匹配仍要生效。"""
        it = SearchIntent(raw_input="A股ETF截面动量轮动",
                          factor_family="momentum", asset_class="etf")
        kws = zh_keywords(it)
        assert "动量" in kws
        assert "轮动" in kws

    def test_empty_intent_still_empty(self):
        assert zh_keywords(SearchIntent()) == []


class TestJoinQuantSkipsPhrases:
    def test_multiword_phrase_not_used_for_like_matching(self, tmp_path):
        """社区帖用 LIKE 匹配，整串短语在正文里几乎不会原样出现。"""
        from autoalpha.sources.joinquant import JoinQuantSource
        src = JoinQuantSource(db_path=str(tmp_path / "t.db"), auto_crawl=False)
        # canonical_zh 整串是「ETF 截面动量 轮动」，拆开后才有可用的单词
        it = SearchIntent(canonical_zh="ETF 截面动量 轮动")
        # 库里无数据时返回空，但不应因整串短语而抛错或误判
        assert src.search(it) == []

    def test_word_level_terms_used(self, tmp_path):
        from autoalpha.sources.joinquant import JoinQuantSource, JoinQuantStore, normalize
        db = str(tmp_path / "t.db")
        store = JoinQuantStore(db)
        store.upsert([normalize(
            {"title": "动量策略回测", "postId": "a", "uniqueKey": "a"},
            content="动量策略正文")])
        src = JoinQuantSource(db_path=db, auto_crawl=False)
        out = src.search(SearchIntent(canonical_zh="ETF 动量 轮动"))
        assert len(out) == 1, "拆开后的单词「动量」应能命中"


class TestIntentParsing:
    def test_recognize_populates_canonical_zh(self, monkeypatch):
        from autoalpha.agents.intent_agent import IntentAgent

        class _Fake:
            def complete(self, **kw):
                return ('{"canonical_en":"momentum ETF",'
                        '"canonical_zh":"ETF 动量 轮动",'
                        '"market":"china_a","factor_family":"momentum",'
                        '"aliases":[],"confidence":0.8}')
        it = IntentAgent(_Fake()).recognize("A股ETF动量")
        assert it.canonical_zh == "ETF 动量 轮动"

    def test_missing_canonical_zh_is_tolerated(self, monkeypatch):
        """旧版 LLM 输出没有该字段时不能崩，降级到词典匹配。"""
        from autoalpha.agents.intent_agent import IntentAgent

        class _Fake:
            def complete(self, **kw):
                return '{"canonical_en":"momentum ETF","market":"china_a"}'
        it = IntentAgent(_Fake()).recognize("A股ETF动量")
        assert it.canonical_zh == ""
