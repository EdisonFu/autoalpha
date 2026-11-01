"""
RSS 灵感池测试。不打网络：解析与存储逻辑用构造数据验证。
"""

import os
import tempfile

import pytest

from autoalpha.sources.feeds import (
    FeedStore, _clean_html, _parse_date, FEEDS,
)


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield FeedStore(path)
    os.unlink(path)


def _item(title, url, summary="", published="2026-10-01", feed="quantpedia"):
    return {"feed": feed, "source_label": "Quantpedia", "title": title,
            "url": url, "summary": summary, "published": published}


# ── 解析 ────────────────────────────────────────────────────────────────────

def test_clean_html_strips_tags_and_entities():
    raw = "<p>Momentum &amp; <b>reversal</b></p><script>bad()</script>"
    out = _clean_html(raw)
    assert "Momentum & reversal" in out
    assert "bad()" not in out and "<" not in out


def test_parse_date_handles_rfc822():
    assert _parse_date("Thu, 23 Oct 2026 10:00:00 +0000") == "2026-10-23"
    assert _parse_date("2026-10-23T10:00:00Z") == "2026-10-23"
    assert _parse_date("") == ""
    assert _parse_date("garbage") == ""


def test_feed_registry_is_not_empty():
    assert set(FEEDS) >= {"alphaarchitect", "quantpedia", "nber"}
    for url, label in FEEDS.values():
        assert url.startswith("http") and label


# ── 存储 ────────────────────────────────────────────────────────────────────

def test_upsert_is_idempotent(store):
    items = [_item("A", "u1"), _item("B", "u2")]
    assert store.upsert(items) == 2
    assert store.upsert(items) == 0, "同一 url 重复抓取不得重复入库"
    assert store.stats()["total"] == 2


def test_upsert_skips_items_without_url(store):
    assert store.upsert([_item("A", ""), _item("B", "u2")]) == 1


def test_search_filters_by_keyword(store):
    store.upsert([
        _item("ETF momentum rotation", "u1"),
        _item("Bond carry strategies", "u2"),
        _item("Deep learning for options", "u3", summary="uses momentum signals"),
    ])
    hits = store.search(["momentum"])
    titles = {h["title"] for h in hits}
    assert "ETF momentum rotation" in titles
    assert "Deep learning for options" in titles, "摘要命中也应算"
    assert "Bond carry strategies" not in titles


def test_search_is_case_insensitive(store):
    store.upsert([_item("ETF Momentum", "u1")])
    assert len(store.search(["momentum"])) == 1
    assert len(store.search(["MOMENTUM"])) == 1


def test_search_without_keywords_returns_latest(store):
    store.upsert([_item("old", "u1", published="2020-01-01"),
                  _item("new", "u2", published="2026-10-01")])
    out = store.search(None, limit=10)
    assert out[0]["title"] == "new", "应按发布时间倒序"


def test_search_respects_limit(store):
    store.upsert([_item(f"t{i}", f"u{i}") for i in range(10)])
    assert len(store.search(None, limit=3)) == 3


def test_stats_groups_by_source(store):
    store.upsert([_item("a", "u1"), _item("b", "u2")])
    st = store.stats()
    assert st["total"] == 2 and st["by_source"]["Quantpedia"] == 2
