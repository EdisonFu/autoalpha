"""
量化博客/机构 RSS 灵感池。

与 Step 2 的检索源不同：RSS 只给最新 N 条，**不能按关键词查**。
所以它不是检索源，而是「定期抓取 → 本地索引 → 离线筛选」的灵感池，
用来回答「最近业界在研究什么」，而不是「关于 X 有哪些文献」。

实测可用（均带摘要）：
    alphaarchitect  5 条   quantpedia 10 条
    quantinsti     15 条   nber       35 条
"""

from __future__ import annotations

import re
import sqlite3
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

UA = "Mozilla/5.0 (compatible; AutoAlpha/1.0)"
TIMEOUT = 20

FEEDS = {
    "alphaarchitect": ("https://alphaarchitect.com/feed/", "AlphaArchitect"),
    "quantpedia":     ("https://www.quantpedia.com/feed/", "Quantpedia"),
    "quantinsti":     ("https://blog.quantinsti.com/rss/", "QuantInsti"),
    "nber":           ("https://www.nber.org/rss/new.xml", "NBER"),
}


def fetch_feed(key: str) -> List[Dict]:
    """抓单个源。失败返回空列表，不抛 —— 单源挂掉不应影响其它源。"""
    if key not in FEEDS:
        return []
    url, label = FEEDS[key]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        raw = urllib.request.urlopen(req, timeout=TIMEOUT).read()
        root = ET.fromstring(raw)
    except Exception as e:
        print(f"[Feeds] {key} 抓取失败: {type(e).__name__}: {e}")
        return []

    out = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        out.append({
            "feed": key,
            "source_label": label,
            "title": title,
            "url": (it.findtext("link") or "").strip(),
            "summary": _clean_html(it.findtext("description") or "")[:800],
            "published": _parse_date(it.findtext("pubDate") or ""),
        })
    return out


def fetch_all(keys: Optional[List[str]] = None) -> List[Dict]:
    items: List[Dict] = []
    for k in (keys or list(FEEDS)):
        items.extend(fetch_feed(k))
    return items


def _clean_html(s: str) -> str:
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&lt;", "<").replace("&gt;", ">").replace("&#8217;", "'"))
    return re.sub(r"\s+", " ", s).strip()


def _parse_date(s: str) -> str:
    if not s:
        return ""
    try:
        return parsedate_to_datetime(s).astimezone(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        return m.group(0) if m else ""


# ── 本地索引 ─────────────────────────────────────────────────────────────────

class FeedStore:
    """
    SQLite 存已抓条目。按 url 唯一，重复抓取只累计不重复。
    关键词筛选在本地做 —— RSS 本身不支持按词检索。
    """

    def __init__(self, db_path: str = "autoalpha.db"):
        self.db_path = db_path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as c:
            c.execute("""
CREATE TABLE IF NOT EXISTS feed_items (
    url           TEXT PRIMARY KEY,
    feed          TEXT,
    source_label  TEXT,
    title         TEXT NOT NULL,
    summary       TEXT,
    published     TEXT,
    fetched_at    TEXT
)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_feed_pub "
                      "ON feed_items(published DESC)")

    def upsert(self, items: List[Dict]) -> int:
        """返回新增条数（已存在的不计）。"""
        now = datetime.now().isoformat()
        added = 0
        with self._conn() as c:
            for it in items:
                if not it.get("url"):
                    continue
                cur = c.execute(
                    "INSERT OR IGNORE INTO feed_items"
                    "(url,feed,source_label,title,summary,published,fetched_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (it["url"], it.get("feed", ""), it.get("source_label", ""),
                     it["title"], it.get("summary", ""), it.get("published", ""),
                     now))
                added += cur.rowcount
        return added

    def search(self, keywords: Optional[List[str]] = None,
               limit: int = 30) -> List[Dict]:
        """本地关键词筛选（标题或摘要命中任一词）。无关键词则返回最新。"""
        sql = ("SELECT url,feed,source_label,title,summary,published "
               "FROM feed_items")
        args: List = []
        kws = [k.strip() for k in (keywords or []) if k and k.strip()]
        if kws:
            terms = []
            for k in kws[:6]:
                terms.append("(LOWER(title) LIKE ? OR LOWER(summary) LIKE ?)")
                args += [f"%{k.lower()}%"] * 2
            sql += " WHERE " + " OR ".join(terms)
        sql += " ORDER BY published DESC, fetched_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            cur = c.execute(sql, args)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def stats(self) -> Dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM feed_items").fetchone()[0]
            by = dict(c.execute(
                "SELECT source_label, COUNT(*) FROM feed_items "
                "GROUP BY source_label").fetchall())
            latest = c.execute(
                "SELECT MAX(published) FROM feed_items").fetchone()[0]
        return {"total": total, "by_source": by, "latest": latest or ""}


def refresh(db_path: str = "autoalpha.db",
            keys: Optional[List[str]] = None) -> Dict:
    store = FeedStore(db_path)
    items = fetch_all(keys)
    added = store.upsert(items)
    return {"fetched": len(items), "added": added, **store.stats()}
