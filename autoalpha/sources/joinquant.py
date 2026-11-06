"""
聚宽社区：中文量化研究资料源。

**为什么是这个源**：实测各中文候选后只有它同时满足「内容对口」与「可合法获取」。

    源              量化内容密度    门槛
    东方财富研报      0.1%           无反爬，但「策略」类实为宏观/市场展望
    知网 / 百度学术   —              指纹验证、安全验证
    米筐             —              robots 全站禁止
    BigQuant         —              robots 禁止 /community/search
    聚宽社区          52%            robots `Allow: /`，纯 HTTP 即可

120 帖里 63 帖与因子/策略/回测直接相关，标题如「著名的Fama-French三因子模型
策略完美实现」「单因子有效性分析：分层法 / 多空组合 / 各分位数收益」——
后者几乎就是本项目回测层的教学版。

**两个必须知道的坑**（都实测撞过）：
  1. `listV2` 必须带 `x-requested-with: XMLHttpRequest`，且 referer 要是
     `/view/community/list?listType=1`。用社区首页当 referer 会返回
     `{"code":"20000","msg":"系统异常"}`，即使带 cookie 也一样。
  2. `listV2` 的 `keyword`/`title` 参数**被服务端忽略**（传不同词返回完全
     相同的列表）。所以它和 RSS 一样只能「抓列表 → 本地索引 → 离线筛选」，
     不能实时按词检索。

**合规**：robots.txt 为 `Allow: /`，仅禁后台与 clone 接口。
正文只保留前 `ABSTRACT_CHARS` 字作为摘要并附原文链接，不做整站镜像。
聚宽公告自 2026-08 起不再服务非中国大陆 IP，境外部署需自行评估。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
BASE = "https://www.joinquant.com"
LIST_API = BASE + "/community/post/listV2"
DETAIL_API = BASE + "/community/post/detailV2"
# 实测有效的 referer。用 /community/index 会被判为非法来源。
REFERER = BASE + "/view/community/list?listType=1"
TIMEOUT = 25

ABSTRACT_CHARS = 2000     # 只留摘要长度，不整篇入库

# 代码痕迹在【完整正文】上检测，只存布尔标志。
# 摘要被截断且剥掉了 Markdown 代码围栏，单看它常常判不出这篇帖附了策略代码，
# 导致可复现性被低估。存全文则既没必要也不该做。
_CODE_RE = re.compile(
    r"def initialize|handle_data\(|import pandas|import numpy|"
    r"get_fundamentals|order_target|run_daily|def compute|```python", re.I)
DEFAULT_PAGES = 6         # 6 页 × 20 ≈ 120 帖，实测约 1 秒


def _headers() -> Dict[str, str]:
    return {"User-Agent": UA, "referer": REFERER,
            "x-requested-with": "XMLHttpRequest",
            "accept": "application/json, text/plain, */*"}


def _get_json(url: str) -> Dict:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _clean_md(text: str) -> str:
    """Markdown → 纯文本摘要。图片与链接 URL 对检索和概述都是噪声。"""
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text or "")     # 图片
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)            # 链接保留文字
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[#>*`\-]{1,}\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def fetch_list(page: int, limit: int = 20) -> List[Dict]:
    q = urllib.parse.urlencode({"limit": limit, "page": page,
                                "cate": 3, "type": "isNew"})
    try:
        d = _get_json(f"{LIST_API}?{q}")
    except Exception as e:
        print(f"[JoinQuant] 第 {page} 页抓取失败: {type(e).__name__}")
        return []
    return (d.get("data") or {}).get("list") or []


def fetch_detail(post_id: str) -> str:
    try:
        d = _get_json(f"{DETAIL_API}?postId={post_id}")
    except Exception:
        return ""
    return (d.get("data") or {}).get("content") or ""


def normalize(raw: Dict, content: str = "") -> Dict:
    """统一成 paper schema，与 arXiv/OpenAlex/Crossref 的结果可混合去重排序。"""
    body = content or raw.get("content") or ""
    key = raw.get("uniqueKey") or raw.get("postId") or ""
    return {
        "title": (raw.get("title") or "").strip(),
        "abstract": _clean_md(body)[:ABSTRACT_CHARS],
        "pdf_url": f"{BASE}/view/community/detail/{key}" if key else "",
        "date": (raw.get("addTime") or "")[:10],
        "source": "joinquant",
        "venue": "聚宽社区",
        "doc_type": "cn_post",
        "citations": 0,
        "doi": "",
        # 社区特有的质量信号，供排序使用
        "is_best": raw.get("isBest") == "1",
        "views": _to_int(raw.get("viewCount")),
        "collections": _to_int(raw.get("collectionCount")),
        "replies": _to_int(raw.get("replyCount")),
        "backtest_clones": _to_int(raw.get("backtestCloneCount")),
        "has_code": bool(_CODE_RE.search(body)),
        "lang": "zh",
    }


def _to_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def crawl(pages: int = DEFAULT_PAGES, with_content: bool = True,
          workers: int = 6) -> List[Dict]:
    """抓列表 → 抓正文 → 归一化。单页失败不影响其余页。"""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pages_data = list(ex.map(fetch_list, range(1, pages + 1)))
    raws = [it for pg in pages_data for it in pg]
    if not raws:
        return []

    contents: List[str] = [""] * len(raws)
    if with_content:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            contents = list(ex.map(fetch_detail,
                                   [r.get("postId", "") for r in raws]))
    return [normalize(r, c) for r, c in zip(raws, contents)]


# ── 本地索引 ────────────────────────────────────────────────────────────────

class JoinQuantStore:
    """
    SQLite 存已抓帖子。按 url 唯一，重复抓取只累计不重复。

    与 FeedStore 分开建表：社区帖有互动指标（精华/收藏/回测克隆），
    排序逻辑不同，混在一起会让字段语义含糊。
    """

    def __init__(self, db_path: str = "autoalpha.db"):
        self.db_path = db_path
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as c:
            c.execute("""
CREATE TABLE IF NOT EXISTS jq_posts (
    url           TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    abstract      TEXT,
    published     TEXT,
    is_best       INTEGER DEFAULT 0,
    views         INTEGER DEFAULT 0,
    collections   INTEGER DEFAULT 0,
    backtest_clones INTEGER DEFAULT 0,
    has_code      INTEGER DEFAULT 0,
    fetched_at    TEXT
)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_jq_pub "
                      "ON jq_posts(published DESC)")
            # 老库补列：不重建表，已抓的数据不该丢
            cols = {r[1] for r in c.execute("PRAGMA table_info(jq_posts)")}
            if "has_code" not in cols:
                c.execute("ALTER TABLE jq_posts ADD COLUMN has_code "
                          "INTEGER DEFAULT 0")

    def upsert(self, items: List[Dict]) -> int:
        now = datetime.now().isoformat()
        added = 0
        with self._conn() as c:
            for it in items:
                if not it.get("url") and not it.get("pdf_url"):
                    continue
                cur = c.execute(
                    "INSERT OR IGNORE INTO jq_posts"
                    "(url,title,abstract,published,is_best,views,collections,"
                    " backtest_clones,has_code,fetched_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (it["pdf_url"], it["title"], it.get("abstract", ""),
                     it.get("date", ""), int(it.get("is_best", False)),
                     it.get("views", 0), it.get("collections", 0),
                     it.get("backtest_clones", 0),
                     int(it.get("has_code", False)), now))
                added += cur.rowcount
        return added

    def search(self, keywords: Optional[List[str]] = None,
               limit: int = 20) -> List[Dict]:
        """
        本地中文筛选。排序：精华优先，其次收藏数（社区公认的实用度信号），
        再按日期。不用阅读量 —— 置顶帖阅读量虚高，会挤掉新内容。
        """
        sql = ("SELECT url,title,abstract,published,is_best,views,collections,"
               "backtest_clones,has_code FROM jq_posts")
        args: List = []
        kws = [k.strip() for k in (keywords or []) if k and k.strip()]
        if kws:
            terms = []
            for k in kws[:6]:
                terms.append("(title LIKE ? OR abstract LIKE ?)")
                args += [f"%{k}%", f"%{k}%"]
            sql += " WHERE " + " OR ".join(terms)
        sql += (" ORDER BY is_best DESC, collections DESC, published DESC "
                "LIMIT ?")
        args.append(limit)
        with self._conn() as c:
            cur = c.execute(sql, args)
            cols = [d[0] for d in cur.description]
            out = []
            for row in cur.fetchall():
                r = dict(zip(cols, row))
                out.append({
                    "title": r["title"], "abstract": r["abstract"] or "",
                    "pdf_url": r["url"], "date": r["published"] or "",
                    "source": "joinquant", "venue": "聚宽社区",
                    "doc_type": "cn_post", "citations": 0, "doi": "",
                    "is_best": bool(r["is_best"]), "views": r["views"],
                    "collections": r["collections"],
                    "backtest_clones": r["backtest_clones"],
                    "has_code": bool(r["has_code"]), "lang": "zh",
                })
            return out

    def stats(self) -> Dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM jq_posts").fetchone()[0]
            latest = c.execute(
                "SELECT MAX(published) FROM jq_posts").fetchone()[0]
        return {"total": total, "latest": latest or ""}


def refresh(db_path: str = "autoalpha.db", pages: int = DEFAULT_PAGES) -> Dict:
    items = crawl(pages)
    store = JoinQuantStore(db_path)
    added = store.upsert(items)
    return {"fetched": len(items), "added": added, **store.stats()}


# ── 作为检索源接入主链路 ────────────────────────────────────────────────────

from .base import PaperSource          # noqa: E402
from .zh_terms import zh_keywords      # noqa: E402


class JoinQuantSource(PaperSource):
    """
    把聚宽本地库包装成统一检索源，与英文源一起进跨源去重与排序。

    注意它不是实时检索：帖子是预先抓取入库的。库为空时自动抓一次，
    之后靠用户手动「更新中文库」刷新 —— 每次检索都重抓整库既慢也没必要。
    """

    name = "joinquant"

    def __init__(self, db_path: str = "autoalpha.db",
                 auto_crawl: bool = True, pages: int = DEFAULT_PAGES):
        self.db_path = db_path
        self.store = JoinQuantStore(db_path)
        self.auto_crawl = auto_crawl
        self.pages = pages

    def search(self, intent, limit: int = 20) -> List[Dict]:
        kws = zh_keywords(intent)
        if not kws:
            # 没有中文关键词就不该返回整库 —— 那不是「相关」，是噪声
            return []
        if self.store.stats()["total"] == 0:
            if not self.auto_crawl:
                print("[JoinQuant] 本地库为空，跳过（可在界面更新中文库）")
                return []
            print("[JoinQuant] 本地库为空，首次自动抓取…")
            try:
                r = refresh(self.db_path, self.pages)
                print(f"[JoinQuant] 抓取 {r['fetched']} 帖，新增 {r['added']}，"
                      f"库内共 {r['total']}")
            except Exception as e:
                print(f"[JoinQuant] 首次抓取失败: {type(e).__name__}: {e}")
                return []
        return self.store.search(kws, limit)
