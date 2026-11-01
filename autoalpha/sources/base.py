"""
Step 2：多源资料检索的统一接口。

所有源产出同一 schema，便于跨源去重与统一排序：
    title / abstract / url / date / source / venue / doc_type / citations / doi
"""

from __future__ import annotations

import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

# OpenAlex / Crossref 都提供「礼貌池」：请求里带联系邮箱可获得更高、
# 更稳定的速率配额。可用 AUTOALPHA_CONTACT_EMAIL 覆盖。
CONTACT = os.environ.get("AUTOALPHA_CONTACT_EMAIL", "autoalpha@example.com")
UA = f"AutoAlpha/1.0 (research tool; mailto:{CONTACT})"
TIMEOUT = 25
MAX_RETRY = 3


class PaperSource:
    """检索源基类。子类实现 search()。"""

    name = "base"
    doc_type = "paper"

    def search(self, intent, limit: int = 20) -> List[Dict]:
        raise NotImplementedError

    # ── 工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get(url: str, headers: Optional[Dict] = None) -> bytes:
        """
        带退避重试的 GET。429/5xx 是这些公共 API 的常态（实测多源并联时
        OpenAlex 很容易触发 429），重试一次通常就过了。
        """
        last = None
        for attempt in range(MAX_RETRY):
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       **(headers or {})})
            try:
                return urllib.request.urlopen(req, timeout=TIMEOUT).read()
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRY - 1:
                    time.sleep((2 ** attempt) + random.random())
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                if attempt < MAX_RETRY - 1:
                    time.sleep(1 + attempt)
                    continue
                raise
        if last:
            raise last
        return b""

    def _safe_search(self, intent, limit: int) -> List[Dict]:
        """单个源失败不应拖垮整体检索。"""
        try:
            return self.search(intent, limit) or []
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[{self.name}] 网络失败，跳过该源: {e}")
        except Exception as e:
            print(f"[{self.name}] 检索失败，跳过该源: {type(e).__name__}: {e}")
        return []

    @staticmethod
    def _mk(title: str, abstract: str, url: str, date: str, source: str,
            venue: str = "", doc_type: str = "paper",
            citations: int = 0, doi: str = "") -> Dict:
        return {
            "title": (title or "").strip(),
            "abstract": (abstract or "").strip(),
            "pdf_url": url or "",
            "date": date or "",
            "source": source,
            "venue": (venue or "").strip(),
            "doc_type": doc_type,
            "citations": int(citations or 0),
            "doi": (doi or "").lower().strip(),
        }


# ── 跨源去重 ─────────────────────────────────────────────────────────────────

def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())[:80]


def dedupe(papers: List[Dict]) -> List[Dict]:
    """
    按 DOI 优先、标题归一兜底去重。
    同一篇论文常同时出现在 arXiv（preprint）与 Crossref（期刊版），
    保留元数据更全的那条（有 DOI、有摘要、引用数更高）。
    """
    best: Dict[str, Dict] = {}
    order: List[str] = []
    doi_to_key: Dict[str, str] = {}     # DOI → 已建组的标题 key

    for p in papers:
        tkey = _norm_title(p.get("title", ""))
        doi = (p.get("doi") or "").strip()
        # 以标题为主键：同一篇在 arXiv 常无 DOI、在 Crossref 有 DOI，
        # 若用「DOI 优先」建组，两条的 key 不同就永远合并不了 ——
        # 而这正是跨源去重最常见的情形。DOI 只用来补抓标题差异较大的同一篇。
        key = doi_to_key.get(doi) if doi else None
        if key is None:
            key = tkey or doi
        if not key:
            continue
        if doi and doi not in doi_to_key:
            doi_to_key[doi] = key
        if key not in best:
            best[key] = p
            order.append(key)
            continue
        cur = best[key]
        if _richness(p) > _richness(cur):
            # 保留更丰富的那条，但合并来源标记便于界面展示
            p = dict(p)
            p["also_in"] = sorted({cur.get("source", ""), *cur.get("also_in", []),
                                   p.get("source", "")} - {p.get("source", "")})
            best[key] = p
        else:
            cur.setdefault("also_in", [])
            s = p.get("source", "")
            if s and s != cur.get("source") and s not in cur["also_in"]:
                cur["also_in"].append(s)

    return [best[k] for k in order]


def _richness(p: Dict) -> int:
    score = 0
    if p.get("doi"):
        score += 4
    score += min(len(p.get("abstract") or ""), 600) // 150
    if p.get("venue"):
        score += 2
    if p.get("citations"):
        score += 1
    return score
