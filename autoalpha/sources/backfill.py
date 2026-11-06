"""
摘要回填：给缺摘要的条目补上摘要，靠查别家的库，不靠爬网。

**为什么需要**：实测一次真实召回 115 条去重结果里，25 条（22%）只有标题
——没有摘要，正文也拿不到（出版商登录墙）。这些条目带着一个标题进 LLM
打分，几乎必然被打低分丢掉，等于白召回。SSRN 工作论文尤其吃亏，
《How to Improve ETF Sector Momentum》这类标题一看就对口的，系统却一无所知。

**为什么是查库而不是爬网**：缺摘要不等于「摘要被墙了」，而是「这个源的
这条记录里没有」。同一个 DOI 在另一家的库里往往就有。实测 33 条缺摘要样本：

    OpenAlex 批量回查    9/33 = 27%
    Crossref 批量回查    5/33 = 15%
    并集                14/33 = 42%     7 秒，零新增依赖

两边**完全不重叠**（仅 OpenAlex 有 9，仅 Crossref 有 5，交集 0），
互为补充。相比之下所有抓取方案都只有 6%：

    urllib / curl_cffi / Scrapling   均为 2/33 = 6%

原因是这些 URL 几乎全是 doi.org，解析到 Cloudflare 后在部分网络下大面积
超时；能连通的又多是 Elsevier / Wiley / T&F 的登录墙。TLS 指纹和浏览器
指纹对「连不上」和「内容不在页面上」都无效。详见 docs/E2E_REVIEW.md。
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Dict, List

from .base import CONTACT, PaperSource
from .openalex import _rebuild_abstract

_OA_API = "https://api.openalex.org/works"
_CR_API = "https://api.crossref.org/works"

# 单次请求塞多少 DOI。两家都用 URL 查询串传过滤条件，太长会被截断或 414。
_OA_BATCH = 25
_CR_BATCH = 20

# 低于这个长度视同没有摘要 —— 和 orchestrator 判缺摘要的口径保持一致，
# 否则会出现「回填成功」但下游仍当它是空的。
MIN_ABSTRACT = 80


class _Batch(PaperSource):
    """复用基类的退避重试。本身不作为检索源注册。"""
    name = "backfill"


_http = _Batch()


def _norm_doi(doi: str) -> str:
    return (doi or "").replace("https://doi.org/", "").strip().lower()


def _from_openalex(dois: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i in range(0, len(dois), _OA_BATCH):
        chunk = dois[i:i + _OA_BATCH]
        f = "doi:" + "|".join(chunk)
        url = (f"{_OA_API}?per-page=50&mailto={CONTACT}"
               f"&select=doi,abstract_inverted_index"
               f"&filter={urllib.parse.quote(f, safe=':|/.')}")
        try:
            data = json.loads(_http._get(url).decode("utf-8", "replace"))
        except Exception as e:
            print(f"[Backfill] OpenAlex 批次失败，跳过: {type(e).__name__}")
            continue
        for w in data.get("results", []):
            abs_ = _rebuild_abstract(w.get("abstract_inverted_index"))
            if len(abs_) >= MIN_ABSTRACT:
                out[_norm_doi(w.get("doi"))] = abs_
    return out


def _from_crossref(dois: List[str]) -> Dict[str, str]:
    from .crossref import _clean_jats
    out: Dict[str, str] = {}
    for i in range(0, len(dois), _CR_BATCH):
        chunk = dois[i:i + _CR_BATCH]
        f = ",".join(f"doi:{d}" for d in chunk)
        url = (f"{_CR_API}?rows=50&select=DOI,abstract"
               f"&filter={urllib.parse.quote(f, safe=':,/.')}")
        try:
            data = json.loads(_http._get(url).decode("utf-8", "replace"))
        except Exception as e:
            print(f"[Backfill] Crossref 批次失败，跳过: {type(e).__name__}")
            continue
        for it in data.get("message", {}).get("items", []):
            abs_ = _clean_jats(it.get("abstract") or "")
            if len(abs_) >= MIN_ABSTRACT:
                out[_norm_doi(it.get("DOI"))] = abs_
    return out


def backfill_abstracts(papers: List[Dict], verbose: bool = True) -> int:
    """
    就地给缺摘要的条目补摘要，返回补上的条数。

    只查对方没覆盖到的 DOI：先 OpenAlex，剩下的再 Crossref。
    两家实测零重叠，所以这个顺序只影响请求数，不影响结果。
    """
    need = [p for p in papers
            if len(p.get("abstract") or "") < MIN_ABSTRACT and p.get("doi")]
    if not need:
        return 0

    by_doi: Dict[str, List[Dict]] = {}
    for p in need:
        by_doi.setdefault(_norm_doi(p["doi"]), []).append(p)
    dois = list(by_doi)

    found = _from_openalex(dois)
    rest = [d for d in dois if d not in found]
    if rest:
        found.update(_from_crossref(rest))

    n = 0
    for doi, abs_ in found.items():
        for p in by_doi.get(doi, []):
            p["abstract"] = abs_
            p["abstract_backfilled"] = True     # 供界面标注来源
            n += 1

    if verbose and need:
        print(f"[Backfill] 缺摘要 {len(need)} 条 → 回填 {n} 条 "
              f"({n / len(need):.0%})，仍缺 {len(need) - n} 条")
    return n


# ── 非论文条目过滤 ──────────────────────────────────────────────────────────

# 实测 OECD 的图表、书籍章节会混进召回结果，占着打分配额还要花 token
# 让 LLM 判断「这个图表和动量因子相关吗」。典型样本：
#   Figure 7.18. Global exchange-traded fund assets
#   Figure 3.13. Synthetic exchange traded Fund (ETF) example
#   Building Your Exchange-Traded Fund Team
#   The Steps to Start an Exchange-Traded Fund
import re  # noqa: E402

_JUNK_TITLE = re.compile(
    r"^\s*(?:figure|fig\.?|table|chart|box|exhibit|graph|annex|appendix)\s*"
    r"[\dIVX]+[.\s]", re.I)

# 书籍类正文通常是教科书式介绍，不含可复现的因子方法
_JUNK_TYPES = {"book"}


def is_research_item(paper: Dict) -> bool:
    """这条召回是不是一篇可供提取因子的研究文献。"""
    title = (paper.get("title") or "").strip()
    if not title:
        return False
    if _JUNK_TITLE.match(title):
        return False
    if (paper.get("doc_type") or "") in _JUNK_TYPES:
        return False
    return True


def drop_non_research(papers: List[Dict], verbose: bool = True) -> List[Dict]:
    """滤掉图表、书籍这类非研究条目。"""
    keep = [p for p in papers if is_research_item(p)]
    dropped = len(papers) - len(keep)
    if verbose and dropped:
        print(f"[Backfill] 滤除非研究条目 {dropped} 条"
              f"（图表/书籍），余 {len(keep)} 条")
    return keep
