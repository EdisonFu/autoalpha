import json
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from ..llm import LLMClient
from ..json_repair import loads_lenient, JSONRepairError
from ..utils import strip_code_fence

# arXiv 每次最多拉取的论文数
# 去掉 abs: 关键词过滤后召回更宽，适当多拉一些由 LLM 语义过滤
_PREFETCH = 200

# 语义过滤时每批论文数（含摘要，控制 token 量）
_FILTER_BATCH = 20

# 实际送去 LLM 打分的论文数上限。
# arXiv 已按 relevance 排序，深尾部分极少进入最终展示（默认只展示 10 篇），
# 对 200 篇全部打分是纯浪费：实测 score 阶段 10 次调用吃掉 22.9k 输入 token，
# 占检索阶段总成本的 84.5%。截到 60 篇可省约 70%，对召回几无影响。
# 需要更宽召回时调大此值。
_SCORE_LIMIT = 60

# 排序中时间项的权重。实测 0.4 会压倒相关性（见 _combined_sort 注释）。
_RECENCY_W = 0.15
# 时间项半衰期：3 年前的论文时间得分 0.5，而非线性归一化的近 0
_RECENCY_HALFLIFE_DAYS = 365 * 3
# 语义得分低于此值直接淘汰。原 5 太松，放进了大量边缘论文
_MIN_RELEVANCE = 7

# 摘要截断长度（LLM 语义分析用，过长影响批处理性能）
_ABSTRACT_CHARS = 400

_KW_SYSTEM = """\
You are an arXiv quantitative-finance search expert.
Given a research idea (possibly in Chinese), output a JSON object with two fields:

{
  "title_terms":    ["term1", "term2", "term3"],
  "abstract_terms": ["term4", "term5", "term6", "term7"]
}

Rules:
- Translate Chinese to English academic terminology
- title_terms:    3-5 SHORT phrases likely to appear in paper TITLES
                  (strategy type, method, asset class)
- abstract_terms: 4-6 broader concept terms for abstract/keyword search
                  (include synonyms and related methodology terms)
- Focus on: strategy type (momentum / mean-reversion / trend-following / carry /
            factor / rotation / allocation), method (machine learning / deep learning /
            optimization / regression), asset class (ETF / equity / futures / bonds /
            portfolio)
- OMIT country/market names (China / A-share / US) — they hurt recall
- Output ONLY valid JSON, no explanation

Examples
--------
Idea: 基于动量的A股ETF轮动策略
Output:
{
  "title_terms":    ["momentum", "ETF rotation", "sector rotation", "factor rotation"],
  "abstract_terms": ["momentum strategy", "ETF", "sector rotation", "portfolio rebalancing", "return momentum", "asset rotation"]
}

Idea: 机器学习因子选股
Output:
{
  "title_terms":    ["machine learning", "stock selection", "factor investing"],
  "abstract_terms": ["alpha factor", "cross-sectional", "return prediction", "feature importance", "neural network", "equity returns"]
}

Idea: 基于波动率的动态资产配置
Output:
{
  "title_terms":    ["volatility", "dynamic allocation", "risk parity"],
  "abstract_terms": ["volatility targeting", "portfolio optimization", "risk management", "drawdown", "asset allocation", "regime"]
}
"""

_FILTER_SYSTEM = """\
You are a research relevance judge for quantitative finance.

Given a research idea and a numbered list of papers (title + abstract excerpt),
SEMANTICALLY analyze the CONTENT of each abstract to assess how well the paper
matches the research idea — do NOT rely on keyword overlap alone.

Judge whether the paper studies the same strategy type, methodology, or asset class
as described in the idea. A paper is relevant if its abstract describes a strategy
or method that could inform, validate, or extend the research idea.

Output a JSON array of relevance scores (integers 1-10) in the SAME ORDER as the input list.
  8-10 = highly relevant (same strategy type / methodology / asset class)
  5-7  = somewhat relevant (related area, could provide useful reference)
  1-4  = not relevant (different domain or methodology)

Output ONLY the JSON array like [8,3,9,...], no explanation.
"""


class PaperSearchAgent:
    """
    Agent 1：论文检索
    仅在 input_type=idea 且 search_confidence>=0.6 时激活

    搜索策略：
      1. LLM 提取标题关键词（仅 title_terms，不做摘要关键词匹配）
      2. 构造 ti: 字段前缀的 arXiv 查询
      3. 按 relevance 拉取最多 200 篇
      4. 客户端按提交日期倒序排列
      5. LLM 语义过滤：传入标题+摘要段落，按语义内容打分，移除低相关论文
         每批 20 篇，逐批调用，最终合并
      6. 结果按 idea 缓存，翻页直接从内存取
    """

    ARXIV_API  = "https://export.arxiv.org/api/query"
    CATEGORIES = ["q-fin.PM", "q-fin.TR", "q-fin.ST", "q-fin.MF"]

    def __init__(self, llm: LLMClient, local_library_dir: str = None):
        self.llm = llm
        self.local_library_dir = local_library_dir
        self._kw_cache:      Dict[str, dict]       = {}   # idea → {title_terms}
        self._results_cache: Dict[str, List[Dict]] = {}   # idea → 已过滤+排序的论文列表

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def search(self, idea: str, max_results: int = 10) -> Tuple[List[Dict], float]:
        """一次性搜索（非分页）。"""
        papers = self._get_results(idea)[:max_results]
        return (papers, 0.9) if papers else ([], 0.0)

    def search_paged(self, idea: str,
                     page: int = 1,
                     page_size: int = 10) -> Tuple[List[Dict], int]:
        """
        分页搜索，按日期倒序，翻页从缓存取（不重复调 arXiv / LLM）。
        返回 (本页论文列表, 过滤后总数)
        """
        all_papers = self._get_results(idea)
        total      = len(all_papers)
        start      = (page - 1) * page_size
        return all_papers[start: start + page_size], total

    # ── 缓存 & 主流程 ─────────────────────────────────────────────────────────

    def _get_results(self, idea: str) -> List[Dict]:
        """
        首次调用时执行完整搜索流程并缓存；后续直接返回缓存。
        """
        if idea not in self._results_cache:
            self._results_cache[idea] = self._fetch_and_rank(idea)
        return self._results_cache[idea]

    def _fetch_and_rank(self, idea: str) -> List[Dict]:
        """
        完整流程：
          arXiv 宽召回 → LLM 语义打分 → 过滤低分 → 日期+相关性综合排序
        """
        kw = self._get_keywords(idea)
        query = self._build_query(kw)

        print(f"[PaperSearch] arXiv 查询: {query[:120]}")
        papers, arxiv_total = self._fetch_arxiv(query, start=0,
                                                page_size=_PREFETCH,
                                                sort_by="relevance")
        if not papers:
            return []

        print(f"[PaperSearch] arXiv 命中 {arxiv_total} 篇，取前 {len(papers)} 篇")

        # LLM 语义打分（传入标题+摘要段落）
        # 只对 relevance 靠前的 _SCORE_LIMIT 篇打分，其余直接丢弃 —— 见该常量注释
        if len(papers) > _SCORE_LIMIT:
            print(f"[PaperSearch] 打分前截断: {len(papers)} → {_SCORE_LIMIT} 篇"
                  f"（省约 {(1 - _SCORE_LIMIT / len(papers)):.0%} 打分 token）")
            papers = papers[:_SCORE_LIMIT]
        papers, scores = self._llm_score(idea, papers)
        print(f"[PaperSearch] 语义过滤后剩 {len(papers)} 篇，按日期+相关性综合排序")

        # 综合排序：0.6 × 相关性得分（归一化）+ 0.4 × 时间新鲜度（归一化）
        papers = self._combined_sort(papers, scores)
        return papers

    @staticmethod
    def _combined_sort(papers: List[Dict], scores: List[int]) -> List[Dict]:
        """
        综合排序：combined = (1 - _RECENCY_W) × rel_norm + _RECENCY_W × recency

        实测问题（docs/E2E_REVIEW.md P-1）：原权重 0.6/0.4 下，输入
        「ETF截面动量因子」时 rel=7 的 *ChatGPT in Systematic Investing*
        排到 Top1（combined 0.787），而三篇 rel=8 的真·截面动量论文
        全部落后（0.645 / 0.645 / 0.480）—— 0.4 的时间权重足以推翻
        1 分的相关性差距。

        两处修正：
          1. 时间权重 0.4 → 0.15，相关性重新成为主导项
          2. 时间项改用【半衰期衰减】而非线性满量程归一化。
             线性归一化把 2020→2026 拉满 0~1，等于让「新」获得与
             「相关」同等的话语权；半衰期衰减让 3 年前的论文仍有
             0.5 权重，只对真正过时的做惩罚。
        """
        from datetime import datetime
        import math

        now = datetime.now().timestamp()
        half_life_sec = _RECENCY_HALFLIFE_DAYS * 86400

        def _to_ts(date_str: str) -> float:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").timestamp()
            except Exception:
                return 0.0

        ranked = []
        for p, s in zip(papers, scores):
            ts = _to_ts(p.get("date", ""))
            rel_norm = s / 10.0
            if ts > 0:
                age_sec = max(0.0, now - ts)
                recency = 0.5 ** (age_sec / half_life_sec)
            else:
                recency = 0.3          # 无日期：轻微惩罚，不至于排到最后
            combined = ((1.0 - _RECENCY_W) * rel_norm) + (_RECENCY_W * recency)
            p = dict(p)
            p["_relevance_score"] = s
            p["_recency"] = round(recency, 4)
            p["_combined_score"] = round(combined, 4)
            ranked.append(p)

        # 同分时相关性优先，使时间项只在真正平手时起 tie-break 作用
        ranked.sort(key=lambda x: (x["_combined_score"], x["_relevance_score"]),
                    reverse=True)
        return ranked

    # ── 关键词提取 & 查询构造 ─────────────────────────────────────────────────

    def _get_keywords(self, idea: str) -> dict:
        if idea not in self._kw_cache:
            self._kw_cache[idea] = self._extract_keywords(idea)
        return self._kw_cache[idea]

    def _extract_keywords(self, idea: str) -> dict:
        """
        LLM 提取关键词，返回 {title_terms: [...], abstract_terms: [...]}.
        解析失败时降级为简单分词。
        """
        raw = idea
        try:
            raw = self.llm.complete(system=_KW_SYSTEM,
                                    user=f"Research idea: {idea}",
                                    purpose="keyword")
            kw = loads_lenient(raw, expect=dict)
            if isinstance(kw.get("title_terms"), list) and kw["title_terms"]:
                return {
                    "title_terms":    kw["title_terms"],
                    "abstract_terms": kw.get("abstract_terms", []),
                }
        except Exception:
            pass

        # 降级：把 LLM 输出当空格分隔的关键词
        words = raw.replace(",", " ").split()
        return {"title_terms": words[:3], "abstract_terms": words[3:7]}

    def _build_query(self, kw: dict) -> str:
        """
        组装宽召回 arXiv 查询：
          (ti:term1 OR ti:term2 OR abs:term3 OR abs:term4 OR kw:term5 ...) AND (cat:...)

        三路并联召回：
          ti:  标题词（精准，权重高）
          abs: 摘要词（覆盖范围广）
          kw:  论文作者关键词（arXiv 支持 kw: 字段）
        语义相关性判断完全交由 LLM 读摘要内容负责，arXiv 侧只做宽召回。
        多词短语加引号。
        """
        def _q(term: str) -> str:
            t = term.strip()
            return f'"{t}"' if " " in t else t

        title_parts = [f'ti:{_q(t)}'  for t in kw.get("title_terms", [])    if t.strip()]
        abs_parts   = [f'abs:{_q(t)}' for t in kw.get("abstract_terms", []) if t.strip()]
        kw_parts    = [f'kw:{_q(t)}'  for t in kw.get("title_terms", [])    if t.strip()]
        cat_filter  = " OR ".join(f"cat:{c}" for c in self.CATEGORIES)

        all_parts    = title_parts + abs_parts + kw_parts
        keyword_part = " OR ".join(all_parts) if all_parts else "quantitative finance"
        return f"({keyword_part}) AND ({cat_filter})"

    # ── arXiv 请求 ────────────────────────────────────────────────────────────

    def _fetch_arxiv(self, query: str, start: int, page_size: int,
                     sort_by: str = "relevance") -> Tuple[List[Dict], int]:
        params = urllib.parse.urlencode({
            "search_query": query,
            "start":        start,
            "max_results":  page_size,
            "sortBy":       sort_by,
            "sortOrder":    "descending",
        })
        url = f"{self.ARXIV_API}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                xml_data = resp.read()
        except Exception as e:
            print(f"[PaperSearch] arXiv 请求失败: {e}")
            return [], 0

        ns = {
            "atom":       "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        }
        root = ET.fromstring(xml_data)

        total_el = root.find("opensearch:totalResults", ns)
        total    = int(total_el.text) if total_el is not None and total_el.text else 0

        papers = []
        for entry in root.findall("atom:entry", ns):
            id_el      = entry.find("atom:id", ns)
            title_el   = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            pub_el     = entry.find("atom:published", ns)
            upd_el     = entry.find("atom:updated", ns)

            if id_el is None or title_el is None:
                continue

            arxiv_id = id_el.text.split("/abs/")[-1].strip()
            title    = title_el.text.strip().replace("\n", " ")
            abstract = (summary_el.text or "").strip().replace("\n", " ")
            pdf_url  = f"https://arxiv.org/pdf/{arxiv_id}"
            date_el  = pub_el or upd_el
            date_str = date_el.text[:10] if date_el is not None else ""

            papers.append({
                "arxiv_id": arxiv_id,
                "title":    title,
                "abstract": abstract,
                "pdf_url":  pdf_url,
                "date":     date_str,
            })

        return papers, total

    # ── LLM 语义打分 ──────────────────────────────────────────────────────────

    def _llm_score(self, idea: str,
                   papers: List[Dict],
                   min_score: int = _MIN_RELEVANCE) -> Tuple[List[Dict], List[int]]:
        """
        对每篇论文传入「标题 + 摘要段落」，让 LLM 按语义内容打分（1-10）。
        分批处理（每批 _FILTER_BATCH 篇），防止 token 超限。
        返回 (过滤后的论文列表, 对应分数列表)。
        全部被过滤时降级返回原列表（分数全为 5）。
        """
        if not papers:
            return papers, []

        all_scores: List[int] = []

        for batch_start in range(0, len(papers), _FILTER_BATCH):
            batch = papers[batch_start: batch_start + _FILTER_BATCH]
            papers_text = "\n\n".join(
                f"{batch_start + i + 1}. Title: {p['title']}\n"
                f"   Abstract: {p['abstract'][:_ABSTRACT_CHARS]}"
                for i, p in enumerate(batch)
            )
            try:
                raw = self.llm.complete(
                    system=_FILTER_SYSTEM,
                    user=(
                        f"Research idea: {idea}\n\n"
                        f"Papers (score each based on abstract content):\n{papers_text}"
                    ),
                    purpose="score",
                ).strip()

                try:
                    batch_scores = [int(x) for x in loads_lenient(raw, expect=list)]
                except Exception:
                    batch_scores = []
                if batch_scores:
                    while len(batch_scores) < len(batch):
                        batch_scores.append(1)
                    all_scores.extend(batch_scores[:len(batch)])
                else:
                    all_scores.extend([5] * len(batch))

            except Exception as e:
                print(f"[PaperSearch] LLM 语义打分失败（批次 {batch_start}，降级）: {e}")
                all_scores.extend([5] * len(batch))

        pairs = [(p, s) for p, s in zip(papers, all_scores) if s >= min_score]
        if not pairs:
            # 全部被过滤说明评分异常——降级
            return papers, [5] * len(papers)

        filtered_papers = [p for p, _ in pairs]
        filtered_scores = [s for _, s in pairs]
        return filtered_papers, filtered_scores
