import json
import math
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

# 每个检索源各自的召回上限。多源并联后总量约为其 2-3 倍，
# 再经 _SCORE_LIMIT 截断，所以 LLM 打分调用数不随源数量增长。
_PER_SOURCE = 25

# 语义过滤时每批论文数（含摘要，控制 token 量）
_FILTER_BATCH = 20

# 实际送去 LLM 打分的论文数上限。
# arXiv 已按 relevance 排序，深尾部分极少进入最终展示（默认只展示 10 篇），
# 对 200 篇全部打分是纯浪费：实测 score 阶段 10 次调用吃掉 22.9k 输入 token，
# 占检索阶段总成本的 84.5%。截到 60 篇可省约 70%，对召回几无影响。
# 需要更宽召回时调大此值。
_SCORE_LIMIT = 60

# 排序权重。相关性始终是主导项（实测 0.4 的时间权重会压倒它，见 _combined_sort）。
_REL_W     = 0.45    # LLM 语义相关性
_REPRO_W   = 0.20    # 可复现性：能不能从这篇文献里读出可计算的因子
_RECENCY_W = 0.10    # 时效（半衰期衰减）
_DOCTYPE_W = 0.15    # 文献类型
_CITE_W    = 0.10    # 引用量（log 压缩）

# 文献类型权重。研报与 SSRN 工作论文给到接近期刊的权重：对「这个策略
# 实盘能不能用」这类问题，它们往往比纯理论论文更有信息量。
_DOC_WEIGHT = {
    "paper":         1.00,   # 同行评审期刊
    "report":        0.92,   # 机构研报（NBER / Econstor 等）
    "working_paper": 0.88,   # SSRN 工作论文
    "preprint":      0.72,
    "thesis":        0.60,
    "book":          0.50,   # 正常会被 drop_non_research 滤掉，留作兜底
}
# 时间项半衰期：3 年前的论文时间得分 0.5，而非线性归一化的近 0
_RECENCY_HALFLIFE_DAYS = 365 * 3
# 综述/教程类标题。这类文献讲「这个领域有什么」，不讲「怎么算」，
# 对复现因子没有直接价值。实测被顶到第一的 Origin and Evolution of
# Momentum: A Literature Review 就是典型 —— 提取出的 4 个因子全部未过门控。
_REVIEW_TITLE = re.compile(
    r"literature review|a survey|survey of|review of|a review|"
    r"systematic review|tutorial|a primer|introduction to|"
    r"state of the art|bibliometric|meta-analysis|"
    r"综述|研究进展|文献回顾", re.I)

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

    def __init__(self, llm: LLMClient, local_library_dir: str = None,
                 enabled_sources=None):
        self.llm = llm
        self.local_library_dir = local_library_dir
        self.enabled_sources = enabled_sources      # None = 全部默认源
        self.intent = None                          # Step 1 的产物，由编排器注入
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
          多源并联召回 → 跨源去重 → LLM 语义打分 → 过滤低分 → 综合排序

        意图（Step 1）若已由调用方提供则直接复用，否则就地识别一次。
        """
        from ..sources import build_sources, dedupe
        from ..sources.backfill import backfill_abstracts, drop_non_research

        intent = self.intent or self._recognize(idea)

        papers: List[Dict] = []
        for src in build_sources(self.enabled_sources):
            got = src._safe_search(intent, _PER_SOURCE)
            if got:
                print(f"[PaperSearch] {src.name}: {len(got)} 条")
            papers.extend(got)

        if not papers:
            return []

        before = len(papers)
        papers = dedupe(papers)
        print(f"[PaperSearch] 多源合计 {before} 条 → 去重后 {len(papers)} 条")

        # 补摘要、滤非论文 —— 都在打分之前做：
        #   缺摘要的条目只凭标题打分，几乎必然被丢掉（SSRN 工作论文尤其吃亏）；
        #   OECD 图表和书籍章节占着 _SCORE_LIMIT 配额，还要花 token 评判。
        # 实测回填 46%（28 → 11 条仍缺），滤除 6 条，共约 4 秒。
        backfill_abstracts(papers)
        papers = drop_non_research(papers)

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
    def _reproducibility(p: Dict) -> float:
        """
        这篇文献能不能支撑「复现一个因子」，0~1。

        为什么需要这个维度：本项目的目标是复现因子，不是了解领域。
        一次真实运行里，相关性排序把 *Origin and Evolution of Momentum:
        A Literature Review* 顶到第一，系统拿它提取出 4 个因子、
        全部未通过门控 —— 综述根本不含回看窗口、分位切法这些细节，
        提取出的参数只能由模型自己编。

        三个信号，按可靠性排序：
          1. 是否 arXiv。实测只有 arXiv 能拿到全文（其余源正文在登录墙后，
             Unpaywall 给的 PDF 链接 6/6 是绕回 doi.org 的假地址）。
             全文命中 4-6 项复现细节，摘要只有 0-1 项。
          2. 标题是否综述/教程。这类文献结构上就不含可复现方法。
          3. 摘要里已经出现多少复现细节（分位数、回看窗口、调仓、样本期）。
        """
        from ..sources.fulltext import is_arxiv, detail_hits

        score = 0.0
        if is_arxiv(p.get("pdf_url", "")):
            score += 0.5

        title = (p.get("title") or "").lower()
        if _REVIEW_TITLE.search(title):
            score -= 0.6

        n = len(detail_hits(p.get("abstract") or ""))
        score += min(n, 4) * 0.125

        return max(0.0, min(1.0, score))

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

            # 文献类型权重：同等相关度下，经同行评审的期刊论文比预印本更可信，
            # 而 SSRN 工作论文与机构研报对「能不能落地」这类问题信息量更高。
            # 缺省按 paper 处理：未标类型不等于低质量，取 0.5 会无故惩罚
            dw = _DOC_WEIGHT.get(p.get("doc_type") or "paper", 1.00)
            # 引用量做 log 压缩，避免老论文仅靠年头久而垄断前排
            cites = max(0, int(p.get("citations") or 0))
            cw = min(1.0, math.log1p(cites) / math.log1p(200))

            rp = PaperSearchAgent._reproducibility(p)
            combined = (_REL_W * rel_norm + _REPRO_W * rp
                        + _RECENCY_W * recency
                        + _DOCTYPE_W * dw + _CITE_W * cw)
            p = dict(p)
            p["_relevance_score"] = s
            p["_recency"] = round(recency, 4)
            p["_reproducibility"] = round(rp, 4)
            p["_combined_score"] = round(combined, 4)
            ranked.append(p)

        # 同分时相关性优先，使时间项只在真正平手时起 tie-break 作用
        ranked.sort(key=lambda x: (x["_combined_score"], x["_relevance_score"]),
                    reverse=True)
        return ranked

    def _recognize(self, idea: str):
        """调用方没给意图时就地识别一次（CLI 路径会走到这里）。"""
        from .intent_agent import IntentAgent
        return IntentAgent(self.llm).recognize(idea)

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
