"""
论文全文抓取与方法论抽取。

**为什么需要**：提取因子假设原本只喂标题+摘要。实测召回 79 篇的摘要
中位 833 字符（≈120 词），而复现一个因子需要的 8 项细节——回看窗口、
跳过近月、分位切法、加权方式、调仓频率、交易成本、样本期、评价指标——
在摘要里只命中 0–1 项：

    论文            摘要    全文
    ST-Momentum     1/8  →  6/8   past 12 month / decile / rebalancing daily / 1990-2022
    CrossSecLearn   1/8  →  5/8   decile×19 / rebalanced monthly / 1980-2019
    Equity2Vec      0/8  →  4/8   past 5 day / quintile / 2009-2018

也就是说，不读全文时 LLM 产出的 `parameters` 基本是它自己定的，
而不是从论文里读出来的。这个区别必须对用户可见（见 provenance 字段）。

**覆盖范围**：只做 arXiv。实测过的其它路都不通——
  - Unpaywall 的 `url_for_pdf` 对 6/6 条目填的是绕回 doi.org 的假地址，
    下到可解析全文 0/6；
  - Elsevier / Wiley / T&F 正文在登录墙后，不是抓取技术能解决的。
所以非 arXiv 条目一律走摘要，并在 provenance 里如实标注。

**依赖**：pypdf（404KB），可选。未安装时返回 None 并说明原因，
不抛异常——与 data_config 同样的约定：不可用要如实呈现，不能静默降级。
"""

from __future__ import annotations

import io
import re
import urllib.request
from typing import Dict, List, Optional

CONTACT = "autoalpha@example.com"
UA = f"AutoAlpha/1.0 (research tool; mailto:{CONTACT})"
TIMEOUT = 45
MAX_PDF_BYTES = 30 * 1024 * 1024      # 实测论文 0.9–17MB，30MB 封顶防异常文件

# 喂给 LLM 的全文预算。实测论文 7700–8900 词，整篇塞进去 token 成本
# 翻数倍且大半是 related work / 致谢 / 参考文献。只取方法论相关章节。
MAX_CHARS = 12000


def pypdf_available() -> bool:
    try:
        import pypdf          # noqa: F401
        return True
    except Exception:
        return False


def is_arxiv(url: str) -> bool:
    return "arxiv.org" in (url or "").lower()


def fetch_fulltext(pdf_url: str) -> Optional[str]:
    """
    下载并解析 PDF，返回纯文本。任何一步失败都返回 None（调用方回落摘要）。

    arXiv API 返回的是 http:// 链接，而明文 http 在部分网络下直接超时
    （实测 http 超时 / https 2s 拿到 1.4MB）。sources/arxiv.py 已在入库前
    升到 https，这里再兜一次——这个链接也可能来自别处。
    """
    if not pdf_url:
        return None
    if pdf_url.startswith("http://"):
        pdf_url = "https://" + pdf_url[len("http://"):]

    try:
        import pypdf
    except Exception:
        print("[FullText] 未安装 pypdf，跳过全文解析"
              "（pip install -e \".[fulltext]\"）")
        return None

    try:
        req = urllib.request.Request(pdf_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read(MAX_PDF_BYTES)
    except Exception as e:
        print(f"[FullText] 下载失败 {pdf_url[:60]}: {type(e).__name__}")
        return None

    # 出版商常把挡板页 / 验证页也以 200 返回，魔数是唯一可靠判据
    if raw[:4] != b"%PDF":
        print(f"[FullText] 非 PDF 内容（可能是挡板页）: {pdf_url[:60]}")
        return None

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        print(f"[FullText] 解析失败: {type(e).__name__}: {str(e)[:80]}")
        return None

    if len(text.split()) < 500:
        # 扫描版 PDF 抽不出文字层，拿到的是空壳
        print(f"[FullText] 抽到的文字过少（{len(text.split())} 词），按失败处理")
        return None
    return text


# ── 章节定位 ────────────────────────────────────────────────────────────────

# 要保留的章节。按复现需要排序：方法与数据最重要，结果次之。
# 实测各刊标题风格差异大（`OVERVIEW` / `Momentum Strategies` /
# `Factors Description`），所以关键词取宽，宁可多带一段正文。
_WANTED = (
    r"method|methodolog|approach|model|specification|overview|framework|"
    r"data|sample|dataset|universe|feature|variable|"
    r"strateg|portfolio|signal|factor|construct|implement|setup|"
    r"empirical|experiment|result|performance|evaluat|backtest|discussion"
)
# 明确不要的。related work 占篇幅大且对复现无用，参考文献更是纯噪声。
_UNWANTED = (r"related work|literature review|reference|bibliograph|"
             r"acknowledg|conclusion|future work|introduction|abstract")

# 章节标题通常独占一行、较短、以编号或大写词开头
_HEAD = re.compile(
    r"^\s*(?:(?:\d+|[IVX]+)[.\s]+)?([A-Z][A-Za-z \-&]{3,60})\s*$")

# PDF 会把正文长句折成短行，其中不少「首字母大写、无句末标点」，
# 于是被当成章节标题。实测 Equity2Vec 切出 55 段，里面混着
# 「Increasing rate」「Keeping in mind that stock trends are highly influenced」
# 这类正文碎片，真标题反而被淹没。靠这组启发式剔除：
_STOPWORD_HEAD = re.compile(
    r"^(the|a|an|this|that|these|those|we|our|it|its|in|on|at|for|with|"
    r"by|from|to|as|is|are|was|were|be|been|and|or|but|if|when|while|"
    r"where|which|who|how|increasing|keeping|given|most|newer|using)\b",
    re.I)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not (3 < len(s) < 70):
        return False
    if re.search(r"[.,;:]$", s):
        return False
    if not _HEAD.match(s):
        return False
    body = re.sub(r"^\s*(?:(?:\d+|[IVX]+)[.\s]+)", "", s)
    # 真标题极少以虚词/从句词开头；正文碎片几乎总是
    if _STOPWORD_HEAD.match(body):
        return False
    # 标题词数有限，正文碎片往往更长
    return len(body.split()) <= 7


def split_sections(text: str) -> List[tuple]:
    """粗分章节，返回 [(标题, 正文), ...]。分不出来时返回单段。"""
    lines = text.split("\n")
    sections: List[tuple] = []
    cur_title, cur_body = "", []
    for ln in lines:
        if _is_heading(ln):
            if cur_body:
                sections.append((cur_title, "\n".join(cur_body)))
            cur_title, cur_body = re.sub(
                r"^\s*(?:(?:\d+|[IVX]+)[.\s]+)", "", ln.strip()), []
        else:
            cur_body.append(ln)
    if cur_body:
        sections.append((cur_title, "\n".join(cur_body)))
    return sections


# 复现一个因子真正需要的细节。既用于挑选章节，也用于诚实度自检：
# 命中数会写进 provenance，让用户知道这次读到的全文有多少可用信息。
DETAIL_PATTERNS = {
    "lookback": r"(?i)look[- ]?back|lookback|formation period|past \d+ (?:month|day)",
    "quantile": r"(?i)decile|quintile|tercile|top \d+\s?%",
    "weighting": r"(?i)equal[- ]?weight|value[- ]?weight|cap[- ]?weight",
    "rebalance": r"(?i)rebalanc",
    "cost": r"(?i)transaction cost|slippage|basis points|bps",
    "sample": r"(?i)(?:from|between)\s+(?:19|20)\d{2}\s*(?:to|-|–|until)\s*(?:19|20)\d{2}",
    "metric": r"(?i)sharpe|information ratio|maximum drawdown",
}


def detail_hits(text: str) -> List[str]:
    """文本里命中了哪几项复现细节。"""
    return [k for k, p in DETAIL_PATTERNS.items() if re.search(p, text or "")]


def extract_methodology(text: str, max_chars: int = MAX_CHARS) -> str:
    """
    从全文里挑出对复现有用的部分，控制在 max_chars 内。

    **按细节密度挑，不按出现顺序挑。** 顺序挑会被靠前的长章节吃掉预算：
    实测 CrossSecLearn 的分位切法/交易成本/夏普都在后半篇，顺序截取后
    7 项细节只剩 2 项；按密度挑能保住 5 项。

    分不出章节时退化为截取开头——总比不给强，但那种情况下质量无保证。
    """
    if not text:
        return ""
    sections = split_sections(text)
    if len(sections) < 3:
        return text[:max_chars]

    cands = []
    for i, (title, body) in enumerate(sections):
        t = title.lower()
        if re.search(_UNWANTED, t):
            continue
        body = body.strip()
        if len(body) < 200:          # 太短的多是图表题注
            continue
        titled = bool(re.search(_WANTED, t))
        if not titled and not title:
            continue
        n_detail = len(detail_hits(body))
        # 标题相关性只作次级信号：实测各刊标题风格差异太大，
        # 真正可靠的是正文里有没有复现细节。
        score = n_detail * 10 + (3 if titled else 0)
        cands.append((score, i, title, body))

    if not cands:
        return text[:max_chars]

    cands.sort(key=lambda c: (-c[0], c[1]))
    chosen, total = [], 0
    for score, i, title, body in cands:
        if total >= max_chars:
            break
        chunk = f"\n## {title or '(untitled)'}\n{body}"
        if total + len(chunk) > max_chars:
            chunk = chunk[:max_chars - total]
        chosen.append((i, chunk))
        total += len(chunk)

    # 恢复原文顺序，读起来仍是一篇连贯的方法论
    chosen.sort(key=lambda c: c[0])
    return "".join(c[1] for c in chosen)


def paper_context(paper: Dict) -> Dict:
    """
    为一篇论文准备喂给提取 agent 的上下文。

    返回 {"text": ..., "provenance": "fulltext" | "abstract", "note": ...}。
    provenance 必须一路带到 FactorSpec —— 用户要能分辨「参数读自论文」
    和「参数是模型自己定的」。
    """
    abstract = paper.get("abstract") or ""
    url = paper.get("pdf_url") or ""

    if not is_arxiv(url):
        return {"text": abstract, "provenance": "abstract",
                "note": "非 arXiv 来源，正文不可获取（出版商登录墙），仅依据摘要"}

    if not pypdf_available():
        return {"text": abstract, "provenance": "abstract",
                "note": "未安装 pypdf，未读取全文，仅依据摘要"}

    full = fetch_fulltext(url)
    if not full:
        return {"text": abstract, "provenance": "abstract",
                "note": "全文下载或解析失败，仅依据摘要"}

    method = extract_methodology(full)
    return {"text": method, "provenance": "fulltext",
            "note": f"已读取 arXiv 全文（{len(full.split())} 词，"
                    f"取方法论相关章节 {len(method)} 字符）"}
