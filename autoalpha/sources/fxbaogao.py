"""
发现报告（fxbaogao）：中文券商研报检索源。

**为什么值得接**：实测它是唯一同时满足「真正支持关键词搜索」与「命中即
对口」的中文研报源。搜「指数增强」前几名就是《申万宏源金工：因子分域提升
指数增强组合收益研究》《广发金工机器学习选股手册》，全部是券商金融工程
团队的正式研报 —— 对「复现因子」这个目标，这是比社区讨论密度高得多的资料。

对比此前测过的中文源：
    东方财富「策略」类    0.1% 量化（2000 条里 3 条），且无关键词接口
    知网 / 百度学术       反爬（指纹 / 安全验证）
    聚宽社区              52% 对口但只能抓入库本地筛，列表接口不支持关键词
    发现报告              真搜索、免认证、直接命中金工研报

**已知限制**（都要在实现里体现，不能假装不存在）：
  - `pdfPath` 指向的 PDF 需要登录/签名，直接下载 404。所以拿不到全文，
    摘要用接口返回的 `paragraphObjs` —— 那是报告的语音转写，带语气词，
    质量低于正文，但用于检索排序与 LLM 打分足够。
  - `count` 封顶 10000，且高热词（「动量」）会混入「量价齐升」这类
    字面命中的非量化研报，靠 LLM 打分兜底。
  - 标题带 `<em>` 高亮标签，必须清理。
  - robots.txt 只禁 `/pdf /user /page2 /view`，API 域未禁；只读公开
    searchNoAuth 接口，不下载 PDF、不镜像内容。
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Dict, List

from .base import PaperSource
from .zh_terms import zh_keywords

API = "https://api.fxbaogao.com/mofoun/report/searchReport/searchNoAuth"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
TIMEOUT = 30
_PAGE_SIZE = 20

_EM = re.compile(r"</?em>")


def _strip_em(s: str) -> str:
    return _EM.sub("", s or "")


# 机构名常标为「未知机构」（上传者未填），但标题里写着
# 「申万宏源金工」「广发金工」「华泰证券」。从标题补回来，
# 否则用户在界面上看不出这篇是谁写的。
_ORG_RE = re.compile(
    r"(申万宏源|广发|长江|华泰|国泰君安|国信|招商|中信|中信证券|中信建投|"
    r"海通|兴业|光大|银河|方正|中银|开源|东吴|国盛|华西|东北|东方|浙商|"
    r"平安|中泰|天风|国金|西南|东兴|民生|财通|信达|山西|渤海|首创|"
    r"摩根士丹利|高盛|瑞银|摩根大通|野村|瑞穗|汇丰|"
    r"广发期货|中信期货|南华期货|永安期货|浙商期货|"
    r"申万宏源金工|华泰金工|广发金工|长江金工|"
    r"头豹研究院|艾瑞咨询|易观分析|亿欧智库|罗兰贝格|麦肯锡|"
    r"普华永道|德勤|安永|毕马威|OECD|世界银行|国际货币基金组织)")


def _org_from_title(title: str, fallback: str) -> str:
    m = _ORG_RE.search(title or "")
    if m:
        return m.group(1) + ("金工" if "金工" in title[:m.end() + 3] else "")
    return fallback or "券商研报"


def _clean_paragraphs(objs) -> str:
    """语音转写段落拼成摘要。语气词保留 —— LLM 打分不受影响，
    手动清洗反而容易误伤术语。"""
    if not objs or not isinstance(objs, list):
        return ""
    parts = []
    for o in objs:
        c = (o or {}).get("content") or ""
        if c:
            parts.append(c)
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return text[:1500]


class FxBaoGaoSource(PaperSource):
    """中文券商研报（发现报告）。真正支持关键词，无需入库。"""

    name = "fxbaogao"

    def __init__(self):
        self._last_error = ""

    def search(self, intent, limit: int = 20) -> List[Dict]:
        # 券商研报是中文语料，用中文领域词检索；英文检索式在这里命中不了
        kws = zh_keywords(intent)
        if not kws:
            return []
        out: List[Dict] = []
        for kw in kws[:3]:            # 三个词各查一次，够了；太多会拖慢检索
            out.extend(self._query(kw, max(4, limit // 2)))
        return out

    def _query(self, kw: str, n: int) -> List[Dict]:
        payload = {"order": "2", "pdfPage": "-1", "pageNum": 1,
                   "pageSize": min(n, _PAGE_SIZE), "paragraphSize": 3,
                   "keywords": kw}
        req = urllib.request.Request(
            API, data=json.dumps(payload).encode(),
            headers={"User-Agent": UA, "Content-Type": "application/json",
                     "Referer": "https://www.fxbaogao.com/",
                     "Origin": "https://www.fxbaogao.com"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            print(f"[FxBaoGao] 「{kw}」检索失败，跳过该词")
            return []

        res = []
        for it in (data.get("data") or {}).get("dataList") or []:
            doc_id = it.get("docId")
            para = _clean_paragraphs(it.get("paragraphObjs"))
            base = self._mk(
                title=_strip_em(it.get("title") or ""),
                abstract=para,
                # 站内详情页，不暴露需要签名的 pdf 直链
                url=(f"https://www.fxbaogao.com/rp/detail/{doc_id}"
                     if doc_id else ""),
                date=(it.get("pubTimeStr") or "").rstrip("/").replace("/", "-")[:10],
                source="fxbaogao",
                venue=_org_from_title(
                    _strip_em(it.get("title") or ""),
                    _strip_em(it.get("orgName") or "")),
                doc_type="cn_report",
                citations=0,
                doi="",
            )
            # _mk 只定义通用 schema，源特有字段另行附加
            base["is_deep"] = it.get("isDeep")
            base["lang"] = "zh"
            res.append(base)
        return res
