"""
从研究意图提取中文检索词。

**为什么不用分词**：量化领域的术语是有限集合（动量、反转、低波动、
指数增强……），用领域词典做匹配比通用分词更准，还省掉一个依赖。
通用分词会把「截面动量」切成「截面/动量」，把「指数增强」切成
「指数/增强」，反而降低了检索精度。

两个来源：
  1. 用户原始输入与中文复述 —— 保留他真正关心的词
  2. 意图识别出的因子族/资产类别 —— 用户没明说时补上对应中文术语
"""

from __future__ import annotations

from typing import List

# 领域术语表。按长度降序匹配，保证「指数增强」先于「指数」被取到。
CN_TERMS = [
    "指数增强", "行业轮动", "风格轮动", "风险平价", "统计套利", "配对交易",
    "截面动量", "时序动量", "低波动", "小市值", "多因子", "单因子",
    "机器学习", "深度学习", "神经网络", "遗传规划", "强化学习",
    "动量", "反转", "波动率", "波动", "市值", "价值", "成长", "红利",
    "质量", "盈利", "换手率", "换手", "流动性", "择时", "轮动",
    "对冲", "套利", "CTA", "情绪", "北向", "资金流", "量能", "量价",
    "因子", "超额", "alpha", "Alpha", "ALPHA", "回测", "选股",
    "ETF", "etf", "股指期货", "可转债", "融资融券", "新股",
]

_FAMILY_ZH = {
    "momentum": ["动量"],
    "reversal": ["反转"],
    "volatility": ["低波动", "波动率"],
    "value": ["价值"],
    "quality": ["质量", "盈利"],
    "size": ["小市值", "市值"],
    "ml": ["机器学习"],
    "liquidity": ["流动性", "换手率"],
}
_ASSET_ZH = {
    "etf": ["ETF"],
    "equity": ["选股"],
    "bond": ["债券"],
    "commodity": ["商品"],
}


def zh_keywords(intent, limit: int = 6) -> List[str]:
    """
    返回用于中文资料检索的词，按信息量排序（长词优先）。

    三个来源，按可靠度：
      1. intent.canonical_zh —— LLM 在意图识别时产出的中文检索式，
         能覆盖词典收不到的表述（「动量崩溃」「北向资金流」）。
         空格分隔，整串可直接给支持短语的接口（券商研报库），
         拆开的词给 LIKE 匹配的本地库（社区帖）。
      2. 词典匹配 —— 在用户输入上找已收录术语。canonical_zh 缺失时
         这是唯一来源，但它认不出没收录的说法。
      3. 因子族/资产类别映射 —— 用户没明说时补上对应中文术语。

    没有命中时返回空列表 —— 调用方应据此跳过中文源，
    而不是用空词把整库当成「相关」返回。
    """
    found: List[str] = []

    zh = (getattr(intent, "canonical_zh", "") or "").strip()
    if zh:
        # 整串短语优先（给支持短语的源），拆词次之（给 LIKE 匹配）
        if " " in zh:
            found.append(zh)
        for w in zh.split():
            w = w.strip()
            if w and w not in found:
                found.append(w)

    text = " ".join(filter(None, [
        getattr(intent, "raw_input", "") or "",
        getattr(intent, "restated_zh", "") or "",
    ]))
    for term in CN_TERMS:
        if term in text and term not in found:
            found.append(term)

    for family in (_FAMILY_ZH.get(getattr(intent, "factor_family", ""), []) or []):
        if family not in found:
            found.append(family)
    for asset in (_ASSET_ZH.get(getattr(intent, "asset_class", ""), []) or []):
        if asset not in found:
            found.append(asset)

    # 长词信息量大：「指数增强」比「指数」更能定位到想要的内容。
    # canonical_zh 整串始终排最前（信息量最大），其它按长度。
    found.sort(key=lambda t: (0 if t == zh else 1, -len(t), t))
    return found[:limit]
