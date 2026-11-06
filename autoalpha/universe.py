"""
标的分类与截面池构建。

动机（docs/E2E_REVIEW.md P-3）：
全池 117 只 ETF 的年化波动极差达 200 倍（国债 0.2% vs 原油 39.8%）。
把它们放进同一个截面做动量排序，排序结果被【资产类别】主导而非动量强弱，
这使「分层单调性」这个核心判据失去意义 —— 它测出的可能只是波动率阶梯。

实测只保留股票类后，极差从 200 倍降到 2.2 倍（17.0%~36.7%）。
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

# 分类关键词。顺序敏感：先匹配的优先（如「可转债」归 bond 而非 equity）
_RULES = [
    ("reit",      ["REIT"]),
    ("money",     ["货币", "添益", "现金"]),
    ("bond",      ["国债", "城投债", "可转债", "信用债", "政金债", "债"]),
    ("gold",      ["黄金"]),
    ("oil",       ["原油"]),
    ("commodity", ["商品", "豆粕", "有色", "能源化工"]),
]

# 可比的截面池预设
PRESETS = {
    "equity": ["equity"],                         # 默认：仅股票类，截面最可比
    "equity_reit": ["equity", "reit"],
    "risk_assets": ["equity", "reit", "commodity", "gold", "oil"],
    "all": None,                                   # 全池（不推荐做截面排序）
}

DEFAULT_PRESET = "equity"


def classify(name: str) -> str:
    """按标的名称分类。未匹配任何规则视为股票类。"""
    for cat, kws in _RULES:
        if any(k in name for k in kws):
            return cat
    return "equity"


def load_names(klines_dir: str, freq: str = "etf-daily") -> Dict[str, str]:
    """从文件名解析 {代码: 名称}。文件名形如 shetf_510300_沪深300ETF.csv。"""
    d = os.path.join(klines_dir, freq)
    out: Dict[str, str] = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        m = re.match(r"[a-z]+etf_(\d+)_(.+)\.csv$", f)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def build_universe(data: Dict, names: Dict[str, str],
                   preset: str = DEFAULT_PRESET) -> tuple:
    """
    按预设过滤标的。返回 (过滤后的 data, 画像 dict)。

    画像用于注入 LLM prompt（修 P-2：提取阶段此前完全不知道有什么数据）
    和在 UI 上展示。
    """
    cats = {code: classify(nm) for code, nm in names.items()}
    keep = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])

    if keep is None:
        filtered = dict(data)
    else:
        filtered = {c: df for c, df in data.items()
                    if cats.get(c, "equity") in keep}

    # 分布统计
    dist: Dict[str, int] = {}
    for c in data:
        k = cats.get(c, "equity")
        dist[k] = dist.get(k, 0) + 1
    dist_kept: Dict[str, int] = {}
    for c in filtered:
        k = cats.get(c, "equity")
        dist_kept[k] = dist_kept.get(k, 0) + 1

    profile = {
        "preset": preset,
        "n_total": len(data),
        "n_kept": len(filtered),
        "dist_all": dist,
        "dist_kept": dist_kept,
        "excluded": sorted(set(dist) - set(dist_kept)),
        "sample": [f"{c} {names.get(c, '')}" for c in sorted(filtered)[:8]],
    }
    return filtered, profile


def describe_for_prompt(profile: dict, fields: List[str],
                        date_range: tuple = None) -> str:
    """
    生成注入 LLM prompt 的数据说明（修 P-2）。

    必须明确两件事，否则 LLM 会凭空要 SPY/QQQ/VIX/新闻数据：
      1. 这是什么市场的什么标的
      2. 只有哪些字段可用，缺失数据不许自行替代
    """
    kept = profile.get("dist_kept", {})
    comp = ", ".join(f"{k}×{v}" for k, v in sorted(kept.items(), key=lambda x: -x[1]))
    lines = [
        "=== AVAILABLE DATA (hard constraint) ===",
        "",
        "INSTRUMENT TYPE : Exchange-Traded Funds (ETFs) — baskets, NOT individual stocks",
        "MARKET          : China A-share market (Shanghai + Shenzhen exchanges)",
        "FREQUENCY       : End-of-day daily bars. One row per asset per trading day.",
        f"UNIVERSE        : {profile.get('n_kept', 0)} ETFs — {comp}",
        f"FIELDS          : {', '.join(fields)}",
    ]
    if date_range:
        lines.append(f"DATE RANGE      : {date_range[0]} to {date_range[1]}")
    lines += [
        f"SAMPLE TICKERS  : {'; '.join(profile.get('sample', [])[:5])}",
        "",
        "NOT AVAILABLE — the data does not exist, do not reference it:",
        "",
        "  Wrong market      US tickers (SPY, QQQ, DIA, IWM, TLT), US indices,",
        "                    VIX or any volatility index product",
        "  Wrong granularity intraday / minute / tick bars, order book, bid-ask,",
        "                    quotes, trade-by-trade data — we only have daily OHLCV",
        "  Wrong asset type  single-stock data, fundamentals (earnings, book value,",
        "                    P/E, ROE), shares outstanding, market cap, analyst",
        "                    estimates — ETFs are baskets and we hold no constituents",
        "  Not collected     news text, sentiment scores, social media, interest",
        "                    rates, yield curves, macro series (CPI/GDP)",
        "",
        "ETF-SPECIFIC CAVEATS — signals valid for single stocks often break here:",
        "",
        "  * ETFs have NO price limit (no 涨停/跌停). Conditions like",
        "    'daily return >= 9.9%' almost never fire. A factor measured only on",
        "    rare trigger days cannot support a daily cross-sectional ranking.",
        "  * No fundamentals, so no value/quality/profitability factors.",
        "  * Avoid signals that produce only a handful of discrete values",
        "    (0/1 flags, event markers). After cross-sectional ranking they tie",
        "    massively and carry no information.",
        "  * The factor must vary ACROSS ETFs on EVERY trading day.",
        "",
        "If the paper's signal requires unavailable data, DO NOT invent a proxy.",
        "Either express the signal using only the fields above, or set",
        "computable=false and say why in implementation_notes.",
        "A proxy the user did not ask for silently changes what is measured.",
        "=== END DATA CONSTRAINTS ===",
    ]
    return "\n".join(lines)
