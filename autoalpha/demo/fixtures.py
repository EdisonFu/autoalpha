"""演示数据。形状与真实运行一致，内容固定。"""

import math

DEMO_BANNER = ("演示模式：以下结果为预置样例，用于展示流程，"
               "不是真实研究结论。配置 LLM_API_KEY 后可跑真实流水线。")

DEMO_INTENT = {
    "canonical_en": "cross-sectional momentum in Chinese A-share ETFs",
    "aliases": ["Chinese A-share ETF momentum", "ETF rotation strategy",
                "relative strength momentum across ETFs"],
    "market": "china_a", "market_label": "A股", "asset_class": "etf",
    "factor_family": "momentum", "horizon": "daily",
    "doc_preference": "both", "confidence": 0.86,
    "restated_zh": "在A股ETF之间，基于截面动量（相对强弱）进行周期性轮动配置的因子研究",
}

DEMO_PAPERS = [
    {"idx": 0,
     "title": "Transfer Ranking in Finance: Applications to Cross-Sectional Momentum",
     "url": "https://arxiv.org/abs/2302.xxxxx", "date": "2023-02-21",
     "source": "arxiv", "venue": "arXiv", "doc_type": "preprint",
     "citations": 18,
     "abstract": "We study transfer ranking for cross-sectional momentum...",
     "zh_summary": "提出将迁移排序(transfer ranking)用于截面动量，"
                   "通过跨市场共享排序模型缓解单一市场样本不足的问题。",
     "zh_relevance": "对构建A股ETF截面动量因子有直接参考价值，排序学习框架可复用。"},
    {"idx": 1, "title": "Refining ETF Asset Momentum Strategy",
     "url": "https://doi.org/10.2139/ssrn.xxxxx", "date": "2025-06-01",
     "source": "crossref", "venue": "SSRN", "doc_type": "working_paper",
     "citations": 3, "abstract_backfilled": True,
     "abstract": "We refine ETF asset momentum with volatility scaling...",
     "zh_summary": "针对ETF资产动量策略提出改进：引入波动率调整与换手约束，"
                   "降低实盘摩擦成本。",
     "zh_relevance": "实务向，直接讨论换手与成本，对落地性评估很有用。"},
    {"idx": 2, "title": "Factor Momentum and the Momentum Factor",
     "url": "https://www.nber.org/papers/wxxxxx", "date": "2019-01-01",
     "source": "openalex", "venue": "National Bureau of Economic Research",
     "doc_type": "report", "citations": 142, "also_in": ["crossref"],
     "abstract": "Momentum in individual stock returns emanates from "
                 "momentum in factor returns...",
     "zh_summary": "个股动量源于因子收益本身的动量，多数因子的收益序列存在正自相关。",
     "zh_relevance": "揭示动量的底层驱动机制，为因子轮动类信号提供理论基础。"},
]

DEMO_SPECS = [
    {"idx": 0, "factor_name": "etf_cross_sectional_momentum",
     "description": "Ranks ETFs by trailing 60-day return",
     "hypothesis": "Momentum persists due to investor underreaction",
     "formula": "rank(close_t / close_{t-60} - 1)",
     "parameters": {"lookback": 60},
     "zh_what": "衡量每只ETF过去60个交易日的涨跌幅，并在截面上排序。",
     "zh_why": "投资者对信息反应不足与羊群效应使强势资产延续，形成动量。",
     "zh_how": "计算60日收益率，做3倍标准差winsorize去极值，再按日做截面rank归一化。",
     # 这篇是 arXiv 预印本，读到了全文，窗口长度确实写在正文里
     "provenance": "fulltext",
     "provenance_note": "已读取 arXiv 全文（8438 词，取方法论相关章节 12000 字符）",
     "param_sources": {"lookback": "paper"}},
    {"idx": 1, "factor_name": "etf_volume_weighted_momentum",
     "description": "Volume-weighted momentum",
     "hypothesis": "Price moves backed by volume are more likely to persist",
     "formula": "rank(sum(r_t * v_t) / sum(v_t))",
     "parameters": {"lookback": 60},
     "zh_what": "以成交量为权重的动量，强调有量配合的价格变动。",
     "zh_why": "带量上涨比缩量上涨更可能延续，成交量提供信号确认。",
     "zh_how": "对60日窗口内的日收益按成交额加权求和，再做截面rank。",
     # 这篇只有期刊页，正文在登录墙后 —— 窗口是模型自己定的
     "provenance": "abstract",
     "provenance_note": "非 arXiv 来源，正文不可获取（出版商登录墙），仅依据摘要",
     "param_sources": {"lookback": "default"}},
]

DEMO_DATA = {
    "n_symbols": 92, "n_days": 1003,
    "is_range": ["2022-01-04", "2025-02-14"],
    "oos_range": ["2025-03-17", "2026-02-27"],
    "is_days": 752, "oos_days": 231, "universe_width": 92.0,
    "ic_mean_floor": 0.0114, "t_threshold": 1.645,
    "preset": "equity", "n_total": 117, "dist_kept": {"equity": 92},
    "excluded": ["bond", "commodity", "gold", "money", "oil", "reit"],
}


def _nav(seed, n=180, drift=0.0004, vol=0.009):
    """
    带漂移的随机游走。固定种子保证可复现。

    原来用 math.sin 生成，画出来是规整的正弦波 —— 一眼就看得出是假数据，
    而这是新用户看到的第一张图。随机游走更像真实净值。
    """
    import random
    rng = random.Random(int(seed * 1000))
    out, v = [], 1.0
    for i in range(n):
        v *= 1 + drift + rng.gauss(0, vol)
        out.append({"d": f"2024-{1 + i // 30:02d}-{1 + i % 28:02d}",
                    "v": round(max(v, 0.05), 5)})
    return out


def _mk_ic(t, n_obs, win, cov=0.9998):
    return {"ic_mean_1d": round(t / 70, 4), "icir_daily": round(t / 28, 3),
            "icir_annual": round(t / 28 * 15.87, 2), "ic_tstat": t,
            "n_obs": n_obs, "monthly_win_rate": win, "n_months": n_obs // 21,
            "ic_by_lag": {}, "ic_halflife": 20.0, "coverage": cov}


def _mk_layers(mono, shp, net, turn, layers):
    return {"n_quantiles": 5, "layer_ann_returns": layers,
            "monotonicity": mono,
            "long_short": {"ann_return": round(shp * 0.22, 4),
                           "ann_return_net": round(net * 0.22, 4),
                           "sharpe": shp, "sharpe_net": net,
                           "max_drawdown": -0.11, "calmar": 0.8,
                           "turnover_daily": turn},
            "long_only": None}


def _eval(name, status, zh, t, mono, shp, net, turn, layers, nav,
          reason="", decay=None):
    """
    样本外指标按 decay 从样本内推导，而不是直接复用同一组数字。

    曾让 oos_ic 直接指向 is_ic，于是界面上两栏完全一样 ——
    看起来像根本没跑样本外，而样本内外对照正是这个项目的重点。
    OOS 期更短（231 日 vs 752 日），t 值与月数也应随之不同。
    """
    is_ic = _mk_ic(t, 752, 0.63)
    is_lm = _mk_layers(mono, shp, net, turn, layers)

    oos_ic = oos_lm = None
    if status == "PASS":
        d = decay if decay is not None else -0.15
        # OOS 样本更短，t 值按 sqrt(n) 缩放后再乘衰减
        t_oos = round(t * (231 / 752) ** 0.5 * (1 + d) / (1 + d * 0.5), 2)
        oos_ic = _mk_ic(t_oos, 231, round(0.63 * (1 + d * 0.3), 2))
        oos_lm = _mk_layers(
            round(mono * (1 + d * 0.4), 2),
            round(shp * (1 + d), 3), round(net * (1 + d), 3), turn,
            [round(x * (1 + d), 4) for x in layers])

    return {"factor_name": name, "status": status, "fail_reason": reason,
            "description": "", "zh_what": zh, "is_ic": is_ic,
            "is_layers": is_lm, "oos_ic": oos_ic, "oos_layers": oos_lm,
            "sharpe_decay": decay, "max_nav_corr": 0.21,
            "universe_width": 92.0, "nav": nav}


DEMO_EVALS = [
    _eval("etf_cross_sectional_momentum", "PASS",
          "按60日收益在截面上排序，捕捉ETF间的相对强弱。",
          3.42, 0.90, 0.81, 0.68, 0.11,
          [-0.064, -0.012, 0.021, 0.055, 0.118], _nav(0.3), decay=-0.12),
    # 这条是本项目存在的理由：IC 为正、月胜率 66%，按 IC/ICIR 会判「有效」，
    # 但最高分位反而最差，多空组合恰好错过赚钱的层。
    _eval("etf_reversal_overlay", "FAIL_IS",
          "短期反转叠加：IC 为正但分层不单调，最高分位反而最差。",
          2.10, -0.30, -0.12, -0.21, 0.19,
          [0.054, 0.026, -0.023, 0.075, -0.063], _nav(2.1, drift=-0.0002),
          reason="单调性=-0.300 < 0.6; 多空Sharpe=-0.120 < 0.5 "
                 "（IC 显著但分层不单调——这正是只看 IC 会漏掉的情况）"),
    _eval("etf_volume_weighted_momentum", "FAIL_IS",
          "以成交额加权的动量，强调有量配合的价格变动。",
          1.58, 0.60, 0.37, 0.18, 0.31,
          [-0.083, -0.066, -0.072, 0.008, 0.073], _nav(1.7, drift=0.0001),
          reason="|t|=1.58 < 1.96; 换手=0.313 > 0.30"),
]

# 延伸变体：围绕通过门控的基础因子展开。
# 刻意包含一个「延伸后反而变差」的例子 —— 延伸不是免费的，
# 只展示改进会让人误以为变体总能提升。
DEMO_EXTENSIONS = [
    {"base": "etf_cross_sectional_momentum",
     "factor_name": "etf_momentum_vol_scaled",
     "extension_type": "formula_variation",
     "zh_what": "在原动量上按标的波动率缩放，削弱高波品种的权重主导。",
     "rationale": "等权动量会被高波标的主导，波动率缩放让信号更均衡。"},
    {"base": "etf_cross_sectional_momentum",
     "factor_name": "etf_momentum_250d",
     "extension_type": "param_variation",
     "zh_what": "把回看窗口从 60 日延长到 250 日，捕捉更长周期的趋势。",
     "rationale": "长短窗口反映不同投资者的持有期，经济含义不同而非单纯调参。"},
    {"base": "etf_cross_sectional_momentum",
     "factor_name": "etf_momentum_x_lowvol",
     "extension_type": "combination",
     "zh_what": "与低波因子正交组合，取动量强且波动低的标的。",
     "rationale": "动量与低波历史相关性低，组合可降低单一信号的噪声。"},
]

DEMO_EXT_EVALS = [
    _eval("etf_momentum_vol_scaled", "PASS",
          "按波动率缩放后的动量，削弱高波品种的权重主导。",
          3.88, 1.00, 0.94, 0.82, 0.10,
          [-0.071, -0.008, 0.029, 0.067, 0.134], _nav(5.1, drift=0.0006),
          decay=-0.08),
    _eval("etf_momentum_250d", "PASS",
          "回看窗口延长到 250 日的动量。",
          2.96, 0.80, 0.63, 0.55, 0.06,
          [-0.048, 0.004, 0.018, 0.042, 0.096], _nav(6.3, drift=0.00045),
          decay=-0.21),
    # 延伸也会失败：组合后换手升高、扣费后反而不如基础因子
    _eval("etf_momentum_x_lowvol", "FAIL_IS",
          "与低波因子正交组合，取动量强且波动低的标的。",
          2.04, 0.40, 0.44, 0.19, 0.28,
          [-0.032, 0.011, -0.006, 0.038, 0.051], _nav(7.7, drift=0.0002),
          reason="单调性=0.400 < 0.6; 多空Sharpe=0.440 < 0.5"
                 "（组合提高了换手，扣费后不如基础因子）"),
]

DEMO_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
              "total_tokens": 0, "cost_usd": 0.0, "priced": False,
              "model": "(demo)", "by_purpose": {}}


# 反思与下一轮。演示必须覆盖这一步 —— auto research 的主循环在这里，
# 只演示线性流程会让用户以为项目跑完一轮就结束。
DEMO_PROPOSAL = {
    "learned": "截面动量在A股ETF上有效但换手偏高；短期反转因子IC虽显著，"
               "分层单调性却为负，最高分位反而最差，说明IC与可交易性脱节。",
    "next_direction_zh": "低换手的中长周期动量：用波动率缩放与更长回看窗口"
                         "降低换手，同时保持截面单调性",
    "canonical_en": "low turnover momentum volatility scaling ETF",
    "avoid": ["依赖宏观或基本面序列的信号（本数据集只有行情）",
              "仅凭IC/ICIR筛选因子，不看分层单调性",
              "短窗口高频信号，换手会超标"],
    "rationale": "本轮唯一的通过因子受换手约束，而失败因子暴露的是"
                 "「IC显著但不可交易」这一结构性问题，因此下一轮应直接"
                 "优化换手与单调性，而非继续堆叠同类动量变体。",
}

DEMO_ROUNDS = [
    {"round": 1, "query": "cross-sectional momentum exchange-traded fund",
     "n_specs": 2, "n_evals": 6, "n_pass": 3,
     "failures": [{"factor": "etf_reversal_overlay", "status": "FAIL_IS",
                   "reason": "单调性=-0.30，最高分位最差"},
                  {"factor": "etf_volume_weighted_momentum",
                   "status": "FAIL_IS", "reason": "多空Sharpe=0.31 < 0.5"}]},
]
