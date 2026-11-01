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
     "citations": 3,
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
     "zh_how": "计算60日收益率，做3倍标准差winsorize去极值，再按日做截面rank归一化。"},
    {"idx": 1, "factor_name": "etf_volume_weighted_momentum",
     "description": "Volume-weighted momentum",
     "hypothesis": "Price moves backed by volume are more likely to persist",
     "formula": "rank(sum(r_t * v_t) / sum(v_t))",
     "parameters": {"lookback": 60},
     "zh_what": "以成交量为权重的动量，强调有量配合的价格变动。",
     "zh_why": "带量上涨比缩量上涨更可能延续，成交量提供信号确认。",
     "zh_how": "对60日窗口内的日收益按成交额加权求和，再做截面rank。"},
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


def _eval(name, status, zh, t, mono, shp, net, turn, layers, nav,
          reason="", decay=None):
    ic = {"ic_mean_1d": round(t / 70, 4), "icir_daily": round(t / 28, 3),
          "icir_annual": round(t / 28 * 15.87, 2), "ic_tstat": t,
          "n_obs": 752, "monthly_win_rate": 0.63, "n_months": 36,
          "ic_by_lag": {}, "ic_halflife": 20.0, "coverage": 0.9998}
    lm = {"n_quantiles": 5, "layer_ann_returns": layers, "monotonicity": mono,
          "long_short": {"ann_return": round(shp * 0.22, 4),
                         "ann_return_net": round(net * 0.22, 4),
                         "sharpe": shp, "sharpe_net": net,
                         "max_drawdown": -0.11, "calmar": 0.8,
                         "turnover_daily": turn},
          "long_only": None}
    return {"factor_name": name, "status": status, "fail_reason": reason,
            "description": "", "zh_what": zh, "is_ic": ic, "is_layers": lm,
            "oos_ic": ic if status == "PASS" else None,
            "oos_layers": lm if status == "PASS" else None,
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
