#!/usr/bin/env python3
"""
截取各阶段页面，供 README 使用。

页面状态用事件注入驱动（与 tests/test_e2e_ui.py 同一手法），
不依赖真实 LLM 调用——这样截图可重复生成，不会因为某次检索
召回不同而每次长得不一样。

用法：
    python scripts/capture_screens.py --port 8765
"""

from __future__ import annotations

import argparse
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "screenshots")

# 样例数据：取自真实运行的结果形状，但固定下来以保证截图稳定
INTENT = {
    "canonical_en": "cross-sectional momentum in Chinese A-share ETFs",
    "aliases": ["Chinese A-share ETF momentum", "ETF rotation strategy",
                "relative strength momentum across ETFs"],
    "market": "china_a", "market_label": "A股", "asset_class": "etf",
    "factor_family": "momentum", "horizon": "daily",
    "doc_preference": "both", "confidence": 0.86,
    "restated_zh": "在A股ETF之间，基于截面动量（相对强弱）进行周期性轮动配置的因子研究",
}

PAPERS = [
    {"idx": 0, "title": "Transfer Ranking in Finance: Applications to Cross-Sectional Momentum",
     "url": "https://arxiv.org/abs/2302.xxxxx", "date": "2023-02-21",
     "source": "arxiv", "venue": "arXiv", "doc_type": "preprint", "citations": 18,
     "zh_summary": "提出将迁移排序(transfer ranking)用于截面动量，通过跨市场共享排序模型缓解单一市场样本不足的问题。",
     "zh_relevance": "对构建A股ETF截面动量因子有直接参考价值，排序学习框架可复用。"},
    {"idx": 1, "title": "Refining ETF Asset Momentum Strategy",
     "url": "https://doi.org/10.2139/ssrn.xxxxx", "date": "2025-06-01",
     "source": "crossref", "venue": "SSRN", "doc_type": "working_paper", "citations": 3,
     "zh_summary": "针对ETF资产动量策略提出改进：引入波动率调整与换手约束，降低实盘摩擦成本。",
     "zh_relevance": "实务向，directly讨论换手与成本，对落地性评估很有用。"},
    {"idx": 2, "title": "Factor Momentum and the Momentum Factor",
     "url": "https://www.nber.org/papers/wxxxxx", "date": "2019-01-01",
     "source": "openalex", "venue": "National Bureau of Economic Research",
     "doc_type": "report", "citations": 142, "also_in": ["crossref"],
     "zh_summary": "个股动量源于因子收益本身的动量，多数因子的收益序列存在正自相关。",
     "zh_relevance": "揭示动量的底层驱动机制，为因子轮动类信号提供理论基础。"},
]

SPECS = [
    {"idx": 0, "factor_name": "etf_cross_sectional_momentum",
     "formula": "rank(close_t / close_{t-60} - 1)",
     "zh_what": "衡量每只ETF过去60个交易日的涨跌幅，并在截面上排序(cross-sectional rank)。",
     "zh_why": "投资者对信息反应不足(underreaction)与羊群效应使强势资产延续，形成动量。",
     "zh_how": "计算60日收益率，做3倍标准差winsorize去极值，再按日做截面rank归一化。"},
    {"idx": 1, "factor_name": "etf_volume_weighted_momentum",
     "formula": "rank(Σ(r_t · v_t) / Σ v_t)",
     "zh_what": "以成交量为权重的动量，强调有量配合的价格变动。",
     "zh_why": "带量上涨比缩量上涨更可能延续，成交量提供信号确认。",
     "zh_how": "对60日窗口内的日收益按成交额加权求和，再做截面rank。"},
]


def _eval(name, status, zh, t, mono, shp, net, turn, layers, nav, reason="",
          decay=None):
    ic = {"ic_mean_1d": round(t / 70, 4), "icir_daily": round(t / 28, 3),
          "icir_annual": round(t / 28 * 15.87, 2), "ic_tstat": t, "n_obs": 752,
          "monthly_win_rate": 0.63, "n_months": 36, "ic_by_lag": {},
          "ic_halflife": 20.0, "coverage": 0.9998}
    lm = {"n_quantiles": 5, "layer_ann_returns": layers, "monotonicity": mono,
          "long_short": {"ann_return": shp * 0.22, "ann_return_net": net * 0.22,
                         "sharpe": shp, "sharpe_net": net,
                         "max_drawdown": -0.11, "calmar": 0.8,
                         "turnover_daily": turn}, "long_only": None}
    return {"factor_name": name, "status": status, "fail_reason": reason,
            "description": "", "zh_what": zh, "is_ic": ic, "is_layers": lm,
            "oos_ic": ic if status == "PASS" else None,
            "oos_layers": lm if status == "PASS" else None,
            "sharpe_decay": decay, "max_nav_corr": 0.21,
            "universe_width": 92.0, "nav": nav}


def _nav(seed, n=180, drift=0.0004):
    import math
    out, v = [], 1.0
    for i in range(n):
        v *= 1 + drift + 0.012 * math.sin(i * 0.7 + seed) * 0.5
        out.append({"d": f"2024-{1 + i // 30:02d}-{1 + i % 28:02d}",
                    "v": round(v, 5)})
    return out


EVALS = [
    _eval("etf_cross_sectional_momentum", "PASS",
          "按60日收益在截面上排序，捕捉ETF间的相对强弱。",
          3.42, 0.90, 0.81, 0.68, 0.11,
          [-0.064, -0.012, 0.021, 0.055, 0.118], _nav(0.3), decay=-0.12),
    _eval("etf_volume_weighted_momentum", "FAIL_IS",
          "以成交额加权的动量，强调有量配合的价格变动。",
          1.58, 0.60, 0.37, 0.18, 0.31,
          [-0.083, -0.066, -0.072, 0.008, 0.073], _nav(1.7, drift=0.0001),
          reason="|t|=1.58 < 1.96; 换手=0.313 > 0.30"),
]

DATA = {"n_symbols": 92, "n_days": 1003,
        "is_range": ["2022-01-04", "2025-02-14"],
        "oos_range": ["2025-03-17", "2026-02-27"],
        "is_days": 752, "oos_days": 231, "universe_width": 92.0,
        "ic_mean_floor": 0.0114, "t_threshold": 1.645,
        "preset": "equity", "n_total": 117, "dist_kept": {"equity": 92},
        "excluded": ["bond", "commodity", "gold", "money", "oil", "reit"]}

USAGE = {"calls": 16, "input_tokens": 29040, "output_tokens": 7739,
         "total_tokens": 36779, "cost_usd": 0.2032, "priced": True,
         "by_purpose": {"score": {"input": 22884, "output": 440},
                        "codegen": {"input": 2547, "output": 3244},
                        "translate_paper": {"input": 1770, "output": 1943}}}

FEEDS = [
    {"title": "Shorting Leveraged ETFs: The Role of Borrowing Costs",
     "source_label": "Quantpedia", "published": "2026-10-23",
     "url": "https://quantpedia.com/x", "summary": "考察做空杠杆ETF的收益来源，指出借券成本与再平衡效应共同决定策略盈亏。"},
    {"title": "Factor Momentum and the Momentum Factor",
     "source_label": "NBER", "published": "2026-10-11",
     "url": "https://nber.org/x", "summary": "个股动量可分解为底层因子动量之和。"},
    {"title": "Diversification All the Way Down",
     "source_label": "AlphaArchitect", "published": "2026-10-08",
     "url": "https://alphaarchitect.com/x", "summary": "从资产、因子到个券三个层级检验分散化的边际收益。"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--width", type=int, default=1340)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    os.makedirs(OUT, exist_ok=True)

    from playwright.sync_api import sync_playwright

    def launch(p):
        for kw in ({"channel": "chrome"}, {"channel": "msedge"}, {}):
            try:
                return p.chromium.launch(**kw)
            except Exception:
                continue
        raise RuntimeError("没有可用浏览器")

    shots = []
    with sync_playwright() as p:
        br = launch(p)
        pg = br.new_page(viewport={"width": a.width, "height": 1000},
                         device_scale_factor=2)
        pg.goto(base, wait_until="networkidle")

        def snap(name, full=True):
            path = os.path.join(OUT, f"{name}.png")
            pg.screenshot(path=path, full_page=full)
            shots.append(name)
            print(f"  ✓ {name}.png", flush=True)

        # 1 首页
        pg.fill("#idea", "A股ETF截面动量轮动")
        snap("01-home")

        # 2 LLM 配置
        pg.click("text=LLM 配置")
        pg.wait_for_selector("#cfgMask.show")
        pg.wait_for_timeout(350)
        snap("02-llm-config", full=False)
        pg.keyboard.press("Escape")

        # 3 数据源
        pg.click("text=数据源")
        pg.wait_for_selector("#dataMask.show")
        pg.wait_for_timeout(450)
        snap("03-data-sources", full=False)
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(200)

        # 4 Step1 意图识别
        pg.evaluate("""(i) => {
            handle({kind:'stage_start',stage:'intent',message:'识别研究意图',payload:{}});
            handle({kind:'data',stage:'intent',message:i.restated_zh,payload:{intent:i}});
            handle({kind:'stage_end',stage:'intent',message:'意图识别完成',payload:{}});
            handle({kind:'await_input',stage:'intent',message:'确认研究意图后开始检索',
                    payload:{next:'search'}});
        }""", INTENT)
        pg.wait_for_timeout(350)
        snap("04-step1-intent")

        # 5 Step2 多源检索
        pg.evaluate("""(d) => {
            handle({kind:'stage_start',stage:'search',message:'多源检索',payload:{}});
            handle({kind:'data',stage:'search',message:'找到 3 篇论文',
                payload:{page:1,total:12,has_more:true,papers:d}});
            handle({kind:'stage_end',stage:'search',message:'检索完成',payload:{}});
            handle({kind:'await_input',stage:'select',message:'请选择要研究的论文',
                payload:{next:'select'}});
        }""", PAPERS)
        pg.wait_for_timeout(350)
        snap("05-step2-papers")

        # 6 数据准备
        pg.evaluate("""(d) => {
            handle({kind:'stage_start',stage:'data',message:'准备行情数据',payload:{}});
            handle({kind:'data',stage:'data',message:'数据准备就绪',payload:d});
            handle({kind:'stage_end',stage:'data',message:'数据就绪',payload:{}});
        }""", DATA)
        pg.wait_for_timeout(300)
        snap("06-step3-data")

        # 7 因子假设
        pg.evaluate("""(d) => {
            handle({kind:'stage_start',stage:'extract',message:'提取因子假设',payload:{}});
            handle({kind:'data',stage:'extract',message:'提取到 2 个因子假设',
                payload:{append:false,specs:d}});
            handle({kind:'stage_end',stage:'extract',message:'提取完成',payload:{}});
            handle({kind:'await_input',stage:'implement',message:'请选择要实现的因子假设',
                payload:{next:'run'}});
        }""", SPECS)
        pg.wait_for_timeout(350)
        snap("07-step4-hypotheses")

        # 8 回测结果
        pg.evaluate("""(a) => {
            handle({kind:'stage_start',stage:'implement',message:'实现并评估',payload:{}});
            a.evals.forEach(e => handle({kind:'data',stage:'evaluate',
                message:e.factor_name, payload:{evaluation:e}}));
            handle({kind:'stage_end',stage:'evaluate',
                message:'完成：1/2 通过双层门控',payload:{usage:a.usage}});
            handle({kind:'stage_end',stage:'done',message:'本轮完成',
                payload:{can_extend:true}});
        }""", {"evals": EVALS, "usage": USAGE})
        pg.wait_for_timeout(500)
        snap("08-step5-backtest")

        # 9 灵感池
        pg.evaluate("(items) => renderFeeds(items, {total: 65})", FEEDS)
        pg.wait_for_timeout(300)
        snap("09-feeds")

        br.close()

    print(f"\n共 {len(shots)} 张 → {OUT}")


if __name__ == "__main__":
    main()
