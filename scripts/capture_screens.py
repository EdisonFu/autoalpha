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
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "screenshots")

# 演示数据直接取自 /api/demo —— 不要在这里另抄一份。
# 曾经两边各存一份，修了 demo/fixtures.py 而截图没变，白排查半天。


def load_fixtures(base):
    """从运行中的服务拉演示数据，保证截图与实际演示一致。"""
    import urllib.request
    with urllib.request.urlopen(f"{base}/api/demo", timeout=20) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--width", type=int, default=1340)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    os.makedirs(OUT, exist_ok=True)

    fx = load_fixtures(base)
    INTENT, PAPERS = fx["intent"], fx["papers"]
    SPECS, DATA = fx["specs"], fx["data"]
    EVALS, USAGE = fx["evals"], fx["usage"]
    FEEDS = [
        {"title": "Shorting Leveraged ETFs: The Role of Borrowing Costs",
         "source_label": "Quantpedia", "published": "2026-10-23",
         "url": "https://quantpedia.com/x",
         "summary": "考察做空杠杆ETF的收益来源，指出借券成本与再平衡效应共同决定策略盈亏。"},
        {"title": "Factor Momentum and the Momentum Factor",
         "source_label": "NBER", "published": "2026-10-11",
         "url": "https://nber.org/x", "summary": "个股动量可分解为底层因子动量之和。"},
        {"title": "Diversification All the Way Down",
         "source_label": "AlphaArchitect", "published": "2026-10-08",
         "url": "https://alphaarchitect.com/x",
         "summary": "从资产、因子到个券三个层级检验分散化的边际收益。"},
    ]

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

        def snap(name, full=True, step=None, panel=None):
            """
            step: 先切到该步骤再截 —— 单屏布局下只显示当前步骤的面板，
                  不切就会截到最后到达的那个（曾把延伸探索截成「因子评估」）。
            panel: 只截这个面板，去掉页头与无关卡片。
            """
            if step:
                pg.evaluate(f"openStep('{step}')")
                pg.wait_for_timeout(450)
            path = os.path.join(OUT, f"{name}.png")
            if panel:
                el = pg.query_selector(panel)
                if el:
                    el.screenshot(path=path)
                else:
                    pg.screenshot(path=path, full_page=full)
            else:
                pg.screenshot(path=path, full_page=full)
            shots.append(name)
            print(f"  ✓ {name}.png", flush=True)

        # 1 首页
        pg.fill("#idea", "A股ETF截面动量轮动")
        snap("01-home", panel=".hero")

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
        snap("04-step1-intent", step="intent", panel="#pIntent")

        # 演示横幅会遮住卡片标题；重查表单若被前面的操作展开也要收起，
        # 否则截图呈现的就不是用户实际看到的默认状态。
        pg.evaluate("""() => {
            document.querySelectorAll('.demobar')
                .forEach(e => e.style.display='none');
            const b=document.querySelector('#researchBox');
            if(b)b.classList.add('hidden');
        }""")

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
        snap("05-step2-papers", step="search", panel="#pPapers")

        # 6 数据准备
        pg.evaluate("""(d) => {
            handle({kind:'stage_start',stage:'data',message:'准备行情数据',payload:{}});
            handle({kind:'data',stage:'data',message:'数据准备就绪',payload:d});
            handle({kind:'stage_end',stage:'data',message:'数据就绪',payload:{}});
        }""", DATA)
        pg.wait_for_timeout(300)
        snap("06-step3-data", step="data", panel="#pData")

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
        snap("07-step4-hypotheses", step="extract", panel="#pSpecs")

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
        snap("08-step5-backtest", step="evaluate", panel="#pEval")

        # 0 概览：页头 + 输入区 + 跑完的步骤条。README 的第一张图，
        #   要让人一眼看懂整条流程走到哪、每步产出了什么。
        #   先补上延伸探索 —— 否则 Step 7 停在未完成而 Step 8 已完成，
        #   步骤条自相矛盾。
        pg.evaluate("""(d) => {
            const exts=d.extensions, res=d.ext_evals||[];
            handle({kind:'stage_start',stage:'extend',message:'生成延伸变体',payload:{}});
            handle({kind:'data',stage:'extend',
                message:`生成 ${exts.length} 个延伸变体`,
                payload:{specs:exts.map((x,i)=>({idx:1000+i,...x}))}});
            res.forEach(e => handle({kind:'data',stage:'evaluate',
                message:e.factor_name,payload:{evaluation:e}}));
            const np=res.filter(e=>e.status==='PASS').length;
            handle({kind:'stage_end',stage:'extend',
                message:`延伸完成：${np}/${res.length} 通过门控`,payload:{}});
            handle({kind:'stage_end',stage:'done',message:'本轮完成',payload:{}});
            document.querySelector('#cardFeeds').classList.add('hidden');
            window.scrollTo(0, 0);
        }""", {"extensions": fx["extensions"], "ext_evals": fx["ext_evals"]})
        pg.wait_for_timeout(400)
        # 从 header 起（不是 .hero）——以 .hero 为上边界会把页头按钮
        # 拦腰截断，看起来像截坏了。
        box = pg.evaluate("""() => {
            const t=document.querySelector('header').getBoundingClientRect();
            const s=document.querySelector('#steps').getBoundingClientRect();
            return {x:Math.max(0,t.x-16), y:Math.max(0,t.y-12),
                    width:t.width+32, height:(s.y+s.height)-t.y+28};
        }""")
        pg.screenshot(path=os.path.join(OUT, "00-overview.png"), clip=box)
        shots.append("00-overview")
        print("  ✓ 00-overview.png", flush=True)

        # 10 反思与下一轮 —— auto research 的主循环，README 需要这张图
        #   才能说明「跑完一轮不是结束」。
        pg.evaluate("""(d) => {
            handle({kind:'await_input',stage:'done',
                message:'是否按此方向开始下一轮研究',
                payload:{next:'next_round',proposal:d.proposal,
                    can_extend:true,round_no:1,max_rounds:4,
                    rounds:d.rounds||[],n_factor_trials:d.n_factor_trials||0}});
        }""", {"proposal": fx["proposal"], "rounds": fx["rounds"],
               "n_factor_trials": fx["n_factor_trials"]})
        pg.wait_for_timeout(450)
        snap("10-reflect", step="done", panel="#pNext")

        # 9 灵感池
        pg.evaluate("""(items) => {
            document.querySelector('#cardFeeds').classList.remove('hidden');
            renderFeeds(items, {total: 65});
        }""", FEEDS)
        pg.wait_for_timeout(350)
        snap("09-feeds", panel="#cardFeeds")

        br.close()

    print(f"\n共 {len(shots)} 张 → {OUT}")


if __name__ == "__main__":
    main()
