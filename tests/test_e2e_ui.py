"""
真实浏览器 E2E 测试。

为什么需要：此前的前端测试只在 Node 里跑逻辑，验证不了布局 ——
数据源弹窗在小视口下内容被裁掉且无法滚动，逻辑测试全绿却漏掉了。
凡是涉及尺寸、滚动、可见性的断言，必须在真实渲染环境里做。

浏览器来源：优先用系统已装的 Chrome（channel="chrome"），
其次用 playwright 自带的 chromium。本环境的网络白名单拦掉了
cdn.playwright.dev（实测 307 且零字节），装不了自带 chromium，
所以系统 Chrome 是主路径。两者都没有时整个文件跳过。
"""

import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("playwright", reason="playwright 未安装")
from playwright.sync_api import sync_playwright  # noqa: E402

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"


def _pick_browser():
    """返回可用的 launch 参数，没有可用浏览器则 None。"""
    for kw in ({"channel": "chrome"}, {"channel": "msedge"}, {}):
        try:
            with sync_playwright() as p:
                p.chromium.launch(**kw).close()
            return kw
        except Exception:
            continue
    return None


LAUNCH = _pick_browser()
pytestmark = pytest.mark.skipif(LAUNCH is None,
                                reason="无可用浏览器（系统 Chrome 或 chromium）")


@pytest.fixture(scope="module")
def server():
    env = {**os.environ, "LLM_MODEL": os.environ.get("LLM_MODEL", "test-model")}
    proc = subprocess.Popen(
        [sys.executable, "-u", "main.py", "--web", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import urllib.request
    for _ in range(40):
        try:
            urllib.request.urlopen(f"{BASE}/api/health", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        pytest.skip("服务未能启动")
    yield BASE
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture
def page(server):
    with sync_playwright() as p:
        br = p.chromium.launch(**LAUNCH)
        pg = br.new_page(viewport={"width": 1280, "height": 720})
        errors = []
        pg.on("pageerror", lambda e: errors.append(f"JS: {e}"))
        # 只收真正的脚本错误。浏览器自动请求 favicon 等资源产生的
        # 网络 404 会混进 console error，与页面代码无关。
        pg.on("console", lambda m: errors.append(m.text)
              if m.type == "error" and "Failed to load resource" not in m.text
              else None)
        pg.goto(server, wait_until="networkidle")
        pg._errors = errors
        yield pg
        br.close()


# ── 基本加载 ────────────────────────────────────────────────────────────────

def test_page_loads_without_js_errors(page):
    assert page.title()
    assert not page._errors, f"控制台报错: {page._errors[:3]}"


def test_step_bar_says_data_preparation(page):
    """阶段名应为「数据准备」而非「数据切分」。"""
    txt = page.inner_text("#steps")
    assert "数据准备" in txt
    assert "数据切分" not in txt


# ── 弹窗滚动（此前的 bug）────────────────────────────────────────────────

def test_data_modal_scrolls_in_small_viewport(page):
    """
    7 个数据源在矮视口下必然超出。内容必须可滚动，
    且弹窗本身不得超出视口 —— 否则底部的源和按钮都够不到。
    """
    page.set_viewport_size({"width": 900, "height": 560})
    page.click("text=数据源")
    page.wait_for_selector("#dataMask.show", timeout=5000)

    # 页面上有两个 .modal（LLM / 数据源），必须限定到当前这个 ——
    # 泛选择器会命中隐藏的那个，bounding_box() 返回 None
    modal = page.query_selector("#dataMask .modal")
    box = modal.bounding_box()
    assert box["height"] <= 560, f"弹窗高 {box['height']} 超出视口"

    body = page.query_selector("#dataMask .mbody")
    assert body, "缺少可滚动区域"
    metrics = page.evaluate(
        """() => {const e=document.querySelector('#dataMask .mbody');
                  return {sh:e.scrollHeight, ch:e.clientHeight};}""")
    assert metrics["sh"] > metrics["ch"], "内容未溢出，测试前提不成立"

    page.evaluate(
        "() => {const e=document.querySelector('#dataMask .mbody');"
        "e.scrollTop = e.scrollHeight;}")
    top = page.evaluate(
        "() => document.querySelector('#dataMask .mbody').scrollTop")
    assert top > 0, "内容溢出却无法滚动"


def test_close_button_reachable_without_scrolling(page):
    """关闭按钮固定在底部，长列表下也要能点到。"""
    page.set_viewport_size({"width": 900, "height": 560})
    page.click("text=数据源")
    page.wait_for_selector("#dataMask.show")
    btn = page.query_selector("#dataMask .modal > .actions button")
    assert btn and btn.is_visible()
    btn.click()
    page.wait_for_selector("#dataMask.show", state="detached", timeout=3000)


def test_llm_modal_also_scrolls(page):
    page.set_viewport_size({"width": 900, "height": 420})
    page.click("text=LLM 配置")
    page.wait_for_selector("#cfgMask.show")
    box = page.query_selector("#cfgMask .modal").bounding_box()
    assert box["height"] <= 420


# ── 内容要求 ───────────────────────────────────────────────────────────────

def test_data_panel_states_public_interface(page):
    """必须讲清：不配置也能用，走公共数据查询接口。"""
    page.click("text=数据源")
    page.wait_for_selector("#dataMask.show")
    txt = page.inner_text("#dataMask")
    assert "公共数据查询接口" in txt
    assert "不配置任何账号也能用" in txt


def test_data_panel_has_directory_field(page):
    page.click("text=数据源")
    page.wait_for_selector("#dataMask.show")
    assert page.query_selector("#dataDir"), "缺少本地目录输入"
    assert "本地数据目录" in page.inner_text("#dataMask")


def test_llm_panel_has_no_price_inputs(page):
    """单价由系统推断，不该让用户填。"""
    page.click("text=LLM 配置")
    page.wait_for_selector("#cfgMask.show")
    assert page.query_selector("#cfgPin") is None
    assert page.query_selector("#cfgPout") is None
    assert page.query_selector("#cfgPrice"), "应展示推断出的单价"


def test_escape_closes_modal(page):
    page.click("text=数据源")
    page.wait_for_selector("#dataMask.show")
    page.keyboard.press("Escape")
    page.wait_for_selector("#dataMask.show", state="detached", timeout=3000)


def test_no_horizontal_overflow(page):
    """窄屏不得出现横向滚动。"""
    page.set_viewport_size({"width": 390, "height": 780})
    page.wait_for_timeout(200)
    over = page.evaluate(
        "() => document.documentElement.scrollWidth > window.innerWidth + 1")
    assert not over, "窄屏出现横向溢出"


# ── 阶段门控（用户报的问题）────────────────────────────────────────────

def test_buttons_gated_by_phase(page):
    """
    进入某阶段后，上游阶段的按钮必须置灰。
    用户报：因子假设已出来，"开始检索"仍可点。
    """
    page.evaluate("""() => {
        // 直接驱动事件，不依赖真实 LLM
        handle({kind:'await_input',stage:'intent',message:'',payload:{next:'search'}});
        INTENT={canonical_en:'x',aliases:[],asset_class:'etf',factor_family:'momentum',
                horizon:'daily',doc_preference:'both',restated_zh:'测试',confidence:0.8};
        showIntent(true);
    }""")
    page.wait_for_timeout(150)
    assert page.is_enabled("[data-act=search]"), "意图确认后应可检索"

    page.evaluate("""() => {
        handle({kind:'data',stage:'search',message:'',payload:{page:1,total:2,has_more:false,
          papers:[{idx:0,title:'P1'},{idx:1,title:'P2'}]}});
        handle({kind:'await_input',stage:'select',message:'',payload:{next:'select'}});
    }""")
    page.wait_for_timeout(150)
    assert page.is_enabled("[data-act=select]"), "检索完成后应可提取"

    page.evaluate("""() => {
        handle({kind:'data',stage:'extract',message:'',payload:{append:false,
          specs:[{idx:0,factor_name:'f1',zh_what:'测试因子'}]}});
        handle({kind:'await_input',stage:'implement',message:'',payload:{next:'run'}});
    }""")
    page.wait_for_timeout(200)

    assert page.is_disabled("[data-act=search]"), "有假设后「开始检索」必须置灰"
    assert page.is_disabled("[data-act=select]"), "有假设后「选论文」必须置灰"
    assert page.is_enabled("[data-act=run]"), "有假设后「实现并回测」应可用"

    title = page.get_attribute("[data-act=search]", "title")
    assert title, "置灰的按钮要说明原因"


def test_progress_bar_advances(page):
    page.evaluate("""() => {
        handle({kind:'stage_start',stage:'intent',message:'',payload:{}});
        handle({kind:'stage_end',stage:'intent',message:'',payload:{}});
    }""")
    page.wait_for_timeout(150)
    w = page.evaluate(
        "() => document.querySelector('.progress .bar').style.width")
    assert w and w != "0%", f"进度条未推进: {w}"


# ── 单屏布局 ────────────────────────────────────────────────────────────

def test_only_current_step_panel_is_visible(page):
    """
    页面高度不应随步骤累积。实测优化前 3 个因子就涨到 5.6 屏。
    每次只显示当前步骤的面板，完成的收进步骤卡。
    """
    page.evaluate("""() => {
        handle({kind:'await_input',stage:'intent',message:'',payload:{next:'search'}});
        INTENT={canonical_en:'x',aliases:[],market:'china_a',market_label:'A股',
                asset_class:'etf',factor_family:'momentum',horizon:'daily',
                doc_preference:'both',restated_zh:'测试',confidence:0.8};
        showIntent(true);
        handle({kind:'data',stage:'search',message:'',payload:{page:1,total:2,
            has_more:false,papers:[{idx:0,title:'P1'},{idx:1,title:'P2'}]}});
        handle({kind:'await_input',stage:'select',message:'',payload:{next:'select'}});
    }""")
    page.wait_for_timeout(300)

    visible = page.evaluate(
        """() => [...document.querySelectorAll('#panels > .card')]
                 .filter(c => c.style.display !== 'none').map(c => c.id)""")
    assert len(visible) == 1, f"同时只应显示一个步骤面板，实际 {visible}"
    assert visible[0] == "pPapers"


def test_completed_steps_are_clickable_with_summary(page):
    """折叠后要带摘要——只写「已完成」等于逼用户点开才知道发生了什么。"""
    page.evaluate("""() => {
        INTENT={canonical_en:'x',aliases:[],market:'china_a',market_label:'A股',
                asset_class:'etf',factor_family:'momentum',horizon:'daily',
                doc_preference:'both',restated_zh:'测试',confidence:0.8};
        handle({kind:'data',stage:'intent',message:'',payload:{intent:INTENT}});
        handle({kind:'await_input',stage:'intent',message:'',payload:{next:'search'}});
        handle({kind:'data',stage:'search',message:'',payload:{page:1,total:1,
            has_more:false,papers:[{idx:0,title:'P1'}]}});
        handle({kind:'await_input',stage:'select',message:'',payload:{next:'select'}});
    }""")
    page.wait_for_timeout(300)

    sums = page.evaluate(
        "() => [...document.querySelectorAll('.step .sum')].map(e => e.textContent)")
    assert any("A股" in s for s in sums), f"意图步骤应有摘要: {sums}"
    assert page.evaluate("document.querySelectorAll('.step.clickable').length") >= 2

    page.evaluate("openStep('intent')")
    page.wait_for_timeout(300)
    visible = page.evaluate(
        """() => [...document.querySelectorAll('#panels > .card')]
                 .filter(c => c.style.display !== 'none').map(c => c.id)""")
    assert visible == ["pIntent"], f"点步骤应切回该面板，实际 {visible}"


def test_feeds_entry_sits_next_to_the_input(page):
    """灵感池不属于任何步骤，它的用途是「还没想好研究什么」，所以放在输入框旁。"""
    btn = page.query_selector(".heroHead .inspire")
    assert btn, "入口应在输入框标题行"
    assert "还没想好" in btn.inner_text()
    assert page.evaluate(
        "document.querySelector('#cardFeeds').classList.contains('hidden')"), \
        "默认应收起，不占开场版面"


def test_chart_has_axes_in_percent(page):
    """净值曲线要有 X/Y 轴，Y 轴是收益率。"""
    page.evaluate("""() => {
        handle({kind:'data',stage:'evaluate',message:'',payload:{evaluation:{
          factor_name:'f',status:'PASS',fail_reason:'',description:'',zh_what:'',
          is_ic:{ic_mean_1d:0.03,icir_daily:0.1,icir_annual:1.6,ic_tstat:3.1,
                 n_obs:700,monthly_win_rate:0.6,n_months:36,ic_by_lag:{},
                 ic_halflife:20,coverage:1},
          is_layers:{n_quantiles:5,layer_ann_returns:[-0.06,0.02,-0.01,0.05,0.12],
            monotonicity:0.9,long_short:{ann_return:0.2,ann_return_net:0.18,
            sharpe:0.8,sharpe_net:0.7,max_drawdown:-0.1,calmar:1.8,
            turnover_daily:0.11},long_only:null},
          oos_ic:null,oos_layers:null,sharpe_decay:null,max_nav_corr:0,
          nav:[{d:'2024-01-01',v:1},{d:'2024-03-01',v:1.1},{d:'2024-06-01',v:1.25}]}}});
    }""")
    page.wait_for_timeout(300)
    svg = page.evaluate(
        "() => {const s=document.querySelector('.bChart'); return s?s.outerHTML:''}")
    assert svg, "应渲染图表"
    assert svg.count("%<") >= 3, "Y 轴应有百分比刻度（收益率）"
    assert "2024-" in svg, "X 轴应有日期"
    # viewBox 宽高比须接近容器，否则等比缩放会留大片空白
    import re as _re
    m = _re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
    assert m and int(m.group(1)) / int(m.group(2)) > 5, \
        "viewBox 过窄会导致左右留白"


# ── 资料来源与参数出处的可见性 ──────────────────────────────────────────────
#
# 这两组断言守的是同一条线：用户必须能分辨「读到的」和「编出来的」。
# 逻辑测试验证不了徽标到底有没有渲染出来、文案是不是真能看懂，
# 所以放在真实浏览器里。

def test_spec_card_shows_parameter_provenance(page):
    """
    只读摘要时 LLM 给的 lookback 是它自己定的（实测摘要 7 项复现细节命中 0-1）。
    卡片必须把这件事说出来，否则看起来就像复现了论文。
    """
    page.evaluate("""() => {
        handle({kind:'data',stage:'extract',message:'',payload:{append:false,specs:[
          {idx:0,factor_name:'from_fulltext',formula:'f',
           provenance:'fulltext',param_sources:{lookback:'paper'},
           provenance_note:'已读取 arXiv 全文'},
          {idx:1,factor_name:'from_abstract',formula:'f',
           provenance:'abstract',param_sources:{lookback:'default'},
           provenance_note:'非 arXiv 来源，正文不可获取'}]}});
        handle({kind:'await_input',stage:'implement',message:'',
                payload:{next:'run'}});
    }""")
    page.wait_for_timeout(300)

    badges = page.evaluate(
        "() => [...document.querySelectorAll('#sList .prov')]"
        ".map(e => ({t:e.textContent.trim(), cls:e.className}))")
    assert len(badges) == 2, f"两个因子都应有来源徽标，实际 {badges}"

    full, abst = badges[0], badges[1]
    assert "已读全文" in full["t"], full
    assert "ok" in full["cls"], "读到全文应是肯定态"
    assert "1 项来自论文" in full["t"], full

    assert "仅摘要" in abst["t"], abst
    assert "warn" in abst["cls"], "只读摘要必须是警示态，不能和全文一个样"
    assert "0 项来自论文" in abst["t"] and "1 项为模型设定" in abst["t"], abst


def test_spec_badge_survives_missing_fields(page):
    """老会话或延伸变体没有这些字段，不能因此渲染崩掉。"""
    page.evaluate("""() => {
        handle({kind:'data',stage:'extract',message:'',payload:{append:false,
            specs:[{idx:0,factor_name:'legacy',formula:'f'}]}});
        handle({kind:'await_input',stage:'implement',message:'',
                payload:{next:'run'}});
    }""")
    page.wait_for_timeout(300)
    assert page.query_selector("#sList .item"), "缺字段时仍应渲染出卡片"
    assert not page._errors, f"控制台报错: {page._errors[:3]}"


def test_backfilled_abstract_is_labelled(page):
    """
    回填来的摘要不是原源给的，要标出来 —— 用户判断可信度时需要知道
    这条信息是从哪家库查来的。
    """
    page.evaluate("""() => {
        handle({kind:'data',stage:'search',message:'',payload:{page:1,total:2,
          has_more:false,papers:[
            {idx:0,title:'原源自带摘要',abstract:'a',source:'arxiv'},
            {idx:1,title:'跨库回填',abstract:'a',source:'crossref',
             doc_type:'working_paper',abstract_backfilled:true}]}});
        handle({kind:'await_input',stage:'select',message:'',
                payload:{next:'select'}});
    }""")
    page.wait_for_timeout(300)

    items = page.evaluate(
        """() => [...document.querySelectorAll('#pList .item')]
                 .map(e => e.innerHTML)""")
    assert len(items) == 2, f"应渲染两条，实际 {len(items)}"
    assert "摘要回填" not in items[0], "原源自带摘要的不该被标成回填"
    assert "摘要回填" in items[1], "回填来的必须标出"
