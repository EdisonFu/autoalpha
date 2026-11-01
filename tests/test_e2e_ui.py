"""
真实浏览器 E2E 测试。

为什么需要：此前的前端测试只在 Node 里跑逻辑，验证不了布局 ——
数据源弹窗在小视口下内容被裁掉且无法滚动，逻辑测试全绿却漏掉了。
凡是涉及尺寸、滚动、可见性的断言，必须在真实渲染环境里做。

需要 chromium：.venv/bin/playwright install chromium
无浏览器时整个文件跳过，不阻塞其它测试。
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


def _chromium_ready() -> bool:
    try:
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _chromium_ready(),
                                reason="chromium 未安装")


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
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": 1280, "height": 720})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text)
              if m.type == "error" else None)
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

    modal = page.query_selector(".modal")
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
