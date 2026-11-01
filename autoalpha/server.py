"""
FastAPI 服务：分阶段流水线的 Web 外壳。

事件通过 SSE 推送，前端按阶段渲染。流水线在后台线程跑（LLM 调用和
沙箱执行都是阻塞的），事件经线程安全队列转交事件循环。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .flow import FlowSession, FlowEvent, EventKind, Stage
from .orchestrator import FlowOrchestrator, DEFAULT_KLINES

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

app = FastAPI(title="AutoAlpha")

_sessions: Dict[str, FlowSession] = {}
_queues: Dict[str, queue.Queue] = {}
_orch: Optional[FlowOrchestrator] = None
_running: Dict[str, threading.Thread] = {}   # 每会话当前阶段线程
_data_registry = None
_lock = threading.Lock()


def get_orch() -> FlowOrchestrator:
    global _orch
    with _lock:
        if _orch is None:
            _orch = FlowOrchestrator(
                klines_dir=os.environ.get("AUTOALPHA_KLINES", DEFAULT_KLINES))
    return _orch


class StartReq(BaseModel):
    idea: str
    mode: str = "stepwise"       # stepwise | auto
    sources: Optional[List[str]] = None     # 启用的检索源，None = 全部


class IntentReq(BaseModel):
    session_id: str
    intent: dict = {}            # 用户在界面上修正后的意图


class SelectReq(BaseModel):
    session_id: str
    paper_indices: List[int] = []
    append: bool = False          # True = 把新因子追加到现有列表


class MoreReq(BaseModel):
    session_id: str
    page: int = 2
    page_size: int = 10


class RunReq(BaseModel):
    session_id: str
    spec_indices: Optional[List[int]] = None


def _pump(sid: str, gen_factory):
    """后台线程跑生成器，把事件塞进队列。"""
    q = _queues[sid]
    try:
        for ev in gen_factory():
            q.put(ev)
    except Exception as e:
        q.put(FlowEvent(EventKind.ERROR, Stage.ERROR, f"流水线异常: {e}"))
    finally:
        q.put(None)        # 结束哨兵


def _spawn(sid: str, gen_factory) -> bool:
    """
    启动一个阶段生成器。若该会话已有阶段在运行则拒绝。

    不加这道检查时，前一阶段未结束就收到下一阶段请求（界面快速连点、
    或脚本按事件计数推进），两个生成器会并发写同一个 FlowSession ——
    实测 stage 从 evaluate 倒退回 extract，随后卡住。
    """
    _queues.setdefault(sid, queue.Queue())
    with _lock:
        t = _running.get(sid)
        if t is not None and t.is_alive():
            return False
        th = threading.Thread(target=_pump, args=(sid, gen_factory), daemon=True)
        _running[sid] = th
        th.start()
    return True


# ── API ──────────────────────────────────────────────────────────────────────

@app.post("/api/start")
def start(req: StartReq):
    idea = (req.idea or "").strip()
    if not idea:
        raise HTTPException(422, "研究方向不能为空")
    if len(idea) > 500:
        raise HTTPException(422, "研究方向过长（上限 500 字）")
    req = StartReq(idea=idea, mode=req.mode)
    sess = FlowSession(idea=req.idea)
    _sessions[sess.session_id] = sess
    orch = get_orch()

    if req.sources:
        orch.paper_search.enabled_sources = tuple(req.sources)

    if req.mode == "auto":
        _spawn(sess.session_id, lambda: orch.run_all(sess, req.idea))
    else:
        # 分步模式下 Step 1 结束即暂停，让用户确认/修正意图再检索
        def _gen():
            yield from orch.recognize_intent(sess, req.idea)
            yield FlowEvent(EventKind.AWAIT_INPUT, Stage.INTENT,
                            "确认研究意图后开始检索", {"next": "search"})
        _spawn(sess.session_id, _gen)

    return {"session_id": sess.session_id, "mode": req.mode}


@app.post("/api/select")
def select(req: SelectReq):
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    orch = get_orch()
    picked = orch.papers_by_index(sess, req.paper_indices)

    def _gen():
        yield from orch.extract_specs(sess, picked, append=req.append)
        yield FlowEvent(EventKind.AWAIT_INPUT, Stage.IMPLEMENT,
                        "请选择要实现的因子假设", {"next": "run"})

    if not _spawn(req.session_id, _gen):
        raise HTTPException(409, "上一阶段尚未结束，请稍候")
    return {"ok": True, "n_selected": len(picked)}


@app.post("/api/search")
def search(req: IntentReq):
    """Step 2：按（可能被用户修正过的）意图开始多源检索。"""
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    orch = get_orch()

    if req.intent and sess.intent is not None:
        for k, v in req.intent.items():
            if hasattr(sess.intent, k) and v not in (None, "", []):
                setattr(sess.intent, k, v)
        orch.paper_search.intent = sess.intent
        orch.paper_search._results_cache.pop(sess.idea, None)   # 意图变了要重查

    def _gen():
        yield from orch.search_papers(sess, sess.idea)
        yield from orch.prepare_data(sess)
        yield FlowEvent(EventKind.AWAIT_INPUT, Stage.SELECT,
                        "请选择要研究的论文", {"next": "select"})

    if not _spawn(req.session_id, _gen):
        raise HTTPException(409, "上一阶段尚未结束，请稍候")
    return {"ok": True}


@app.post("/api/more_papers")
def more_papers(req: MoreReq):
    """加载下一批论文。结果有缓存，翻页不重复调用 arXiv 与打分。"""
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    orch = get_orch()

    def _gen():
        yield from orch.search_papers(sess, sess.idea,
                                      max_results=req.page_size, page=req.page)
        yield FlowEvent(EventKind.AWAIT_INPUT, Stage.SELECT,
                        f"第 {req.page} 批论文已加载", {"next": "select"})

    if not _spawn(req.session_id, _gen):
        raise HTTPException(409, "上一阶段尚未结束，请稍候")
    return {"ok": True, "page": req.page}


@app.post("/api/run")
def run(req: RunReq):
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    orch = get_orch()

    def _gen():
        yield from orch.run_factors(sess, req.spec_indices)
        yield FlowEvent(EventKind.STAGE_END, Stage.DONE, "本轮完成",
                        {"can_extend": any(e.status == "PASS"
                                           for e in sess.evaluations)})

    if not _spawn(req.session_id, _gen):
        raise HTTPException(409, "上一阶段尚未结束，请稍候")
    return {"ok": True}


@app.post("/api/extend")
def extend(req: RunReq):
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    orch = get_orch()

    def _gen():
        yield from orch.run_extensions(sess)
        yield FlowEvent(EventKind.AWAIT_INPUT, Stage.IMPLEMENT,
                        "延伸假设已生成，可选择实现", {"next": "run"})

    if not _spawn(req.session_id, _gen):
        raise HTTPException(409, "上一阶段尚未结束，请稍候")
    return {"ok": True}


@app.get("/api/events/{session_id}")
async def events(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "session not found")

    async def stream():
        loop = asyncio.get_event_loop()
        while True:
            q = _queues.get(session_id)      # 每轮重取，不缓存引用
            if q is None:
                await asyncio.sleep(0.2)
                continue
            try:
                ev = await loop.run_in_executor(None, q.get, True, 30)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if ev is None:
                yield f"data: {json.dumps({'kind':'idle'})}\n\n"
                continue
            yield f"data: {ev.to_json()}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/session/{session_id}")
def session_state(session_id: str):
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    from .flow import evaluation_to_dict
    return {
        "session_id": sess.session_id, "idea": sess.idea,
        "stage": sess.stage.value, "error": sess.error,
        "n_papers": len(sess.papers), "n_specs": len(sess.specs),
        "evaluations": [evaluation_to_dict(e) for e in sess.evaluations],
        "usage": get_orch().llm.meter.summary(),
    }


class LLMConfigReq(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    price_input: Optional[float] = None
    price_output: Optional[float] = None


def _mask(key: str) -> str:
    if not key:
        return ""
    return f"{key[:6]}…{key[-4:]}" if len(key) > 14 else "已设置"


@app.get("/api/llm/config")
def llm_config_get():
    """当前 LLM 配置。密钥只回传掩码，不回传明文。"""
    import os as _os
    key = next((_os.environ.get(e) for e, _p, _u in
                __import__("autoalpha.llm", fromlist=["_KEY_ENVS"])._KEY_ENVS
                if _os.environ.get(e)), "")
    return {
        "has_key": bool(key),
        "key_masked": _mask(key),
        "model": _os.environ.get("LLM_MODEL", ""),
        "base_url": _os.environ.get("LLM_BASE_URL", ""),
        "price_input": _os.environ.get("LLM_PRICE_INPUT", ""),
        "price_output": _os.environ.get("LLM_PRICE_OUTPUT", ""),
        # 由模型 ID 推断的单价，供界面展示 —— 用户不必自己查价
        "inferred_price": _inferred_price(_os.environ.get("LLM_MODEL", "")),
    }


def _inferred_price(model: str) -> dict:
    from .llm import _price_for
    pin, pout = _price_for(model)
    return {"input": pin, "output": pout, "known": bool(pin or pout)}


@app.post("/api/llm/config")
def llm_config_set(req: LLMConfigReq):
    """
    更新 LLM 配置并重建客户端。

    只写进本进程的环境变量，不落盘：密钥不该被写进仓库或配置文件。
    重启服务后需要重新设置，或用启动前的环境变量。
    """
    import os as _os
    global _orch

    if req.api_key:
        _os.environ["LLM_API_KEY"] = req.api_key.strip()
    if req.model is not None:
        _os.environ["LLM_MODEL"] = req.model.strip()
    if req.base_url is not None:
        v = req.base_url.strip()
        if v:
            _os.environ["LLM_BASE_URL"] = v
        else:
            _os.environ.pop("LLM_BASE_URL", None)
    for field, env in (("price_input", "LLM_PRICE_INPUT"),
                       ("price_output", "LLM_PRICE_OUTPUT")):
        v = getattr(req, field)
        if v is not None:
            _os.environ[env] = str(v)

    if not _os.environ.get("LLM_MODEL"):
        raise HTTPException(422, "必须指定模型 ID")

    # 重建编排器，让新配置对后续会话生效
    with _lock:
        _orch = None
    try:
        get_orch()
    except Exception as e:
        raise HTTPException(422, f"配置无法初始化: {e}")
    return {"ok": True, **llm_config_get()}


@app.post("/api/llm/test")
def llm_test():
    """用一次最小调用验证配置可用，并回报延迟与用量。"""
    import time as _time
    try:
        orch = get_orch()
    except Exception as e:
        raise HTTPException(422, f"配置无效: {e}")
    t0 = _time.time()
    try:
        out = orch.llm.complete(system="Reply with the single word: ok",
                                user="ping", purpose="healthcheck")
    except Exception as e:
        raise HTTPException(502, f"调用失败: {str(e)[:200]}")
    return {"ok": True, "latency_ms": int((_time.time() - t0) * 1000),
            "reply": (out or "").strip()[:60],
            "usage": orch.llm.meter.summary()}


class DataSourceReq(BaseModel):
    name: str
    enabled: bool = True
    credentials: Dict[str, str] = {}


def _registry():
    from .data_config import DataSourceRegistry
    global _data_registry
    if _data_registry is None:
        _data_registry = DataSourceRegistry()
    return _data_registry


@app.get("/api/data/sources")
def data_sources():
    """各数据源状态。不返回任何凭据值。"""
    reg = _registry()
    return {"sources": reg.status(), "active": reg.active(),
            "klines_dir": os.environ.get("AUTOALPHA_KLINES", "")
                          or "(默认) factor-signal/data/klines"}


@app.post("/api/data/sources")
def data_source_set(req: DataSourceReq):
    """
    配置一个数据源。凭据只留在本进程内存，不落盘。
    留空的凭据字段表示保持原值，与 LLM 配置同一约定。
    """
    reg = _registry()
    try:
        cfg = reg.set(req.name, req.enabled, req.credentials)
    except ValueError as e:
        raise HTTPException(422, str(e))

    from .data_config import module_available
    warn = ""
    if not module_available(req.name):
        warn = "依赖库未安装，该源暂不可用"
    elif cfg.missing_credentials():
        warn = f"缺少凭据: {', '.join(cfg.missing_credentials())}"
    return {"ok": True, "active": reg.active(), "warning": warn,
            "sources": reg.status()}


class KlinesDirReq(BaseModel):
    path: str = ""


@app.post("/api/data/dir")
def data_dir_set(req: KlinesDirReq):
    """
    指定本地 K 线目录。留空恢复默认。
    校验目录存在且含频率子目录，否则返回具体原因 ——
    路径写错若静默接受，要等到跑流水线才暴露。
    """
    path = (req.path or "").strip()
    if not path:
        os.environ.pop("AUTOALPHA_KLINES", None)
        _reset_data_cache()
        return {"ok": True, "klines_dir": "(默认) factor-signal/data/klines"}

    if not os.path.isdir(path):
        raise HTTPException(422, f"目录不存在: {path}")
    subs = [d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))]
    if not subs:
        raise HTTPException(422, "目录下没有频率子目录（如 etf-daily/）")

    os.environ["AUTOALPHA_KLINES"] = path
    _reset_data_cache()
    return {"ok": True, "klines_dir": path, "freqs": sorted(subs)[:8]}


def _reset_data_cache():
    """换了数据目录，已加载的行情与切分必须作废。"""
    global _orch
    with _lock:
        if _orch is not None:
            _orch.klines_dir = os.environ.get("AUTOALPHA_KLINES", DEFAULT_KLINES)
            _orch._engine = None
            _orch._raw_data = None
            _orch._profile = {}
            _orch._data_context = ""


@app.get("/api/usage")
def usage():
    """当前进程累计的 LLM token 用量与估算花费。"""
    try:
        return get_orch().llm.meter.summary()
    except Exception as e:
        return {"error": str(e)}


class FeedReq(BaseModel):
    keywords: Optional[List[str]] = None
    limit: int = 30


@app.post("/api/feeds/refresh")
def feeds_refresh():
    """抓取量化博客/机构 RSS 并入库。RSS 不支持按词检索，故先抓后筛。"""
    from .sources.feeds import refresh
    try:
        return refresh(get_orch().db.db_path)
    except Exception as e:
        raise HTTPException(500, f"抓取失败: {e}")


@app.post("/api/feeds")
def feeds_list(req: FeedReq):
    """本地筛选已抓条目。不传关键词则返回最新。"""
    from .sources.feeds import FeedStore
    st = FeedStore(get_orch().db.db_path)
    return {"items": st.search(req.keywords, req.limit), **st.stats()}


@app.get("/api/feeds/suggest/{session_id}")
def feeds_suggest(session_id: str):
    """按当前会话的意图，从灵感池里挑相关条目。"""
    from .sources.feeds import FeedStore
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    kws = []
    if sess.intent is not None:
        kws = [sess.intent.factor_family, sess.intent.asset_class]
        kws += [w for t in sess.intent.query_terms(2) for w in t.split()
                if len(w) > 4][:4]
    st = FeedStore(get_orch().db.db_path)
    return {"items": st.search([k for k in kws if k and k != "unknown"], 12)}


@app.get("/api/library")
def library():
    try:
        return {"factors": get_orch().db.get_passed_factors()}
    except Exception as e:
        return {"factors": [], "error": str(e)}


@app.get("/api/health")
def health():
    import os as _os
    return {"ok": True,
            "has_api_key": any(_os.environ.get(k) for k in
                               ("LLM_API_KEY", "ANTHROPIC_API_KEY",
                                "DEEPSEEK_API_KEY", "OPENAI_API_KEY"))}


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def serve(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
