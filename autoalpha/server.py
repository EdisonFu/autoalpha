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


class SelectReq(BaseModel):
    session_id: str
    paper_indices: List[int] = []


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


def _spawn(sid: str, gen_factory):
    _queues[sid] = queue.Queue()
    t = threading.Thread(target=_pump, args=(sid, gen_factory), daemon=True)
    t.start()


# ── API ──────────────────────────────────────────────────────────────────────

@app.post("/api/start")
def start(req: StartReq):
    sess = FlowSession(idea=req.idea)
    _sessions[sess.session_id] = sess
    orch = get_orch()

    if req.mode == "auto":
        _spawn(sess.session_id, lambda: orch.run_all(sess, req.idea))
    else:
        def _gen():
            yield from orch.search_papers(sess, req.idea)
            yield from orch.prepare_data(sess)
            yield FlowEvent(EventKind.AWAIT_INPUT, Stage.SELECT,
                            "请选择要研究的论文", {"next": "select"})
        _spawn(sess.session_id, _gen)

    return {"session_id": sess.session_id, "mode": req.mode}


@app.post("/api/select")
def select(req: SelectReq):
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    orch = get_orch()
    picked = [sess.papers[i] for i in req.paper_indices
              if 0 <= i < len(sess.papers)]

    def _gen():
        yield from orch.extract_specs(sess, picked)
        yield FlowEvent(EventKind.AWAIT_INPUT, Stage.IMPLEMENT,
                        "请选择要实现的因子假设", {"next": "run"})

    _spawn(req.session_id, _gen)
    return {"ok": True, "n_selected": len(picked)}


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

    _spawn(req.session_id, _gen)
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

    _spawn(req.session_id, _gen)
    return {"ok": True}


@app.get("/api/events/{session_id}")
async def events(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, "session not found")

    async def stream():
        loop = asyncio.get_event_loop()
        while True:
            q = _queues.get(session_id)
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


@app.get("/api/usage")
def usage():
    """当前进程累计的 LLM token 用量与估算花费。"""
    try:
        return get_orch().llm.meter.summary()
    except Exception as e:
        return {"error": str(e)}


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
            "has_api_key": bool(_os.environ.get("ANTHROPIC_API_KEY")
                                or _os.environ.get("DEEPSEEK_API_KEY")
                                or _os.environ.get("OPENAI_API_KEY"))}


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def serve(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
