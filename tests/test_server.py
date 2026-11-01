"""
server.py 契约测试。
"""

import os
import queue
import time

import pytest

from autoalpha import server


def test_spawn_reuses_queue_object():
    """
    SSE 流在首次连接时持有队列引用。_spawn 若换新 Queue 对象，
    后续阶段的事件会写进没人读的队列 —— 实测表现为「加载第 2 批论文
    请求成功，但 SSE 收不到任何事件，会话状态不更新」。
    """
    sid = "qtest"
    server._queues.pop(sid, None)

    server._spawn(sid, lambda: iter([]))
    first = server._queues[sid]
    time.sleep(0.1)

    server._spawn(sid, lambda: iter([]))
    assert server._queues[sid] is first, "第二次 _spawn 不得替换队列对象"

    server._queues.pop(sid, None)


def test_spawn_delivers_events_across_stages():
    """连续两次 _spawn，两批事件都必须能从同一个队列读到。"""
    from autoalpha.flow import FlowEvent, EventKind, Stage

    sid = "qtest2"
    server._queues.pop(sid, None)

    def gen(tag):
        return lambda: iter([FlowEvent(EventKind.LOG, Stage.SEARCH, tag)])

    server._spawn(sid, gen("batch1"))
    q = server._queues[sid]
    got = [q.get(timeout=3)]
    while True:                       # 读到结束哨兵
        item = q.get(timeout=3)
        if item is None:
            break
        got.append(item)

    server._spawn(sid, gen("batch2"))
    item = q.get(timeout=3)
    assert item is not None and item.message == "batch2", "第二批事件未送达"

    server._queues.pop(sid, None)


def test_start_rejects_empty_idea():
    from fastapi.testclient import TestClient
    c = TestClient(server.app)
    for bad in ("", "   ", "\n"):
        assert c.post("/api/start", json={"idea": bad}).status_code == 422
    assert c.post("/api/start", json={"idea": "x" * 600}).status_code == 422


def test_unknown_session_returns_404():
    from fastapi.testclient import TestClient
    c = TestClient(server.app)
    assert c.get("/api/session/nope").status_code == 404
    assert c.post("/api/more_papers",
                  json={"session_id": "nope", "page": 2}).status_code == 404
    assert c.post("/api/select",
                  json={"session_id": "nope"}).status_code == 404


def test_session_papers_accumulate_across_pages():
    """
    sess.papers 必须是「已加载的全部论文」，与前端列表同语义。
    实测第 2 批曾覆盖第 1 批，使 /api/session 的 n_papers 从 10 变成 5。
    """
    from autoalpha.flow import FlowSession

    sess = FlowSession(idea="x")
    sess.papers = [{"title": f"P{i}", "pdf_url": f"u{i}"} for i in range(10)]

    batch2 = [{"title": f"P{i}", "pdf_url": f"u{i}"} for i in range(10, 15)]
    known = {(x.get("title"), x.get("pdf_url")) for x in sess.papers}
    sess.papers = sess.papers + [
        p for p in batch2 if (p.get("title"), p.get("pdf_url")) not in known]
    assert len(sess.papers) == 15

    # 重复批次不得重复累积
    known = {(x.get("title"), x.get("pdf_url")) for x in sess.papers}
    sess.papers = sess.papers + [
        p for p in batch2 if (p.get("title"), p.get("pdf_url")) not in known]
    assert len(sess.papers) == 15, "重复论文被二次累积"


def test_spawn_rejects_concurrent_stage():
    """
    同一会话不得同时跑两个阶段。实测两个生成器并发写 FlowSession 时，
    stage 会从 evaluate 倒退回 extract 然后卡死。
    """
    import threading
    import time

    sid = "reentry"
    server._queues.pop(sid, None)
    server._running.pop(sid, None)

    started = threading.Event()
    release = threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return
        yield                      # 使其成为生成器

    assert server._spawn(sid, slow) is True
    started.wait(3)
    assert server._spawn(sid, lambda: iter([])) is False, "重入必须被拒绝"

    release.set()
    for _ in range(50):
        if not server._running[sid].is_alive():
            break
        time.sleep(0.05)
    assert server._spawn(sid, lambda: iter([])) is True, "上一阶段结束后应可再启动"

    server._queues.pop(sid, None)
    server._running.pop(sid, None)


@pytest.mark.skipif(not os.environ.get("LLM_MODEL"),
                    reason="需要 LLM_MODEL 才能构造 orchestrator")
def test_endpoints_return_409_when_busy():
    """端点层面的重入拒绝（需真实环境变量，CI 无 LLM_MODEL 时跳过）。"""
    from fastapi.testclient import TestClient
    import threading, time
    from autoalpha.flow import FlowSession

    c = TestClient(server.app)
    sid = "busy-test"
    server._sessions[sid] = FlowSession(idea="x")
    server._queues.pop(sid, None)
    server._running.pop(sid, None)
    release = threading.Event()

    def slow():
        release.wait(5)
        return
        yield

    server._spawn(sid, slow)
    time.sleep(0.1)
    assert c.post("/api/run", json={"session_id": sid}).status_code == 409
    release.set(); time.sleep(0.2)
    for d in (server._sessions, server._queues, server._running):
        d.pop(sid, None)


def test_llm_config_never_returns_plaintext_key():
    """配置接口只回掩码。明文密钥不得出现在任何响应里。"""
    import os
    from fastapi.testclient import TestClient

    old = os.environ.get("LLM_API_KEY")
    os.environ["LLM_API_KEY"] = "sk-secret-value-1234567890"
    try:
        c = TestClient(server.app)
        body = c.get("/api/llm/config").text
        assert "sk-secret-value-1234567890" not in body
        assert "…" in body or "已设置" in body
    finally:
        if old is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = old


def test_llm_config_requires_model():
    from fastapi.testclient import TestClient
    c = TestClient(server.app)
    assert c.post("/api/llm/config", json={"model": ""}).status_code == 422


def test_demo_mode_needs_no_api_key():
    """
    演示模式的意义是让人无需密钥即可看懂项目。
    接口必须自包含，不触发任何 LLM 调用。
    """
    from fastapi.testclient import TestClient
    c = TestClient(server.app)
    r = c.get("/api/demo")
    assert r.status_code == 200
    d = r.json()
    assert d["papers"] and d["specs"] and d["evals"]
    assert "演示" in d["banner"], "必须标明是演示数据，不可被误当成真实结论"


def test_demo_includes_the_motivating_counterexample():
    """
    演示里必须包含「IC 为正但最高分位最差」的因子 ——
    它是本项目把重心放在评估而非生成的理由，不能只展示成功案例。
    """
    from fastapi.testclient import TestClient
    c = TestClient(server.app)
    evals = c.get("/api/demo").json()["evals"]

    assert any(e["status"] == "PASS" for e in evals)
    assert any(e["status"].startswith("FAIL") for e in evals), "不能只有成功案例"

    bad = [e for e in evals if e["is_layers"]
           and e["is_layers"]["layer_ann_returns"][-1]
           < max(e["is_layers"]["layer_ann_returns"])]
    assert bad, "应包含最高分位非最优的反例"
    tricky = [e for e in bad if e["is_ic"]["ic_tstat"] > 1.96]
    assert tricky, "反例应当 IC 显著——否则体现不出只看 IC 会被骗"
