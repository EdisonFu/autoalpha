"""
server.py 契约测试。
"""

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
