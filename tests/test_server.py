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
