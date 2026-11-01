#!/usr/bin/env python3
"""
驱动 Web 流水线跑完整一轮，用于计时与冒烟测试。

为什么不用 shell + SSE 事件计数：
    事件条数随论文数、重试次数、源数量变化，不是稳定的阶段边界。
    按条数推进会在阶段未结束时发下一个请求（实测撞 409）。

两条判定规则（都踩过坑）：
  1. 阶段是否结束，看 /api/session 的真实计数（n_papers / n_specs /
     len(evaluations)），不看 stage 字段 —— 分步模式下 stage 停在
     "intent" 既表示进行中也表示已完成，无法区分。
  2. 下一阶段请求返回 409 表示上一阶段仍在跑，重试即可。

用法：
    python scripts/drive_pipeline.py --port 8765 --idea "ETF截面动量因子"
    python scripts/drive_pipeline.py --ab 8766 8765     # 串行 vs 并发对照
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def _post(base, path, payload=None, timeout=30):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def _state(base, sid, timeout=20):
    with urllib.request.urlopen(f"{base}/api/session/{sid}", timeout=timeout) as r:
        return json.load(r)


def _advance(base, sid, path, payload, done, timeout=600, label=""):
    """发请求（409 则重试），然后等 done(state) 为真。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, _ = _post(base, path, {"session_id": sid, **payload})
        if code == 200:
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"{label}: 请求始终未被接受")

    while time.time() < deadline:
        try:
            d = _state(base, sid)
            if done(d):
                return d
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{label}: 等待结果超时")


def run_once(port, idea, verbose=True):
    base = f"http://127.0.0.1:{port}"
    code, r = _post(base, "/api/start", {"idea": idea, "mode": "stepwise"})
    if code != 200:
        raise RuntimeError(f"start 失败 HTTP {code}")
    sid = r["session_id"]
    say = ((lambda m: print(f"    {m}", flush=True))
           if verbose else (lambda m: None))
    say(f"session {sid}")

    t_search = time.time()
    d = _advance(base, sid, "/api/search", {},
                 lambda d: d.get("n_papers", 0) > 0, label="search")
    say(f"检索 {d['n_papers']} 篇  ({time.time() - t_search:.0f}s)")

    t_ex = time.time()
    d = _advance(base, sid, "/api/select", {"paper_indices": [0]},
                 lambda d: d.get("n_specs", 0) > 0, label="select")
    n = d["n_specs"]
    say(f"提取 {n} 个假设  ({time.time() - t_ex:.0f}s)")

    t_run = time.time()
    d = _advance(base, sid, "/api/run", {},
                 lambda d: len(d.get("evaluations", [])) >= n,
                 timeout=1200, label="run")
    elapsed = time.time() - t_run
    evals = d.get("evaluations", [])
    npass = sum(1 for e in evals if e.get("status") == "PASS")
    say(f"实现+评估 {len(evals)} 个（{npass} 通过）  ({elapsed:.0f}s)")
    return {"session_id": sid, "n_specs": n, "n_evals": len(evals),
            "n_pass": npass, "run_seconds": elapsed,
            "usage": d.get("usage", {})}


def main():
    # 后台运行时 stdout 默认全缓冲，进度看不到，容易误判为卡死
    try:
        import sys
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--idea", default="ETF截面动量因子")
    ap.add_argument("--ab", nargs=2, type=int, metavar=("SERIAL", "CONCURRENT"))
    a = ap.parse_args()

    if a.ab:
        results = {}
        for port, label in zip(a.ab, ("串行", "并发")):
            print(f"\n[{label}] port={port}")
            try:
                results[label] = run_once(port, a.idea)
            except Exception as e:
                print(f"    失败: {e}")
        if len(results) == 2:
            s = results["串行"]["run_seconds"]
            c = results["并发"]["run_seconds"]
            print(f"\n串行 {s:.0f}s → 并发 {c:.0f}s   加速比 {s / c:.2f}x")
    else:
        out = run_once(a.port, a.idea)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
