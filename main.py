#!/usr/bin/env python3
"""
AutoAlpha 多Agent流水线入口
用法：python main.py [论文PDF路径 / arXiv URL / 研究想法]
"""
import sys
import os
import json

try:
    import readline  # noqa: F401  — 启用上下键历史
except ImportError:
    pass

from autoalpha.llm import LLMClient
from autoalpha.db import AutoAlphaDB
from autoalpha.pipeline import Pipeline
from autoalpha.agents.input_classifier import classify_input


# ── 显示 ─────────────────────────────────────────────────────────────────────

def _banner():
    print("\n" + "=" * 62)
    print("  AutoAlpha  多Agent量化研究流水线")
    print("=" * 62)
    print("  直接输入研究想法即可，系统会自动搜索相关论文：")
    print("    示例：基于动量的A股ETF轮动策略")
    print()
    print("  也可以直接指定已有论文：")
    print("    arXiv URL  →  https://arxiv.org/abs/2401.xxxxx")
    print("    本地 PDF   →  /path/to/paper.pdf")
    print("    长文本粘贴 →  输入 --- 开始，再次 --- 结束")
    print()
    print("    /help  帮助    q  退出")
    print("-" * 62)


def _print_help():
    print("""
命令说明：
  <短句>         触发论文搜索 → 选择 → 流水线（idea模式）
  <URL>          直接解析 arXiv / 论文链接（url模式）
  <PDF路径>      解析本地PDF（pdf模式）
  ---            进入多行粘贴模式（粘贴摘要/全文，再输入 --- 结束）
  q / quit       退出

论文搜索后可同时选多篇（逗号分隔，例：1,2,3）进行对比。
""")


# ── 输入 ─────────────────────────────────────────────────────────────────────

def _read_multiline() -> str:
    """收集多行输入直到用户再次输入 ---"""
    print("  [多行模式] 粘贴内容后输入 --- 结束")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "---":
            break
        lines.append(line)
    return "\n".join(lines)


def _prompt() -> str:
    """读一行输入；若首行是 --- 则进入多行模式"""
    try:
        line = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"
    if line == "---":
        return _read_multiline()
    return line


# ── 论文选择 ─────────────────────────────────────────────────────────────────

def _show_papers(papers: list):
    print(f"\n  找到 {len(papers)} 篇相关论文：\n")
    for i, p in enumerate(papers, 1):
        title = p.get("title", "(无标题)")
        url = p.get("pdf_url", "")
        abstract = (p.get("abstract", "")[:130]).replace("\n", " ")
        print(f"  [{i}] {title}")
        if url:
            print(f"       {url}")
        print(f"       {abstract}…")
        print()


def _select_papers(papers: list) -> list:
    """展示论文列表，返回用户选中的子集（空 = 不使用搜索结果）"""
    _show_papers(papers)
    print("  选择论文（Enter = 第1篇，1,2,3 = 多选，skip = 用原始想法直接分析）：", end="")
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice.lower() == "skip":
        return []
    if not choice:
        return [papers[0]]

    selected = []
    for tok in choice.split(","):
        try:
            idx = int(tok.strip()) - 1
            if 0 <= idx < len(papers):
                selected.append(papers[idx])
        except ValueError:
            pass
    return selected if selected else [papers[0]]


# ── 结果打印 ──────────────────────────────────────────────────────────────────

def _print_result(result: dict):
    status = result.get("status", "UNKNOWN")
    if status == "CACHED":
        print("\n  ✓ 命中历史记录，已跳过复现")
        rec = result.get("record", {})
        print(f"    策略：{rec.get('strategy_name', 'N/A')}")
        print(f"    OOS Sharpe：{rec.get('oos_sharpe', 'N/A')}")
        return

    if status != "PASS":
        print(f"\n  ✗ 流水线未通过（{status}）")
        print(f"    原因：{result.get('reason') or result.get('fail_context', 'N/A')}")
        return

    paper_url = result.get("paper_url", "")
    paper_title = result.get("paper_title", "")
    report_path = result.get("report_path", "")

    print(f"\n  ✓ 质控通过！")
    print(f"    策略名称  ：{result.get('strategy_name', 'N/A')}")
    if paper_title:
        print(f"    论文标题  ：{paper_title}")
    if paper_url:
        print(f"    论文链接  ：{paper_url}")
    is_s = result.get("is_sharpe")
    oos_s = result.get("oos_sharpe")
    print(f"    IS  Sharpe：{is_s:.3f}" if isinstance(is_s, float) else "    IS  Sharpe：N/A")
    print(f"    OOS Sharpe：{oos_s:.3f}" if isinstance(oos_s, float) else "    OOS Sharpe：N/A")
    if report_path:
        print(f"    HTML 报告 ：{os.path.abspath(report_path)}")


def _print_comparison(results: list):
    """多篇论文对比汇总"""
    if len(results) < 2:
        return
    W = 70
    print("\n" + "=" * W)
    print(f"{'  多论文对比汇总':^{W}}")
    print("=" * W)
    fmt = "  {:<32}  {:<8}  {:>9}  {:>9}"
    print(fmt.format("策略/论文", "状态", "IS Sharpe", "OOS Sharpe"))
    print("  " + "-" * (W - 2))
    for r in results:
        name = (r.get("paper_title") or r.get("strategy_name") or "Unknown")[:32]
        status = r.get("status", "N/A")
        is_s = r.get("is_sharpe")
        oos_s = r.get("oos_sharpe")
        is_str = f"{is_s:.3f}" if isinstance(is_s, float) else "N/A"
        oos_str = f"{oos_s:.3f}" if isinstance(oos_s, float) else "N/A"
        url = r.get("paper_url", "")
        print(fmt.format(name, status, is_str, oos_str))
        if url:
            print(f"    {url}")
    print("=" * W)


# ── 主逻辑 ───────────────────────────────────────────────────────────────────

def main():
    llm = LLMClient()
    db = AutoAlphaDB(db_path="autoalpha.db")
    pipeline = Pipeline(
        llm=llm,
        db=db,
        output_dir="./paper/research",
        data_cache_dir="./data_cache",
    )

    if len(sys.argv) > 1:
        # 命令行直接传入（非交互）
        raw_input = " ".join(sys.argv[1:])
        print(f"[AutoAlpha] Running pipeline for: {raw_input[:100]}")
        result = pipeline.run(raw_input)
        print("\n[AutoAlpha] Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    # ── 交互模式 ──────────────────────────────────────────────────────────────
    _banner()

    while True:
        raw = _prompt()

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if raw.lower() in ("/help", "help"):
            _print_help()
            continue

        # 分类输入
        input_type, confidence = classify_input(raw)

        # idea 模式 → 先搜索论文，让用户选择
        if input_type == "idea" and confidence >= 0.6:
            print(f"\n[搜索] 正在搜索相关论文（confidence={confidence:.2f}）…")
            papers = pipeline.search_papers(raw, max_results=10)

            if papers:
                selected = _select_papers(papers)
                if not selected:
                    # 用户选 skip → 用原始想法
                    print("[流水线] 用原始想法直接运行…")
                    result = pipeline.run(raw)
                    _print_result(result)
                else:
                    # 运行选中的一篇或多篇
                    results = []
                    for paper in selected:
                        print(f"\n[流水线] 处理论文：{paper['title'][:60]}")
                        result = pipeline.run(raw, selected_paper=paper)
                        results.append(result)
                        _print_result(result)
                    _print_comparison(results)
            else:
                print("[搜索] 未找到相关论文，用原始想法运行…")
                result = pipeline.run(raw)
                _print_result(result)

        else:
            # url / pdf / text / low-confidence idea
            result = pipeline.run(raw)
            _print_result(result)


if __name__ == "__main__":
    main()
