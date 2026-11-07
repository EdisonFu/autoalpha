#!/usr/bin/env python3
"""
AutoAlpha 多Agent流水线入口
用法：python main.py [论文PDF路径 / arXiv URL / 研究想法]
"""
import sys
import os
import json
from dataclasses import asdict

if sys.platform != "darwin":
    try:
        import readline  # noqa: F401  — 启用上下键历史
    except ImportError:
        pass

from autoalpha.local_config import load_local_env

# 必须在创建任何 LLM 客户端之前加载本地 .env —— 否则密钥和模型
# 每次重启都要重配一遍。进程环境变量优先，便于临时覆盖。
load_local_env(verbose=True)

from autoalpha.pipeline import AutoAlphaPipeline
from autoalpha.pipeline_state import FactorReport, FactorResult
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

def _print_factor_metrics(title: str, results: list[FactorResult]):
    if not results:
        return

    print(f"\n  {title}")
    for result in results:
        print(f"    - {result.factor_name} [{result.status}]")
        print(
            "      "
            f"IC={result.ic_mean:.4f}  "
            f"ICIR={result.icir:.4f}  "
            f"Decay={result.ic_decay_ratio:.4f}  "
            f"Coverage={result.coverage:.4f}  "
            f"WinRate={result.monthly_win_rate:.4f}  "
            f"MaxCorr={result.max_correlation:.4f}"
        )
        if result.fail_reason:
            print(f"      原因：{result.fail_reason}")
        if result.code_path:
            print(f"      文件：{os.path.abspath(result.code_path)}")


def _print_result(report: FactorReport):
    passed_count = len(report.passed)
    failed_count = len(report.failed)
    ext_count = len(report.extensions_passed)
    paper = report.papers_used[0] if report.papers_used else {}

    print("\n  ✓ 流水线执行完成")
    if paper.get("title"):
        print(f"    论文标题  ：{paper['title']}")
    if paper.get("pdf_url"):
        print(f"    论文链接  ：{paper['pdf_url']}")
    print(f"    提取因子数：{report.total_extracted}")
    print(f"    通过因子数：{passed_count}")
    print(f"    失败因子数：{failed_count}")
    print(f"    延伸通过数：{ext_count}")
    if report.report_path:
        print(f"    HTML 报告 ：{os.path.abspath(report.report_path)}")

    _print_factor_metrics("通过因子指标：", report.passed)
    _print_factor_metrics("失败因子指标：", report.failed)
    _print_factor_metrics("延伸因子指标：", report.extensions_passed)


def _print_comparison(results: list[FactorReport]):
    """多篇论文对比汇总"""
    if len(results) < 2:
        return
    W = 70
    print("\n" + "=" * W)
    print(f"{'  多论文对比汇总':^{W}}")
    print("=" * W)
    fmt = "  {:<32}  {:>8}  {:>8}  {:>8}"
    print(fmt.format("论文", "提取", "通过", "延伸"))
    print("  " + "-" * (W - 2))
    for r in results:
        paper = r.papers_used[0] if r.papers_used else {}
        name = (paper.get("title") or r.idea or "Unknown")[:32]
        print(fmt.format(name, r.total_extracted, len(r.passed), len(r.extensions_passed)))
        url = paper.get("pdf_url", "")
        if url:
            print(f"    {url}")
    print("=" * W)


# ── 主逻辑 ───────────────────────────────────────────────────────────────────

def _serve_web(argv):
    """启动 Web 控制台。详见 README_WEB.md。"""
    port = 8000
    host = "127.0.0.1"
    for i, a in enumerate(argv):
        if a == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
        if a == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
    import os
    if "--demo" in argv:
        os.environ["AUTOALPHA_DEMO"] = "1"
    from autoalpha.server import serve
    from autoalpha.orchestrator import DEFAULT_KLINES
    key = any(os.environ.get(k) for k in
              ("LLM_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
               "OPENAI_API_KEY"))
    klines = os.environ.get("AUTOALPHA_KLINES")
    print("\n" + "=" * 54, flush=True)
    print(f"  AutoAlpha Web  →  http://{host}:{port}", flush=True)
    print("=" * 54, flush=True)
    if os.environ.get("AUTOALPHA_DEMO") == "1":
        print("  模式         : 演示（预置样例，无需 API Key 与行情数据）",
              flush=True)
    else:
        print(f"  LLM API Key : {'已配置' if key else '未配置 ⚠ 请设置 LLM_API_KEY'}",
              flush=True)
    print(f"  数据目录     : {klines or DEFAULT_KLINES}", flush=True)
    print("  停止服务     : Ctrl+C\n", flush=True)
    serve(host=host, port=port)


def main():
    if "--web" in sys.argv:
        _serve_web(sys.argv[1:])
        return

    pipeline = AutoAlphaPipeline(
        db_path="autoalpha.db",
        output_dir="./output",
    )

    if len(sys.argv) > 1:
        # 命令行直接传入（非交互）
        raw_input = " ".join(sys.argv[1:])
        print(f"[AutoAlpha] Running pipeline for: {raw_input[:100]}")
        result = pipeline.run(raw_input)
        print("\n[AutoAlpha] Result:")
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
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
            papers, _ = pipeline.search_papers_paged(raw, page=1, page_size=10)

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
                        result = pipeline.run(raw, selected_papers=[paper])
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
