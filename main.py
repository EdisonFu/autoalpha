#!/usr/bin/env python3
"""
AutoAlpha  多Agent因子挖掘流水线入口
用法：python main.py [研究方向]
"""
import sys
import os
import json
import threading
import itertools
import time
import textwrap
import shutil

# ── 颜色 / 格式（仅 tty 生效）────────────────────────────────────────────────

_TTY = sys.stdout.isatty()

def _c(text: str, *codes: str) -> str:
    if not _TTY:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"

def bold(t):    return _c(t, "1")
def dim(t):     return _c(t, "2")
def green(t):   return _c(t, "32")
def red(t):     return _c(t, "31")
def yellow(t):  return _c(t, "33")
def cyan(t):    return _c(t, "36")
def blue(t):    return _c(t, "34")
def magenta(t): return _c(t, "35")
def b_green(t): return _c(t, "1", "32")
def b_red(t):   return _c(t, "1", "31")
def b_blue(t):  return _c(t, "1", "34")
def b_cyan(t):  return _c(t, "1", "36")


# ── Spinner ───────────────────────────────────────────────────────────────────

class Spinner:
    """线程 spinner，不干扰主线程打印——仅在搜索等短暂等待时使用。"""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, msg: str = ""):
        self.msg = msg
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        if not _TTY:
            print(f"  {self.msg}", flush=True)
            return self
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_):
        if not _TTY:
            return
        self._stop.set()
        if self._t:
            self._t.join()
        sys.stdout.write("\r\033[K")  # 清除当前行
        sys.stdout.flush()

    def _run(self):
        for ch in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {cyan(ch)}  {dim(self.msg)}")
            sys.stdout.flush()
            time.sleep(0.09)


# ── Tab 补全 ──────────────────────────────────────────────────────────────────

_COMPLETIONS = [
    "/help", "/data", "/data clear",
    "/source", "/source add", "/source add tushare --token ",
    "/source add wind", "/source add baostock",
    "/source add joinquant --username ", "/source add ricequant --username ",
    "/library",
    "---", "quit", "q",
]

def _setup_readline():
    try:
        import readline

        def _completer(text, state):
            opts = [c for c in _COMPLETIONS if c.startswith(text)]
            return opts[state] if state < len(opts) else None

        readline.set_completer(_completer)
        # macOS 用 libedit，绑定方式不同
        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
    except (ImportError, TypeError):
        pass


# ── 布局工具 ──────────────────────────────────────────────────────────────────

def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns

def _box(lines: list, title: str = "", color_fn=None) -> str:
    """用 Unicode box-drawing 包裹文本行，返回字符串。"""
    cf = color_fn or (lambda x: x)
    inner_w = max((len(l) for l in lines), default=0)
    if title:
        inner_w = max(inner_w, len(title) + 2)
    inner_w = max(inner_w, 20)

    top = cf("╭" + "─" * (inner_w + 2) + "╮")
    bot = cf("╰" + "─" * (inner_w + 2) + "╯")
    mid = cf("│")

    rows = [top]
    if title:
        rows.append(f"{mid} {bold(title):<{inner_w}} {mid}")
        rows.append(cf("├" + "─" * (inner_w + 2) + "┤"))
    for l in lines:
        rows.append(f"{mid} {l:<{inner_w}} {mid}")
    rows.append(bot)
    return "\n".join(rows)

def _hr(char: str = "─", label: str = "") -> str:
    w = min(_term_width(), 72)
    if label:
        side = (w - len(label) - 2) // 2
        return dim(char * side) + f" {label} " + dim(char * (w - side - len(label) - 2))
    return dim(char * w)

def _fmt_float(v, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if isinstance(v, (int, float)) else "N/A"


# ── Banner / Help ─────────────────────────────────────────────────────────────

def _banner():
    w = min(_term_width(), 72)
    print()
    print(b_blue("╭" + "─" * (w - 2) + "╮"))
    title = "AutoAlpha  多Agent因子挖掘流水线"
    pad = (w - 2 - len(title)) // 2
    print(b_blue("│") + " " * pad + bold(title) + " " * (w - 2 - pad - len(title)) + b_blue("│"))
    print(b_blue("╰" + "─" * (w - 2) + "╯"))
    print()
    print(f"  {bold('直接输入研究方向')}，系统自动搜索相关论文并挖掘因子")
    print(f"  示例：{cyan('基于动量的截面因子')}")
    print(f"  示例：{cyan('volatility anomaly equity factors')}")
    print()
    print(f"  {dim('其他输入方式：')}")
    print(f"    {yellow('多行粘贴')}    先输入 {cyan('---')}，粘贴后再输入 {cyan('---')} 结束")
    print()
    cmds = (f"{cyan('/data')} 本地数据  {cyan('/source')} VIP数据源  "
            f"{cyan('/library')} 因子库  {cyan('/help')} 帮助  {cyan('q')} 退出")
    print(f"  {cmds}")
    print(_hr())
    print()


def _print_help():
    print()
    print(_hr(label="命令帮助"))
    sections = [
        ("研究输入",
         ["<研究方向>     搜索论文 → 选择 → 提取因子 → 评估（支持中文）",
          "---            进入多行粘贴模式（再输入 --- 结束）"]),
        ("本地数据  /data",
         ["/data <路径>   注册文件或目录（CSV/Parquet/Excel/Pickle/Feather）",
          "/data          查看已注册的本地数据",
          "/data clear    清除注册"]),
        ("VIP 数据源  /source",
         ["/source                   查看当前数据源列表",
          "/source add tushare --token <token>",
          "/source add wind",
          "/source add joinquant --username <手机号> --password <密码>",
          "/source add ricequant --username <账号> --password <密码>",
          "/source add baostock      免费，A股日线",
          "/source add ./my.py MyClass [--key val]  自定义适配器"]),
        ("因子库  /library",
         ["/library       查看已通过质量门控的所有因子",
          "               显示 IC均值 / ICIR / IC衰减比 等核心指标"]),
        ("质量门控标准",
         ["IC均值 > 0.03", "ICIR > 0.5", "IC衰减比(5d/1d) > 0.5",
          "因子覆盖率 > 0.8", "月胜率 > 0.55", "与已有因子最大相关 < 0.7"]),
        ("数据优先级",
         ["本地文件  >  VIP 适配器（按 priority）  >  AKShare（兜底）"]),
    ]
    for title, lines in sections:
        print()
        print(f"  {bold(title)}")
        for l in lines:
            print(f"    {dim('·')} {l}")
    print()
    print(_hr())
    print()


# ── 输入 ─────────────────────────────────────────────────────────────────────

def _read_multiline() -> str:
    print(f"  {dim('[多行模式] 粘贴内容，输入')} {cyan('---')} {dim('结束')}")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "---":
            break
        lines.append(line)
    text = "\n".join(lines)
    print(f"  {dim(f'已读取 {len(lines)} 行，{len(text)} 字符')}")
    return text


def _prompt() -> str:
    try:
        line = input(f"\n{b_cyan('❯')} ").strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"
    if line == "---":
        return _read_multiline()
    return line


# ── 论文展示 / 分页选择 ───────────────────────────────────────────────────────

PAGE_SIZE  = 10
MAX_PAGES  = 10


def _show_papers(papers: list, page: int = 1, total_pages: int = 1, total: int = 0):
    """展示一页论文列表，含页码和日期。"""
    print()
    page_info = f"第 {page}/{total_pages} 页"
    total_info = f"共约 {total} 篇" if total else ""
    header = f"  {bold('相关论文')}  {dim(page_info)}  {dim(total_info)}"
    print(header)
    print()
    for i, p in enumerate(papers, 1):
        title  = p.get("title", "(无标题)")
        url    = p.get("pdf_url", "")
        date   = p.get("date", "")
        ab     = p.get("abstract", "").replace("\n", " ")
        ab_cut = textwrap.shorten(ab, width=100, placeholder="…")

        rel   = p.get("_relevance_score")
        comb  = p.get("_combined_score")
        num   = b_cyan(f"[{i:>2}]")

        # 相关性徽章
        if rel is not None:
            if rel >= 8:
                rel_tag = f"  {b_green('★★★')}"
            elif rel >= 6:
                rel_tag = f"  {yellow('★★☆')}"
            else:
                rel_tag = f"  {dim('★☆☆')}"
        else:
            rel_tag = ""

        date_tag  = f"  {yellow(date)}" if date else ""
        score_tag = f"  {dim(f'综合 {comb:.2f}')}" if comb is not None else ""
        print(f"  {num}  {bold(title)}{rel_tag}{date_tag}{score_tag}")
        if url:
            print(f"        {cyan(url)}")
        print(f"        {dim(ab_cut)}")
        print()


def _select_papers_paged(pipeline, idea: str) -> list:
    """
    带分页的交互式论文选择。
    - 每页 10 篇，最多 10 页，按提交日期倒序
    - n=下一页  p=上一页  数字=选择  s=用原始想法
    """
    page = 1

    while True:
        with Spinner(f"搜索第 {page} 页…"):
            papers, total = pipeline.search_papers_paged(
                idea, page=page, page_size=PAGE_SIZE
            )

        if not papers:
            print(f"\n  {yellow('⚠')}  第 {page} 页无结果")
            if page > 1:
                page -= 1
                continue
            return []

        total_pages = min(
            max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
            MAX_PAGES
        )

        _show_papers(papers, page=page, total_pages=total_pages, total=total)

        # 构造提示行
        hints = [
            f"{cyan('Enter')}={dim('第1篇')}",
            f"{cyan('1,2,3')}={dim('多选')}",
        ]
        if page < total_pages:
            hints.append(f"{cyan('n')}={dim('下一页')}")
        if page > 1:
            hints.append(f"{cyan('p')}={dim('上一页')}")
        hints.append(f"{cyan('s')}={dim('跳过论文搜索')}")

        prompt = f"  {dim('选择')}  {'  '.join(hints)}  {b_cyan('❯')} "
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []

        if choice == "n":
            if page < total_pages:
                page += 1
            else:
                print(f"  {dim('已是最后一页（最多 10 页）')}")
            continue

        if choice == "p":
            if page > 1:
                page -= 1
            else:
                print(f"  {dim('已是第一页')}")
            continue

        if choice in ("s", "skip"):
            return []

        # 解析数字选择
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


# ── 因子结果展示 ──────────────────────────────────────────────────────────────

def _sharpe_color_ic(v):
    """IC均值着色：>0.05绿，>0.03黄，否则红"""
    if not isinstance(v, (int, float)):
        return dim("N/A")
    s = f"{v:.4f}"
    return b_green(s) if v >= 0.05 else (yellow(s) if v >= 0.03 else red(s))

def _icir_color(v):
    """ICIR着色：>0.7绿，>0.5黄，否则红"""
    if not isinstance(v, (int, float)):
        return dim("N/A")
    s = f"{v:.4f}"
    return b_green(s) if v >= 0.7 else (yellow(s) if v >= 0.5 else red(s))


def _print_factor_table(report):
    """打印因子评估结果表格"""
    all_results = report.passed + report.extensions_passed + report.failed
    if not all_results:
        print(f"\n  {dim('（无因子评估结果）')}")
        return

    all_results_sorted = sorted(all_results, key=lambda x: -x.icir)

    print()
    print(_hr(label=f"因子评估结果（{len(all_results)} 个）"))
    print()

    # Header
    hdr_fmt = f"  {'因子名':<32}  {'IC均值':>8}  {'ICIR':>8}  {'衰减比':>8}  {'覆盖率':>6}  {'月胜率':>6}  {'状态':<6}"
    print(dim(hdr_fmt))
    print(dim("  " + "─" * 80))

    for r in all_results_sorted:
        name = r.factor_name[:32]
        st_colored = b_green("PASS") if r.status == "PASS" else red("FAIL")
        ext_tag = dim(" [ext]") if hasattr(r, '_is_extension') and r._is_extension else ""
        print(
            f"  {name:<32}  "
            f"{_sharpe_color_ic(r.ic_mean):>8}  "
            f"{_icir_color(r.icir):>8}  "
            f"{_fmt_float(r.ic_decay_ratio):>8}  "
            f"{_fmt_float(r.coverage, 3):>6}  "
            f"{_fmt_float(r.monthly_win_rate, 3):>6}  "
            f"{st_colored}{ext_tag}"
        )
        if r.status == "FAIL" and r.fail_reason:
            print(f"  {dim('  ' + r.fail_reason[:70])}")

    print(dim("  " + "─" * 80))
    print()


def _print_factor_report(report):
    """打印完整因子研究报告摘要"""
    total_pass = len(report.passed) + len(report.extensions_passed)
    total_fail = len(report.failed)

    lines = [
        f"{bold('研究方向')}  {report.idea[:60]}",
        f"{bold('参考论文')}  {len(report.papers_used)} 篇",
        f"{bold('提取因子')}  {report.total_extracted} 个",
        f"{bold('测试因子')}  {report.total_tested} 个",
        "",
        f"通过因子  {b_green(str(total_pass))} 个"
        + (f"  （含 {len(report.extensions_passed)} 个延伸变体）" if report.extensions_passed else ""),
        f"未通过    {red(str(total_fail))} 个",
    ]
    if report.report_path:
        lines.append("")
        lines.append(f"{bold('HTML 报告')}  {cyan(os.path.abspath(report.report_path))}")

    color_fn = b_green if total_pass > 0 else (red if total_fail > 0 else dim)
    title_str = "因子研究完成" + (f"  {total_pass} 个因子通过质量门控" if total_pass > 0 else "  暂无因子通过门控")
    print()
    print(_box(lines, title=title_str, color_fn=color_fn))


# ── /library 命令 ─────────────────────────────────────────────────────────────

def _handle_library_command(pipeline) -> None:
    factors = pipeline.db.get_passed_factors()
    if not factors:
        print(f"\n  {dim('因子库为空，尚无因子通过质量门控')}")
        return

    print()
    print(_hr(label=f"因子库（{len(factors)} 个已通过因子）"))
    print()
    hdr = f"  {'因子名':<32}  {'IC均值':>8}  {'ICIR':>8}  {'衰减比':>8}  {'来源论文':<30}"
    print(dim(hdr))
    print(dim("  " + "─" * 90))
    for f in factors:
        name = (f.get("factor_name") or "")[:32]
        ic   = f.get("ic_mean") or 0
        icir = f.get("icir") or 0
        decay = f.get("ic_decay_ratio") or 0
        paper = (f.get("source_paper_title") or "")[:30]
        ext_tag = dim(" [ext]") if f.get("variant_of") else ""
        print(
            f"  {name:<32}  "
            f"{_sharpe_color_ic(ic):>8}  "
            f"{_icir_color(icir):>8}  "
            f"{_fmt_float(decay):>8}  "
            f"{dim(paper)}{ext_tag}"
        )
    print(dim("  " + "─" * 90))
    print()


# ── /data 命令 ────────────────────────────────────────────────────────────────

def _handle_data_command(cmd: str, pipeline) -> None:
    parts = cmd.split(None, 1)
    arg = parts[1].strip() if len(parts) > 1 else ""

    if not arg:
        print(f"\n  {dim('本地数据功能：注册数据文件供因子计算使用')}")
        print(f"  用法：{cyan('/data /path/to/file.csv')}  或  {cyan('/data /path/to/dir/')}")
        print(f"  清除：{cyan('/data clear')}")
        return

    if arg == "clear":
        pipeline.set_local_data(None)
        print(f"  {green('已清除本地数据注册')}")
        return

    if not os.path.exists(arg):
        print(f"\n  {red('路径不存在:')} {arg}")
        return

    pipeline.set_local_data([arg])
    print(f"\n  {green('已注册本地数据路径:')} {arg}")
    print()


# ── /source 命令 ──────────────────────────────────────────────────────────────

def _handle_source_command(cmd: str, pipeline) -> None:
    from autoalpha.data_source import create_adapter, load_adapter_from_file

    tokens = cmd.split()
    sub = tokens[1].lower() if len(tokens) > 1 else "list"

    if sub in ("list", ""):
        adapters = pipeline.list_data_adapters()
        if not adapters:
            print(f"\n  {dim('当前无注册适配器（pipeline 使用内置 AKShare）')}")
            return
        print()
        print(f"  {bold('数据源适配器')}  {dim('（按优先级）')}")
        print()
        for a in sorted(adapters, key=lambda x: x["priority"]):
            ok_sym = green("✓") if a["available"] else red("✗")
            pri    = dim(f"[{a['priority']:>3}]")
            name   = cyan(f"{a['name']:<16}")
            cls    = dim(f"({a['class']})")
            print(f"    {ok_sym}  {pri}  {name}  {cls}")
        print()
        return

    if sub == "add":
        if len(tokens) < 3:
            print(f"\n  {dim('用法：')} {cyan('/source add <名称> [--key value ...]')}")
            return

        name_or_path = tokens[2]
        kwargs: dict = {}
        i = 3
        while i < len(tokens):
            if tokens[i].startswith("--") and i + 1 < len(tokens):
                kwargs[tokens[i][2:]] = tokens[i + 1]
                i += 2
            else:
                i += 1

        if name_or_path.endswith(".py") or os.path.isfile(name_or_path):
            class_name = tokens[3] if len(tokens) > 3 and not tokens[3].startswith("--") \
                         else "CustomAdapter"
            try:
                adapter = load_adapter_from_file(name_or_path, class_name, kwargs)
                pipeline.add_data_adapter(adapter)
                ok = adapter.test_connection()
                status = green("通过") if ok else yellow("失败（检查SDK/凭证）")
                print(f"\n  {green('已注册自定义适配器')} {cyan(adapter.name)}  连接测试: {status}")
            except Exception as e:
                print(f"\n  {red('加载失败:')} {e}")
            return

        try:
            adapter = create_adapter(name_or_path, **kwargs)
            pipeline.add_data_adapter(adapter)
            ok = adapter.test_connection()
            status = green("通过") if ok else yellow("失败（检查SDK/凭证）")
            print(f"\n  {green('已注册')} {cyan(adapter.name)} "
                  f"{dim(f'priority={adapter.priority}')}  连接测试: {status}")
        except ValueError as e:
            print(f"\n  {red(str(e))}")
        except Exception as e:
            print(f"\n  {red('初始化失败:')} {e}")
        print()
        return

    print(f"\n  {yellow('未知子命令')} {sub!r}  →  输入 {cyan('/source')} 查看帮助")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    _setup_readline()

    from autoalpha.pipeline import AutoAlphaPipeline
    from autoalpha.agents.input_classifier import classify_input

    pipeline = AutoAlphaPipeline(
        db_path="autoalpha.db",
        output_dir="./output",
    )

    # ── 命令行直接传入（非交互）──────────────────────────────────────────────
    if len(sys.argv) > 1:
        raw_input = " ".join(sys.argv[1:])
        print(f"[AutoAlpha] Running: {raw_input[:100]}")
        report = pipeline.run(raw_input)
        print(f"\n通过因子: {len(report.passed) + len(report.extensions_passed)}")
        print(f"未通过:   {len(report.failed)}")
        if report.report_path:
            print(f"报告路径: {os.path.abspath(report.report_path)}")
        return

    # ── 交互模式 ──────────────────────────────────────────────────────────────
    _banner()

    while True:
        raw = _prompt()

        if not raw:
            continue

        if raw.lower() in ("q", "quit", "exit"):
            print(f"\n  {dim('再见！')}\n")
            break

        if raw.lower() in ("/help", "help"):
            _print_help()
            continue

        if raw.startswith("/data"):
            _handle_data_command(raw, pipeline)
            continue

        if raw.startswith("/source"):
            _handle_source_command(raw, pipeline)
            continue

        if raw.startswith("/library"):
            _handle_library_command(pipeline)
            continue

        # ── 分类输入 ─────────────────────────────────────────────────────────
        input_type, confidence = classify_input(raw)

        if input_type == "idea" and confidence >= 0.6:
            selected = _select_papers_paged(pipeline, raw)

            if selected is None or selected == []:
                print(f"\n  {dim('跳过论文选择，直接基于研究方向提取因子…')}")
                with Spinner("运行因子挖掘流水线…"):
                    report = pipeline.run(raw)
            else:
                print()
                paper_titles = ", ".join(p["title"][:40] for p in selected)
                print(_hr(label=f"已选 {len(selected)} 篇论文"))
                with Spinner("运行因子挖掘流水线…"):
                    report = pipeline.run(raw, selected_papers=selected)
        else:
            with Spinner("运行因子挖掘流水线…"):
                report = pipeline.run(raw)

        _print_factor_table(report)
        _print_factor_report(report)


if __name__ == "__main__":
    main()
