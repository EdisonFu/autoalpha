import os
from datetime import datetime
from typing import List
from ..pipeline_state import FactorReport, FactorResult
from .factor_eval_agent import GATES

_PASS_COLOR = "#22c55e"
_FAIL_COLOR = "#ef4444"
_EXT_COLOR  = "#3b82f6"


def _sparkline(ic_series: list, width: int = 90, height: int = 22) -> str:
    if len(ic_series) < 2:
        return ""
    mn, mx = min(ic_series), max(ic_series)
    span = mx - mn if mx != mn else 1.0
    # y coordinate for a given value (0=top, height=bottom in SVG)
    def _y(v: float) -> float:
        return height - (v - mn) / span * height
    # zero-line y position
    zero_y = _y(0.0)
    zero_y = max(0.0, min(float(height), zero_y))
    n = len(ic_series)
    step = width / (n - 1)
    points = " ".join(
        f"{i * step:.2f},{_y(v):.2f}" for i, v in enumerate(ic_series)
    )
    mean_ic = sum(ic_series) / n
    line_color = "#22c55e" if mean_ic > 0 else "#ef4444"
    return (
        f'<svg width="{width}" height="{height}" style="vertical-align:middle" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<line x1="0" y1="{zero_y:.2f}" x2="{width}" y2="{zero_y:.2f}" '
        f'stroke="#cbd5e1" stroke-width="0.8"/>'
        f'<polyline points="{points}" fill="none" stroke="{line_color}" stroke-width="1.5"/>'
        f'</svg>'
    )


def _avg_judge_score(results: list) -> str:
    scores = [getattr(r, "judge_score", 0.0) for r in results if getattr(r, "judge_score", 0.0) > 0]
    if not scores:
        return "—"
    return f"{sum(scores) / len(scores):.1f}"


def _badge(status: str) -> str:
    color = _PASS_COLOR if status == "PASS" else _FAIL_COLOR
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.85em">{status}</span>'


def _judge_badge(score: float) -> str:
    if score <= 0:
        return ""
    color = "#22c55e" if score >= 7 else ("#f59e0b" if score >= 5 else "#ef4444")
    return f'<span style="background:{color};color:white;padding:1px 6px;border-radius:4px;font-size:0.8em">{score:.1f}</span>'


def _row(r: FactorResult, row_class: str = "") -> str:
    judge_score = getattr(r, "judge_score", 0.0)
    judge_comment = getattr(r, "judge_comment", "")
    fail = f'<span style="color:#888;font-size:0.8em">{r.fail_reason}</span>' if r.fail_reason else ""
    judge_cell = _judge_badge(judge_score)
    if judge_comment and judge_score > 0:
        judge_cell += f' <span style="color:#64748b;font-size:0.78em" title="{judge_comment}">ℹ</span>'
    return f"""<tr class="{row_class}">
  <td>{r.factor_name}</td>
  <td>{r.ic_mean:.4f}</td>
  <td>{r.icir:.4f}</td>
  <td>{r.ic_decay_ratio:.4f}</td>
  <td>{r.coverage:.3f}</td>
  <td>{r.monthly_win_rate:.3f}</td>
  <td>{r.max_correlation:.3f}</td>
  <td>{_sparkline(getattr(r, 'ic_series', []))}</td>
  <td>{judge_cell}</td>
  <td>{_badge(r.status)} {fail}</td>
</tr>"""


class ReportAgent:
    """生成因子质量检测 HTML 报告"""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, report: FactorReport) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"factor_report_{ts}.html"
        path = os.path.join(self.output_dir, fname)

        all_results = report.passed + report.extensions_passed + report.failed
        rows_html = "\n".join(
            _row(r, "pass-row" if r.status=="PASS" else "fail-row")
            for r in sorted(all_results, key=lambda x: -x.icir)
        )

        paper_items = "\n".join(
            f'<li><a href="{p.get("pdf_url","#")}" target="_blank">{p.get("title","")}</a> '
            f'<span style="color:#888">({p.get("date","")})</span></li>'
            for p in report.papers_used
        )

        _GATE_LABELS = {
            "ic_mean": "IC 均值", "icir": "ICIR", "ic_decay_ratio": "IC 衰减比 (5d/1d)",
            "coverage": "因子覆盖率", "monthly_win_rate": "月度胜率",
            "max_correlation": "与已有因子最大相关",
        }
        _GATE_OPS = {"max_correlation": "<"}
        gates_html = "\n".join(
            f'<tr><td>{_GATE_LABELS.get(k, k)}</td>'
            f'<td>{_GATE_OPS.get(k, ">")} {v}</td></tr>'
            for k, v in GATES.items()
        )

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>AutoAlpha 因子质量报告</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; color: #1e293b; }}
  .header {{ background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 2rem 3rem; }}
  .header h1 {{ margin: 0 0 .5rem; font-size: 1.8rem; }}
  .header p {{ margin: 0; opacity: .8; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.1); padding: 1.5rem 2rem; margin-bottom: 1.5rem; }}
  .stats {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
  .stat {{ text-align: center; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: #1e40af; }}
  .stat-label {{ color: #64748b; font-size: .85rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th {{ background: #f1f5f9; padding: .6rem 1rem; text-align: left; font-weight: 600; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: .55rem 1rem; border-bottom: 1px solid #f1f5f9; }}
  .pass-row {{ background: #f0fdf4; }}
  .fail-row {{ background: #fff7f7; }}
  h2 {{ color: #1e40af; margin-top: 0; }}
  ul {{ padding-left: 1.5rem; }}
  li {{ margin: .4rem 0; }}
  a {{ color: #3b82f6; }}
</style>
</head>
<body>
<div class="header">
  <h1>AutoAlpha 因子质量报告</h1>
  <p>研究方向：{report.idea} &nbsp;·&nbsp; 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>
<div class="container">

<div class="card">
  <h2>概要</h2>
  <div class="stats">
    <div class="stat"><div class="stat-value">{len(report.papers_used)}</div><div class="stat-label">参考论文</div></div>
    <div class="stat"><div class="stat-value">{report.total_extracted}</div><div class="stat-label">提取因子</div></div>
    <div class="stat"><div class="stat-value">{report.total_tested}</div><div class="stat-label">测试因子</div></div>
    <div class="stat"><div class="stat-value" style="color:{_PASS_COLOR}">{len(report.passed) + len(report.extensions_passed)}</div><div class="stat-label">通过因子</div></div>
    <div class="stat"><div class="stat-value" style="color:{_FAIL_COLOR}">{len(report.failed)}</div><div class="stat-label">未通过因子</div></div>
    <div class="stat"><div class="stat-value" style="color:#f59e0b">{_avg_judge_score(report.passed + report.extensions_passed)}</div><div class="stat-label">平均Judge评分</div></div>
  </div>
</div>

<div class="card">
  <h2>质量门控标准</h2>
  <table><tr><th>指标</th><th>门槛</th></tr>{gates_html}</table>
</div>

<div class="card">
  <h2>因子评估结果</h2>
  <table>
    <tr><th>因子名</th><th>IC均值</th><th>ICIR</th><th>IC衰减比</th><th>覆盖率</th><th>月胜率</th><th>最大相关</th><th>IC走势</th><th>Judge评分</th><th>状态</th></tr>
    {rows_html}
  </table>
</div>

<div class="card">
  <h2>参考论文</h2>
  <ul>{paper_items}</ul>
</div>

</div>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        report.report_path = path
        print(f"[Report] 报告已保存: {path}")
        return path
