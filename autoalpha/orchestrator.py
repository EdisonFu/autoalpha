"""
分阶段编排器：把流水线拆成可中断、可观测的阶段。

与 pipeline.AutoAlphaPipeline 的关系：
  复用它的 agents（检索/提取/实现），但替换评估层为 engine.SplitEngine（IS/OOS 双层门控），
  并把控制流改成生成器 —— 每阶段 yield FlowEvent，调用方决定是否继续。
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, Iterator, List, Optional

import pandas as pd

from .flow import (
    Stage, EventKind, FlowEvent, FlowSession,
    evaluation_to_dict, nav_to_points,
)
from .llm import LLMClient
from .db import FactorDB
from .engine import SplitEngine, DATA_START, DATA_END
from .data_loader import load_klines
from .agents.input_classifier import classify_input
from .agents.paper_search_agent import PaperSearchAgent
from .agents.factor_extract_agent import FactorExtractAgent
from .agents.factor_implement_agent import FactorImplementAgent, classify_error
from .agents.factor_extension_agent import FactorExtensionAgent
from .agents.translate_agent import TranslateAgent

DEFAULT_KLINES = "/Users/fuzhendong/GolandProjects/quant/factor-signal/data/klines"
MAX_IMPLEMENT_RETRIES = 2
MAX_EXTENSIONS = 3


class FlowOrchestrator:
    def __init__(self, klines_dir: str = DEFAULT_KLINES, freq: str = "etf-daily",
                 db_path: str = "autoalpha.db", output_dir: str = "./output",
                 llm_config: dict = None):
        self.klines_dir = klines_dir
        self.freq = freq
        self.output_dir = output_dir
        self.db = FactorDB(db_path)
        self.llm = LLMClient(llm_config)

        self.paper_search = PaperSearchAgent(self.llm)
        self.extract = FactorExtractAgent(self.llm)
        self.implement = FactorImplementAgent(
            self.llm, output_dir=os.path.join(output_dir, "factors"))
        self.extension = FactorExtensionAgent(self.llm)
        self.translate = TranslateAgent(self.llm)

        self._engine: Optional[SplitEngine] = None
        self._raw_data: Optional[Dict] = None

    # ── 阶段 1：论文检索 ────────────────────────────────────────────────────

    def search_papers(self, sess: FlowSession, idea: str,
                      max_results: int = 10) -> Iterator[FlowEvent]:
        sess.idea = idea
        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.SEARCH,
                                    f"检索与「{idea}」相关的论文"))
        kind, conf = classify_input(idea)
        yield sess.record(FlowEvent(EventKind.LOG, Stage.SEARCH,
                                    f"输入类型={kind} 置信度={conf:.2f}"))
        papers: List[Dict] = []
        if kind == "idea" and conf >= 0.6:
            try:
                papers, _ = self.paper_search.search(idea, max_results=max_results)
            except Exception as e:
                yield sess.record(FlowEvent(EventKind.LOG, Stage.SEARCH,
                                            f"检索失败，改用原始想法: {e}"))
        sess.papers = papers

        zh: List[Dict] = []
        if papers:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.SEARCH,
                                        f"生成 {len(papers)} 篇中文概述（批量单次调用）"))
            zh = self.translate.describe_papers(papers)

        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.SEARCH, f"找到 {len(papers)} 篇论文",
            {"papers": [{"idx": i, "title": p.get("title", ""),
                         "url": p.get("pdf_url", ""), "date": p.get("date", ""),
                         "abstract": (p.get("abstract") or "")[:400],
                         "zh_summary": (zh[i].get("summary", "") if i < len(zh) else ""),
                         "zh_relevance": (zh[i].get("relevance", "") if i < len(zh) else "")}
                        for i, p in enumerate(papers)]}))
        yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.SEARCH, "检索完成"))

    # ── 阶段 2：提取因子假设 ────────────────────────────────────────────────

    def extract_specs(self, sess: FlowSession,
                      selected: List[Dict]) -> Iterator[FlowEvent]:
        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.EXTRACT, "提取因子假设"))
        if not selected:
            selected = [{"title": sess.idea, "abstract": sess.idea,
                         "pdf_url": "", "date": ""}]
            yield sess.record(FlowEvent(EventKind.LOG, Stage.EXTRACT,
                                        "未选论文，直接基于研究方向生成假设"))
        sess.selected_papers = selected

        specs = []
        for i, paper in enumerate(selected, 1):
            yield sess.record(FlowEvent(
                EventKind.PROGRESS, Stage.EXTRACT,
                f"解析论文 {i}/{len(selected)}: {paper.get('title','')[:50]}",
                {"current": i, "total": len(selected)}))
            try:
                specs.extend(self.extract.extract(paper))
            except Exception as e:
                yield sess.record(FlowEvent(EventKind.LOG, Stage.EXTRACT,
                                            f"解析失败: {e}"))
        sess.specs = specs

        zh: List[Dict] = []
        if specs:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.EXTRACT,
                                        f"生成 {len(specs)} 个因子中文说明（批量单次调用）"))
            zh = self.translate.describe_factors(specs)
        sess.spec_zh = zh

        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.EXTRACT, f"提取到 {len(specs)} 个因子假设",
            {"specs": [{"idx": i, "factor_name": sp.factor_name,
                        "description": sp.description, "hypothesis": sp.hypothesis,
                        "formula": sp.formula, "parameters": sp.parameters,
                        "zh_what": (zh[i].get("what", "") if i < len(zh) else ""),
                        "zh_why": (zh[i].get("why", "") if i < len(zh) else ""),
                        "zh_how": (zh[i].get("how", "") if i < len(zh) else "")}
                       for i, sp in enumerate(specs)]}))
        yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.EXTRACT, "提取完成"))

    # ── 阶段 3：加载数据 + 建切分 ──────────────────────────────────────────

    def prepare_data(self, sess: FlowSession) -> Iterator[FlowEvent]:
        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.DATA, "加载行情数据"))
        if self._engine is not None:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.DATA, "复用已加载数据"))
        else:
            try:
                self._raw_data = load_klines(self.klines_dir, self.freq)
                self._engine = SplitEngine(self._raw_data)
            except Exception as e:
                sess.error = f"数据加载失败: {e}"
                yield sess.record(FlowEvent(EventKind.ERROR, Stage.ERROR, sess.error))
                return
        e = self._engine
        g = e.gates_for("is")
        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.DATA, "数据与切分就绪",
            {"n_symbols": int(e.price_full.shape[1]),
             "n_days": int(len(e.price_full)),
             "is_range": [e.price_is.index[0].strftime("%Y-%m-%d"),
                          e.is_end.strftime("%Y-%m-%d")],
             "oos_range": [e.oos_start.strftime("%Y-%m-%d"),
                           e.price_oos.index[-1].strftime("%Y-%m-%d")],
             "is_days": int(len(e.price_is)), "oos_days": int(len(e.price_oos)),
             "universe_width": round(g["_width"], 1),
             "ic_mean_floor": round(g["ic_mean_floor"], 5),
             "t_threshold": round(g["t_threshold"], 3)}))
        yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.DATA, "数据就绪"))

    # ── 阶段 4+5：实现 + 评估 ──────────────────────────────────────────────

    def run_factors(self, sess: FlowSession,
                    spec_indices: Optional[List[int]] = None
                    ) -> Iterator[FlowEvent]:
        if self._engine is None:
            yield sess.record(FlowEvent(EventKind.ERROR, Stage.ERROR, "数据未就绪"))
            return
        specs = sess.specs if spec_indices is None else \
            [sess.specs[i] for i in spec_indices if 0 <= i < len(sess.specs)]

        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.IMPLEMENT,
                                    f"实现并评估 {len(specs)} 个因子"))
        passed_navs = [e.nav_net for e in sess.evaluations
                       if e.status == "PASS" and e.nav_net is not None]

        for i, spec in enumerate(specs, 1):
            yield sess.record(FlowEvent(
                EventKind.PROGRESS, Stage.IMPLEMENT,
                f"[{i}/{len(specs)}] {spec.factor_name}",
                {"current": i, "total": len(specs), "factor": spec.factor_name}))

            factor, err = yield from self._implement_with_retry(sess, spec)
            if factor is None:
                ev = _failed_eval(spec.factor_name, err)
                sess.evaluations.append(ev)
                yield sess.record(FlowEvent(
                    EventKind.DATA, Stage.EVALUATE, f"{spec.factor_name} 实现失败",
                    {"evaluation": evaluation_to_dict(ev)}))
                continue

            yield sess.record(FlowEvent(EventKind.LOG, Stage.EVALUATE,
                                        f"评估 {spec.factor_name}（IS/OOS 双层门控）"))
            try:
                ev = self._engine.full_evaluate(
                    factor, spec.factor_name, existing_navs=passed_navs,
                    n_trials=max(1, len(sess.evaluations) + 1))
            except Exception as e:
                ev = _failed_eval(spec.factor_name, f"评估异常: {e}")
            sess.evaluations.append(ev)

            if ev.status == "PASS":
                if ev.nav_net is not None:
                    passed_navs.append(ev.nav_net)
                try:
                    self.db.save_factor(_db_row(ev), spec)
                except Exception as e:
                    yield sess.record(FlowEvent(EventKind.LOG, Stage.EVALUATE,
                                                f"入库失败(非致命): {e}"))

            yield sess.record(FlowEvent(
                EventKind.DATA, Stage.EVALUATE,
                f"{spec.factor_name} → {ev.status}",
                {"evaluation": evaluation_to_dict(ev)}))

        n_pass = sum(1 for e in sess.evaluations if e.status == "PASS")
        usage = self.llm.meter.summary()
        yield sess.record(FlowEvent(
            EventKind.STAGE_END, Stage.EVALUATE,
            f"完成：{n_pass}/{len(sess.evaluations)} 通过双层门控",
            {"n_pass": n_pass, "n_total": len(sess.evaluations),
             "usage": usage}))

    def _implement_with_retry(self, sess: FlowSession, spec):
        """生成代码 → 沙箱执行 → 读回因子值。返回 (factor_series|None, err)。"""
        err_ctx = None
        for attempt in range(1, MAX_IMPLEMENT_RETRIES + 2):
            csv_path, impl_err = self.implement.implement(
                spec, self._raw_data, err_ctx)
            if impl_err:
                etype = classify_error(impl_err)
                yield sess.record(FlowEvent(
                    EventKind.LOG, Stage.IMPLEMENT,
                    f"第{attempt}次失败[{etype}]: {impl_err[:120]}"))
                if etype in ("TIMEOUT", "ENV"):
                    return None, f"[{etype}] 不可恢复: {impl_err[:200]}"
                if attempt > MAX_IMPLEMENT_RETRIES:
                    return None, f"实现失败({attempt}次): {impl_err[:200]}"
                err_ctx = impl_err
                continue
            try:
                factor = _load_factor_csv(csv_path)
            except Exception as e:
                if attempt > MAX_IMPLEMENT_RETRIES:
                    return None, f"因子文件读取失败: {e}"
                err_ctx = f"输出的 CSV 无法解析为 MultiIndex(date,symbol) 序列: {e}"
                continue

            bad = _sparse_dates(factor, self._engine) or _cross_section_degenerate(factor)
            if bad is None:
                return factor, None
            yield sess.record(FlowEvent(
                EventKind.LOG, Stage.IMPLEMENT, f"第{attempt}次截面退化: {bad}"))
            if attempt > MAX_IMPLEMENT_RETRIES:
                return None, f"截面无区分度: {bad}"
            err_ctx = (
                f"[FACTOR_SHAPE_BAD] 代码能运行，但产出的因子形状不可用：{bad}\n"
                "若是日期稀疏：必须为【每个交易日】输出因子值，不要自行重采样到月末；\n"
                "  调仓频率由回测框架决定，想表达月度逻辑请用更长的 lookback 窗口。\n"
                "若是截面无区分度：\n"
                "这类因子的 IC 恒为 0，无法用于选股。常见成因：\n"
                "  1. 用 np.tile / broadcast 把一个时序标量复制给所有标的 —— "
                "     截面因子必须让【不同标的在同一天取不同值】\n"
                "  2. 只算了市场层面的聚合量（如指数动量），没有落到单个标的\n"
                "修复方向：对每个标的分别计算其自身的特征，再做截面 rank 归一化。"
            )
        return None, "未知失败"

    # ── 阶段 6：延伸 ───────────────────────────────────────────────────────

    def run_extensions(self, sess: FlowSession) -> Iterator[FlowEvent]:
        base = [e for e in sess.evaluations if e.status == "PASS"]
        if not base:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.EXTEND,
                                        "无通过因子，跳过延伸"))
            return
        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.EXTEND,
                                    f"为 {len(base)} 个通过因子生成延伸"))
        for ev in base:
            spec = next((s for s in sess.specs if s.factor_name == ev.factor_name), None)
            if spec is None:
                continue
            try:
                exts = self.extension.propose_extensions(spec, _db_row(ev))
            except Exception as e:
                yield sess.record(FlowEvent(EventKind.LOG, Stage.EXTEND,
                                            f"延伸生成失败: {e}"))
                continue
            if exts:
                sess.specs.extend(exts[:MAX_EXTENSIONS])
                yield sess.record(FlowEvent(
                    EventKind.DATA, Stage.EXTEND,
                    f"{ev.factor_name} → {len(exts[:MAX_EXTENSIONS])} 个延伸",
                    {"specs": [{"factor_name": s.factor_name,
                                "hypothesis": s.hypothesis,
                                "extension_type": s.extension_type}
                               for s in exts[:MAX_EXTENSIONS]]}))
        yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.EXTEND, "延伸完成"))

    # ── 全自动串联 ─────────────────────────────────────────────────────────

    def run_all(self, sess: FlowSession, idea: str,
                auto_select: int = 1) -> Iterator[FlowEvent]:
        """不中断地跑完整条链（CLI 用）。Web 端用分阶段接口。"""
        try:
            yield from self.search_papers(sess, idea)
            yield from self.prepare_data(sess)
            if sess.error:
                return
            yield from self.extract_specs(sess, sess.papers[:auto_select])
            if not sess.specs:
                yield sess.record(FlowEvent(EventKind.ERROR, Stage.ERROR,
                                            "未提取到任何因子假设"))
                return
            yield from self.run_factors(sess)
            yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.DONE, "流水线完成"))
        except Exception as e:
            sess.error = f"{e}\n{traceback.format_exc()[-800:]}"
            yield sess.record(FlowEvent(EventKind.ERROR, Stage.ERROR, str(e)))


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _load_factor_csv(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=[0, 1], parse_dates=[0])
    df.index.names = ["date", "symbol"]
    df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
    return df.iloc[:, 0].rename("factor_value")


def _sparse_dates(factor: pd.Series, engine, min_ratio: float = 0.5) -> Optional[str]:
    """
    检测日期稀疏：因子只在少数日期有值（典型成因是 LLM 自行做了月末重采样）。

    调仓频率应由回测框架决定。因子若只输出月末值，coverage 会塌到 ~5%，
    分层组合只剩几十个调仓点，收益严重失真（实测出现过 ±290% 的分层年化）。
    """
    if engine is None:
        return None
    try:
        fdates = factor.index.get_level_values(0).unique()
    except Exception:
        return None
    lo, hi = engine.price_full.index[0], engine.price_full.index[-1]
    expected = len(engine.price_full)
    got = len([d for d in fdates if lo <= d <= hi])
    if expected and got / expected < min_ratio:
        return (f"因子仅覆盖 {got}/{expected} 个交易日（{got/expected:.1%}）—— "
                f"疑似自行做了月末/周末重采样")
    return None


def _cross_section_degenerate(factor: pd.Series, sample_days: int = 60,
                              min_unique_ratio: float = 0.1) -> Optional[str]:
    """
    检测截面退化：同一天所有标的取值几乎相同。

    这类因子（典型成因是把市场级标量广播给全部标的）IC 恒为 0，
    但代码能正常运行、CSV 也正常产出，不做检测就会白白消耗一轮回测。
    返回 None 表示正常，否则返回问题描述。
    """
    try:
        wide = factor.unstack(level=-1)
    except Exception:
        return None
    if wide.empty or wide.shape[1] < 3:
        return None
    rows = wide.iloc[::max(1, len(wide) // sample_days)]
    flat = 0
    checked = 0
    for _, row in rows.iterrows():
        v = row.dropna()
        if len(v) < 3:
            continue
        checked += 1
        if v.nunique() / len(v) < min_unique_ratio:
            flat += 1
    if checked == 0:
        return "所有采样日的有效标的数均不足 3 个"
    if flat / checked > 0.8:
        return (f"{flat}/{checked} 个采样日上，截面取值几乎完全相同"
                f"（唯一值占比 < {min_unique_ratio:.0%}）")
    return None


def _failed_eval(name: str, reason: str):
    from .metrics import FactorEvaluation
    return FactorEvaluation(factor_name=name, status="FAIL_IS", fail_reason=reason)


def _db_row(ev):
    """FactorEvaluation → 旧 FactorResult 形状，供 db / extension agent 复用。"""
    from .pipeline_state import FactorResult
    ic = ev.oos_ic or ev.is_ic
    lm = ev.oos_layers or ev.is_layers
    return FactorResult(
        factor_name=ev.factor_name,
        ic_mean=ic.ic_mean_1d if ic else 0.0,
        icir=ic.icir_daily if ic else 0.0,
        ic_decay_ratio=(ic.ic_halflife if ic and ic.ic_halflife == ic.ic_halflife else 0.0),
        coverage=ic.coverage if ic else 0.0,
        monthly_win_rate=ic.monthly_win_rate if ic else 0.0,
        max_correlation=ev.max_nav_corr,
        status="PASS" if ev.status == "PASS" else "FAIL",
        fail_reason=ev.fail_reason,
        code_path=ev.code_path,
        ic_series=[],
    )
