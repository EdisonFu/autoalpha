"""
分阶段编排器：把流水线拆成可中断、可观测的阶段。

与 pipeline.AutoAlphaPipeline 的关系：
  复用它的 agents（检索/提取/实现），但替换评估层为 engine.SplitEngine（IS/OOS 双层门控），
  并把控制流改成生成器 —— 每阶段 yield FlowEvent，调用方决定是否继续。
"""

from __future__ import annotations

import os
import re
import traceback
from typing import Dict, Iterator, List, Optional

import pandas as pd

from .flow import (
    Stage, EventKind, FlowEvent, FlowSession,
    evaluation_to_dict,
)
from .llm import LLMClient
from .db import FactorDB
from .data_loader import load_klines
from .engine import SplitEngine
from .universe import (load_names, build_universe, describe_for_prompt,
                       DEFAULT_PRESET)
from .agents.input_classifier import classify_input
from .agents.paper_search_agent import PaperSearchAgent
from .agents.factor_extract_agent import FactorExtractAgent
from .agents.factor_implement_agent import FactorImplementAgent, classify_error
from .agents.factor_extension_agent import FactorExtensionAgent
from .agents.translate_agent import TranslateAgent
from .agents.intent_agent import IntentAgent
from .agents.research_loop_agent import ResearchLoopAgent

# 默认读项目内 data/klines（仓库附沪深300ETF样例数据）。
# 调用方（server.py / main.py）会先查 AUTOALPHA_KLINES 环境变量，
# 此处只是最后的回退 —— 此前硬编码本机绝对路径，别人 clone 下来
# 不设环境变量时拿到的是死路径。
DEFAULT_KLINES = "./data/klines"
MAX_IMPLEMENT_RETRIES = 2

# 因子形状不可用时给 LLM 的修复提示。串行与并发两条实现路径共用，
# 避免两边提示词各自演化出不一致的说法。
_SHAPE_HINT = (
    "[FACTOR_SHAPE_BAD] 代码能运行，但产出的因子形状不可用：{reason}\n"
    "若是日期稀疏：必须为【每个交易日】输出因子值，不要自行重采样到月末；\n"
    "  调仓频率由回测框架决定，想表达月度逻辑请用更长的 lookback 窗口。\n"
    "若是截面无区分度：这类因子的 IC 恒为 0，无法用于选股。常见成因：\n"
    "  1. 用 np.tile / broadcast 把时序标量复制给所有标的 —— "
    "截面因子必须让【不同标的在同一天取不同值】\n"
    "  2. 只算了市场层面的聚合量（如指数动量），没有落到单个标的\n"
    "修复方向：对每个标的分别计算其自身的特征，再做截面 rank 归一化。"
)
MAX_EXTENSIONS = 3


class FlowOrchestrator:
    def __init__(self, klines_dir: str = DEFAULT_KLINES, freq: str = "etf-daily",
                 db_path: str = "autoalpha.db", output_dir: str = "./output",
                 llm_config: dict = None, universe_preset: str = DEFAULT_PRESET,
                 max_workers: int = 3):
        self.klines_dir = klines_dir
        self.freq = freq
        self.output_dir = output_dir
        self.db = FactorDB(db_path)
        self.llm = LLMClient(llm_config)

        self.paper_search = PaperSearchAgent(self.llm, db_path=db_path)
        self.extract = FactorExtractAgent(self.llm)
        self.implement = FactorImplementAgent(
            self.llm, output_dir=os.path.join(output_dir, "factors"))
        self.extension = FactorExtensionAgent(self.llm)
        self.translate = TranslateAgent(self.llm)
        self.intent_agent = IntentAgent(self.llm)
        self.research_loop = ResearchLoopAgent(self.llm)

        self.universe_preset = universe_preset
        self.max_workers = max(1, int(max_workers))
        self._engine: Optional[SplitEngine] = None
        self._raw_data: Optional[Dict] = None
        self._profile: Dict = {}
        self._data_context: str = ""

    # ── 阶段 1：论文检索 ────────────────────────────────────────────────────

    # ── Step 1：意图识别 ───────────────────────────────────────────────────

    def recognize_intent(self, sess: FlowSession, idea: str,
                         override: Optional[dict] = None
                         ) -> Iterator[FlowEvent]:
        """
        把用户输入归一成结构化检索意图。实测同一意图不同表述的召回量
        相差 800 倍（"ETF轮动" 18 篇 vs 规范英文表述 14,494 篇），
        这一步就是填这个落差，并决定去哪些源找。
        """
        sess.idea = idea
        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.INTENT,
                                    "识别研究意图"))
        intent = self.intent_agent.recognize(idea)

        # 用户在界面上改过意图就以用户的为准
        if override:
            for k, v in override.items():
                if hasattr(intent, k) and v not in (None, "", []):
                    setattr(intent, k, v)

        sess.intent = intent
        self.paper_search.intent = intent

        d = intent.to_dict()
        d["market_label"] = intent.market_label
        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.INTENT,
            intent.restated_zh or intent.canonical_en, {"intent": d}))

        if intent.market == "unknown":
            yield sess.record(FlowEvent(
                EventKind.LOG, Stage.INTENT,
                "未识别到市场。不同市场制度与微观结构差异大，"
                "建议在研究方向里写明（如「A股」「港股」「美股」）"))

        if intent.confidence < 0.5:
            yield sess.record(FlowEvent(
                EventKind.LOG, Stage.INTENT,
                f"意图置信度较低（{intent.confidence:.2f}），"
                f"建议确认或补充研究方向后再检索"))

        yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.INTENT, "意图识别完成"))

    # ── Step 2：多源资料检索 ───────────────────────────────────────────────

    def search_papers(self, sess: FlowSession, idea: str,
                      max_results: int = 10, page: int = 1
                      ) -> Iterator[FlowEvent]:
        sess.idea = idea
        if sess.intent is not None:
            self.paper_search.intent = sess.intent
        more = page > 1
        yield sess.record(FlowEvent(
            EventKind.STAGE_START, Stage.SEARCH,
            f"加载第 {page} 批参考资料" if more else f"检索与「{idea}」相关的文献与研报"))
        kind, conf = classify_input(idea)
        if not more:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.SEARCH,
                                        f"输入类型={kind} 置信度={conf:.2f}"))
        papers: List[Dict] = []
        total = 0
        if kind == "idea" and conf >= 0.6:
            try:
                # search_paged 内部按 idea 缓存，翻页不重复调 arXiv 与 LLM 打分
                papers, total = self.paper_search.search_paged(
                    idea, page=page, page_size=max_results)
            except Exception as e:
                yield sess.record(FlowEvent(EventKind.LOG, Stage.SEARCH,
                                            f"检索失败，改用原始想法: {e}"))
        sess.paper_total = total
        if papers:
            sess.paper_page = page
            if page > 1:
                # 累积而非替换：sess.papers 是「已加载的全部论文」，
                # 与前端列表保持同一语义。此前第 2 批会覆盖第 1 批，
                # 导致 /api/session 的 n_papers 只报最后一批的数量。
                {id(x) for x in sess.papers}
                known = {(x.get("title"), x.get("pdf_url")) for x in sess.papers}
                sess.papers = sess.papers + [
                    p for p in papers
                    if (p.get("title"), p.get("pdf_url")) not in known]
            else:
                sess.papers = papers
        elif page > 1:
            # 这一批为空：保留上一批，不要把列表清掉
            yield sess.record(FlowEvent(
                EventKind.LOG, Stage.SEARCH,
                f"没有更多资料了（共 {total} 篇，已全部加载）"))
            papers = sess.papers
        else:
            sess.papers = papers

        zh: List[Dict] = []
        if papers:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.SEARCH,
                                        f"生成 {len(papers)} 篇中文概述（批量单次调用）"))
            zh = self.translate.describe_papers(papers)

        base = (page - 1) * max_results if papers else 0
        has_more = bool(papers) and total > base + len(papers)
        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.SEARCH,
            f"找到 {len(papers)} 篇参考资料"
            + (f"（第 {page} 批，已加载 {len(sess.papers)}/{total} 篇）"
               if total > len(papers) else ""),
            {"page": page, "total": total, "has_more": has_more,
             "papers": [{"idx": base + i, "title": p.get("title", ""),
                         "url": p.get("pdf_url", ""), "date": p.get("date", ""),
                         "abstract": (p.get("abstract") or "")[:400],
                         "source": p.get("source", ""),
                         "venue": p.get("venue", ""),
                         "doc_type": p.get("doc_type", "paper"),
                         "citations": p.get("citations", 0),
                         "also_in": p.get("also_in", []),
                         # 摘要是原源就有，还是跨库回填来的
                         "abstract_backfilled": bool(
                             p.get("abstract_backfilled")),
                         # 中文社区帖的质量信号（精华/收藏）
                         "is_best": bool(p.get("is_best")),
                         "collections": int(p.get("collections") or 0),
                         "zh_summary": (zh[i].get("summary", "") if i < len(zh) else ""),
                         "zh_relevance": (zh[i].get("relevance", "") if i < len(zh) else "")}
                        for i, p in enumerate(papers)]}))
        yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.SEARCH, "检索完成"))

    # ── 阶段 2：提取因子假设 ────────────────────────────────────────────────

    def papers_by_index(self, sess: FlowSession, idxs: List[int]) -> List[Dict]:
        """按全局索引取论文（支持跨页选择）。"""
        allp = self.paper_search._get_results(sess.idea) if sess.idea else []
        out = []
        for i in idxs:
            if 0 <= i < len(allp):
                out.append(allp[i])
            elif 0 <= i < len(sess.papers):
                out.append(sess.papers[i])
        return out

    def extract_specs(self, sess: FlowSession,
                      selected: List[Dict],
                      append: bool = False) -> Iterator[FlowEvent]:
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
                specs.extend(self.extract.extract(paper, self._data_context))
            except Exception as e:
                yield sess.record(FlowEvent(EventKind.LOG, Stage.EXTRACT,
                                            f"解析失败: {e}"))
        base_n = len(sess.specs) if append else 0
        sess.specs = (sess.specs + specs) if append else specs

        zh: List[Dict] = []
        if specs:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.EXTRACT,
                                        f"生成 {len(specs)} 个因子中文说明（批量单次调用）"))
            zh = self.translate.describe_factors(specs)
        sess.spec_zh = zh

        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.EXTRACT, f"提取到 {len(specs)} 个因子假设",
            {"append": append,
             "specs": [{"idx": base_n + i, "factor_name": sp.factor_name,
                        "description": sp.description, "hypothesis": sp.hypothesis,
                        "formula": sp.formula, "parameters": sp.parameters,
                        "zh_what": (zh[i].get("what", "") if i < len(zh) else ""),
                        "zh_why": (zh[i].get("why", "") if i < len(zh) else ""),
                        "zh_how": (zh[i].get("how", "") if i < len(zh) else ""),
                        # 参数读自论文还是模型自己定的 —— 用户必须能分辨
                        "provenance": sp.provenance,
                        "param_sources": sp.param_sources,
                        "provenance_note": (sp.source_paper or {}).get(
                            "provenance_note", ""),
                        # 市场迁移声明：用户需要知道哪些是原文、哪些是替换
                        "migration": sp.migration or {}}
                       for i, sp in enumerate(specs)]}))
        yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.EXTRACT, "提取完成"))

    # ── 阶段 3：加载数据 + 建切分 ──────────────────────────────────────────

    def prepare_data(self, sess: FlowSession) -> Iterator[FlowEvent]:
        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.DATA, "准备行情数据"))
        if self._engine is not None:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.DATA, "复用已加载数据"))
        else:
            try:
                raw = load_klines(self.klines_dir, self.freq)
                names = load_names(self.klines_dir, self.freq)
                # 按资产类别过滤，保证截面可比（修 E2E_REVIEW P-3）
                self._raw_data, self._profile = build_universe(
                    raw, names, self.universe_preset)
                self._engine = SplitEngine(self._raw_data)
                self._data_context = describe_for_prompt(
                    self._profile,
                    ["open", "high", "low", "close", "volume", "amount"],
                    (self._engine.price_full.index[0].strftime("%Y-%m-%d"),
                     self._engine.price_full.index[-1].strftime("%Y-%m-%d")))
            except Exception as e:
                sess.error = f"数据加载失败: {e}"
                yield sess.record(FlowEvent(EventKind.ERROR, Stage.ERROR, sess.error))
                return
        e = self._engine
        g = e.gates_for("is")
        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.DATA, "数据准备就绪",
            {"n_symbols": int(e.price_full.shape[1]),
             "n_days": int(len(e.price_full)),
             "is_range": [e.price_is.index[0].strftime("%Y-%m-%d"),
                          e.is_end.strftime("%Y-%m-%d")],
             "oos_range": [e.oos_start.strftime("%Y-%m-%d"),
                           e.price_oos.index[-1].strftime("%Y-%m-%d")],
             "is_days": int(len(e.price_is)), "oos_days": int(len(e.price_oos)),
             "universe_width": round(g["_width"], 1),
             "ic_mean_floor": round(g["ic_mean_floor"], 5),
             "t_threshold": round(g["t_threshold"], 3),
             "preset": self._profile.get("preset", ""),
             "n_total": self._profile.get("n_total", 0),
             "dist_kept": self._profile.get("dist_kept", {}),
             "excluded": self._profile.get("excluded", [])}))
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

        available = set()
        for df in list(self._raw_data.values())[:3]:
            available.update(df.columns.tolist())

        # 数据可用性前置检查（修 E2E_REVIEW P-2/P-4）：
        # 不可计算的因子直接跳过，不浪费 codegen 调用与回测。
        # 用户必须能区分「数据没有」和「测了但不达标」。
        runnable = []
        for spec in specs:
            skip = _check_data_available(spec, available)
            if skip:
                ev = _skipped_eval(spec.factor_name, skip)
                _attach_desc(ev, spec, sess)
                sess.evaluations.append(ev)
                yield sess.record(FlowEvent(
                    EventKind.DATA, Stage.EVALUATE,
                    f"{spec.factor_name} → SKIP_NO_DATA",
                    {"evaluation": evaluation_to_dict(ev)}))
            else:
                runnable.append(spec)

        # 实现阶段并发。实测单因子回测仅 1.9s，而整轮耗时约 10 分钟 ——
        # 时间几乎全在 LLM codegen 与沙箱执行上，这两者因子间互不依赖。
        # 评估仍串行：passed_navs 是去重基线，必须按确定顺序累积，
        # 否则同一批因子的判重结果会随线程调度变化。
        impl_results = {}
        if len(runnable) > 1 and self.max_workers > 1:
            yield from self._implement_concurrent(sess, runnable, impl_results)

        for i, spec in enumerate(runnable, 1):
            yield sess.record(FlowEvent(
                EventKind.PROGRESS, Stage.IMPLEMENT,
                f"[{i}/{len(runnable)}] {spec.factor_name}",
                {"current": i, "total": len(runnable), "factor": spec.factor_name}))

            if spec.factor_name in impl_results:
                factor, err = impl_results[spec.factor_name]
            else:
                factor, err = yield from self._implement_with_retry(sess, spec)
            if factor is None:
                ev = _failed_eval(spec.factor_name, err)
                _attach_desc(ev, spec, sess)
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
            _attach_desc(ev, spec, sess)
            sess.evaluations.append(ev)

            if ev.status == "PASS":
                if ev.nav_net is not None:
                    passed_navs.append(ev.nav_net)
                # 入库前再查一次：PASS 的因子会成为后续去重基线，
                # 一条脏数据会污染所有后续判断。实测库里曾存有
                # llm_news_momentum_score（依赖不存在的新闻数据，
                # 靠 LLM 自编代理变量通过了旧门控）。
                dirty = _check_data_available(spec, available)
                if dirty:
                    ev.status = "SKIP_NO_DATA"
                    ev.fail_reason = f"通过门控但数据不可得，拒绝入库: {dirty}"
                    yield sess.record(FlowEvent(
                        EventKind.LOG, Stage.EVALUATE,
                        f"{spec.factor_name} 拒绝入库: {dirty}"))
                    continue
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
        n_skip = sum(1 for e in sess.evaluations if e.status == "SKIP_NO_DATA")
        n_tested = len(sess.evaluations) - n_skip
        usage = self.llm.meter.summary()
        msg = f"完成：{n_pass}/{n_tested} 通过双层门控"
        if n_skip:
            msg += f"（另有 {n_skip} 个因数据缺失未测）"
        yield sess.record(FlowEvent(
            EventKind.STAGE_END, Stage.EVALUATE, msg,
            {"n_pass": n_pass, "n_total": len(sess.evaluations),
             "n_tested": n_tested, "n_skip": n_skip, "usage": usage}))

    def _implement_concurrent(self, sess: FlowSession, specs: List,
                              out: Dict) -> Iterator[FlowEvent]:
        """
        并发跑 codegen + 沙箱。事件在主线程按完成顺序发出，
        因为生成器不能跨线程 yield。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = min(self.max_workers, len(specs))
        yield sess.record(FlowEvent(
            EventKind.LOG, Stage.IMPLEMENT,
            f"并发生成 {len(specs)} 个因子的代码（{n} 路并行）"))

        logs: List[tuple] = []          # (factor_name, message)

        def _work(spec):
            return spec.factor_name, self._implement_once(spec, logs)

        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(_work, sp): sp for sp in specs}
            done = 0
            for fut in as_completed(futs):
                sp = futs[fut]
                done += 1
                try:
                    name, res = fut.result()
                    out[name] = res
                except Exception as e:
                    out[sp.factor_name] = (None, f"并发实现异常: {e}")
                yield sess.record(FlowEvent(
                    EventKind.PROGRESS, Stage.IMPLEMENT,
                    f"代码生成 {done}/{len(specs)}: {sp.factor_name}",
                    {"current": done, "total": len(specs)}))

        for name, msg in logs:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.IMPLEMENT,
                                        f"{name}: {msg}"))

    def _implement_once(self, spec, logs: List):
        """
        线程内执行：生成代码 → 沙箱 → 读回因子值。
        不 yield 事件（跨线程不可行），消息先收集到 logs。
        """
        err_ctx = None
        for attempt in range(1, MAX_IMPLEMENT_RETRIES + 2):
            csv_path, impl_err = self.implement.implement(
                spec, self._raw_data, err_ctx)
            if impl_err:
                etype = classify_error(impl_err)
                logs.append((spec.factor_name,
                             f"第{attempt}次失败[{etype}]: {impl_err[:110]}"))
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

            bad = (_sparse_dates(factor, self._engine)
                   or _cross_section_degenerate(factor))
            if bad is None:
                return factor, None
            logs.append((spec.factor_name, f"第{attempt}次因子形状不可用: {bad[:110]}"))
            if attempt > MAX_IMPLEMENT_RETRIES:
                return None, f"因子形状不可用: {bad}"
            err_ctx = _SHAPE_HINT.format(reason=bad)
        return None, "未知失败"

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
            err_ctx = _SHAPE_HINT.format(reason=bad)
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

    # ── 阶段 7：反思 → 下一轮（auto research 的 loop）─────────────────────

    MAX_ROUNDS = 4

    def reflect_round(self, sess: FlowSession) -> Iterator[FlowEvent]:
        """
        本轮结束，把结论变成下一轮该查什么。

        记账在反思之前做：无论反思成功与否，这一轮的试验次数都要留下，
        否则选择性偏差会静默累积。
        """
        n_pass = sum(1 for e in sess.evaluations if e.status == "PASS")
        n_tried = len(sess.evaluations)
        sess.n_factor_trials += n_tried
        sess.rounds.append({
            "round": sess.round_no,
            "query": getattr(sess.intent, "canonical_en", ""),
            "n_specs": len(sess.specs),
            "n_evals": n_tried,
            "n_pass": n_pass,
            "failures": [
                {"factor": e.factor_name, "status": e.status,
                 "reason": (e.fail_reason or "")[:200]}
                for e in sess.evaluations if e.status != "PASS"],
        })

        yield sess.record(FlowEvent(
            EventKind.LOG, Stage.DONE,
            f"第 {sess.round_no} 轮结束：{n_tried} 个因子，{n_pass} 通过；"
            f"累计试验 {sess.n_factor_trials} 个"))

        if sess.round_no >= self.MAX_ROUNDS:
            yield sess.record(FlowEvent(
                EventKind.LOG, Stage.DONE,
                f"已达轮次上限 {self.MAX_ROUNDS}。注意：{sess.n_factor_trials} 次"
                f"试验里挑出的因子存在选择性偏差，门控通过率不代表泛化能力。"))
            yield FlowEvent(EventKind.AWAIT_INPUT, Stage.DONE,
                            "研究结束", {"next": "done", "can_extend": False,
                                        "rounds": sess.rounds,
                                        "n_factor_trials": sess.n_factor_trials})
            return

        yield sess.record(FlowEvent(EventKind.STAGE_START, Stage.DONE,
                                    "反思本轮，规划下一轮研究方向"))
        nxt = self.research_loop.reflect(sess.intent, sess.evaluations,
                                        sess.round_no, self.MAX_ROUNDS)
        if not nxt:
            # 失败必须可见：静默退化成「再跑一遍」会让用户以为系统在工作
            err = self.research_loop.last_error or "反思失败"
            yield sess.record(FlowEvent(EventKind.LOG, Stage.DONE,
                                        f"无法规划下一轮：{err}"))
            yield FlowEvent(EventKind.AWAIT_INPUT, Stage.DONE,
                            f"反思失败：{err}", {"next": "done",
                                              "can_extend": False,
                                              "rounds": sess.rounds})
            return

        sess.next_proposal = nxt
        yield sess.record(FlowEvent(
            EventKind.DATA, Stage.DONE, nxt.get("next_direction_zh", ""),
            {"proposal": nxt, "round": sess.round_no,
             "rounds": sess.rounds, "n_factor_trials": sess.n_factor_trials}))
        yield FlowEvent(EventKind.AWAIT_INPUT, Stage.DONE,
                        "是否按此方向开始下一轮研究",
                        {"next": "next_round", "proposal": nxt,
                         "can_extend": n_pass > 0,
                         "round_no": sess.round_no,
                         "max_rounds": self.MAX_ROUNDS,
                         "rounds": sess.rounds,
                         "n_factor_trials": sess.n_factor_trials})

    def start_next_round(self, sess: FlowSession,
                         proposal: Dict = None) -> Iterator[FlowEvent]:
        """
        带着上轮结论进入下一轮。保留记账，重置其余状态。

        复用已加载的数据与 IS/OOS 切分 —— 切分不能变，否则跨轮结果不可比。
        """
        nxt = proposal or getattr(sess, "next_proposal", None)
        if not nxt:
            yield sess.record(FlowEvent(EventKind.LOG, Stage.DONE,
                                        "没有下一轮方向，结束"))
            return

        sess.round_no += 1
        sess.papers = []
        sess.selected_papers = []
        sess.specs = []
        sess.evaluations = []
        sess.paper_page = 1
        sess.paper_total = 0
        # 检索缓存按 query 失效，新一轮必须真正重新召回
        self.paper_search._results_cache.clear()

        avoid = nxt.get("avoid") or []
        direction = nxt.get("next_direction_zh") or nxt.get("canonical_en", "")
        yield sess.record(FlowEvent(
            EventKind.STAGE_START, Stage.INTENT,
            f"第 {sess.round_no} 轮：{direction}"))
        if avoid:
            yield sess.record(FlowEvent(
                EventKind.LOG, Stage.INTENT,
                "本轮规避：" + "；".join(avoid[:4])))

        # 意图直接由反思结果构造，不再让 intent agent 重新理解一遍 ——
        # 它已经知道上一轮为什么失败，重新识别会丢掉这些约束。
        from dataclasses import replace
        base = sess.intent
        new_intent = replace(
            base,
            canonical_en=nxt["canonical_en"],
            canonical_zh=(nxt.get("canonical_zh") or "").strip(),
            aliases=[a for a in (base.aliases or [])
                     if a.lower() != nxt["canonical_en"].lower()][:3],
            restated_zh=direction,
            raw_input=direction,
        ) if base is not None else None

        if new_intent is None:
            yield from self.recognize_intent(sess, direction)
        else:
            sess.intent = new_intent
            self.paper_search.intent = new_intent
            yield sess.record(FlowEvent(
                EventKind.DATA, Stage.INTENT, direction,
                {"intent": new_intent.__dict__}))
            yield sess.record(FlowEvent(EventKind.STAGE_END, Stage.INTENT,
                                        "意图已根据上轮结论更新"))

        yield from self.search_papers(sess, sess.idea)
        yield FlowEvent(EventKind.AWAIT_INPUT, Stage.SELECT,
                        f"第 {sess.round_no} 轮：请选择参考资料",
                        {"next": "select", "round": sess.round_no})

    # ── 全自动串联 ─────────────────────────────────────────────────────────

    def run_all(self, sess: FlowSession, idea: str,
                auto_select: int = 1) -> Iterator[FlowEvent]:
        """不中断地跑完整条链（CLI 用）。Web 端用分阶段接口。"""
        try:
            yield from self.recognize_intent(sess, idea)
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

    # 日期齐全但绝大多数格子是 NaN：条件触发型因子（如"涨停后"）
    # 在 ETF 上几乎从不触发。实测 limit_up_volume_strength 仅 4.6% 的日期有值。
    try:
        wide = factor.unstack(level=-1)
        live = (wide.notna().sum(axis=1) >= 3).sum()
        if len(wide) and live / len(wide) < 0.3:
            return (f"因子只在 {live}/{len(wide)} 个日期（{live/len(wide):.1%}）"
                    f"有足够取值 —— 触发条件过于罕见，无法支撑逐日截面排序。"
                    f"ETF 不存在涨跌停，个股专属的极端事件条件在此数据上几乎不触发。")
    except Exception:
        pass
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

    # 只在【确实有值】的日期上采样。因子常在早期历史全为 NaN
    # （lookback 未满、标的未上市），若按全样本等间隔取点，
    # 采样点会大量落进空白区，导致判定只基于个位数的有效日。
    # 实测 limit_up_volume_strength：61 个采样点里仅 3 个有效。
    usable = wide[wide.notna().sum(axis=1) >= 3]
    if usable.empty:
        return "因子在所有日期上的有效标的数均不足 3 个"

    rows = usable.iloc[::max(1, len(usable) // sample_days)]
    flat = sum(1 for _, r in rows.iterrows()
               if (v := r.dropna()).nunique() / len(v) < min_unique_ratio)
    checked = len(rows)

    if flat / checked > 0.8:
        d = usable.iloc[len(usable) // 2].dropna()
        return (f"{flat}/{checked} 个有效采样日上截面取值几乎相同"
                f"（样例日仅 {d.nunique()} 种取值 / {len(d)} 个标的）。"
                f"常见成因：因子只输出少数离散值（如 0/1 标记），"
                f"rank 后大量并列，无法区分标的。")
    return None


# 论文常用但本数据集没有的东西。命中即判定不可计算。
_UNAVAILABLE_TOKENS = {
    "spy", "qqq", "dia", "iwm", "tlt", "gld", "vxx", "ndx", "spx",
    "vix", "vvix", "move", "skew",
    "news", "sentiment", "headline", "tweet", "social",
    "earnings", "eps", "book_value", "pe_ratio", "pb_ratio", "roe", "roa",
    "fundamental", "analyst", "estimate", "revision",
    "interest_rate", "yield_curve", "treasury", "fed", "cpi", "gdp", "macro",
    "order_book", "bid_ask", "tick", "intraday", "minute",
    "market_cap", "shares_outstanding", "float",
}


# 每个 token 对应的缺失类别，用于给出可操作的说明
_TOKEN_REASON = {
    **{t: "市场不符：这是美股标的/指数，本数据集是中国A股 ETF"
       for t in ("spy", "qqq", "dia", "iwm", "tlt", "gld", "vxx", "ndx", "spx",
                 "vix", "vvix", "move", "skew")},
    **{t: "粒度不符：需要日内/逐笔数据，本数据集只有日频收盘 OHLCV"
       for t in ("order_book", "bid_ask", "tick", "intraday", "minute")},
    **{t: "资产类型不符：ETF 是一篮子，本数据集不含成分股与基本面"
       for t in ("earnings", "eps", "book_value", "pe_ratio", "pb_ratio",
                 "roe", "roa", "fundamental", "analyst", "estimate",
                 "revision", "market_cap", "shares_outstanding", "float")},
    **{t: "未采集：本数据集只有行情，没有文本/宏观序列"
       for t in ("news", "sentiment", "headline", "tweet", "social",
                 "interest_rate", "yield_curve", "treasury", "fed",
                 "cpi", "gdp", "macro")},
}


def _check_data_available(spec, available_cols: set) -> Optional[str]:
    """
    返回 None 表示可计算，否则返回不可计算的原因。

    三道判断：
      1. LLM 自报 computable=False
      2. required_fields 里有本数据集没有的列
      3. 因子名/假设/公式里出现明确不可得的数据源（SPY/VIX/news/...）

    第 3 条是必要的兜底：实测 LLM 会一边说 computable=true，
    一边在代码里给 VIX 编一个代理变量（且沿用美股校准的阈值 25.0）。
    """
    if not getattr(spec, "computable", True):
        return f"LLM 标记为不可计算: {spec.implementation_notes[:150]}"

    req = {f.lower() for f in (getattr(spec, "required_fields", None) or [])}
    avail = {c.lower() for c in available_cols}
    missing = {f for f in req if f not in avail and f in _UNAVAILABLE_TOKENS}
    if missing:
        return f"需要本数据集不存在的字段: {', '.join(sorted(missing))}"

    # 已声明市场迁移的因子豁免文本扫描：迁移版在描述里提到原文的美股
    # 数据是「引用原文」（「论文用 VIX，本版用已实现波动率」），不是
    # 数据依赖。实测误杀：realized_vol_scaled_momentum（名字就是迁移版）
    # 因 description 提到 vix 被拒。硬检查只有上面两条 —— required_fields
    # 与 LLM 自报，它们是结构化的、不会被引用原文误触发。
    # 未声明迁移仍命中 → 拒绝：这防的是「LLM 偷偷给 VIX 编代理变量」，
    # 老的教训，扫描必须保留。
    if not getattr(spec, "migration", {}).get("changes"):
        blob = " ".join([spec.factor_name, spec.hypothesis, spec.formula,
                         spec.description]).lower()
        hits = {t for t in _UNAVAILABLE_TOKENS
                if re.search(rf"(?<![a-z0-9_]){re.escape(t)}(?![a-z0-9_])", blob)}
        if hits:
            # 按类别说明缺什么，而不是只抛一个词 —— 用户需要知道
            # 是市场不对、粒度不对、还是资产类型不对
            groups = {}
            for t in sorted(hits):
                groups.setdefault(_TOKEN_REASON.get(t, "未采集的数据"), []).append(t)
            parts = [f"{why}（{', '.join(ts)}）" for why, ts in groups.items()]
            return ("需要本数据集没有的数据：" + "；".join(parts)
                    + "。本数据集 = 中国A股市场 ETF 的日频 OHLCV。"
                    "若该信号可迁移到价格/成交量数据，请在假设里声明迁移"
                    "（migration.changes），而不是放弃。")
    return None


def _attach_desc(ev, spec, sess) -> None:
    """
    把因子描述带到评估结果上。结果卡片只显示名字时，用户得回头翻
    假设列表才知道这个因子在做什么。优先用中文说明（若已生成）。
    """
    ev.description = (spec.description or spec.hypothesis or "")[:300]
    zh = getattr(sess, "spec_zh", None) or []
    try:
        idx = sess.specs.index(spec)
    except (ValueError, AttributeError):
        idx = -1
    if 0 <= idx < len(zh):
        ev.zh_what = (zh[idx].get("what") or "")[:300]


def _skipped_eval(name: str, reason: str):
    from .metrics import FactorEvaluation
    return FactorEvaluation(factor_name=name, status="SKIP_NO_DATA",
                            fail_reason=reason)


def _failed_eval(name: str, reason: str):
    from .metrics import FactorEvaluation
    return FactorEvaluation(factor_name=name, status="FAIL_IS", fail_reason=reason)


def _db_row(ev):
    """FactorEvaluation → 旧 FactorResult 形状，供 db / extension agent 复用。"""
    from .pipeline_state import FactorResult
    ic = ev.oos_ic or ev.is_ic
    ev.oos_layers or ev.is_layers
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
