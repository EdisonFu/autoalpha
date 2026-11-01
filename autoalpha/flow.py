"""
分阶段可中断流水线。

为什么不直接复用 pipeline.run()：
  它是一次性阻塞函数，中间状态只通过 print() 输出。产品化需要流水线能
  分阶段暂停、把中间结果交出去、等用户确认再继续。

设计：每个阶段 yield 一个 FlowEvent，调用方（CLI 或 Web）决定如何呈现、
是否继续。事件既是 UI 数据源，也是进度日志。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

import pandas as pd


class Stage(str, Enum):
    INIT      = "init"
    INTENT    = "intent"        # 意图识别：把用户输入归一成结构化检索意图
    SEARCH    = "search"        # 多源资料检索
    SELECT    = "select"        # 等待用户选论文（可中断点）
    EXTRACT   = "extract"       # 提取因子假设
    DATA      = "data"          # 加载数据 + 建切分
    IMPLEMENT = "implement"     # 生成代码并执行
    EVALUATE  = "evaluate"      # IS/OOS 双层评估
    EXTEND    = "extend"        # 因子延伸
    REPORT    = "report"
    DONE      = "done"
    ERROR     = "error"


class EventKind(str, Enum):
    STAGE_START = "stage_start"
    STAGE_END   = "stage_end"
    PROGRESS    = "progress"
    DATA        = "data"         # 携带结构化结果
    AWAIT_INPUT = "await_input"  # 需要用户决策，流水线在此暂停
    LOG         = "log"
    ERROR       = "error"


@dataclass
class FlowEvent:
    kind: EventKind
    stage: Stage
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["stage"] = self.stage.value
        return json.dumps(d, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (pd.Series, pd.Index)):
        return list(o.astype(str)) if o.dtype == object else list(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


# ── 会话 ─────────────────────────────────────────────────────────────────────

@dataclass
class FlowSession:
    """一次完整挖掘的状态容器。Web 端按 id 索引。"""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    idea: str = ""
    stage: Stage = Stage.INIT
    intent: Optional[Any] = None       # SearchIntent
    papers: List[Dict] = field(default_factory=list)
    paper_page: int = 1
    paper_total: int = 0
    selected_papers: List[Dict] = field(default_factory=list)
    specs: List[Any] = field(default_factory=list)
    spec_zh: List[Dict] = field(default_factory=list)
    evaluations: List[Any] = field(default_factory=list)
    events: List[FlowEvent] = field(default_factory=list)
    error: str = ""
    # 用户决策通过这个 future 回传（Web 端用）
    _pending: Optional[asyncio.Future] = None

    def record(self, ev: FlowEvent) -> FlowEvent:
        self.events.append(ev)
        self.stage = ev.stage
        return ev


def nav_to_points(nav: pd.Series, max_points: int = 300) -> List[Dict]:
    """净值序列 → 前端图表点。超长序列等间隔下采样，避免传输过大。"""
    if nav is None or len(nav) == 0:
        return []
    s = nav.dropna()
    if len(s) > max_points:
        step = max(1, len(s) // max_points)
        s = s.iloc[::step]
    return [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 5)}
            for d, v in s.items()]


def _safe(v):
    """NaN/Inf → None。JSON 不支持这些值，FastAPI 的编码器会直接 500。"""
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def evaluation_to_dict(ev) -> Dict[str, Any]:
    """FactorEvaluation → 前端可渲染的扁平结构。所有浮点经 _safe 清洗。"""
    def _pm(m):
        return None if m is None else {
            "ann_return": _safe(m.ann_return), "ann_return_net": _safe(m.ann_return_net),
            "sharpe": _safe(m.sharpe), "sharpe_net": _safe(m.sharpe_net),
            "max_drawdown": _safe(m.max_drawdown), "calmar": _safe(m.calmar),
            "turnover_daily": _safe(m.turnover_daily),
        }

    def _ic(m):
        return None if m is None else {
            "ic_mean_1d": _safe(m.ic_mean_1d), "icir_daily": _safe(m.icir_daily),
            "icir_annual": _safe(m.icir_annual), "ic_tstat": _safe(m.ic_tstat),
            "n_obs": m.n_obs, "monthly_win_rate": _safe(m.monthly_win_rate),
            "n_months": m.n_months,
            "ic_by_lag": {str(k): _safe(v) for k, v in (m.ic_by_lag or {}).items()},
            "ic_halflife": _safe(m.ic_halflife), "coverage": _safe(m.coverage),
        }

    def _lm(m):
        return None if m is None else {
            "n_quantiles": m.n_quantiles,
            "layer_ann_returns": [_safe(x) for x in (m.layer_ann_returns or [])],
            "monotonicity": _safe(m.monotonicity),
            "long_short": _pm(m.long_short), "long_only": _pm(m.long_only),
        }

    return {
        "factor_name": ev.factor_name, "status": ev.status,
        "fail_reason": ev.fail_reason,
        "is_ic": _ic(ev.is_ic), "is_layers": _lm(ev.is_layers),
        "oos_ic": _ic(ev.oos_ic), "oos_layers": _lm(ev.oos_layers),
        "sharpe_decay": _safe(ev.sharpe_decay), "max_nav_corr": _safe(ev.max_nav_corr),
        "universe_width": _safe(ev.universe_width),
        "nav": nav_to_points(ev.nav_net),
    }
