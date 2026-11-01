"""
数据源配置与选路。

设计意图（docs/CLAUDE.md「数据准备」）：
  - 用户不配置 → 走免费平台（本地 klines / AKShare / BaoStock）
  - 用户配置了账号 → 优先走他配的平台（Tushare / 聚宽 / 米筐 / Wind）

凭据只存在于本进程内存，不落盘：密钥不该进仓库或配置文件。
依赖库未安装时该源标记为 unavailable 而非报错 —— 本环境实测
akshare/baostock/tushare 等全部未安装，所以「不可用」是常态，
界面必须如实呈现而不是让用户配了却静默失败。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 每个源：所需 python 包、需要的凭据字段、是否免费、优先级（小=优先）
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "local": {
        "label": "本地 K 线文件",
        "module": None,
        "credentials": [],
        "free": True,
        # 兜底位：必须排在所有可配置源之后，否则用户配了账号也永远走不到。
        # 曾把它设成 5（最高优先），导致配置 Tushare 后 active 仍是 local。
        "priority": 999,
        "note": "公共接口：读取本地目录，无需联网、无需账号",
    },
    "akshare": {
        "label": "AKShare",
        "module": "akshare",
        "credentials": [],
        "free": True,
        "priority": 90,
        "note": "公共数据查询接口，免费无需账号；pip install akshare",
    },
    "baostock": {
        "label": "BaoStock",
        "module": "baostock",
        "credentials": [],
        "free": True,
        "priority": 80,
        "note": "公共数据查询接口，免费无需账号；pip install baostock",
    },
    "tushare_pro": {
        "label": "Tushare Pro",
        "module": "tushare",
        "credentials": ["token"],
        "free": False,
        "priority": 30,
        "note": "积分制，数据质量高；pip install tushare",
    },
    "joinquant": {
        "label": "聚宽 JoinQuant",
        "module": "jqdatasdk",
        "credentials": ["username", "password"],
        "free": False,
        "priority": 20,
        "note": "需账号；pip install jqdatasdk",
    },
    "ricequant": {
        "label": "米筐 RiceQuant",
        "module": "rqdatac",
        "credentials": ["username", "password"],
        "free": False,
        "priority": 20,
        "note": "需账号；pip install rqdatac",
    },
    "wind": {
        "label": "Wind",
        "module": "WindPy",
        "credentials": [],
        "free": False,
        "priority": 10,
        "note": "需本地安装 Wind 终端并登录",
    },
}


def module_available(name: str) -> bool:
    mod = PROVIDERS.get(name, {}).get("module")
    if not mod:
        return True            # local 不依赖第三方库
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


@dataclass
class DataSourceConfig:
    """一个源的用户配置。credentials 不出现在任何对外响应里。"""
    name: str
    enabled: bool = True
    credentials: Dict[str, str] = field(default_factory=dict)

    def missing_credentials(self) -> List[str]:
        need = PROVIDERS.get(self.name, {}).get("credentials", [])
        return [k for k in need if not self.credentials.get(k)]

    def is_ready(self) -> bool:
        # 通过模块属性查找而非直接引用，使依赖可被测试替换
        # （直接引用会把函数绑死在定义时刻，monkeypatch 打不进来）
        import sys as _sys
        avail = getattr(_sys.modules[__name__], "module_available")
        return (self.enabled and avail(self.name)
                and not self.missing_credentials())


class DataSourceRegistry:
    """
    持有各源配置，决定用哪个。

    选路规则：就绪的源里按 priority 升序，用户配置过凭据的源优先于免费源
    （priority 本身已体现：付费源数值更小）。全不就绪时回落到 local。
    """

    def __init__(self):
        self._configs: Dict[str, DataSourceConfig] = {
            "local": DataSourceConfig("local"),
        }

    def set(self, name: str, enabled: bool = True,
            credentials: Optional[Dict[str, str]] = None) -> DataSourceConfig:
        if name not in PROVIDERS:
            raise ValueError(f"未知数据源: {name}")
        cur = self._configs.get(name) or DataSourceConfig(name)
        cur.enabled = enabled
        if credentials:
            # 只覆盖传入的字段，留空表示保持原值（与 LLM 配置同样的约定）
            cur.credentials.update({k: v for k, v in credentials.items() if v})
        self._configs[name] = cur
        return cur

    def remove(self, name: str) -> None:
        if name != "local":
            self._configs.pop(name, None)

    def ready_sources(self) -> List[str]:
        ready = [n for n, c in self._configs.items() if c.is_ready()]
        return sorted(ready, key=lambda n: PROVIDERS[n]["priority"])

    def active(self) -> str:
        """当前会用哪个源。"""
        r = self.ready_sources()
        return r[0] if r else "local"

    def status(self) -> List[Dict[str, Any]]:
        """供界面展示。绝不包含凭据值。"""
        out = []
        for name, meta in sorted(PROVIDERS.items(),
                                 key=lambda kv: kv[1]["priority"]):
            cfg = self._configs.get(name)
            avail = module_available(name)
            missing = cfg.missing_credentials() if cfg else \
                meta.get("credentials", [])
            out.append({
                "name": name,
                "label": meta["label"],
                "free": meta["free"],
                "note": meta["note"],
                "needs": meta.get("credentials", []),
                "module_available": avail,
                "configured": bool(cfg),
                "enabled": bool(cfg and cfg.enabled),
                "missing_credentials": missing,
                "ready": bool(cfg and cfg.is_ready()),
            })
        return out
