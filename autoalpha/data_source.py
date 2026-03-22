"""
数据源适配器体系

每个适配器只需实现 fetch()，DataAgent 按优先级瀑布式尝试，
只为仍未命中的标的请求后续数据源。

内置适配器：
  AKShareAdapter    — 免费，默认兜底（腾讯财经/东方财富）
  BaoStockAdapter   — 免费，A股日线/月线（pip install baostock）
  TushareProAdapter — Tushare Pro token（积分制）
  WindAdapter       — Wind 终端 windpy（需本地安装）
  JoinQuantAdapter  — 聚宽 jqdatasdk（需账号）
  RiceQuantAdapter  — 米筐 rqdatac（需账号）

自定义适配器（写一个 Python 文件，继承 DataSourceAdapter）：
  class MyAdapter(DataSourceAdapter):
      name = "my_source"
      def fetch(self, symbols, fields, freq, start, end):
          ...
          return {"513100": df, ...}
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


# ── 基类 ──────────────────────────────────────────────────────────────────────

class DataSourceAdapter(ABC):
    """
    数据源适配器基类。

    子类只需覆盖 name 和 fetch()，其余都有默认实现。
    fetch() 约定：
      - 输入 symbols 为字符串列表（格式由适配器自行处理）
      - 输出 {symbol: DataFrame}，只返回成功获取的标的
      - DataFrame 须有 DatetimeIndex，至少含 close 列（数值）
      - 不能获取的标的直接跳过（不抛异常），由下一个适配器接力
    """

    name: str = "base"
    priority: int = 99  # 数字越小优先级越高

    @abstractmethod
    def fetch(self, symbols: List[str], fields: List[str],
              freq: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
        ...

    def test_connection(self) -> bool:
        """快速测试数据源是否可用，失败时打印提示。"""
        return True

    def normalize_symbol(self, sym: str) -> str:
        """适配器内部的 symbol 标准化（可选覆盖）。"""
        return sym

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r} priority={self.priority}>"


# ── AKShare（默认兜底）────────────────────────────────────────────────────────

class AKShareAdapter(DataSourceAdapter):
    """
    AKShare 适配器（免费，腾讯财经/东方财富）。
    已内置于 DataAgent，无需额外配置。
    """
    name = "akshare"
    priority = 90  # 优先级最低，作为兜底

    # 与 DataAgent.INDEX_MAP 保持同步（英文全名 + 中文 + 纯数字）
    INDEX_MAP = {
        "000300": "sh000300", "000300.sh": "sh000300",
        "csi300": "sh000300", "csi 300": "sh000300",
        "hushen300": "sh000300", "沪深300": "sh000300", "hs300": "sh000300",
        "000905": "sh000905", "000905.sh": "sh000905",
        "csi500": "sh000905", "csi 500": "sh000905",
        "中证500": "sh000905", "zhongzheng500": "sh000905",
        "000001": "sh000001", "000001.sh": "sh000001",
        "上证指数": "sh000001", "上证综指": "sh000001",
        "shanghaicomposite": "sh000001", "ssecomposite": "sh000001",
        "ssecompositeindex": "sh000001", "shanghaicompositeindex": "sh000001",
        "399001": "sz399001", "399001.sz": "sz399001",
        "深证成指": "sz399001", "szsecomponent": "sz399001",
        "szsecomponentindex": "sz399001",
        "000016": "sh000016", "000016.sh": "sh000016",
        "sse50": "sh000016", "上证50": "sh000016",
        "shanghaistock50": "sh000016", "sse50index": "sh000016",
        "399006": "sz399006", "399006.sz": "sz399006",
        "创业板指": "sz399006", "创业板": "sz399006",
        "chinext": "sz399006", "chinextindex": "sz399006",
        "000852": "sh000852", "csi1000": "sh000852", "中证1000": "sh000852",
        "000688": "sh000688", "star50": "sh000688", "科创50": "sh000688",
        "000903": "sh000903", "csi100": "sh000903", "中证100": "sh000903",
    }

    _FUZZY_RULES = [
        ("ssecomposite",      "sh000001"),
        ("shanghaicomposite", "sh000001"),
        ("szsecomponent",     "sz399001"),
        ("sse50",             "sh000016"),
        ("chinextindex",      "sz399006"),
        ("chinext",           "sz399006"),
        ("csi1000",           "sh000852"),
        ("star50",            "sh000688"),
        ("csi300",            "sh000300"),
        ("csi500",            "sh000905"),
    ]

    def fetch(self, symbols, fields, freq, start, end):
        try:
            import akshare as ak
        except ImportError:
            print("[AKShare] akshare 未安装，pip install akshare")
            return {}

        data = {}
        start_d, end_d = start.replace("-", ""), end.replace("-", "")
        for raw_sym in symbols:
            inst = self._norm(raw_sym)
            df = self._fetch_one(ak, inst, freq, start, end, start_d, end_d)
            if df is not None and not df.empty:
                data[raw_sym] = df
            else:
                print(f"[AKShare] 无数据: {raw_sym} ({inst})")
        return data

    def test_connection(self):
        try:
            import akshare as ak  # noqa: F401
            return True
        except ImportError:
            return False

    def _norm(self, raw: str) -> str:
        m = re.search(r'\b(\d{6})\b', raw)
        if m:
            code = m.group(1)
            return self.INDEX_MAP.get(code, code)
        cleaned = re.sub(r"\s*\(.*?\)", "", raw).strip()
        key = cleaned.lower().replace("-", "").replace("_", "").replace(" ", "")
        if key in self.INDEX_MAP:
            return self.INDEX_MAP[key]
        for keyword, target in self._FUZZY_RULES:
            if keyword in key:
                return target
        return cleaned

    def _fetch_one(self, ak, inst, freq, start, end, start_d, end_d):
        try:
            if inst.startswith(("sh0", "sz3", "sz0")):
                df = ak.stock_zh_index_daily_tx(symbol=inst)
                df = df[(df["date"].astype(str) >= start) &
                        (df["date"].astype(str) <= end)].copy()
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date").sort_index()
            elif re.match(r"^\d{6}$", inst) and inst[:2] in ("51","58","50","15","16"):
                import io, contextlib
                with contextlib.redirect_stderr(io.StringIO()):
                    df = ak.fund_etf_fund_info_em(fund=inst, start_date=start_d, end_date=end_d)
                if df.empty:
                    return None
                df = df.rename(columns={"净值日期":"date","单位净值":"close",
                                        "累计净值":"nav_cum","日增长率":"pct_chg"})
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
                for c in ("open", "high", "low"):
                    df[c] = df["close"]
                df["volume"] = 0.0
                return df
            elif re.match(r"^[A-Z]{1,5}$", inst.strip()):
                df = ak.stock_us_daily(symbol=inst, adjust="qfq")
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            else:
                code = inst.zfill(6)
                tx_sym = f"sh{code}" if code[:1] in ("6","5") else f"sz{code}"
                df = ak.stock_zh_a_daily(symbol=tx_sym, start_date=start_d,
                                         end_date=end_d, adjust="qfq")
                df["date"] = pd.to_datetime(df["date"])
                return df.set_index("date").sort_index()
        except Exception as e:
            print(f"[AKShare] {inst}: {e}")
            return None


# ── BaoStock（免费，A股日线）─────────────────────────────────────────────────

class BaoStockAdapter(DataSourceAdapter):
    """
    BaoStock 免费适配器（A股日/周/月线，复权价）。
    pip install baostock
    """
    name = "baostock"
    priority = 80

    def __init__(self):
        self._logged_in = False

    def _login(self):
        if not self._logged_in:
            import baostock as bs
            bs.login()
            self._logged_in = True

    def fetch(self, symbols, fields, freq, start, end):
        try:
            import baostock as bs
        except ImportError:
            print("[BaoStock] 未安装 baostock，pip install baostock")
            return {}
        self._login()

        freq_map = {"daily": "d", "weekly": "w", "monthly": "m"}
        bs_freq = freq_map.get(freq, "d")
        data = {}
        for sym in symbols:
            code = self._to_bs_code(sym)
            if not code:
                continue
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,open,high,low,close,volume,pctChg",
                    start_date=start, end_date=end,
                    frequency=bs_freq, adjustflag="2"  # 前复权
                )
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue
                df = pd.DataFrame(rows, columns=rs.fields)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                if not df.empty:
                    data[sym] = df
                    print(f"[BaoStock] {sym}: {len(df)} 行")
            except Exception as e:
                print(f"[BaoStock] {sym}: {e}")
        return data

    def test_connection(self):
        try:
            import baostock as bs  # noqa: F401
            return True
        except ImportError:
            return False

    def _to_bs_code(self, sym: str) -> Optional[str]:
        m = re.search(r'\b(\d{6})\b', sym)
        if not m:
            return None
        code = m.group(1)
        prefix = "sh" if code[:1] in ("6", "5", "0") else "sz"
        return f"{prefix}.{code}"


# ── Tushare Pro ───────────────────────────────────────────────────────────────

class TushareProAdapter(DataSourceAdapter):
    """
    Tushare Pro 适配器（积分制，A股/ETF/指数全覆盖，数据质量高）。
    pip install tushare
    使用：TushareProAdapter(token="your_token")
    """
    name = "tushare_pro"
    priority = 30

    def __init__(self, token: str):
        self.token = token
        self._pro = None

    def _get_pro(self):
        if self._pro is None:
            import tushare as ts
            ts.set_token(self.token)
            self._pro = ts.pro_api()
        return self._pro

    def fetch(self, symbols, fields, freq, start, end):
        try:
            pro = self._get_pro()
        except Exception as e:
            print(f"[TusharePro] 初始化失败: {e}")
            return {}

        freq_map = {"daily": "D", "weekly": "W", "monthly": "M"}
        ts_freq = freq_map.get(freq, "D")
        start_ts = start.replace("-", "")
        end_ts = end.replace("-", "")
        data = {}

        for sym in symbols:
            ts_code = self._to_ts_code(sym)
            if not ts_code:
                continue
            try:
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start_ts, end_date=end_ts,
                    fields="trade_date,open,high,low,close,vol"
                ) if ts_freq == "D" else pro.weekly(
                    ts_code=ts_code,
                    start_date=start_ts, end_date=end_ts,
                    fields="trade_date,open,high,low,close,vol"
                )
                if df is None or df.empty:
                    continue
                df = df.rename(columns={"trade_date": "date", "vol": "volume"})
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                if not df.empty:
                    data[sym] = df
                    print(f"[TusharePro] {sym}: {len(df)} 行")
            except Exception as e:
                print(f"[TusharePro] {sym}: {e}")
        return data

    def test_connection(self):
        try:
            pro = self._get_pro()
            pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20240102")
            return True
        except Exception:
            return False

    def _to_ts_code(self, sym: str) -> Optional[str]:
        m = re.search(r'\b(\d{6})\b', sym)
        if not m:
            return None
        code = m.group(1)
        suffix = "SH" if code[:1] in ("6", "5") else "SZ"
        return f"{code}.{suffix}"


# ── Wind（windpy）────────────────────────────────────────────────────────────

class WindAdapter(DataSourceAdapter):
    """
    Wind 适配器（需本地安装 Wind 终端 + windpy）。
    无需额外参数，通过 Wind 终端登录态授权。
    使用：WindAdapter()
    """
    name = "wind"
    priority = 10  # Wind 数据质量高，优先级设高

    def __init__(self):
        self._w = None

    def _get_w(self):
        if self._w is None:
            from WindPy import w
            w.start()
            self._w = w
        return self._w

    def fetch(self, symbols, fields, freq, start, end):
        try:
            w = self._get_w()
        except Exception as e:
            print(f"[Wind] 初始化失败（是否已启动 Wind 终端？）: {e}")
            return {}

        freq_map = {"daily": "D", "weekly": "W", "monthly": "M"}
        w_freq = freq_map.get(freq, "D")
        # Wind 字段映射
        w_fields = "open,high,low,close,volume"
        data = {}

        for sym in symbols:
            w_code = self._to_wind_code(sym)
            try:
                result = w.wsd(w_code, w_fields, start, end,
                               f"Period={w_freq};PriceAdj=F")  # F=前复权
                if result.ErrorCode != 0:
                    print(f"[Wind] {sym}: ErrorCode={result.ErrorCode}")
                    continue
                df = pd.DataFrame(
                    dict(zip(result.Fields, result.Data)),
                    index=pd.to_datetime(result.Times)
                )
                df.columns = [c.lower() for c in df.columns]
                df.index.name = "date"
                df = df.sort_index().dropna(subset=["close"])
                if not df.empty:
                    data[sym] = df
                    print(f"[Wind] {sym}: {len(df)} 行")
            except Exception as e:
                print(f"[Wind] {sym}: {e}")
        return data

    def test_connection(self):
        try:
            from WindPy import w
            w.start()
            return w.isconnected()
        except Exception:
            return False

    def _to_wind_code(self, sym: str) -> str:
        m = re.search(r'\b(\d{6})\b', sym)
        if m:
            code = m.group(1)
            suffix = "SH" if code[:1] in ("6", "5") else "SZ"
            return f"{code}.{suffix}"
        return sym


# ── JoinQuant / 聚宽 ──────────────────────────────────────────────────────────

class JoinQuantAdapter(DataSourceAdapter):
    """
    聚宽 jqdatasdk 适配器。
    pip install jqdatasdk
    使用：JoinQuantAdapter(username="手机号", password="密码")
    """
    name = "joinquant"
    priority = 20

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._initialized = False

    def _init(self):
        if not self._initialized:
            import jqdatasdk as jq
            jq.auth(self.username, self.password)
            self._initialized = True
            self._jq = jq

    def fetch(self, symbols, fields, freq, start, end):
        try:
            self._init()
            jq = self._jq
        except Exception as e:
            print(f"[JoinQuant] 初始化失败: {e}")
            return {}

        freq_map = {"daily": "1d", "weekly": "1w", "monthly": "1M"}
        jq_freq = freq_map.get(freq, "1d")
        data = {}

        for sym in symbols:
            jq_code = self._to_jq_code(sym)
            try:
                df = jq.get_price(
                    jq_code,
                    start_date=start, end_date=end,
                    frequency=jq_freq,
                    fields=["open", "high", "low", "close", "volume"],
                    fq="pre"  # 前复权
                )
                if df is None or df.empty:
                    continue
                df.index.name = "date"
                df = df.sort_index().dropna(subset=["close"])
                if not df.empty:
                    data[sym] = df
                    print(f"[JoinQuant] {sym}: {len(df)} 行")
            except Exception as e:
                print(f"[JoinQuant] {sym}: {e}")
        return data

    def test_connection(self):
        try:
            self._init()
            return True
        except Exception:
            return False

    def _to_jq_code(self, sym: str) -> str:
        m = re.search(r'\b(\d{6})\b', sym)
        if m:
            code = m.group(1)
            suffix = "XSHG" if code[:1] in ("6", "5") else "XSHE"
            return f"{code}.{suffix}"
        return sym


# ── RiceQuant / 米筐 ──────────────────────────────────────────────────────────

class RiceQuantAdapter(DataSourceAdapter):
    """
    米筐 rqdatac 适配器。
    pip install rqdatac
    使用：RiceQuantAdapter(username="账号", password="密码")
         或 RiceQuantAdapter()  # 使用环境变量 RQ_USERNAME / RQ_PASSWORD
    """
    name = "ricequant"
    priority = 20

    def __init__(self, username: str = None, password: str = None):
        self.username = username or os.environ.get("RQ_USERNAME", "")
        self.password = password or os.environ.get("RQ_PASSWORD", "")
        self._initialized = False

    def _init(self):
        if not self._initialized:
            import rqdatac as rq
            rq.init(self.username, self.password)
            self._initialized = True
            self._rq = rq

    def fetch(self, symbols, fields, freq, start, end):
        try:
            self._init()
            rq = self._rq
        except Exception as e:
            print(f"[RiceQuant] 初始化失败: {e}")
            return {}

        freq_map = {"daily": "1d", "weekly": "1w", "monthly": "1M"}
        rq_freq = freq_map.get(freq, "1d")
        data = {}

        for sym in symbols:
            rq_code = self._to_rq_code(sym)
            try:
                df = rq.get_price(
                    rq_code,
                    start_date=start, end_date=end,
                    frequency=rq_freq,
                    fields=["open", "high", "low", "close", "volume"],
                    adjust_type="pre"  # 前复权
                )
                if df is None or df.empty:
                    continue
                df.index.name = "date"
                df = df.sort_index().dropna(subset=["close"])
                if not df.empty:
                    data[sym] = df
                    print(f"[RiceQuant] {sym}: {len(df)} 行")
            except Exception as e:
                print(f"[RiceQuant] {sym}: {e}")
        return data

    def test_connection(self):
        try:
            self._init()
            return True
        except Exception:
            return False

    def _to_rq_code(self, sym: str) -> str:
        m = re.search(r'\b(\d{6})\b', sym)
        if m:
            code = m.group(1)
            suffix = "XSHG" if code[:1] in ("6", "5") else "XSHE"
            return f"{code}.{suffix}"
        return sym


# ── 自定义适配器动态加载 ───────────────────────────────────────────────────────

def load_adapter_from_file(filepath: str, class_name: str,
                           init_kwargs: dict = None) -> DataSourceAdapter:
    """
    从 Python 文件动态加载自定义适配器。

    Parameters
    ----------
    filepath   : str   Python 文件路径（含 class_name 类定义）
    class_name : str   适配器类名
    init_kwargs: dict  传给 __init__ 的关键字参数（可选）

    示例
    ----
    adapter = load_adapter_from_file(
        "./my_wind_adapter.py", "MyWindAdapter",
        init_kwargs={"token": "xxx"}
    )
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_user_adapter", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    if not issubclass(cls, DataSourceAdapter):
        raise TypeError(f"{class_name} 必须继承 DataSourceAdapter")
    return cls(**(init_kwargs or {}))


# ── 适配器注册表 ──────────────────────────────────────────────────────────────

_BUILTIN_ADAPTERS = {
    "akshare":    AKShareAdapter,
    "baostock":   BaoStockAdapter,
    "tushare":    TushareProAdapter,
    "tushare_pro": TushareProAdapter,
    "wind":       WindAdapter,
    "joinquant":  JoinQuantAdapter,
    "jq":         JoinQuantAdapter,
    "ricequant":  RiceQuantAdapter,
    "rq":         RiceQuantAdapter,
}


def create_adapter(name: str, **kwargs) -> DataSourceAdapter:
    """
    按名称创建内置适配器。

    示例：
      create_adapter("tushare", token="xxx")
      create_adapter("joinquant", username="13800138000", password="xxx")
      create_adapter("akshare")
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    cls = _BUILTIN_ADAPTERS.get(key)
    if cls is None:
        raise ValueError(
            f"未知适配器: {name!r}。"
            f"内置支持: {list(_BUILTIN_ADAPTERS.keys())}"
        )
    return cls(**kwargs)
