"""
本地数据加载器：自动识别用户本地数据文件并转换为 DataAgent 兼容格式。

支持文件格式：CSV / Parquet / Excel / Pickle / Feather
支持数据布局：
  - 目录        每个文件对应一个标的（filename stem = symbol）
  - 单文件长格式  含 symbol 列 + date 列（多标的混合）
  - 单文件宽格式  行=日期，列=symbol（每列是该标的的价格）
  - 单文件单标的  只含 date + OHLCV 列
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SUPPORTED_EXTS = {".csv", ".parquet", ".pkl", ".pickle", ".xlsx", ".xls", ".feather"}

_DATE_HINTS = {
    "date", "datetime", "tradedate", "trade_date", "tradingdate",
    "time", "day", "dt", "index", "tdate", "交易日期", "日期",
}
_SYMBOL_HINTS = {
    "symbol", "code", "ticker", "tscode", "ts_code", "stockcode",
    "stock_code", "seccode", "sec_code", "instrument", "asset", "id",
    "证券代码", "股票代码",
}
_CLOSE_HINTS = {
    "close", "adjclose", "adj_close", "nav", "netvalue", "net_value",
    "unitnav", "unit_nav", "price", "收盘", "收盘价", "单位净值", "累计净值",
}


class LocalDataLoader:
    """
    自动识别本地数据文件，返回 {symbol: DataFrame(DatetimeIndex, close, ...)}。
    """

    def __init__(self, paths: List[str]):
        self.paths = [str(p) for p in paths if p]
        self._cache: Optional[Dict[str, pd.DataFrame]] = None

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """加载所有路径，返回 {normalized_symbol: DataFrame}。结果内存缓存。"""
        if self._cache is not None:
            return self._cache
        result: Dict[str, pd.DataFrame] = {}
        for path in self.paths:
            result.update(self._load_path(path))
        self._cache = result
        return result

    def get_data(self, symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        """
        按 symbol 列表从本地数据中取数并按日期过滤。
        只返回本地能命中的标的，未命中的由 DataAgent 走远程拉取。
        """
        all_data = self.load_all()
        if not all_data:
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self._match_symbol(sym, all_data)
            if df is not None:
                result[sym] = _filter_dates(df, start, end)
        return result

    def invalidate_cache(self):
        """清除内存缓存，强制下次重新加载文件。"""
        self._cache = None

    def summary(self) -> str:
        """返回已加载数据的摘要，用于终端展示。"""
        all_data = self.load_all()
        if not all_data:
            return "  （未加载到任何数据）"
        lines = []
        for sym, df in sorted(all_data.items()):
            if df.empty:
                lines.append(f"  {sym:<16}  空数据")
            else:
                rng = f"{df.index.min().date()} ~ {df.index.max().date()}"
                cols = ", ".join(df.columns.tolist()[:6])
                lines.append(f"  {sym:<16}  {len(df):>5} 行  {rng}  [{cols}]")
        return "\n".join(lines)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _load_path(self, path: str) -> Dict[str, pd.DataFrame]:
        p = Path(path)
        if not p.exists():
            print(f"[LocalData] 路径不存在: {path}")
            return {}
        if p.is_dir():
            return self._load_directory(p)
        if p.suffix.lower() in SUPPORTED_EXTS:
            return self._load_file(p)
        print(f"[LocalData] 不支持的格式（{p.suffix}）: {path}")
        return {}

    def _load_directory(self, dirpath: Path) -> Dict[str, pd.DataFrame]:
        files = sorted(
            f for f in dirpath.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
        if not files:
            print(f"[LocalData] 目录中无可识别数据文件: {dirpath}")
            return {}
        print(f"[LocalData] 扫描目录 {dirpath}，发现 {len(files)} 个文件")
        result: Dict[str, pd.DataFrame] = {}
        for f in files:
            result.update(self._load_file(f, default_symbol=f.stem))
        return result

    def _load_file(self, filepath: Path,
                   default_symbol: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        try:
            df_raw = _read_file(filepath)
        except Exception as e:
            print(f"[LocalData] 读取失败 {filepath.name}: {e}")
            return {}
        if df_raw is None or df_raw.empty:
            return {}

        layout, info = _detect_layout(df_raw)

        if layout == "long":
            sym_col = info["symbol_col"]
            date_col = info["date_col"]
            result: Dict[str, pd.DataFrame] = {}
            for sym, grp in df_raw.groupby(sym_col):
                key = _normalize_key(str(sym))
                df = _standardize(grp.drop(columns=[sym_col]), date_col)
                if df is not None and not df.empty:
                    result[key] = df
            n = len(result)
            print(f"[LocalData] {filepath.name}  长格式 → {n} 个标的")
            return result

        elif layout == "wide":
            date_col = info["date_col"]
            sym_cols = info["sym_cols"]
            df_raw = df_raw.copy()
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
            df_raw = (df_raw.dropna(subset=[date_col])
                      .set_index(date_col).sort_index())
            result = {}
            for col in sym_cols:
                key = _normalize_key(col)
                series = pd.to_numeric(df_raw[col], errors="coerce").dropna()
                if not series.empty:
                    result[key] = pd.DataFrame({"close": series})
            n = len(result)
            print(f"[LocalData] {filepath.name}  宽格式 → {n} 个标的")
            return result

        else:  # single
            date_col = info.get("date_col")
            key = _normalize_key(default_symbol or filepath.stem)
            df = _standardize(df_raw, date_col)
            if df is not None and not df.empty:
                cols_preview = ", ".join(df.columns.tolist()[:5])
                print(f"[LocalData] {filepath.name}  单标的 → {key}"
                      f"  {len(df)} 行  [{cols_preview}]")
                return {key: df}
            return {}

    def _match_symbol(self, sym: str,
                      all_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """多级匹配：精确 → normalize 后精确 → 去后缀模糊"""
        if sym in all_data:
            return all_data[sym]
        norm = _normalize_key(sym)
        if norm in all_data:
            return all_data[norm]
        # 去交易所后缀（.SZ/.SH）再比
        stripped = re.sub(r"\.(SZ|SH|BJ|SS|XSHG|XSHE)$", "",
                          norm, flags=re.IGNORECASE)
        for k, v in all_data.items():
            k_strip = re.sub(r"\.(SZ|SH|BJ|SS|XSHG|XSHE)$", "",
                             k, flags=re.IGNORECASE)
            if k_strip == stripped:
                return v
        return None


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _read_file(filepath: Path) -> Optional[pd.DataFrame]:
    ext = filepath.suffix.lower()
    if ext == ".csv":
        for enc in ("utf-8", "gbk", "gb2312"):
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(filepath, encoding="utf-8", errors="replace")
    elif ext == ".parquet":
        return pd.read_parquet(filepath)
    elif ext in (".pkl", ".pickle"):
        return pd.read_pickle(filepath)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    elif ext == ".feather":
        return pd.read_feather(filepath)
    return None


def _detect_layout(df: pd.DataFrame) -> Tuple[str, dict]:
    """
    推断 DataFrame 布局：
      "long"   含 symbol 列 + date 列（多标的行混合）
      "wide"   date 列 + 其余列均为 symbol（宽格式）
      "single" 单标的 OHLCV
    """
    # 规范化列名查找
    low2orig = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}

    # 找 date 列
    date_col: Optional[str] = None
    for low, orig in low2orig.items():
        if low in _DATE_HINTS or "date" in low or "time" in low:
            date_col = orig
            break
    if date_col is None:
        # fallback：试第一列能否解析为日期
        try:
            pd.to_datetime(df.iloc[:5, 0], errors="raise")
            date_col = df.columns[0]
        except Exception:
            pass

    # 找 symbol 列
    sym_col: Optional[str] = None
    for low, orig in low2orig.items():
        if low in _SYMBOL_HINTS:
            sym_col = orig
            break

    # 长格式判断
    if sym_col and date_col and df[sym_col].nunique() > 1:
        return "long", {"date_col": date_col, "symbol_col": sym_col}

    # 宽格式判断：date 列 + 所有其余列不像 OHLCV 字段名
    if date_col:
        non_date = [c for c in df.columns if c != date_col]
        ohlcv_names = {
            "open", "high", "low", "close", "volume", "vol", "amount",
            "adjclose", "pctchg", "pct_chg", "change",
        }
        ohlcv_cols = {
            c for c in non_date
            if c.lower().replace("_", "") in ohlcv_names
            or c.lower().replace("_", "") in _CLOSE_HINTS
        }
        sym_like = [c for c in non_date if c not in ohlcv_cols]
        if len(sym_like) >= 2 and not ohlcv_cols:
            return "wide", {"date_col": date_col, "sym_cols": non_date}

    return "single", {"date_col": date_col}


def _standardize(df: pd.DataFrame,
                 date_col: Optional[str]) -> Optional[pd.DataFrame]:
    """设置 DatetimeIndex，统一列名（close / open / high / low / volume）。"""
    df = df.copy()

    # 设置日期索引
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
        except Exception:
            return None

    df = df.sort_index()

    # 列名标准化
    low2orig = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    rename: Dict[str, str] = {}

    for std_name, hints in [
        ("close",  _CLOSE_HINTS | {"close", "收盘价", "收盘"}),
        ("open",   {"open", "开盘价", "开盘"}),
        ("high",   {"high", "最高价", "最高"}),
        ("low",    {"low", "最低价", "最低"}),
        ("volume", {"volume", "vol", "成交量", "amount", "成交额"}),
    ]:
        if std_name in df.columns:
            continue
        for hint in hints:
            key = hint.lower().replace("_", "").replace(" ", "")
            if key in low2orig and low2orig[key] not in rename.values():
                rename[low2orig[key]] = std_name
                break

    if rename:
        df = df.rename(columns=rename)

    # 必须有 close 列
    if "close" not in df.columns:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return None
        df = df.rename(columns={num_cols[0]: "close"})

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df if not df.empty else None


def _normalize_key(sym: str) -> str:
    """统一 symbol：提取6位A股数字代码，或保持原大写美股代码。"""
    s = sym.strip()
    m = re.search(r"\b(\d{6})\b", s)
    if m:
        return m.group(1)
    s = re.sub(r"\.(SZ|SH|BJ|SS|XSHG|XSHE)$", "", s, flags=re.IGNORECASE).strip()
    return s


def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    try:
        return df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    except Exception:
        return df
