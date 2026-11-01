"""
Local klines data loader.

Supports the directory structure:
  klines/
    etf-daily/   *.csv    daily bars
    min1/        *.csv    1-minute bars
    min5/        *.csv    5-minute bars
    min60/       *.csv    60-minute bars

CSV columns: trade_date, open, high, low, close, volume, amount, open_interest

Returns dict[symbol, pd.DataFrame] with DatetimeIndex and OHLCV columns,
matching the format expected by compute_factor() and FactorEvalAgent.
"""

import os
import re
import pandas as pd
from typing import Dict, Optional


def load_klines(
    klines_dir: str,
    freq: str = "etf-daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbols: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load local klines CSV files into the pipeline-compatible format.

    Args:
        klines_dir: Root directory containing freq sub-directories.
        freq: Sub-directory name: 'etf-daily', 'min1', 'min5', 'min60'.
        start_date: Filter rows >= this date (YYYY-MM-DD), inclusive.
        end_date:   Filter rows <= this date (YYYY-MM-DD), inclusive.
        symbols:    If given, only load these symbol codes (e.g. ['510300', '510500']).

    Returns:
        dict[symbol_code, DataFrame] — DatetimeIndex, columns: open/high/low/close/volume/amount
    """
    # 两种布局都接受：
    #   1. <dir>/<freq>/*.csv   —— 按频率分子目录（factor-signal 的结构）
    #   2. <dir>/*.csv          —— CSV 直接放在目录里
    # 目录名不该是硬约束，用户给什么目录就读什么；判据是里面有没有 CSV。
    freq_dir = os.path.join(klines_dir, freq)
    if not os.path.isdir(freq_dir):
        if _has_csv(klines_dir):
            freq_dir = klines_dir
        else:
            raise FileNotFoundError(
                f"目录下既没有 {freq}/ 子目录，也没有 CSV 文件: {klines_dir}")

    data: Dict[str, pd.DataFrame] = {}
    for fname in sorted(os.listdir(freq_dir)):
        if not fname.endswith(".csv"):
            continue

        # Extract symbol code from filename pattern: {exchange}etf_{code}_{name}.csv
        m = re.match(r"[a-z]+etf_(\d+)_", fname)
        code = m.group(1) if m else fname.replace(".csv", "")

        if symbols and code not in symbols:
            continue

        path = os.path.join(freq_dir, fname)
        try:
            df = pd.read_csv(path, parse_dates=["trade_date"])
            df = df.rename(columns={"trade_date": "date"})
            df = df.set_index("date").sort_index()

            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            if not df.empty:
                data[code] = df
        except Exception as e:
            print(f"[DataLoader] 跳过 {fname}: {e}")

    print(f"[DataLoader] 加载 {freq} 数据: {len(data)} 个标的, "
          f"日期范围 {_date_range(data)}")
    return data


def _has_csv(path: str) -> bool:
    try:
        return any(f.lower().endswith(".csv") for f in os.listdir(path))
    except OSError:
        return False


def find_csv_dirs(root: str, max_depth: int = 2) -> Dict[str, int]:
    """
    扫描目录找含 CSV 的位置，返回 {相对路径: csv 数量}。
    "." 表示 CSV 直接在 root 下。供界面提示用户目录里有什么。
    """
    out: Dict[str, int] = {}
    if not os.path.isdir(root):
        return out
    n = sum(1 for f in os.listdir(root) if f.lower().endswith(".csv"))
    if n:
        out["."] = n
    for d in sorted(os.listdir(root)):
        sub = os.path.join(root, d)
        if not os.path.isdir(sub):
            continue
        try:
            c = sum(1 for f in os.listdir(sub) if f.lower().endswith(".csv"))
        except OSError:
            continue
        if c:
            out[d] = c
    return out


def _date_range(data: Dict[str, pd.DataFrame]) -> str:
    if not data:
        return "无数据"
    all_idx = pd.concat([df.index.to_series() for df in data.values()])
    return f"{all_idx.min().date()} ~ {all_idx.max().date()}"
