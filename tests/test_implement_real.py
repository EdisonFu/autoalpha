"""
真实实现路径：mock LLM 但真跑沙箱。

此前一次真实运行报「FactorImplementAgent object has no attribute
'_check_lookahead_bias'」——插入 _check_factor_csv 时把类方法挤成了
模块级函数里的死代码。260 条测试全过，因为它们都 mock 掉了 implement。
这类 bug 只有真跑 implement() 才暴露。
"""

import os
import tempfile

import pandas as pd
import pytest

from autoalpha.agents.factor_implement_agent import FactorImplementAgent
from autoalpha.pipeline_state import FactorSpec

# 一段满足契约的因子代码（读 argv pickle、to_csv、FACTOR_OK）
GOOD_CODE = '''import pandas as pd
import numpy as np
import pickle, sys


def compute_factor(data: dict) -> pd.Series:
    frames = []
    for symbol, df in data.items():
        v = df["close"].pct_change(5, fill_method=None)
        frames.append(v.rename(symbol))
    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide.rank(axis=1, pct=True)
    s = wide.stack()
    s.index.names = ["date", "symbol"]
    s.name = "factor_value"
    return s


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f)
    result = compute_factor(data)
    result.to_csv(sys.argv[2])
    print("FACTOR_OK")
'''


class _FakeLLM:
    def __init__(self, code):
        self.code = code

    def complete(self, **kw):
        return self.code


def _spec():
    return FactorSpec(factor_name="mom", description="d", hypothesis="h",
                      formula="f", implementation_notes="",
                      data_requirements={"freq": "daily"},
                      parameters={}, source_paper={})


def _data():
    idx = pd.date_range("2024-01-01", periods=80)
    return {c: pd.DataFrame({"close": range(100, 100 + 80)}, index=idx)
            for c in ("AAA", "BBB", "CCC")}


def test_implement_runs_sandbox_and_writes_csv():
    """真跑 implement()，不允许出现 AttributeError。"""
    with tempfile.TemporaryDirectory() as td:
        ag = FactorImplementAgent(_FakeLLM(GOOD_CODE), output_dir=td)
        csv_path, err = ag.implement(_spec(), _data())
        assert err is None, f"实现失败: {err}"
        assert csv_path and os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["date", "symbol", "factor_value"]


def test_lookahead_check_is_a_class_method():
    """_check_lookahead_bias 必须在类上，不是模块函数里的死代码。"""
    ag = FactorImplementAgent.__new__(FactorImplementAgent)
    assert hasattr(ag, "_check_lookahead_bias")
    assert FactorImplementAgent._check_lookahead_bias("x = df.shift(-1)")


def test_shape_check_rejects_single_day_output():
    """返回单日截面时必须被形状校验拦下。"""
    code = GOOD_CODE.replace(
        "s = wide.stack()",
        "s = wide.iloc[[-1]].stack()")     # 只取最后一天 → 行数 < 50
    with tempfile.TemporaryDirectory() as td:
        ag = FactorImplementAgent(_FakeLLM(code), output_dir=td)
        csv_path, err = ag.implement(_spec(), _data())
        assert csv_path is None
        assert "形状" in err or "历史过短" in err
