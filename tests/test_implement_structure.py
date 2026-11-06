"""
代码生成的结构契约。

实测一次真实运行：LLM 把 data(dict[股票代码, DataFrame]) 当成
dict[字段名, DataFrame]，在 data 里找 "close" 键 → KeyError。
它被归成 DATA，拿到 dropna()/min_periods 的提示，三次重试全在原地打转。
这里守三件事：
  1. 这类错误必须归类为 STRUCTURE，修复提示要让它重写数据访问层；
  2. prompt 必须给出真实数据形状，而不是只让 LLM 读文字描述去猜；
  3. 骨架本身必须能在真实数据上跑通（不是看起来对）。
"""

import os
import pickle
import subprocess
import sys
import tempfile

import pandas as pd
import pytest

from autoalpha.agents.factor_implement_agent import (
    _CODEGEN_SYSTEM, _REPAIR_HINTS, _REPAIR_SYSTEM, classify_error,
)

REAL_TRACEBACK = (
    'ERROR: Traceback (most recent call last):\n'
    '  File "x.py", line 103, in <module>\n'
    '    result = compute_factor(data)\n'
    '  File "x.py", line 74, in compute_factor\n'
    '    raise KeyError("close price data not found")\n'
    "KeyError: 'close price data not found'"
)


class TestClassify:
    def test_structure_misread_is_not_data(self):
        assert classify_error(REAL_TRACEBACK) == "STRUCTURE"

    def test_plain_column_keyerror_stays_data(self):
        # 真的只是缺列时不该被当成结构误解
        assert classify_error("KeyError: 'some_custom_column'") == "DATA"

    def test_other_categories_unaffected(self):
        assert classify_error("SyntaxError: invalid syntax") == "SYNTAX"
        assert classify_error("NameError: name 'x' is not defined") == "RUNTIME"
        assert classify_error("TIMEOUT: exceeded 300s") == "TIMEOUT"


class TestRepairGuidance:
    def test_structure_hint_demands_rewrite(self):
        hint = _REPAIR_HINTS["STRUCTURE"]
        assert "REWRITE" in hint, "结构性误解必须要求重写，最小修改会再次失败"
        assert "STOCK CODE" in hint
        assert "data.items()" in hint

    def test_repair_system_allows_structure_rewrite(self):
        assert "STRUCTURE" in _REPAIR_SYSTEM
        assert "MUST be rewritten" in _REPAIR_SYSTEM


class TestPromptDescribesRealShape:
    def test_skeleton_warns_against_field_keying(self):
        assert "NOT keyed by field name" in _CODEGEN_SYSTEM
        assert "data[\"510300\"][\"close\"]" in _CODEGEN_SYSTEM

    def test_skeleton_has_verified_main_block(self):
        assert 'result.to_csv(sys.argv[2])' in _CODEGEN_SYSTEM
        assert 'print("FACTOR_OK")' in _CODEGEN_SYSTEM

    def test_skeleton_keeps_output_contract(self):
        assert 's.index.names = ["date", "symbol"]' in _CODEGEN_SYSTEM
        assert 's.name = "factor_value"' in _CODEGEN_SYSTEM


class TestSkeletonActuallyRuns:
    """
    prompt 里的骨架必须真能用。上一版只给函数签名，LLM 自己发明了错的
    数据访问层；这版给完整骨架，那就要保证它本身是对的。
    """

    def _run(self, code):
        # 要长于骨架示例的 20 日窗口，否则全是 NaN —— 那是测试数据太短，
        # 不是骨架的错
        df = pd.DataFrame(
            {"close": [10 + i * 0.5 for i in range(40)]},
            index=pd.date_range("2024-01-01", periods=40))
        data = {"AAA": df, "BBB": df * 2}
        with tempfile.TemporaryDirectory() as td:
            pkl = os.path.join(td, "d.pkl")
            csv = os.path.join(td, "out.csv")
            src = os.path.join(td, "f.py")
            with open(pkl, "wb") as f:
                pickle.dump(data, f)
            with open(src, "w") as f:
                f.write(code)
            r = subprocess.run([sys.executable, src, pkl, csv],
                               capture_output=True, text=True, timeout=120)
            # 必须在 with 块内读出来：TemporaryDirectory 退出即删目录，
            # 把路径带出去再读会得到 FileNotFoundError（曾误判成骨架的 bug）
            out = pd.read_csv(csv) if os.path.exists(csv) else None
            return r, out

    def test_skeleton_executes_and_writes_contract_csv(self):
        import re
        m = re.search(r"```python\n(import pandas as pd.*?)```",
                      _CODEGEN_SYSTEM, re.S)
        assert m, "prompt 里必须能找到完整骨架"
        code = m.group(1)
        r, out = self._run(code)
        assert r.returncode == 0, f"骨架跑挂了: {r.stderr[:400]}"
        assert "FACTOR_OK" in r.stdout
        assert out is not None, "骨架未写出 CSV"
        assert list(out.columns) == ["date", "symbol", "factor_value"]
        assert set(out["symbol"]) == {"AAA", "BBB"}
        # 截面 rank 后非空值应落在 [0,1]
        vals = out["factor_value"].dropna()
        assert len(vals) and vals.between(0, 1).all()
