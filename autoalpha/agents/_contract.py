"""
因子代码的出口契约兜底。

实测一类失败：LLM 不按骨架写 main 块，自己发明 _load_data() /
_make_demo_data() 自造演示数据（里面长度 400 vs 399），既没读
sys.argv[1] 的 pickle，也没 to_csv / FACTOR_OK。这类错误被归成 DATA
（dropna 提示），三次重试都没纠正 —— 因为它不是「数据问题」，
是「没遵守出口契约」。

契约是确定的，不该靠 LLM 重试碰运气：直接补齐/替换为正确的 main 块。
只动 main 块与自造数据的辅助函数，因子逻辑一字不改。
"""

import re

# 自造数据/自造加载器的辅助函数名 —— 出现即视为偏离契约
_FAKE_HELPERS = ("_make_demo_data", "_load_data", "_generate_sample",
                 "_demo_data", "_make_synthetic")


def _find_entry_fn(code: str) -> str:
    """主计算函数名：优先 compute_factor，否则第一个公开的顶层 def。"""
    if re.search(r"^def compute_factor\s*\(", code, re.M):
        return "compute_factor"
    for m in re.finditer(r"^def (\w+)\s*\(", code, re.M):
        name = m.group(1)
        if not name.startswith("_"):
            return name
    return "compute_factor"


def _strip_main_block(code: str) -> str:
    """去掉已有 main 块（不论写得多歪），返回纯逻辑部分。"""
    idx = code.find('if __name__ == "__main__":')
    if idx == -1:
        idx = code.find("if __name__ == '__main__':")
    if idx != -1:
        return code[:idx].rstrip() + "\n"
    return code.rstrip() + "\n"


def _main_template(fn: str) -> str:
    return (
        "\nimport pickle, sys\n"
        'if __name__ == "__main__":\n'
        '    with open(sys.argv[1], "rb") as f:\n'
        "        data = pickle.load(f)\n"
        f"    result = {fn}(data)\n"
        "    result.to_csv(sys.argv[2])\n"
        '    print("FACTOR_OK")\n'
    )


def enforce_main_contract(code: str) -> str:
    """
    保证出口契约：读 argv[1] pickle → 调主函数 → to_csv(argv[2]) → FACTOR_OK。

    什么情况下动手：
      - 没有 main 块；
      - main 块里没有 to_csv / FACTOR_OK / pickle；
      - 出现自造数据辅助函数（_make_demo_data 等）。
    其它情况原样返回，避免误改已经正确的代码。
    """
    needs = ('if __name__' not in code
             or 'to_csv' not in code
             or 'FACTOR_OK' not in code
             or 'pickle' not in code
             or any(h in code for h in _FAKE_HELPERS))
    if not needs:
        return code

    body = _strip_main_block(code)
    fn = _find_entry_fn(body)
    return body + _main_template(fn)
