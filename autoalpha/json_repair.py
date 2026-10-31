"""
容错解析 LLM 的 JSON 输出。

动机（实测）：
    [Translate] translate_factor 失败: Expecting ',' delimiter: line 4 column 72

LLM 生成中文时习惯给英文术语加引号（如 衡量"动量"强度），这些未转义的
双引号直接让 json.loads 抛错。原实现的降级是返回空结果，于是中文静默消失 ——
用户看到的是「中文没了」，而不是「翻译失败了」。

同类脆弱解析在 5 个 agent 里都有，后果从「中文消失」到「提取到 0 个因子」
再到「judge 默认放过坏因子」，所以统一到这里处理。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

__all__ = ["loads_lenient", "JSONRepairError"]


class JSONRepairError(ValueError):
    """所有修复策略都失败。异常消息里带原文片段便于定位。"""


def loads_lenient(raw: str, expect: Optional[type] = None) -> Any:
    """
    尽力把 LLM 输出解析成 JSON。按代价从低到高依次尝试各策略，
    任一策略成功即返回。

    expect: 传 list 或 dict 可要求顶层类型；类型不符视为该策略失败，
            继续尝试后续策略（LLM 有时会把数组包进 {"items": [...]}）。
    """
    if raw is None:
        raise JSONRepairError("输入为 None")
    text = raw.strip()
    if not text:
        raise JSONRepairError("输入为空字符串")

    errors = []
    for name, candidate in _candidates(text):
        if not candidate:
            continue
        try:
            val = json.loads(candidate)
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        val = _unwrap(val, expect)
        if expect is not None and not isinstance(val, expect):
            errors.append(f"{name}: 顶层类型为 {type(val).__name__}")
            continue
        return val

    raise JSONRepairError(
        "全部修复策略失败 → " + " | ".join(errors[:4])
        + f"\n原文前 300 字: {text[:300]}"
    )


# ── 候选生成 ─────────────────────────────────────────────────────────────────

def _candidates(text: str):
    """按代价递增产出 (策略名, 候选文本)。"""
    yield "raw", text

    fenced = _strip_fence(text)
    if fenced != text:
        yield "defence", fenced

    block = _extract_block(fenced)
    if block and block != fenced:
        yield "block", block

    base = block or fenced
    yield "ctrl", _escape_control_chars(base)
    yield "smart", _normalize_smart_quotes(base)
    yield "trailing", _strip_trailing_commas(base)
    yield "inner", _escape_inner_quotes(base)

    # 组合修复：实测中文输出常同时有全角引号 + 内部裸引号 + 裸换行
    combo = _strip_trailing_commas(
        _escape_control_chars(
            _escape_inner_quotes(
                _normalize_smart_quotes(base))))
    yield "combo", combo

    # _extract_block 的括号配平依赖引号状态，字符串内的裸引号会把它带偏
    # （实测返回 None）。所以在修完引号后，再尝试一次切块。
    if block is None:
        repaired = _strip_trailing_commas(
            _escape_control_chars(
                _escape_inner_quotes(
                    _normalize_smart_quotes(fenced))))
        late = _extract_block(repaired)
        if late:
            yield "late_block", late
        # 最后兜底：先切块再修引号（顺序相反，覆盖另一类损坏）
        rough = _rough_block(fenced)
        if rough:
            yield "rough", _strip_trailing_commas(
                _escape_control_chars(
                    _escape_inner_quotes(
                        _normalize_smart_quotes(rough))))


def _strip_fence(text: str) -> str:
    t = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text.strip())
    return re.sub(r"\n?```\s*$", "", t).strip()


def _extract_block(text: str) -> Optional[str]:
    """
    取第一个完整的 [...] 或 {...}。LLM 常在 JSON 前后加解释文字
    （"以下是结果："），用括号配平扫描而非贪婪正则，避免截断嵌套结构。
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        return None

    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _rough_block(text: str) -> Optional[str]:
    """
    不看引号状态的粗暴切块：第一个 [ 或 { 到最后一个 ] 或 }。
    用于 _extract_block 被裸引号带偏的情况，配合引号修复使用。
    """
    first = min((i for i, c in enumerate(text) if c in "[{"), default=None)
    if first is None:
        return None
    last = max((i for i, c in enumerate(text) if c in "]}"), default=None)
    if last is None or last <= first:
        return None
    return text[first:last + 1]


def _escape_control_chars(text: str) -> str:
    """把字符串字面量内部的裸换行/制表符转义（JSON 不允许裸控制字符）。"""
    out = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            continue
        out.append(ch)
    return "".join(out)


# 全角/弯引号 → 不参与 JSON 配平的替代字符。
#
# 曾经把 “ ” 转成 \\" ，结果引号配平状态被打乱，后续的控制字符与内部引号
# 修复全部基于错误状态工作（实测组合损坏样本解析失败）。中文书名号 「」
# 既保留了「这是引用」的语义，又不会被 JSON 解析器当作字符串边界。
_SMART_PAIRS = {
    "“": "「", "”": "」",   # “ ” → 「 」
    "‘": "'", "’": "'",         # ‘ ’ → '
}


def _normalize_smart_quotes(text: str) -> str:
    """
    全角/弯引号转换。仅在字符串内部做：结构性引号不能动。
    中文概述里 “动量” 这种写法很常见，转成转义双引号后 JSON 才合法。
    """
    out = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            continue
        if in_str and ch in _SMART_PAIRS:
            out.append(_SMART_PAIRS[ch])
            continue
        out.append(ch)
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _escape_inner_quotes(text: str) -> str:
    """
    转义字符串值内部未转义的双引号 —— 本模块要解决的核心病症。

    判定依据：一个 `"` 若其后（跳过空白）不是 JSON 结构字符 `,:}]`，
    也不在行尾，则它更可能是内容的一部分而非字符串结束符。
    这是启发式，所以放在候选链较后位置，前面的廉价策略优先。
    """
    out = []
    in_str = False
    esc = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                nxt = text[j] if j < n else ""
                if nxt in (",", ":", "}", "]", ""):
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')          # 内容里的引号
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _unwrap(val: Any, expect: Optional[type]) -> Any:
    """
    LLM 有时把数组包一层，如 {"items": [...]} / {"factors": [...]}。
    期望 list 却拿到单键 dict 且其值是 list 时，解包。
    """
    if expect is list and isinstance(val, dict):
        lists = [v for v in val.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return val
