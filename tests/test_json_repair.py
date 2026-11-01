"""
json_repair 回归测试。

样本取自真实失败：服务日志实测
  [Translate] translate_factor 失败: Expecting ',' delimiter: line 4 column 72
"""

import json

import pytest

from autoalpha.json_repair import loads_lenient, JSONRepairError


def test_plain_json_still_works():
    assert loads_lenient('[{"a": 1}]') == [{"a": 1}]
    assert loads_lenient('{"a": 1}') == {"a": 1}


def test_markdown_fence():
    assert loads_lenient('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert loads_lenient('```\n{"a": 2}\n```') == {"a": 2}


def test_surrounding_prose():
    """LLM 常在 JSON 前后加解释。"""
    raw = '以下是提取结果：\n[{"factor_name": "mom"}]\n希望对你有帮助。'
    assert loads_lenient(raw) == [{"factor_name": "mom"}]


def test_unescaped_inner_quotes():
    """核心病症：中文里给英文术语加了引号。"""
    raw = '[{"what": "衡量"动量"强度", "why": "反应不足", "how": "rank"}]'
    out = loads_lenient(raw, expect=list)
    assert len(out) == 1
    assert "动量" in out[0]["what"]
    assert out[0]["why"] == "反应不足"


def test_full_width_quotes():
    raw = '[{"what": "衡量“动量”强度", "why": "x", "how": "y"}]'
    out = loads_lenient(raw, expect=list)
    assert "动量" in out[0]["what"]


def test_bare_newlines_in_strings():
    raw = '[{"summary": "第一行\n第二行", "relevance": "ok"}]'
    out = loads_lenient(raw, expect=list)
    assert "第一行" in out[0]["summary"]


def test_trailing_comma():
    assert loads_lenient('[{"a": 1},]', expect=list) == [{"a": 1}]
    assert loads_lenient('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_combined_damage():
    """实测中文输出常同时出现多种问题。"""
    raw = ('```json\n这是结果:\n[{"what": "衡量“动量”的"强度",\n'
           '  "why": "因为\n投资者反应不足", "how": "rank"},]\n```')
    out = loads_lenient(raw, expect=list)
    assert len(out) == 1 and "动量" in out[0]["what"]


def test_expect_list_unwraps_single_key_dict():
    """LLM 有时把数组包一层。"""
    raw = '{"factors": [{"a": 1}, {"b": 2}]}'
    assert loads_lenient(raw, expect=list) == [{"a": 1}, {"b": 2}]


def test_nested_structures_survive():
    raw = '[{"data_requirements": {"fields": ["close", "volume"]}, "n": 1}]'
    out = loads_lenient(raw, expect=list)
    assert out[0]["data_requirements"]["fields"] == ["close", "volume"]


def test_scores_array():
    """打分用的纯数字数组。"""
    assert loads_lenient("[8, 3, 9, 5]", expect=list) == [8, 3, 9, 5]
    assert loads_lenient("打分如下：[7,2]", expect=list) == [7, 2]


def test_hopeless_input_raises_with_context():
    with pytest.raises(JSONRepairError) as ei:
        loads_lenient("这完全不是 JSON，没有任何括号")
    assert "原文前" in str(ei.value)

    with pytest.raises(JSONRepairError):
        loads_lenient("")


def test_does_not_corrupt_escaped_quotes():
    """已经正确转义的引号不能被二次转义。"""
    raw = '[{"formula": "rank(\\"close\\")"}]'
    out = loads_lenient(raw, expect=list)
    assert out[0]["formula"] == 'rank("close")'


# ── 调用点行为契约 ──────────────────────────────────────────────────────────

class _StubLLM:
    """返回预设文本的假 LLM。"""
    def __init__(self, reply):
        self.reply = reply
        self.model = "stub"
        from autoalpha.llm import TokenMeter
        self.meter = TokenMeter()

    def complete(self, **kw):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_translate_failure_is_visible_not_silent():
    """
    中文消失的根因：解析失败时返回空串，界面上看起来就是「中文没了」。
    失败必须留下可见提示。
    """
    from autoalpha.agents.translate_agent import TranslateAgent
    ag = TranslateAgent(_StubLLM("这根本不是 JSON"))
    out = ag.describe_papers([{"title": "T", "abstract": "A"}])
    assert len(out) == 1
    assert out[0]["summary"], "失败时不得返回空串"
    assert "失败" in out[0]["summary"]


def test_translate_recovers_from_unescaped_quotes():
    """真实失败样本：中文里给术语加了英文引号。"""
    from autoalpha.agents.translate_agent import TranslateAgent
    reply = ('[{"what": "衡量"动量"强度", "why": "反应不足", "how": "rank"}]')
    ag = TranslateAgent(_StubLLM(reply))
    from autoalpha.pipeline_state import FactorSpec
    spec = FactorSpec("f", "d", "h", "rank(x)", "n", {}, {}, {})
    out = ag.describe_factors([spec])
    assert "动量" in out[0]["what"]
    assert "失败" not in out[0]["what"]


def test_judge_rejects_when_parse_fails():
    """
    judge 原本解析失败就默认 APPROVE —— 一次解析故障等于放过一个
    未经语义审查的因子。必须 fail-closed。
    """
    from autoalpha.agents.factor_judge_agent import FactorJudgeAgent
    from autoalpha.pipeline_state import FactorSpec, FactorResult
    ag = FactorJudgeAgent(_StubLLM("完全不是 JSON"))
    spec = FactorSpec("f", "d", "h", "rank(x)", "n", {}, {}, {})
    res = FactorResult.failed("f", "", "")
    v = ag.judge(spec, res)
    assert v["verdict"] == "REJECT", "解析失败必须拒绝，不得放行"
    assert v["overall_score"] == 0.0


def test_extract_surfaces_parse_error():
    from autoalpha.agents.factor_extract_agent import FactorExtractAgent
    ag = FactorExtractAgent(_StubLLM("不是 JSON"))
    specs = ag.extract({"title": "T", "abstract": "A"})
    assert specs == []
    assert ag.last_error, "解析失败必须记录原因"


def test_extract_recovers_damaged_json():
    from autoalpha.agents.factor_extract_agent import FactorExtractAgent
    reply = ('以下是结果：\n[{"factor_name": "mom", "description": "d",\n'
             '  "hypothesis": "用"动量"解释", "formula": "rank(r)",\n'
             '  "implementation_notes": "n", "data_requirements": {"fields": ["close"]},\n'
             '  "parameters": {"lookback": 20}},]')
    ag = FactorExtractAgent(_StubLLM(reply))
    specs = ag.extract({"title": "T", "abstract": "A"})
    assert len(specs) == 1 and specs[0].factor_name == "mom"
    assert not ag.last_error


def test_prompt_forbids_placeholder_summaries():
    """
    不得让模型输出「摘要信息不足」「需查看全文」这类占位话。
    用户要的是有用信息，而不是被告知信息不够 —— 摘要短时
    应就标题与已有内容写出能写的部分。
    """
    from autoalpha.agents.translate_agent import _PAPER_SYSTEM
    assert "不要写" in _PAPER_SYSTEM
    assert "摘要信息不足" in _PAPER_SYSTEM, "须显式点名禁止该话术"
    # 禁止句必须出现在否定语境里，而不是作为输出模板
    idx = _PAPER_SYSTEM.index("摘要信息不足")
    assert "不要写" in _PAPER_SYSTEM[max(0, idx - 60):idx], \
        "该短语只能出现在禁止说明中，不可作为建议输出"
