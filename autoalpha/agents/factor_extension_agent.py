import json
from typing import List
from ..llm import LLMClient
from ..pipeline_state import FactorSpec, FactorResult
from ..utils import strip_code_fence

_EXT_SYSTEM = """\
You are a quantitative researcher proposing factor extensions.

Given a base factor that passed quality evaluation, propose 3-5 extensions.
Each extension must have a clear economic rationale (not just parameter tweaking).

Extension types:
1. param_variation   — different lookback/decay/threshold (must have different economic meaning)
2. formula_variation — alternative formula measuring similar concept differently
3. combination       — combine with another well-known factor (momentum, reversal, volatility)

Output JSON array, each element:
{
  "factor_name": "extended_factor_name",
  "description": "what this extension measures",
  "hypothesis": "why this extension adds value over the base factor",
  "formula": "mathematical formula",
  "implementation_notes": "key implementation notes",
  "data_requirements": {"freq": "daily", "fields": [...], "lookback_days": N},
  "parameters": {...},
  "extension_type": "param_variation|formula_variation|combination",
  "rationale": "economic logic for why this extension should improve upon the base"
}

Output ONLY valid JSON array, no explanation.
"""


class FactorExtensionAgent:
    """基于已通过的因子，提出延伸变体"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose_extensions(self, spec: FactorSpec, result: FactorResult) -> List[FactorSpec]:
        print(f"[Extension] 为因子 '{spec.factor_name}' 生成延伸假设...")
        user_msg = (
            f"Base factor: {spec.factor_name}\n"
            f"Description: {spec.description}\n"
            f"Formula: {spec.formula}\n"
            f"IC mean: {result.ic_mean:.4f}, ICIR: {result.icir:.4f}, "
            f"IC decay: {result.ic_decay_ratio:.4f}\n"
            f"Source paper: {spec.source_paper.get('title','')}\n\n"
            f"Propose 3-5 extensions."
        )
        try:
            raw = strip_code_fence(self.llm.complete(system=_EXT_SYSTEM, user=user_msg))
            items = json.loads(raw)
            if not isinstance(items, list):
                items = [items]
        except Exception as e:
            print(f"[Extension] 生成失败: {e}")
            return []

        extensions = []
        for item in items:
            try:
                ext = FactorSpec(
                    factor_name=item.get("factor_name", ""),
                    description=item.get("description", ""),
                    hypothesis=item.get("hypothesis", ""),
                    formula=item.get("formula", ""),
                    implementation_notes=item.get("implementation_notes", ""),
                    data_requirements=item.get("data_requirements", spec.data_requirements),
                    parameters=item.get("parameters", {}),
                    source_paper=spec.source_paper,
                    variant_of=spec.factor_name,
                    extension_type=item.get("extension_type", "param_variation"),
                )
                extensions.append(ext)
            except Exception as e:
                print(f"[Extension] FactorSpec 构建失败: {e}")
        print(f"[Extension] 生成 {len(extensions)} 个延伸假设")
        return extensions
