import json
from typing import List
from ..llm import LLMClient
from ..pipeline_state import FactorSpec, FactorResult
from ..json_repair import loads_lenient, JSONRepairError
from ..utils import strip_code_fence

_EXT_SYSTEM = """\
You are a quantitative researcher proposing factor extensions.

Given a base factor that passed quality evaluation, propose 3-5 extensions.
Each extension MUST have a distinct economic rationale — not just parameter tweaking.

Extension types (use all three, not just one):
1. param_variation   — different lookback/decay/threshold with DIFFERENT economic meaning
                       (e.g., short-term vs long-term captures different investor behavior)
2. formula_variation — alternative formula measuring the same concept via different math
                       (e.g., exponential vs linear decay, ratio vs difference)
3. combination       — combine with an ORTHOGONAL factor (momentum ↔ reversal, price ↔ volume)
                       to capture interaction effects

Diversity rule: each extension must be structurally different from the others and from the base.
Avoid: renaming the same formula, trivially adjusting a single parameter without economic justification.

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
    """基于已通过的因子，提出延伸变体（含 Jaccard 多样性过滤）"""

    # 延伸间 Jaccard 相似度上限（来自 BrainAlpha / AlphaAgent 论文）
    DIVERSITY_THRESHOLD = 0.65

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
            f"Propose 3-5 extensions that are structurally diverse (not just parameter changes)."
        )
        try:
            raw = self.llm.complete(system=_EXT_SYSTEM, user=user_msg,
                                    purpose="extension")
            items = loads_lenient(raw, expect=list)
            if not isinstance(items, list):
                items = [items]
        except (JSONRepairError, Exception) as e:
            print(f"[Extension] 生成失败: {str(e)[:300]}")
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

        # 多样性过滤：去除与基础因子或其他延伸高度相似的变体
        diverse = self._filter_diverse(extensions, base_spec=spec)
        if len(diverse) < len(extensions):
            print(f"[Extension] 多样性过滤：{len(extensions)} → {len(diverse)} 个延伸")

        print(f"[Extension] 最终生成 {len(diverse)} 个延伸假设")
        return diverse

    # ── 多样性工具 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """计算两段文本的 Jaccard 词级相似度（算子/公式词汇重叠度）。"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _filter_diverse(self, extensions: List[FactorSpec],
                         base_spec: FactorSpec) -> List[FactorSpec]:
        """
        过滤掉与基础因子或已保留延伸高度相似的变体。
        相似度基于 formula + description 的联合 Jaccard 相似度。
        """
        base_text = base_spec.formula + " " + base_spec.description
        diverse: List[FactorSpec] = []

        for ext in extensions:
            ext_text = ext.formula + " " + ext.description
            # 与基础因子对比
            if self._jaccard_similarity(ext_text, base_text) > self.DIVERSITY_THRESHOLD:
                continue
            # 与已保留延伸对比
            too_similar = any(
                self._jaccard_similarity(ext_text, kept.formula + " " + kept.description)
                > self.DIVERSITY_THRESHOLD
                for kept in diverse
            )
            if not too_similar:
                diverse.append(ext)

        return diverse
