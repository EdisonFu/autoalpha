from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class FactorSpec:
    """从论文中提取的因子假设规范"""
    factor_name: str
    description: str              # 自然语言描述因子含义
    hypothesis: str               # 该因子预测什么、为什么有效
    formula: str                  # 数学公式（LaTeX 或伪代码）
    implementation_notes: str     # 实现要注意的关键细节
    data_requirements: Dict       # {"freq": "daily", "fields": ["close","volume",...], "lookback_days": 252}
    parameters: Dict              # {"lookback": 20, "decay": 0.5, ...}
    source_paper: Dict            # {"title": ..., "url": ..., "date": ..., "abstract": ...}
    variant_of: Optional[str] = None   # 若是延伸因子，填父因子名
    extension_type: Optional[str] = None  # param_variation / formula_variation / combination
    # LLM 自报：该因子能否用现有数据算出来（修 E2E_REVIEW P-2）
    computable: bool = True
    required_fields: List[str] = field(default_factory=list)
    # 这个因子是读了全文还是只读摘要提取的："fulltext" | "abstract"
    provenance: str = "abstract"
    # 每个参数的来源："paper"（论文明确写了）/ "inferred"（论文暗示）/
    # "default"（模型自己定的）。只读摘要时一律 default —— 摘要几乎不写
    # 回看窗口、分位数这类数字，标成「来自论文」是误导。
    param_sources: Dict = field(default_factory=dict)
    # 市场迁移声明：{"source_market": "us", "changes": ["VIX → 已实现波动率"]}
    # 非空表示 LLM 已把原文的方法论迁到了本数据集上 —— 此时因子描述里
    # 出现原文的美股词是引用原文，不是数据依赖。
    migration: Dict = field(default_factory=dict)


@dataclass
class FactorResult:
    """因子评估结果"""
    factor_name: str
    ic_mean: float
    icir: float
    ic_decay_ratio: float
    coverage: float
    monthly_win_rate: float
    max_correlation: float
    status: str           # "PASS" | "FAIL"
    fail_reason: str = ""
    code_path: str = ""
    ic_series: List[float] = field(default_factory=list)
    judge_score: float = 0.0    # LLM judge overall score (0-10); 0 = not judged
    judge_comment: str = ""     # judge improvement suggestions or rejection reason

    @classmethod
    def failed(cls, name: str, code_path: str, reason: str) -> "FactorResult":
        return cls(
            factor_name=name, ic_mean=0.0, icir=0.0, ic_decay_ratio=0.0,
            coverage=0.0, monthly_win_rate=0.0, max_correlation=0.0,
            status="FAIL", fail_reason=reason, code_path=code_path,
        )


@dataclass
class FactorReport:
    """最终因子研究报告"""
    idea: str
    papers_used: List[Dict]
    total_extracted: int
    total_tested: int
    passed: List[FactorResult]
    failed: List[FactorResult]
    extensions_passed: List[FactorResult]
    report_path: str = ""
