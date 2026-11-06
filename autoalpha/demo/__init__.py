"""
演示模式：无需 API Key 与真实数据即可走通全流程。

目的是让人五分钟内看到这个项目在做什么，而不是先去申请密钥、配数据。
所有内容取自真实运行的结果形状并固定下来，包括那个「IC 为正但最高分位
最差」的因子 —— 它正是本项目存在的理由，演示里必须出现。

数据明确标注为演示用，不会被误当成真实研究结论。
"""

from .fixtures import (
    DEMO_INTENT, DEMO_PAPERS, DEMO_SPECS, DEMO_DATA,
    DEMO_EVALS, DEMO_USAGE, DEMO_BANNER,
    DEMO_EXTENSIONS, DEMO_EXT_EVALS,
    DEMO_PROPOSAL, DEMO_ROUNDS,
)

__all__ = ["DEMO_INTENT", "DEMO_PAPERS", "DEMO_SPECS", "DEMO_DATA",
           "DEMO_EVALS", "DEMO_USAGE", "DEMO_BANNER",
           "DEMO_EXTENSIONS", "DEMO_EXT_EVALS",
           "DEMO_PROPOSAL", "DEMO_ROUNDS"]
