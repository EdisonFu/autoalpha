"""多源资料检索。"""

from .base import PaperSource, dedupe
from .arxiv import ArxivSource
from .openalex import OpenAlexSource
from .crossref import CrossrefSource
from .joinquant import JoinQuantSource

# 默认启用的源。界面上可逐个开关。
DEFAULT_SOURCES = ("arxiv", "openalex", "crossref", "joinquant")

_REGISTRY = {
    "arxiv": ArxivSource,
    "openalex": OpenAlexSource,
    "crossref": CrossrefSource,
    "joinquant": JoinQuantSource,
}


def build_sources(names=None, db_path: str = "autoalpha.db"):
    """db_path 只被需要本地索引的源使用（聚宽社区帖库）。"""
    names = names or DEFAULT_SOURCES
    out = []
    for n in names:
        cls = _REGISTRY.get(n)
        if cls is None:
            continue
        out.append(cls(db_path=db_path) if n == "joinquant" else cls())
    return out


__all__ = ["PaperSource", "dedupe", "ArxivSource", "OpenAlexSource",
           "CrossrefSource", "JoinQuantSource", "build_sources",
           "DEFAULT_SOURCES"]
