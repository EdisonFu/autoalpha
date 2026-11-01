"""多源资料检索。"""

from .base import PaperSource, dedupe
from .arxiv import ArxivSource
from .openalex import OpenAlexSource
from .crossref import CrossrefSource

# 默认启用的源。界面上可逐个开关。
DEFAULT_SOURCES = ("arxiv", "openalex", "crossref")

_REGISTRY = {
    "arxiv": ArxivSource,
    "openalex": OpenAlexSource,
    "crossref": CrossrefSource,
}


def build_sources(names=None):
    names = names or DEFAULT_SOURCES
    return [_REGISTRY[n]() for n in names if n in _REGISTRY]


__all__ = ["PaperSource", "dedupe", "ArxivSource", "OpenAlexSource",
           "CrossrefSource", "build_sources", "DEFAULT_SOURCES"]
