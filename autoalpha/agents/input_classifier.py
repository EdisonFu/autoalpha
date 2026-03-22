import mimetypes
import os
from urllib.parse import urlparse


def classify_input(raw_input: str) -> tuple[str, float]:
    """
    3-5行规则分类，不调LLM。
    返回 (input_type, search_confidence)
    input_type: idea / pdf / url / text
    search_confidence: 0.0-1.0，用于决定是否触发PaperSearchAgent
    """
    s = raw_input.strip()

    # URL
    parsed = urlparse(s)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return "url", 0.0

    # 本地PDF路径
    if os.path.isfile(s):
        mime, _ = mimetypes.guess_type(s)
        if mime == "application/pdf" or s.lower().endswith(".pdf"):
            return "pdf", 0.0

    # 长文本（含换行或超过200字符）→ 当做已有论文文本
    if "\n" in s or len(s) > 200:
        return "text", 0.0

    # 短句 → 想法，估算搜索置信度
    # 英文按空格分词；中文无空格，用字符数估算（4-60字符视为有效想法）
    word_count = len(s.split())
    char_count = len(s)
    is_short_idea = (2 <= word_count <= 15) or (word_count == 1 and 4 <= char_count <= 60)
    confidence = 0.8 if is_short_idea else 0.5
    return "idea", confidence
