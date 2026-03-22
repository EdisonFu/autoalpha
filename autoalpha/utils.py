import re


def strip_code_fence(text: str) -> str:
    """Remove LLM markdown code fences (```python ... ```) from a string."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    return re.sub(r"\n?```$", "", text).strip()
