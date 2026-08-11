import re
from typing import Literal


COMPLEX_MARKERS = re.compile(
    r"\b(compare|research|investigate|analy[sz]e|design|architect|implement|"
    r"debug|refactor|migrate|evaluate|trade-?offs?|step[- ]by[- ]step|plan)\b",
    re.IGNORECASE,
)


def classify_complexity(task: str, force_mode: str | None = None) -> Literal["fast", "deliberate"]:
    """Cheap gate that intentionally avoids spending an LLM call on routing."""
    if force_mode in {"fast", "deliberate"}:
        return force_mode
    words = task.split()
    multi_part = len(re.findall(r"(?:\band\b|;|\n|\bthen\b)", task, re.I)) >= 2
    return "deliberate" if len(words) > 45 or multi_part or COMPLEX_MARKERS.search(task) else "fast"

