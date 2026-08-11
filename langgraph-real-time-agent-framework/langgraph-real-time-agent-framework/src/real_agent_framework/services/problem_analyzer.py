import re
from typing import Literal

_COMPLEX_HINTS = re.compile(
    r"\b(research|investigate|compare|evaluate|analyze|analyse|report|sources?|citations?|"
    r"search|latest|current|study|literature|recommend|database|document|file|api|"
    r"multi-step|pros and cons|create|write)\b",
    re.IGNORECASE,
)


class ProblemAnalyzer:
    """Low-latency deterministic routing; uncertain or action-oriented work is planned."""

    def classify(self, query: str, force_plan: bool = False) -> Literal["simple", "planned"]:
        if force_plan:
            return "planned"
        clean = query.strip()
        if _COMPLEX_HINTS.search(clean) or len(clean.split()) > 30:
            return "planned"
        return "simple"
