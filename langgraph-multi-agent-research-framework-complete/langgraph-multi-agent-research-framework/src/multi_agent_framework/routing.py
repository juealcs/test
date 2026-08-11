import re
from typing import Literal

_MATH = re.compile(r"^[\s\d.+*/%()^-]+$")
_COMPLEX = re.compile(
    r"\b(research|investigate|compare|evaluate|analyze|analyse|report|sources?|citations?|"
    r"web|search|latest|current|study|literature|pros and cons|recommend|file|create|write)\b",
    re.IGNORECASE,
)


def classify(query: str, force_plan: bool = False) -> Literal["fast", "planner"]:
    if force_plan:
        return "planner"
    clean = query.strip()
    possible_math = re.sub(
        r"^(what is|calculate|compute)\s+", "", clean, flags=re.IGNORECASE
    ).rstrip("? ")
    if possible_math and _MATH.fullmatch(possible_math):
        return "fast"
    if _COMPLEX.search(clean) or len(clean.split()) > 30:
        return "planner"
    return "fast"


def extract_math(query: str) -> str | None:
    clean = re.sub(
        r"^(what is|calculate|compute)\s+", "", query.strip(), flags=re.IGNORECASE
    ).rstrip("? ")
    if clean and _MATH.fullmatch(clean):
        return clean.replace("^", "**")
    return None
