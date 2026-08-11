import re
from typing import Literal

_MATH = re.compile(r"^[\s\d.+*/%()^-]+$")
_AGENT_HINTS = re.compile(
    r"\b(search|find|browse|read|write|create|save|remember|file|fetch|http|compare|research)\b",
    re.IGNORECASE,
)


def classify(text: str) -> Literal["deterministic", "fast", "agent"]:
    clean = text.strip()
    possible_math = re.sub(
        r"^(what is|calculate|compute)\s+", "", clean, flags=re.IGNORECASE
    ).rstrip("? ")
    if possible_math and _MATH.fullmatch(possible_math):
        return "deterministic"
    if _AGENT_HINTS.search(clean) or len(clean.split()) > 45:
        return "agent"
    return "fast"


def extract_math(text: str) -> str:
    return (
        re.sub(r"^(what is|calculate|compute)\s+", "", text.strip(), flags=re.IGNORECASE)
        .rstrip("? ")
        .replace("^", "**")
    )
