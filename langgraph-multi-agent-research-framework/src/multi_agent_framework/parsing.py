import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def json_payload(text: str) -> object:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start_candidates = [i for i in (clean.find("{"), clean.find("[")) if i >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        end = max(clean.rfind("}"), clean.rfind("]"))
        if end <= start:
            raise
        return json.loads(clean[start : end + 1])


def parse_model(text: str, model: type[T]) -> T | None:
    try:
        return model.model_validate(json_payload(text))
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return None
