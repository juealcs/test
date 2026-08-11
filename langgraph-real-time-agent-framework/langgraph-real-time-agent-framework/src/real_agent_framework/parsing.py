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


def parse_json(text: str) -> object:
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        candidates = [index for index in (clean.find("{"), clean.find("[")) if index >= 0]
        if not candidates:
            raise
        start = min(candidates)
        end = max(clean.rfind("}"), clean.rfind("]"))
        if end <= start:
            raise
        return json.loads(clean[start : end + 1])


def parse_model(text: str, schema: type[T]) -> T | None:
    try:
        return schema.model_validate(parse_json(text))
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        return None
