from typing import Literal

from pydantic import BaseModel, Field


class PlanOutput(BaseModel):
    goal: str
    steps: list[str] = Field(min_length=1, max_length=5)


class VerificationOutput(BaseModel):
    verdict: Literal["pass", "retry", "replan"]
    feedback: str
    verified_facts: list[str] = Field(default_factory=list, max_length=5)


class MemoryItem(BaseModel):
    kind: Literal["fact", "preference", "task_summary", "important_evidence"]
    content: str


class MemoryOutput(BaseModel):
    memories: list[MemoryItem] = Field(default_factory=list, max_length=3)

