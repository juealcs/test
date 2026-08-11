from enum import Enum

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStep(BaseModel):
    id: str
    objective: str
    success_criteria: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    tool_hint: str | None = None


class Plan(BaseModel):
    goal: str
    assumptions: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(min_length=1)


class StepResult(BaseModel):
    step_id: str
    status: StepStatus
    output: str
    evidence: list[str] = Field(default_factory=list)
    error: str | None = None


class Review(BaseModel):
    passed: bool
    summary: str
    missing: list[str] = Field(default_factory=list)
    revised_plan: Plan | None = None


class RunResult(BaseModel):
    answer: str
    plan: Plan
    steps: list[StepResult]
    review: Review
    replan_count: int = 0
