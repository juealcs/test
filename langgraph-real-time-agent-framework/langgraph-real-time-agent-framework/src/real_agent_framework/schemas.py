from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: str
    description: str
    expected_output: str
    suggested_tools: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    objective: str
    reasoning: str = ""
    steps: list[PlanStep]


class StepResult(BaseModel):
    step_id: str
    description: str
    output: str
    tools_used: list[str] = Field(default_factory=list)
    observations: list[dict] = Field(default_factory=list)


class VerificationResult(BaseModel):
    status: Literal["pass", "fail"]
    feedback: str
    missing_items: list[str] = Field(default_factory=list)
    replan_required: bool = False
    confidence: float = Field(default=0.5, ge=0, le=1)


class TraceEvent(BaseModel):
    run_id: str
    stage: str
    actor: str
    kind: str
    message: str
    details: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
