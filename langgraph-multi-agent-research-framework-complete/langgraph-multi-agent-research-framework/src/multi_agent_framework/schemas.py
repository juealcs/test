from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

AgentRole = Literal["researcher", "analyst", "writer"]


class PlanTask(BaseModel):
    id: str
    agent: AgentRole
    instruction: str
    expected_output: str = "A concise, evidence-based result"
    depends_on: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    objective: str
    rationale: str = ""
    tasks: list[PlanTask]


class TaskResult(BaseModel):
    task_id: str
    agent: AgentRole
    instruction: str
    output: str
    tools_used: list[str] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    approved: bool
    feedback: str
    missing_items: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    run_id: str
    node: str
    agent: str
    kind: str
    message: str
    details: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
