import json
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .models import Plan, PlanStep, Review, StepResult
from .prompts import PLANNER, REVIEWER, SOLVER, SYNTHESIZER

T = TypeVar("T", bound=BaseModel)


class SolverBackend(Protocol):
    async def plan(self, goal: str) -> Plan: ...
    async def execute(self, goal: str, step: PlanStep, prior: list[StepResult]) -> StepResult: ...
    async def review(self, goal: str, plan: Plan, results: list[StepResult]) -> Review: ...
    async def synthesize(self, goal: str, results: list[StepResult], review: Review) -> str: ...


class VllmAgentBackend:
    """Microsoft Agent Framework using a local OpenAI-compatible vLLM server."""

    def __init__(self, *, base_url: str, api_key: str, model: str):
        from agent_framework.openai import OpenAIChatCompletionClient

        client = OpenAIChatCompletionClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self.planner = client.as_agent(name="Planner", instructions=PLANNER)
        self.solver = client.as_agent(name="Solver", instructions=SOLVER)
        self.reviewer = client.as_agent(name="Reviewer", instructions=REVIEWER)
        self.synthesizer = client.as_agent(name="Synthesizer", instructions=SYNTHESIZER)

    async def _structured(self, agent, prompt: str, response_type: type[T]) -> T:
        from agent_framework import ChatOptions, Message

        response = await agent.run(
            [Message(role="user", contents=prompt)],
            options=ChatOptions(response_format=response_type),
        )
        return response_type.model_validate_json(response.messages[-1].text)

    async def plan(self, goal: str) -> Plan:
        return await self._structured(self.planner, goal, Plan)

    async def execute(self, goal: str, step: PlanStep, prior: list[StepResult]) -> StepResult:
        payload = {
            "goal": goal,
            "step": step.model_dump(),
            "prior_results": [item.model_dump() for item in prior],
        }
        return await self._structured(self.solver, json.dumps(payload, default=str), StepResult)

    async def review(self, goal: str, plan: Plan, results: list[StepResult]) -> Review:
        payload = {
            "goal": goal,
            "plan": plan.model_dump(),
            "results": [item.model_dump() for item in results],
        }
        return await self._structured(self.reviewer, json.dumps(payload, default=str), Review)

    async def synthesize(self, goal: str, results: list[StepResult], review: Review) -> str:
        payload = {
            "goal": goal,
            "results": [item.model_dump() for item in results],
            "review": review.model_dump(),
        }
        response = await self.synthesizer.run(json.dumps(payload, default=str))
        return response.text
