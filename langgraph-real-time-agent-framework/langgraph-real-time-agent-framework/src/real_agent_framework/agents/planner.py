import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..parsing import message_text, parse_model
from ..prompts import PLANNER_PROMPT
from ..schemas import ExecutionPlan, PlanStep


class PlannerAgent:
    def __init__(self, model: BaseChatModel, max_steps: int):
        self.model = model
        self.max_steps = max_steps

    def _fallback(self, query: str) -> ExecutionPlan:
        tools: list[str] = []
        if re.search(r"\b(current|latest|web|search|source|research|news)\b", query, re.IGNORECASE):
            tools.append("web_search")
        if re.search(r"\b(calculate|number|percent|average|total|cost)\b", query, re.IGNORECASE):
            tools.append("calculator")
        return ExecutionPlan(
            objective=query,
            reasoning="Safe fallback plan used because the model did not return a valid plan.",
            steps=[
                PlanStep(
                    id="step-1",
                    description=f"Solve the user's problem completely: {query}",
                    expected_output="A supported result that directly addresses the user",
                    suggested_tools=tools,
                    success_criteria=["Answer the full request", "Do not invent evidence"],
                )
            ],
        )

    def create_plan(
        self,
        query: str,
        summary: str,
        memories: list[dict],
        previous_results: list[dict],
        verifier_feedback: dict,
    ) -> ExecutionPlan:
        request = (
            f"User problem:\n{query}\n\n"
            f"Conversation summary:\n{summary or '(none)'}\n\n"
            f"Relevant user memories:\n{json.dumps(memories, ensure_ascii=False, default=str)}\n\n"
            f"Previous execution results:\n{json.dumps(previous_results, ensure_ascii=False)}\n\n"
            f"Verifier feedback:\n{json.dumps(verifier_feedback, ensure_ascii=False)}"
        )
        response = self.model.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT.format(max_steps=self.max_steps)),
                HumanMessage(content=request),
            ]
        )
        plan = parse_model(message_text(response.content), ExecutionPlan)
        if plan is None or not plan.steps:
            return self._fallback(query)
        plan.steps = plan.steps[: self.max_steps]
        return plan
