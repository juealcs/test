import json
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..parsing import message_text
from ..prompts import SOLVER_PROMPT
from ..schemas import PlanStep, StepResult
from ..services.tool_router import ToolRouter

Emit = Callable[[str, str, dict], None]


class SolverAgent:
    def __init__(
        self,
        model: BaseChatModel,
        tool_router: ToolRouter,
        max_tool_loops: int,
    ):
        self.model = model
        self.tool_router = tool_router
        self.max_tool_loops = max_tool_loops

    def solve_step(
        self,
        query: str,
        step: PlanStep,
        summary: str,
        memories: list[dict],
        completed_results: list[dict],
        emit: Emit,
    ) -> StepResult:
        task = (
            f"Original user problem:\n{query}\n\n"
            f"Current step ({step.id}):\n{step.description}\n\n"
            f"Expected output:\n{step.expected_output}\n\n"
            f"Success criteria:\n{json.dumps(step.success_criteria)}\n\n"
            f"Suggested tools:\n{json.dumps(step.suggested_tools)}\n\n"
            f"Conversation summary:\n{summary or '(none)'}\n\n"
            f"Relevant long-term memories:\n{json.dumps(memories, ensure_ascii=False, default=str)}\n\n"
            f"Completed results from this run:\n"
            f"{json.dumps(completed_results, ensure_ascii=False, default=str)}"
        )
        dialog = [SystemMessage(content=SOLVER_PROMPT), HumanMessage(content=task)]
        observations: list[dict] = []
        tools_used: list[str] = []
        bound = self.model.bind_tools(self.tool_router.tools)

        for _iteration in range(1, self.max_tool_loops + 1):
            response = bound.invoke(dialog)
            dialog.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return StepResult(
                    step_id=step.id,
                    description=step.description,
                    output=message_text(response.content),
                    tools_used=tools_used,
                    observations=observations,
                )
            for call in tool_calls:
                name = str(call.get("name", ""))
                observation = self.tool_router.execute(name, call.get("args", {}), emit)
                observations.append(observation)
                tools_used.append(name)
                dialog.append(
                    ToolMessage(
                        content=observation["output"],
                        tool_call_id=str(call.get("id") or f"tool-{len(observations)}"),
                        name=name,
                    )
                )

        response = self.model.invoke(
            [
                *dialog,
                SystemMessage(
                    content=(
                        "Tool-call limit reached. Return the best result from existing observations. "
                        "Do not request another tool."
                    )
                ),
            ]
        )
        return StepResult(
            step_id=step.id,
            description=step.description,
            output=message_text(response.content),
            tools_used=tools_used,
            observations=observations,
        )

    def synthesize(self, query: str, results: list[dict], verifier_feedback: dict) -> str:
        request = (
            f"Original user problem:\n{query}\n\n"
            f"Completed step results and evidence:\n"
            f"{json.dumps(results, ensure_ascii=False, default=str, indent=2)}\n\n"
            f"Previous verifier feedback, if any:\n"
            f"{json.dumps(verifier_feedback, ensure_ascii=False)}\n\n"
            "Produce the complete candidate answer. Preserve useful source URLs."
        )
        response = self.model.invoke(
            [SystemMessage(content=SOLVER_PROMPT), HumanMessage(content=request)]
        )
        return message_text(response.content)
