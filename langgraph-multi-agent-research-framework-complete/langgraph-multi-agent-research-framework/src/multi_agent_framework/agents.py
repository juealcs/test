import json
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from .parsing import message_text

EmitEvent = Callable[[str, str, dict], dict]


class SpecialistAgent:
    """A bounded tool-calling specialist used by researcher, analyst, and writer nodes."""

    def __init__(self, model: BaseChatModel, tools: list[BaseTool], max_tool_loops: int):
        self.model = model
        self.tools = tools
        self.max_tool_loops = max_tool_loops
        self.tool_map = {item.name: item for item in tools}

    @staticmethod
    def _arguments(value: object) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {"input": value}
            except json.JSONDecodeError:
                return {"input": value}
        return {}

    def run(
        self,
        system_prompt: str,
        task_prompt: str,
        emit: EmitEvent,
    ) -> tuple[str, list[str], list[dict]]:
        dialog = [SystemMessage(content=system_prompt), HumanMessage(content=task_prompt)]
        tools_used: list[str] = []
        events: list[dict] = []
        bound_model = self.model.bind_tools(self.tools) if self.tools else self.model

        for iteration in range(1, self.max_tool_loops + 1):
            response = bound_model.invoke(dialog)
            dialog.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                return message_text(response.content), tools_used, events

            for call in tool_calls:
                name = str(call.get("name", ""))
                arguments = self._arguments(call.get("args", {}))
                call_id = str(call.get("id") or f"tool-{iteration}-{len(tools_used) + 1}")
                tool = self.tool_map.get(name)
                if tool is None:
                    output = f"Unknown or unauthorized tool: {name}"
                else:
                    try:
                        output = str(tool.invoke(arguments))
                    except Exception as exc:  # noqa: BLE001 - third-party tools may raise anything.
                        output = f"Tool error ({type(exc).__name__}): {exc}"
                tools_used.append(name)
                events.append(
                    emit(
                        "tool",
                        f"Called {name}",
                        {
                            "iteration": iteration,
                            "arguments": arguments,
                            "output_preview": output[:800],
                        },
                    )
                )
                dialog.append(ToolMessage(content=output, tool_call_id=call_id, name=name))

        limit_message = SystemMessage(
            content=(
                "The tool-call limit is reached. Give the best complete result using the tool "
                "outputs already available. Do not request another tool."
            )
        )
        response = self.model.invoke([*dialog, limit_message])
        if isinstance(response, AIMessage):
            return message_text(response.content), tools_used, events
        return str(response), tools_used, events
