import json
from collections.abc import Callable
from time import perf_counter

from langchain_core.tools import BaseTool

Emit = Callable[[str, str, dict], None]


class ToolRouter:
    """Single governed gateway for all Solver tool execution."""

    def __init__(self, tools: list[BaseTool]):
        self.tools = tools
        self._by_name = {item.name: item for item in tools}

    @property
    def descriptions(self) -> list[dict]:
        return [
            {"name": item.name, "description": item.description, "args": item.args}
            for item in self.tools
        ]

    @staticmethod
    def normalize_arguments(value: object) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {"input": value}
            except json.JSONDecodeError:
                return {"input": value}
        return {}

    def execute(self, name: str, arguments: object, emit: Emit) -> dict:
        args = self.normalize_arguments(arguments)
        emit("tool_start", f"Running {name}", {"tool": name, "arguments": args})
        started = perf_counter()
        tool = self._by_name.get(name)
        if tool is None:
            output = f"Unknown or unauthorized tool: {name}"
            success = False
        else:
            try:
                output = str(tool.invoke(args))[:60000]
                success = True
            except Exception as exc:  # noqa: BLE001 - third-party tools can raise arbitrary errors.
                output = f"Tool error ({type(exc).__name__}): {exc}"
                success = False
        elapsed = round(perf_counter() - started, 3)
        observation = {
            "tool": name,
            "arguments": args,
            "output": output,
            "success": success,
            "elapsed_seconds": elapsed,
        }
        emit(
            "tool_result",
            f"{name} {'completed' if success else 'failed'} in {elapsed}s",
            observation | {"output": output[:2000]},
        )
        return observation
