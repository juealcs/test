import ast
import operator
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx
from langchain_core.tools import BaseTool, StructuredTool, tool

from .memory import LongTermMemory

_OPS: dict[type[ast.AST], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> int | float:
    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.operand))
        raise ValueError("Only numeric arithmetic is allowed")

    if len(expression) > 200:
        raise ValueError("Expression is too long")
    result = evaluate(ast.parse(expression, mode="eval"))
    if isinstance(result, complex) or abs(result) > 1e100:
        raise ValueError("Result is outside the allowed range")
    return result


@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression safely."""
    return str(safe_calculate(expression))


@tool
def utc_now() -> str:
    """Return the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


class ToolRegistry:
    def __init__(self, workspace: Path, memory: LongTermMemory, enable_http: bool = False):
        self.workspace = workspace.resolve()
        self.memory = memory
        self._tools: list[BaseTool] = [calculator, utc_now]
        self._add_workspace_tools()
        if enable_http:
            self._add_http_tool()

    def _safe_path(self, relative_path: str) -> Path:
        target = (self.workspace / relative_path).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ValueError("Path must stay inside WORKSPACE_DIR")
        return target

    def _add_workspace_tools(self) -> None:
        def read_file(relative_path: str) -> str:
            """Read a UTF-8 text file inside the agent workspace."""
            return self._safe_path(relative_path).read_text(encoding="utf-8")[:50000]

        def write_file(relative_path: str, content: str) -> str:
            """Write a UTF-8 text file inside the agent workspace."""
            target = self._safe_path(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {target.relative_to(self.workspace)}"

        def list_files(relative_path: str = ".") -> str:
            """List files inside a directory in the agent workspace."""
            target = self._safe_path(relative_path)
            return "\n".join(
                str(p.relative_to(self.workspace)) for p in list(target.iterdir())[:200]
            )

        self._tools.extend(
            [
                StructuredTool.from_function(read_file),
                StructuredTool.from_function(write_file),
                StructuredTool.from_function(list_files),
            ]
        )

    def _add_http_tool(self) -> None:
        def http_get(url: str) -> str:
            """Fetch public HTTP(S) text. Enable only for trusted deployments."""
            if not url.startswith(("https://", "http://")):
                raise ValueError("Only HTTP(S) URLs are allowed")
            response = httpx.get(url, timeout=10, follow_redirects=True)
            response.raise_for_status()
            return response.text[:50000]

        self._tools.append(StructuredTool.from_function(http_get))

    def for_user(self, user_id: str) -> list[BaseTool]:
        def save_memory(text: str) -> str:
            """Save a durable user fact or preference for future conversations."""
            return "Saved" if self.memory.save(user_id, text) else "Already saved"

        def search_memory(query: str) -> str:
            """Search this user's durable memories."""
            return "\n".join(self.memory.search(user_id, query)) or "No memories found"

        return self._tools + [
            StructuredTool.from_function(save_memory),
            StructuredTool.from_function(search_memory),
        ]

    def add(self, new_tool: BaseTool) -> None:
        self._tools.append(new_tool)
