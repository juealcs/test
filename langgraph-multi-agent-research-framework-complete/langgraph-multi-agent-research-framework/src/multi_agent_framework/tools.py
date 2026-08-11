import ast
import ipaddress
import operator
import socket
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

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
    ast.UAdd: operator.pos,
}


def safe_calculate(expression: str) -> int | float:
    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is outside the allowed range")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.operand))
        raise ValueError("Only numeric arithmetic is allowed")

    if len(expression) > 300:
        raise ValueError("Expression is too long")
    result = evaluate(ast.parse(expression, mode="eval"))
    if isinstance(result, complex) or abs(result) > 1e100:
        raise ValueError("Result is outside the allowed range")
    return result


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a numeric arithmetic expression."""
    return str(safe_calculate(expression.replace("^", "**")))


@tool
def utc_now() -> str:
    """Return the current UTC date and time."""
    return datetime.now(timezone.utc).isoformat()


def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only HTTP(S) URLs are allowed")
    for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError("Private, loopback, and link-local addresses are blocked")


class ToolRegistry:
    """Role-based tool registry. Add tools without changing the graph."""

    def __init__(
        self,
        workspace: Path,
        memory: LongTermMemory,
        enable_web_search: bool = True,
        enable_fetch_url: bool = True,
        web_search_max_results: int = 5,
    ):
        self.workspace = workspace.resolve()
        self.memory = memory
        self.web_search_max_results = web_search_max_results
        self._role_tools: dict[str, list[BaseTool]] = {
            "researcher": [utc_now],
            "analyst": [calculator, utc_now],
            "writer": [],
        }
        self._add_workspace_tools()
        if enable_web_search:
            self._add_web_search()
        if enable_fetch_url:
            self._add_fetch_url()

    def _safe_path(self, relative_path: str) -> Path:
        target = (self.workspace / relative_path).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ValueError("Path must stay inside WORKSPACE_DIR")
        return target

    def _add_workspace_tools(self) -> None:
        def read_file(relative_path: str) -> str:
            """Read a UTF-8 text file inside the configured workspace."""
            return self._safe_path(relative_path).read_text(encoding="utf-8")[:50000]

        def write_file(relative_path: str, content: str) -> str:
            """Write a UTF-8 text file inside the configured workspace."""
            target = self._safe_path(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {target.relative_to(self.workspace)}"

        def list_files(relative_path: str = ".") -> str:
            """List up to 200 entries in a workspace directory."""
            target = self._safe_path(relative_path)
            return "\n".join(
                str(item.relative_to(self.workspace)) for item in list(target.iterdir())[:200]
            )

        file_tools = [
            StructuredTool.from_function(read_file),
            StructuredTool.from_function(write_file),
            StructuredTool.from_function(list_files),
        ]
        for role in self._role_tools:
            self._role_tools[role].extend(file_tools)

    def _add_web_search(self) -> None:
        max_results = self.web_search_max_results

        def web_search(query: str) -> str:
            """Search the public web and return titles, URLs, and snippets."""
            from ddgs import DDGS

            results = list(DDGS().text(query, max_results=max_results))
            if not results:
                return "No web results found."
            blocks = []
            for index, result in enumerate(results, 1):
                blocks.append(
                    f"[{index}] {result.get('title', 'Untitled')}\n"
                    f"URL: {result.get('href', '')}\n"
                    f"Snippet: {result.get('body', '')}"
                )
            return "\n\n".join(blocks)

        self._role_tools["researcher"].append(StructuredTool.from_function(web_search))

    def _add_fetch_url(self) -> None:
        def fetch_url(url: str) -> str:
            """Fetch text from a public HTTP(S) URL after blocking private-network addresses."""
            _assert_public_url(url)
            response = httpx.get(
                url,
                timeout=15,
                follow_redirects=True,
                headers={"User-Agent": "LangGraphResearchFramework/0.1"},
            )
            response.raise_for_status()
            return response.text[:60000]

        self._role_tools["researcher"].append(StructuredTool.from_function(fetch_url))

    def for_agent(self, role: str, user_id: str, thread_id: str) -> list[BaseTool]:
        def search_memory(query: str) -> str:
            """Search durable memories belonging to the current user."""
            rows = self.memory.search(user_id, query)
            return "\n".join(row["text"] for row in rows) or "No memories found."

        def save_memory(text: str, category: str = "profile") -> str:
            """Save an explicit durable user fact or preference."""
            saved = self.memory.save(user_id, text, category, thread_id)
            return "Memory saved." if saved else "Memory was not saved."

        memory_tools = [
            StructuredTool.from_function(search_memory),
            StructuredTool.from_function(save_memory),
        ]
        return [*self._role_tools.get(role, []), *memory_tools]

    def add_tool(self, roles: Sequence[str], new_tool: BaseTool) -> None:
        for role in roles:
            self._role_tools.setdefault(role, []).append(new_tool)
