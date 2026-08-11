import ast
import ipaddress
import json
import operator
import socket
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from langchain_core.tools import BaseTool, StructuredTool, tool

from ..config import Settings

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


def _safe_path(workspace: Path, relative_path: str) -> Path:
    root = workspace.resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path must stay inside WORKSPACE_DIR")
    return target


def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only HTTP(S) URLs are allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for info in socket.getaddrinfo(parsed.hostname, port):
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError("Private, loopback, and link-local network addresses are blocked")


def _public_get(url: str, timeout: float) -> httpx.Response:
    current = url
    with httpx.Client(
        timeout=timeout, follow_redirects=False, headers={"User-Agent": "RealAgentFramework/1.0"}
    ) as client:
        for _ in range(4):
            _assert_public_url(current)
            response = client.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    return response
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            return response
    raise ValueError("Too many HTTP redirects")


def build_tools(settings: Settings) -> list[BaseTool]:
    workspace = settings.workspace_dir
    tools: list[BaseTool] = [calculator, utc_now]

    def read_file(relative_path: str) -> str:
        """Read a UTF-8 text file inside the configured workspace."""
        return _safe_path(workspace, relative_path).read_text(encoding="utf-8")[:60000]

    def list_files(relative_path: str = ".") -> str:
        """List up to 200 entries in a workspace directory."""
        target = _safe_path(workspace, relative_path)
        return "\n".join(
            str(item.relative_to(workspace.resolve())) for item in list(target.iterdir())[:200]
        )

    def document_search(query: str, relative_path: str = ".", max_results: int = 8) -> str:
        """Search text-like workspace documents and return file, line, and matching snippets."""
        root = _safe_path(workspace, relative_path)
        terms = [term.lower() for term in query.split() if len(term) > 1][:12]
        if not terms:
            return "No searchable terms provided."
        extensions = {".txt", ".md", ".py", ".json", ".csv", ".rst"}
        matches: list[tuple[int, str]] = []
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                ):
                    score = sum(term in line.lower() for term in terms)
                    if score:
                        relative = path.relative_to(workspace.resolve())
                        matches.append((score, f"{relative}:{line_number}: {line.strip()[:500]}"))
            except OSError:
                continue
        matches.sort(key=lambda item: item[0], reverse=True)
        return "\n".join(text for _, text in matches[:max_results]) or "No matches found."

    tools.extend(
        [
            StructuredTool.from_function(read_file),
            StructuredTool.from_function(list_files),
            StructuredTool.from_function(document_search),
        ]
    )

    if settings.enable_file_write:

        def write_file(relative_path: str, content: str) -> str:
            """Write a UTF-8 file inside the workspace. This tool is disabled by default."""
            target = _safe_path(workspace, relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {target.relative_to(workspace.resolve())}"

        tools.append(StructuredTool.from_function(write_file))

    if settings.enable_web_search:
        max_results = settings.web_search_max_results

        def web_search(query: str) -> str:
            """Search the public web and return source titles, URLs, and snippets."""
            from ddgs import DDGS

            results = list(DDGS().text(query, max_results=max_results))
            blocks = []
            for index, result in enumerate(results, 1):
                blocks.append(
                    f"[{index}] {result.get('title', 'Untitled')}\n"
                    f"URL: {result.get('href', '')}\n"
                    f"Snippet: {result.get('body', '')}"
                )
            return "\n\n".join(blocks) or "No web results found."

        tools.append(StructuredTool.from_function(web_search))

    if settings.enable_url_fetch:
        timeout = settings.tool_timeout_seconds

        def fetch_url(url: str) -> str:
            """Fetch text from a public HTTP(S) page while blocking private-network targets."""
            response = _public_get(url, timeout)
            return response.text[:60000]

        tools.append(StructuredTool.from_function(fetch_url))

    if settings.enable_api_get:
        timeout = settings.tool_timeout_seconds

        def api_get(url: str) -> str:
            """Send GET to a public JSON API. Private-network targets are blocked."""
            response = _public_get(url, timeout)
            try:
                return json.dumps(response.json(), ensure_ascii=False, indent=2)[:60000]
            except ValueError:
                return response.text[:60000]

        tools.append(StructuredTool.from_function(api_get))

    if settings.enable_database_read:

        def sqlite_read_query(database_path: str, query: str) -> str:
            """Run one read-only SELECT/WITH query on a SQLite database inside the workspace."""
            target = _safe_path(workspace, database_path)
            if target.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
                raise ValueError("Only SQLite database files are allowed")
            normalized = query.strip().lower()
            if not normalized.startswith(("select", "with", "pragma table_info")):
                raise ValueError("Only read-only SELECT, WITH, or PRAGMA table_info is allowed")
            connection = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            try:
                rows = connection.execute(query).fetchmany(100)
                return json.dumps([dict(row) for row in rows], ensure_ascii=False, default=str)
            finally:
                connection.close()

        tools.append(StructuredTool.from_function(sqlite_read_query))

    return tools
