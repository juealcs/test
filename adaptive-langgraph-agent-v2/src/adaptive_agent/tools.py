from __future__ import annotations

import ast
import ipaddress
import operator
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from langchain_core.tools import BaseTool, tool


_OPS: dict[type, Callable] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float | int:
    def evaluate(node: ast.AST):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.operand))
        raise ValueError("Only numeric arithmetic is allowed")
    if len(expression) > 200:
        raise ValueError("Expression is too long")
    return evaluate(ast.parse(expression, mode="eval").body)


def default_tools(workspace: Path) -> list[BaseTool]:
    root = workspace.resolve()

    def resolve(relative_path: str) -> Path:
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Path must remain inside the configured workspace")
        return candidate

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a numeric arithmetic expression safely."""
        return str(safe_calculate(expression))

    @tool
    def current_utc_time() -> str:
        """Return the current UTC date and time."""
        return datetime.now(timezone.utc).isoformat()

    @tool
    def list_files(relative_directory: str = ".") -> str:
        """List files under a directory in the configured workspace."""
        directory = resolve(relative_directory)
        return "\n".join(str(p.relative_to(root)) for p in sorted(directory.iterdir())[:200])

    @tool
    def read_text_file(relative_path: str) -> str:
        """Read a UTF-8 text file from the configured workspace."""
        path = resolve(relative_path)
        if path.stat().st_size > 200_000:
            raise ValueError("File is larger than 200 KB")
        return path.read_text(encoding="utf-8")

    @tool
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the public web for current or external information and return cited URLs."""
        limit = max(1, min(max_results, 8))
        rows = DDGS().text(query, max_results=limit)
        results = [
            {"title": row.get("title", ""), "summary": row.get("body", "")[:500], "url": row.get("href", "")}
            for row in rows
        ]
        return "\n\n".join(
            f"[{index}] {item['title']}\n{item['summary']}\nURL: {item['url']}"
            for index, item in enumerate(results, 1)
        ) or "No search results found."

    @tool
    def fetch_url(url: str) -> str:
        """Fetch public HTTP(S) text content. Use for a known URL, not general search."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("A public HTTP(S) URL is required")
        for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
            if ipaddress.ip_address(info[4][0]).is_private:
                raise ValueError("Private network destinations are blocked")
        response = httpx.get(url, follow_redirects=True, timeout=10)
        response.raise_for_status()
        return response.text[:50_000]

    return [calculator, current_utc_time, list_files, read_text_file, web_search, fetch_url]
