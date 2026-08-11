# Agentic LangGraph Framework

A small, production-oriented Python starter for agents that should answer easy requests quickly and use tools only when useful.

## What is included

- **LangGraph orchestration:** explicit routes and bounded tool loops.
- **Short-term memory:** complete per-thread message state in a SQLite checkpointer.
- **Long-term memory:** user-scoped SQLite/FTS5 facts, isolated by `user_id`.
- **Fast path:** greetings, arithmetic, time, and direct factual prompts skip planning loops.
- **Agent path:** the model selects tools and may iterate up to `MAX_TOOL_LOOPS`.
- **Safe tools:** calculator, UTC time, sandboxed file read/write/list, optional HTTP GET, and memory save/search.
- **Extensible registry:** add any LangChain-compatible `@tool` without changing the graph.
- **CLI and tests:** interactive shell plus unit coverage that does not need an API key.

This project uses architectural ideas from LangGraph's state graph, prebuilt `ToolNode`, and SQLite checkpoint packages. It does not copy the supplied benchmark script.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY (or OPENAI_BASE_URL for an OpenAI-compatible server)
agentic-ai chat --user alice --thread support-1
```

Single request:

```powershell
agentic-ai ask "What is (81 / 9) + 4?" --user alice --thread demo
```

## Runtime flow

```text
request -> load long-term memories -> route
                                      |-> fast answer -> save useful facts -> end
                                      |-> agent -> tools -> agent (bounded) -> save useful facts -> end
```

The graph does **not** create a plan node. The router sends simple work through one model call; deterministic arithmetic can run without a model. Complex work gets a ReAct-style tool loop with a hard iteration limit.

## Memory model

- `thread_id` selects a short-term conversation checkpoint.
- `user_id` selects long-term facts shared across that user's threads.
- The agent can explicitly call `save_memory`; a final memory-curation node also extracts only durable preferences/facts.
- Use different `user_id` values for tenant isolation. For a larger deployment, replace `LongTermMemory` with Postgres/vector search while keeping its interface.

## Add a tool

```python
from langchain_core.tools import tool
from agentic_framework import AgentRuntime


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order in the company's system."""
    return "..."


runtime = AgentRuntime(extra_tools=[lookup_order])
```

Tools execute code. Review permissions, validate arguments, and add human approval before sensitive writes in a real deployment.

## Test

```powershell
pytest
ruff check .
```

See `ARCHITECTURE.md` for design and extension points.

