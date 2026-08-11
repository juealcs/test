# LangGraph Multi-Agent Research Framework

A reusable planner-supervisor framework for research experiments. It runs from a terminal, supports any OpenAI-compatible model server, and shows what each agent, tool, and memory layer is doing.

## Included agents

- **Router:** keeps trivial work out of expensive planning loops.
- **Planner:** produces a structured, dependency-aware task plan.
- **Supervisor:** delegates each task and tracks shared progress.
- **Researcher:** web search, URL retrieval, files, time, and memory tools.
- **Analyst:** calculations, comparison, reasoning, files, and memory tools.
- **Writer:** synthesis, report drafting, file output, and memory tools.
- **Reviewer:** quality gate for completeness, support, calculations, and clarity.
- **Reviser:** fixes material reviewer findings within a bounded loop.
- **Finalizer:** produces the final user-facing result.
- **Memory manager:** retrieves and writes user-scoped durable memory.

## Memory and observability

- Short-term chat history: LangGraph `SqliteSaver` in `.data/short_term_checkpoints.sqlite`.
- Long-term memory: user-scoped SQLite FTS5 in `.data/long_term_memory.sqlite`.
- Working/collaboration memory: plan, task outputs, draft, and review in LangGraph state.
- Audit trail: every stage and tool call in `.data/audit.sqlite`, replayable by `run_id`.
- Live terminal trace: stage names, plan, delegation, tool inputs/results, agent output, review, and memory operations.

## Start

Read [RUN_GUIDE.md](RUN_GUIDE.md) for complete PyCharm and model-server instructions.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env                     # Windows PowerShell: Copy-Item .env.example .env
research-agents ask "Research the benefits and limitations of agent memory" --force-plan
```

Interactive mode:

```bash
research-agents chat --user alice --thread research-1 --force-plan
```

## Inspect state

```bash
research-agents memory list --user alice
research-agents history --user alice --thread research-1
research-agents trace YOUR_RUN_ID
research-agents graph
```

## Extend with a domain tool

See `examples/custom_tool.py`. `MultiAgentRuntime(extra_tools=...)` accepts any LangChain-compatible tool and the roles allowed to use it.

## Verify

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

This is a research framework, not an unrestricted autonomous system. Put human approval before consequential tools such as email, purchases, database writes, deployment, or deletion.

