# LangGraph Real-Time Agent Framework

A durable, observable problem-solving framework with exactly three agents:

1. **Planner Agent** — decomposes complex work and replans from feedback.
2. **Solver Agent** — executes every step through a governed tool gateway.
3. **Verifier Agent** — checks correctness, evidence, freshness, completeness, and safety.

Problem analysis, tools, memory, chat history, summaries, and output streaming are services—not extra agents.

## What is real and persistent

- Full ChatGPT-style conversation archive with list/resume/rename/delete commands.
- LangGraph SQLite checkpoints for each `user_id + thread_id` session.
- Short-term context using recent messages plus rolling summaries.
- User-isolated long-term memory with FTS5 retrieval and source provenance.
- Immediate terminal events for every agent stage, plan step, and tool call.
- Bounded tool and replan loops.
- Per-run audit events and complete `.data/results/RUN_ID.json` result files.
- Paid OpenAI and local/hosted OpenAI-compatible APIs, including vLLM.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env
agent-framework ask "Research current agent-memory approaches and compare them with sources" --force-plan
```

Windows PowerShell activation and file copy:

```powershell
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

See [RUN_GUIDE.md](RUN_GUIDE.md) for OpenAI, vLLM, PyCharm, session, memory, and troubleshooting instructions.

## Session commands

```bash
agent-framework sessions new --user alice --title "Memory research"
agent-framework sessions list --user alice
agent-framework chat --user alice --thread THREAD_ID --force-plan
agent-framework history --user alice --thread THREAD_ID
agent-framework runs --user alice --thread THREAD_ID
agent-framework trace RUN_ID
agent-framework memory list --user alice
```

## Safety defaults

File writes are disabled by default. Files are confined to `WORKSPACE_DIR`. URL tools block private and loopback targets. SQLite access is read-only. Arbitrary shell and Python execution are deliberately not included; add them only behind an actual sandbox and approval layer.

## Verification

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

