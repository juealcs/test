# PyCharm terminal run guide

## 1. Open the extracted directory

Open `langgraph-real-time-agent-framework` as the PyCharm project. Select a Python 3.10–3.12 interpreter.

In the PyCharm terminal:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
Copy-Item .env.example .env
```

## 2. Choose a model connection

### Official paid OpenAI

Put this in `.env`:

```env
MODEL_NAME=gpt-4.1-mini
OPENAI_API_KEY=sk-your-real-key
OPENAI_BASE_URL=
```

### Local vLLM

Use a separate terminal/environment for the model server:

```bash
python -m pip install vllm
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000
```

Keep the server terminal running. In the project `.env`:

```env
MODEL_NAME=Qwen/Qwen3-8B
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://localhost:8000/v1
```

Verify from Bash:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
```

Verify from PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/v1/models
```

Use the exact returned model `id` for `MODEL_NAME`. The Solver model must support OpenAI-style tool calls for tool use. vLLM generally requires Linux/WSL or a Linux server with a compatible GPU stack.

### Other OpenAI-compatible providers

Set their model ID, API key, and `/v1` base URL. This includes compatible hosted endpoints and local servers. Protocol differences vary, so test structured output and tool calling before a long experiment.

## 3. Run and see every stage

```bash
agent-framework ask \
  "Research current short-term and long-term memory designs for AI agents, compare them, and cite sources" \
  --user alice \
  --thread memory-study \
  --force-plan
```

Typical live output:

```text
SYSTEM                     run_start       Started durable graph run
MEMORY_SERVICE             memory_read     Loaded recent messages and memories
PROBLEM_ANALYZER_SERVICE   routing         Complex request sent to Planner
PLANNER_AGENT              plan_created    Created 3 execution steps
SOLVER_AGENT               step_start      Executing step-1
TOOL_ROUTER_SERVICE        tool_start      Running web_search
TOOL_ROUTER_SERVICE        tool_result     web_search completed
SOLVER_AGENT               step_result     Completed step-1
VERIFIER_AGENT             verification    Verification PASS
OUTPUT_SERVICE             final_start     Streaming verified answer
MEMORY_SERVICE             memory_write    Saved durable memories
```

Use `--compact` to hide result previews or `--quiet` to show only the final answer.

## 4. Durable ChatGPT-style conversation

Create a named session:

```bash
agent-framework sessions new --user alice --title "Agent memory study"
```

Copy the returned thread ID, then start chat:

```bash
agent-framework chat --user alice --thread THREAD_ID --force-plan
```

Exit and run the same command later. The complete transcript and checkpoint are restored. Inside chat:

```text
/history
/memory
/quit
```

Manage conversations:

```bash
agent-framework sessions list --user alice
agent-framework sessions rename THREAD_ID "New title" --user alice
agent-framework sessions delete THREAD_ID --user alice
```

Deleting a session removes its transcript, summary, and LangGraph checkpoint. It does not automatically delete user-wide long-term memories.

## 5. Inspect memory, runs, and results

```bash
agent-framework history --user alice --thread THREAD_ID
agent-framework memory list --user alice
agent-framework memory delete MEMORY_ID --user alice
agent-framework runs --user alice --thread THREAD_ID
agent-framework trace RUN_ID
```

Each run also produces:

```text
.data/results/RUN_ID.json
```

That file contains the query, plan, every Solver step result, tool observations, verification, replan count, final answer, models, event log, and chat message IDs.

## 6. Tools and workspace

The following tools are included:

- `web_search`
- `fetch_url`
- `api_get`
- `calculator`
- `utc_now`
- `read_file`
- `list_files`
- `document_search`
- `sqlite_read_query`
- optional `write_file`

Place local files and SQLite databases under `workspace/`. File paths outside this directory are rejected. To enable writing:

```env
ENABLE_FILE_WRITE=true
```

## Troubleshooting

`agent-framework: command not found`: run `pip install -e . --no-deps` from the project root, or use `python -m real_agent_framework.cli ...`.

You can also use the PyCharm-friendly root entry point:

```bash
python main.py ask "Your question" --user alice --thread demo --force-plan
```

`vllm: command not found`: vLLM is not installed in the active model-serving environment.

`Connection refused`: the model server is stopped, on another port, or `localhost` refers to another container/machine.

Malformed plan or verification JSON: the framework validates structured output and uses safe fallbacks, but a stronger instruction model will be more reliable.

Tool-call errors: confirm the served model and chat template support OpenAI-compatible function/tool calling.
