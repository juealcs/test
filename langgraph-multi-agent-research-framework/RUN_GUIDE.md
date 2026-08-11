# PyCharm run guide

You need two PyCharm terminal tabs: one for the model server and one for this framework.

## 1. Open and configure the project

Extract the ZIP and open `langgraph-multi-agent-research-framework` as the PyCharm project. In **Settings → Project → Python Interpreter**, select your existing environment or create Python 3.10–3.12 environment.

In Terminal 2, from the project root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env
```

Windows PowerShell uses:

```powershell
Copy-Item .env.example .env
```

## 2. Start an OpenAI-compatible model server in Terminal 1

### vLLM

vLLM generally requires Linux/WSL, a supported GPU, and a compatible CUDA/PyTorch environment. Install it in the model-serving environment:

```bash
python -m pip install vllm
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000
```

Leave this terminal running. Model loading can take several minutes.

### Existing Ollama server

```bash
ollama serve
ollama pull qwen3:8b
```

For Ollama, change `.env` to port `11434` and model `qwen3:8b`.

## 3. Verify the server from Terminal 2

Bash:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
```

PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/v1/models
```

Copy the exact model `id` from this response into `.env`.

## 4. Edit `.env`

For vLLM on port 8000:

```env
MODEL_NAME=Qwen/Qwen3-8B
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://localhost:8000/v1
DATA_DIR=.data
WORKSPACE_DIR=workspace
MAX_PLAN_TASKS=6
MAX_TOOL_LOOPS=4
MAX_REVISIONS=1
MEMORY_RECALL_LIMIT=6
ENABLE_WEB_SEARCH=true
ENABLE_FETCH_URL=true
WEB_SEARCH_MAX_RESULTS=5
```

## 5. Run a visible multi-agent query

```bash
research-agents ask "Research how short-term and long-term memory improve AI agents. Compare approaches and cite sources." --user alice --thread memory-study --force-plan
```

The terminal will show:

```text
SYSTEM          lifecycle      Started a new multi-agent run
MEMORY_MANAGER  memory_read    Recalled long-term memories
ROUTER          routing        Selected planner orchestration
PLANNER         plan           Created a task plan
SUPERVISOR      delegation     Delegated task-1 to researcher
RESEARCHER      tool           Called web_search
RESEARCHER      agent_output   Completed task-1
...
REVIEWER        review         Draft approved / requires revision
FINALIZER       final_answer   Produced final answer
MEMORY_MANAGER  memory_write   Saved durable memories
```

## 6. Interactive research chat

```bash
research-agents chat --user alice --thread memory-study --force-plan
```

Use the same user and thread to continue with checkpointed history. Use the same user and a new thread to start a new conversation while retaining long-term profile memory.

## 7. Inspect memory and traces

```bash
research-agents memory list --user alice
research-agents history --user alice --thread memory-study
research-agents trace RUN_ID_PRINTED_AFTER_THE_ANSWER
research-agents graph
```

## Common errors

`vllm: command not found`: install vLLM in Terminal 1's active environment, or use an already installed model server.

`Connection refused`: the model server is not running, is on another port, or `localhost` refers to a different container/remote host.

`research-agents: command not found`: from the project root run `pip install -e . --no-deps`, then try `python -m multi_agent_framework.cli ...`.

Tool-calling error from a local model: use a model/server configuration that supports OpenAI-style tool calls, or disable complex tools while testing. Qwen instruction models served with an appropriate vLLM chat template generally support this workflow.

