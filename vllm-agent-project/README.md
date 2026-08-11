# Local vLLM Agentic Problem Solver

A reusable **plan → execute → verify → replan → answer** framework using Microsoft Agent Framework with a local open-source model served by vLLM. No paid model API is required.

## 1. Install the application

Python 3.10+ is required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

## 2. Start vLLM

vLLM is normally run on Linux or WSL2 with a supported NVIDIA GPU. In a separate environment/terminal:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto --api-key local-dev-key
```

The model can be replaced with any instruction model that fits your hardware and reliably supports JSON structured output. Update `VLLM_MODEL` in `.env` to exactly match the served model ID.

## 3. Run a problem

```powershell
solve "Create a phased migration plan from a monolith to microservices with rollback criteria"
```

Or:

```powershell
python -m agentic_solver.cli "Explain and solve this business problem"
```

## Configuration

```dotenv
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=local-dev-key
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
MAX_STEPS=8
MAX_REPLANS=2
```

The API key is local authentication between this app and your vLLM server; it is not a paid service key. The client uses vLLM's OpenAI-compatible Chat Completions endpoint and structured JSON responses.

## Test

```powershell
pytest
```
