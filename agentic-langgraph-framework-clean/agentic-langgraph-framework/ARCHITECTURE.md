# Architecture

`AgentRuntime` owns the model, tool registry, SQLite resources, and compiled graph. Call `invoke(text, user_id, thread_id)`; never put identity in prompt text.

## Graph nodes

1. `recall`: retrieve relevant long-term memories.
2. `route`: classify deterministic, simple, or agentic work using cheap rules.
3. `fast`: answer directly; arithmetic is evaluated locally and other simple work uses one LLM call.
4. `agent`: bind registered tools to the model and decide whether to call one.
5. `tools`: LangGraph `ToolNode` executes validated tool calls.
6. `curate_memory`: ask the model for a small JSON list of durable facts and persist them.

The conditional edge after `agent` enforces `MAX_TOOL_LOOPS`. This prevents accidental infinite loops and returns the best available answer.

## Production extension points

- Swap `ChatOpenAI` for another `BaseChatModel` in the constructor.
- Replace SQLite with Postgres checkpointers/stores for multi-process deployments.
- Put approval nodes before tools that send, purchase, delete, or mutate external state.
- Add retrieval, browser, database, MCP, email, calendar, and code execution as registered tools.
- Add tracing through LangSmith or OpenTelemetry at the runtime boundary.
- Run one runtime per process; SQLite is intended for a single-process starter.

