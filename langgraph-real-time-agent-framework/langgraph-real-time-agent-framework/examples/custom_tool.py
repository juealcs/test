from langchain_core.tools import tool

from real_agent_framework import AgentRuntime


@tool
def paper_catalog(query: str) -> str:
    """Search a private paper catalog. Replace this example with a real integration."""
    return f"Internal catalog result for: {query}"


with AgentRuntime(extra_tools=[paper_catalog]) as runtime:
    result = runtime.run(
        "Find internal work on durable agent memory and explain its relevance.",
        user_id="researcher-1",
        thread_id="experiment-1",
        force_plan=True,
        on_event=lambda event: print(event["actor"], event["message"]),
    )
    print(result.answer)
