from langchain_core.tools import tool

from multi_agent_framework import MultiAgentRuntime


@tool
def internal_paper_search(query: str) -> str:
    """Search an organization's internal paper index."""
    return f"Replace this function with a real paper index lookup for: {query}"


with MultiAgentRuntime(extra_tools=[(("researcher",), internal_paper_search)]) as runtime:
    result = runtime.run(
        "Find relevant internal work on long-term agent memory.",
        user_id="researcher-1",
        thread_id="experiment-1",
        force_plan=True,
        on_event=lambda event: print(event["agent"], event["message"]),
    )
    print(result.answer)
