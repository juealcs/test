from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    thread_id: str
    run_id: str
    query: str
    force_plan: bool
    route: Literal["simple", "planned"]
    conversation_summary: str
    recent_chat: list[dict]
    recalled_memories: list[dict]
    plan: dict
    step_results: list[dict]
    tool_observations: list[dict]
    candidate_answer: str
    verification: dict
    replan_count: int
    final_answer: str
    status: str
