from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class MultiAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    user_id: str
    thread_id: str
    run_id: str
    force_plan: bool
    route: Literal["fast", "planner"]
    memories: list[str]
    plan: dict
    task_results: list[dict]
    next_agent: str
    draft: str
    review: dict
    revision_count: int
    final_answer: str
    events: list[dict]
