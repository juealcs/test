import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    mode: Literal["fast", "deliberate"]
    plan: list[str]
    goal: str
    memories: list[dict[str, Any]]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    failed_attempts: Annotated[list[str], operator.add]
    feedback: str
    verdict: Literal["pass", "retry", "replan"]
    iterations: int
    final_answer: str
    verified_facts: list[str]
    memories_written: int
