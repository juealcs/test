from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from ..prompts import EXECUTE, FAST
from ..state import AgentState


class ExecutorAgent:
    """Works through the plan and decides which registered tools to call."""

    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.model = model.bind_tools(tools)

    def run(self, state: AgentState) -> dict:
        working_context = (
            f"Original task: {state['task']}\n"
            f"Goal: {state.get('goal', state['task'])}\n"
            f"Plan: {state.get('plan', [])}\n"
            f"Relevant memory: {state.get('memories', [])}\n"
            f"Verified facts: {state.get('verified_facts', [])}\n"
            f"Failed attempts: {state.get('failed_attempts', [])}\n"
            f"Verifier feedback: {state.get('feedback', '')}"
        )
        response = self.model.invoke(
            [SystemMessage(content=EXECUTE), HumanMessage(content=working_context), *state["messages"]]
        )
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}


class FastAgent:
    """Direct model/tool loop used when planning would be wasteful."""

    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.model = model.bind_tools(tools)

    def run(self, state: AgentState) -> dict:
        response = self.model.invoke([SystemMessage(content=FAST), *state["messages"]])
        update = {"messages": [response]}
        if not response.tool_calls:
            update["final_answer"] = str(response.content)
        return update

