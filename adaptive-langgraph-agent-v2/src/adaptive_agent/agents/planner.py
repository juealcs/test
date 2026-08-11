from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..prompts import PLAN
from ..state import AgentState
from .schemas import PlanOutput


class PlannerAgent:
    """Produces a small plan; it never executes tools or writes memory."""

    def __init__(self, model: BaseChatModel):
        self.model = model.with_structured_output(PlanOutput)

    def run(self, state: AgentState) -> dict:
        prompt = (
            f"Task: {state['task']}\n"
            f"Relevant long-term memory: {state.get('memories', [])}\n"
            f"Verified facts this run: {state.get('verified_facts', [])}\n"
            f"Failed attempts: {state.get('failed_attempts', [])}\n"
            f"Verifier feedback: {state.get('feedback', '')}"
        )
        result = self.model.invoke([SystemMessage(content=PLAN), HumanMessage(content=prompt)])
        return {"plan": result.steps, "goal": result.goal}

