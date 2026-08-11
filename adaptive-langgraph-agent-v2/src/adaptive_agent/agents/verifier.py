from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..prompts import VERIFY
from ..state import AgentState
from .schemas import VerificationOutput


class VerifierAgent:
    """Independently accepts, retries, or rejects the current strategy."""

    def __init__(self, model: BaseChatModel, max_iterations: int):
        self.model = model.with_structured_output(VerificationOutput)
        self.max_iterations = max_iterations

    def run(self, state: AgentState) -> dict:
        candidate = str(state["messages"][-1].content)
        tool_evidence = [
            str(message.content)
            for message in state["messages"]
            if getattr(message, "type", "") == "tool"
        ][-8:]
        prompt = (
            f"Original task: {state['task']}\nCandidate: {candidate}\n"
            f"Plan: {state.get('plan', [])}\nTool evidence: {tool_evidence}\n"
            f"Other evidence: {state.get('evidence', [])}"
        )
        result = self.model.invoke([SystemMessage(content=VERIFY), HumanMessage(content=prompt)])
        verdict, feedback = result.verdict, result.feedback
        if state.get("iterations", 0) >= self.max_iterations:
            verdict = "pass"
            feedback += " Iteration limit reached; this is the best available answer."
        failed = [feedback] if verdict != "pass" else []
        return {
            "verdict": verdict,
            "feedback": feedback,
            "verified_facts": result.verified_facts,
            "failed_attempts": failed,
            "final_answer": candidate,
        }
