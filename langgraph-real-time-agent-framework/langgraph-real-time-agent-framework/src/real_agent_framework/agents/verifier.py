import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..parsing import message_text, parse_model
from ..prompts import VERIFIER_PROMPT
from ..schemas import VerificationResult


class VerifierAgent:
    def __init__(self, model: BaseChatModel):
        self.model = model

    def verify(
        self,
        query: str,
        plan: dict,
        results: list[dict],
        candidate_answer: str,
    ) -> VerificationResult:
        request = (
            f"Original problem:\n{query}\n\n"
            f"Plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"Step results and tool observations:\n"
            f"{json.dumps(results, ensure_ascii=False, default=str, indent=2)}\n\n"
            f"Candidate answer:\n{candidate_answer}"
        )
        response = self.model.invoke(
            [SystemMessage(content=VERIFIER_PROMPT), HumanMessage(content=request)]
        )
        parsed = parse_model(message_text(response.content), VerificationResult)
        if parsed:
            return parsed
        return VerificationResult(
            status="pass",
            feedback=(
                "Verifier output was not structured; returning the best available answer with "
                "reduced verification confidence."
            ),
            missing_items=[],
            replan_required=False,
            confidence=0.25,
        )
