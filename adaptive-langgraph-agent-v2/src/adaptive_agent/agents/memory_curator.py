from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..memory import LongTermMemory
from ..prompts import MEMORY
from ..state import AgentState
from .schemas import MemoryOutput


class MemoryCurator:
    """The only agent allowed to promote verified working state to long-term memory."""

    def __init__(self, model: BaseChatModel, memory: LongTermMemory):
        self.model = model.with_structured_output(MemoryOutput)
        self.memory = memory

    def run(self, state: AgentState, user_id: str) -> dict:
        prompt = (
            f"Task: {state['task']}\nVerified answer: {state['final_answer']}\n"
            f"Verified facts: {state.get('verified_facts', [])}"
        )
        result = self.model.invoke([SystemMessage(content=MEMORY), HumanMessage(content=prompt)])
        for item in result.memories:
            self.memory.put(user_id, item.kind, item.content, state["task"])
        return {"memories_written": len(result.memories)}

