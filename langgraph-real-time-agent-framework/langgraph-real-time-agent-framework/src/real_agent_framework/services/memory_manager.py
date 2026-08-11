import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..parsing import message_text, parse_json
from ..prompts import MEMORY_PROMPT, SUMMARY_PROMPT
from .chat_history import ChatHistoryStore
from .memory import LongTermMemoryStore


class MemoryManager:
    def __init__(
        self,
        long_term: LongTermMemoryStore,
        chat: ChatHistoryStore,
        recall_limit: int,
        summary_trigger: int,
        summary_interval: int,
    ):
        self.long_term = long_term
        self.chat = chat
        self.recall_limit = recall_limit
        self.summary_trigger = summary_trigger
        self.summary_interval = summary_interval

    def recall(self, user_id: str, query: str) -> list[dict]:
        return self.long_term.search(user_id, query, self.recall_limit)

    def extract_and_save(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        source_message_id: int | None,
        model: BaseChatModel,
    ) -> list[dict]:
        should_extract = re.search(
            r"\b(remember|i prefer|my preference|my name is|i am allergic|i live in|"
            r"my goal|my project|always answer|never answer|my constraint)\b",
            query,
            re.IGNORECASE,
        )
        if not should_extract:
            return []
        response = model.invoke([SystemMessage(content=MEMORY_PROMPT), HumanMessage(content=query)])
        try:
            items = parse_json(message_text(response.content))
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
        saved: list[dict] = []
        if not isinstance(items, list):
            return saved
        for item in items[:5]:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            record = {
                "text": item["text"],
                "category": str(item.get("category", "profile")),
                "confidence": float(item.get("confidence", 1.0)),
            }
            if self.long_term.save(
                user_id=user_id,
                text=record["text"],
                category=record["category"],
                confidence=record["confidence"],
                source_thread=thread_id,
                source_message_id=source_message_id,
            ):
                saved.append(record)
        return saved

    def update_summary_if_needed(
        self,
        user_id: str,
        thread_id: str,
        final_answer: str,
        model: BaseChatModel,
    ) -> str | None:
        stored_count = self.chat.message_count(user_id, thread_id)
        effective_count = stored_count + 1  # Final answer is appended immediately after the graph.
        existing = self.chat.get_summary(user_id, thread_id)
        summarized = int(existing.get("summarized_message_count", 0))
        if effective_count < self.summary_trigger:
            return None
        if effective_count - summarized < self.summary_interval:
            return None
        recent = self.chat.get_messages(
            user_id, thread_id, limit=max(self.summary_interval * 2, 12)
        )
        transcript = "\n".join(f"{row['role']}: {row['content']}" for row in recent)
        transcript += f"\nassistant: {final_answer}"
        request = (
            f"Previous summary:\n{existing.get('summary') or '(none)'}\n\n"
            f"Recent conversation:\n{transcript}"
        )
        response = model.invoke(
            [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=request)]
        )
        summary = message_text(response.content).strip()
        if summary:
            self.chat.save_summary(user_id, thread_id, summary, effective_count)
            return summary
        return None
