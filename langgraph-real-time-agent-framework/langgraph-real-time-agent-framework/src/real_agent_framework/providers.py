from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings


class ModelPool:
    """Shared and role-specific models through OpenAI-compatible APIs."""

    def __init__(self, settings: Settings, injected: dict[str, BaseChatModel] | None = None):
        self.settings = settings
        self.injected = injected or {}
        self._cache: dict[str, BaseChatModel] = {}

    def get(self, role: str) -> BaseChatModel:
        if role in self.injected:
            return self.injected[role]
        if "default" in self.injected:
            return self.injected["default"]
        model_name = self.settings.model_for(role)
        if model_name not in self._cache:
            self._cache[model_name] = ChatOpenAI(
                model=model_name,
                api_key=self.settings.openai_api_key or "EMPTY",
                base_url=self.settings.openai_base_url or None,
                temperature=0,
                max_retries=2,
                timeout=self.settings.tool_timeout_seconds * 3,
            )
        return self._cache[model_name]
