from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = "Qwen/Qwen3-8B"
    openai_api_key: str = "EMPTY"
    openai_base_url: str | None = "http://localhost:8000/v1"
    planner_model_name: str = ""
    solver_model_name: str = ""
    verifier_model_name: str = ""

    data_dir: Path = Path(".data")
    workspace_dir: Path = Path("workspace")
    max_plan_steps: int = Field(default=6, ge=1, le=20)
    max_tool_loops_per_step: int = Field(default=4, ge=1, le=12)
    max_replans: int = Field(default=2, ge=0, le=5)
    recent_message_limit: int = Field(default=12, ge=4, le=50)
    summary_trigger_messages: int = Field(default=12, ge=4, le=100)
    summary_update_interval: int = Field(default=6, ge=2, le=50)
    memory_recall_limit: int = Field(default=6, ge=1, le=20)

    enable_web_search: bool = True
    enable_url_fetch: bool = True
    enable_api_get: bool = True
    enable_file_write: bool = False
    enable_database_read: bool = True
    web_search_max_results: int = Field(default=5, ge=1, le=10)
    tool_timeout_seconds: float = Field(default=20, ge=2, le=120)

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "results").mkdir(parents=True, exist_ok=True)

    def model_for(self, role: str) -> str:
        configured = getattr(self, f"{role}_model_name", "")
        return configured or self.model_name
