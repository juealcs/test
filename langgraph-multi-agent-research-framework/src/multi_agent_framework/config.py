from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = "Qwen/Qwen3-8B"
    openai_api_key: str = "EMPTY"
    openai_base_url: str | None = "http://localhost:8000/v1"

    planner_model_name: str = ""
    researcher_model_name: str = ""
    analyst_model_name: str = ""
    writer_model_name: str = ""
    reviewer_model_name: str = ""

    data_dir: Path = Path(".data")
    workspace_dir: Path = Path("workspace")
    max_plan_tasks: int = Field(default=6, ge=1, le=20)
    max_tool_loops: int = Field(default=4, ge=1, le=12)
    max_revisions: int = Field(default=1, ge=0, le=5)
    memory_recall_limit: int = Field(default=6, ge=1, le=20)

    enable_web_search: bool = True
    enable_fetch_url: bool = True
    web_search_max_results: int = Field(default=5, ge=1, le=10)

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def model_for(self, role: str) -> str:
        value = getattr(self, f"{role}_model_name", "")
        return value or self.model_name
