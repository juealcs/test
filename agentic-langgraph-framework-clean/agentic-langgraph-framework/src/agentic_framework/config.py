from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = "gpt-4.1-mini"
    openai_api_key: str = ""
    openai_base_url: str | None = None
    data_dir: Path = Path(".data")
    workspace_dir: Path = Path("workspace")
    max_tool_loops: int = 4
    enable_http_tool: bool = False

    def prepare(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
