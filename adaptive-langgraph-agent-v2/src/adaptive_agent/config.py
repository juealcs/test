from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("AGENT_MODEL", "openai:gpt-4.1-mini")
    planner_model: str | None = os.getenv("PLANNER_MODEL") or None
    executor_model: str | None = os.getenv("EXECUTOR_MODEL") or None
    verifier_model: str | None = os.getenv("VERIFIER_MODEL") or None
    checkpoint_db_path: Path = Path(os.getenv("AGENT_CHECKPOINT_DB", ".data/checkpoints.sqlite"))
    memory_db_path: Path = Path(os.getenv("AGENT_MEMORY_DB", ".data/long_term_memory.sqlite"))
    workspace: Path = Path(os.getenv("AGENT_WORKSPACE", ".")).resolve()
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "3"))
