from dataclasses import dataclass


@dataclass(frozen=True)
class AgentContext:
    """Immutable request identity/configuration, separate from graph state."""

    user_id: str
    force_mode: str | None = None

