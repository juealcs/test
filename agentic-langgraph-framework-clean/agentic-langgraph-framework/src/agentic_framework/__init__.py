__all__ = ["AgentRuntime"]


def __getattr__(name: str):
    # Keep lightweight modules importable even before optional runtime dependencies are installed.
    if name == "AgentRuntime":
        from .runtime import AgentRuntime

        return AgentRuntime
    raise AttributeError(name)
