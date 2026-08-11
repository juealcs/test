__all__ = ["AgentRuntime", "RunResult"]


def __getattr__(name: str):
    if name in __all__:
        from .runtime import AgentRuntime, RunResult

        return {"AgentRuntime": AgentRuntime, "RunResult": RunResult}[name]
    raise AttributeError(name)
