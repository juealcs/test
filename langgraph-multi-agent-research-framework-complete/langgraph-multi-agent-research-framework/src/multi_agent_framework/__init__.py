__all__ = ["MultiAgentRuntime", "RunResult"]


def __getattr__(name: str):
    if name in __all__:
        from .runtime import MultiAgentRuntime, RunResult

        return {"MultiAgentRuntime": MultiAgentRuntime, "RunResult": RunResult}[name]
    raise AttributeError(name)
