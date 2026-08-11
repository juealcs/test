import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import Settings
from .memory import LongTermMemory
from .models import ModelPool
from .observability import AuditLog
from .tools import ToolRegistry
from .workflow import WorkflowNodes, build_workflow


@dataclass(frozen=True)
class RunResult:
    answer: str
    run_id: str
    route: str
    plan: dict
    task_results: list[dict]
    review: dict


class MultiAgentRuntime:
    """Public facade around the compiled LangGraph and its durable resources."""

    def __init__(
        self,
        settings: Settings | None = None,
        models: dict[str, BaseChatModel] | None = None,
        extra_tools: Sequence[tuple[Sequence[str], BaseTool]] = (),
    ):
        self.settings = settings or Settings()
        self.settings.prepare()
        self.memory = LongTermMemory(self.settings.data_dir / "long_term_memory.sqlite")
        self.audit = AuditLog(self.settings.data_dir / "audit.sqlite")
        self.models = ModelPool(self.settings, models)
        self.tools = ToolRegistry(
            workspace=self.settings.workspace_dir,
            memory=self.memory,
            enable_web_search=self.settings.enable_web_search,
            enable_fetch_url=self.settings.enable_fetch_url,
            web_search_max_results=self.settings.web_search_max_results,
        )
        for roles, new_tool in extra_tools:
            self.tools.add_tool(roles, new_tool)
        self._checkpoint_conn = sqlite3.connect(
            self.settings.data_dir / "short_term_checkpoints.sqlite",
            check_same_thread=False,
        )
        self.checkpointer = SqliteSaver(self._checkpoint_conn)
        self.nodes = WorkflowNodes(self.settings, self.models, self.memory, self.tools, self.audit)
        self.graph = build_workflow(self.nodes, self.checkpointer)

    @staticmethod
    def _config(user_id: str, thread_id: str) -> dict:
        return {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}

    def run(
        self,
        query: str,
        user_id: str = "default",
        thread_id: str = "default",
        force_plan: bool = False,
        on_event: Callable[[dict], None] | None = None,
    ) -> RunResult:
        config = self._config(user_id, thread_id)
        input_state = {
            "messages": [HumanMessage(content=query)],
            "user_id": user_id,
            "thread_id": thread_id,
            "force_plan": force_plan,
        }
        for update in self.graph.stream(input_state, config=config, stream_mode="updates"):
            for payload in update.values():
                if isinstance(payload, dict):
                    for event in payload.get("events", []):
                        if on_event:
                            on_event(event)
        state = self.graph.get_state(config).values
        return RunResult(
            answer=state.get("final_answer", ""),
            run_id=state.get("run_id", ""),
            route=state.get("route", ""),
            plan=state.get("plan", {}),
            task_results=state.get("task_results", []),
            review=state.get("review", {}),
        )

    def invoke(
        self,
        query: str,
        user_id: str = "default",
        thread_id: str = "default",
        force_plan: bool = False,
    ) -> str:
        return self.run(query, user_id, thread_id, force_plan).answer

    def history(self, user_id: str, thread_id: str) -> list[dict]:
        state = self.graph.get_state(self._config(user_id, thread_id)).values
        history: list[dict] = []
        for message in state.get("messages", []):
            history.append(
                {
                    "type": getattr(message, "type", type(message).__name__),
                    "content": message.content,
                }
            )
        return history

    def graph_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def close(self) -> None:
        self.memory.close()
        self.audit.close()
        self._checkpoint_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
