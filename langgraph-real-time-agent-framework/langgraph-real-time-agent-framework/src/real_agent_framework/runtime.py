import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver

from .agents import PlannerAgent, SolverAgent, VerifierAgent
from .config import Settings
from .graph import WorkflowNodes, build_graph
from .providers import ModelPool
from .services.chat_history import ChatHistoryStore
from .services.memory import LongTermMemoryStore
from .services.memory_manager import MemoryManager
from .services.observability import AuditLog
from .services.problem_analyzer import ProblemAnalyzer
from .services.tool_router import ToolRouter
from .tools import build_tools


@dataclass(frozen=True)
class RunResult:
    run_id: str
    user_id: str
    thread_id: str
    route: str
    answer: str
    status: str
    plan: dict
    step_results: list[dict]
    verification: dict
    replan_count: int
    result_path: str


class AgentRuntime:
    """Durable public runtime for CLI, tests, notebooks, or an API layer."""

    def __init__(
        self,
        settings: Settings | None = None,
        models: dict[str, BaseChatModel] | None = None,
        extra_tools: Sequence[BaseTool] = (),
    ):
        self.settings = settings or Settings()
        self.settings.prepare()
        self.chat = ChatHistoryStore(self.settings.data_dir / "chat_history.sqlite")
        self.long_term_memory = LongTermMemoryStore(
            self.settings.data_dir / "long_term_memory.sqlite"
        )
        self.audit = AuditLog(self.settings.data_dir / "audit.sqlite")
        self.model_pool = ModelPool(self.settings, models)
        all_tools = [*build_tools(self.settings), *extra_tools]
        self.tool_router = ToolRouter(all_tools)
        self.memory_manager = MemoryManager(
            long_term=self.long_term_memory,
            chat=self.chat,
            recall_limit=self.settings.memory_recall_limit,
            summary_trigger=self.settings.summary_trigger_messages,
            summary_interval=self.settings.summary_update_interval,
        )
        self._checkpoint_connection = sqlite3.connect(
            self.settings.data_dir / "langgraph_checkpoints.sqlite", check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.nodes = WorkflowNodes(
            settings=self.settings,
            analyzer=ProblemAnalyzer(),
            planner=PlannerAgent(self.model_pool.get("planner"), self.settings.max_plan_steps),
            solver=SolverAgent(
                self.model_pool.get("solver"),
                self.tool_router,
                self.settings.max_tool_loops_per_step,
            ),
            verifier=VerifierAgent(self.model_pool.get("verifier")),
            memory=self.memory_manager,
            chat=self.chat,
            audit=self.audit,
            memory_model=self.model_pool.get("solver"),
        )
        self.graph = build_graph(self.nodes, self.checkpointer)

    @staticmethod
    def checkpoint_thread_id(user_id: str, thread_id: str) -> str:
        return f"{user_id}:{thread_id}"

    def _config(self, user_id: str, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": self.checkpoint_thread_id(user_id, thread_id)},
            "recursion_limit": 12 + self.settings.max_replans * 4,
        }

    def _save_result(self, result: dict) -> Path:
        path = self.settings.data_dir / "results" / f"{result['run_id']}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def run(
        self,
        query: str,
        user_id: str = "default",
        thread_id: str = "default",
        force_plan: bool = False,
        on_event: Callable[[dict], None] | None = None,
    ) -> RunResult:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty")
        run_id = str(uuid4())
        self.chat.ensure_conversation(user_id, thread_id, clean_query[:80])
        user_message_id = self.chat.append_message(user_id, thread_id, "user", clean_query, run_id)
        self.audit.start_run(run_id, user_id, thread_id, clean_query)
        emitted: list[dict] = []
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            inputs = {
                "messages": [HumanMessage(content=clean_query)],
                "user_id": user_id,
                "thread_id": thread_id,
                "run_id": run_id,
                "force_plan": force_plan,
            }
            for event in self.graph.stream(
                inputs,
                config=self._config(user_id, thread_id),
                stream_mode="custom",
            ):
                emitted.append(event)
                if on_event:
                    on_event(event)
            state = self.graph.get_state(self._config(user_id, thread_id)).values
            assistant_message_id = self.chat.append_message(
                user_id,
                thread_id,
                "assistant",
                state.get("final_answer", ""),
                run_id,
                {"status": state.get("status", "completed")},
            )
            result_document = {
                "run_id": run_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "query": clean_query,
                "route": state.get("route", ""),
                "plan": state.get("plan", {}),
                "step_results": state.get("step_results", []),
                "tool_observations": state.get("tool_observations", []),
                "verification": state.get("verification", {}),
                "replan_count": state.get("replan_count", 0),
                "final_answer": state.get("final_answer", ""),
                "status": state.get("status", "completed"),
                "chat_message_ids": {
                    "user": user_message_id,
                    "assistant": assistant_message_id,
                },
                "models": {
                    role: self.settings.model_for(role)
                    for role in ("planner", "solver", "verifier")
                },
                "events": self.audit.get_events(run_id),
            }
            result_path = self._save_result(result_document)
            self.audit.finish_run(run_id, state.get("status", "completed"), str(result_path))
            return RunResult(
                run_id=run_id,
                user_id=user_id,
                thread_id=thread_id,
                route=state.get("route", ""),
                answer=state.get("final_answer", ""),
                status=state.get("status", "completed"),
                plan=state.get("plan", {}),
                step_results=state.get("step_results", []),
                verification=state.get("verification", {}),
                replan_count=state.get("replan_count", 0),
                result_path=str(result_path),
            )
        except Exception as exc:
            error_document = {
                "run_id": run_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "query": clean_query,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "events": emitted,
            }
            error_path = self._save_result(error_document)
            self.audit.finish_run(run_id, "failed", str(error_path), error_document["error"])
            raise

    def invoke(
        self,
        query: str,
        user_id: str = "default",
        thread_id: str = "default",
        force_plan: bool = False,
    ) -> str:
        return self.run(query, user_id, thread_id, force_plan).answer

    def delete_conversation(self, user_id: str, thread_id: str) -> bool:
        deleted = self.chat.delete(user_id, thread_id)
        if deleted:
            self.checkpointer.delete_thread(self.checkpoint_thread_id(user_id, thread_id))
        return deleted

    def graph_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def close(self) -> None:
        self.chat.close()
        self.long_term_memory.close()
        self.audit.close()
        self._checkpoint_connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
