from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import StreamWriter

from .agents import PlannerAgent, SolverAgent, VerifierAgent
from .config import Settings
from .parsing import message_text
from .schemas import PlanStep, TraceEvent
from .services.chat_history import ChatHistoryStore
from .services.memory_manager import MemoryManager
from .services.observability import AuditLog
from .services.problem_analyzer import ProblemAnalyzer
from .state import AgentState


class WorkflowNodes:
    def __init__(
        self,
        settings: Settings,
        analyzer: ProblemAnalyzer,
        planner: PlannerAgent,
        solver: SolverAgent,
        verifier: VerifierAgent,
        memory: MemoryManager,
        chat: ChatHistoryStore,
        audit: AuditLog,
        memory_model,
    ):
        self.settings = settings
        self.analyzer = analyzer
        self.planner_agent = planner
        self.solver_agent = solver
        self.verifier_agent = verifier
        self.memory = memory
        self.chat = chat
        self.audit = audit
        self.memory_model = memory_model

    def emit(
        self,
        state: AgentState,
        writer: StreamWriter,
        stage: str,
        actor: str,
        kind: str,
        message: str,
        details: dict | None = None,
    ) -> dict:
        payload = self.audit.record(
            TraceEvent(
                run_id=state["run_id"],
                stage=stage,
                actor=actor,
                kind=kind,
                message=message,
                details=details or {},
            )
        )
        writer(payload)
        return payload

    @staticmethod
    def latest_user_text(state: AgentState) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return message_text(message.content)
        return ""

    def bootstrap(self, state: AgentState, writer: StreamWriter) -> dict:
        query = self.latest_user_text(state)
        update: dict = {
            "query": query,
            "route": "simple",
            "plan": {},
            "step_results": [],
            "tool_observations": [],
            "candidate_answer": "",
            "verification": {},
            "replan_count": 0,
            "final_answer": "",
            "status": "running",
        }
        messages = state.get("messages", [])
        if len(messages) > self.settings.recent_message_limit:
            update["messages"] = [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *messages[-self.settings.recent_message_limit :],
            ]
        self.emit(
            state,
            writer,
            "bootstrap",
            "system",
            "run_start",
            "Started durable graph run",
            {"user_id": state["user_id"], "thread_id": state["thread_id"]},
        )
        return update

    def recall(self, state: AgentState, writer: StreamWriter) -> dict:
        summary_record = self.chat.get_summary(state["user_id"], state["thread_id"])
        recent = self.chat.get_messages(
            state["user_id"], state["thread_id"], self.settings.recent_message_limit
        )
        memories = self.memory.recall(state["user_id"], state["query"])
        self.emit(
            state,
            writer,
            "memory_recall",
            "memory_service",
            "memory_read",
            f"Loaded {len(recent)} recent messages and {len(memories)} long-term memories",
            {
                "summary_present": bool(summary_record.get("summary")),
                "recalled_memories": [item["text"] for item in memories],
            },
        )
        return {
            "conversation_summary": summary_record.get("summary", ""),
            "recent_chat": recent,
            "recalled_memories": memories,
        }

    def analyze(self, state: AgentState, writer: StreamWriter) -> dict:
        route = self.analyzer.classify(state["query"], state.get("force_plan", False))
        self.emit(
            state,
            writer,
            "problem_analyzer",
            "problem_analyzer_service",
            "routing",
            "Simple request sent directly to Solver"
            if route == "simple"
            else "Complex request sent to Planner",
            {"route": route, "force_plan": state.get("force_plan", False)},
        )
        return {"route": route}

    def planner(self, state: AgentState, writer: StreamWriter) -> dict:
        self.emit(
            state,
            writer,
            "planner",
            "planner_agent",
            "agent_start",
            "Creating execution plan"
            if state.get("replan_count", 0) == 0
            else f"Replanning after verification failure {state['replan_count']}",
        )
        plan = self.planner_agent.create_plan(
            query=state["query"],
            summary=state.get("conversation_summary", ""),
            memories=state.get("recalled_memories", []),
            previous_results=state.get("step_results", []),
            verifier_feedback=state.get("verification", {}),
        )
        payload = plan.model_dump()
        self.emit(
            state,
            writer,
            "planner",
            "planner_agent",
            "plan_created",
            f"Created {len(plan.steps)} execution steps",
            payload,
        )
        return {
            "plan": payload,
            "step_results": [],
            "tool_observations": [],
            "candidate_answer": "",
        }

    def solver(self, state: AgentState, writer: StreamWriter) -> dict:
        if state.get("plan", {}).get("steps"):
            steps = [PlanStep.model_validate(item) for item in state["plan"]["steps"]]
        else:
            steps = [
                PlanStep(
                    id="direct",
                    description=state["query"],
                    expected_output="A direct and correct answer",
                    success_criteria=["Address the user's request"],
                )
            ]
        results: list[dict] = []
        observations: list[dict] = []

        def tool_emit(kind: str, message: str, details: dict) -> None:
            self.emit(state, writer, "tool_router", "tool_router_service", kind, message, details)

        for index, step in enumerate(steps, 1):
            self.emit(
                state,
                writer,
                "solver",
                "solver_agent",
                "step_start",
                f"Executing {step.id} ({index}/{len(steps)}): {step.description}",
                step.model_dump(),
            )
            result = self.solver_agent.solve_step(
                query=state["query"],
                step=step,
                summary=state.get("conversation_summary", ""),
                memories=state.get("recalled_memories", []),
                completed_results=results,
                emit=tool_emit,
            )
            result_payload = result.model_dump()
            results.append(result_payload)
            observations.extend(result.observations)
            self.emit(
                state,
                writer,
                "solver",
                "solver_agent",
                "step_result",
                f"Completed {step.id}",
                {
                    "tools_used": result.tools_used,
                    "output": result.output[:3000],
                },
            )
        if len(results) == 1:
            candidate = results[0]["output"]
        else:
            candidate = self.solver_agent.synthesize(
                state["query"], results, state.get("verification", {})
            )
        self.emit(
            state,
            writer,
            "solver",
            "solver_agent",
            "candidate_answer",
            "Produced candidate answer",
            {"answer": candidate[:4000]},
        )
        return {
            "step_results": results,
            "tool_observations": observations,
            "candidate_answer": candidate,
        }

    def verifier(self, state: AgentState, writer: StreamWriter) -> dict:
        self.emit(
            state,
            writer,
            "verifier",
            "verifier_agent",
            "agent_start",
            "Checking correctness, completeness, evidence, freshness, and safety",
        )
        verification = self.verifier_agent.verify(
            state["query"],
            state.get("plan", {}),
            state.get("step_results", []),
            state["candidate_answer"],
        )
        payload = verification.model_dump()
        self.emit(
            state,
            writer,
            "verifier",
            "verifier_agent",
            "verification",
            f"Verification {verification.status.upper()} (confidence {verification.confidence:.2f})",
            payload,
        )
        return {"verification": payload}

    def increment_replan(self, state: AgentState, writer: StreamWriter) -> dict:
        count = state.get("replan_count", 0) + 1
        self.emit(
            state,
            writer,
            "replan_gate",
            "system",
            "replan",
            f"Returning verifier feedback to Planner (replan {count}/{self.settings.max_replans})",
            state.get("verification", {}),
        )
        return {"replan_count": count}

    def final_response(self, state: AgentState, writer: StreamWriter) -> dict:
        answer = state.get("candidate_answer", "").strip()
        verification = state.get("verification", {})
        exhausted = verification.get("status") == "fail"
        if exhausted:
            answer += (
                "\n\nVerification note: The replan limit was reached. Remaining concerns: "
                + str(verification.get("feedback", "unknown"))
            )
        self.emit(
            state,
            writer,
            "final_response",
            "output_service",
            "final_start",
            "Streaming verified answer",
        )
        for start in range(0, len(answer), 48):
            writer(
                {
                    "run_id": state["run_id"],
                    "stage": "final_response",
                    "actor": "output_service",
                    "kind": "token",
                    "message": answer[start : start + 48],
                    "details": {},
                }
            )
        self.emit(
            state,
            writer,
            "final_response",
            "output_service",
            "final_end",
            "Final answer stream completed",
        )
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "status": "completed_with_warning" if exhausted else "completed",
        }

    def summarize(self, state: AgentState, writer: StreamWriter) -> dict:
        summary = self.memory.update_summary_if_needed(
            state["user_id"],
            state["thread_id"],
            state["final_answer"],
            self.memory_model,
        )
        self.emit(
            state,
            writer,
            "chat_summary",
            "memory_service",
            "summary_update",
            "Updated rolling conversation summary" if summary else "Rolling summary is current",
            {"updated": bool(summary), "summary": (summary or "")[:2000]},
        )
        return {"conversation_summary": summary or state.get("conversation_summary", "")}

    def update_memory(self, state: AgentState, writer: StreamWriter) -> dict:
        recent = self.chat.get_messages(state["user_id"], state["thread_id"], limit=1)
        source_message_id = recent[-1]["id"] if recent else None
        saved = self.memory.extract_and_save(
            state["user_id"],
            state["thread_id"],
            state["query"],
            source_message_id,
            self.memory_model,
        )
        self.emit(
            state,
            writer,
            "long_term_memory",
            "memory_service",
            "memory_write",
            f"Saved {len(saved)} durable memories",
            {"saved": saved, "checkpoint_persisted": True},
        )
        return {}

    def route_after_analyze(self, state: AgentState) -> str:
        return state["route"]

    def route_after_verify(self, state: AgentState) -> str:
        verification = state.get("verification", {})
        should_replan = (
            verification.get("status") == "fail"
            and verification.get("replan_required", True)
            and state.get("replan_count", 0) < self.settings.max_replans
        )
        return "replan" if should_replan else "final"


def build_graph(nodes: WorkflowNodes, checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("bootstrap", nodes.bootstrap)
    graph.add_node("recall", nodes.recall)
    graph.add_node("analyze", nodes.analyze)
    graph.add_node("planner", nodes.planner)
    graph.add_node("solver", nodes.solver)
    graph.add_node("verifier", nodes.verifier)
    graph.add_node("increment_replan", nodes.increment_replan)
    graph.add_node("final_response", nodes.final_response)
    graph.add_node("summarize", nodes.summarize)
    graph.add_node("update_memory", nodes.update_memory)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "recall")
    graph.add_edge("recall", "analyze")
    graph.add_conditional_edges(
        "analyze", nodes.route_after_analyze, {"simple": "solver", "planned": "planner"}
    )
    graph.add_edge("planner", "solver")
    graph.add_edge("solver", "verifier")
    graph.add_conditional_edges(
        "verifier",
        nodes.route_after_verify,
        {"replan": "increment_replan", "final": "final_response"},
    )
    graph.add_edge("increment_replan", "planner")
    graph.add_edge("final_response", "summarize")
    graph.add_edge("summarize", "update_memory")
    graph.add_edge("update_memory", END)
    return graph.compile(checkpointer=checkpointer)
