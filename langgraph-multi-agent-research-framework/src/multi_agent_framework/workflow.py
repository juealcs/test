import json
import re
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .agents import SpecialistAgent
from .config import Settings
from .memory import LongTermMemory
from .models import ModelPool
from .observability import AuditLog
from .parsing import json_payload, message_text, parse_model
from .prompts import (
    FAST_PROMPT,
    FINALIZER_PROMPT,
    MEMORY_PROMPT,
    PLANNER_PROMPT,
    REVIEWER_PROMPT,
    SPECIALIST_PROMPTS,
)
from .routing import classify, extract_math
from .schemas import PlanTask, ResearchPlan, ReviewDecision, TaskResult, TraceEvent
from .state import MultiAgentState
from .tools import ToolRegistry, safe_calculate


class WorkflowNodes:
    def __init__(
        self,
        settings: Settings,
        models: ModelPool,
        memory: LongTermMemory,
        tools: ToolRegistry,
        audit: AuditLog,
    ):
        self.settings = settings
        self.models = models
        self.memory = memory
        self.tools = tools
        self.audit = audit

    def emit(
        self,
        state: MultiAgentState,
        node: str,
        agent: str,
        kind: str,
        message: str,
        details: dict | None = None,
    ) -> dict:
        return self.audit.record(
            TraceEvent(
                run_id=state["run_id"],
                node=node,
                agent=agent,
                kind=kind,
                message=message,
                details=details or {},
            )
        )

    @staticmethod
    def latest_user_text(state: MultiAgentState) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return message_text(message.content)
        return ""

    @staticmethod
    def conversation_context(state: MultiAgentState, limit: int = 8) -> str:
        lines: list[str] = []
        for message in state.get("messages", [])[-limit:]:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            lines.append(f"{role}: {message_text(message.content)[:2000]}")
        return "\n".join(lines)

    @staticmethod
    def memory_context(state: MultiAgentState) -> str:
        return "\n".join(f"- {item}" for item in state.get("memories", [])) or "(none)"

    def bootstrap(self, state: MultiAgentState) -> dict:
        query = self.latest_user_text(state)
        run_id = str(uuid4())
        initialized = {
            "query": query,
            "run_id": run_id,
            "plan": {},
            "task_results": [],
            "next_agent": "",
            "draft": "",
            "review": {},
            "revision_count": 0,
            "final_answer": "",
        }
        event_state = dict(state) | initialized
        event = self.emit(
            event_state,
            "bootstrap",
            "system",
            "lifecycle",
            "Started a new multi-agent run",
            {"query": query, "user_id": state["user_id"], "thread_id": state["thread_id"]},
        )
        return initialized | {"events": [event]}

    def recall(self, state: MultiAgentState) -> dict:
        rows = self.memory.search(
            state["user_id"], state["query"], self.settings.memory_recall_limit
        )
        memories = [row["text"] for row in rows]
        event = self.emit(
            state,
            "recall",
            "memory_manager",
            "memory_read",
            f"Recalled {len(memories)} long-term memories",
            {"memories": memories},
        )
        return {"memories": memories, "events": [event]}

    def router(self, state: MultiAgentState) -> dict:
        route = classify(state["query"], state.get("force_plan", False))
        explanation = (
            "Simple request selected for the low-latency path"
            if route == "fast"
            else "Research/action request selected for planner orchestration"
        )
        event = self.emit(
            state,
            "router",
            "router",
            "routing",
            explanation,
            {"route": route, "force_plan": state.get("force_plan", False)},
        )
        return {"route": route, "events": [event]}

    def fast(self, state: MultiAgentState) -> dict:
        expression = extract_math(state["query"])
        if expression:
            try:
                answer = str(safe_calculate(expression))
                method = "deterministic_calculator"
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
                answer = f"I could not evaluate that expression: {exc}"
                method = "calculator_error"
        else:
            response = self.models.get("analyst").invoke(
                [
                    SystemMessage(
                        content=f"{FAST_PROMPT}\n\nRelevant memories:\n{self.memory_context(state)}"
                    ),
                    HumanMessage(content=state["query"]),
                ]
            )
            answer = message_text(response.content)
            method = "single_model_call"
        event = self.emit(
            state,
            "fast",
            "fast_agent",
            "agent_output",
            "Produced a direct answer without a plan",
            {"method": method, "answer_preview": answer[:1000]},
        )
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "events": [event],
        }

    def _fallback_plan(self, query: str) -> ResearchPlan:
        needs_research = bool(
            re.search(
                r"\b(research|search|web|latest|current|source|study|evidence|literature)\b",
                query,
                re.IGNORECASE,
            )
        )
        tasks: list[PlanTask] = []
        if needs_research:
            tasks.append(
                PlanTask(
                    id="task-1",
                    agent="researcher",
                    instruction=f"Find reliable evidence needed to answer: {query}",
                    expected_output="Source-backed findings with URLs",
                )
            )
        analysis_id = f"task-{len(tasks) + 1}"
        tasks.append(
            PlanTask(
                id=analysis_id,
                agent="analyst",
                instruction=f"Analyze the request and available evidence: {query}",
                expected_output="Reasoned analysis with calculations when useful",
                depends_on=[tasks[-1].id] if needs_research else [],
            )
        )
        tasks.append(
            PlanTask(
                id=f"task-{len(tasks) + 1}",
                agent="writer",
                instruction="Synthesize collaborator findings into a complete draft answer",
                expected_output="A clear, well-supported draft",
                depends_on=[analysis_id],
            )
        )
        return ResearchPlan(
            objective=query,
            rationale="Fallback plan used because structured planner output was unavailable.",
            tasks=tasks,
        )

    def planner(self, state: MultiAgentState) -> dict:
        prompt = PLANNER_PROMPT.format(max_tasks=self.settings.max_plan_tasks)
        request = (
            f"User objective:\n{state['query']}\n\n"
            f"Relevant long-term memory:\n{self.memory_context(state)}\n\n"
            f"Recent conversation:\n{self.conversation_context(state)}"
        )
        response = self.models.get("planner").invoke(
            [SystemMessage(content=prompt), HumanMessage(content=request)]
        )
        raw = message_text(response.content)
        plan = parse_model(raw, ResearchPlan) or self._fallback_plan(state["query"])
        plan.tasks = plan.tasks[: self.settings.max_plan_tasks]
        if not plan.tasks:
            plan = self._fallback_plan(state["query"])
        payload = plan.model_dump()
        event = self.emit(
            state,
            "planner",
            "planner",
            "plan",
            f"Created a {len(plan.tasks)}-task execution plan",
            payload,
        )
        return {"plan": payload, "events": [event]}

    def supervisor(self, state: MultiAgentState) -> dict:
        tasks = state.get("plan", {}).get("tasks", [])
        cursor = len(state.get("task_results", []))
        if cursor >= len(tasks):
            next_agent = "reviewer"
            message = "All planned tasks completed; sending the shared draft to review"
            details = {"completed_tasks": cursor}
        else:
            task = tasks[cursor]
            next_agent = task["agent"]
            message = f"Delegated {task['id']} to the {next_agent} agent"
            details = {"task": task, "completed_tasks": cursor}
        event = self.emit(state, "supervisor", "supervisor", "delegation", message, details)
        return {"next_agent": next_agent, "events": [event]}

    def _run_specialist(self, state: MultiAgentState, role: str) -> dict:
        cursor = len(state.get("task_results", []))
        task = state["plan"]["tasks"][cursor]
        previous = json.dumps(state.get("task_results", []), ensure_ascii=False, indent=2)
        task_prompt = (
            f"Original user query:\n{state['query']}\n\n"
            f"Your assigned task:\n{task['instruction']}\n\n"
            f"Expected output:\n{task.get('expected_output', '')}\n\n"
            f"Relevant long-term memories:\n{self.memory_context(state)}\n\n"
            f"Shared results from earlier collaborators:\n{previous or '(none)'}"
        )

        def agent_emit(kind: str, message: str, details: dict) -> dict:
            return self.emit(state, role, role, kind, message, details)

        runner = SpecialistAgent(
            model=self.models.get(role),
            tools=self.tools.for_agent(role, state["user_id"], state["thread_id"]),
            max_tool_loops=self.settings.max_tool_loops,
        )
        output, tools_used, events = runner.run(SPECIALIST_PROMPTS[role], task_prompt, agent_emit)
        result = TaskResult(
            task_id=task["id"],
            agent=role,
            instruction=task["instruction"],
            output=output,
            tools_used=tools_used,
        ).model_dump()
        events.append(
            self.emit(
                state,
                role,
                role,
                "agent_output",
                f"Completed {task['id']}",
                {"tools_used": tools_used, "output_preview": output[:1600]},
            )
        )
        update = {"task_results": [*state.get("task_results", []), result], "events": events}
        if role == "writer":
            update["draft"] = output
        return update

    def researcher(self, state: MultiAgentState) -> dict:
        return self._run_specialist(state, "researcher")

    def analyst(self, state: MultiAgentState) -> dict:
        return self._run_specialist(state, "analyst")

    def writer(self, state: MultiAgentState) -> dict:
        return self._run_specialist(state, "writer")

    def reviewer(self, state: MultiAgentState) -> dict:
        draft = state.get("draft") or "\n\n".join(
            result["output"] for result in state.get("task_results", [])
        )
        request = (
            f"Original query:\n{state['query']}\n\n"
            f"Plan and specialist results:\n{json.dumps(state.get('task_results', []), indent=2)}\n\n"
            f"Draft to review:\n{draft}"
        )
        response = self.models.get("reviewer").invoke(
            [SystemMessage(content=REVIEWER_PROMPT), HumanMessage(content=request)]
        )
        raw = message_text(response.content)
        decision = parse_model(raw, ReviewDecision)
        if decision is None:
            decision = ReviewDecision(
                approved=True,
                feedback="Reviewer returned unstructured feedback; preserving the best available draft.",
            )
        event = self.emit(
            state,
            "reviewer",
            "reviewer",
            "review",
            "Draft approved" if decision.approved else "Draft requires revision",
            decision.model_dump(),
        )
        return {"review": decision.model_dump(), "draft": draft, "events": [event]}

    def reviser(self, state: MultiAgentState) -> dict:
        request = (
            f"Original query:\n{state['query']}\n\n"
            f"Current draft:\n{state.get('draft', '')}\n\n"
            f"Reviewer feedback:\n{json.dumps(state.get('review', {}), indent=2)}\n\n"
            "Revise the draft to address every material issue. Return only the revised draft."
        )
        response = self.models.get("writer").invoke(
            [SystemMessage(content=SPECIALIST_PROMPTS["writer"]), HumanMessage(content=request)]
        )
        draft = message_text(response.content)
        revision_count = state.get("revision_count", 0) + 1
        event = self.emit(
            state,
            "reviser",
            "writer",
            "revision",
            f"Completed revision {revision_count}",
            {"draft_preview": draft[:1600]},
        )
        return {"draft": draft, "revision_count": revision_count, "events": [event]}

    def finalizer(self, state: MultiAgentState) -> dict:
        request = (
            f"Original query:\n{state['query']}\n\n"
            f"Draft:\n{state.get('draft', '')}\n\n"
            f"Reviewer decision:\n{json.dumps(state.get('review', {}), indent=2)}"
        )
        response = self.models.get("writer").invoke(
            [SystemMessage(content=FINALIZER_PROMPT), HumanMessage(content=request)]
        )
        answer = message_text(response.content).strip() or state.get("draft", "")
        event = self.emit(
            state,
            "finalizer",
            "finalizer",
            "final_answer",
            "Produced the final answer",
            {"answer_preview": answer[:2000]},
        )
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "events": [event],
        }

    def memory_manager(self, state: MultiAgentState) -> dict:
        signal = re.search(
            r"\b(remember|i prefer|my preference|my name is|i am allergic|i live in|"
            r"my goal|my project|always answer|never answer)\b",
            state["query"],
            re.IGNORECASE,
        )
        saved: list[str] = []
        if signal:
            response = self.models.get("analyst").invoke(
                [
                    SystemMessage(content=MEMORY_PROMPT),
                    HumanMessage(content=state["query"]),
                ]
            )
            try:
                items = json_payload(message_text(response.content))
            except (json.JSONDecodeError, TypeError, ValueError):
                items = []
            if isinstance(items, list):
                for item in items[:5]:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        text = item["text"]
                        category = str(item.get("category", "profile"))
                        if self.memory.save(state["user_id"], text, category, state["thread_id"]):
                            saved.append(text)
        event = self.emit(
            state,
            "memory_manager",
            "memory_manager",
            "memory_write",
            f"Saved {len(saved)} durable memories",
            {"saved": saved, "short_term_checkpoint": True},
        )
        return {"events": [event]}

    def route_after_router(self, state: MultiAgentState) -> str:
        return state["route"]

    def route_after_supervisor(self, state: MultiAgentState) -> str:
        return state["next_agent"]

    def route_after_review(self, state: MultiAgentState) -> str:
        approved = bool(state.get("review", {}).get("approved", True))
        if not approved and state.get("revision_count", 0) < self.settings.max_revisions:
            return "revise"
        return "finalize"


def build_workflow(nodes: WorkflowNodes, checkpointer):
    graph = StateGraph(MultiAgentState)
    graph.add_node("bootstrap", nodes.bootstrap)
    graph.add_node("recall", nodes.recall)
    graph.add_node("router", nodes.router)
    graph.add_node("fast", nodes.fast)
    graph.add_node("planner", nodes.planner)
    graph.add_node("supervisor", nodes.supervisor)
    graph.add_node("researcher", nodes.researcher)
    graph.add_node("analyst", nodes.analyst)
    graph.add_node("writer", nodes.writer)
    graph.add_node("reviewer", nodes.reviewer)
    graph.add_node("reviser", nodes.reviser)
    graph.add_node("finalizer", nodes.finalizer)
    graph.add_node("memory_manager", nodes.memory_manager)

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "recall")
    graph.add_edge("recall", "router")
    graph.add_conditional_edges(
        "router", nodes.route_after_router, {"fast": "fast", "planner": "planner"}
    )
    graph.add_edge("fast", "memory_manager")
    graph.add_edge("planner", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        nodes.route_after_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "reviewer": "reviewer",
        },
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_conditional_edges(
        "reviewer", nodes.route_after_review, {"revise": "reviser", "finalize": "finalizer"}
    )
    graph.add_edge("reviser", "reviewer")
    graph.add_edge("finalizer", "memory_manager")
    graph.add_edge("memory_manager", END)
    return graph.compile(checkpointer=checkpointer)
