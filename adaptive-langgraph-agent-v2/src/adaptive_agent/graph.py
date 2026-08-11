from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from typing import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from .agents import ExecutorAgent, FastAgent, MemoryCurator, PlannerAgent, VerifierAgent
from .config import Settings
from .context import AgentContext
from .memory import LongTermMemory
from .router import classify_complexity
from .state import AgentState
from .tools import default_tools


def _text(content: object) -> str:
    return content if isinstance(content, str) else str(content)


class AdaptiveAgent(AbstractContextManager):
    def __init__(self, settings: Settings | None = None, extra_tools: Sequence[BaseTool] = ()):
        self.settings = settings or Settings()
        self.settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory = LongTermMemory(self.settings.memory_db_path)
        self._connection = sqlite3.connect(self.settings.checkpoint_db_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(self._connection)
        self.model = init_chat_model(self.settings.model)
        self.planner_model = init_chat_model(self.settings.planner_model or self.settings.model)
        self.executor_model = init_chat_model(self.settings.executor_model or self.settings.model)
        self.verifier_model = init_chat_model(self.settings.verifier_model or self.settings.model)
        self.tools = [*default_tools(self.settings.workspace), *extra_tools]
        self.planner = PlannerAgent(self.planner_model)
        self.executor = ExecutorAgent(self.executor_model, self.tools)
        self.fast_agent = FastAgent(self.executor_model, self.tools)
        self.verifier = VerifierAgent(self.verifier_model, self.settings.max_iterations)
        self.memory_curator = MemoryCurator(self.verifier_model, self.memory)
        self.graph = self._compile()

    def _compile(self):
        def route(state: AgentState, runtime: Runtime[AgentContext]):
            task = state.get("task") or _text(state["messages"][-1].content)
            mode = classify_complexity(task, runtime.context.force_mode)
            return {"task": task, "mode": mode, "iterations": 0, "evidence": [], "failed_attempts": []}

        def route_mode(state: AgentState) -> str:
            return "fast" if state["mode"] == "fast" else "retrieve"

        def retrieve(state: AgentState, runtime: Runtime[AgentContext]):
            return {"memories": self.memory.search(runtime.context.user_id, state["task"])}

        def plan(state: AgentState):
            return self.planner.run(state)

        def fast(state: AgentState):
            return self.fast_agent.run(state)

        def execute(state: AgentState):
            return self.executor.run(state)

        def after_agent(state: AgentState) -> str:
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return "tools"
            return "final" if state["mode"] == "fast" else "verify"

        def after_tools(state: AgentState) -> str:
            return "fast" if state["mode"] == "fast" else "execute"

        def verify(state: AgentState):
            return self.verifier.run(state)

        def after_verify(state: AgentState) -> str:
            return {"pass": "remember", "retry": "execute", "replan": "plan"}[state["verdict"]]

        def remember(state: AgentState, runtime: Runtime[AgentContext]):
            return self.memory_curator.run(state, runtime.context.user_id)

        graph = StateGraph(AgentState, context_schema=AgentContext)
        graph.add_node("route", route)
        graph.add_node("retrieve", retrieve)
        graph.add_node("plan", plan)
        graph.add_node("fast", fast)
        graph.add_node("execute", execute)
        graph.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        graph.add_node("verify", verify)
        graph.add_node("remember", remember)
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", route_mode, {"fast": "fast", "retrieve": "retrieve"})
        graph.add_edge("retrieve", "plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges("fast", after_agent, {"tools": "tools", "final": END})
        graph.add_conditional_edges("execute", after_agent, {"tools": "tools", "verify": "verify"})
        graph.add_conditional_edges("tools", after_tools, {"fast": "fast", "execute": "execute"})
        graph.add_conditional_edges("verify", after_verify, {"remember": "remember", "execute": "execute", "plan": "plan"})
        graph.add_edge("remember", END)
        return graph.compile(checkpointer=self._checkpointer)

    def invoke(self, task: str, context: AgentContext, thread_id: str) -> AgentState:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        return self.graph.invoke({"task": task, "messages": [HumanMessage(content=task)]}, config=config, context=context)

    def close(self) -> None:
        self._connection.close()

    def __exit__(self, *args) -> None:
        self.close()


def build_agent(settings: Settings | None = None, extra_tools: Sequence[BaseTool] = ()) -> AdaptiveAgent:
    return AdaptiveAgent(settings=settings, extra_tools=extra_tools)
