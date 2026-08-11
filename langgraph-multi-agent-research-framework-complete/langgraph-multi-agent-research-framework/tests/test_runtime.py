from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from multi_agent_framework import MultiAgentRuntime
from multi_agent_framework.config import Settings


class WorkflowTestModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "workflow-test-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        system = str(messages[0].content)
        if "planning agent" in system:
            content = """{
              "objective":"Produce an analysis",
              "rationale":"Analyze then synthesize",
              "tasks":[
                {"id":"task-1","agent":"analyst","instruction":"Analyze the topic",
                 "expected_output":"Analysis","depends_on":[]},
                {"id":"task-2","agent":"writer","instruction":"Write the draft",
                 "expected_output":"Draft","depends_on":["task-1"]}
              ]} """
        elif "Analyst Agent" in system:
            content = "Analyst result based on the available evidence."
        elif "Writer Agent" in system:
            content = "Integrated draft from the collaborating agents."
        elif "Reviewer Agent" in system:
            content = '{"approved":true,"feedback":"Complete","missing_items":[]}'
        elif "Finalizer Agent" in system:
            content = "Final research answer."
        else:
            content = "Direct answer."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def test_complete_planner_supervisor_workflow(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
        enable_web_search=False,
        enable_fetch_url=False,
    )
    events: list[dict] = []
    with MultiAgentRuntime(settings=settings, models={"default": WorkflowTestModel()}) as runtime:
        result = runtime.run(
            "Analyze agent memory architecture",
            user_id="alice",
            thread_id="research",
            force_plan=True,
            on_event=events.append,
        )
        history = runtime.history("alice", "research")

    assert result.answer == "Final research answer."
    assert [item["agent"] for item in result.task_results] == ["analyst", "writer"]
    assert result.review["approved"] is True
    assert len(history) == 2
    agents = {event["agent"] for event in events}
    assert {"planner", "supervisor", "analyst", "writer", "reviewer", "finalizer"} <= agents


def test_fast_math_avoids_planner(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        workspace_dir=tmp_path / "workspace",
        enable_web_search=False,
        enable_fetch_url=False,
    )
    with MultiAgentRuntime(settings=settings, models={"default": WorkflowTestModel()}) as runtime:
        result = runtime.run("What is (81 / 9) + 4?", "alice", "math")
    assert result.answer == "13.0"
    assert result.route == "fast"
    assert result.plan == {}
