import json
from pathlib import Path

from conftest import WorkflowTestModel

from real_agent_framework import AgentRuntime
from real_agent_framework.config import Settings


def settings_for(tmp_path, **overrides):
    values = {
        "data_dir": tmp_path / "data",
        "workspace_dir": tmp_path / "workspace",
        "enable_web_search": False,
        "enable_url_fetch": False,
        "enable_api_get": False,
        "enable_database_read": False,
        "summary_trigger_messages": 100,
    }
    values.update(overrides)
    return Settings(**values)


def test_full_plan_tool_verify_replan_and_result_file(tmp_path):
    model = WorkflowTestModel(use_tool=True, fail_first_verification=True)
    events: list[dict] = []
    with AgentRuntime(
        settings=settings_for(tmp_path, max_replans=1), models={"default": model}
    ) as runtime:
        result = runtime.run(
            "Calculate a result and verify it",
            user_id="alice",
            thread_id="calculation",
            force_plan=True,
            on_event=events.append,
        )
        history = runtime.chat.get_messages("alice", "calculation")

    assert result.answer == "The verified result is 4."
    assert result.replan_count == 1
    assert result.verification["status"] == "pass"
    assert [row["role"] for row in history] == ["user", "assistant"]
    kinds = [event["kind"] for event in events]
    assert "plan_created" in kinds
    assert "tool_start" in kinds
    assert "tool_result" in kinds
    assert kinds.count("verification") == 2
    document = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert document["tool_observations"][0]["tool"] == "calculator"
    assert document["status"] == "completed"


def test_session_checkpoint_summary_and_memory_survive_restart(tmp_path):
    settings = settings_for(
        tmp_path,
        summary_trigger_messages=4,
        summary_update_interval=2,
    )
    with AgentRuntime(settings=settings, models={"default": WorkflowTestModel()}) as first:
        first.run(
            "Remember that I prefer concise answers",
            user_id="alice",
            thread_id="durable",
        )
        first.run("Continue the test", user_id="alice", thread_id="durable")
        assert first.chat.get_summary("alice", "durable")["summary"]
        assert first.long_term_memory.list("alice")[0]["category"] == "preference"

    with AgentRuntime(settings=settings, models={"default": WorkflowTestModel()}) as restarted:
        history = restarted.chat.get_messages("alice", "durable")
        checkpoint = restarted.graph.get_state(restarted._config("alice", "durable")).values
        assert len(history) == 4
        assert checkpoint["final_answer"] == "The verified result is 4."
        assert restarted.delete_conversation("alice", "durable")
        assert restarted.chat.get_messages("alice", "durable") == []


def test_simple_query_bypasses_planner(tmp_path):
    events: list[dict] = []
    with AgentRuntime(
        settings=settings_for(tmp_path), models={"default": WorkflowTestModel()}
    ) as runtime:
        result = runtime.run("Hello", on_event=events.append)
    assert result.route == "simple"
    assert not result.plan
    assert all(event["actor"] != "planner_agent" for event in events)
