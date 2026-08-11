from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentic_framework import AgentRuntime
from agentic_framework.config import Settings


def test_graph_runs_deterministic_fast_lane(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", workspace_dir=tmp_path / "workspace")
    with AgentRuntime(settings=settings, model=FakeListChatModel(responses=["unused"])) as runtime:
        assert runtime.invoke("What is (81 / 9) + 4?", "alice", "math") == "13.0"
