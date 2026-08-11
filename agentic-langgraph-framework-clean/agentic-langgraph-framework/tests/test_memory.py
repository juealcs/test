from agentic_framework.memory import LongTermMemory


def test_memory_is_user_scoped(tmp_path):
    memory = LongTermMemory(tmp_path / "memory.sqlite")
    assert memory.save("alice", "Alice prefers concise answers")
    assert memory.search("alice", "concise") == ["Alice prefers concise answers"]
    assert memory.search("bob", "concise") == []
    memory.close()
