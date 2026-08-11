from multi_agent_framework.memory import LongTermMemory


def test_long_term_memory_is_user_scoped(tmp_path):
    memory = LongTermMemory(tmp_path / "memory.sqlite")
    memory.save("alice", "Alice prefers concise research summaries", "preference", "t1")
    assert memory.search("alice", "concise")[0]["category"] == "preference"
    assert memory.search("bob", "concise") == []
    rows = memory.list("alice")
    assert len(rows) == 1
    assert memory.delete("alice", rows[0]["id"])
    memory.close()
