from real_agent_framework.services.chat_history import ChatHistoryStore
from real_agent_framework.services.memory import LongTermMemoryStore


def test_chat_history_survives_restart(tmp_path):
    path = tmp_path / "chat.sqlite"
    first = ChatHistoryStore(path)
    first.append_message("alice", "thread-1", "user", "Hello", "run-1")
    first.append_message("alice", "thread-1", "assistant", "Hi", "run-1")
    first.save_summary("alice", "thread-1", "A greeting occurred.", 2)
    first.close()

    second = ChatHistoryStore(path)
    assert [row["content"] for row in second.get_messages("alice", "thread-1")] == [
        "Hello",
        "Hi",
    ]
    assert second.get_summary("alice", "thread-1")["summary"] == "A greeting occurred."
    assert second.list_conversations("alice")[0]["message_count"] == 2
    second.close()


def test_long_term_memory_is_user_isolated(tmp_path):
    memory = LongTermMemoryStore(tmp_path / "memory.sqlite")
    memory.save("alice", "Alice prefers concise answers", "preference", 1.0, "thread-1")
    assert memory.search("alice", "concise")[0]["text"] == "Alice prefers concise answers"
    assert memory.search("bob", "concise") == []
    memory_id = memory.list("alice")[0]["id"]
    assert memory.delete("alice", memory_id)
    memory.close()
