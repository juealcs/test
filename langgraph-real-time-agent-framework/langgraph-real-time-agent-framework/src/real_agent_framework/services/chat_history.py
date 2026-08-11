import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatHistoryStore:
    """Complete ChatGPT-style conversation archive, independent of model context limits."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.executescript(
                """CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, thread_id TEXT NOT NULL,
                title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}', UNIQUE(user_id,thread_id));
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations(user_id,updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, run_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id,id);

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id TEXT PRIMARY KEY, summary TEXT NOT NULL,
                summarized_message_count INTEGER NOT NULL, updated_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE);"""
            )

    def ensure_conversation(self, user_id: str, thread_id: str, title: str | None = None) -> dict:
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM conversations WHERE user_id=? AND thread_id=?",
                (user_id, thread_id),
            ).fetchone()
            if existing:
                return dict(existing)
            now = _now()
            conversation_id = str(uuid4())
            with self._conn:
                self._conn.execute(
                    """INSERT INTO conversations
                    (id,user_id,thread_id,title,created_at,updated_at)
                    VALUES(?,?,?,?,?,?)""",
                    (conversation_id, user_id, thread_id, title or "New conversation", now, now),
                )
            return dict(
                self._conn.execute(
                    "SELECT * FROM conversations WHERE id=?", (conversation_id,)
                ).fetchone()
            )

    def append_message(
        self,
        user_id: str,
        thread_id: str,
        role: str,
        content: str,
        run_id: str = "",
        metadata: dict | None = None,
    ) -> int:
        title = " ".join(content.split())[:80] if role == "user" else None
        conversation = self.ensure_conversation(user_id, thread_id, title)
        now = _now()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """INSERT INTO messages
                (conversation_id,role,content,run_id,created_at,metadata_json)
                VALUES(?,?,?,?,?,?)""",
                (
                    conversation["id"],
                    role,
                    content,
                    run_id,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                ),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation["id"])
            )
        return int(cursor.lastrowid)

    def get_messages(self, user_id: str, thread_id: str, limit: int | None = None) -> list[dict]:
        conversation = self.ensure_conversation(user_id, thread_id)
        with self._lock:
            if limit:
                rows = self._conn.execute(
                    """SELECT * FROM (SELECT * FROM messages WHERE conversation_id=?
                    ORDER BY id DESC LIMIT ?) ORDER BY id""",
                    (conversation["id"], limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
                    (conversation["id"],),
                ).fetchall()
        return [dict(row) | {"metadata": json.loads(row["metadata_json"])} for row in rows]

    def message_count(self, user_id: str, thread_id: str) -> int:
        conversation = self.ensure_conversation(user_id, thread_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE conversation_id=?",
                (conversation["id"],),
            ).fetchone()
        return int(row["count"])

    def get_summary(self, user_id: str, thread_id: str) -> dict:
        conversation = self.ensure_conversation(user_id, thread_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversation_summaries WHERE conversation_id=?",
                (conversation["id"],),
            ).fetchone()
        return dict(row) if row else {"summary": "", "summarized_message_count": 0}

    def save_summary(self, user_id: str, thread_id: str, summary: str, message_count: int) -> None:
        conversation = self.ensure_conversation(user_id, thread_id)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO conversation_summaries
                (conversation_id,summary,summarized_message_count,updated_at)
                VALUES(?,?,?,?) ON CONFLICT(conversation_id) DO UPDATE SET
                summary=excluded.summary,
                summarized_message_count=excluded.summarized_message_count,
                updated_at=excluded.updated_at""",
                (conversation["id"], summary, message_count, _now()),
            )

    def list_conversations(self, user_id: str, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT c.*, COUNT(m.id) AS message_count FROM conversations c
                LEFT JOIN messages m ON m.conversation_id=c.id WHERE c.user_id=?
                GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def rename(self, user_id: str, thread_id: str, title: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE conversations SET title=?,updated_at=? WHERE user_id=? AND thread_id=?",
                (title.strip()[:200], _now(), user_id, thread_id),
            )
        return cursor.rowcount > 0

    def delete(self, user_id: str, thread_id: str) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id FROM conversations WHERE user_id=? AND thread_id=?",
                (user_id, thread_id),
            ).fetchone()
            if not row:
                return False
            self._conn.execute("DELETE FROM messages WHERE conversation_id=?", (row["id"],))
            self._conn.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id=?", (row["id"],)
            )
            self._conn.execute("DELETE FROM conversations WHERE id=?", (row["id"],))
        return True

    def close(self) -> None:
        self._conn.close()
