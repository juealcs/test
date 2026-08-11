import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class LongTermMemory:
    """Durable, user-scoped memory with lexical SQLite FTS5 retrieval."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'profile',
                source_thread TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, text)
                )"""
            )
            self._conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(text, content='memories', content_rowid='id')"""
            )
            self._conn.executescript(
                """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text)
                VALUES('delete', old.id, old.text); END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text)
                VALUES('delete', old.id, old.text);
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;"""
            )

    def save(
        self,
        user_id: str,
        text: str,
        category: str = "profile",
        source_thread: str = "",
    ) -> bool:
        clean = " ".join(text.split()).strip()[:2000]
        if not user_id or not clean:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """INSERT INTO memories(user_id,text,category,source_thread,created_at,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(user_id,text) DO UPDATE SET
                category=excluded.category, source_thread=excluded.source_thread,
                updated_at=excluded.updated_at""",
                (user_id, clean, category[:50], source_thread[:200], now, now),
            )
        return cursor.rowcount > 0

    def search(self, user_id: str, query: str, limit: int = 6) -> list[dict]:
        terms = re.findall(r"[A-Za-z0-9_]{2,}", query)[:16]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        with self._lock:
            rows = self._conn.execute(
                """SELECT m.text,m.category,m.source_thread,m.updated_at
                FROM memories_fts f JOIN memories m ON m.id=f.rowid
                WHERE memories_fts MATCH ? AND m.user_id=?
                ORDER BY bm25(memories_fts) LIMIT ?""",
                (expression, user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def list(self, user_id: str, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT id,text,category,source_thread,created_at,updated_at FROM memories
                WHERE user_id=? ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, user_id: str, memory_id: int) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE user_id=? AND id=?", (user_id, memory_id)
            )
        return cursor.rowcount > 0

    def close(self) -> None:
        self._conn.close()
