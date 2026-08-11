import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class LongTermMemory:
    """Small durable, user-scoped memory store with SQLite FTS5 search."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                created_at TEXT NOT NULL, UNIQUE(user_id, text))"""
            )
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content='memories', content_rowid='id')"
            )
            self._conn.executescript(
                """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text); END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text); END;"""
            )

    def save(self, user_id: str, text: str) -> bool:
        clean = " ".join(text.split()).strip()[:2000]
        if not user_id or not clean:
            return False
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO memories(user_id,text,created_at) VALUES(?,?,?)",
                (user_id, clean, datetime.now(timezone.utc).isoformat()),
            )
        return cursor.rowcount > 0

    def search(self, user_id: str, query: str, limit: int = 5) -> list[str]:
        terms = re.findall(r"[A-Za-z0-9_]{2,}", query)[:12]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        with self._lock:
            rows = self._conn.execute(
                """SELECT m.text FROM memories_fts f JOIN memories m ON m.id=f.rowid
                WHERE memories_fts MATCH ? AND m.user_id=? ORDER BY bm25(memories_fts) LIMIT ?""",
                (expression, user_id, limit),
            ).fetchall()
        return [row["text"] for row in rows]

    def close(self) -> None:
        self._conn.close()
