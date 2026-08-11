import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class LongTermMemoryStore:
    """User-isolated durable memory with FTS5 retrieval and source provenance."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.executescript(
                """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                category TEXT NOT NULL, confidence REAL NOT NULL,
                source_thread TEXT NOT NULL, source_message_id INTEGER,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                last_used_at TEXT, UNIQUE(user_id,text));
                CREATE INDEX IF NOT EXISTS idx_memory_user ON memories(user_id,updated_at DESC);
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(text, content='memories', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid,text) VALUES(new.id,new.text); END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts,rowid,text)
                VALUES('delete',old.id,old.text); END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts,rowid,text)
                VALUES('delete',old.id,old.text);
                INSERT INTO memories_fts(rowid,text) VALUES(new.id,new.text); END;"""
            )

    def save(
        self,
        user_id: str,
        text: str,
        category: str,
        confidence: float,
        source_thread: str,
        source_message_id: int | None = None,
    ) -> bool:
        clean = " ".join(text.split()).strip()[:2000]
        if not user_id or not clean:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """INSERT INTO memories
                (user_id,text,category,confidence,source_thread,source_message_id,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(user_id,text) DO UPDATE SET
                category=excluded.category, confidence=MAX(memories.confidence,excluded.confidence),
                source_thread=excluded.source_thread, source_message_id=excluded.source_message_id,
                updated_at=excluded.updated_at""",
                (
                    user_id,
                    clean,
                    category[:50],
                    max(0, min(float(confidence), 1)),
                    source_thread,
                    source_message_id,
                    now,
                    now,
                ),
            )
        return cursor.rowcount > 0

    def search(self, user_id: str, query: str, limit: int = 6) -> list[dict]:
        terms = re.findall(r"[A-Za-z0-9_]{2,}", query)[:16]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            rows = self._conn.execute(
                """SELECT m.* FROM memories_fts f JOIN memories m ON m.id=f.rowid
                WHERE memories_fts MATCH ? AND m.user_id=?
                ORDER BY bm25(memories_fts),m.confidence DESC LIMIT ?""",
                (expression, user_id, limit),
            ).fetchall()
            if rows:
                self._conn.executemany(
                    "UPDATE memories SET last_used_at=? WHERE id=?",
                    [(now, row["id"]) for row in rows],
                )
        return [dict(row) for row in rows]

    def list(self, user_id: str, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
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
