from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4


class LongTermMemory:
    """Cross-thread memory. This is intentionally independent of checkpoints."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL,
                    content TEXT NOT NULL, source_task TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON long_term_memories(user_id, created_at)")

    def put(self, user_id: str, kind: str, content: str, source_task: str, metadata: dict[str, Any] | None = None) -> None:
        if not content.strip():
            return
        with self._connect() as db:
            db.execute(
                "INSERT INTO long_term_memories(id,user_id,kind,content,source_task,metadata) VALUES(?,?,?,?,?,?)",
                (str(uuid4()), user_id, kind, content.strip(), source_task, json.dumps(metadata or {})),
            )

    def search(self, user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = [t.lower() for t in re_words(query) if len(t) > 2][:12]
        if not terms:
            return []
        score = " + ".join(["CASE WHEN lower(content) LIKE ? THEN 1 ELSE 0 END"] * len(terms))
        params: list[Any] = [f"%{term}%" for term in terms] + [user_id, limit]
        sql = f"""SELECT kind,content,source_task,created_at,({score}) AS relevance
                  FROM long_term_memories WHERE user_id=? ORDER BY relevance DESC,created_at DESC LIMIT ?"""
        with self._connect() as db:
            return [dict(row) for row in db.execute(sql, params) if row["relevance"] > 0]


def re_words(value: str) -> list[str]:
    import re
    return re.findall(r"[\w'-]+", value)

