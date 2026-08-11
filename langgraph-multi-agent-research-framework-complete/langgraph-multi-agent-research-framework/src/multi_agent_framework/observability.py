import json
import sqlite3
import threading
from pathlib import Path

from .schemas import TraceEvent


class AuditLog:
    """Append-only event log for replaying and analysing agent runs."""

    def __init__(self, path: Path):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                node TEXT NOT NULL, agent TEXT NOT NULL, kind TEXT NOT NULL,
                message TEXT NOT NULL, details_json TEXT NOT NULL)"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id,id)")

    def record(self, event: TraceEvent) -> dict:
        payload = event.model_dump()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO events(run_id,timestamp,node,agent,kind,message,details_json)
                VALUES(?,?,?,?,?,?,?)""",
                (
                    event.run_id,
                    event.timestamp,
                    event.node,
                    event.agent,
                    event.kind,
                    event.message,
                    json.dumps(event.details, ensure_ascii=False, default=str),
                ),
            )
        return payload

    def get_run(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(row) | {"details": json.loads(row["details_json"])} for row in rows]

    def close(self) -> None:
        self._conn.close()
