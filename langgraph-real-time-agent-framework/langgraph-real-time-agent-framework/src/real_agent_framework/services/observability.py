import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import TraceEvent


class AuditLog:
    def __init__(self, path: Path):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._conn:
            self._conn.executescript(
                """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,thread_id TEXT NOT NULL,
                query TEXT NOT NULL,status TEXT NOT NULL,started_at TEXT NOT NULL,
                finished_at TEXT,result_path TEXT,error TEXT);
                CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs(user_id,thread_id,started_at DESC);
                CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,run_id TEXT NOT NULL,timestamp TEXT NOT NULL,
                stage TEXT NOT NULL,actor TEXT NOT NULL,kind TEXT NOT NULL,
                message TEXT NOT NULL,details_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id,id);"""
            )

    def start_run(self, run_id: str, user_id: str, thread_id: str, query: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs(run_id,user_id,thread_id,query,status,started_at) VALUES(?,?,?,?,?,?)",
                (
                    run_id,
                    user_id,
                    thread_id,
                    query,
                    "running",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        result_path: str = "",
        error: str = "",
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE runs SET status=?,finished_at=?,result_path=?,error=? WHERE run_id=?""",
                (
                    status,
                    datetime.now(timezone.utc).isoformat(),
                    result_path,
                    error,
                    run_id,
                ),
            )

    def record(self, event: TraceEvent) -> dict:
        payload = event.model_dump()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO events
                (run_id,timestamp,stage,actor,kind,message,details_json)
                VALUES(?,?,?,?,?,?,?)""",
                (
                    event.run_id,
                    event.timestamp,
                    event.stage,
                    event.actor,
                    event.kind,
                    event.message,
                    json.dumps(event.details, ensure_ascii=False, default=str),
                ),
            )
        return payload

    def get_events(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(row) | {"details": json.loads(row["details_json"])} for row in rows]

    def list_runs(self, user_id: str, thread_id: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock:
            if thread_id:
                rows = self._conn.execute(
                    """SELECT * FROM runs WHERE user_id=? AND thread_id=?
                    ORDER BY started_at DESC LIMIT ?""",
                    (user_id, thread_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM runs WHERE user_id=? ORDER BY started_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()
