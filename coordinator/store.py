"""Persistencia SQLite: idempotencia (INSERT OR IGNORE), transiciones con
control optimista y bitácora de eventos."""
import json
import sqlite3
import threading
import time
from pathlib import Path
#hola soy chris
SCHEMA = Path(__file__).with_name("schema.sql")


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False,
                                  isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA.read_text())

    def _q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def _x(self, sql, args=()) -> int:
        with self.lock:
            return self.db.execute(sql, args).rowcount

    def insert_job(self, job: dict) -> bool:
        cols = ",".join(job)
        qs = ",".join("?" * len(job))
        return self._x(f"INSERT OR IGNORE INTO jobs ({cols}) VALUES ({qs})",
                       tuple(job.values())) == 1

    def get_job(self, job_id: str):
        rows = self._q("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        return dict(rows[0]) if rows else None

    def cas_update(self, job_id: str, expected_version: int, fields: dict) -> bool:
        sets = ",".join(f"{k}=?" for k in fields)
        sql = f"UPDATE jobs SET {sets}, version=version+1 WHERE job_id=? AND version=?"
        return self._x(sql, (*fields.values(), job_id, expected_version)) == 1

    def update_progress(self, job_id: str, worker: str, progress: float) -> None:
        self._x("UPDATE jobs SET progress=? WHERE job_id=? AND status='RUNNING' "
                "AND assigned_worker=?", (progress, job_id, worker))

    def running_jobs(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM jobs WHERE status='RUNNING'")]

    def due_jobs(self, now: float) -> list[dict]:
        rows = self._q("SELECT * FROM jobs WHERE status IN ('PENDING','REASSIGNED') "
                       "AND next_attempt_at<=? ORDER BY next_attempt_at LIMIT 50", (now,))
        return [dict(r) for r in rows]

    def running_counts(self) -> dict:
        out = {}
        rows = self._q("SELECT assigned_worker AS w, model_type AS m, COUNT(*) AS n "
                       "FROM jobs WHERE status='RUNNING' GROUP BY w, m")
        for r in rows:
            d = out.setdefault(r["w"], {"total": 0, "CNN": 0})
            d["total"] += r["n"]
            if r["m"] == "CNN":
                d["CNN"] += r["n"]
        return out

    def event(self, etype: str, worker, job_id, detail=None) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO job_events (ts,event_type,worker_id,job_id,detail_json) "
                "VALUES (?,?,?,?,?)",
                (time.time(), etype, worker, job_id, json.dumps(detail or {})))
            return cur.lastrowid

    def events_since(self, event_id: int, limit: int = 200) -> list[dict]:
        rows = self._q("SELECT * FROM job_events WHERE event_id > ? "
                       "ORDER BY event_id LIMIT ?", (event_id, limit))
        return [dict(r) for r in rows]
