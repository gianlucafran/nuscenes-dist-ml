"""Scheduler: cola, Bulkhead, timeouts, backoff con jitter y reasignación.
Toda transición pasa por Store.cas_update; publicar a Kafka es un callback."""
import json
import logging
import random
import threading
import time

log = logging.getLogger("scheduler")

# Bulkhead: capacidad máxima por nodo
WORKERS = {
    "sergio": {"max_jobs": 2, "max_cnn": 1},
    "joshua": {"max_jobs": 1, "max_cnn": 1},
    "karina": {"max_jobs": 1, "max_cnn": 1},
    "patsy":  {"max_jobs": 1, "max_cnn": 1},
}
TIMEOUT_S = {"LINEAR_REGRESSION": 120, "MLP": 900, "CNN": 3600}
HEARTBEAT_TTL_S = 15
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 5


class Scheduler:
    def __init__(self, store, publish, clock=time.time, seed: int = 0):
        self.store = store
        self.publish = publish          # publish(worker: str, msg: dict)
        self.clock = clock
        self.rng = random.Random(seed)
        self.last_hb: dict[str, float] = {}
        self.down: set[str] = set()
        self.lock = threading.RLock()

    # ---- entradas desde Kafka ----

    def on_heartbeat(self, worker: str, job_id, progress: float) -> None:
        with self.lock:
            self.last_hb[worker] = self.clock()
            if worker in self.down:
                self.down.discard(worker)
                log.info("worker %s recuperado", worker)
            if job_id:
                self.store.update_progress(job_id, worker, float(progress))

    def on_result(self, msg: dict) -> None:
        with self.lock:
            job = self.store.get_job(msg["job_id"])
            stale = (job is None
                     or job["status"] != "RUNNING"
                     or job["assigned_worker"] != msg["worker"]
                     or job["attempt"] != msg["attempt"])
            if stale:
                log.info("resultado descartado (tardío o duplicado): %s", msg["job_id"])
                return
            if msg["ok"]:
                self._update(job, status="COMPLETED", progress=1.0, deadline_at=None,
                             metrics_json=json.dumps(msg["metrics"]),
                             model_artifact_path=msg["artifact"], error_message=None)
                self.store.event("JOB_COMPLETED", msg["worker"], job["job_id"],
                                 {"metrics": msg["metrics"]})
            else:
                self._retry(job, msg.get("error", "error desconocido"),
                            permanent=msg.get("permanent", False))

    # ---- ciclo periódico ----

    def tick(self) -> None:
        with self.lock:
            now = self.clock()
            self._detect_failures(now)
            self._dispatch(now)

    # ---- internos ----

    def _update(self, job: dict, **fields) -> bool:
        fields["updated_at"] = self.clock()
        ok = self.store.cas_update(job["job_id"], job["version"], fields)
        if not ok:
            log.warning("conflicto de versión en %s", job["job_id"])
        return ok

    def _retry(self, job: dict, error: str, permanent: bool) -> None:
        worker = job["assigned_worker"]
        if permanent or job["attempt"] >= MAX_ATTEMPTS:
            self._update(job, status="FAILED", error_message=error,
                         assigned_worker=None, deadline_at=None)
            self.store.event("JOB_FAILED", worker, job["job_id"], {"error": error})
            return
        now = self.clock()
        delay = BACKOFF_BASE_S * (2 ** job["attempt"]) + self.rng.uniform(0, BACKOFF_BASE_S)
        self._update(job, status="REASSIGNED", error_message=error,
                     assigned_worker=None, deadline_at=None, progress=0.0,
                     next_attempt_at=now + delay)
        self.store.event("JOB_REASSIGNED", worker, job["job_id"],
                         {"error": error, "retry_in_s": round(delay, 1)})

    def _detect_failures(self, now: float) -> None:
        for job in self.store.running_jobs():
            worker = job["assigned_worker"]
            if job["deadline_at"] and now > job["deadline_at"]:
                self._retry(job, "timeout", permanent=False)
            elif now - self.last_hb.get(worker, 0.0) > HEARTBEAT_TTL_S:
                if worker not in self.down:
                    self.down.add(worker)
                    self.store.event("NODE_DOWN", worker, None, {})
                self._retry(job, f"sin heartbeat de {worker}", permanent=False)

    def _pick_worker(self, model_type: str, now: float):
        busy = self.store.running_counts()
        for w, cap in sorted(WORKERS.items()):
            if now - self.last_hb.get(w, -1e9) > HEARTBEAT_TTL_S:
                continue
            c = busy.get(w, {"total": 0, "CNN": 0})
            if c["total"] >= cap["max_jobs"]:
                continue
            if model_type == "CNN" and c["CNN"] >= cap["max_cnn"]:
                continue
            return w
        return None

    def _dispatch(self, now: float) -> None:
        for job in self.store.due_jobs(now):
            worker = self._pick_worker(job["model_type"], now)
            if worker is None:
                continue
            attempt = job["attempt"] + 1
            ok = self._update(job, status="RUNNING", assigned_worker=worker,
                              attempt=attempt, progress=0.0,
                              deadline_at=now + TIMEOUT_S[job["model_type"]])
            if not ok:
                continue
            self.publish(worker, {
                "job_id": job["job_id"],
                "attempt": attempt,
                "model_type": job["model_type"],
                "config": json.loads(job["config_json"]),
                "primary_metric": job["primary_metric"],
            })
            self.store.event("JOB_ASSIGNED", worker, job["job_id"], {"attempt": attempt})
