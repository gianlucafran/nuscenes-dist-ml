PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS jobs (
  job_id              TEXT PRIMARY KEY,
  model_type          TEXT NOT NULL CHECK (model_type IN ('LINEAR_REGRESSION','MLP','CNN')),
  config_json         TEXT NOT NULL,
  dataset_version     TEXT NOT NULL,
  split_version       TEXT NOT NULL,
  target_variable     TEXT NOT NULL,
  primary_metric      TEXT NOT NULL,
  seed                INTEGER NOT NULL,
  status              TEXT NOT NULL CHECK (status IN
                        ('PENDING','RUNNING','COMPLETED','FAILED','REASSIGNED','CANCELLED')),
  assigned_worker     TEXT,
  attempt             INTEGER NOT NULL DEFAULT 0,
  next_attempt_at     REAL NOT NULL DEFAULT 0,
  deadline_at         REAL,
  progress            REAL NOT NULL DEFAULT 0,
  metrics_json        TEXT,
  model_artifact_path TEXT,
  error_message       TEXT,
  created_at          REAL NOT NULL,
  updated_at          REAL NOT NULL,
  version             INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_due ON jobs(status, next_attempt_at);

CREATE TABLE IF NOT EXISTS job_events (
  event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  ts           REAL NOT NULL,
  event_type   TEXT NOT NULL,
  worker_id    TEXT,
  job_id       TEXT,
  detail_json  TEXT NOT NULL DEFAULT '{}'
);
