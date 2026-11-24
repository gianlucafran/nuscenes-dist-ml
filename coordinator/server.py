"""Coordinador: API gRPC, consumo de heartbeats y resultados desde Kafka, y tick del scheduler."""
import hashlib
import itertools
import json
import logging
import os
import re
import threading
import time
from concurrent import futures

import grpc
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from google.protobuf.timestamp_pb2 import Timestamp

import trainer_pb2 as pb
import trainer_pb2_grpc as pb_grpc
from coordinator.scheduler import WORKERS, Scheduler
from coordinator.store import Store

log = logging.getLogger("coordinator")

KAFKA = os.environ["KAFKA_BOOTSTRAP"]
DB = os.environ.get("COORD_DB", "data/db/coordinator.db")
PORT = os.environ.get("GRPC_PORT", "50051")
CODE_VERSION = os.environ.get("CODE_VERSION", "dev")

MODEL_KEY = {pb.LINEAR_REGRESSION: "linear", pb.MLP: "mlp", pb.CNN: "cnn"}
TARGET = {"LINEAR_REGRESSION": "distance_m", "MLP": "distance_m", "CNN": "category"}
METRICS = {"LINEAR_REGRESSION": {"rmse", "mae"}, "MLP": {"rmse", "mae"},
           "CNN": {"accuracy", "macro_f1"}}
RULES = {
    "LINEAR_REGRESSION": {"alpha": "nonneg", "normalize": "any"},
    "MLP": {"hidden_layers": "layers", "learning_rate": "pos", "epochs": "pos_int"},
    "CNN": {"filters": "pos_int", "conv_layers": "pos_int", "learning_rate": "pos",
            "epochs": "pos_int", "image_size": "size"},
}


# ---------- validación y expansión ----------

def _check(model, field, kind, values):
    if not values:
        raise ValueError(f"{model}.{field} no puede estar vacío")
    for v in values:
        if kind == "nonneg" and not v >= 0:
            raise ValueError(f"{model}.{field} debe ser >= 0")
        if kind == "pos" and not v > 0:
            raise ValueError(f"{model}.{field} debe ser > 0")
        if kind == "pos_int" and (not v > 0 or int(v) != v):
            raise ValueError(f"{model}.{field} debe ser entero > 0")
        if kind == "layers" and not re.fullmatch(r"\d+(,\d+)*", v):
            raise ValueError(f"{model}.{field} formato inválido, ej. '64,32'")
        if kind == "size" and not re.fullmatch(r"\d+x\d+", v):
            raise ValueError(f"{model}.{field} formato inválido, ej. '64x64'")


def validate(req) -> None:
    if req.model_type not in MODEL_KEY:
        raise ValueError("model_type inválido")
    name = pb.ModelType.Name(req.model_type)
    if req.WhichOneof("params") != MODEL_KEY[req.model_type]:
        raise ValueError("params no corresponde a model_type")
    if not req.dataset_version or not req.split_version:
        raise ValueError("dataset_version y split_version son obligatorios")
    if not req.HasField("seed"):
        raise ValueError("seed es obligatorio")
    if req.target_variable != TARGET[name]:
        raise ValueError(f"target_variable debe ser '{TARGET[name]}' para {name}")
    if req.primary_metric not in METRICS[name]:
        raise ValueError(f"primary_metric debe ser uno de {sorted(METRICS[name])}")
    params = getattr(req, MODEL_KEY[req.model_type])
    for field, kind in RULES[name].items():
        _check(name, field, kind, list(getattr(params, field)))


def expand(req):
    name = pb.ModelType.Name(req.model_type)
    params = getattr(req, MODEL_KEY[req.model_type])
    keys = list(RULES[name])
    for combo in itertools.product(*(list(getattr(params, k)) for k in keys)):
        yield name, dict(zip(keys, combo))


def make_job(req, name: str, hp: dict):
    identity = {
        "model_type": name,
        "dataset_version": req.dataset_version,
        "split_version": req.split_version,
        "target_variable": req.target_variable,
        "seed": req.seed,
        "hparams": hp,
        "code_version": CODE_VERSION,
    }
    canon = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:32], canon


def to_info(row: dict) -> pb.JobInfo:
    return pb.JobInfo(
        job_id=row["job_id"],
        model_type=pb.ModelType.Value(row["model_type"]),
        status=pb.JobStatus.Value(row["status"]),
        assigned_worker=row["assigned_worker"] or "",
        attempt=row["attempt"],
        error_message=row["error_message"] or "",
        progress=row["progress"],
        config_json=row["config_json"],
        metrics_json=row["metrics_json"] or "",
        model_artifact_path=row["model_artifact_path"] or "",
    )


# ---------- servicio gRPC ----------

class Servicer(pb_grpc.TrainerServiceServicer):
    def __init__(self, store: Store):
        self.store = store

    def SubmitTraining(self, req, ctx):
        try:
            validate(req)
            combos = [make_job(req, name, hp) for name, hp in expand(req)]
        except ValueError as e:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
            return pb.SubmitResponse()  # ctx.abort ya lanza; esto solo hace explícito el flujo

        now = time.time()
        infos, any_dup = [], False
        for job_id, canon in combos:
            row = {
                "job_id": job_id,
                "model_type": pb.ModelType.Name(req.model_type),
                "config_json": canon,
                "dataset_version": req.dataset_version,
                "split_version": req.split_version,
                "target_variable": req.target_variable,
                "primary_metric": req.primary_metric,
                "seed": req.seed,
                "status": "PENDING",
                "attempt": 0,
                "next_attempt_at": now,
                "progress": 0.0,
                "created_at": now,
                "updated_at": now,
                "version": 0,
            }
            if self.store.insert_job(row):
                self.store.event("JOB_SUBMITTED", None, job_id, {})
            else:
                any_dup = True
            infos.append(to_info(self.store.get_job(job_id)))
        return pb.SubmitResponse(jobs=infos, deduplicated=any_dup)

    def GetJobStatus(self, req, ctx):
        row = self.store.get_job(req.job_id)
        if row is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, "job no existe")
        return to_info(row)

    def GetJobResult(self, req, ctx):
        row = self.store.get_job(req.job_id)
        if row is None:
            ctx.abort(grpc.StatusCode.NOT_FOUND, "job no existe")
        if row["status"] != "COMPLETED":
            ctx.abort(grpc.StatusCode.FAILED_PRECONDITION,
                      f"job en estado {row['status']}")
        return to_info(row)

    def GetGridSummary(self, req, ctx):
        rows = []
        for jid in req.job_ids:
            r = self.store.get_job(jid)
            if r and r["status"] == "COMPLETED" and r["metrics_json"]:
                if req.primary_metric in json.loads(r["metrics_json"]):
                    rows.append(r)
        if not rows:
            return pb.GridSummaryResponse()
        pick = max if req.higher_is_better else min
        best = pick(rows, key=lambda r: json.loads(r["metrics_json"])[req.primary_metric])
        return pb.GridSummaryResponse(best_job=to_info(best),
                                      all_jobs=[to_info(r) for r in rows])

    def StreamEvents(self, req, ctx):
        last = req.since_event_id
        while ctx.is_active():
            rows = self.store.events_since(last)
            for r in rows:
                last = r["event_id"]
                ts = Timestamp()
                ts.FromNanoseconds(int(r["ts"] * 1e9))
                yield pb.SystemEvent(
                    timestamp=ts,
                    event_type=r["event_type"],
                    worker_id=r["worker_id"] or "",
                    job_id=r["job_id"] or "",
                    detail_json=r["detail_json"],
                    event_id=r["event_id"],
                )
            if not rows:
                time.sleep(0.5)


# ---------- Kafka ----------

def ensure_topics() -> None:
    admin = AdminClient({"bootstrap.servers": KAFKA})
    wanted = [NewTopic(f"jobs.{w}", num_partitions=1, replication_factor=3) for w in WORKERS]
    wanted += [NewTopic("heartbeats", num_partitions=3, replication_factor=3),
               NewTopic("results", num_partitions=3, replication_factor=3)]
    for topic, fut in admin.create_topics(wanted).items():
        try:
            fut.result()
            log.info("tópico creado: %s", topic)
        except Exception as e:  # ya existe u otro error no fatal
            log.info("tópico %s: %s", topic, e)


def make_publisher(prod: Producer):
    def publish(worker: str, msg: dict) -> None:
        prod.produce(f"jobs.{worker}", json.dumps(msg).encode(),
                     key=msg["job_id"].encode())
        prod.flush(5)
    return publish


def consume(sched: Scheduler) -> None:
    c = Consumer({
        "bootstrap.servers": KAFKA,
        "group.id": "coordinator",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    c.subscribe(["heartbeats", "results"])
    while True:
        m = c.poll(1.0)
        if m is None:
            continue
        if m.error():
            log.warning("error de Kafka: %s", m.error())
            continue
        msg = json.loads(m.value())
        if m.topic() == "heartbeats":
            sched.on_heartbeat(msg["worker"], msg.get("job_id"), msg.get("progress", 0.0))
        else:
            sched.on_result(msg)
        c.commit(message=m, asynchronous=False)


def tick_loop(sched: Scheduler) -> None:
    while True:
        try:
            sched.tick()
        except Exception:
            log.exception("tick falló")
        time.sleep(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ensure_topics()
    store = Store(DB)
    prod = Producer({"bootstrap.servers": KAFKA, "enable.idempotence": True, "acks": "all"})
    sched = Scheduler(store, publish=make_publisher(prod))

    threading.Thread(target=consume, args=(sched,), daemon=True).start()
    threading.Thread(target=tick_loop, args=(sched,), daemon=True).start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_TrainerServiceServicer_to_server(Servicer(store), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    log.info("coordinador escuchando en :%s", PORT)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
