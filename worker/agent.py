"""Worker: consume jobs.<worker>, ejecuta el pipeline y publica heartbeats y resultados.
Commit del offset solo después de publicar el resultado (at-least-once;
el coordinador descarta duplicados y resultados tardíos)."""
import argparse
import json
import logging
import os
import threading
import time

from confluent_kafka import Consumer, Producer

from worker.pipelines import REGISTRY

log = logging.getLogger("worker")
HB_EVERY_S = 5


def send(prod: Producer, topic: str, payload: dict, key: str) -> None:
    prod.produce(topic, json.dumps(payload).encode(), key=key.encode())
    prod.poll(0)


def heartbeat_loop(worker: str, prod: Producer, state: dict, lock: threading.Lock) -> None:
    while True:
        with lock:
            payload = {"worker": worker, "job_id": state["job_id"],
                       "progress": state["progress"], "ts": time.time()}
        send(prod, "heartbeats", payload, worker)
        time.sleep(HB_EVERY_S)


def build_cfg(config: dict) -> dict:
    """La identidad canónica trae hparams y seed; el pipeline espera un dict plano."""
    return {"seed": config["seed"], **config["hparams"]}


def run_job(job: dict, worker: str, state: dict, lock: threading.Lock) -> dict:
    base = {"job_id": job["job_id"], "worker": worker, "attempt": job["attempt"]}

    def progress(p: float) -> None:
        with lock:
            state["progress"] = float(p)

    try:
        out = REGISTRY[job["model_type"]].run(build_cfg(job["config"]), job["job_id"], progress)
        return {**base, "ok": True, "metrics": out["metrics"], "artifact": out["artifact_path"]}
    except (ValueError, KeyError) as e:  # configuración o datos inválidos: error permanente
        log.error("job %s inválido: %s", job["job_id"], e)
        return {**base, "ok": False, "permanent": True, "error": str(e)[:500]}
    except Exception as e:               # transitorio: el coordinador aplicará backoff
        log.exception("job %s falló", job["job_id"])
        return {**base, "ok": False, "permanent": False,
                "error": f"{type(e).__name__}: {e}"[:500]}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", required=True)
    worker = ap.parse_args().worker_id

    boot = os.environ["KAFKA_BOOTSTRAP"]
    prod = Producer({"bootstrap.servers": boot, "enable.idempotence": True, "acks": "all"})
    cons = Consumer({
        "bootstrap.servers": boot,
        "group.id": f"worker-{worker}",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    cons.subscribe([f"jobs.{worker}"])

    state = {"job_id": None, "progress": 0.0}
    lock = threading.Lock()
    threading.Thread(target=heartbeat_loop, args=(worker, prod, state, lock), daemon=True).start()
    log.info("worker %s escuchando jobs.%s", worker, worker)

    while True:
        m = cons.poll(1.0)
        if m is None:
            continue
        if m.error():
            log.warning("error de Kafka: %s", m.error())
            continue

        job = json.loads(m.value())
        with lock:
            state.update(job_id=job["job_id"], progress=0.0)

        result = run_job(job, worker, state, lock)
        send(prod, "results", result, job["job_id"])
        prod.flush(10)
        cons.commit(message=m, asynchronous=False)

        with lock:
            state.update(job_id=None, progress=0.0)


if __name__ == "__main__":
    main()
