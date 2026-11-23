experiments/run_benchmark.py
python
"""Benchmark distribuido: envía el mismo grid al coordinador, espera a que termine
y registra tiempo, % de reducción frente a T_base y estado de los jobs.

Requisito: coordinador con base de datos nueva y N workers activos."""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import grpc

import trainer_pb2 as pb
import trainer_pb2_grpc as pb_grpc
from experiments.grid import SEED, expand

TERMINAL = {pb.COMPLETED, pb.FAILED, pb.CANCELLED}
TARGET = {"LINEAR_REGRESSION": "distance_m", "MLP": "distance_m", "CNN": "category"}
METRIC = {"LINEAR_REGRESSION": "rmse", "MLP": "rmse", "CNN": "accuracy"}


def build_request(model: str, hp: dict) -> pb.TrainRequest:
    req = pb.TrainRequest(
        model_type=pb.ModelType.Value(model),
        dataset_version="nuscenes-v1.0-mini",
        split_version="nuscenes-mini-official",
        target_variable=TARGET[model],
        primary_metric=METRIC[model],
        seed=SEED,
    )
    if model == "LINEAR_REGRESSION":
        req.linear.CopyFrom(pb.LinearParams(alpha=[hp["alpha"]], normalize=[hp["normalize"]]))
    elif model == "MLP":
        req.mlp.CopyFrom(pb.MlpParams(hidden_layers=[hp["hidden_layers"]],
                                      learning_rate=[hp["learning_rate"]],
                                      epochs=[hp["epochs"]]))
    else:
        req.cnn.CopyFrom(pb.CnnParams(filters=[hp["filters"]], conv_layers=[hp["conv_layers"]],
                                      learning_rate=[hp["learning_rate"]],
                                      epochs=[hp["epochs"]], image_size=[hp["image_size"]]))
    return req


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordinator", default="localhost:50051")
    ap.add_argument("--nodes", type=int, required=True, help="Nodos activos (etiqueta de la bitácora)")
    ap.add_argument("--baseline", default="data/bitacora/baseline.json")
    ap.add_argument("--out", default="data/bitacora/benchmark.csv")
    ap.add_argument("--poll-s", type=float, default=2.0)
    a = ap.parse_args()

    t_base = json.loads(Path(a.baseline).read_text())["T_base_s"]
    stub = pb_grpc.TrainerServiceStub(grpc.insecure_channel(a.coordinator))

    t0 = time.perf_counter()
    job_ids = []
    for model, hp in expand():
        resp = stub.SubmitTraining(build_request(model, hp))
        job_ids += [j.job_id for j in resp.jobs]

    while True:
        infos = [stub.GetJobStatus(pb.JobQuery(job_id=j)) for j in job_ids]
        if all(i.status in TERMINAL for i in infos):
            break
        time.sleep(a.poll_s)
    wall = time.perf_counter() - t0

    completed = sum(i.status == pb.COMPLETED for i in infos)
    failed = sum(i.status == pb.FAILED for i in infos)
    reduction = 100.0 * (1.0 - wall / t_base)

    row = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "nodos": a.nodes,
        "jobs": len(job_ids),
        "completed": completed,
        "failed": failed,
        "T_dist_s": round(wall, 2),
        "T_base_s": t_base,
        "reduccion_pct": round(reduction, 2),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
