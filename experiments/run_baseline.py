experiments/run_baseline.py
python
"""Baseline no distribuido: ejecuta el grid secuencialmente en este proceso y mide T_base."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from experiments.grid import SEED, expand
from worker.pipelines import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bitacora/baseline.json")
    a = ap.parse_args()

    tag = datetime.now().strftime("%Y%m%dT%H%M%S")
    runs = []
    t0 = time.perf_counter()
    for i, (model, hp) in enumerate(expand()):
        job_id = f"baseline-{tag}-{i:03d}"   # id único: no reutiliza checkpoints previos
        out = REGISTRY[model].run({"seed": SEED, **hp}, job_id, lambda p: None)
        runs.append({"model": model, "hparams": hp, "metrics": out["metrics"]})
    t_base = time.perf_counter() - t0

    result = {"T_base_s": round(t_base, 2), "seed": SEED, "fecha": tag, "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"T_base = {t_base:.2f} s ({len(runs)} configuraciones)")


if __name__ == "__main__":
    main()
