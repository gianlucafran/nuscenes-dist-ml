"""Utilidades compartidas: datos, semillas, métricas y escritura atómica."""
#K
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(os.environ.get("NUSC_DATA", "data"))
PROCESSED = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
CKPT_DIR = DATA_DIR / "checkpoints"

# Columnas de objetos.csv (generadas por prep_data a partir de sample_annotation.json)
FEATURES = ["size_w", "size_l", "size_h", "num_lidar_pts", "visibility", "yaw"]
TARGET_REG = "distance_m"   # distancia ego -> objeto (lineal y MLP)
TARGET_CLS = "category"     # categoría agrupada (CNN)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_objects() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "objetos.csv")


def load_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = json.loads((PROCESSED / "particion_escenas.json").read_text())
    tr = df[df["scene_token"].isin(split["train"])]
    va = df[df["scene_token"].isin(split["val"])]
    if tr.empty or va.empty:
        # ValueError = error no recuperable: el agente lo marca FAILED sin reintentar
        raise ValueError("Partición vacía: revisar particion_escenas.json")
    return tr, va


def regression_metrics(y, pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def atomic_write(path: Path, writer) -> None:
    """writer(file_obj) escribe el contenido. El rename final es atómico en el mismo filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
