"""Grid y definiciones compartidas por baseline, benchmark y coordinador.
Fuente única de: versiones de dataset/split, target y métrica por modelo."""
import itertools

__all__ = ["SEED", "GRID", "TARGET", "METRIC", "DATASET_VERSION", "SPLIT_VERSION", "expand"]

SEED = 42

GRID = {
    "LINEAR_REGRESSION": {"alpha": [0.1, 1.0, 10.0], "normalize": [True, False]},
    "MLP": {"hidden_layers": ["64,32", "128,64"], "learning_rate": [1e-3], "epochs": [20]},
    "CNN": {"filters": [16], "conv_layers": [2, 3], "learning_rate": [1e-3],
            "epochs": [5], "image_size": ["64x64"]},
}

# Target y métrica primaria por modelo.
TARGET = {"LINEAR_REGRESSION": "distance_m", "MLP": "distance_m", "CNN": "category"}
METRIC = {"LINEAR_REGRESSION": "rmse", "MLP": "rmse", "CNN": "accuracy"}
DATASET_VERSION = "nuscenes-v1.0-mini"
SPLIT_VERSION = "nuscenes-mini-official"


def expand():
    for model, space in GRID.items():
        keys = list(space)
        for vals in itertools.product(*(space[k] for k in keys)):
            yield model, dict(zip(keys, vals))
