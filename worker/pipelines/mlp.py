
"""MLP para regresión de distancia. Entrena por épocas con partial_fit para reportar progreso."""

import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import common as C


def run(cfg: dict, job_id: str, progress) -> dict:
    """cfg: {seed, hidden_layers ("64,32"), learning_rate, epochs}."""
    C.set_seed(cfg["seed"])
    tr, va = C.load_split(C.load_objects())

    scaler = StandardScaler().fit(tr[C.FEATURES])  # ajustado solo con train
    X_tr = scaler.transform(tr[C.FEATURES])
    y_tr = tr[C.TARGET_REG].to_numpy()
    X_va = scaler.transform(va[C.FEATURES])

    hidden = tuple(int(x) for x in cfg["hidden_layers"].split(",") if x.strip())
    net = MLPRegressor(
        hidden_layer_sizes=hidden,
        learning_rate_init=cfg["learning_rate"],
        random_state=cfg["seed"],
    )

    epochs = cfg["epochs"]
    for ep in range(epochs):
        net.partial_fit(X_tr, y_tr)
        progress(0.9 * (ep + 1) / epochs)

    metrics = C.regression_metrics(va[C.TARGET_REG].to_numpy(), net.predict(X_va))

    path = C.MODELS_DIR / job_id / "model.joblib"
    C.atomic_write(path, lambda f: joblib.dump({"scaler": scaler, "net": net}, f))
    progress(1.0)

    return {"metrics": metrics, "artifact_path": str(path)}
