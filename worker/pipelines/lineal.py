"""Regresión lineal (Ridge) sobre features tabulares de objetos.csv."""
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import common as C


def run(cfg: dict, job_id: str, progress) -> dict:
    """cfg: {seed, alpha, normalize}. progress(float 0..1). Devuelve métricas y ruta del artefacto."""
    C.set_seed(cfg["seed"])
    tr, va = C.load_split(C.load_objects())

    steps = [StandardScaler()] if cfg["normalize"] else []
    model = make_pipeline(*steps, Ridge(alpha=cfg["alpha"]))

    progress(0.1)
    model.fit(tr[C.FEATURES], tr[C.TARGET_REG])
    progress(0.8)

    metrics = C.regression_metrics(va[C.TARGET_REG].to_numpy(), model.predict(va[C.FEATURES]))

    path = C.MODELS_DIR / job_id / "model.joblib"
    C.atomic_write(path, lambda f: joblib.dump(model, f))
    progress(1.0)
    return {"metrics": metrics, "artifact_path": str(path)}
