from . import cnn, lineal, mlp

REGISTRY = {
    "LINEAR_REGRESSION": lineal,
    "MLP": mlp,
    "CNN": cnn,
}
