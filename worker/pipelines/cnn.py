"""CNN para clasificación de categoría sobre recortes CAM_* (recortes.npy).
Checkpoint por época con reanudación. El artefacto final se escribe de forma atómica."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from . import common as C

BATCH = 64


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, conv_layers: int, filters: int):
        super().__init__()
        layers, c_in = [], 3
        for i in range(conv_layers):
            c_out = filters * (2 ** i)
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.net = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_in, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _images(rows: np.ndarray, recortes, size: tuple[int, int]) -> torch.Tensor:
    X = torch.from_numpy(np.ascontiguousarray(recortes[rows])).permute(0, 3, 1, 2).float() / 255.0
    if tuple(X.shape[-2:]) != size:
        X = F.interpolate(X, size=size, mode="bilinear", align_corners=False)
    return X


def run(cfg: dict, job_id: str, progress) -> dict:
    """cfg: {seed, filters, conv_layers, learning_rate, epochs, image_size ("64x64")}."""
    C.set_seed(cfg["seed"])
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

    df = C.load_objects()
    df = df[df["recorte_idx"] >= 0]              # solo objetos con recorte CAM
    tr, va = C.load_split(df)

    classes = sorted(tr[C.TARGET_CLS].unique())
    cls2i = {c: i for i, c in enumerate(classes)}
    va = va[va[C.TARGET_CLS].isin(classes)]

    recortes = np.load(C.PROCESSED / "recortes.npy", mmap_mode="r")
    size = tuple(int(v) for v in cfg["image_size"].lower().split("x"))

    rows_tr = tr["recorte_idx"].to_numpy()
    y_tr = tr[C.TARGET_CLS].map(cls2i).to_numpy()

    model = SmallCNN(len(classes), cfg["conv_layers"], cfg["filters"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    # Reanudar desde checkpoint si existe (p. ej. tras reasignación)
    ckpt = C.CKPT_DIR / job_id / "last.pt"
    start = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start = state["epoch"] + 1

    epochs = cfg["epochs"]
    for ep in range(start, epochs):
        # Semilla por época: el orden de batches es el mismo aunque el job se reanude
        rng = np.random.default_rng(cfg["seed"] + ep)
        order = rng.permutation(len(rows_tr))
        model.train()
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            X = _images(rows_tr[idx], recortes, size)
            y = torch.as_tensor(y_tr[idx], dtype=torch.long)
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()

        state = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep}
        C.atomic_write(ckpt, lambda f, s=state: torch.save(s, f))
        progress(0.9 * (ep + 1) / epochs)

    # Evaluación en validación
    model.eval()
    rows_va = va["recorte_idx"].to_numpy()
    y_va = va[C.TARGET_CLS].map(cls2i).to_numpy()
    preds = []
    with torch.no_grad():
        for b in range(0, len(rows_va), BATCH):
            X = _images(rows_va[b:b + BATCH], recortes, size)
            preds.append(model(X).argmax(dim=1).numpy())
    pred = np.concatenate(preds)

    metrics = {
        "accuracy": float(accuracy_score(y_va, pred)),
        "macro_f1": float(f1_score(y_va, pred, average="macro")),
    }

    artifact = {
        "model": model.state_dict(),
        "classes": [str(c) for c in classes],
        "image_size": size,
        "conv_layers": cfg["conv_layers"],
        "filters": cfg["filters"],
    }
    path = C.MODELS_DIR / job_id / "model.pt"
    C.atomic_write(path, lambda f: torch.save(artifact, f))
    progress(1.0)
    return {"metrics": metrics, "artifact_path": str(path)}
