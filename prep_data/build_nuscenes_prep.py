"""Genera objetos.csv, recortes.npy y particion_escenas.json a partir de nuScenes v1.0-mini.

- Solo keyframes (samples/): las anotaciones 3D están ligadas a ellos.
- Solo cámaras CAM_*: LIDAR_TOP (.pcd.bin) y RADAR_* (.pcd) no son imágenes.
- Cada fila de objetos.csv es un recorte: una anotación visible en una cámara.
- recorte_idx es el índice de fila en recortes.npy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from nuscenes.utils.splits import create_splits_scenes

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CROP = 64
MIN_SIDE = 4
VIS_ORDER = {"v0-40": 0, "v40-60": 1, "v60-80": 2, "v80-100": 3}
CATEGORY_GROUPS = {"vehicle", "human", "movable_object", "static_object", "animal"}


def crop(img: Image.Image, corners_xy: np.ndarray):
    w, h = img.size
    x0, y0 = corners_xy.min(axis=1)
    x1, y1 = corners_xy.max(axis=1)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        return None
    patch = img.crop((x0, y0, x1, y1)).resize((CROP, CROP), Image.BILINEAR)
    return np.asarray(patch, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True,
                    help="Carpeta que contiene samples/, sweeps/, maps/ y v1.0-mini/ (JSON)")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=a.dataroot, verbose=False)

    # Partición oficial mini por escena (no por sample, para evitar fuga)
    scene_token = {s["name"]: s["token"] for s in nusc.scene}
    official = create_splits_scenes()
    split = {
        "train": [scene_token[n] for n in official["mini_train"] if n in scene_token],
        "val": [scene_token[n] for n in official["mini_val"] if n in scene_token],
    }
    (out / "particion_escenas.json").write_text(json.dumps(split, indent=2))

    rows, crops = [], []
    for sample in nusc.sample:
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_xy = np.array(ego["translation"][:2])

        for cam in CAMS:
            path, boxes, K = nusc.get_sample_data(
                sample["data"][cam], box_vis_level=BoxVisibility.ANY)
            img = Image.open(path).convert("RGB")

            for box in boxes:
                if box.center[2] <= 0:
                    continue
                corners = view_points(box.corners(), K, normalize=True)[:2]
                patch = crop(img, corners)
                if patch is None:
                    continue

                ann = nusc.get("sample_annotation", box.token)
                cat = ann["category_name"].split(".")[0]
                if cat not in CATEGORY_GROUPS:
                    cat = "other"
                vis = nusc.get("visibility", ann["visibility_token"])["level"]

                rows.append({
                    "scene_token": sample["scene_token"],
                    "sample_token": sample["token"],
                    "ann_token": box.token,
                    "camera": cam,
                    "category": cat,
                    "size_w": ann["size"][0],
                    "size_l": ann["size"][1],
                    "size_h": ann["size"][2],
                    "num_lidar_pts": ann["num_lidar_pts"],
                    "visibility": VIS_ORDER.get(vis, -1),
                    "yaw": Quaternion(ann["rotation"]).yaw_pitch_roll[0],
                    "distance_m": float(np.linalg.norm(np.array(ann["translation"][:2]) - ego_xy)),
                    "recorte_idx": len(crops),
                })
                crops.append(patch)

    pd.DataFrame(rows).to_csv(out / "objetos.csv", index=False)
    arr = np.stack(crops) if crops else np.zeros((0, CROP, CROP, 3), dtype=np.uint8)
    np.save(out / "recortes.npy", arr)
    print(f"objetos: {len(rows)} | recortes: {arr.shape} | "
          f"train escenas: {len(split['train'])} | val escenas: {len(split['val'])}")


if __name__ == "__main__":
    main()
