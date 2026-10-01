"""Run the date detector on real frames and their NuRec renders, side by side.

Pairs SCENE/images/left/NNNNN.jpg with RENDERS/isaac_NNNNN.png (render_nurec_views.py),
draws both sets of boxes, and reports per-class agreement (IoU-matched detections).

  ~/nurec_data/yolo_venv/bin/python isaac/nurec/perception_check.py SCENE RENDERS OUT [--weights W]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# The robot's date model as PyTorch weights (the Jetson runs TensorRT engines built from it).
# The .pt is no longer tracked on curobo-v2, so fall back to a copy kept outside the repo.
_WEIGHTS_NAME = "yolo26_zedone4k_3classes.pt"
_REPO_WEIGHTS = Path(__file__).resolve().parents[2] / "ur_ws_new/src/zed_date_detector/models" / _WEIGHTS_NAME
DEFAULT_WEIGHTS = Path(os.environ.get(
    "NUREC_DETECTOR_WEIGHTS",
    _REPO_WEIGHTS if _REPO_WEIGHTS.exists() else Path.home() / "nurec_data/models" / _WEIGHTS_NAME,
))
COLORS = [(0, 200, 255), (60, 220, 60), (255, 120, 40)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("renders", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.3, help="IoU for a real/render box to count as matched.")
    return parser.parse_args()


def detect(model, image, conf):
    result = model.predict(image, conf=conf, verbose=False)[0]
    return [(int(c), float(s), b) for c, s, b in zip(
        result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()
    )]


def iou(a, b) -> float:
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(real, sim, threshold):
    """Greedy same-class IoU matching; returns (matched, real_only, sim_only) counts per class."""
    stats = {}
    used = set()
    for cls, _, box in sorted(real, key=lambda d: -d[1]):
        best, best_iou = None, threshold
        for j, (sim_cls, _, sim_box) in enumerate(sim):
            if j not in used and sim_cls == cls and (value := iou(box, sim_box)) >= best_iou:
                best, best_iou = j, value
        entry = stats.setdefault(cls, [0, 0, 0])
        if best is None:
            entry[1] += 1
        else:
            used.add(best)
            entry[0] += 1
    for j, (cls, _, _) in enumerate(sim):
        if j not in used:
            stats.setdefault(cls, [0, 0, 0])[2] += 1
    return stats


def draw(image, detections, names, label):
    image = image.copy()
    for cls, score, box in detections:
        x0, y0, x1, y1 = box.astype(int)
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cv2.putText(image, f"{names[cls]} {score:.2f}", (x0, max(15, y0 - 5)), 0, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(image, label, (12, 32), 0, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, label, (12, 32), 0, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    names = model.names
    totals = {}
    for render_path in sorted(args.renders.glob("isaac_*.png")):
        index = render_path.stem.split("_")[1]
        real = cv2.imread(str(args.scene / "images" / "left" / f"{index}.jpg"))
        sim = cv2.imread(str(render_path))
        real_det, sim_det = detect(model, real, args.conf), detect(model, sim, args.conf)
        for cls, counts in match(real_det, sim_det, args.iou).items():
            totals.setdefault(cls, np.zeros(3, int))[:] += counts
        panel = np.hstack([draw(real, real_det, names, f"real {index}"), draw(sim, sim_det, names, "Isaac NuRec")])
        cv2.imwrite(str(args.out / f"compare_{index}.jpg"), panel)
        print(f"{index}: real {len(real_det):2d} boxes, Isaac {len(sim_det):2d} boxes")

    print("\nper class: matched / real only / Isaac only")
    for cls, (matched, real_only, sim_only) in sorted(totals.items()):
        print(f"  {names[cls]:18s} {matched:3d} / {real_only:3d} / {sim_only:3d}")
    print(f"[DONE] side-by-side images in {args.out}")


if __name__ == "__main__":
    main()
