"""Find dates in every image of a NuRec scene (input for localize_dates.py).

--method yolo     the date detector's boxes (date classes only).
--method fastsam  FastSAM "segment everything", keeping compact, solid, date-sized masks whose
                  colour matches the dates the detector found confidently. Recovers the dense
                  inner dates the detector misses; localize_dates.py's multi-view voting removes
                  the remaining false positives.

Runs in the YOLO environment:

  ~/nurec_data/yolo_venv/bin/python isaac/nurec/detect_dates.py ~/nurec_data/scenes/palm3 --method fastsam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, FastSAM

from perception_check import DEFAULT_WEIGHTS, detect


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--method", choices=("yolo", "fastsam"), default="fastsam")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--sam_weights",
        type=Path,
        default=Path.home() / "nurec_data" / "models" / "FastSAM-x.pt",
        help="FastSAM checkpoint (downloaded there on first use).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence (yolo boxes).")
    parser.add_argument("--class_prefix", default="date", help="Classes whose name starts with this are dates.")
    parser.add_argument("--colour_conf", type=float, default=0.5, help="Detector boxes used for the colour model.")
    parser.add_argument("--max_colour_distance", type=float, default=2.5, help="Median Mahalanobis distance in Lab.")
    parser.add_argument("--min_solidity", type=float, default=0.85)
    parser.add_argument("--max_aspect", type=float, default=2.8)
    parser.add_argument("--area", type=float, nargs=2, default=(150, 40000), help="Mask area range [px].")
    return parser.parse_args()


def yolo_boxes(args, model, date_classes, image):
    return [
        [*map(float, box), float(score)]
        for cls, score, box in detect(model, image, args.conf)
        if cls in date_classes
    ]


def colour_model(args, scene: Path, boxes: dict):
    """Lab mean / inverse covariance of the centres of confidently detected dates."""
    samples = []
    for name, found in boxes.items():
        lab = None
        for x0, y0, x1, y1, score in found:
            if score < args.colour_conf:
                continue
            if lab is None:
                lab = cv2.cvtColor(cv2.imread(str(scene / "images" / name)), cv2.COLOR_BGR2LAB)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
            samples.append(lab[int(cy - h):int(cy + h), int(cx - w):int(cx + w)].reshape(-1, 3))
    if not samples:
        raise SystemExit("No confident detector boxes to learn the date colour from; lower --colour_conf.")
    samples = np.concatenate(samples).astype(np.float32)
    return samples.mean(0), np.linalg.inv(np.cov(samples.T) + np.eye(3) * 4.0)


def fastsam_dates(args, sam, image, colour, robot_mask):
    """Instance label image (0 = none) of date-like FastSAM masks, and their boxes."""
    mean, inv_cov = colour
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    result = sam(image, retina_masks=True, imgsz=1280, conf=0.3, iou=0.8, verbose=False)[0]
    labels = np.zeros(image.shape[:2], np.uint16)
    boxes = []
    if result.masks is None:
        return labels, boxes
    for mask in result.masks.data.cpu().numpy().astype(bool):
        area = int(mask.sum())
        if not args.area[0] < area < args.area[1]:
            continue
        if robot_mask is not None and (robot_mask[mask] == 0).mean() > 0.3:
            continue  # gripper
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5 or area / max(cv2.contourArea(cv2.convexHull(contour)), 1) < args.min_solidity:
            continue
        _, (a, b), _ = cv2.fitEllipse(contour)
        if max(a, b) / max(min(a, b), 1) > args.max_aspect:
            continue
        diff = lab[mask] - mean
        if np.median(np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))) > args.max_colour_distance:
            continue
        labels[mask & (labels == 0)] = len(boxes) + 1
        ys, xs = np.nonzero(mask)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), 1.0])
    return labels, boxes


def main():
    args = parse_args()
    model = YOLO(str(args.weights))
    date_classes = {c for c, name in model.names.items() if name.startswith(args.class_prefix)}
    images = sorted((args.scene / "images").glob("*/*.jpg"))
    names = [str(path.relative_to(args.scene / "images")) for path in images]

    boxes = {name: yolo_boxes(args, model, date_classes, cv2.imread(str(path))) for name, path in zip(names, images)}
    if args.method == "fastsam":
        colour = colour_model(args, args.scene, boxes)
        args.sam_weights.parent.mkdir(parents=True, exist_ok=True)
        sam = FastSAM(str(args.sam_weights))
        for name, path in zip(names, images):
            robot_path = args.scene / "masks" / f"{name}.png"
            robot_mask = cv2.imread(str(robot_path), cv2.IMREAD_GRAYSCALE) if robot_path.exists() else None
            labels, boxes[name] = fastsam_dates(args, sam, cv2.imread(str(path)), colour, robot_mask)
            out = args.scene / "date_masks" / Path(name).with_suffix(".png")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), labels)

    total = sum(len(found) for found in boxes.values())
    out = args.scene / "date_detections.json"
    out.write_text(json.dumps({
        "method": args.method,
        "classes": sorted(model.names[c] for c in date_classes),
        "masks": "date_masks" if args.method == "fastsam" else None,
        "boxes": boxes,
    }))
    print(f"[INFO] {args.method}: {len(images)} images, {total} dates, "
          f"{sum(bool(found) for found in boxes.values())} images with dates -> {out}")


if __name__ == "__main__":
    main()
