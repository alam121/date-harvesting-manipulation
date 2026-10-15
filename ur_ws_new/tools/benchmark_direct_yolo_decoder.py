#!/usr/bin/env python3
"""Compare Ultralytics predict() with a minimal YOLO26 end-to-end TRT decoder.

This is intentionally standalone. It does not publish ROS topics or alter the
harvesting detector. Stop the harvesting/vision nodes before running it so the
timings are not distorted by GPU contention.
"""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class DirectDecoder:
    def __init__(self, engine: str, imgsz: int, conf: float, class_id: int,
                 max_det: int):
        self.imgsz = imgsz
        self.conf = conf
        self.class_id = class_id
        self.max_det = max_det
        self.device = torch.device("cuda:0")
        self.backend = AutoBackend(
            model=engine, device=self.device, fp16=True, verbose=False)
        self.letterbox = LetterBox(
            new_shape=(imgsz, imgsz), auto=False, stride=32)

    def preprocess(self, image: np.ndarray):
        padded = self.letterbox(image=image)
        tensor = torch.from_numpy(
            np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)))
        tensor = tensor.to(self.device, non_blocking=True)
        # Respect the actual TensorRT input binding. This engine computes in
        # FP16 internally but exposes an FP32 input tensor, so forcing .half()
        # corrupts the buffer interpreted by TensorRT.
        tensor = tensor.half() if self.backend.fp16 else tensor.float()
        tensor /= 255.0
        return tensor.unsqueeze(0)

    def decode(self, image: np.ndarray):
        t0 = perf_counter()
        tensor = self.preprocess(image)
        synchronize()
        t1 = perf_counter()

        raw = self.backend(tensor)
        synchronize()
        t2 = perf_counter()

        # Export metadata says end2end=True. output0 is Bx300x38:
        # xyxy, confidence, class, 32 mask coefficients. output1 is the
        # Bx32x312x312 mask prototype tensor.
        predictions, prototypes = raw[0], raw[1]
        rows = predictions[0]
        keep = ((rows[:, 4] >= self.conf) &
                (rows[:, 5].round().long() == self.class_id))
        rows = rows[keep]
        if rows.shape[0] > self.max_det:
            order = rows[:, 4].argsort(descending=True)[:self.max_det]
            rows = rows[order]

        if rows.shape[0]:
            masks = ops.process_mask(
                prototypes[0], rows[:, 6:], rows[:, :4],
                tensor.shape[2:], upsample=True)
            boxes = rows[:, :4].clone()
            ops.scale_boxes(tensor.shape[2:], boxes, image.shape[:2])
            boxes_np = boxes.detach().cpu().numpy()
            conf_np = rows[:, 4].detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy().astype(bool)
        else:
            boxes_np = np.empty((0, 4), np.float32)
            conf_np = np.empty((0,), np.float32)
            masks_np = np.empty((0, self.imgsz, self.imgsz), bool)
        synchronize()
        t3 = perf_counter()
        return boxes_np, conf_np, masks_np, {
            "pre_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "decode_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=1248)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--class-id", type=int, default=1)
    parser.add_argument("--max-det", type=int, default=3)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    direct = DirectDecoder(
        args.engine, args.imgsz, args.conf, args.class_id, args.max_det)
    # Warm up TensorRT and CUDA allocations before timing.
    for _ in range(3):
        direct.decode(image)

    samples = []
    direct_result = None
    for _ in range(args.runs):
        direct_result = direct.decode(image)
        samples.append(direct_result[3])

    print("DIRECT DECODER")
    for key in ("pre_ms", "infer_ms", "decode_ms", "total_ms"):
        values = np.asarray([sample[key] for sample in samples])
        print(f"  {key}: mean={values.mean():.2f} p95={np.percentile(values, 95):.2f}")

    # One reference result is enough for equivalence: both paths are
    # deterministic for the same engine and image.
    reference = YOLO(args.engine, task="segment").predict(
        image, imgsz=args.imgsz, conf=args.conf, iou=0.3,
        max_det=args.max_det, classes=[args.class_id], device="cuda:0",
        retina_masks=False, verbose=False)[0]
    ref_boxes = reference.boxes.xyxy.detach().cpu().numpy()
    ref_conf = reference.boxes.conf.detach().cpu().numpy()
    ref_masks = (
        reference.masks.data.detach().cpu().numpy().astype(bool)
        if reference.masks is not None else np.empty((0, args.imgsz, args.imgsz), bool))
    direct_boxes, direct_conf, direct_masks, _ = direct_result

    print("EQUIVALENCE")
    print(f"  direct_count={len(direct_boxes)} ultralytics_count={len(ref_boxes)}")
    for index in range(min(len(direct_boxes), len(ref_boxes))):
        mask_intersection = np.count_nonzero(direct_masks[index] & ref_masks[index])
        mask_union = np.count_nonzero(direct_masks[index] | ref_masks[index])
        mask_iou = mask_intersection / mask_union if mask_union else 1.0
        print(
            f"  #{index + 1}: box_iou={box_iou(direct_boxes[index], ref_boxes[index]):.5f} "
            f"mask_iou={mask_iou:.5f} "
            f"direct_conf={direct_conf[index]:.5f} reference_conf={ref_conf[index]:.5f}")


if __name__ == "__main__":
    main()
