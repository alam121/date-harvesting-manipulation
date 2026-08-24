"""YOLO detection helpers and mask conversion."""

from typing import List

import cv2
import numpy as np
import pyzed.sl as sl


# Global storage for sl.Mat objects to prevent garbage collection
_sl_mats: List[sl.Mat] = []


def xywh2abcd(xywh: np.ndarray) -> np.ndarray:
    """Convert xywh bounding box format to 4-corner format."""
    out = np.zeros((4, 2), dtype=np.float32)
    x_min = xywh[0] - 0.5 * xywh[2]
    x_max = xywh[0] + 0.5 * xywh[2]
    y_min = xywh[1] - 0.5 * xywh[3]
    y_max = xywh[1] + 0.5 * xywh[3]
    out[0] = [x_min, y_min]
    out[1] = [x_max, y_min]
    out[2] = [x_max, y_max]
    out[3] = [x_min, y_max]
    return out


def detections_to_custom_masks(dets, trunk_class_ids=None, bunch_class_ids=None,
                               class_names=None, build_custom_masks=True):
    """Convert YOLO detections to ZED CustomMaskObjectData format.

    Returns (fruit_dets, trunk_boxes, bunch_boxes, raw_viz) where:
      - fruit_dets: list of sl.CustomMaskObjectData for non-trunk/bunch classes
      - trunk_boxes: list of (x1, y1, x2, y2) int tuples for trunk detections
      - bunch_boxes: list of dicts {"bb": (x1,y1,x2,y2), "polygon": np.ndarray|None, "conf": float}
    """
    global _sl_mats
    fruit_output = []
    trunk_boxes = []
    bunch_boxes = []
    raw_viz = []
    _sl_mats = []
    if trunk_class_ids is None:
        trunk_class_ids = set()
    if bunch_class_ids is None:
        bunch_class_ids = set()
    H, W = dets.orig_shape
    # One GPU→CPU synchronization per tensor and one polygon conversion per
    # result set. Accessing .item()/.cpu()/.masks.xy inside the loop caused a
    # separate synchronization (and repeated contour generation) per fruit.
    classes = dets.boxes.cls.detach().cpu().numpy().astype(np.int32)
    boxes_xyxy = dets.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confidences = dets.boxes.conf.detach().cpu().numpy().astype(np.float32)
    masks_xy = dets.masks.xy if dets.masks is not None else None

    for di in range(len(dets.boxes)):
        cls_id = int(classes[di])
        x1f, y1f, x2f, y2f = boxes_xyxy[di]
        abcd = np.array(
            [[x1f, y1f], [x2f, y1f], [x2f, y2f], [x1f, y2f]],
            dtype=np.float32)
        abcd[:, 0] = np.clip(abcd[:, 0], 0, W - 1)
        abcd[:, 1] = np.clip(abcd[:, 1], 0, H - 1)

        x1 = int(abcd[0, 0])
        y1 = int(abcd[0, 1])
        x2 = int(abcd[2, 0])
        y2 = int(abcd[2, 1])

        polygon = None
        if masks_xy is not None:
            xy = masks_xy[di]
            if len(xy) > 2:
                polygon = xy.astype(np.float32)
        if class_names is None:
            class_name = cls_id
        elif hasattr(class_names, "get"):
            class_name = class_names.get(cls_id, cls_id)
        else:
            class_name = class_names[cls_id]
        raw_viz.append({
            "bb": (x1, y1, x2, y2),
            "class": str(class_name),
            "conf": float(confidences[di]),
            "polygon": polygon,
        })

        if not build_custom_masks:
            continue

        # Trunk → just save bbox, don't create ZED object (only one trunk)
        if cls_id in trunk_class_ids:
            if not trunk_boxes:
                trunk_boxes.append((x1, y1, x2, y2))
            continue

        # Bunch → save bbox + segmentation polygon (only one bunch)
        if cls_id in bunch_class_ids:
            if not bunch_boxes:
                conf = float(confidences[di])
                bunch_boxes.append({"bb": (x1, y1, x2, y2), "polygon": polygon, "conf": conf})
            continue

        obj = sl.CustomMaskObjectData()
        obj.bounding_box_2d = abcd
        obj.label = cls_id
        obj.probability = float(confidences[di])
        obj.is_grounded = False

        if dets.masks is not None:
            x_min = int(abcd[0, 0])
            y_min = int(abcd[0, 1])
            x_max = int(abcd[2, 0])
            y_max = int(abcd[2, 1])
            roi_h = max(1, y_max - y_min + 1)
            roi_w = max(1, x_max - x_min + 1)
            mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            if masks_xy is not None:
                xy = masks_xy[di]
                # fillPoly on bbox ROI only with contour translated locally.
                xy_local = xy.copy()
                xy_local[:, 0] -= x_min
                xy_local[:, 1] -= y_min
                if len(xy_local) >= 3:
                    cv2.fillPoly(mask_roi, [xy_local.astype(np.int32).reshape(-1, 1, 2)], 255)
            if not mask_roi.flags.c_contiguous:
                mask_roi = np.ascontiguousarray(mask_roi)
            sl_mat = sl.Mat(
                width=mask_roi.shape[1],
                height=mask_roi.shape[0],
                mat_type=sl.MAT_TYPE.U8_C1,
                memory_type=sl.MEM.CPU,
            )
            np.copyto(sl_mat.get_data(), mask_roi)
            _sl_mats.append(sl_mat)
            obj.box_mask = sl_mat

        fruit_output.append(obj)
    raw_viz.sort(key=lambda item: item["conf"], reverse=True)
    return fruit_output, trunk_boxes, bunch_boxes, raw_viz
