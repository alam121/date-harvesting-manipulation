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


def detections_to_custom_masks(dets, trunk_class_ids=None, bunch_class_ids=None):
    """Convert YOLO detections to ZED CustomMaskObjectData format.

    Returns (fruit_dets, trunk_boxes, bunch_boxes) where:
      - fruit_dets: list of sl.CustomMaskObjectData for non-trunk/bunch classes
      - trunk_boxes: list of (x1, y1, x2, y2) int tuples for trunk detections
      - bunch_boxes: list of dicts {"bb": (x1,y1,x2,y2), "polygon": np.ndarray|None, "conf": float}
    """
    global _sl_mats
    fruit_output = []
    trunk_boxes = []
    bunch_boxes = []
    _sl_mats = []
    if trunk_class_ids is None:
        trunk_class_ids = set()
    if bunch_class_ids is None:
        bunch_class_ids = set()
    H, W = dets.orig_shape

    for di in range(len(dets.boxes)):
        cls_id = int(dets.boxes.cls[di].item())
        xywh = dets.boxes.xywh[di].cpu().numpy().astype(np.float32)
        abcd = xywh2abcd(xywh)
        abcd[:, 0] = np.clip(abcd[:, 0], 0, W - 1)
        abcd[:, 1] = np.clip(abcd[:, 1], 0, H - 1)

        x1 = int(abcd[0, 0])
        y1 = int(abcd[0, 1])
        x2 = int(abcd[2, 0])
        y2 = int(abcd[2, 1])

        # Trunk → just save bbox, don't create ZED object (only one trunk)
        if cls_id in trunk_class_ids:
            if not trunk_boxes:
                trunk_boxes.append((x1, y1, x2, y2))
            continue

        # Bunch → save bbox + segmentation polygon (only one bunch)
        if cls_id in bunch_class_ids:
            if not bunch_boxes:
                conf = float(dets.boxes.conf[di].item())
                polygon = None
                if dets.masks is not None and dets.masks.xy is not None:
                    xy = dets.masks.xy[di]
                    if len(xy) > 2:
                        polygon = xy.astype(np.float32)
                bunch_boxes.append({"bb": (x1, y1, x2, y2), "polygon": polygon, "conf": conf})
            continue

        obj = sl.CustomMaskObjectData()
        obj.bounding_box_2d = abcd
        obj.label = cls_id
        obj.probability = float(dets.boxes.conf[di].item())
        obj.is_grounded = False

        if dets.masks is not None and dets.masks.data is not None:
            m = dets.masks.data[di].cpu().numpy()
            mask_bin = (m * 255).astype(np.uint8)
            # retina_masks=False returns masks at model resolution, not image resolution.
            # Resize to image dims so bbox coords (in image space) align correctly.
            if mask_bin.shape[0] != H or mask_bin.shape[1] != W:
                mask_bin = cv2.resize(mask_bin, (W, H), interpolation=cv2.INTER_NEAREST)
            x_min = int(abcd[0, 0])
            y_min = int(abcd[0, 1])
            x_max = int(abcd[2, 0])
            y_max = int(abcd[2, 1])
            mask_roi = mask_bin[y_min: y_max + 1, x_min: x_max + 1]
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
    return fruit_output, trunk_boxes, bunch_boxes
