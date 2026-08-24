"""Minimal YOLO26 end-to-end TensorRT segmentation runtime.

The exported engine already performs end-to-end box suppression. This wrapper
keeps only the requested rows before reconstructing masks, then returns an
Ultralytics Results object so the rest of the harvesting pipeline is unchanged.
"""

from time import perf_counter

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops


class DirectTensorRTSegmenter:
    """Fast decoder for an Ultralytics end-to-end segmentation engine."""

    def __init__(self, engine: str, imgsz: int, class_names, device,
                 conf: float, class_ids, max_det: int):
        self.imgsz = int(imgsz)
        self.names = class_names
        self.device = device
        self.conf = float(conf)
        self.class_ids = tuple(int(value) for value in (class_ids or ()))
        self.max_det = int(max_det)
        self.backend = AutoBackend(
            model=engine, device=device, fp16=True, verbose=False)
        self.letterbox = LetterBox(
            new_shape=(self.imgsz, self.imgsz), auto=False, stride=32)

    def predict(self, image: np.ndarray) -> Results:
        t0 = perf_counter()
        padded = self.letterbox(image=image)
        tensor = torch.from_numpy(np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)))
        tensor = tensor.to(self.device, non_blocking=True)
        tensor = tensor.half() if self.backend.fp16 else tensor.float()
        tensor = tensor.unsqueeze(0) / 255.0
        t1 = perf_counter()

        raw = self.backend(tensor)
        t2 = perf_counter()

        predictions, prototypes = raw[0], raw[1]
        rows = predictions[0]
        # Match Ultralytics end-to-end filtering order exactly: confidence,
        # max_det, then requested classes. Exported rows are confidence-sorted.
        rows = rows[rows[:, 4] > self.conf][:self.max_det]
        if self.class_ids and rows.shape[0]:
            wanted = torch.tensor(
                self.class_ids, device=rows.device, dtype=rows[:, 5].dtype)
            rows = rows[(rows[:, 5:6] == wanted).any(1)]

        masks = None
        if rows.shape[0]:
            masks = ops.process_mask(
                prototypes[0], rows[:, 6:], rows[:, :4],
                tensor.shape[2:], upsample=True)
            keep = masks.amax((-2, -1)) > 0
            if not bool(torch.all(keep)):
                rows, masks = rows[keep], masks[keep]
            boxes = rows[:, :6].clone()
            ops.scale_boxes(tensor.shape[2:], boxes[:, :4], image.shape[:2])
        else:
            boxes = rows[:, :6]
        t3 = perf_counter()

        result = Results(
            image, path="direct_trt_frame", names=self.names,
            boxes=boxes, masks=masks)
        result.speed = {
            # These are CPU wall intervals for enqueue/construction. The
            # downstream masks.xy conversion is the natural synchronization
            # point and is reported separately as mask_cpu by YoloThread.
            "preprocess": (t1 - t0) * 1000.0,
            "inference": (t2 - t1) * 1000.0,
            "postprocess": (t3 - t2) * 1000.0,
        }
        return result
