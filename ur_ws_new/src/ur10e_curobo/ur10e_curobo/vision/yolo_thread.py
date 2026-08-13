"""YOLO inference thread for background processing."""

import json
from threading import Lock, Event
from time import time
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .detection import detections_to_custom_masks


def engine_imgsz(weights: str) -> Optional[int]:
    """Return the square input size a TensorRT .engine was compiled for, read
    from the JSON metadata header ultralytics prepends to the file. Returns
    None for non-engine weights (.pt/.onnx are size-flexible) or if the header
    can't be parsed. A TRT engine's input size is fixed at export time and
    cannot be changed at inference — passing any other imgsz raises an
    AssertionError deep in the backend, so this is the authoritative size."""
    if not str(weights).endswith(".engine"):
        return None
    try:
        with open(weights, "rb") as fh:
            meta_len = int.from_bytes(fh.read(4), "little")
            meta = json.loads(fh.read(meta_len).decode("utf-8"))
        imgsz = meta.get("imgsz")
        if isinstance(imgsz, (list, tuple)):
            return int(imgsz[0])
        if imgsz is not None:
            return int(imgsz)
    except Exception:
        pass
    return None


class YoloThread:
    """Background thread for YOLO inference."""

    def __init__(self, weights: str, img_size=640, conf_thres: float = 0.35):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres

        self.lock = Lock()
        self.run_event = Event()
        self.dets_ready = Event()
        self.stopped = Event()  # set when run() loop exits
        # Set whenever no inference is queued or in flight; cleared for the
        # duration of a predict() call. Lets wait_until_idle() block instead
        # of polling for the GPU-handoff check before cuRobo moves the arm.
        self.idle_event = Event()
        self.idle_event.set()
        # Held during active GPU inference — cuRobo acquires this before planning
        # to prevent concurrent CUDA ops that corrupt shared GPU memory on Jetson.
        self.inference_lock = Lock()
        self.paused = False   # when True, set_image() drops frames — YOLO idles
        self.exit_signal = False

        self.image_net: Optional[np.ndarray] = None
        self.detections = None       # fruit-only detections (for ZED)
        self.trunk_boxes = []        # trunk bbox list: [(x1,y1,x2,y2), ...]
        self.bunch_boxes = []        # bunch bbox list: [(x1,y1,x2,y2), ...]
        self.net_fps = 0.0

        self._model: Optional[YOLO] = None
        self.class_names: dict = {}  # {class_id: class_name} from model

    def run(self) -> None:
        """Main thread loop - runs YOLO inference on available images."""
        print("Initializing Network...")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for YOLO; GPU not available.")

        device = torch.device("cuda")

        # A TRT engine's input size is baked in at export and cannot be resized
        # at inference. Force img_size to the engine's compiled size so a stale
        # DEFAULT_IMG_SIZE can never trigger the "input size N not equal to max
        # model size M" AssertionError. .pt/.onnx return None here and keep the
        # configured size (they resize freely). 
        native = engine_imgsz(self.weights)
        if native is not None and native != self.img_size:
            print(f"[YoloThread] img_size {self.img_size} does not match engine "
                  f"'{self.weights}' (compiled for {native}); using {native}.")
            self.img_size = native

        self._model = YOLO(self.weights, task='segment')
        self.class_names = getattr(self._model, 'names', {})

        # Separate fruit vs trunk vs bunch class IDs
        skip_names = {"stem"}
        self._trunk_class_ids = set(
            cid for cid, name in self.class_names.items()
            if name.lower() == "trunk"
        )
        self._bunch_class_ids = set(
            cid for cid, name in self.class_names.items()
            if name.lower() == "bunch"
        )
        # Classes to run inference on: fruit + trunk + bunch (skip stem only)
        self._detect_class_ids = [
            cid for cid, name in self.class_names.items()
            if name.lower() not in skip_names
        ]
        print(f"Network Initialized... classes: {self.class_names}")
        print(f"  Detect class IDs: {self._detect_class_ids} (skipping: {skip_names})")
        print(f"  Trunk class IDs: {self._trunk_class_ids}")
        print(f"  Bunch class IDs: {self._bunch_class_ids}")

        while not self.exit_signal:
            if not self.run_event.wait(timeout=0.1):
                continue

            self.idle_event.clear()
            with self.lock:
                img = cv2.cvtColor(self.image_net, cv2.COLOR_RGBA2RGB)

            t0 = time()
            with self.inference_lock:
                det = self._model.predict(
                     img,
                     save=False,
                     retina_masks=False,
                     imgsz=self.img_size,
                     conf=self.conf_thres,
                     iou=0.3,
                     max_det=10,
                     device=device,
                     verbose=False,
                     classes=self._detect_class_ids if self._detect_class_ids else None,
                 )[0]

            dt = time() - t0
            self.net_fps = (1.0 / dt) if dt > 0 else 0.0
            self._log_inference_time(dt)

            fruit_dets, trunk_boxes, bunch_boxes = detections_to_custom_masks(
                det, trunk_class_ids=self._trunk_class_ids,
                bunch_class_ids=self._bunch_class_ids
            )
            with self.lock:
                self.detections = fruit_dets
                self.trunk_boxes = trunk_boxes
                self.bunch_boxes = bunch_boxes

            self.run_event.clear()
            self.dets_ready.set()
            self.idle_event.set()

        self.stopped.set()  # signal that the loop has fully exited

    def set_image(self, image: np.ndarray) -> None:
        """Set new image for inference. Dropped silently when paused."""
        if self.paused:
            return
        with self.lock:
            self.image_net = image
            self.run_event.set()

    def _log_inference_time(self, dt: float) -> None:
        """Append inference time (ms) to CSV when UR10E_YOLO_TIMING_CSV is set.
        No-op (and no overhead beyond one attribute check) when unset. Used only
        for benchmarking the GPU handoff — safe to leave in place."""
        path = getattr(self, "_timing_csv_path", -1)
        if path == -1:
            import os
            path = os.getenv("UR10E_YOLO_TIMING_CSV")
            self._timing_csv_path = path
            self._timing_t0 = time()
            if path:
                try:
                    with open(path, "w") as fh:
                        fh.write("t_rel_s,inference_ms\n")
                except Exception:
                    self._timing_csv_path = None
        if not path:
            return
        try:
            with open(path, "a") as fh:
                fh.write(f"{time() - self._timing_t0:.4f},{dt * 1000.0:.3f}\n")
        except Exception:
            pass

    def wait_until_idle(self, timeout: float = 0.5) -> bool:
        """Block until no inference is pending or in flight, so the GPU is free
        for cuRobo. Returns True once idle, False if still busy after timeout."""
        return self.idle_event.wait(timeout=timeout)

    def get_detections(self):
        """Get latest fruit detections (thread-safe). Trunk is excluded."""
        with self.lock:
            return self.detections

    def get_trunk_boxes(self):
        """Get latest trunk bounding boxes (thread-safe)."""
        with self.lock:
            return list(self.trunk_boxes)

    def get_bunch_boxes(self):
        """Get latest bunch bounding boxes (thread-safe)."""
        with self.lock:
            return list(self.bunch_boxes)

    def stop(self) -> None:
        """Signal thread to stop."""
        self.exit_signal = True
