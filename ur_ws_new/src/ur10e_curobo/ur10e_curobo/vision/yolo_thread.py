"""YOLO inference thread for background processing."""

from threading import Lock, Event
from time import sleep, time
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .detection import detections_to_custom_masks


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
            if self.run_event.is_set():
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
                         max_det=25,
                         device=device,
                         verbose=False,
                         classes=self._detect_class_ids if self._detect_class_ids else None,
                     )[0]

                dt = time() - t0
                self.net_fps = (1.0 / dt) if dt > 0 else 0.0

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

            sleep(0.005)

        self.stopped.set()  # signal that the loop has fully exited

    def set_image(self, image: np.ndarray) -> None:
        """Set new image for inference. Dropped silently when paused."""
        if self.paused:
            return
        with self.lock:
            self.image_net = image
            self.run_event.set()

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
