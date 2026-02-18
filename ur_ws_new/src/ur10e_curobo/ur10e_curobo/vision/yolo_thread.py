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

    def __init__(self, weights: str, img_size: int = 512, conf_thres: float = 0.35):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres

        self.lock = Lock()
        self.run_event = Event()
        self.dets_ready = Event()
        self.exit_signal = False

        self.image_net: Optional[np.ndarray] = None
        self.detections = None
        self.net_fps = 0.0

        self._model: Optional[YOLO] = None
        self.class_names: dict = {}  # {class_id: class_name} from model

    def run(self) -> None:
        """Main thread loop - runs YOLO inference on available images."""
        print("Initializing Network...")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for YOLO; GPU not available.")

        device = torch.device("cuda")
        self._model = YOLO(self.weights)
        self.class_names = getattr(self._model, 'names', {})
        print(f"Network Initialized... classes: {self.class_names}")

        while not self.exit_signal:
            if self.run_event.is_set():
                with self.lock:
                    img = cv2.cvtColor(self.image_net, cv2.COLOR_RGBA2RGB)

                t0 = time()
                det = self._model.predict(
                    img,
                    save=False,
                    retina_masks=True,
                    imgsz=self.img_size,
                    conf=self.conf_thres,
                    device=device,
                    verbose=False,
                )[0]
                dt = time() - t0
                self.net_fps = (1.0 / dt) if dt > 0 else 0.0

                with self.lock:
                    self.detections = detections_to_custom_masks(det)

                self.run_event.clear()
                self.dets_ready.set()

            sleep(0.005)

    def set_image(self, image: np.ndarray) -> None:
        """Set new image for inference."""
        with self.lock:
            self.image_net = image
            self.run_event.set()

    def get_detections(self):
        """Get latest detections (thread-safe)."""
        with self.lock:
            return self.detections

    def stop(self) -> None:
        """Signal thread to stop."""
        self.exit_signal = True
