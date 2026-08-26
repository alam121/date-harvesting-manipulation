"""YOLO inference thread for background processing."""

import json
import os
from collections import deque
from threading import Lock, Event
from time import time
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .detection import detections_to_custom_masks
from .direct_trt_decoder import DirectTensorRTSegmenter


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

    def __init__(self, weights: str, img_size=640, conf_thres: float = 0.35,
                 raw_view: bool = False, use_numpy_masks: bool = False):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.raw_view = raw_view
        self.use_numpy_masks = bool(use_numpy_masks)

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
        self.raw_viz = []            # unfiltered YOLO boxes/classes for model inspection
        self.raw_result = None        # exact source frame + timings for raw inspection
        self.net_fps = 0.0
        self._next_frame_id = 0
        self._pending_frame_id = -1
        self._pending_capture_time = 0.0
        self._dropped_frames = 0
        self._profile_count = 0
        self._last_profile_dropped = 0
        self._recent_dropped_interval = 0
        self._fps_window = deque(maxlen=20)

        self._model: Optional[YOLO] = None
        self._direct_model = None
        self._direct_decoder_enabled = False
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

        use_direct = (
            str(self.weights).endswith(".engine") and
            os.getenv("UR10E_DIRECT_TRT_DECODER", "1").strip().lower()
            not in {"0", "false", "no", "off"})
        # Read class names from the engine metadata before choosing a backend.
        engine_names = {}
        if str(self.weights).endswith(".engine"):
            try:
                with open(self.weights, "rb") as fh:
                    meta_len = int.from_bytes(fh.read(4), "little")
                    metadata = json.loads(fh.read(meta_len).decode("utf-8"))
                engine_names = {
                    int(key): value for key, value in metadata.get("names", {}).items()}
            except Exception:
                engine_names = {}
        if use_direct:
            self.class_names = engine_names
        else:
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
        fruit_class_ids = [
            cid for cid, name in self.class_names.items()
            if name.lower() not in skip_names | {"trunk", "bunch"}
        ]
        # Harvesting and raw inspection both operate on the three selected
        # dates only. Trunk/bunch obstacle processing has been removed from this
        # workflow, so do not spend inference/postprocessing time on those masks.
        self._detect_class_ids = fruit_class_ids
        if use_direct:
            try:
                self._direct_model = DirectTensorRTSegmenter(
                    self.weights, self.img_size, self.class_names, device,
                    self.conf_thres, self._detect_class_ids, 3)
                self._direct_decoder_enabled = True
                print("[YoloThread] Direct TensorRT decoder ENABLED "
                      "(set UR10E_DIRECT_TRT_DECODER=0 for Ultralytics fallback).")
            except Exception as exc:
                print(f"[YoloThread] Direct decoder initialization failed: {exc}; "
                      "falling back to Ultralytics.")
                self._model = YOLO(self.weights, task='segment')
                self.class_names = getattr(self._model, 'names', self.class_names)
        print(f"Network Initialized... classes: {self.class_names}")
        print(f"  Detect class IDs: {self._detect_class_ids} (skipping: {skip_names})")
        print(f"  Trunk class IDs: {self._trunk_class_ids}")
        print(f"  Bunch class IDs: {self._bunch_class_ids}")

        while not self.exit_signal:
            if not self.run_event.wait(timeout=0.1):
                continue

            self.idle_event.clear()
            with self.lock:
                source_image = self.image_net
                frame_id = self._pending_frame_id
                capture_time = self._pending_capture_time

            preprocess_start = time()
            img = cv2.cvtColor(source_image, cv2.COLOR_RGBA2RGB)
            predict_start = time()

            t0 = time()
            with self.inference_lock:
                if self._direct_decoder_enabled:
                    try:
                        det = self._direct_model.predict(img)
                    except Exception as exc:
                        print(f"[YoloThread] Direct decode failed: {exc}; "
                              "switching permanently to Ultralytics fallback.")
                        self._direct_decoder_enabled = False
                        self._model = YOLO(self.weights, task='segment')
                        det = self._model.predict(
                            img, save=False, retina_masks=False,
                            imgsz=self.img_size, conf=self.conf_thres, iou=0.3,
                            max_det=3, device=device, verbose=False,
                            classes=self._detect_class_ids or None)[0]
                else:
                    det = self._model.predict(
                         img,
                         save=False,
                         retina_masks=False,
                         imgsz=self.img_size,
                         conf=self.conf_thres,
                         iou=0.3,
                         max_det=3,
                         device=device,
                         verbose=False,
                         classes=self._detect_class_ids if self._detect_class_ids else None,
                     )[0]

            dt = time() - t0
            predict_end = time()
            if dt > 0:
                self._fps_window.append(1.0 / dt)
            self.net_fps = (
                sum(self._fps_window) / len(self._fps_window)
                if self._fps_window else 0.0)
            self._log_inference_time(dt)

            decode_start = time()
            fruit_dets, trunk_boxes, bunch_boxes, raw_viz = (
                detections_to_custom_masks(
                    det,
                    trunk_class_ids=self._trunk_class_ids,
                    bunch_class_ids=self._bunch_class_ids,
                    class_names=self.class_names,
                    build_custom_masks=not self.raw_view,
                    use_numpy_masks=self.use_numpy_masks and not self.raw_view,
                ))
            if self.raw_view:
                raw_viz = raw_viz[:3]
            decode_end = time()
            speed = getattr(det, "speed", {}) or {}
            timing = {
                "frame_id": frame_id,
                "capture_time": capture_time,
                "capture_to_start_ms": max(0.0, (preprocess_start - capture_time) * 1000.0),
                "color_ms": (predict_start - preprocess_start) * 1000.0,
                "predict_wall_ms": (predict_end - predict_start) * 1000.0,
                "preprocess_ms": float(speed.get("preprocess", 0.0)),
                "inference_ms": float(speed.get("inference", 0.0)),
                "postprocess_ms": float(speed.get("postprocess", 0.0)),
                "decode_mask_ms": (decode_end - decode_start) * 1000.0,
                "ready_time": decode_end,
                "dropped": self._dropped_frames,
                "dropped_interval": self._recent_dropped_interval,
            }
            with self.lock:
                self.detections = fruit_dets
                self.trunk_boxes = trunk_boxes
                self.bunch_boxes = bunch_boxes
                self.raw_viz = raw_viz
                self.raw_result = {
                    "image": source_image,
                    "detections": raw_viz,
                    "timing": timing,
                }

            self._profile_count += 1
            if self._profile_count % 30 == 0:
                dropped_interval = (
                    self._dropped_frames - self._last_profile_dropped)
                self._last_profile_dropped = self._dropped_frames
                self._recent_dropped_interval = dropped_interval
                timing["dropped_interval"] = dropped_interval
                if self.raw_view:
                    print(
                        "[RAW_TIMING] "
                        f"id={frame_id} queue={timing['capture_to_start_ms']:.1f}ms "
                        f"color={timing['color_ms']:.1f}ms "
                        f"pre={timing['preprocess_ms']:.1f}ms "
                        f"infer={timing['inference_ms']:.1f}ms "
                        f"post={timing['postprocess_ms']:.1f}ms "
                        f"mask_cpu={timing['decode_mask_ms']:.1f}ms "
                        f"predict_wall={timing['predict_wall_ms']:.1f}ms "
                        f"fps_avg={self.net_fps:.1f} "
                        f"dropped_30={dropped_interval}"
                    )
                else:
                    print(
                        "[YOLO_TIMING] "
                        f"id={frame_id} queue={timing['capture_to_start_ms']:.1f}ms "
                        f"color={timing['color_ms']:.1f}ms "
                        f"pre={timing['preprocess_ms']:.1f}ms "
                        f"infer={timing['inference_ms']:.1f}ms "
                        f"post={timing['postprocess_ms']:.1f}ms "
                        f"mask_cpu={timing['decode_mask_ms']:.1f}ms "
                        f"predict_wall={timing['predict_wall_ms']:.1f}ms "
                        f"fps_avg={self.net_fps:.1f} "
                        f"dropped_30={dropped_interval}"
                    )

            self.run_event.clear()
            self.dets_ready.set()
            self.idle_event.set()

        self.stopped.set()  # signal that the loop has fully exited

    def set_image(self, image: np.ndarray, capture_time: Optional[float] = None) -> None:
        """Set new image for inference. Dropped silently when paused."""
        if self.paused or self.run_event.is_set():
            self._dropped_frames += 1
            return
        with self.lock:
            # ZED reuses its SDK image buffer on the next grab. Own this frame
            # so inference never observes a buffer being rewritten underneath it.
            self.image_net = image.copy()
            self._pending_frame_id = self._next_frame_id
            self._next_frame_id += 1
            self._pending_capture_time = capture_time if capture_time is not None else time()
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

    def get_raw_viz(self):
        """Return raw YOLO detections before depth/scoring/tracking filters."""
        with self.lock:
            return list(self.raw_viz)

    def get_raw_result(self):
        """Return detections with the exact immutable frame used for inference."""
        with self.lock:
            if self.raw_result is None:
                return None
            return {
                "image": self.raw_result["image"],
                "detections": list(self.raw_result["detections"]),
                "timing": dict(self.raw_result["timing"]),
            }

    def get_latest_timing(self):
        """Return timing for the latest completed inference without its image."""
        with self.lock:
            if self.raw_result is None:
                return None
            return dict(self.raw_result["timing"])

    def stop(self) -> None:
        """Signal thread to stop."""
        self.exit_signal = True
