"""Sim twin of the robot's wrist date detector, for the Isaac hardware mirror (full harvest cycle).

isaac_hardware_scene.py (NuRec mode) writes the newest wrist RGB-D frame and the camera pose in
base_link to /dev/shm. This runs the robot's YOLO segmentation model on it, turns every date mask
into the point the real vision node publishes (front-surface depth, median pixel of the front
points; ur10e_curobo/vision/node.py _extract_target_3d), finds the trunk, and sends the result back
to Isaac over UDP. Isaac forwards it to the ROS bridge, which publishes /vision/all_fruit_poses,
/external_goal_pose, /fruit_radius and /trunk_position for the cuRobo control node on the Jetson.

  ~/nurec_data/yolo_venv/bin/python isaac/nurec/sim_vision.py [--show]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from perception_check import COLORS, DEFAULT_WEIGHTS

FRAME_PATH = Path("/dev/shm/isaac_wrist_frame.npz")
# ur10e_curobo/vision/config.py
FRUIT_RADIUS_DEFAULT, FRUIT_RADIUS_MIN, FRUIT_RADIUS_MAX = 0.035, 0.020, 0.060
DEPTH_MAX = 7.0
FRONT_BAND = 0.03  # front points: within 3 cm of the nearest (5th percentile) in-mask depth
TRACK_RADIUS, TRACK_KEEP_S = 0.05, 2.0


class Tracks:
    """Dates seen in the last TRACK_KEEP_S, merged within TRACK_RADIUS (detections flicker frame to
    frame; the real node tracks too, vision/tracking.py). Position follows the latest sighting."""

    def __init__(self):
        self.items: list[dict] = []

    def update(self, fruits: list[list[float]], now: float) -> list[list[float]]:
        for x, y, z, score, radius in fruits:
            xyz = np.array([x, y, z])
            near = [t for t in self.items if np.linalg.norm(t["xyz"] - xyz) < TRACK_RADIUS]
            if near:
                track = min(near, key=lambda t: np.linalg.norm(t["xyz"] - xyz))
                track.update(xyz=xyz, score=max(score, 0.8 * track["score"]), radius=radius, seen=now)
            else:
                self.items.append({"xyz": xyz, "score": score, "radius": radius, "seen": now})
        self.items = [t for t in self.items if now - t["seen"] < TRACK_KEEP_S]
        ranked = sorted(self.items, key=lambda t: -t["score"])
        return [[*t["xyz"].tolist(), t["score"], t["radius"]] for t in ranked]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--isaac_port", type=int, default=55002, help="Isaac's detection input port.")
    parser.add_argument("--show", action="store_true", help="Window with the detections.")
    return parser.parse_args()


def quat_to_matrix(w, x, y, z) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def upright_turn(R_bc: np.ndarray) -> int:
    """Quarter turns (cv2 rotate code or -1) that bring world-down to image-down.

    The detector was trained on upright images; at the lab home the wrist camera is rolled ~90 deg
    and finds nothing. The real vision node does not rotate (see the README).
    """
    down = R_bc.T @ np.array([0.0, 0.0, -1.0])  # world down in the optical frame (y = image down)
    quarter = int(np.round(np.degrees(np.arctan2(down[0], down[1])) / 90.0)) % 4
    return {0: -1, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[quarter]


def unrotate(points: np.ndarray, turn: int, w: int, h: int) -> np.ndarray:
    """Pixel coordinates in the turned image -> original image (w x h)."""
    u, v = points[:, 0], points[:, 1]
    if turn == cv2.ROTATE_90_CLOCKWISE:
        return np.stack([v, h - 1 - u], 1)
    if turn == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return np.stack([w - 1 - v, u], 1)
    if turn == cv2.ROTATE_180:
        return np.stack([w - 1 - u, h - 1 - v], 1)
    return points


def front_point(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, stats: dict | None = None):
    """Camera-frame front-surface point of a mask, or None (too few valid depth pixels).

    stats, if given, receives the real vision node's quality inputs: vis_ratio (share of the mask
    with valid depth) and z_std (depth scatter of the front points, metres).
    """
    v, u = np.nonzero(mask)
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0.05) & (z < DEPTH_MAX)
    if valid.sum() < 15:
        return None
    u, v, z = u[valid], v[valid], z[valid]
    front = z <= np.percentile(z, 5) + FRONT_BAND
    zc = float(np.median(z[front]))
    uc, vc = float(np.median(u[front])), float(np.median(v[front]))
    if stats is not None:
        stats["vis_ratio"] = float(valid.sum()) / max(len(valid), 1)
        stats["z_std"] = float(np.std(z[front]))
    return np.array([(uc - K[0, 2]) * zc / K[0, 0], (vc - K[1, 2]) * zc / K[1, 1], zc])


def main():
    args = parse_args()
    model = YOLO(str(args.weights))
    names = model.names
    date_class = next(k for k, n in names.items() if n.startswith("date"))
    trunk_class = next((k for k, n in names.items() if n == "trunk"), None)
    bunch_class = next((k for k, n in names.items() if n == "bunch"), None)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    last_mtime, frames, started = 0.0, 0, time.time()
    tracks = Tracks()
    print(f"[SIM VISION] {args.weights.name}: waiting for {FRAME_PATH}", flush=True)
    while True:
        try:
            mtime = FRAME_PATH.stat().st_mtime
        except FileNotFoundError:
            time.sleep(0.2)
            continue
        if mtime == last_mtime:
            time.sleep(0.01)
            continue
        last_mtime = mtime
        try:
            frame = np.load(FRAME_PATH)
            rgb, depth, K = frame["rgb"], frame["depth"].astype(np.float32), frame["K"]
            R_bc, t_bc = quat_to_matrix(*frame["cam_quat_b"]), frame["cam_pos_b"]
            frame_id = int(frame["frame_id"])
        except (OSError, ValueError, KeyError, EOFError):
            continue  # caught mid-replace; take the next one
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = depth.shape
        turn = upright_turn(R_bc)
        result = model.predict(image if turn < 0 else cv2.rotate(image, turn), conf=args.conf, verbose=False)[0]

        fruits, trunk, bunch_box, best_box = [], None, None, None
        observed = []  # this frame only: [x, y, z, vis_ratio, z_std, confidence, score, edge_margin_norm]
        polygons = [unrotate(p, turn, w, h) for p in result.masks.xy] if result.masks is not None \
            else [None] * len(result.boxes)
        boxes = []
        for box in result.boxes.xyxy.cpu().numpy():
            corners = unrotate(np.array([box[:2], box[2:]]), turn, w, h)
            boxes.append(np.concatenate([corners.min(0), corners.max(0)]))
        order = np.argsort(-result.boxes.conf.cpu().numpy())
        for i in order:
            cls = int(result.boxes.cls[i])
            score = float(result.boxes.conf[i])
            box = boxes[i]
            mask = np.zeros((h, w), np.uint8)
            if polygons[i] is not None and len(polygons[i]) >= 3:
                cv2.fillPoly(mask, [polygons[i].astype(np.int32)], 1)
            else:
                x0, y0, x1, y1 = box.astype(int)
                mask[y0:y1, x0:x1] = 1
            if cls == bunch_class and bunch_box is None:
                bunch_box = box
            stats = {}
            point = front_point(mask > 0, depth, K, stats) if cls in (date_class, trunk_class) else None
            if point is None:
                continue
            base = R_bc @ point + t_bc
            size_m = float(min(box[2] - box[0], box[3] - box[1])) * point[2] / K[0, 0]
            if cls == date_class:
                radius = float(np.clip(size_m / 2.0, FRUIT_RADIUS_MIN, FRUIT_RADIUS_MAX))
                fruits.append([*base.tolist(), score, radius])
                edge = min(box[0], box[1], w - box[2], h - box[3])
                edge_norm = float(np.clip(edge / max(1.0, min(w, h) / 2.0), 0.0, 1.0))
                observed.append([*base.tolist(), stats["vis_ratio"], stats["z_std"], score, score, edge_norm])
                best_box = box if best_box is None else best_box
            elif cls == trunk_class and trunk is None:
                # Front surface -> axis: push back by the trunk's apparent radius along the ray.
                axis_cam = point + (size_m / 2.0) * point / np.linalg.norm(point)
                trunk = (R_bc @ axis_cam + t_bc).tolist()

        bbox = [0.5, 0.5, 0.5, 0.5]
        if best_box is not None:
            cx, cy = (best_box[0] + best_box[2]) / 2.0, (best_box[1] + best_box[3]) / 2.0
            bbox = [cx / w, cy / h, 0.5, 0.5]
            if bunch_box is not None:
                bw, bh = max(bunch_box[2] - bunch_box[0], 1.0), max(bunch_box[3] - bunch_box[1], 1.0)
                bbox[2:] = [float(np.clip((cx - bunch_box[0]) / bw, 0, 1)), float(np.clip((cy - bunch_box[1]) / bh, 0, 1))]
        detected = len(fruits)
        fruits = tracks.update(fruits, time.time())
        payload = {"frame_id": frame_id, "fruits": fruits, "observed": observed, "trunk": trunk, "bbox": [float(v) for v in bbox],
                   "reason": f"dates={len(fruits)} (in frame {detected}) trunk={'yes' if trunk else 'no'}"}
        sock.sendto(json.dumps(payload).encode(), ("127.0.0.1", args.isaac_port))

        frames += 1
        if frames % 50 == 0:
            print(f"[SIM VISION] {frames / (time.time() - started):.1f} Hz, {payload['reason']}", flush=True)
        if args.show:
            for i in order:
                cls, score = int(result.boxes.cls[i]), float(result.boxes.conf[i])
                x0, y0, x1, y1 = boxes[i].astype(int)
                colour = COLORS[cls % len(COLORS)]
                cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2)
                cv2.putText(image, f"{names[cls]} {score:.2f}", (x0, max(12, y0 - 4)), 0, 0.45, colour, 1, cv2.LINE_AA)
            cv2.imshow("sim wrist camera - date detector", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    os.environ.setdefault("YOLO_VERBOSE", "False")
    main()
