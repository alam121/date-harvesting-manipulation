"""Run the date detector live on the simulated wrist camera streamed by isaac_hardware_scene.py.

  bin/launch_isaac_gpu --nurec_usdz SCENE/scene_nurec.usdz --nurec_pose ... --wrist_stream_port 55010
  ~/nurec_data/yolo_venv/bin/python isaac/nurec/live_detector.py --port 55010

Drive the sim arm yourself with --jog (do not run replay_bag_joints.py at the same time):

  ... live_detector.py --port 55010 --jog --start_joints J1 .. J6   (from replay_bag_joints.py --print_start)

Keys in the window: a/z s/x d/c f/v g/b h/n jog joints 1-6 (+/-), ]/[ step size, r back to start,
q quit. --record OUT.mp4 also saves the annotated stream.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from perception_check import COLORS, DEFAULT_WEIGHTS, detect

WINDOW = "sim wrist camera - date detector"
JOG_KEYS = {key: (joint, sign) for joint, (plus, minus) in enumerate(["az", "sx", "dc", "fv", "gb", "hn"])
            for key, sign in ((plus, 1.0), (minus, -1.0))}
JOINT_LIMITS = [2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=55010)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--jog", action="store_true", help="Drive the sim arm from this window.")
    parser.add_argument("--start_joints", type=float, nargs=6, default=None)
    parser.add_argument("--command_port", type=int, default=55000, help="isaac_hardware_scene.py --command_port")
    return parser.parse_args()


class Jogger:
    """Sends arm joint states to the scene in "state" mode, like the digital-twin bridge."""

    def __init__(self, start: list[float], port: int):
        self.start = np.array(start, dtype=float)
        self.joints = self.start.copy()
        self.step = math.radians(1.0)
        self.address = ("127.0.0.1", port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_send = 0.0

    def handle(self, key: str) -> None:
        if key in JOG_KEYS:
            joint, sign = JOG_KEYS[key]
            limit = JOINT_LIMITS[joint]
            self.joints[joint] = float(np.clip(self.joints[joint] + sign * self.step, -limit, limit))
        elif key == "]":
            self.step = min(self.step * 2, math.radians(8.0))
        elif key == "[":
            self.step = max(self.step / 2, math.radians(0.125))
        elif key == "r":
            self.joints = self.start.copy()

    def send(self) -> None:
        # Resend continuously: the scene treats a silent sender as "hold last state".
        now = time.monotonic()
        if now - self.last_send < 0.03:
            return
        self.last_send = now
        self.socket.sendto(json.dumps({
            "arm_target": self.joints.tolist(),
            "arm_velocity": [0.0] * 6,
            "arm_mode": "state",
            "gripper_target": [],
        }).encode(), self.address)

    def overlay(self, frame: np.ndarray) -> None:
        text = "  ".join(f"j{i + 1} {math.degrees(v):6.1f}" for i, v in enumerate(self.joints))
        for y, line in ((frame.shape[0] - 30, text),
                        (frame.shape[0] - 10, f"step {math.degrees(self.step):.2f} deg   a/z s/x d/c f/v g/b h/n  [ ]  r  q")):
            cv2.putText(frame, line, (10, y), 0, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def latest_frame(sock: socket.socket):
    """Drain the socket and decode only the newest datagram, so detection never lags the sim."""
    packet = None
    while True:
        try:
            packet = sock.recv(65535)
        except BlockingIOError:
            break
    if packet is None:
        return None
    return cv2.imdecode(np.frombuffer(packet, np.uint8), cv2.IMREAD_COLOR)


def main():
    args = parse_args()
    if args.jog and args.start_joints is None:
        raise SystemExit("--jog needs --start_joints (replay_bag_joints.py BAG --print_start)")
    model = YOLO(str(args.weights))
    names = model.names
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)
    jogger = Jogger(args.start_joints, args.command_port) if args.jog else None
    writer = None
    print(f"[INFO] waiting for wrist frames on udp://127.0.0.1:{args.port} (q to quit)")

    frame_times = []
    last_frame = time.monotonic()
    while True:
        if jogger is not None:
            jogger.send()
        frame = latest_frame(sock)
        if frame is None:
            if time.monotonic() - last_frame > 2.0:
                idle = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(idle, "waiting for Isaac wrist stream...", (40, 180), 0, 0.8, (200, 200, 200), 2)
                cv2.imshow(WINDOW, idle)
            key = cv2.waitKey(5) & 0xFF
        else:
            last_frame = time.monotonic()
            frame_times = [t for t in frame_times if last_frame - t < 2.0] + [last_frame]

            for cls, score, box in detect(model, frame, args.conf):
                x0, y0, x1, y1 = box.astype(int)
                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, f"{names[cls]} {score:.2f}", (x0, max(15, y0 - 5)), 0, 0.5, color, 1, cv2.LINE_AA)
            fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
            cv2.putText(frame, f"Isaac NuRec wrist cam  {fps:4.1f} fps", (10, 24), 0, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if jogger is not None:
                jogger.overlay(frame)

            if args.record is not None:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame.shape[1], frame.shape[0])
                    )
                writer.write(frame)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if jogger is not None and key != 255:
            jogger.handle(chr(key))

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
