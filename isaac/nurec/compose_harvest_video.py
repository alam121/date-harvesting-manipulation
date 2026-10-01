"""Compose the harvest video: overview camera | wrist camera with live date detections.

Input: the frames and events.jsonl written by date_pick_test.py --record DIR. The date detector
runs on every wrist frame (as live_detector.py does) and a banner shows the current action and
how many dates have been picked.

  ~/nurec_data/yolo_venv/bin/python isaac/nurec/compose_harvest_video.py DIR OUT.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from perception_check import COLORS, DEFAULT_WEIGHTS, detect

PANEL_W, PANEL_H, BANNER_H = 960, 540, 56


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def label(image, text, origin, scale=0.8, colour=(255, 255, 255)):
    cv2.putText(image, text, origin, 0, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, 0, scale, colour, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    model = YOLO(str(args.weights))
    events = [json.loads(line) for line in (args.frames / "events.jsonl").read_text().splitlines() if line.strip()]
    overview_frames = sorted(args.frames.glob("overview_*.jpg"))
    if not overview_frames:
        raise SystemExit(f"no frames in {args.frames}")
    # Some recordings render the photo-coloured date/strand meshes solid black (intermittent RTX
    # material issue, cause not pinned down); re-recording has fixed it every time.
    samples = [cv2.imread(str(args.frames / p.name.replace("overview_", "wrist_"))) for p in overview_frames[::100]]
    black = float(np.mean([(s.max(2) < 20).mean() for s in samples])) * 100
    if black > 1.0:
        print(f"[VIDEO] WARNING: {black:.1f}% of wrist pixels are black - meshes likely rendered black; re-record.")

    encoder = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{2 * PANEL_W}x{PANEL_H + BANNER_H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE,
    )
    event_index, current = 0, {"text": "starting", "picked": 0, "total": events[0]["total"] if events else 0}
    detections_total = 0
    for number, overview_path in enumerate(overview_frames):
        while event_index < len(events) and events[event_index]["frame"] <= number:
            current = events[event_index]
            event_index += 1
        overview = cv2.resize(cv2.imread(str(overview_path)), (PANEL_W, PANEL_H))
        wrist = cv2.imread(str(args.frames / overview_path.name.replace("overview_", "wrist_")))
        found = detect(model, wrist, args.conf)
        detections_total += len(found)
        for cls, score, box in found:
            x0, y0, x1, y1 = box.astype(int)
            colour = COLORS[cls % len(COLORS)]
            cv2.rectangle(wrist, (x0, y0), (x1, y1), colour, 2)
            label(wrist, f"{model.names[cls]} {score:.2f}", (x0, max(14, y0 - 5)), 0.45, colour)
        wrist = cv2.resize(wrist, (PANEL_W, PANEL_H))
        label(overview, "Isaac Sim - NuRec palm3, UR10e + Delto hand", (14, 32), 0.7)
        label(wrist, "wrist camera - date detector", (14, 32), 0.7)

        banner = np.full((BANNER_H, 2 * PANEL_W, 3), 28, np.uint8)
        label(banner, current["text"], (16, 37), 0.8, (120, 230, 255))
        label(banner, f"picked {current['picked']} / {current['total']}", (2 * PANEL_W - 250, 37), 0.9, (120, 255, 120))
        encoder.stdin.write(np.vstack([banner, np.hstack([overview, wrist])]).tobytes())
    encoder.stdin.close()
    encoder.wait()
    seconds = len(overview_frames) / args.fps
    print(f"[VIDEO] {len(overview_frames)} frames ({seconds:.0f} s), {detections_total} detections -> {args.out}")


if __name__ == "__main__":
    main()
