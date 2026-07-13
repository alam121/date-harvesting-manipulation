#!/usr/bin/env python3
"""Measure YOLO inference time per frame over a ZED X One (mono) SVO recording.

The vision node only wires --svo into the stereo Camera path, so ZED X One
(QHDPLUS, CameraOne API) recordings cannot be replayed through it. This
standalone timer opens such an SVO directly and runs the *same* YOLO engine and
inference parameters the pipeline uses, so the inference-time distribution is
representative of the live detector.

Output CSV matches the UR10E_YOLO_TIMING_CSV format (t_rel_s, inference_ms) so
plot_handshake_results.py can consume it directly.

    python3 benchmarks/bench_svo_yolo_timing.py \
        --svo /home/datepalm2/Documents/ZED/QHD+_SN318493032_13-50-07.svo2 \
        --out /tmp/yolo_timing.csv --max-frames 600
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from the package root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ur10e_curobo.vision.config import (  # noqa: E402
    DEFAULT_WEIGHTS, DEFAULT_IMG_SIZE, DEFAULT_CONF_THRES,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svo", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF_THRES)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="/tmp/yolo_timing.csv")
    ap.add_argument("--gpu-contention", action="store_true",
                    help="run a background GPU load to emulate concurrent cuRobo "
                         "planning — reproduces the contention the handshake guards against")
    args = ap.parse_args()

    import cv2
    import torch
    import pyzed.sl as sl
    from ultralytics import YOLO

    # Optional background GPU load: continuous large matmuls on CUDA to contend
    # for the GPU the way cuRobo motion planning would during a harvest cycle.
    contention_stop = None
    if args.gpu_contention:
        import threading
        contention_stop = threading.Event()

        def _hog():
            a = torch.randn(4096, 4096, device="cuda")
            b = torch.randn(4096, 4096, device="cuda")
            while not contention_stop.is_set():
                (a @ b).sum().item()  # force sync so the GPU stays busy

        threading.Thread(target=_hog, daemon=True).start()
        print("GPU contention load ON (emulating concurrent cuRobo)")

    zed = sl.CameraOne()
    ip = sl.InitParametersOne()
    ip.set_from_svo_file(args.svo)
    ip.svo_real_time_mode = False   # run as fast as inference allows
    ip.coordinate_units = sl.UNIT.METER
    if zed.open(ip) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open SVO (is it a ZED X One recording?)")
        return
    total = zed.get_svo_number_of_frames()
    print(f"SVO opened: {total} frames")

    device = torch.device("cuda")
    model = YOLO(args.weights, task="segment")
    predict = lambda f: model.predict(
        f, save=False, retina_masks=False, imgsz=args.img_size, conf=args.conf,
        iou=0.3, max_det=10, device=device, verbose=False)[0]

    mat = sl.Mat()
    rows = []          # (t_rel_s, inference_ms)
    times = []         # inference_ms only
    t_start = None
    grabbed = 0
    limit = min(args.max_frames, total) if args.max_frames > 0 else total

    while grabbed < limit + args.warmup:
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            break
        zed.retrieve_image(mat, sl.VIEW.LEFT)
        frame = cv2.cvtColor(mat.get_data(), cv2.COLOR_RGBA2RGB)

        t0 = time.time()
        r = predict(frame)
        dt_ms = (time.time() - t0) * 1000.0
        grabbed += 1

        if grabbed <= args.warmup:
            continue  # discard TRT warm-up frames
        if t_start is None:
            t_start = time.time()
        rows.append((time.time() - t_start, dt_ms))
        times.append(dt_ms)
        if len(times) % 50 == 0:
            print(f"  {len(times):4d}/{limit}  last={dt_ms:6.1f} ms")

    zed.close()
    if contention_stop is not None:
        contention_stop.set()

    with open(args.out, "w") as fh:
        fh.write("t_rel_s,inference_ms\n")
        for t, ms in rows:
            fh.write(f"{t:.4f},{ms:.3f}\n")

    if not times:
        print("No frames timed.")
        return
    s = sorted(times)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    over = sum(1 for x in times if x > 120.0)
    print("\n===== YOLO INFERENCE TIMING (SVO) =====")
    print(f"frames timed : {len(times)} (warmup {args.warmup} discarded)")
    print(f"mean         : {sum(times)/len(times):.2f} ms")
    print(f"median       : {p(0.50):.2f} ms")
    print(f"p95          : {p(0.95):.2f} ms")
    print(f"max          : {max(times):.2f} ms")
    print(f"> 120 ms     : {over} / {len(times)}  "
          f"(frames the old fixed sleep would have overlapped)")
    print(f"CSV -> {args.out}")
    print("=======================================\n")


if __name__ == "__main__":
    main()
