#!/usr/bin/env python3
"""Plot GPU-handoff benchmark results into presentation-ready PNGs.

Reads the two CSVs produced by the benchmark run:
  - YOLO inference timing   (UR10E_YOLO_TIMING_CSV) : t_rel_s, inference_ms
  - handshake latency       (bench_gpu_handshake --out): cycle, acked, latency_ms

Produces:
  - yolo_inference_scatter.png : per-frame inference time vs the old 120 ms line
                                 (the money shot — every point above the line is
                                  a frame the fixed sleep would have overlapped)
  - yolo_inference_hist.png    : inference-time distribution
  - handshake_latency_hist.png : drain-and-ack latency distribution

Usage:
    python3 benchmarks/plot_handshake_results.py \
        --yolo /tmp/yolo_timing.csv \
        --handshake /tmp/handshake_latency.csv \
        --outdir /tmp/handshake_plots
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # headless — no display needed on the Jetson
import matplotlib.pyplot as plt

OLD_SLEEP_MS = 120.0


def _read(path, col):
    vals = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                vals.append(float(row[col]))
            except (KeyError, ValueError):
                pass
    return vals


def _read2(path, xcol, ycol):
    xs, ys = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                xs.append(float(row[xcol]))
                ys.append(float(row[ycol]))
            except (KeyError, ValueError):
                pass
    return xs, ys


def plot_yolo_scatter(path, outdir):
    ts, ms = _read2(path, "t_rel_s", "inference_ms")
    if not ms:
        print(f"[skip] no YOLO rows in {path}")
        return
    over = sum(1 for x in ms if x > OLD_SLEEP_MS)
    colors = ["#d1495b" if x > OLD_SLEEP_MS else "#2e86ab" for x in ms]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(ts, ms, c=colors, s=14, alpha=0.8)
    ax.axhline(OLD_SLEEP_MS, color="#e07a00", ls="--", lw=2,
               label=f"old fixed sleep = {OLD_SLEEP_MS:.0f} ms")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("YOLO inference time (ms)")
    ax.set_title(f"YOLO inference time vs fixed-sleep budget\n"
                 f"{over}/{len(ms)} frames exceed 120 ms "
                 f"(old sleep would have overlapped cuRobo)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(outdir, "yolo_inference_scatter.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}  ({over}/{len(ms)} frames > 120 ms)")


def plot_hist(vals, title, xlabel, out, ref=None):
    if not vals:
        print(f"[skip] no data for {out}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=30, color="#2e86ab", alpha=0.85, edgecolor="white")
    if ref is not None:
        ax.axvline(ref, color="#e07a00", ls="--", lw=2, label=f"{ref:.0f} ms")
        ax.legend()
    mean = sum(vals) / len(vals)
    ax.axvline(mean, color="#111", ls=":", lw=1.5, label=f"mean {mean:.1f} ms")
    ax.set_xlabel(xlabel); ax.set_ylabel("count"); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[ok] {out}  (n={len(vals)}, mean={mean:.1f} ms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yolo", type=str, default="/tmp/yolo_timing.csv")
    ap.add_argument("--handshake", type=str, default="/tmp/handshake_latency.csv")
    ap.add_argument("--outdir", type=str, default="/tmp/handshake_plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if os.path.exists(args.yolo):
        plot_yolo_scatter(args.yolo, args.outdir)
        plot_hist(_read(args.yolo, "inference_ms"),
                  "YOLO inference-time distribution", "inference time (ms)",
                  os.path.join(args.outdir, "yolo_inference_hist.png"),
                  ref=OLD_SLEEP_MS)
    else:
        print(f"[skip] YOLO CSV not found: {args.yolo}")

    if os.path.exists(args.handshake):
        plot_hist(_read(args.handshake, "latency_ms"),
                  "Handshake drain-and-ack latency", "latency (ms)",
                  os.path.join(args.outdir, "handshake_latency_hist.png"),
                  ref=OLD_SLEEP_MS)
    else:
        print(f"[skip] handshake CSV not found: {args.handshake}")


if __name__ == "__main__":
    main()
