#!/usr/bin/env python3
"""Create a scientific results graph for GPU contention and loop response."""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ur_ws_new/src/ur10e_curobo/benchmarks/results"
OUT = ROOT / "presentation_output"


def load(name):
    with (RESULTS / name).open() as f:
        rows = list(csv.DictReader(f))
    return (
        np.array([float(r["t_rel_s"]) for r in rows]),
        np.array([float(r["inference_ms"]) for r in rows]),
    )


idle_t, idle_ms = load("yolo_timing_idle.csv")
load_t, load_ms = load("yolo_timing_contended.csv")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#526777",
    "axes.labelcolor": "#253746",
    "xtick.color": "#526777",
    "ytick.color": "#526777",
})

fig = plt.figure(figsize=(16, 8.6), facecolor="#f5f6f1")
grid = fig.add_gridspec(2, 2, height_ratios=[2.4, 1], width_ratios=[2.25, 1],
                        hspace=.34, wspace=.24)

# A: Measured response over time
ax = fig.add_subplot(grid[0, 0])
ax.set_facecolor("white")
ax.scatter(idle_t, idle_ms, s=13, alpha=.58, color="#26a6b2",
           edgecolors="none", label=f"GPU free (n={len(idle_ms)})")
ax.scatter(load_t, load_ms, s=16, alpha=.48, color="#ed8b3a",
           edgecolors="none", label=f"GPU contended (n={len(load_ms)})")
ax.axhline(120, color="#c74747", lw=2.2, ls="--",
           label="Old fixed wait: 120 ms")
ax.set_title("A. YOLO inference response over time", loc="left", pad=12)
ax.set_xlabel("Elapsed experiment time (s)")
ax.set_ylabel("Inference response time (ms)")
ax.set_ylim(0, 900)
ax.grid(True, axis="y", alpha=.18)
ax.legend(frameon=False, loc="upper right")
ax.text(.02, .93,
        "GPU contention produces a sustained latency increase;\n"
        "the fixed 120 ms wait is below every contended sample.",
        transform=ax.transAxes, va="top", color="#253746",
        bbox=dict(boxstyle="round,pad=.55", fc="#eef4f5", ec="none"))

# B: Distribution summary
ax = fig.add_subplot(grid[0, 1])
ax.set_facecolor("white")
parts = ax.violinplot([idle_ms, load_ms], positions=[1, 2],
                      showmeans=False, showmedians=False, showextrema=False,
                      widths=.72)
for body, color in zip(parts["bodies"], ["#26a6b2", "#ed8b3a"]):
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(.65)

for xpos, values, color in [(1, idle_ms, "#137f89"), (2, load_ms, "#c96d24")]:
    median = np.median(values)
    p95 = np.percentile(values, 95)
    ax.scatter([xpos], [median], s=75, color="white", edgecolor=color,
               linewidth=2.4, zorder=4)
    ax.vlines(xpos, median, p95, color=color, lw=4, zorder=3)
    ax.scatter([xpos], [p95], marker="_", s=250, color=color,
               linewidth=3, zorder=4)
    p95_y = 92 if xpos == 1 else p95 + 32
    median_y = 68 if xpos == 1 else median - 42
    ax.text(xpos, p95_y, f"P95 {p95:.1f} ms", ha="center",
            color=color, fontweight="bold")
    ax.text(xpos, median_y, f"Median {median:.1f} ms",
            ha="center", color="#253746")

ax.axhline(120, color="#c74747", lw=2, ls="--")
ax.text(2.38, 129, "120 ms", ha="right", va="bottom",
        color="#c74747", fontweight="bold")
ax.set_title("B. Response-time distribution", loc="left", pad=12)
ax.set_ylabel("Inference response time (ms)")
ax.set_xticks([1, 2], ["GPU free", "GPU contended"])
ax.set_ylim(0, 900)
ax.grid(True, axis="y", alpha=.18)

# C: Perception-loop throughput response
ax = fig.add_subplot(grid[1, :])
ax.set_facecolor("white")
labels = ["Previous loop limit", "Measured: GPU free", "Measured: GPU load"]
fps = [15.0, 35.9, 28.5]
colors = ["#8395a1", "#38a46a", "#ed8b3a"]
bars = ax.barh(labels, fps, color=colors, height=.52)
ax.invert_yaxis()
ax.set_xlim(0, 40)
ax.set_xlabel("Perception-loop throughput (frames per second)")
ax.set_title("C. Perception-loop response under operating conditions",
             loc="left", pad=10)
ax.grid(True, axis="x", alpha=.18)
for bar, value in zip(bars, fps):
    ax.text(value + .6, bar.get_y() + bar.get_height()/2,
            f"{value:.1f} FPS", va="center", fontweight="bold",
            color="#253746", fontsize=13)
ax.text(37.8, 0.08, "+139% vs limit", ha="right", va="center",
        color="#38a46a", fontweight="bold")
ax.text(37.8, 2.08, "+90% vs limit", ha="right", va="center",
        color="#c96d24", fontweight="bold")

fig.suptitle("Measured Vision-System Response to GPU Contention",
             fontsize=24, fontweight="bold", color="#0c1b2d", y=.98)
fig.text(.5, .935,
         "1,000 YOLO samples plus live perception-loop FPS measurements",
         ha="center", fontsize=13, color="#526777")
fig.text(.5, .015,
         "Interpretation: GPU contention increases inference latency, but the optimized perception loop remains above the old 15 FPS limit. "
         "The drain-and-ack handshake prevents motion planning from starting until inference has actually released the GPU.",
         ha="center", fontsize=10.5, color="#526777")

OUT.mkdir(exist_ok=True)
png = OUT / "gpu_contention_response_study.png"
pdf = OUT / "gpu_contention_response_study.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
print(png)
print(pdf)
