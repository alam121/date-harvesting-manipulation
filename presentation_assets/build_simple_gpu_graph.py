#!/usr/bin/env python3
"""Create a simple presentation graph for the GPU response study."""

from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_output"

labels = ["Previous\nlimit", "GPU free", "GPU under\nload"]
values = [15.0, 35.9, 28.5]
colors = ["#8597a3", "#35a56b", "#ed8b38"]

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
ax.set_facecolor("white")

bars = ax.bar(labels, values, color=colors, width=.62)
ax.set_ylim(0, 42)
ax.set_ylabel("Perception loop (FPS)", fontsize=15, color="#263746")
ax.set_title("Perception Response Under GPU Load",
             fontsize=24, fontweight="bold", color="#0c1b2d", pad=20)
ax.grid(axis="y", alpha=.18)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="x", labelsize=14, colors="#263746", length=0)
ax.tick_params(axis="y", labelsize=11, colors="#607482", length=0)

for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, value + 1,
            f"{value:.1f} FPS", ha="center", va="bottom",
            fontsize=18, fontweight="bold", color="#263746")

ax.annotate("System remains responsive\nwhile the GPU is shared",
            xy=(2, 28.5), xytext=(1.55, 39),
            ha="center", va="center", fontsize=13, color="#b8601e",
            arrowprops=dict(arrowstyle="-|>", color="#ed8b38", lw=2))

fig.text(.5, .035,
         "Result: the optimized loop stays 90% above the previous limit under load.\n"
         "A drain-and-ack handshake prevents YOLO inference and cuRobo planning from conflicting on the GPU.",
         ha="center", fontsize=12.5, color="#526777")

plt.subplots_adjust(left=.12, right=.96, top=.84, bottom=.22)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "simple_gpu_response_graph.png", dpi=220,
            facecolor="white", bbox_inches="tight")
fig.savefig(OUT / "simple_gpu_response_graph.pdf",
            facecolor="white", bbox_inches="tight")
