"""Findings chart: perception Loop FPS vs YOLO FPS, GPU-free vs under contention.
Numbers are the live-HUD values captured this session (docs/images/vision_fps_*).
One axis (FPS), two series, dashed reference at the old 15 fps cap."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/datepalm2/manipulatorsdatepalm/presentation_assets/farm_media/findings_fps.png"

conds = ["GPU free\n(arm idle)", "Under GPU load\n(cuRobo planning)"]
loop = [35.9, 28.5]     # perception loop fps (live HUD)
yolo = [22.1, 10.2]     # YOLO inference fps (live HUD)

NAVY, BLUE, ORANGE, MUTED, GREEN = "#0e1e30", "#2a78d6", "#d67628", "#6e7e8c", "#2d965f"
x = np.arange(len(conds)); w = 0.34

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

b1 = ax.bar(x - w/2, loop, w, label="Perception loop", color=BLUE, zorder=3)
b2 = ax.bar(x + w/2, yolo, w, label="YOLO inference", color=ORANGE, zorder=3)

ax.axhline(15, ls="--", lw=1.3, color=MUTED, zorder=2)
ax.text(len(conds) - 0.5, 15.7, "old loop cap = 15 fps", ha="right", fontsize=9, color=MUTED)
ax.axhline(30, ls=":", lw=1.2, color=GREEN, zorder=2)
ax.text(0.5, 31.0, "camera target 30 fps", ha="center", fontsize=9, color=GREEN)

for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.6,
                f"{r.get_height():.1f}", ha="center", fontsize=10, weight="bold", color=NAVY)

ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=10.5, color=NAVY)
ax.set_ylabel("frames per second", fontsize=10.5, color=NAVY)
ax.set_ylim(0, 40)
ax.tick_params(colors=NAVY, labelsize=9.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(MUTED)
ax.grid(axis="y", ls=":", lw=0.6, color="#d5dde2", zorder=0)
ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=1)
ax.set_title("Perception loop holds ~30 fps; YOLO inference dips under GPU contention",
             fontsize=12.5, color=NAVY, weight="bold", pad=12)
fig.text(0.5, -0.02,
         "Live HUD, ZED X Mini harvest mode. Loop rate ≈2× the old 15 fps cap in both cases; "
         "YOLO halves (22→10) only when it shares the Orin GPU with cuRobo planning.",
         ha="center", fontsize=8.3, color=MUTED, wrap=True)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
