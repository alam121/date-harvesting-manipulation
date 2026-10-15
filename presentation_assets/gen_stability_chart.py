"""Generate a clean stability-metrics chart from the golf-cart validation data
(docs/GOLFCART_STABILITY_VALIDATION.md, run 2026-06-11). Two panels, one axis
each (jitter deg / force N) — no dual-axis. White background, restrained palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/datepalm2/manipulatorsdatepalm/presentation_assets/farm_media/stability_metrics.png"

scen = ["Cart moving\n(idle arm)", "Cart static\n(idle arm)", "Trajectory\ngoals"]
jitter = [0.000672, 0.000681, 2.728]      # position jitter RMS, deg
force = [0.249, 0.275, 1.053]             # force noise RMS, N
NAVY, CYAN, ORANGE, MUTED = "#0c1b2d", "#23b3bf", "#ee8b3f", "#697b8b"
bars = [CYAN, CYAN, ORANGE]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5), dpi=200)
fig.patch.set_facecolor("white")

a1.bar(scen, jitter, color=bars, width=0.6, zorder=3)
a1.set_yscale("log")
a1.set_title("Position jitter RMS  (deg, log scale)", fontsize=11, color=NAVY, weight="bold", pad=10)
a1.axhline(0.001, ls="--", lw=1, color=MUTED, zorder=2)
a1.text(2.4, 0.0012, "0.001° idle band", fontsize=8, color=MUTED, ha="right")
for i, v in enumerate(jitter):
    a1.text(i, v * 1.25, f"{v:.4g}", ha="center", fontsize=8.5, color=NAVY, weight="bold")

a2.bar(scen, force, color=bars, width=0.6, zorder=3)
a2.set_title("Force noise RMS  (N)", fontsize=11, color=NAVY, weight="bold", pad=10)
for i, v in enumerate(force):
    a2.text(i, v + 0.03, f"{v:.3g} N", ha="center", fontsize=8.5, color=NAVY, weight="bold")

for ax in (a1, a2):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=NAVY, labelsize=8.5)
    ax.grid(axis="y", ls=":", lw=0.6, color="#d5dde2", zorder=0)

fig.suptitle("Golf-cart mounted UR10e — all runs Cleared (2026-06-11, ~31 Hz)",
             fontsize=12, color=NAVY, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
