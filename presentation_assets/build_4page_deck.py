#!/usr/bin/env python3
"""4-page Date-Harvesting progress deck — plain white, diagram-forward.

Slides: (1) Golf-cart stability validation, (2) Vision & GPU performance,
(3) New gripper & how it works, (4) Field evidence (last two videos).
Content is grounded in the repo docs + git history + verified session results.
"""
from pathlib import Path
import sys

sys.path.insert(0, "/tmp/datepalm_pydeps")
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_output" / "Date_Harvesting_4Page_July_2026.pptx"
MEDIA = ROOT / "presentation_assets" / "farm_media"
BENCH = ROOT / "ur_ws_new/src/ur10e_curobo/benchmarks/results"
DOCIMG = ROOT / "docs/images"

NAVY = RGBColor(14, 30, 48); BLUE = RGBColor(17, 96, 140); CYAN = RGBColor(20, 150, 160)
GREEN = RGBColor(45, 150, 95); ORANGE = RGBColor(214, 118, 40); RED = RGBColor(196, 66, 66)
WHITE = RGBColor(255, 255, 255); INK = RGBColor(28, 40, 52); MUTED = RGBColor(110, 126, 140)
PANEL = RGBColor(244, 247, 249); LINE = RGBColor(210, 219, 225)

prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)


def set_bg(slide, color):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def textbox(slide, text, x, y, w, h, size=16, color=INK, bold=False,
            align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    b = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = b.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06); tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.name = "Aptos"; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return b


def rrect(slide, x, y, w, h, fill, border=None, radius=True):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = border or fill; s.line.width = Pt(1.0)
    s.shadow.inherit = False
    return s


def title(slide, headline, kicker):
    textbox(slide, kicker.upper(), 0.62, 0.34, 11.8, 0.3, 10.5, CYAN, True)
    textbox(slide, headline, 0.58, 0.64, 12.2, 0.72, 26, NAVY, True)
    rrect(slide, 0.62, 1.36, 1.0, 0.05, CYAN, radius=False)


def footer(slide, n):
    textbox(slide, "Date-Harvesting Manipulator · July 2026", 0.62, 7.16, 9.0, 0.2, 8, MUTED)
    textbox(slide, f"{n} / 4", 12.0, 7.12, 0.8, 0.24, 9, CYAN, True, PP_ALIGN.RIGHT)


def bullets(slide, items, x, y, w, h, size=13.5, spacing=9):
    b = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = b.text_frame; tf.clear(); tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.font.name = "Aptos"; p.space_after = Pt(spacing)
        head, _, rest = it.partition("|")
        r = p.add_run(); r.text = "▸  " + head.strip()
        r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = INK
        if rest:
            r2 = p.add_run(); r2.text = "  " + rest.strip()
            r2.font.size = Pt(size); r2.font.color.rgb = MUTED
    return b


def metric(slide, value, label, x, y, w=2.5, accent=CYAN):
    rrect(slide, x, y, w, 1.15, PANEL, LINE)
    textbox(slide, value, x + .14, y + .13, w - .28, .5, 22, accent, True)
    textbox(slide, label, x + .14, y + .68, w - .28, .4, 9.5, MUTED, True)


def node(slide, x, y, w, h, top, sub=None, fill=PANEL, border=CYAN, tc=NAVY):
    rrect(slide, x, y, w, h, fill, border)
    if sub:
        textbox(slide, top, x, y + h*0.16, w, h*0.4, 12.5, tc, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        textbox(slide, sub, x, y + h*0.52, w, h*0.42, 9, MUTED, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    else:
        textbox(slide, top, x, y, w, h, 12.5, tc, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


def arrow(slide, x, y, w=0.42, h=0.34, color=CYAN):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background(); s.shadow.inherit = False


def img(slide, path, x, y, w, h=None):
    path = Path(path)
    if not path.exists():
        rrect(slide, x, y, w, h or w*.56, PANEL, LINE)
        textbox(slide, "(missing)", x, y + (h or w*.56)/2 - .15, w, .3, 10, MUTED, True, PP_ALIGN.CENTER)
        return
    if h is None:
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def movie(slide, mp4, poster, x, y, w, h):
    try:
        slide.shapes.add_movie(str(mp4), Inches(x), Inches(y), Inches(w), Inches(h),
                               poster_frame_image=str(poster), mime_type="video/mp4")
    except Exception:
        img(slide, poster, x, y, w, h)


def slide():
    s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, WHITE); return s


# ── 1 · GOLF-CART STABILITY VALIDATION ───────────────────────────────────────
s = slide()
title(s, "UR10e Golf-Cart Stability Validation", "Field validation · 2026-06-11 · ~31 Hz logging")
# system diagram (left)
node(s, 0.62, 1.7, 3.05, 0.85, "Golf cart", "mobile field base")
arrow(s, 2.0, 2.62, 0.3, 0.34)
node(s, 0.62, 3.0, 3.05, 0.85, "UR10e arm", "rear-mounted")
arrow(s, 2.0, 3.92, 0.3, 0.34)
node(s, 0.62, 4.3, 3.05, 0.85, "Sensors", "joint encoders · F/T", border=ORANGE)
arrow(s, 2.0, 5.22, 0.3, 0.34)
node(s, 0.62, 5.6, 3.05, 0.8, "Stability recorder", "31 Hz logs → CSV", border=GREEN)
# metrics + chart (right)
metric(s, "3 / 3", "runs Cleared", 3.95, 1.7, 2.35, GREEN)
metric(s, "<0.001°", "idle jitter RMS", 6.45, 1.7, 2.35, CYAN)
metric(s, "±0.05 mm", "UR10e spec ref", 8.95, 1.7, 2.35, MUTED)
img(s, MEDIA / "stability_metrics.png", 3.95, 3.05, 8.75)
textbox(s, "Idle platform (moving or static) stays below 0.001° joint jitter; "
           "commanded-trajectory run reached all 14 goals. All acceptance criteria Cleared.",
        3.95, 6.55, 8.75, 0.6, 11, MUTED)
footer(s, 1)

# ── 2 · VISION & GPU PERFORMANCE ─────────────────────────────────────────────
s = slide()
title(s, "Vision & GPU Performance", "Perception pipeline · 2× loop rate")
# pipeline diagram (top band)
py = 1.62
node(s, 0.62, py, 2.0, 0.82, "ZED X Mini", "30 fps RGBD")
arrow(s, 2.68, py+0.24)
node(s, 3.16, py, 2.0, 0.82, "YOLO seg", "GPU inference")
arrow(s, 5.22, py+0.24)
node(s, 5.70, py, 1.75, 0.82, "Targets", "3-D + score")
arrow(s, 7.51, py+0.24)
node(s, 7.99, py, 2.0, 0.82, "cuRobo plan", "GPU motion-gen")
arrow(s, 10.05, py+0.24)
node(s, 10.53, py, 2.15, 0.82, "UR10e", "collision-free", border=GREEN)
# GPU-sharing band under YOLO + cuRobo
rrect(s, 3.16, py+0.9, 6.83, 0.42, RGBColor(233, 244, 245), CYAN)
textbox(s, "one Orin GPU — drain-and-ack handshake serializes YOLO ↔ cuRobo access (replaces timed handoff)",
        3.16, py+0.92, 6.83, 0.38, 9.5, BLUE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
# left: metrics + bullets
metric(s, "30.1 fps", "loop, under YOLO load", 0.62, 3.5, 2.55, GREEN)
metric(s, "~29 fps", "raw YOLO (GPU free)", 3.3, 3.5, 2.55, CYAN)
bullets(s, [
    "15 → 30 fps loop | ZEDMINI_RGBD_FPS raised, hardware-validated",
    "imgsz auto-read from engine | input-size crash can't recur",
    "Event-driven loop | busy-poll → Event.wait(), no poll latency",
], 0.62, 4.85, 5.5, 2.2, size=12.5, spacing=8)
# right: live HUD screenshots (before / after) — the fps improvement, in the app
img(s, DOCIMG / "vision_fps_uncontended.png", 6.55, 3.5, 3.05)
textbox(s, "Live HUD — GPU free  (Loop 36 · YOLO 22)", 6.55, 5.28, 3.05, 0.26, 8.5, MUTED, True)
img(s, DOCIMG / "vision_fps_under_gpu_load.png", 9.7, 3.5, 3.05)
textbox(s, "Live HUD — under GPU load  (Loop 28 · YOLO 10)", 9.7, 5.28, 3.05, 0.26, 8.5, MUTED, True)
textbox(s, "Loop rate holds ~30 under contention; YOLO inference is what dips when it "
           "shares the GPU with cuRobo — the reason the drain-and-ack handshake matters.",
        6.55, 5.75, 6.2, 0.8, 11, MUTED)
footer(s, 2)

# ── 3 · NEW GRIPPER & HOW IT WORKS ───────────────────────────────────────────
s = slide()
title(s, "New Gripper: DG-3F-M — Integration & Operation", "End effector · ported Jul 16–19")
# grasp-cycle flow diagram (top)
gy = 1.62
node(s, 0.62, gy, 1.75, 0.8, "Open", "profile pose")
arrow(s, 2.43, gy+0.23)
node(s, 2.85, gy, 1.75, 0.8, "Approach", "to target")
arrow(s, 4.66, gy+0.23)
node(s, 5.08, gy, 1.75, 0.8, "Close", "3 fingers")
arrow(s, 6.89, gy+0.23)
node(s, 7.31, gy, 1.9, 0.8, "Grip check", "force feedback", border=ORANGE)
arrow(s, 9.27, gy+0.23)
node(s, 9.66, gy, 3.05, 0.8, "Success · Weak · Slip", "labeled outcome", border=GREEN)
# left bullets
bullets(s, [
    "DG-3F-M driver port | Modbus changes for the 3-finger hardware path",
    "Old + New profiles | per-robot finger-joint indices & open/close poses",
    "Grasp + feedback | commands, force read-back, grip check, slip analysis",
    "Finger test tooling | 240-line diagnostic (dg3fm_finger_test)",
    "Wired into launch | gripper profiles selectable from the launcher/GUI",
], 0.62, 2.85, 6.15, 4.0, size=13, spacing=11)
# right: metrics + field photo
metric(s, "3", "fingers", 6.95, 2.75, 1.75, CYAN)
metric(s, "2", "profiles (old + new)", 8.85, 2.75, 1.9, ORANGE)
metric(s, "240", "line finger test", 10.9, 2.75, 1.8, GREEN)
img(s, MEDIA / "image4.png", 8.0, 4.15, 3.7)
textbox(s, "DG-3F-M closing on a date cluster in the field", 6.95, 6.72, 5.75, 0.3, 10, MUTED, True, PP_ALIGN.CENTER)
footer(s, 3)

# ── 4 · FIELD EVIDENCE (latest two videos) ───────────────────────────────────
s = slide()
title(s, "Field Evidence", "Latest field runs · embedded video")
movie(s, MEDIA / "media4.mp4", MEDIA / "image4.png", 0.62, 1.7, 5.95, 3.35)
textbox(s, "Field run — approach & grasp", 0.62, 5.1, 5.95, 0.3, 12, NAVY, True, PP_ALIGN.CENTER)
movie(s, MEDIA / "media5.mp4", MEDIA / "image5.png", 6.78, 1.7, 5.95, 3.35)
textbox(s, "Field run — harvest cycle", 6.78, 5.1, 5.95, 0.3, 12, NAVY, True, PP_ALIGN.CENTER)
rrect(s, 0.62, 5.75, 12.1, 0.95, PANEL, LINE)
textbox(s, "Videos play in PowerPoint / Keynote / Impress.   Next: reduce cuRobo GPU footprint to lift "
           "YOLO under contention · prune old model iterations · finish operator-doc pass.",
        0.85, 5.9, 11.7, 0.7, 12, INK, False, PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE)
footer(s, 4)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"saved: {OUT}  ({len(prs.slides._sldIdLst)} slides)")
