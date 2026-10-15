#!/usr/bin/env python3
"""Build the July 2026 Date Harvesting progress presentation."""

from pathlib import Path
import csv
import statistics
import sys

sys.path.insert(0, "/tmp/datepalm_pydeps")
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.chart.data import ChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_output" / "Date_Harvesting_Progress_July_2026.pptx"
ASSETS = ROOT / "presentation_assets"
MEDIA = ASSETS / "farm_media"
BENCH = ROOT / "ur_ws_new/src/ur10e_curobo/benchmarks/results"
DOCIMG = ROOT / "docs/images"

NAVY = RGBColor(12, 27, 45)
BLUE = RGBColor(17, 96, 140)
CYAN = RGBColor(35, 179, 191)
GREEN = RGBColor(55, 164, 105)
ORANGE = RGBColor(238, 139, 63)
RED = RGBColor(204, 70, 70)
CREAM = RGBColor(245, 246, 241)
WHITE = RGBColor(255, 255, 255)
MUTED = RGBColor(105, 123, 139)
INK = RGBColor(30, 42, 54)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def set_bg(slide, color=NAVY):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def rect(slide, x, y, w, h, color, radius=False, line=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.color.rgb = line or color
    return shp


def textbox(slide, text, x, y, w, h, size=20, color=INK, bold=False,
            align=PP_ALIGN.LEFT, font="Aptos", margin=0.08, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return box


def title(slide, headline, kicker=None, dark=False):
    color = WHITE if dark else NAVY
    if kicker:
        textbox(slide, kicker.upper(), 0.65, 0.34, 11.8, 0.3, 10, CYAN, True)
    textbox(slide, headline, 0.62, 0.67, 12.0, 0.65, 27, color, True)
    rect(slide, 0.65, 1.31, 1.05, 0.055, CYAN)


def footer(slide, n, source="Repository and project evidence, accessed 27 Jul 2026"):
    textbox(slide, source, 0.65, 7.13, 11.4, 0.2, 8, MUTED)
    textbox(slide, f"{n:02d}", 12.22, 7.08, 0.45, 0.25, 9, CYAN, True, PP_ALIGN.RIGHT)


def bullet_list(slide, items, x, y, w, h, size=16, color=INK, spacing=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(spacing)
        p.text = "•  " + p.text
    return box


def metric(slide, value, label, x, y, w=2.2, accent=CYAN):
    rect(slide, x, y, w, 1.2, WHITE, True, RGBColor(222, 228, 232))
    textbox(slide, value, x + .14, y + .15, w - .28, .48, 25, accent, True)
    textbox(slide, label, x + .14, y + .70, w - .28, .32, 10, MUTED, True)


def add_link(slide, label, url, x, y, w, h=.3, size=10):
    box = textbox(slide, label, x, y, w, h, size, BLUE, True)
    box.text_frame.paragraphs[0].runs[0].hyperlink.address = url
    return box


def add_image(slide, path, x, y, w, h=None):
    path = str(path)
    if h is None:
        return slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    return slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))


def add_slide(light=True):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s, CREAM if light else NAVY)
    return s


# 1 — Cover
s = add_slide(False)
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 8.65, 0, 4.683, 7.5, BLUE)
rect(s, 8.65, 4.95, 4.683, 2.55, CYAN)
textbox(s, "DATE PALM\nHARVESTING", .7, .65, 7.7, 1.35, 37, WHITE, True)
textbox(s, "Three-week engineering progress review", .72, 2.18, 7.2, .48, 19, CYAN, True)
textbox(s, "6–27 JULY 2026", .73, 2.82, 4.0, .35, 12, WHITE, True)
bullet_list(s, [
    "Vision performance and GPU coordination",
    "Operator controls, LiDAR and camera workflows",
    "DG‑3F‑M gripper integration and field evidence",
    "Stability, calibration and reproducible benchmarks",
], .72, 3.45, 7.2, 2.1, 17, WHITE, 10)
textbox(s, "UR10e  •  cuRobo\nZED  •  Livox  •  ROS 2", 8.95, 1.02, 3.9, 1.1, 17, WHITE, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
textbox(s, "Evidence from Git history,\nbenchmarks, screenshots,\nwork items and field videos", 8.98, 5.28, 3.8, 1.3, 14, NAVY, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
footer(s, 1)

# 2 — Executive summary
s = add_slide()
title(s, "Executive summary", "Outcome")
metric(s, "9", "COMMITS IN PERIOD", .7, 1.65, 2.15)
metric(s, "+7,462", "LINES ADDED", 2.98, 1.65, 2.15, GREEN)
metric(s, "59", "CORE FILES CHANGED", 5.26, 1.65, 2.15, BLUE)
metric(s, "1,000", "YOLO TIMING SAMPLES", 7.54, 1.65, 2.15, ORANGE)
metric(s, "5", "EMBEDDED FIELD VIDEOS", 9.82, 1.65, 2.15, RED)
rect(s, .7, 3.2, 12.0, 3.35, WHITE, True, RGBColor(222, 228, 232))
bullet_list(s, [
    "Perception loop increased from the former 15 fps cap to 35.9 fps when uncontended and held 28.5 fps under synthetic GPU contention.",
    "GPU handoff changed from a fixed 120 ms delay to a drain-and-ack handshake, eliminating the known timing race by waiting for actual inference completion.",
    "A substantially richer RViz/operator surface now covers launch, motion, goals, camera capture, monitoring, LiDAR scans, heat and safe-zone controls.",
    "DG‑3F‑M gripper support, profiles and test tooling were integrated; field attempts now provide labeled success, weak-grip and slip evidence.",
    "Calibration profiles and capture tooling support ZED X Mini RGBD and ZED X One + ZED X Mini dual-camera workflows.",
], 1.0, 3.53, 11.4, 2.65, 15, INK, 7)
footer(s, 2, "Git log/stat, benchmark CSVs, project documentation and embedded field media")

# 3 — Timeline
s = add_slide()
title(s, "What changed, week by week", "Delivery timeline")
events = [
    ("06 Jul", "Deterministic GPU handoff", "Drain-and-ack replaces fixed delay"),
    ("13 Jul", "Benchmarks + safety", "1,000 samples; LiDAR, safe-zone and operator docs"),
    ("16 Jul", "RViz + capture + gripper", "Pallet assets, ZED capture tools, DG‑3F‑M port"),
    ("19 Jul", "Integrated launch workflow", "GUI, camera controls and gripper profiles"),
    ("22–26 Jul", "Field-ready polish", "README evidence, FPS, calibration and UI tuning"),
]
rect(s, 1.05, 3.34, 11.2, .08, BLUE)
xs = [1.05, 3.45, 5.85, 8.25, 10.65]
for i, ((date, head, body), x) in enumerate(zip(events, xs)):
    rect(s, x, 3.17, .42, .42, CYAN if i < 4 else ORANGE, True)
    textbox(s, date, x-.25, 2.62, 1.0, .3, 11, NAVY, True, PP_ALIGN.CENTER)
    textbox(s, head, x-.72, 3.78, 1.9, .66, 12, NAVY, True, PP_ALIGN.CENTER)
    textbox(s, body, x-.75, 4.52, 1.96, .95, 9, MUTED, False, PP_ALIGN.CENTER)
textbox(s, "Concurrency safety  →  measurement  →  operator integration  →  field evidence", 1.15, 5.88, 11.0, .55, 16, BLUE, True, PP_ALIGN.CENTER)
footer(s, 3, "Commit dates and subjects from operator-docs-rviz-camera-lidar")

# 4 — Scope map
s = add_slide()
title(s, "Six workstreams moved together", "System scope")
cards = [
    ("VISION", "FPS, YOLO timing,\ncalibration profiles", CYAN),
    ("PLANNING", "cuRobo GPU handoff,\ngoals and execution", BLUE),
    ("SAFETY", "safe zones, pallets,\nLiDAR obstacle scans", GREEN),
    ("OPERATOR", "RViz panel, launch GUI,\nmonitoring and capture", ORANGE),
    ("GRIPPER", "DG‑3F‑M port, profiles,\nforce / slip evidence", RED),
    ("EVIDENCE", "CSV benchmarks,\nscreenshots and videos", RGBColor(122, 92, 160)),
]
for i, (h, b, c) in enumerate(cards):
    col, row = i % 3, i // 3
    x, y = .72 + col*4.12, 1.72 + row*2.25
    rect(s, x, y, 3.75, 1.78, WHITE, True, RGBColor(222,228,232))
    rect(s, x, y, .12, 1.78, c)
    textbox(s, h, x+.3, y+.22, 3.1, .34, 13, c, True)
    textbox(s, b, x+.3, y+.72, 3.1, .72, 16, INK, True)
textbox(s, "Integration is the headline: sensing, planning, execution, UI and evidence capture moved together.", .82, 6.28, 11.8, .55, 13, NAVY, True, PP_ALIGN.CENTER)
footer(s, 4)

# 5 — Architecture
s = add_slide(False)
title(s, "Updated harvesting workflow", "Architecture", dark=True)
nodes = [
    ("ZED / LiDAR", .55, CYAN), ("YOLO + depth", 2.75, CYAN),
    ("Goal scoring", 4.95, ORANGE), ("Safe-zone + cuRobo", 7.15, BLUE),
    ("UR10e execute", 9.35, GREEN), ("DG‑3F‑M feedback", 11.05, RED),
]
for label, x, c in nodes:
    rect(s, x, 2.3, 1.75, 1.15, c, True)
    textbox(s, label, x+.08, 2.52, 1.59, .65, 11, WHITE, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
for x in [2.35, 4.55, 6.75, 8.95, 10.75]:
    textbox(s, "→", x, 2.6, .38, .4, 23, WHITE, True, PP_ALIGN.CENTER)
rect(s, 2.25, 4.45, 8.85, 1.1, RGBColor(24, 51, 74), True)
textbox(s, "Operator supervision + evidence capture", 2.55, 4.72, 8.25, .38, 19, WHITE, True, PP_ALIGN.CENTER)
textbox(s, "RViz panel  •  launch GUI  •  camera recording  •  stability CSV  •  issue-linked field video", 1.25, 5.92, 10.85, .48, 14, CYAN, True, PP_ALIGN.CENTER)
footer(s, 5, "Architecture synthesized from README and changed packages")

# 6 — Vision FPS
s = add_slide()
title(s, "Perception loop remains fast under GPU load", "Measured performance")
add_image(s, DOCIMG / "vision_fps_uncontended.png", .67, 1.62, 5.9, 3.75)
add_image(s, DOCIMG / "vision_fps_under_gpu_load.png", 6.77, 1.62, 5.9, 3.75)
textbox(s, "GPU free", .75, 5.51, 2.0, .32, 13, BLUE, True)
textbox(s, "YOLO 22.1 fps  |  Loop 35.9 fps", .75, 5.88, 4.9, .35, 18, NAVY, True)
textbox(s, "GPU contended", 6.85, 5.51, 2.2, .32, 13, ORANGE, True)
textbox(s, "YOLO 10.2 fps  |  Loop 28.5 fps", 6.85, 5.88, 5.1, .35, 18, NAVY, True)
textbox(s, "Loop throughput remains near 30 fps; GPU sharing primarily affects detector inference.", .83, 6.55, 11.7, .36, 14, GREEN, True, PP_ALIGN.CENTER)
footer(s, 6, "docs/VISION_FPS.md — live /vision/display captures")

# 7 — Timing benchmark
s = add_slide()
title(s, "YOLO timing exposes why a fixed wait was unsafe", "1,000-frame benchmark")
add_image(s, BENCH / "yolo_inference_scatter.png", .68, 1.6, 6.15, 4.25)
add_image(s, BENCH / "yolo_inference_hist.png", 7.02, 1.6, 5.62, 4.25)
metric(s, "39.8 ms", "IDLE MEAN INFERENCE", .85, 6.02, 2.25, GREEN)
metric(s, "544.2 ms", "CONTENDED MEAN", 3.28, 6.02, 2.25, RED)
metric(s, "47.6 ms", "IDLE P95", 5.71, 6.02, 2.25, BLUE)
metric(s, "710.7 ms", "CONTENDED P95", 8.14, 6.02, 2.25, ORANGE)
metric(s, "120 ms", "OLD FIXED WAIT", 10.57, 6.02, 1.9, MUTED)
footer(s, 7, "600 idle + 400 contended samples from benchmark CSVs")

# 8 — Handshake
s = add_slide(False)
title(s, "Concurrency fix: wait for reality, not a timer", "GPU handoff", dark=True)
textbox(s, "OLD", .8, 1.72, 1.0, .3, 12, RED, True)
rect(s, .8, 2.15, 5.45, 1.35, RGBColor(55, 37, 47), True)
textbox(s, "Pause request  →  sleep 120 ms  →  plan", 1.05, 2.55, 4.95, .42, 18, WHITE, True, PP_ALIGN.CENTER)
textbox(s, "Failure mode: inference may still be using the GPU.", .95, 3.74, 5.1, .55, 14, RGBColor(244,170,170), True, PP_ALIGN.CENTER)
textbox(s, "NEW", 7.05, 1.72, 1.0, .3, 12, GREEN, True)
rect(s, 7.05, 2.15, 5.45, 1.35, RGBColor(26, 66, 62), True)
textbox(s, "Pause request  →  drain  →  ACK  →  plan", 7.28, 2.55, 4.98, .42, 18, WHITE, True, PP_ALIGN.CENTER)
textbox(s, "Guarantee: planning starts only after inference exits.", 7.18, 3.74, 5.18, .55, 14, RGBColor(163,235,195), True, PP_ALIGN.CENTER)
rect(s, 2.15, 5.18, 9.05, 1.1, RGBColor(24, 51, 74), True)
textbox(s, "Variable latency is handled deterministically; the benchmark demonstrates the timing risk removed.", 2.42, 5.44, 8.5, .48, 16, CYAN, True, PP_ALIGN.CENTER)
footer(s, 8, "Commit cff95f262 and benchmarks/README.md")

# 9 — Operator controls
s = add_slide()
title(s, "Operator workflow expanded substantially", "Human-in-the-loop")
tabs = [
    ("MOTION", "Home, dropoff, execute,\ngripper, plan confirmation"),
    ("GOAL", "Manual goals, capture,\nreachability and LiDAR scan"),
    ("MONITOR", "Harvest results, joints,\nforces and stability recorder"),
    ("CAMERA", "Snapshots and videos\nfrom /vision/display"),
    ("HEAT", "Joint and tool\ntemperature visibility"),
    ("SETTINGS", "Velocity and process\nrefresh controls"),
]
for i, (h, b) in enumerate(tabs):
    col, row = i % 3, i // 3
    x, y = .72 + col*4.12, 1.66 + row*2.25
    rect(s, x, y, 3.72, 1.82, WHITE, True, RGBColor(222,228,232))
    rect(s, x, y, 3.72, .42, [BLUE,CYAN,ORANGE,GREEN,RED,RGBColor(122,92,160)][i], True)
    textbox(s, h, x+.18, y+.54, 3.35, .31, 13, NAVY, True)
    textbox(s, b, x+.18, y+.97, 3.35, .66, 14, INK, False)
textbox(s, "737 lines added in the principal RViz panel implementation across the period comparison.", .8, 6.36, 11.7, .45, 15, BLUE, True, PP_ALIGN.CENTER)
footer(s, 9, "README, operator guide and git diff statistics")

# 10 — LiDAR, safe zone, calibration
s = add_slide()
title(s, "Sensing and safety became configurable workflows", "Field robustness")
items = [
    ("LiDAR scan", "Preview, preflight validation and bag recording; improved scan parsing and safe-zone integration.", GREEN),
    ("Physical scene", "Golf-cart pallet meshes and static obstacles added for collision-aware planning.", BLUE),
    ("Camera modes", "ZED X Mini RGBD and ZED X One + Mini depth modes select matching calibration profiles.", CYAN),
    ("Calibration", "Hand-eye improvements plus dual-camera extrinsic calibration entry point.", ORANGE),
]
for i, (h,b,c) in enumerate(items):
    y = 1.62 + i*1.27
    rect(s, .75, y, 11.85, 1.02, WHITE, True, RGBColor(222,228,232))
    rect(s, .75, y, .16, 1.02, c)
    textbox(s, h, 1.12, y+.16, 1.78, .35, 15, c, True)
    textbox(s, b, 2.9, y+.14, 9.23, .62, 13, INK)
footer(s, 10, "Changed launch, vision, LiDAR, safe-zone, obstacle and calibration code")

# 11 — Gripper
s = add_slide(False)
title(s, "DG‑3F‑M gripper integration reached field-test stage", "End effector", dark=True)
metric(s, "3", "FINGERS / PROFILE CONTROL", .75, 1.68, 2.45, CYAN)
metric(s, "240", "LINES IN FINGER TEST", 3.42, 1.68, 2.45, GREEN)
metric(s, "5", "LOCAL FIELD CLIPS", 6.09, 1.68, 2.45, ORANGE)
metric(s, "6", "LABELED ATTEMPTS", 8.76, 1.68, 2.45, RED)
bullet_list(s, [
    "New launcher profiles and GUI paths reduce setup friction.",
    "Driver and Modbus changes support the DG‑3F‑M hardware path.",
    "Grasp commands, force feedback, grip checks and slip analysis are surfaced in the integrated workflow.",
    "Finger-test tooling enables focused commissioning outside the full harvest stack.",
], .95, 3.42, 6.7, 2.65, 16, WHITE, 10)
rect(s, 8.0, 3.42, 4.35, 2.35, RGBColor(24,51,74), True)
textbox(s, "Field outcome mix", 8.32, 3.71, 3.7, .35, 17, CYAN, True, PP_ALIGN.CENTER)
textbox(s, "3 success\n1 weak grip\n2 slips", 8.35, 4.22, 3.65, 1.25, 23, WHITE, True, PP_ALIGN.CENTER)
footer(s, 11, "Commits 67dd844a and 5b790675; Farm_experiment.pptx labels")

# 12 — Field outcome chart
s = add_slide()
title(s, "Field attempts create a tuning dataset", "Observed outcomes")
data = ChartData()
data.categories = ["Success", "Weak grip", "Slipped"]
data.add_series("Attempts", (3, 1, 2))
chart = s.shapes.add_chart(
    XL_CHART_TYPE.DOUGHNUT, Inches(.8), Inches(1.55), Inches(5.6), Inches(4.8), data
).chart
chart.has_legend = True
chart.legend.position = XL_LEGEND_POSITION.BOTTOM
chart.legend.font.size = Pt(12)
chart.plots[0].vary_by_categories = True
chart.plots[0].hole_size = 58
metric(s, "50%", "SUCCESS LABEL RATE", 7.0, 1.75, 2.35, GREEN)
metric(s, "66.7%", "NO-SLIP RATE", 9.58, 1.75, 2.35, BLUE)
rect(s, 6.92, 3.25, 5.25, 2.67, WHITE, True, RGBColor(222,228,232))
bullet_list(s, [
    "Success cases: preserve approach and closing behavior.",
    "Weak grip: tune force/profile or target alignment.",
    "Slips: inspect contact, regrip logic and motion after closure.",
], 7.18, 3.55, 4.78, 1.95, 14, INK, 8)
add_link(s, "GitLab field-test evidence (#74)", "https://gitlab.kaust.edu.sa/dsa-kaust/projects/date-palm-automation/-/issues/74", 7.18, 5.55, 4.5)
footer(s, 12, "Six labeled outcomes in existing farm-experiment presentation")

# 13–15 — Embedded video evidence
video_labels = [
    ("Attempt 1 — Success", "media1.mp4", "image1.png"),
    ("Attempt 2 — Success", "media2.mp4", "image2.png"),
    ("Attempt 3 — Success", "media3.mp4", "image3.png"),
    ("Attempt 4 — Weak grip", "media4.mp4", "image4.png"),
    ("Attempt 5 — Slipped", "media5.mp4", "image5.png"),
]
for slide_idx, group in enumerate((video_labels[:2], video_labels[2:4], video_labels[4:]), start=13):
    s = add_slide(False)
    title(s, "Embedded field evidence", f"Video review {slide_idx-12}/3", dark=True)
    if len(group) == 2:
        positions = [(.68, 1.55, 5.95, 4.75), (6.73, 1.55, 5.95, 4.75)]
    else:
        positions = [(2.15, 1.55, 9.03, 4.95)]
    for (label, movie, poster), (x,y,w,h) in zip(group, positions):
        try:
            s.shapes.add_movie(
                str(MEDIA/movie), Inches(x), Inches(y), Inches(w), Inches(h),
                poster_frame_image=str(MEDIA/poster), mime_type="video/mp4"
            )
        except Exception:
            add_image(s, MEDIA/poster, x, y, w, h)
        textbox(s, label, x, y+h+.14, w, .42, 16, WHITE, True, PP_ALIGN.CENTER)
    footer(s, slide_idx, "Embedded MP4 evidence recovered from Farm_experiment.pptx")

# 16 — Stability
s = add_slide()
title(s, "Mounted-platform stability tests cleared", "Field validation")
rows = [
    ("Moving cart, idle arm", "189.24 s", "0.000672°", "0.249 N", "Cleared"),
    ("Static cart, idle arm", "15.90 s", "0.000681°", "0.275 N", "Cleared"),
    ("Robot trajectory goals", "104.32 s", "0.323° tracking RMS", "1.053 N", "Cleared"),
]
headers = ["SCENARIO", "DURATION", "JITTER / TRACKING", "FORCE RMS", "RESULT"]
xs = [.7, 4.05, 5.65, 8.15, 10.45]
ws = [3.25, 1.5, 2.4, 2.2, 1.8]
for x,w,h in zip(xs,ws,headers):
    rect(s,x,1.62,w,.55,NAVY)
    textbox(s,h,x+.08,1.75,w-.16,.22,10,WHITE,True)
for r,row in enumerate(rows):
    y=2.22+r*1.06
    for c,(x,w,val) in enumerate(zip(xs,ws,row)):
        rect(s,x,y,w,.9,WHITE,False,RGBColor(222,228,232))
        textbox(s,val,x+.08,y+.22,w-.16,.42,13,GREEN if c==4 else INK,c in (0,4))
metric(s, "31.26–31.31 Hz", "RECORDED SAMPLE RATE", .78, 5.72, 2.75, BLUE)
metric(s, "14", "TRAJECTORY IDS", 3.8, 5.72, 2.15, ORANGE)
metric(s, "0.291°", "REPORTED PEAK ERROR", 6.22, 5.72, 2.45, RED)
metric(s, "3", "VALIDATION RUNS", 8.94, 5.72, 2.2, GREEN)
footer(s, 16, "docs/GOLFCART_STABILITY_VALIDATION.md")

# 17 — GitLab / issue mapping
s = add_slide()
title(s, "Evidence mapped back to project work items", "Traceability")
evidence = [
    ("#74", "KAUST field test", "Six attempt labels + 16/22 Jul videos", "https://gitlab.kaust.edu.sa/dsa-kaust/projects/date-palm-automation/-/issues/74"),
    ("#17", "Harvest technique", "Regrip, proper-grip checks and slip detection", "https://gitlab.kaust.edu.sa/dsa-kaust/projects/date-palm-automation/-/issues/17"),
    ("#15", "Y2 data collection", "Outdoor videos tracked as field dataset", "https://gitlab.kaust.edu.sa/dsa-kaust/projects/date-palm-automation/-/issues/15"),
    ("#30", "LiDAR + mono plan", "Architecture path for RGB/depth separation", "https://gitlab.kaust.edu.sa/risc/manipulatorsdatepalm/-/work_items/30"),
    ("#7", "Detection → coordinates", "Open reliability requirement for pose success", "https://gitlab.kaust.edu.sa/risc/manipulatorsdatepalm/-/work_items/7"),
]
for i,(iid,name,result,url) in enumerate(evidence):
    y=1.55+i*1.02
    rect(s,.75,y,11.85,.82,WHITE,True,RGBColor(222,228,232))
    link=add_link(s,iid,url,.95,y+.20,.65,.28,13)
    textbox(s,name,1.68,y+.17,2.75,.34,14,NAVY,True)
    textbox(s,result,4.45,y+.17,7.75,.42,13,INK)
textbox(s, "Note: recent field attachments are cited by link; the separate private project was not imported into this workspace.", .9, 6.76, 11.6, .28, 10, MUTED, False, PP_ALIGN.CENTER)
footer(s, 17, "Repository GitLab inventory + README field-evidence references")

# 18 — Next actions
s = add_slide(False)
title(s, "Recommended next experiments", "Decision points", dark=True)
actions = [
    ("1", "Close the loop on slips", "Annotate contact/alignment/force for every failed attempt; tune regrip trigger."),
    ("2", "Run full GPU-contention validation", "Count planning runs and confirm no illegal-memory failures with the physical stack."),
    ("3", "Quantify harvesting KPIs", "Time-to-target, plan success, grasp success, slip rate and cycle time per bunch."),
    ("4", "Validate calibration repeatability", "Repeat dual-camera / hand-eye calibration and report reprojection + 3D error."),
    ("5", "Expand field dataset", "Capture comparable scenes, lighting, bunch maturity and target geometry."),
]
for i,(n,h,b) in enumerate(actions):
    y=1.52+i*1.04
    rect(s,.72,y,11.9,.83,RGBColor(24,51,74),True)
    rect(s,.88,y+.13,.55,.55,[CYAN,GREEN,ORANGE,RED,RGBColor(122,92,160)][i],True)
    textbox(s,n,.9,y+.19,.5,.24,13,WHITE,True,PP_ALIGN.CENTER)
    textbox(s,h,1.65,y+.15,2.75,.3,12,WHITE,True)
    textbox(s,b,4.35,y+.14,7.82,.43,11,RGBColor(210,224,234))
footer(s, 18, "Recommendations inferred from measured bottlenecks and labeled field outcomes")

# 19 — Closing
s = add_slide(False)
rect(s, 0, 0, 13.333, 7.5, NAVY)
textbox(s, "FROM WORKING COMPONENTS\nTO AN OPERABLE FIELD SYSTEM", .95, 1.15, 11.45, 1.55, 31, WHITE, True, PP_ALIGN.CENTER)
textbox(s, "The last three weeks produced measurable throughput gains, safer GPU coordination,\nstronger operator control, integrated gripper support and a growing evidence base.", 1.45, 3.1, 10.45, 1.1, 16, CYAN, True, PP_ALIGN.CENTER)
rect(s, 2.3, 4.75, 8.75, 1.1, BLUE, True)
textbox(s, "Next milestone: convert labeled field evidence into repeatable harvesting KPIs.", 2.58, 5.03, 8.2, .42, 17, WHITE, True, PP_ALIGN.CENTER)
textbox(s, "Questions / video review", 4.6, 6.35, 4.15, .4, 16, WHITE, True, PP_ALIGN.CENTER)
footer(s, 19)

# Core properties and save
prs.core_properties.title = "Date Palm Harvesting — Three-Week Progress Review"
prs.core_properties.subject = "UR10e harvesting progress, 6–27 July 2026"
prs.core_properties.author = "Date Palm Automation Team"
prs.core_properties.comments = "Generated from repository history, benchmarks, documentation and embedded field evidence."
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(OUT)
