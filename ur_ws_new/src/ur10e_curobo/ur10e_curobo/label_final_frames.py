"""Label FINAL frames for detector training, zoomed on the predicted date.

Frames come from ~/harvest_logs/final_frames (saved during FINAL by main and
vision, with the track's predicted pixel). Each is shown as a 640x640 crop
around that prediction, pre-filled with YOLO's own detections there; you fix
the boxes so that EVERY date in the crop is labelled -- an unlabelled date
teaches the model "not a date". Mark trunk/bunch too when they are in the
crop: the engine has those classes as well.

Mouse:  left-drag      draw a box of the current class
        right-click    delete the box under the cursor
Keys:   SPACE / ENTER  save and next
        n              save, recording that the target date is NOT visible
        d / t / b      current class: date / trunk / bunch
        z              undo last added box
        s              skip (not saved)          q  quit

Writes ~/harvest_logs/final_frames_dataset/:
  images/*.jpg          the crops (--crop px square, default 640)
  labels/*.txt          YOLO segmentation (dates as ellipses inscribed in
                        the box, trunk/bunch as rectangles) -- class ids
                        match the engine: 0 bunch, 1 date, 2 trunk
  labels_box/*.txt      YOLO detection boxes
  index.jsonl           one record per labelled frame (source, crop, target)

Usage:
    python3 -m ur10e_curobo.label_final_frames [--since 20261102_1556]
            [--kind miss|hit|all] [--crop 400] [--prefill 0.25]
"""
import argparse
import glob
import json
import math
import os

import cv2

CLASSES = {"b": 0, "d": 1, "t": 2}
NAMES = {0: "bunch", 1: "date", 2: "trunk"}
COLORS = {0: (255, 200, 0), 1: (0, 220, 0), 2: (255, 0, 255)}
CROP = 640
SCALE = 1.5          # display magnification


def crop_window(pred, width, height, size=CROP):
    """Top-left of a size x size crop centred on pred, kept inside the image."""
    cx, cy = (pred if pred else (width / 2.0, height / 2.0))
    x0 = int(round(min(max(cx - size / 2.0, 0), max(0, width - size))))
    y0 = int(round(min(max(cy - size / 2.0, 0), max(0, height - size))))
    return x0, y0


def prefill(raw, x0, y0, size=CROP, min_conf=0.25):
    """YOLO's date boxes (image px) clipped into the crop: [cls, x1, y1, x2, y2]."""
    out = []
    for x1, y1, x2, y2, conf in raw or []:
        if conf < min_conf:
            continue
        bx1, by1 = max(0.0, x1 - x0), max(0.0, y1 - y0)
        bx2, by2 = min(float(size), x2 - x0), min(float(size), y2 - y0)
        if bx2 - bx1 >= 4 and by2 - by1 >= 4:
            out.append([1, bx1, by1, bx2, by2])
    return out


def label_lines(boxes, size=CROP, ellipse_points=16):
    """(segmentation lines, detection lines) in YOLO normalised format."""
    seg, det = [], []
    for cls, x1, y1, x2, y2 in boxes:
        cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
        det.append(f"{cls} {cx / size:.6f} {cy / size:.6f} {w / size:.6f} {h / size:.6f}")
        if cls == 1:
            pts = [(cx + w / 2.0 * math.cos(2 * math.pi * k / ellipse_points),
                    cy + h / 2.0 * math.sin(2 * math.pi * k / ellipse_points))
                   for k in range(ellipse_points)]
        else:
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        seg.append(f"{cls} " + " ".join(
            f"{min(max(x / size, 0.0), 1.0):.6f} {min(max(y / size, 0.0), 1.0):.6f}"
            for x, y in pts))
    return seg, det


def _frames(root, since, kind):
    for meta_path in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
        attempt = os.path.basename(os.path.dirname(meta_path))
        name = os.path.splitext(os.path.basename(meta_path))[0]
        if attempt < since or (kind != "all" and f"_{kind}_" not in name):
            continue
        img = meta_path[:-5] + ".jpg"
        if os.path.isfile(img):
            yield attempt, name, img, meta_path


def _draw(crop, boxes, pred, sigma, cls, title):
    view = cv2.resize(crop, None, fx=SCALE, fy=SCALE)
    for c, x1, y1, x2, y2 in boxes:
        cv2.rectangle(view, (int(x1 * SCALE), int(y1 * SCALE)),
                      (int(x2 * SCALE), int(y2 * SCALE)), COLORS[c], 2)
        cv2.putText(view, NAMES[c], (int(x1 * SCALE), int(y1 * SCALE) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[c], 1)
    if pred is not None:
        p = (int(pred[0] * SCALE), int(pred[1] * SCALE))
        cv2.drawMarker(view, p, (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
        if sigma:
            cv2.circle(view, p, max(6, int(min(sigma, 2 * CROP) * SCALE)), (0, 255, 255), 1)
    cv2.putText(view, f"{title}  class={NAMES[cls]}  [space]save [n]not-visible "
                "[d/t/b]class [z]undo [s]skip [q]quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return view


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="~/harvest_logs/final_frames")
    ap.add_argument("--out", default="~/harvest_logs/final_frames_dataset")
    ap.add_argument("--since", default="")
    ap.add_argument("--kind", default="all", choices=("miss", "hit", "all"))
    ap.add_argument("--crop", type=int, default=CROP,
                    help="crop size in px; smaller = fewer dates to label per frame")
    ap.add_argument("--prefill", type=float, default=0.25,
                    help="pre-fill YOLO date boxes at or above this confidence")
    args = ap.parse_args(argv)
    root, out = os.path.expanduser(args.dir), os.path.expanduser(args.out)
    for sub in ("images", "labels", "labels_box"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    index_path = os.path.join(out, "index.jsonl")
    done = set()
    if os.path.isfile(index_path):
        with open(index_path) as fh:
            done = {json.loads(line)["source"] for line in fh if line.strip()}
    todo = [f for f in _frames(root, args.since, args.kind) if f[2] not in done]
    if not todo:
        print(f"Nothing to label in {root}.")
        return
    print(f"{len(todo)} frame(s) to label.")

    win = "label_final_frames"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {"boxes": [], "added": [], "drag": None, "cls": 1}

    def on_mouse(event, x, y, flags, _):
        x, y = x / SCALE, y / SCALE
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["drag"] = state["drag"][:2] + (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x1, y1, x2, y2 = state["drag"]
            state["drag"] = None
            box = [state["cls"], min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            if box[3] - box[1] >= 4 and box[4] - box[2] >= 4:
                state["boxes"].append(box)
                state["added"].append(box)
        elif event == cv2.EVENT_RBUTTONDOWN:
            hits = [b for b in state["boxes"] if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
            if hits:
                state["boxes"].remove(min(hits, key=lambda b: (b[3] - b[1]) * (b[4] - b[2])))

    cv2.setMouseCallback(win, on_mouse)
    saved = 0
    for i, (attempt, name, img_path, meta_path) in enumerate(todo, 1):
        img = cv2.imread(img_path)
        meta = json.load(open(meta_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        pred = meta.get("predicted_px")
        size = min(args.crop, w, h)
        x0, y0 = crop_window(pred, w, h, size)
        crop = img[y0:y0 + size, x0:x0 + size].copy()
        pred_c = [pred[0] - x0, pred[1] - y0] if pred else None
        state.update(boxes=prefill(meta.get("raw_detections_xyxy_conf"), x0, y0,
                                   size=size, min_conf=args.prefill),
                     added=[], drag=None)
        title = f"[{i}/{len(todo)}] {attempt}/{name}"
        key = None
        while True:
            boxes = state["boxes"] + ([[state["cls"], *state["drag"]]] if state["drag"] else [])
            cv2.imshow(win, _draw(crop, boxes, pred_c, meta.get("predicted_sigma_px"),
                                  state["cls"], title))
            k = cv2.waitKey(30) & 0xFF
            if k in (ord(" "), 13, ord("n"), ord("s"), ord("q")):
                key = chr(k) if k != 13 else " "
                break
            if k < 128 and chr(k) in CLASSES:
                state["cls"] = CLASSES[chr(k)]
            elif k == ord("z") and state["added"]:
                b = state["added"].pop()
                if b in state["boxes"]:
                    state["boxes"].remove(b)
        if key == "q":
            break
        if key == "s":
            continue
        stem = f"{attempt}_{name}"
        cv2.imwrite(os.path.join(out, "images", stem + ".jpg"), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        seg, det = label_lines(state["boxes"], size=size)
        with open(os.path.join(out, "labels", stem + ".txt"), "w") as fh:
            fh.write("\n".join(seg) + ("\n" if seg else ""))
        with open(os.path.join(out, "labels_box", stem + ".txt"), "w") as fh:
            fh.write("\n".join(det) + ("\n" if det else ""))
        covers = bool(pred_c) and any(
            b[0] == 1 and b[1] <= pred_c[0] <= b[3] and b[2] <= pred_c[1] <= b[4]
            for b in state["boxes"])
        with open(index_path, "a") as fh:
            fh.write(json.dumps({
                "source": img_path, "image": stem + ".jpg", "crop_xy": [x0, y0],
                "kind": meta.get("kind"), "track_status": meta.get("track_status"),
                "predicted_px_in_crop": pred_c,
                "target_visible": key != "n",
                "date_box_at_prediction": covers,
                "n_boxes": {NAMES[c]: sum(1 for b in state["boxes"] if b[0] == c)
                            for c in NAMES},
            }) + "\n")
        saved += 1
    cv2.destroyAllWindows()
    print(f"{saved} frame(s) labelled -> {out}")


if __name__ == "__main__":
    main()
