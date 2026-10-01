"""Find individual dates in a trained NuRec scene by multi-view voting over the Gaussians.

For every image, the Gaussians whose centres project inside a date box and lie on the front
surface there (within --surface_depth of the nearest ones) get a vote. A Gaussian is a date
Gaussian when it is voted in enough of the views that see it; those are clustered into
individual dates. Everything is in the scene's metric frame (first left camera, ROS body axes).

  cd ~/3dgrut && source .venv/bin/activate
  python ~/manipulatorsdatepalm/isaac/nurec/localize_dates.py ~/nurec_data/scenes/palm3

Writes SCENE/dates.json (one entry per date) and SCENE/date_labels.npy (date id per Gaussian,
-1 for none), plus date_overlay_*.jpg previews.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_frame import read_cameras, read_images  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: newest run's ckpt_last.pt")
    parser.add_argument("--min_opacity", type=float, default=0.05, help="MCMC leaves most Gaussians faint (median ~0.03).")
    parser.add_argument("--surface_depth", type=float, default=0.03, help="Front-surface slab in a box [m].")
    parser.add_argument("--box_shrink", type=float, default=0.15, help="Shrink boxes by this fraction per side.")
    parser.add_argument("--min_votes", type=int, default=3)
    parser.add_argument("--min_ratio", type=float, default=0.3, help="Votes / views in which the Gaussian is visible.")
    parser.add_argument("--cluster_radius", type=float, default=0.008, help="Link distance for clustering [m].")
    parser.add_argument("--min_cluster", type=int, default=30, help="Minimum Gaussians per date.")
    parser.add_argument("--max_extent", type=float, default=0.08, help="Reject clusters larger than this [m].")
    return parser.parse_args()


def load_gaussians(checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    positions = state["positions"].detach().float().numpy()
    density = state["density"].detach().float()
    opacity = torch.sigmoid(density).reshape(-1).numpy()
    colour = None
    if "features_albedo" in state:
        # Degree-0 SH to RGB: 0.5 + C0 * f.
        colour = np.clip(0.5 + 0.28209479 * state["features_albedo"].detach().float().numpy()[:, :3], 0, 1)
    return positions, opacity, colour


def project(points, image, camera):
    fx, fy, cx, cy, width, height = camera
    cam = points @ image["R"].T + image["t"]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * cam[:, 0] / z + cx
        v = fy * cam[:, 1] / z + cy
    inside = (z > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, z, inside


def main():
    args = parse_args()
    scene = args.scene
    checkpoint = args.checkpoint or sorted((scene / "runs" / "nurec").glob("*/ckpt_last.pt"))[-1]
    detection_file = json.loads((scene / "date_detections.json").read_text())
    detections = detection_file["boxes"]
    positions, opacity, colour = load_gaussians(checkpoint)
    solid = opacity >= args.min_opacity
    print(f"[INFO] {len(positions)} Gaussians, {solid.sum()} with opacity >= {args.min_opacity} ({checkpoint.parent.name})")

    cameras = read_cameras(scene / "sparse_metric_txt" / "cameras.txt")
    images = read_images(scene / "sparse_metric_txt" / "images.txt")
    index = np.flatnonzero(solid)
    points = positions[index]
    votes = np.zeros(len(points), np.int32)
    seen = np.zeros(len(points), np.int32)

    mask_dir = scene / detection_file["masks"] if detection_file.get("masks") else None
    for image in images:
        camera = cameras[int(image["fields"][8])]
        u, v, z, inside = project(points, image, camera)
        seen += inside
        if mask_dir is not None:
            # Instance masks: vote the Gaussians on the front surface of each date mask.
            labels_image = cv2.imread(str(mask_dir / Path(image["name"]).with_suffix(".png")), cv2.IMREAD_UNCHANGED)
            if labels_image is None:
                continue
            instance = np.zeros(len(points), np.int32)
            instance[inside] = labels_image[v[inside].astype(int), u[inside].astype(int)]
            for label in np.unique(instance[instance > 0]):
                in_mask = instance == label
                if in_mask.sum() < 5:
                    continue
                front = np.percentile(z[in_mask], 5)
                votes[in_mask & (z <= front + args.surface_depth)] += 1
            continue
        for x0, y0, x1, y1, _ in detections.get(image["name"], []):
            dx, dy = (x1 - x0) * args.box_shrink, (y1 - y0) * args.box_shrink
            in_box = inside & (u >= x0 + dx) & (u <= x1 - dx) & (v >= y0 + dy) & (v <= y1 - dy)
            if in_box.sum() < 5:
                continue
            front = np.percentile(z[in_box], 5)
            votes[in_box & (z <= front + args.surface_depth)] += 1

    ratio = votes / np.maximum(seen, 1)
    is_date = (votes >= args.min_votes) & (ratio >= args.min_ratio)
    date_points = points[is_date]
    print(f"[INFO] {is_date.sum()} date Gaussians (votes >= {args.min_votes}, ratio >= {args.min_ratio})")
    if len(date_points) == 0:
        raise SystemExit("No date Gaussians found; check date_detections.json or lower --min_ratio.")

    # Single-linkage clustering: connect Gaussians closer than cluster_radius.
    pairs = cKDTree(date_points).query_pairs(args.cluster_radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(date_points),) * 2)
    _, component = connected_components(graph, directed=False)

    labels = np.full(len(positions), -1, np.int32)
    dates = []
    too_small, too_large = 0, []
    for comp in np.unique(component):
        members = np.flatnonzero(component == comp)
        if len(members) < args.min_cluster:
            too_small += 1
            continue
        cluster = date_points[members]
        centre = cluster.mean(0)
        _, spread, axes = np.linalg.svd(cluster - centre, full_matrices=False)  # rows of axes: principal axes
        extents = 4.0 * spread / np.sqrt(len(cluster))  # ~ +-2 sigma full width per principal axis
        if extents.max() > args.max_extent:
            too_large.append(round(float(extents.max()) * 100, 1))
            continue
        date_id = len(dates)
        labels[index[np.flatnonzero(is_date)[members]]] = date_id
        entry = {
            "id": date_id,
            "centre": centre.round(5).tolist(),
            "extents": extents.round(4).tolist(),
            "axes": axes.round(4).tolist(),
            "n_gaussians": int(len(members)),
            "mean_votes": float(votes[np.flatnonzero(is_date)[members]].mean()),
        }
        if colour is not None:
            entry["colour_rgb"] = colour[index[np.flatnonzero(is_date)[members]]].mean(0).round(3).tolist()
        dates.append(entry)

    np.save(scene / "date_labels.npy", labels)
    (scene / "dates.json").write_text(json.dumps({
        "frame": "scene metric frame (first left camera, ROS body axes), metres",
        "checkpoint": str(checkpoint),
        "dates": dates,
    }, indent=2))
    sizes = np.array([d["extents"] for d in dates]) * 100 if dates else np.zeros((0, 3))
    print(f"[INFO] {len(dates)} dates; median extents (cm) {np.median(sizes, 0).round(1).tolist() if len(sizes) else '-'}")
    print(f"[INFO] rejected: {too_small} clusters < {args.min_cluster} Gaussians, "
          f"{len(too_large)} larger than {100 * args.max_extent:.0f} cm {sorted(too_large, reverse=True)[:8]}")

    # Previews: date centres projected into a few left images, with the detector boxes.
    lefts = [image for image in images if image["name"].startswith("left/")]
    centres = np.array([d["centre"] for d in dates])
    for image in lefts[:: max(1, len(lefts) // 4)][:4]:
        frame = cv2.imread(str(scene / "images" / image["name"]))
        for x0, y0, x1, y1, _ in detections.get(image["name"], []):
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 1)
        u, v, _, inside = project(centres, image, cameras[int(image["fields"][8])])
        for i in np.flatnonzero(inside):
            cv2.circle(frame, (int(u[i]), int(v[i])), 7, (60, 255, 60), 2)
            cv2.putText(frame, str(dates[i]["id"]), (int(u[i]) + 8, int(v[i]) - 8), 0, 0.5, (60, 255, 60), 1, cv2.LINE_AA)
        cv2.imwrite(str(scene / f"date_overlay_{Path(image['name']).stem}.jpg"), frame)
    print(f"[DONE] {scene / 'dates.json'}")


if __name__ == "__main__":
    main()
