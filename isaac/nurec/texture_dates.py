"""Build a photo-textured ellipsoid mesh for every split date (NuRec volumes cannot depth-sort
against the tree volume in Isaac Sim 5.1; meshes composite with NuRec correctly).

Each vertex takes its colour from the real frames: views where that surface point faces the
camera and projects inside a date mask (detect_dates.py --method fastsam), weighted by how
head-on the view is. Output per date: SCENE/split/date_XX_mesh.npz with vertices / normals in
the date body frame (scene orientation, date centre at origin), triangle indices, sRGB colours.

  /usr/bin/python3 isaac/nurec/texture_dates.py ~/nurec_data/scenes/palm3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from metric_frame import read_cameras, read_images


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--segments", type=int, default=48, help="Ellipsoid longitude segments.")
    parser.add_argument("--min_facing", type=float, default=0.3, help="cos(angle) between normal and view ray.")
    parser.add_argument("--top_views", type=int, default=6, help="Colour from the most head-on views.")
    return parser.parse_args()


def unit_sphere(segments: int):
    rings = segments // 2
    theta = np.linspace(0, np.pi, rings + 1)
    phi = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    points = np.stack([np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)], -1).reshape(-1, 3)
    faces = []
    for r in range(rings):
        for s in range(segments):
            a, b = r * segments + s, r * segments + (s + 1) % segments
            c, d = a + segments, b + segments
            faces += [(a, c, b), (b, c, d)]
    return points, np.array(faces, np.int32)


def main():
    args = parse_args()
    split = args.scene / "split"
    physics = json.loads((split / "dates_physics.json").read_text())
    located = {d["id"]: d for d in json.loads((args.scene / "dates.json").read_text())["dates"]}
    cameras = read_cameras(args.scene / "sparse_metric_txt" / "cameras.txt")
    views = read_images(args.scene / "sparse_metric_txt" / "images.txt")
    images, masks = {}, {}

    unit, faces = unit_sphere(args.segments)
    for date in physics["dates"]:
        axes = np.array(located[date["id"]]["axes"])  # rows: principal axes (scene frame)
        if np.linalg.det(axes) < 0:
            axes[2] *= -1  # SVD can return a reflection; a mirrored ellipsoid renders inside-out (black)
        radii = np.array(date["collider_radii"])
        local = unit * radii
        vertices = local @ axes  # body frame = scene orientation, origin at the date centre
        normals = (unit / radii) @ axes
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        world = vertices + np.array(date["centre"])

        samples = []  # per view: (colours, weights)
        for view in views:
            fx, fy, cx, cy, width, height = cameras[int(view["fields"][8])]
            cam = world @ view["R"].T + view["t"]
            if np.any(cam[:, 2] < 0.05):
                continue
            u = fx * cam[:, 0] / cam[:, 2] + cx
            v = fy * cam[:, 1] / cam[:, 2] + cy
            inside = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
            centre_cam = -view["R"].T @ view["t"]
            to_camera = centre_cam - world
            facing = (normals * to_camera).sum(1) / np.linalg.norm(to_camera, axis=1)
            name = view["name"]
            if name not in masks:
                mask_path = args.scene / "date_masks" / Path(name).with_suffix(".png")
                masks[name] = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.exists() else None
            if masks[name] is None:
                continue
            ok = inside & (facing > args.min_facing)
            ok[ok] &= masks[name][v[ok].astype(int), u[ok].astype(int)] > 0
            if ok.sum() < 0.05 * len(world):
                continue
            if name not in images:
                images[name] = cv2.cvtColor(cv2.imread(str(args.scene / "images" / name)), cv2.COLOR_BGR2RGB)
            colours = np.zeros((len(world), 3), np.float32)
            colours[ok] = cv2.remap(
                images[name], u[ok].astype(np.float32).reshape(-1, 1), v[ok].astype(np.float32).reshape(-1, 1),
                cv2.INTER_LINEAR,
            ).reshape(-1, 3) / 255.0
            samples.append((colours, np.where(ok, facing, 0.0)))

        if not samples:
            print(f"[WARN] date {date['id']:2d}: no usable views; using its mean splat colour")
            colour = np.tile(np.array(located[date["id"]].get("colour_rgb", [0.45, 0.3, 0.15])), (len(world), 1))
            coverage = 0.0
        else:
            colours = np.stack([c for c, _ in samples])
            weights = np.stack([w for _, w in samples])
            # Keep the most head-on views per vertex (less blur from oblique views).
            order = np.argsort(-weights, axis=0)[: args.top_views]
            top = np.take_along_axis(weights, order, 0)
            colour = (np.take_along_axis(colours, order[..., None], 0) * top[..., None]).sum(0) / np.maximum(
                top.sum(0), 1e-6
            )[:, None]
            seen = top.sum(0) > 0
            coverage = float(seen.mean())
            if not seen.all():  # unseen (e.g. back side): mean of the seen surface
                colour[~seen] = colour[seen].mean(0)

        out = split / f"date_{date['id']:02d}_mesh.npz"
        np.savez_compressed(out, vertices=vertices.astype(np.float32), normals=normals.astype(np.float32),
                            faces=faces, colours_srgb=colour.astype(np.float32))
        date["mesh"] = out.name
        print(f"[INFO] date {date['id']:2d}: {len(samples):3d} views, {100 * coverage:4.0f}% of surface photographed")

    (split / "dates_physics.json").write_text(json.dumps(physics, indent=2))
    print(f"[DONE] meshes for {len(physics['dates'])} dates in {split}")


if __name__ == "__main__":
    main()
