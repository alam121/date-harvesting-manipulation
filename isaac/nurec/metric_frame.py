"""Put a COLMAP text model in metres and in the first left camera's ROS body frame.

COLMAP reconstructions have arbitrary scale and orientation. The stereo baseline from
CameraInfo fixes the scale; the frame is anchored at the first left image with ROS body
axes (x forward, y left, z up), so the NuRec asset can be placed in Isaac Sim at the pose
the camera had when recording started.

  python metric_frame.py SPARSE_TXT_IN SPARSE_TXT_OUT stereo_calibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# Optical (x right, y down, z forward) -> ROS body (x forward, y left, z up).
R_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def rot_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    trace = np.trace(r)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return tuple(q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1))


def read_images(path: Path):
    """images.txt: header lines, then pairs of (pose line, 2D points line)."""
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    images = []
    for pose_line, points_line in zip(lines[0::2], lines[1::2]):
        fields = pose_line.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
        images.append({
            "fields": fields,
            "name": fields[9],
            "R": quat_to_rot(qw, qx, qy, qz),
            "t": np.array([tx, ty, tz]),
            "points": points_line,
        })
    return images


def read_cameras(path: Path) -> dict[int, tuple[float, float, float, float, int, int]]:
    cameras = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if fields[1] not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise SystemExit(f"camera model {fields[1]} not supported (rectified PINHOLE expected)")
        width, height = int(fields[2]), int(fields[3])
        params = list(map(float, fields[4:]))
        fx, fy, cx, cy = params if fields[1] == "PINHOLE" else (params[0], params[0], params[1], params[2])
        cameras[int(fields[0])] = (fx, fy, cx, cy, width, height)
    return cameras


def center(image) -> np.ndarray:
    return -image["R"].T @ image["t"]


def main():
    src, dst, calib_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    baseline = json.loads(calib_path.read_text())["baseline_m"]
    images = read_images(src / "images.txt")
    by_name = {image["name"]: image for image in images}

    pairs = []
    for name, image in by_name.items():
        if name.startswith("left/") and (partner := by_name.get("right/" + name[5:])):
            relative = partner["R"] @ image["R"].T
            angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
            pairs.append((name, partner["name"], np.linalg.norm(center(image) - center(partner)), angle))
    if len(pairs) < 5:
        raise SystemExit(f"Only {len(pairs)} stereo pairs registered together; cannot recover metric scale.")
    distances = np.array([p[2] for p in pairs])
    scale = baseline / np.median(distances)

    # A rigid rig must reproduce the baseline and near-zero relative rotation in every pair;
    # pairs that do not have at least one misregistered image, which would smear the splat.
    rejected = {name for left, right, dist, angle in pairs
                if abs(dist * scale / baseline - 1) > 0.05 or angle > 1.0 for name in (left, right)}
    kept = np.array([d for left, _, d, _ in pairs if left not in rejected]) * scale
    spread = np.median(np.abs(kept / baseline - 1))
    print(
        f"[INFO] scale {scale:.5f} from {len(pairs)} pairs; {len(pairs) - len(kept)} inconsistent pairs dropped; "
        f"baseline error of kept pairs {100 * spread:.2f}% (median)"
    )
    if len(kept) < 0.8 * len(pairs):
        print("[WARN] More than 20% of stereo pairs are inconsistent; SfM likely drifted. Check the COLMAP model.")

    images = [image for image in images if image["name"] not in rejected]
    dropped_ids = {int(image["fields"][0]) for image in by_name.values() if image["name"] in rejected}

    first = by_name[min(name for name in by_name if name.startswith("left/") and name not in rejected)]
    r_anchor = R_BODY_FROM_OPTICAL @ first["R"]
    t_anchor = R_BODY_FROM_OPTICAL @ first["t"]

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "cameras.txt").write_text((src / "cameras.txt").read_text())

    out = []
    kept_points = set()
    for line in (src / "points3D.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split()
        track = [(fields[i], fields[i + 1]) for i in range(8, len(fields), 2) if int(fields[i]) not in dropped_ids]
        if len(track) < 2:
            continue
        xyz = scale * (r_anchor @ np.array(list(map(float, fields[1:4]))) + t_anchor)
        fields = fields[:1] + [f"{v:.9g}" for v in xyz] + fields[4:8] + [v for pair in track for v in pair]
        kept_points.add(fields[0])
        out.append(" ".join(fields))
    (dst / "points3D.txt").write_text("\n".join(out) + "\n")

    out = ["# Image list with two lines of data per image:"]
    for image in images:
        r_new = image["R"] @ r_anchor.T
        t_new = scale * (image["t"] - r_new @ t_anchor)
        fields = image["fields"]
        fields[1:8] = [f"{v:.12g}" for v in (*rot_to_quat(r_new), *t_new)]
        observations = image["points"].split()
        for i in range(2, len(observations), 3):
            if observations[i] not in kept_points:
                observations[i] = "-1"
        out += [" ".join(fields), " ".join(observations)]
    (dst / "images.txt").write_text("\n".join(out) + "\n")

    (dst / "metric_frame.json").write_text(json.dumps({
        "scale": scale,
        "anchor_image": first["name"],
        "frame": "first left camera, ROS body axes (x forward, y left, z up), metres",
        "registered_stereo_pairs": len(pairs),
        "dropped_images": sorted(rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
