"""Turn a recorded stereo rosbag into a COLMAP scene folder for NuRec training.

Writes:
  <scene>/images/left/NNNNN.jpg, <scene>/images/right/NNNNN.jpg  (synced, sharpest per time bin)
  <scene>/stereo_calibration.json                               (intrinsics + baseline, metres)
  <scene>/pairs.txt                                             (COLMAP match list for the stereo sequence)
  <scene>/masks/{left,right}/NNNNN.jpg.png                      (--mask_auto / --mask_bottom; 0 = ignore)

Run with the ROS system interpreter: $NUREC_ROS_PYTHON extract_frames.py BAG SCENE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
INFO_TYPE = "sensor_msgs/msg/CameraInfo"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--fps", type=float, default=2.0, help="Stereo pairs kept per second of recording.")
    parser.add_argument("--max_pairs", type=int, default=300)
    parser.add_argument("--width", type=int, default=0, help="Resize to this width (0 keeps native).")
    parser.add_argument("--sync_tolerance", type=float, default=0.01, help="Max left/right stamp gap [s].")
    parser.add_argument("--match_window", type=int, default=12, help="Temporal neighbours matched per frame.")
    parser.add_argument(
        "--min_change",
        type=float,
        default=3.0,
        help="Skip pairs whose left thumbnail differs from the last kept one by less than this (0-255 mean).",
    )
    parser.add_argument(
        "--mask_bottom",
        type=float,
        default=0.0,
        help="Ignore this fraction of image height at the bottom (wrist camera sees the gripper there).",
    )
    parser.add_argument(
        "--mask_auto",
        action="store_true",
        help="Detect robot parts fixed in the image (gripper) and mask them; --mask_bottom is the fallback.",
    )
    parser.add_argument("--left_image", default="")
    parser.add_argument("--right_image", default="")
    parser.add_argument("--left_info", default="")
    parser.add_argument("--right_info", default="")
    return parser.parse_args()


def open_reader(bag: Path, topics: list[str] | None = None):
    storage_id = "mcap" if any(bag.glob("*.mcap")) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


def resolve_topics(args) -> dict[str, str]:
    types = {t.name: t.type for t in open_reader(args.bag).get_all_topics_and_types()}

    def find(side: str, wanted: set[str], override: str) -> str:
        if override:
            return override
        candidates = [
            name for name, kind in types.items()
            if kind in wanted and side in name and not any(k in name for k in ("depth", "disparity", "confidence"))
        ]
        if not candidates:
            raise SystemExit(f"No {side} topic of type {sorted(wanted)} in bag. Topics: {types}")
        return sorted(candidates, key=lambda n: ("rect" not in n, len(n)))[0]

    topics = {
        "left_image": find("left", IMAGE_TYPES, args.left_image),
        "right_image": find("right", IMAGE_TYPES, args.right_image),
        "left_info": find("left", {INFO_TYPE}, args.left_info),
        "right_info": find("right", {INFO_TYPE}, args.right_info),
    }
    msg_types = {role: get_message(types[topic]) for role, topic in topics.items()}
    return topics, msg_types


def decode(msg) -> np.ndarray:
    if hasattr(msg, "format"):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[msg.encoding]
    image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = image[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    conversion = {
        "mono8": cv2.COLOR_GRAY2BGR,
        "rgb8": cv2.COLOR_RGB2BGR,
        "rgba8": cv2.COLOR_RGBA2BGR,
        "bgra8": cv2.COLOR_BGRA2BGR,
    }.get(msg.encoding)
    return cv2.cvtColor(image, conversion) if conversion is not None else image.copy()


def stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def sharpness_and_thumbnail(image: np.ndarray) -> tuple[float, np.ndarray]:
    gray = cv2.cvtColor(cv2.resize(image, (640, int(640 * image.shape[0] / image.shape[1]))), cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), thumbnail


def scan(args, topics, msg_types):
    """First pass: stamps and sharpness per image, first CameraInfo per side."""
    frames = {"left_image": [], "right_image": []}
    infos = {}
    reader = open_reader(args.bag, list(topics.values()))
    role_by_topic = {topic: role for role, topic in topics.items()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        role = role_by_topic[topic]
        msg = deserialize_message(data, msg_types[role])
        if role.endswith("info"):
            infos.setdefault(role, msg)
            continue
        entry = (stamp(msg),)
        if role == "left_image":
            entry += sharpness_and_thumbnail(decode(msg))
        frames[role].append(entry)
    if "left_info" not in infos or "right_info" not in infos:
        raise SystemExit("Bag has no CameraInfo for one side; record the camera_info topics too.")
    return frames, infos


def select_pairs(args, frames) -> list[tuple[float, float]]:
    right_stamps = np.array(sorted(s for (s,) in frames["right_image"]))
    if right_stamps.size == 0:
        raise SystemExit("No right images in bag.")
    bins: dict[int, tuple[float, float, float, np.ndarray]] = {}
    for left_stamp, score, thumbnail in frames["left_image"]:
        nearest = right_stamps[np.argmin(np.abs(right_stamps - left_stamp))]
        if abs(nearest - left_stamp) > args.sync_tolerance:
            continue
        key = int(left_stamp * args.fps)
        if key not in bins or score > bins[key][2]:
            bins[key] = (left_stamp, float(nearest), score, thumbnail)

    # Frames where the camera sat still add no parallax and bias SfM initialisation.
    chosen = []
    for key in sorted(bins):
        if chosen and np.abs(bins[key][3] - chosen[-1][3]).mean() < args.min_change:
            continue
        chosen.append(bins[key])
    print(f"[INFO] {len(bins)} synced pairs, {len(bins) - len(chosen)} dropped as static (--min_change)")
    if len(chosen) > args.max_pairs:
        keep = np.linspace(0, len(chosen) - 1, args.max_pairs).round().astype(int)
        chosen = [chosen[i] for i in keep]
    # Drop the blurriest 10%: they mostly come from fast rotations and hurt both SfM and the splat.
    if len(chosen) > 20:
        floor = np.percentile([c[2] for c in chosen], 10)
        chosen = [c for c in chosen if c[2] >= floor]
    return [(left, right) for left, right, _, _ in chosen]


def write_images(args, topics, msg_types, pairs) -> tuple[int, int, float]:
    wanted = {"left_image": {}, "right_image": {}}
    for index, (left, right) in enumerate(pairs):
        wanted["left_image"][left] = index
        wanted["right_image"][right] = index
    for side in ("left", "right"):
        (args.scene / "images" / side).mkdir(parents=True, exist_ok=True)

    scale = 1.0
    size = None
    reader = open_reader(args.bag, [topics["left_image"], topics["right_image"]])
    role_by_topic = {topics["left_image"]: "left_image", topics["right_image"]: "right_image"}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        role = role_by_topic[topic]
        msg = deserialize_message(data, msg_types[role])
        index = wanted[role].pop(stamp(msg), None)
        if index is None:
            continue
        image = decode(msg)
        if args.width and image.shape[1] != args.width:
            scale = args.width / image.shape[1]
            image = cv2.resize(image, (args.width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        size = (image.shape[1], image.shape[0])
        path = args.scene / "images" / role.split("_")[0] / f"{index:05d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return size[0], size[1], scale


def band_mask(width: int, height: int, fraction: float) -> np.ndarray:
    mask = np.full((height, width), 255, np.uint8)
    mask[height - round(fraction * height):] = 0
    return mask


def robot_mask(images: list[Path], width: int, height: int) -> np.ndarray | None:
    """Mask robot parts rigidly attached to the camera (wrist camera: gripper at the bottom).

    While the camera moves, the scene blurs out in the per-pixel statistics but the gripper stays
    put: low temporal variance, and edges whose direction never changes. Uniform scene regions
    (ceiling, curtain) are also low-variance, so a region must also hold such fixed edges and
    touch the bottom border. Returns None when detection is implausible (e.g. camera barely moved).
    """
    w, h = 320, round(320 * height / width)
    acc = None
    for path in images:
        gray = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (w, h)).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        sample = np.stack([gx, gy, np.hypot(gx, gy), gray, gray * gray])
        acc = sample if acc is None else acc + sample
    acc /= len(images)
    fixed_edges = (np.hypot(acc[0], acc[1]) / (acc[2] + 1e-3) > 0.8) & (acc[2] > 15)
    still = (np.sqrt(np.maximum(acc[4] - acc[3] ** 2, 0)) < 9).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(still, connectivity=4)
    keep = np.zeros_like(still)
    for label in range(1, count):
        region = labels == label
        grown = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if region[-1].any() and fixed_edges[grown].sum() >= 10 and stats[label, cv2.CC_STAT_AREA] < 0.08 * region.size:
            keep[region] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if not 0.005 < keep.mean() < 0.15:
        return None
    return np.where(cv2.resize(keep, (width, height), interpolation=cv2.INTER_NEAREST) > 0, 0, 255).astype(np.uint8)


def write_masks(args, count: int, width: int, height: int) -> None:
    """COLMAP masks (masks/<image>.png, 0 = ignore); reconstruct.sh copies them next to images for 3dgrut."""
    for side in ("left", "right"):
        mask = None
        if args.mask_auto:
            mask = robot_mask(sorted((args.scene / "images" / side).glob("*.jpg")), width, height)
            if mask is None:
                print(f"[WARN] {side}: no plausible robot mask found; falling back to --mask_bottom {args.mask_bottom}")
            else:
                print(f"[INFO] {side}: robot mask covers {100 * (mask == 0).mean():.1f}% of the image")
        if mask is None:
            if args.mask_bottom <= 0:
                continue
            mask = band_mask(width, height, args.mask_bottom)
        (args.scene / "masks" / side).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            cv2.imwrite(str(args.scene / "masks" / side / f"{index:05d}.jpg.png"), mask)
        preview = cv2.imread(str(args.scene / "images" / side / f"{count // 2:05d}.jpg"))
        preview[mask == 0] = (0.4 * preview[mask == 0] + [0, 0, 150]).astype(np.uint8)
        cv2.imwrite(str(args.scene / f"mask_preview_{side}.jpg"), preview)


def calibration(infos, width: int, height: int, scale: float) -> dict:
    left, right = infos["left_info"], infos["right_info"]
    projection = np.array(right.p).reshape(3, 4)
    baseline = -projection[0, 3] / projection[0, 0] if projection[0, 0] else 0.0
    if baseline <= 0.0:
        raise SystemExit(
            "Right CameraInfo P has no baseline (P[0,3] == 0); metric scaling needs rectified stereo info."
        )

    def intrinsics(info):
        k = np.array(info.k).reshape(3, 3)
        return {
            "fx": k[0, 0] * scale,
            "fy": k[1, 1] * scale,
            "cx": k[0, 2] * scale,
            "cy": k[1, 2] * scale,
            "distortion_model": info.distortion_model,
            "d": [float(v) for v in info.d],
        }

    return {
        "width": width,
        "height": height,
        "baseline_m": float(baseline),
        "left": intrinsics(left),
        "right": intrinsics(right),
        "rectified": all(abs(v) < 1e-9 for v in list(left.d) + list(right.d)),
    }


def write_pairs(scene: Path, count: int, window: int) -> int:
    pairs = set()
    for i in range(count):
        pairs.add((f"left/{i:05d}.jpg", f"right/{i:05d}.jpg"))
        for j in range(i + 1, min(count, i + window + 1)):
            for a in ("left", "right"):
                for b in ("left", "right"):
                    pairs.add((f"{a}/{i:05d}.jpg", f"{b}/{j:05d}.jpg"))
    (scene / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    return len(pairs)


def main():
    args = parse_args()
    args.scene.mkdir(parents=True, exist_ok=True)
    topics, msg_types = resolve_topics(args)
    print("[INFO] Topics:", json.dumps(topics, indent=2))

    frames, infos = scan(args, topics, msg_types)
    pairs = select_pairs(args, frames)
    if len(pairs) < 20:
        raise SystemExit(f"Only {len(pairs)} synced stereo pairs; record longer or raise --sync_tolerance.")
    width, height, scale = write_images(args, topics, msg_types, pairs)
    if args.mask_auto or args.mask_bottom > 0:
        write_masks(args, len(pairs), width, height)

    calib = calibration(infos, width, height, scale)
    calib["topics"] = topics
    calib["pairs"] = len(pairs)
    calib["left_stamps"] = [left for left, _ in pairs]
    (args.scene / "stereo_calibration.json").write_text(json.dumps(calib, indent=2))
    matches = write_pairs(args.scene, len(pairs), args.match_window)

    print(f"[INFO] {len(pairs)} stereo pairs at {width}x{height}, baseline {calib['baseline_m']:.4f} m")
    print(f"[INFO] {matches} match pairs -> {args.scene / 'pairs.txt'}")
    if not calib["rectified"]:
        print("[WARN] Images carry distortion; COLMAP will use the OPENCV model with CameraInfo D.")


if __name__ == "__main__":
    main()
