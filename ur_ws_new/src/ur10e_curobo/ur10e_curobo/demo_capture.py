"""Record operator corrections to the FINAL pose, for training a residual policy.

The pipeline drives to where it thinks the date is, which is wrong by a
range-dependent amount nobody has pinned down (see low_center_final_depth_offset
in config.py). This captures the correction instead of hand-tuning it: the arm
stops at its FINAL pose, the operator jogs the teach pendant to the pose that
would actually grasp, and the delta between the two is one training sample.

Deliberately NOT a full-trajectory demo recorder. The policy being trained fixes
a small endpoint error from a consistent start, so the sample that matters is
(what the camera saw, how far off the pipeline was). The jog path in between is
recorded too -- it is nearly free, and an ACT-style chunk model can use it -- but
nothing here assumes it will be.

The correction is recorded in the TOOL frame as well as base_link. Base-frame
deltas are what the offsets in config.py are expressed in, but a policy reading
the wrist camera sees the error in tool axes, and the two differ by the whole
wrist orientation.

Off by default: planner.demo_capture_enabled. A normal harvest must not stop and
wait for a human who is not there.
"""

import copy
import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def _pose_now(node) -> Optional[List[float]]:
    try:
        pose = node.get_end_effector_pose()
        return [float(v) for v in pose] if pose else None
    except Exception:
        return None


def _quat_rotate_inverse(quat_wxyz, vec):
    """Express a base-frame vector in the tool frame. quat is [w,x,y,z]."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    # conjugate rotates base -> tool
    x, y, z = -x, -y, -z
    vx, vy, vz = (float(v) for v in vec)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx)]


def _write_crop(image, path, name) -> bool:
    if image is None:
        return False
    try:
        import cv2
        return bool(cv2.imwrite(os.path.join(path, name), image))
    except Exception:
        return False


def capture_correction(node) -> Optional[str]:
    node._demo_confirm = None
    node._demo_capture_ready = False
    node._demo_capture_active = True
    try:
        return _capture_correction(node)
    finally:
        node._demo_capture_ready = False
        node._demo_capture_active = False


def _capture_correction(node) -> Optional[str]:
    """Hold at FINAL, wait for the operator's corrected pose, write one sample.

    Returns the directory written, or None if disabled/skipped/timed out.
    The caller stops the attempt if recording fails.
    """
    planner = getattr(getattr(node, "cfg", None), "planner", None)
    if not bool(getattr(planner, "demo_capture_enabled", False)):
        return None

    # Vision has just resumed. Do not pair a stale approach image with FINAL.
    requested_at = time.time()
    deadline = time.monotonic() + 3.0
    while (getattr(node, "_latest_grasp_crop_t", 0.0) <= requested_at
           or getattr(node, "_latest_grasp_crop", None) is None):
        if getattr(node, "stop_requested", False):
            return None
        if time.monotonic() >= deadline:
            node.get_logger().warn("[DEMO] no fresh FINAL crop; stopping attempt")
            return None
        time.sleep(0.02)
    before_image = node._latest_grasp_crop.copy()
    before_image_t = node._latest_grasp_crop_t
    before_pose = _pose_now(node)
    if before_pose is None:
        node.get_logger().warn("[DEMO] no TCP pose at FINAL; not recording")
        return None

    before = {
        "pose": before_pose,
        "joints": list(getattr(node, "current_joint_positions", None) or []),
        "goal_pose": getattr(node, "latest_goal_pose", None),
        "final_retarget_xyz": getattr(node, "_final_retarget_xyz", None),
        "crop_meta": copy.deepcopy(getattr(node, "_latest_grasp_crop_meta", None)),
        "crop_time": before_image_t,
        "fruit_radius": getattr(node, "latest_fruit_radius", None),
    }

    timeout = float(getattr(planner, "demo_capture_timeout_s", 0.0))
    hz = max(1.0, float(getattr(planner, "demo_capture_sample_hz", 10.0)))
    node._demo_capture_ready = True
    node.get_logger().warn(
        "[DEMO] holding at FINAL. Jog the pendant to the pose that would "
        "actually grasp, restore ROS control, then press Close in RViz to save "
        "and continue. Uncheck Demo to discard and stop this attempt.")

    path: List[Dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while timeout <= 0 or time.monotonic() < deadline:
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("[DEMO] stop requested; discarding")
            return None
        if getattr(node, "_demo_confirm", None) is not None:
            break
        pose = _pose_now(node)
        if pose is not None:
            path.append({"t": round(time.time(), 3), "pose": pose,
                         "joints": list(getattr(node, "current_joint_positions", None) or [])})
        time.sleep(1.0 / hz)

    accepted = getattr(node, "_demo_confirm", None)
    if accepted is None:
        node.get_logger().warn(f"[DEMO] no confirm within {timeout:.0f}s; discarding")
        return None
    if not accepted:
        node.get_logger().info("[DEMO] operator discarded this one")
        return None

    after_pose = _pose_now(node)
    if after_pose is None:
        node.get_logger().warn("[DEMO] no TCP pose after the jog; discarding")
        return None

    delta_base = [after_pose[i] - before_pose[i] for i in range(3)]
    # Tool frame of the pose the pipeline chose: that is the frame the wrist
    # camera observed the error from, and the frame a policy would correct in.
    delta_tool = (_quat_rotate_inverse(before_pose[3:7], delta_base)
                  if len(before_pose) >= 7 else None)

    root = os.path.expanduser(str(getattr(
        planner, "demo_capture_dir", "~/harvest_logs/demos")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out = os.path.join(root, stamp)
    try:
        os.makedirs(out, exist_ok=True)
    except Exception as exc:
        node.get_logger().warn(f"[DEMO] cannot create {out}: {exc}")
        return None

    sample = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "before": before,
        "after": {"pose": after_pose,
                  "joints": list(getattr(node, "current_joint_positions", None) or [])},
        "correction_base_m": delta_base,
        "correction_tool_m": delta_tool,
        "correction_mm": round(math.dist([0, 0, 0], delta_base) * 1000.0, 2),
        "range_m": (round(math.dist([0, 0, 0], before["goal_pose"][:3]), 4)
                    if before.get("goal_pose") else None),
        "jog_path": path,
        "jog_samples": len(path),
    }
    try:
        with open(os.path.join(out, "sample.json"), "w") as handle:
            json.dump(sample, handle, indent=1, default=str)
    except Exception as exc:
        node.get_logger().warn(f"[DEMO] cannot write sample.json: {exc}")
        return None
    if not _write_crop(before_image, out, "crop_before.jpg"):
        node.get_logger().warn("[DEMO] cannot save FINAL crop; stopping attempt")
        os.remove(os.path.join(out, "sample.json"))
        return None
    _write_crop(getattr(node, "_latest_grasp_crop", None), out, "crop_after.jpg")

    node.get_logger().warn(
        f"[DEMO] correction {sample['correction_mm']:.1f}mm "
        f"base=[{delta_base[0]*1000:+.1f},{delta_base[1]*1000:+.1f},"
        f"{delta_base[2]*1000:+.1f}]mm"
        + (f" tool=[{delta_tool[0]*1000:+.1f},{delta_tool[1]*1000:+.1f},"
           f"{delta_tool[2]*1000:+.1f}]mm" if delta_tool else "")
        + f" at {sample['range_m']}m -> {out}")
    return out


if __name__ == "__main__":
    # Identity: no rotation, tool frame == base frame.
    got = _quat_rotate_inverse([1.0, 0.0, 0.0, 0.0], [0.01, -0.02, 0.03])
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, [0.01, -0.02, 0.03])), got

    # 90deg about base Z: a +x base offset is -y... in the tool frame.
    s = math.sqrt(0.5)
    got = _quat_rotate_inverse([s, 0.0, 0.0, s], [1.0, 0.0, 0.0])
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, [0.0, -1.0, 0.0])), got

    # Rotation preserves length, whatever the axis.
    got = _quat_rotate_inverse([0.5, 0.5, 0.5, 0.5], [0.03, 0.01, -0.02])
    assert abs(math.dist([0, 0, 0], got) - math.dist([0, 0, 0], [0.03, 0.01, -0.02])) < 1e-9
    print("demo_capture self-check OK")
