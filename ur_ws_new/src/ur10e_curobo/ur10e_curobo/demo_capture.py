"""Record operator corrections to the APPROACH pose, for training a residual policy.

The pipeline drives to where it thinks the date is, which is wrong by a
range-dependent amount nobody has pinned down (see low_center_final_depth_offset
in config.py). This captures the correction instead of hand-tuning it: the arm
stops at its APPROACH pose, the operator jogs the teach pendant to the pose that
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


def target_preview(image, metadata, goal):
    """Label every candidate; proximity is a suggestion, never verified identity."""
    import cv2
    marked = image.copy()
    candidates = copy.deepcopy(metadata.get('candidates', []))
    width, height = metadata.get('source_size', [image.shape[1], image.shape[0]])
    for item in candidates:
        item['goal_distance_m'] = math.dist(item['base_xyz'], goal) if goal else None
    ranked = sorted(candidates, key=lambda c: c['goal_distance_m'] if c['goal_distance_m'] is not None else math.inf)
    suggested = (ranked[0] if ranked and goal and ranked[0]['goal_distance_m'] <= .05
                 and (len(ranked) == 1 or ranked[1]['goal_distance_m'] - ranked[0]['goal_distance_m'] >= .02) else None)
    for item in candidates:
        box = [round(v * (image.shape[1] / width if i % 2 == 0 else image.shape[0] / height))
               for i, v in enumerate(item['bbox_xyxy'])]
        item['saved_bbox_xyxy'] = box
        chosen = item is suggested
        color = (0, 255, 255) if chosen else (255, 180, 0)
        cv2.rectangle(marked, tuple(box[:2]), tuple(box[2:]), color, 2)
        cv2.putText(marked, f"ID {item['fruit_id']}" + (' TARGET? VERIFY' if chosen else ''),
                    (max(0, box[0]), max(45, box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    cv2.putText(marked, 'VERIFY TARGET - discard if wrong' if suggested else 'TARGET UNCONFIRMED - discard if ambiguous',
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    return marked, {'attempt_goal_base_xyz': goal, 'candidates': candidates,
                    'suggested_fruit_id': suggested['fruit_id'] if suggested else None,
                    'identity_verified': False, 'association': 'base_proximity_50mm_margin_20mm'}


def capture_correction(node) -> Optional[str]:
    node._demo_jog_joint_path = []
    node._demo_target_selection = None
    node._demo_confirm = None
    node._demo_capture_ready = False
    node._demo_capture_active = True
    try:
        return _capture_correction(node)
    finally:
        node._demo_capture_ready = False
        node._demo_capture_active = False


def _capture_correction(node) -> Optional[str]:
    """Hold at APPROACH, wait for the operator's corrected pose, write one sample.

    Returns the directory written, or None if disabled/skipped/timed out.
    The caller stops the attempt if recording fails.
    """
    planner = getattr(getattr(node, "cfg", None), "planner", None)
    if not bool(getattr(planner, "demo_capture_enabled", False)):
        return None

    # Vision has just resumed. Do not pair a stale approach image with APPROACH.
    requested_at = time.time()
    deadline = time.monotonic() + 3.0
    while (getattr(node, "_latest_grasp_crop_t", 0.0) <= requested_at
           or getattr(node, "_latest_grasp_crop", None) is None
           or getattr(node, "_demo_full_frame", None) is None
           or node._demo_full_frame[1] <= requested_at):
        if getattr(node, "stop_requested", False):
            return None
        if time.monotonic() >= deadline:
            node.get_logger().warn("[DEMO] no fresh APPROACH crop/full frame; stopping attempt")
            return None
        time.sleep(0.02)
    snapshot = node._demo_full_frame
    full_frame, full_received, full_stamp = snapshot[:3]
    frame_ns = snapshot[3] if len(snapshot) > 3 else None
    target_meta = {}
    if frame_ns is not None:
        meta_deadline = time.monotonic() + 1.0
        while frame_ns not in node._demo_targets and time.monotonic() < meta_deadline:
            if getattr(node, 'stop_requested', False):
                return None
            time.sleep(.02)
        target_meta = copy.deepcopy(node._demo_targets.get(frame_ns, {}))
    before_full = full_frame.copy()
    marked_full, target_label = target_preview(before_full, target_meta,
        copy.deepcopy(getattr(node, '_demo_attempt_goal_xyz', None)))
    node._demo_publish_preview(marked_full)
    before_image = node._latest_grasp_crop.copy()
    before_image_t = node._latest_grasp_crop_t
    before_pose = _pose_now(node)
    if before_pose is None:
        node.get_logger().warn("[DEMO] no TCP pose at APPROACH; not recording")
        return None

    before = {
        "planned_final_pose": copy.deepcopy(getattr(node, "_demo_planned_final_pose", None)),
        "target_label": target_label,
        "pose": before_pose,
        "joints": list(getattr(node, "current_joint_positions", None) or []),
        "goal_pose": getattr(node, "latest_goal_pose", None),
        "final_retarget_xyz": getattr(node, "_final_retarget_xyz", None),
        "crop_meta": copy.deepcopy(getattr(node, "_latest_grasp_crop_meta", None)),
        "crop_time": before_image_t,
        "full_frame_received_time": full_received,
        "full_frame_source_stamp": full_stamp,
        "full_frame_shape": list(before_full.shape),
        "full_frame_topic": "/vision/demo_full/compressed",
        "full_frame_annotated": False,
        "fruit_radius": getattr(node, "latest_fruit_radius", None),
    }

    timeout = float(getattr(planner, "demo_capture_timeout_s", 0.0))
    hz = max(1.0, float(getattr(planner, "demo_capture_sample_hz", 10.0)))
    node._demo_capture_ready = True
    node.get_logger().warn(
        "[DEMO] holding at APPROACH. Jog the pendant to the pose that would "
        "actually grasp, restore ROS control, then press Close in RViz to save "
        "and continue. Uncheck Demo to discard and stop this attempt.")

    path: List[Dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while timeout <= 0 or time.monotonic() < deadline:
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("[DEMO] stop requested; discarding")
            return None
        if getattr(node, "_demo_confirm", None) is not None:
            if node._demo_confirm and not getattr(node, '_demo_target_selection', None):
                node._demo_confirm = None
                node.get_logger().warn('[DEMO] Click the intended date in the Motion panel image, then press Close again.')
            else:
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

    if not getattr(node, "_demo_target_selection", None):
        node.get_logger().warn("[DEMO] No target click; discarding")
        return None
    selection = copy.deepcopy(node._demo_target_selection)
    px, py = selection['pixel_xy']
    hits = [c['fruit_id'] for c in target_label['candidates']
            if c['saved_bbox_xyxy'][0] <= px <= c['saved_bbox_xyxy'][2]
            and c['saved_bbox_xyxy'][1] <= py <= c['saved_bbox_xyxy'][3]]
    target_label['operator_selection'] = selection
    target_label['clicked_candidate_ids'] = hits
    target_label['selected_fruit_id'] = hits[0] if len(hits) == 1 else None
    target_label['identity_verified'] = len(hits) == 1
    target_label['operator_point_confirmed'] = True
    import cv2
    cv2.circle(marked_full, (round(px), round(py)), 15, (0, 255, 0), 3)

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

    planned_final = before.get('planned_final_pose')
    residual = ([after_pose[i] - planned_final[i] for i in range(3)]
                if planned_final else None)
    sample = {
        "level_insertion_enabled": bool(getattr(planner, "low_center_level_insertion_enabled", False)),
        "level_final_z_m": getattr(node, "_level_final_z", None),
        "schema_version": 2,
        "observation_stage": "APPROACH",
        "desired_motion_base_m": delta_base,
        "desired_motion_tool_m": delta_tool,
        "residual_from_planned_final_base_m": residual,
        "correction_reference": "actual_approach_pose",

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
    if not (_write_crop(before_image, out, "crop_before.jpg")
            and _write_crop(before_full, out, "full_before.jpg")
            and _write_crop(marked_full, out, "full_before_marked.jpg")):
        node.get_logger().warn("[DEMO] cannot save APPROACH crop; stopping attempt")
        os.remove(os.path.join(out, "sample.json"))
        return None
    _write_crop(getattr(node, "_latest_grasp_crop", None), out, "crop_after.jpg")
    after_frame = getattr(node, '_demo_full_frame', None)
    if after_frame is not None and time.time() - after_frame[1] < 1.0:
        _write_crop(after_frame[0], out, 'full_after.jpg')

    node.get_logger().warn(
        f"[DEMO] correction {sample['correction_mm']:.1f}mm "
        f"base=[{delta_base[0]*1000:+.1f},{delta_base[1]*1000:+.1f},"
        f"{delta_base[2]*1000:+.1f}]mm"
        + (f" tool=[{delta_tool[0]*1000:+.1f},{delta_tool[1]*1000:+.1f},"
           f"{delta_tool[2]*1000:+.1f}]mm" if delta_tool else "")
        + f" at {sample['range_m']}m -> {out}")
    node._demo_jog_joint_path = [list(before['joints'])]
    node._demo_jog_joint_path.extend(list(item['joints']) for item in path if len(item['joints']) == 6)
    node._demo_jog_joint_path.append(list(sample['after']['joints']))
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
