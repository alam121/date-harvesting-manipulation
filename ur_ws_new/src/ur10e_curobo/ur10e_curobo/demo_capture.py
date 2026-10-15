"""Record what the arm saw at APPROACH, for training the last 7cm.

Both modes observe the same thing -- a crop centered on the operator's click,
the full APPROACH frame, the depth trace, the planned final pose -- and differ
only in what label gets attached. planner.demo_capture_jog_enabled picks:

  capture_observation (default)
      Holds at APPROACH only until the operator clicks the intended date, then
      runs the grasp automatically. The operator grades good/edge/fail in RViz;
      the automatic gripper verdict is not part of policy data. No pendant jog
      or Close.

  capture_correction
      Holds at APPROACH for discrete RViz tool-X/tool-Y alignment. Tool Z is
      locked, each step is guarded, and the same planar offset is applied to
      FINAL before the automatic insertion. The accumulated XY offset is the
      policy label.

The correction is recorded in the TOOL frame as well as base_link. Base-frame
deltas are what the offsets in config.py are expressed in, but a policy reading
the wrist camera sees the error in tool axes, and the two differ by the whole
wrist orientation.

Correction-mode full-frame JPEGs and joint states are recorded with source
timestamps. Sequence quality is saved separately; this is one-shot residual
policy data, not an ACT endpoint demonstration.

Both are off unless planner.demo_capture_enabled -- a normal harvest records
episodes, not training samples.
"""

import copy
import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


OPERATOR_LABELS = ("good", "edge", "fail")
OPERATOR_CROP_SIZE_PX = 256
DEFAULT_XY_STEP_M = 0.005
DEFAULT_XY_LIMIT_M = 0.020


def _demo_root(node) -> str:
    planner = getattr(getattr(node, "cfg", None), "planner", None)
    return os.path.expanduser(str(getattr(
        planner, "demo_capture_dir", "~/harvest_logs/demos")))


def _unlabelled_records(root: str) -> List[str]:
    """Return stable, oldest-first record directories needing a label."""
    if not os.path.isdir(root):
        return []
    pending = []
    for name in sorted(os.listdir(root)):
        directory = os.path.join(root, name)
        path = os.path.join(directory, "sample.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as handle:
                sample = json.load(handle)
        except Exception:
            continue
        if sample.get("operator_label") not in OPERATOR_LABELS:
            pending.append(directory)
    return pending


def refresh_label_queue(node) -> List[str]:
    """Merge unlabelled on-disk records into the node's exact-path queue."""
    queue = list(getattr(node, "_demo_label_queue", []) or [])
    queue = [directory for directory in queue
             if os.path.isfile(os.path.join(directory, "sample.json"))]
    for directory in _unlabelled_records(_demo_root(node)):
        if directory not in queue:
            queue.append(directory)
    node._demo_label_queue = queue
    return queue


def queue_for_label(node, directory: str) -> None:
    """Offer the exact completed attempt to RViz for operator grading."""
    queue = list(getattr(node, "_demo_label_queue", []) or [])
    if directory and directory not in queue:
        queue.append(directory)
    node._demo_label_queue = queue


def label_status(node) -> Dict[str, Any]:
    """Small JSON-safe status block consumed by the RViz Motion panel."""
    queue = list(getattr(node, "_demo_label_queue", []) or [])
    while queue:
        path = os.path.join(queue[0], "sample.json")
        try:
            with open(path) as handle:
                sample = json.load(handle)
        except Exception:
            queue.pop(0)
            continue
        if sample.get("operator_label") in OPERATOR_LABELS:
            queue.pop(0)
            continue
        break
    node._demo_label_queue = queue
    if not queue:
        return {"pending": 0, "record": "", "image": ""}
    directory = queue[0]
    image = os.path.join(directory, "full_before_marked.jpg")
    if not os.path.isfile(image):
        image = os.path.join(directory, "full_before.jpg")
    return {
        "pending": len(queue),
        "record": os.path.basename(directory),
        "image": image if os.path.isfile(image) else "",
    }


def apply_operator_label(node, label: str, record_id=None) -> bool:
    """Write one RViz grade to the exact oldest pending demo record."""
    label = str(label).strip().lower()
    if label not in OPERATOR_LABELS:
        node.get_logger().warn(f"[DEMO] invalid operator label: {label!r}")
        return False
    status = label_status(node)
    queue = node._demo_label_queue
    if record_id is not None and record_id != status['record']:
        node.get_logger().warn('[DEMO] stale grade rejected; refresh the displayed record')
        return False
    if not queue:
        node.get_logger().warn("[DEMO] no recorded attempt is waiting for a label")
        return False
    directory = queue[0]
    path = os.path.join(directory, "sample.json")
    try:
        with open(path) as handle:
            sample = json.load(handle)
        sample["operator_label"] = label
        with open(path, "w") as handle:
            json.dump(sample, handle, indent=1, default=str)
    except Exception as exc:
        node.get_logger().warn(f"[DEMO] label was not saved to {path}: {exc}")
        return False
    from ur10e_curobo.demo_sequence import update_training_readiness
    update_training_readiness(directory, label)
    queue.pop(0)
    node._demo_last_operator_label = {
        "record": status["record"], "label": label}
    node.get_logger().info(
        f"[DEMO] operator label {label.upper()} saved for {status['record']}")
    return True


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


def _quat_rotate(quat_wxyz, vec):
    """Rotate a tool-frame vector into the base frame. quat is [w,x,y,z]."""
    w, x, y, z = (float(v) for v in quat_wxyz)
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
    cv2.putText(marked, 'CLICK THE INTENDED TARGET',
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    return marked, {'attempt_goal_base_xyz': goal, 'candidates': candidates,
                    'suggested_fruit_id': suggested['fruit_id'] if suggested else None,
                    'identity_verified': False, 'association': 'base_proximity_50mm_margin_20mm'}


def _apply_target_selection(marked_full, target_label, selection) -> None:
    """Persist the operator's clicked target point and any unique detector hit."""
    selection = copy.deepcopy(selection)
    px, py = selection['pixel_xy']
    hits = [candidate['fruit_id'] for candidate in target_label['candidates']
            if candidate['saved_bbox_xyxy'][0] <= px <= candidate['saved_bbox_xyxy'][2]
            and candidate['saved_bbox_xyxy'][1] <= py <= candidate['saved_bbox_xyxy'][3]]
    target_label['operator_selection'] = selection
    target_label['clicked_candidate_ids'] = hits
    target_label['selected_fruit_id'] = hits[0] if len(hits) == 1 else None
    target_label['identity_verified'] = len(hits) == 1
    target_label['operator_point_confirmed'] = True
    import cv2
    cv2.circle(marked_full, (round(px), round(py)), 15, (0, 255, 0), 3)


def _operator_target_crop(image, pixel_xy, size=OPERATOR_CROP_SIZE_PX):
    """Crop around the operator click without requiring a detector target."""
    if image is None or image.size == 0:
        return None, None
    height, width = image.shape[:2]
    px, py = (int(round(float(value))) for value in pixel_xy)
    if not (0 <= px < width and 0 <= py < height):
        return None, None
    crop_width = min(int(size), width)
    crop_height = min(int(size), height)
    x1 = min(max(0, px - crop_width // 2), width - crop_width)
    y1 = min(max(0, py - crop_height // 2), height - crop_height)
    x2, y2 = x1 + crop_width, y1 + crop_height
    metadata = {
        "source": "operator_click",
        "source_size": [width, height],
        "crop_xyxy": [x1, y1, x2, y2],
        "operator_pixel_full": [float(pixel_xy[0]), float(pixel_xy[1])],
        "operator_pixel_crop": [float(pixel_xy[0]) - x1, float(pixel_xy[1]) - y1],
    }
    return image[y1:y2, x1:x2].copy(), metadata


def _wait_for_target_click(node, planner, next_step: str) -> bool:
    """Hold motion at APPROACH until the frozen-frame target is confirmed."""
    timeout = float(getattr(planner, "demo_capture_timeout_s", 0.0))
    deadline = time.monotonic() + timeout
    node.get_logger().warn(
        f"[DEMO] holding at APPROACH. Click the intended date in RViz; {next_step}. "
        "Uncheck Demo to discard and stop this attempt.")
    while not getattr(node, '_demo_target_selection', None):
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("[DEMO] stop requested before target confirmation")
            return False
        if getattr(node, "_demo_confirm", None) is False:
            node.get_logger().info("[DEMO] operator discarded before target confirmation")
            return False
        if timeout > 0 and time.monotonic() >= deadline:
            node.get_logger().warn(
                f"[DEMO] no target click within {timeout:.0f}s; discarding")
            return False
        time.sleep(0.02)
    return True


def capture_correction(node, move_fn=None) -> Optional[str]:
    node._demo_target_selection = None
    node._demo_confirm = None
    node._demo_xy_pending_step_tool_m = None
    node._demo_xy_offset_tool_m = [0.0, 0.0, 0.0]
    node._demo_xy_offset_base_m = [0.0, 0.0, 0.0]
    node._demo_xy_busy = False
    node._demo_xy_error = ""
    node._demo_capture_ready = False
    from ur10e_curobo.demo_sequence import DemoSequence
    finish_sequence(node)  # Retain an interrupted previous recording as incomplete.
    node._demo_sequence = DemoSequence(
        node.joint_order,
        action_source='operator_xy_alignment_then_automatic_final')
    node._demo_capture_active = True
    saved = None
    try:
        saved = _capture_correction(node, move_fn)
        return saved
    finally:
        if not saved:
            finish_sequence(node)
        node._demo_xy_busy = False
        node._demo_xy_pending_step_tool_m = None
        node._demo_capture_ready = False
        node._demo_capture_active = False


def _observe(node):
    """Everything seen at APPROACH, before anyone touches the robot.

    Shared by both capture modes. Both require target confirmation; correction
    mode then holds for bounded RViz XY alignment, while outcome mode continues.
    Returns (before, before_image, before_full, marked_full, target_label), or
    None if no usable observation was available.
    """
    # Vision has just resumed. Only the full frame is required: the training
    # crop is derived from the operator click, not an automatic detector box.
    requested_at = time.time()
    deadline = time.monotonic() + 3.0
    while (getattr(node, "_demo_full_frame", None) is None
           or node._demo_full_frame[1] <= requested_at):
        if getattr(node, "stop_requested", False):
            return None
        if time.monotonic() >= deadline:
            node.get_logger().warn("[DEMO] no fresh APPROACH full frame; stopping attempt")
            return None
        time.sleep(0.02)
    snapshot = node._demo_full_frame
    full_frame, full_received, full_stamp = snapshot[:3]
    frame_ns = snapshot[3] if len(snapshot) > 3 else None
    target_meta = {}
    if frame_ns is not None:
        target_meta = copy.deepcopy(node._demo_targets.get(frame_ns, {}))
    before_full = full_frame.copy()
    marked_full, target_label = target_preview(before_full, target_meta,
        copy.deepcopy(getattr(node, '_demo_attempt_goal_xyz', None)))
    node._demo_capture_ready = True
    node._demo_publish_preview(marked_full)
    before_pose = _pose_now(node)
    if before_pose is None:
        node.get_logger().warn("[DEMO] no TCP pose at APPROACH; not recording")
        return None

    before = {
        "neighbor_diagnostic": copy.deepcopy(getattr(node, "_last_neighbor_diagnostic", None)),
        "depth_trace": copy.deepcopy(getattr(node, "_demo_depth_trace", None)),
        "planned_final_pose": copy.deepcopy(getattr(node, "_demo_planned_final_pose", None)),
        "target_label": target_label,
        "pose": before_pose,
        "joints": list(getattr(node, "current_joint_positions", None) or []),
        "goal_pose": getattr(node, "latest_goal_pose", None),
        "final_retarget_xyz": getattr(node, "_final_retarget_xyz", None),
        "crop_meta": None,
        "crop_time": full_received,
        "crop_source": "operator_click",
        "full_frame_received_time": full_received,
        "full_frame_source_stamp": full_stamp,
        "full_frame_shape": list(before_full.shape),
        "full_frame_topic": "/vision/demo_full/compressed",
        "full_frame_annotated": False,
        "fruit_radius": getattr(node, "latest_fruit_radius", None),
    }

    return before, None, before_full, marked_full, target_label


def capture_observation(node) -> Optional[str]:
    """Record the automatic APPROACH-to-grasp trajectory after a target click."""
    from ur10e_curobo.demo_sequence import DemoSequence
    finish_sequence(node)
    node._demo_sequence = DemoSequence(node.joint_order, action_source='automatic_execution')
    node._demo_target_selection = None
    node._demo_confirm = None
    node._demo_capture_ready = False
    node._demo_capture_active = True
    saved = None
    try:
        saved = _capture_observation(node)
        return saved
    finally:
        if not saved:
            finish_sequence(node)
        node._demo_capture_ready = False
        node._demo_capture_active = False


def _capture_observation(node) -> Optional[str]:
    """Record a click-confirmed APPROACH observation, then get out of the way.

    The robot holds only for a click on the intended date. There is no pendant
    jog and no Close press: after confirmation the cycle runs to the grasp on
    its own, then the operator grades Good / Edge / Fail in RViz. Automatic
    gripper verdicts are not recorded in policy data.

    Records the robot's own trajectory, not a demonstrated error correction.
    Success labels support imitation; all labels support outcome prediction.

    Returns the directory written, or None if nothing usable was observed.
    """
    planner = getattr(getattr(node, "cfg", None), "planner", None)
    if not bool(getattr(planner, "demo_capture_enabled", False)):
        return None
    observed = _observe(node)
    if observed is None:
        return None
    before, before_image, before_full, marked_full, target_label = observed
    if not _wait_for_target_click(
            node, planner, "the automatic FINAL move will then continue"):
        return None
    _apply_target_selection(
        marked_full, target_label, node._demo_target_selection)
    before_image, crop_meta = _operator_target_crop(
        before_full, node._demo_target_selection["pixel_xy"])
    if before_image is None:
        node.get_logger().warn("[DEMO] operator click could not produce a target crop")
        return None
    before["crop_meta"] = crop_meta
    before["target_label"] = target_label
    if hasattr(node, 'get_clock'):
        node._demo_sequence.command('recording.motion_start', 1.0, time.time_ns(),
                                   source_stamp_ns=node.get_clock().now().nanoseconds)

    out = _make_dir(node, planner)
    if out is None:
        return None
    node._demo_sequence.set_target('../full_before.jpg',
        node._demo_target_selection['pixel_xy'], before_full.shape, target_label)
    sequence_quality = node._demo_sequence.stage(out)
    sample = {
        "schema_version": 5,
        "trajectory_source": "automatic_execution",
        "label_source": "operator",
        "sequence_path": "sequence/sequence.json",
        "sequence_quality": sequence_quality,
        "observation_stage": "APPROACH",
        "capture_mode": "click_confirmed_outcome",
        "target_confirmation_required": True,
        "target_region_preference": getattr(planner, "demo_target_region", "default"),
        "level_insertion_enabled": bool(getattr(
            planner, "low_center_level_insertion_enabled", False)),
        "level_final_z_m": getattr(node, "_level_final_z", None),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "before": before,
        "operator_label": None,
        "range_m": (round(math.dist([0, 0, 0], before["goal_pose"][:3]), 4)
                    if before.get("goal_pose") else None),
    }
    if not _write_sample(node, out, sample):
        return None
    if not (_write_crop(before_image, out, "crop_before.jpg")
            and _write_crop(before_full, out, "full_before.jpg")
            and _write_crop(marked_full, out, "full_before_marked.jpg")):
        node.get_logger().warn("[DEMO] cannot save APPROACH images; discarding")
        os.remove(os.path.join(out, "sample.json"))
        return None
    node.get_logger().info(
        f"[DEMO] click-confirmed APPROACH at {sample['range_m']}m -> {out}")
    return out


def complete_capture(node) -> None:
    """Finalize the policy record after the grasp, without detector labels."""
    out = getattr(node, "_demo_pending_record", None)
    node._demo_pending_record = None
    if not out:
        return
    path = os.path.join(out, "sample.json")
    if not os.path.isfile(path):
        node.get_logger().warn(f"[DEMO] cannot finalize missing record: {path}")
        return
    finish_sequence(node, complete=True)
    queue_for_label(node, out)
    node.get_logger().info(f"[DEMO] attempt recorded; grade it Good / Edge / Fail")


def finish_sequence(node, complete=False):
    recorder = getattr(node, '_demo_sequence', None)
    if recorder is None:
        return
    node._demo_sequence = None
    try:
        if recorder.staged:
            quality = recorder.finalize(complete=complete)
            path = recorder.path.parent / 'sample.json'
            if path.exists():
                sample = json.loads(path.read_text())
                sample['sequence_quality'] = quality
                path.write_text(json.dumps(sample, indent=1, default=str))
    finally:
        recorder.close()


def _make_dir(node, planner) -> Optional[str]:
    root = _demo_root(node)
    out = os.path.join(root, datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
    try:
        os.makedirs(out, exist_ok=True)
        return out
    except Exception as exc:
        node.get_logger().warn(f"[DEMO] cannot create {out}: {exc}")
        return None


def _write_sample(node, out, sample) -> bool:
    try:
        with open(os.path.join(out, "sample.json"), "w") as handle:
            json.dump(sample, handle, indent=1, default=str)
        return True
    except Exception as exc:
        node.get_logger().warn(f"[DEMO] cannot write sample.json: {exc}")
        return False


def _capture_correction(node, move_fn) -> Optional[str]:
    """Collect bounded tool-XY alignment at APPROACH and write one sample."""
    planner = getattr(getattr(node, "cfg", None), "planner", None)
    if not bool(getattr(planner, "demo_capture_enabled", False)):
        return None
    if not callable(move_fn):
        node.get_logger().warn("[DEMO_XY] guarded move callback unavailable")
        return None
    observed = _observe(node)
    if observed is None:
        return None
    before, _, before_full, marked_full, target_label = observed
    before_pose = before["pose"]
    if not _wait_for_target_click(
            node, planner, "use Tool X/Y buttons, then press XY Aligned"):
        return None
    _apply_target_selection(
        marked_full, target_label, node._demo_target_selection)
    before_image, crop_meta = _operator_target_crop(
        before_full, node._demo_target_selection["pixel_xy"])
    if before_image is None:
        node.get_logger().warn("[DEMO_XY] operator click could not produce a target crop")
        return None
    before["crop_meta"] = crop_meta
    before["target_label"] = target_label
    if hasattr(node, 'get_clock'):
        node._demo_sequence.command(
            'recording.motion_start', 1.0, time.time_ns(),
            source_stamp_ns=node.get_clock().now().nanoseconds)

    step_m = float(getattr(planner, "demo_xy_step_m", DEFAULT_XY_STEP_M))
    limit_m = float(getattr(planner, "demo_xy_limit_m", DEFAULT_XY_LIMIT_M))
    if not (0.0 < step_m <= limit_m <= 0.050):
        node.get_logger().warn(
            f"[DEMO_XY] invalid limits step={step_m}m limit={limit_m}m")
        return None
    timeout = float(getattr(planner, "demo_capture_timeout_s", 0.0))
    node.get_logger().warn(
        f"[DEMO_XY] target confirmed. Align with RViz Tool X/Y in "
        f"{step_m*1000:.0f}mm steps (limit +/-{limit_m*1000:.0f}mm per axis); "
        "Tool Z/depth is locked. Press XY Aligned to continue.")

    path: List[Dict[str, Any]] = [{
        "t": round(time.time(), 3), "pose": list(before_pose),
        "joints": list(getattr(node, "current_joint_positions", None) or []),
        "command_tool_m": [0.0, 0.0, 0.0],
        "cumulative_tool_m": [0.0, 0.0, 0.0],
    }]
    deadline = time.monotonic() + timeout
    while timeout <= 0 or time.monotonic() < deadline:
        if getattr(node, "stop_requested", False):
            node.get_logger().warn("[DEMO_XY] stop requested; discarding")
            return None
        pending = getattr(node, "_demo_xy_pending_step_tool_m", None)
        if pending is not None:
            node._demo_xy_pending_step_tool_m = None
            current = list(node._demo_xy_offset_tool_m)
            proposed = [current[0] + float(pending[0]),
                        current[1] + float(pending[1]), 0.0]
            if abs(proposed[0]) > limit_m + 1e-9 or abs(proposed[1]) > limit_m + 1e-9:
                node._demo_xy_error = (
                    f"limit +/-{limit_m*1000:.0f}mm; requested "
                    f"X={proposed[0]*1000:+.0f} Y={proposed[1]*1000:+.0f}mm")
                node.get_logger().warn(f"[DEMO_XY] {node._demo_xy_error}")
                continue
            offset_base = _quat_rotate(before_pose[3:7], proposed)
            target = [before_pose[i] + offset_base[i] for i in range(3)] + list(before_pose[3:7])
            node._demo_xy_busy = True
            node._demo_xy_error = ""
            axis = "x" if abs(float(pending[0])) > 0.0 else "y"
            amount = float(pending[0] if axis == "x" else pending[1])
            if hasattr(node, 'get_clock'):
                node._demo_sequence.command(
                    f'alignment.tool_{axis}', amount, time.time_ns(),
                    source_stamp_ns=node.get_clock().now().nanoseconds)
            try:
                moved = bool(move_fn(target))
            finally:
                node._demo_xy_busy = False
            if not moved:
                node._demo_xy_error = "guarded XY move failed"
                node.get_logger().warn("[DEMO_XY] guarded move failed; stopping attempt")
                return None
            node._demo_xy_offset_tool_m = proposed
            node._demo_xy_offset_base_m = offset_base
            after_step = _pose_now(node)
            path.append({
                "t": round(time.time(), 3), "pose": after_step,
                "joints": list(getattr(node, "current_joint_positions", None) or []),
                "command_tool_m": [float(pending[0]), float(pending[1]), 0.0],
                "cumulative_tool_m": list(proposed),
            })
            node.get_logger().info(
                f"[DEMO_XY] aligned offset tool="
                f"[{proposed[0]*1000:+.0f},{proposed[1]*1000:+.0f},+0]mm")
            moved_at = time.time()
            preview_deadline = time.monotonic() + 1.5
            while time.monotonic() < preview_deadline:
                frame = getattr(node, '_demo_full_frame', None)
                if frame is not None and frame[1] > moved_at:
                    node._demo_publish_preview(frame[0].copy())
                    break
                if getattr(node, "stop_requested", False):
                    return None
                time.sleep(0.02)
            continue
        if getattr(node, "_demo_confirm", None) is not None:
            break
        time.sleep(0.02)

    accepted = getattr(node, "_demo_confirm", None)
    if accepted is None:
        node.get_logger().warn(f"[DEMO_XY] no confirm within {timeout:.0f}s; discarding")
        return None
    if not accepted:
        node.get_logger().info("[DEMO_XY] operator discarded this one")
        return None

    after_pose = _pose_now(node)
    if after_pose is None:
        node.get_logger().warn("[DEMO_XY] no TCP pose after alignment; discarding")
        return None

    commanded_tool = list(node._demo_xy_offset_tool_m)
    commanded_base = list(node._demo_xy_offset_base_m)
    measured_base = [after_pose[i] - before_pose[i] for i in range(3)]
    measured_tool = _quat_rotate_inverse(before_pose[3:7], measured_base)
    planned_final = before.get('planned_final_pose')
    corrected_final = (
        [planned_final[i] + commanded_base[i] for i in range(3)]
        + list(planned_final[3:]) if planned_final else None)
    node._demo_xy_corrected_final_pose = corrected_final

    out = _make_dir(node, planner)
    if out is None:
        return None

    node._demo_sequence.set_target('../full_before.jpg',
        node._demo_target_selection['pixel_xy'], before_full.shape, target_label)
    sequence_quality = node._demo_sequence.stage(out)
    node.get_logger().info(f'[DEMO_XY] sequence quality: {sequence_quality}')
    trace = before.get('depth_trace')
    if trace is not None:
        trace['approach_xy_correction_tool_m'] = commanded_tool
        trace['approach_xy_correction_base_m'] = commanded_base
        trace['aligned_approach_tcp_base_xyz_m'] = list(after_pose[:3])
        trace['corrected_final_tcp_base_xyz_m'] = (
            list(corrected_final[:3]) if corrected_final else None)
    sample = {
        "schema_version": 6,
        "capture_mode": "approach_xy_alignment",
        "trajectory_source": "rviz_discrete_tool_xy",
        "label_source": "operator",
        "operator_label": None,
        "target_confirmation_required": True,
        "level_insertion_enabled": bool(getattr(planner, "low_center_level_insertion_enabled", False)),
        "nominal_level_final_z_m": getattr(node, "_level_final_z", None),
        "level_final_z_m": (
            corrected_final[2] if corrected_final is not None and bool(getattr(
                planner, "low_center_level_insertion_enabled", False))
            else getattr(node, "_level_final_z", None)),
        "sequence_quality": sequence_quality,
        "sequence_path": "sequence/sequence.json",
        "observation_stage": "APPROACH",
        "target_region_preference": getattr(planner, "demo_target_region", "default"),
        "desired_motion_base_m": commanded_base,
        "desired_motion_tool_m": commanded_tool,
        "correction_reference": "nominal_approach_pose",
        "depth_axis_locked": True,
        "automatic_final_retarget_disabled": True,
        "alignment_step_m": step_m,
        "alignment_limit_per_axis_m": limit_m,
        "nominal_final_pose": planned_final,
        "corrected_final_pose": corrected_final,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "before": before,
        "after": {"pose": after_pose,
                  "joints": list(getattr(node, "current_joint_positions", None) or [])},
        "correction_base_m": commanded_base,
        "correction_tool_m": commanded_tool,
        "measured_approach_delta_base_m": measured_base,
        "measured_approach_delta_tool_m": measured_tool,
        "correction_mm": round(math.hypot(commanded_tool[0], commanded_tool[1]) * 1000.0, 2),
        "range_m": (round(math.dist([0, 0, 0], before["goal_pose"][:3]), 4)
                    if before.get("goal_pose") else None),
        "alignment_path": path,
        "alignment_steps": max(0, len(path) - 1),
    }
    if not _write_sample(node, out, sample):
        return None
    if not (_write_crop(before_image, out, "crop_before.jpg")
            and _write_crop(before_full, out, "full_before.jpg")
            and _write_crop(marked_full, out, "full_before_marked.jpg")):
        node.get_logger().warn("[DEMO_XY] cannot save APPROACH crop; stopping attempt")
        os.remove(os.path.join(out, "sample.json"))
        return None
    after_frame = getattr(node, '_demo_full_frame', None)
    if after_frame is not None and time.time() - after_frame[1] < 1.0:
        _write_crop(after_frame[0], out, 'full_after.jpg')

    node.get_logger().warn(
        f"[DEMO_XY] correction {sample['correction_mm']:.1f}mm "
        f"tool=[{commanded_tool[0]*1000:+.1f},{commanded_tool[1]*1000:+.1f},+0.0]mm "
        f"base=[{commanded_base[0]*1000:+.1f},{commanded_base[1]*1000:+.1f},"
        f"{commanded_base[2]*1000:+.1f}]mm at {sample['range_m']}m -> {out}")
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
