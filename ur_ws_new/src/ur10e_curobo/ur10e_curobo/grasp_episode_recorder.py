"""Persist one record per grasp attempt, for later model training.

Phase 0 of the learned grasp-state work. A model that judges whether the date
is positioned to be grasped needs episodes labelled by what actually happened;
none have been recorded, and every field day run without this is data that
cannot be recovered afterwards.

Deliberately dumb: no model, no inference, no effect on behaviour. It writes
what the system already knows at the moment the outcome is decided.

The label is the outcome itself, which is now trustworthy -- before 2026-09-13
the classifier reported WEAK for every grasp however good (it required contact
on three fingers and the CENTER fingertip cannot reach the fruit), so anything
recorded earlier would have taught a model that bug.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


def _jsonable(value: Any) -> Any:
    """numpy and ROS types are not JSON-serialisable; flatten what we can."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    for attr in ("x", "y", "z"):
        if hasattr(value, attr):
            return {a: float(getattr(value, a))
                    for a in ("x", "y", "z", "w") if hasattr(value, a)}
    return str(value)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def record_episode(node, outcome: str, end: str) -> Optional[str]:
    """Write one episode directory. Returns its path, or None if disabled."""
    cfg = getattr(node, "cfg", None)
    planner = getattr(cfg, "planner", None) if cfg else None
    if not bool(getattr(planner, "grasp_episode_recording", False)):
        return None

    root = os.path.expanduser(str(getattr(
        planner, "grasp_episode_dir", "~/harvest_logs/episodes")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(root, f"{stamp}_{outcome}_{end}")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:
        node.get_logger().warn(f"[EPISODE] cannot create {path}: {exc}")
        return None

    gc = getattr(node, "gripper_controller", None)
    evidence = getattr(gc, "contact_evidence", None) if gc else None

    episode: Dict[str, Any] = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "wall_time": time.time(),
        # --- label ---
        "outcome": outcome,
        "end": end,
        "slip": bool(getattr(node, "slip_detection", False)),
        "miss": bool(getattr(node, "grab_miss", False)),
        "weak": bool(getattr(node, "weak_grab", False)),
        # --- where we thought the date was, and where we went ---
        "goal_pose": _jsonable(getattr(node, "latest_goal_pose", None)),
        "final_retarget_xyz": _jsonable(getattr(node, "_final_retarget_xyz", None)),
        "reacquire_result": str(getattr(node, "reacquire_result", "") or ""),
        "motion_phase": str(getattr(node, "motion_phase", "") or ""),
        "grasp_mode": str(getattr(node, "active_goal_grasp_mode", "") or ""),
        "fruit_radius": _jsonable(getattr(node, "latest_fruit_radius", None)),
        # --- arm ---
        "joint_positions": _jsonable(getattr(node, "current_joint_positions", None)),
        "ee_pose": _jsonable(_safe(node.get_end_effector_pose)),
        # --- gripper: the signal that actually decides the label ---
        "gripper_joint_position": _jsonable(
            getattr(gc, "actual_joint_position", None) if gc else None),
        "gripper_joint_effort": _jsonable(
            getattr(gc, "actual_joint_effort", None) if gc else None),
        "contact_evidence": _jsonable(evidence),
        "force_triplet": _jsonable(
            getattr(getattr(node, "classifier", None), "last_forces", None)),
        # --- vision during the final move ---
        "final_inflight": _jsonable(getattr(node, "_last_inflight_report", None)),
        "depth_diagnostics": _jsonable(
            getattr(node, "_latest_depth_diagnostics", None)),
        "fingertip_verification": str(
            getattr(node, "latest_fingertip_verification", "") or ""),
        "all_fruit_poses": _jsonable(getattr(node, "all_fruit_poses", None)),
        "all_fruit_quality": _jsonable(getattr(node, "all_fruit_quality", None)),
    }

    saved = []
    for name in ("_grasp_pair_before_frame", "_grasp_pair_after_frame"):
        frame = getattr(node, name, None)
        if frame is None:
            continue
        try:
            import cv2
            out = os.path.join(path, name.replace("_grasp_pair_", "").replace("_frame", "") + ".jpg")
            cv2.imwrite(out, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved.append(os.path.basename(out))
        except Exception as exc:
            node.get_logger().warn(f"[EPISODE] frame {name} not saved: {exc}")
    episode["frames"] = saved

    try:
        with open(os.path.join(path, "episode.json"), "w") as fh:
            json.dump(episode, fh, indent=2, sort_keys=True)
    except Exception as exc:
        node.get_logger().warn(f"[EPISODE] cannot write episode.json: {exc}")
        return None

    node.get_logger().info(
        f"[EPISODE] {outcome}/{end} -> {path} ({len(saved)} frame(s))")
    return path
