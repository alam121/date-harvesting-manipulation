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
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


def snapshot_close_geometry(node, fruit_seed_xyz, commanded_tcp_xyz,
                            max_age_s=1.5, match_radius_m=0.025,
                            inflight_report=None, inflight_max_age_s=1.5):
    """Where the date is relative to the fingers at the moment of closing.

    Splits a miss into its parts:
      tracking   = actual TCP - commanded TCP              (the arm)
      perception = date seen at close - date planned on    (vision, far vs near)
      target_rel = date seen at close - actual closure centre, in TOOL frame;
                   z is the pointing axis, so +z means the date sits beyond
                   the closure point (arm short) and -z means the arm overshot.
    The endpoint error alone proves only that the arm reached its command; this
    measures against the fruit. Stored on node._close_geometry; never raises.
    """
    try:
        from .goals import active_closure_offset, quat_rotate_vec
        pose = node.get_end_effector_pose()
        if not pose:
            return None
        tcp = np.asarray(pose[:3], dtype=float)
        q = [float(v) for v in pose[3:7]]                       # wxyz
        q_inv = [q[0], -q[1], -q[2], -q[3]]
        seed = np.asarray(fruit_seed_xyz, dtype=float)
        closure = tcp + np.asarray(quat_rotate_vec(q, active_closure_offset(node)))
        pointing = np.asarray(quat_rotate_vec(q, [0.0, 0.0, 1.0]))
        snap = {
            "tcp_actual": tcp, "tcp_commanded": list(commanded_tcp_xyz),
            "closure_actual": closure, "fruit_seed": seed,
            "tracking_mm": float(np.linalg.norm(
                tcp - np.asarray(commanded_tcp_xyz, dtype=float)) * 1000.0),
            "approach_pitch_deg": float(math.degrees(math.asin(
                max(-1.0, min(1.0, -pointing[2]))))),            # + = pointing down
            "approach_yaw_deg": float(math.degrees(math.atan2(pointing[1], pointing[0]))),
        }
        # Nearest fresh detection to the planned date. At close the fruit may be
        # partly hidden by the fingers, so this can miss it -- or, in a tight
        # bunch, pick a neighbour. match_radius_m matches the in-flight
        # monitor's 25mm max: at 40mm a grasp that succeeded logged a 30.7mm
        # "error" the monitor had refused five times as a different fruit.
        obs = getattr(node, "_fruit_observation", None)
        date = None
        why = "no observation"
        if obs is not None and obs.poses:
            age = (node.get_clock().now().nanoseconds - obs.stamp_ns) / 1e9
            nearest = min(obs.poses, key=lambda p: math.dist(p, seed))
            if age <= max_age_s and math.dist(nearest, seed) <= match_radius_m:
                date = np.asarray(nearest, dtype=float)
                snap["date_source"] = "latest_frame"
                snap["observation_age_s"] = float(age)
            else:
                why = (f"latest frame {age:.2f}s old, nearest "
                       f"{math.dist(nearest, seed) * 1000:.0f}mm")
        elif obs is not None:
            why = "latest frame has no dates"
        # As the fingers close around it the date is often hidden in the last
        # frame; the in-flight monitor's last sighting from the final move is
        # the next best measurement (the fruit is close to stationary).
        last = (inflight_report or {}).get("last_match")
        if date is None and last is not None:
            age = time.time() - float(inflight_report.get("last_match_wall", 0.0))
            if age <= inflight_max_age_s and math.dist(last, seed) <= match_radius_m:
                date = np.asarray(last, dtype=float)
                snap["date_source"] = "inflight_last_match"
                snap["observation_age_s"] = float(age)
        if date is not None:
            snap["date_at_close"] = date
        if date is not None:
            snap["perception_mm"] = (date - seed) * 1000.0
            rel_tool = np.asarray(quat_rotate_vec(q_inv, date - closure)) * 1000.0
            snap["target_rel_tool_mm"] = rel_tool
            snap["target_rel_mm"] = float(np.linalg.norm(rel_tool))
            try:
                tf = node.tf_buffer.lookup_transform(
                    "base_link",
                    os.environ.get("UR10E_CAM_FRAME", "zed_mini_left_camera_frame"),
                    __import__("rclpy").time.Time())
                t = tf.transform.translation
                snap["camera_range_m"] = float(np.linalg.norm(
                    date - np.array([t.x, t.y, t.z])))
            except Exception:
                pass
        node._close_geometry = snap
        if date is not None:
            r = snap["target_rel_tool_mm"]
            node.get_logger().info(
                f"[GRASP_ACCURACY] target_rel_tool=[{r[0]:+.1f},{r[1]:+.1f},"
                f"{r[2]:+.1f}]mm (z+ = date beyond closure) "
                f"tracking={snap['tracking_mm']:.1f}mm "
                f"perception={np.linalg.norm(snap['perception_mm']):.1f}mm "
                f"range={snap.get('camera_range_m', float('nan')):.2f}m "
                f"pitch={snap['approach_pitch_deg']:+.0f}deg "
                f"source={snap['date_source']} age={snap['observation_age_s']:.2f}s")
        else:
            snap["no_date_reason"] = why
            node.get_logger().info(
                f"[GRASP_ACCURACY] no fresh detection of the date at close "
                f"({why}; tracking={snap['tracking_mm']:.1f}mm)")
        return snap
    except Exception as exc:
        node.get_logger().warn(f"[GRASP_ACCURACY] snapshot failed: {exc}")
        return None


def snapshot_after_reverse(node, t_close: float) -> None:
    """Add the after-reverse camera crop and gripper state to this attempt's
    episode, without delaying the cycle.

    The crop and gripper state are copied now; a thread waits for the episode
    (the outcome classifier usually decides ~1.9s after close, often after
    this) and writes them into it as after_reverse.jpg / "after_reverse".
    Retention is judged from crop_close.jpg vs after_reverse.jpg.
    """
    planner = getattr(getattr(node, "cfg", None), "planner", None)
    if not bool(getattr(planner, "grasp_episode_recording", False)):
        return
    gc = getattr(node, "gripper_controller", None)
    img = getattr(node, "_latest_grasp_crop", None)
    snap = {
        "wall_time": time.time(),
        "crop_age_s": _safe(lambda: round(
            time.time() - float(node._latest_grasp_crop_t), 3)),
        "gripper_joint_position": _jsonable(
            getattr(gc, "actual_joint_position", None) if gc else None),
        "gripper_joint_effort": _jsonable(
            getattr(gc, "actual_joint_effort", None) if gc else None),
    }
    img = img.copy() if img is not None else None

    def _write():
        # ponytail: pairs by time -- the first episode recorded after this
        # close. Cycles are >10s apart; tag episodes with an attempt id if
        # that ever stops holding.
        deadline = time.time() + 6.0
        while time.time() < deadline:
            last = getattr(node, "_last_episode", None)
            if last is not None and last[1] >= t_close:
                break
            time.sleep(0.1)
        else:
            node.get_logger().warn("[EPISODE] after-reverse snapshot: no episode")
            return
        path = last[0]
        try:
            if img is not None:
                import cv2
                cv2.imwrite(os.path.join(path, "after_reverse.jpg"), img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                snap["image"] = "after_reverse.jpg"
            fn = os.path.join(path, "episode.json")
            with open(fn) as fh:
                episode = json.load(fh)
            episode["after_reverse"] = snap
            with open(fn, "w") as fh:
                json.dump(episode, fh, indent=2, sort_keys=True)
        except Exception as exc:
            node.get_logger().warn(f"[EPISODE] after-reverse not saved: {exc}")

    import threading
    threading.Thread(target=_write, daemon=True).start()


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
        # None = level insertion off or not a low-centre goal (A/B on hit rate).
        "level_final_z": _jsonable(getattr(node, "_level_final_z", None)),
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
        # Geometry at the moment of closing (see snapshot_close_geometry). The
        # ee_pose above is taken at outcome time, after the reverse has begun.
        "close_geometry": _jsonable(getattr(node, "_close_geometry", None)),
        "depth_diagnostics": _jsonable(
            getattr(node, "_latest_depth_diagnostics", None)),
        "fingertip_verification": str(
            getattr(node, "latest_fingertip_verification", "") or ""),
        "all_fruit_poses": _jsonable(getattr(node, "all_fruit_poses", None)),
        "all_fruit_quality": _jsonable(getattr(node, "all_fruit_quality", None)),
        "close_peak_current_a": _jsonable(
            getattr(gc, "close_peak_current_a", None) if gc else None),
        "close_current_limited": _jsonable(
            getattr(gc, "close_current_limited", None) if gc else None),
        # Set by label_episodes.py from what the operator saw; authoritative
        # over "outcome", which only reflects the gripper's force signal.
        "human_label": None,
    }

    # The crop is the one that matters for a learned grasp-state model: it
    # shows the date WITH the fingertips, so a model reasons about
    # gripper-relative geometry instead of absolute position -- which is how it
    # sidesteps the hand-eye chain rather than inheriting its error.
    episode["grasp_crop_meta"] = _jsonable(
        getattr(node, "_latest_grasp_crop_meta", None))
    episode["grasp_crop_age_s"] = _jsonable(_safe(
        lambda: round(time.time() - float(node._latest_grasp_crop_t), 3)))

    episode["grasp_crop_at_close_meta"] = _jsonable(
        getattr(node, "_grasp_crop_at_close_meta", None))
    episode["grasp_crop_at_close_age_s"] = _jsonable(
        getattr(node, "_grasp_crop_at_close_age", None))

    saved = []
    # crop_close is the MODEL INPUT: frozen at the end of the FINAL move, just
    # before the gripper closes. crop is whatever was latest at outcome time,
    # which is after the close and the reverse -- kept for context, not for
    # training a close/don't-close decision.
    for _img, _name in (
            (getattr(node, "_grasp_crop_at_close", None), "crop_close.jpg"),
            (getattr(node, "_latest_grasp_crop", None), "crop.jpg")):
        if _img is None:
            continue
        try:
            import cv2
            cv2.imwrite(os.path.join(path, _name), _img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved.append(_name)
        except Exception as exc:
            node.get_logger().warn(f"[EPISODE] {_name} not saved: {exc}")

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
