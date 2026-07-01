import os
import subprocess
import time
from datetime import datetime
import math
import numpy as np

from . import fk as fk_mod
from . import markers as markers_mod
from . import motions as motions_mod
from .config import PLAN_CFG_SCAN_PREFLIGHT
from .goals import yaw_only_align_local_axis


def _scan_center(node):
    goals = (
        node.goal_poses.snapshot()
        if hasattr(node.goal_poses, "snapshot")
        else [g for g in node.goal_poses]
    )
    if goals:
        return list(goals[0][:3]), list(goals[0][3:7]) if len(goals[0]) >= 7 else None, "queued goal"
    cur = node.get_end_effector_pose()
    if cur is not None and len(cur) >= 7:
        return [float(v) for v in cur[:3]], list(cur[3:7]), "current TCP"
    latest = getattr(node, "latest_goal_pose", None)
    if latest is not None and len(latest) >= 3:
        quat = list(latest[3:7]) if len(latest) >= 7 else None
        return [float(v) for v in latest[:3]], quat, "latest goal"
    best = getattr(node, "best_goal_xyz", None)
    if best is not None and len(best) >= 3:
        cur = node.get_end_effector_pose()
        quat = list(cur[3:7]) if cur and len(cur) >= 7 else None
        return [float(v) for v in best[:3]], quat, "best tracked goal"
    return None, None, "none"


def _angle_near_reference(angle: float, reference: float) -> float:
    while angle - reference > math.pi:
        angle -= 2.0 * math.pi
    while angle - reference < -math.pi:
        angle += 2.0 * math.pi
    return angle


def arc_descriptor(node):
    """Geometry of the scan semicircle around the current TCP, used to highlight reachability
    points that sit on/near the arc. The reachability cloud is computed from the current
    posture, so the arc is centered on the TCP to share that one reference. Returns None if
    the TCP pose is unavailable."""
    cur = node.get_end_effector_pose()
    if cur is None or len(cur) < 3:
        return None
    cfg = node.cfg.lidar_scan
    center = [float(cur[0]), float(cur[1]), float(cur[2])]
    arc = math.radians(float(getattr(cfg, "semicircle_arc_deg", 180.0)))
    # n only affects the sampled angle list; we just need its span, so 2 is enough.
    radial, tangent, angles, _mode = _scan_basis_and_angles(node, center, cur, 2, arc)
    return {
        "center_xy": np.array(center[:2], dtype=float),
        "radial": np.asarray(radial, dtype=float),
        "tangent": np.asarray(tangent, dtype=float),
        "amin": float(min(angles)),
        "amax": float(max(angles)),
        "radius": float(getattr(cfg, "semicircle_radius_m", 0.22)),
        "target_z": center[2] + float(getattr(cfg, "semicircle_z_offset_m", 0.0)),
        "z_tol": float(getattr(cfg, "semicircle_z_tolerance_m", 0.02)),
    }


def point_on_arc(desc, x: float, y: float, z: float, tol: float) -> bool:
    """True if (x,y,z) lies within ``tol`` (radial) and the z-band of the scan arc ``desc``."""
    if desc is None:
        return False
    if abs(float(z) - desc["target_z"]) > desc["z_tol"]:
        return False
    vec = np.array([float(x), float(y)], dtype=float) - desc["center_xy"]
    r = float(np.linalg.norm(vec))
    if r < 1e-4 or abs(r - desc["radius"]) > float(tol):
        return False
    angle = math.atan2(float(np.dot(vec, desc["tangent"])),
                       float(np.dot(vec, desc["radial"])))
    mid = 0.5 * (desc["amin"] + desc["amax"])
    angle = _angle_near_reference(angle, mid)
    return (desc["amin"] - 0.05) <= angle <= (desc["amax"] + 0.05)


def _scan_basis_and_angles(node, center, cur, n: int, arc: float):
    cfg = node.cfg.lidar_scan
    if bool(getattr(cfg, "semicircle_use_base_angles", False)):
        start = math.radians(float(getattr(cfg, "semicircle_start_deg", -90.0)))
        end = math.radians(float(getattr(cfg, "semicircle_end_deg", 90.0)))
        span = abs(end - start)
        endpoint = span < (2.0 * math.pi - 1e-6)
        angles = np.linspace(start, end, n, endpoint=endpoint)
        return (
            np.array([1.0, 0.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            angles,
            "base",
        )

    center_xy = np.array(center[:2], dtype=float)
    cur_xy = np.array(cur[:2], dtype=float)
    radial = cur_xy - center_xy
    if np.linalg.norm(radial) < 1e-4:
        radial = np.array([1.0, 0.0], dtype=float)
    radial = radial / np.linalg.norm(radial)
    tangent = np.array([-radial[1], radial[0]], dtype=float)
    angles = np.linspace(-arc / 2.0, arc / 2.0, n)
    return radial, tangent, angles, "current"


def _semicircle_scan_poses(node, *, publish_preview: bool = True, log: bool = True):
    cfg = node.cfg.lidar_scan
    center, ref_quat, source = _scan_center(node)
    if center is None:
        if log:
            node.get_logger().error("Lidar scan: no scan center available")
        return []

    cur = node.get_end_effector_pose()
    if cur is None or len(cur) < 7:
        if log:
            node.get_logger().error("Lidar scan: current TCP pose unavailable")
        return []
    if ref_quat is None:
        ref_quat = list(cur[3:7])

    n = max(3, int(getattr(cfg, "semicircle_points", 7)))
    radius = max(0.02, float(getattr(cfg, "semicircle_radius_m", 0.22)))
    arc = math.radians(float(getattr(cfg, "semicircle_arc_deg", 180.0)))
    local_axis = tuple(getattr(cfg, "semicircle_local_axis", [0.0, 0.0, 1.0]))
    radial, tangent, angles, angle_mode = _scan_basis_and_angles(node, center, cur, n, arc)
    if bool(getattr(cfg, "semicircle_use_reachability", True)):
        poses = _semicircle_from_reachability(
            node, center, ref_quat, source, n, radius, arc, local_axis,
            radial, tangent, angles,
            publish_preview=publish_preview, log=log)
        if poses:
            return poses
        if not bool(getattr(cfg, "semicircle_allow_geometric_fallback", False)):
            if log:
                node.get_logger().warn(
                    "Lidar scan: no horizontal validated reachability arc found; "
                    "aborting scan instead of using an unvalidated geometric arc.")
            return []
        if log:
            node.get_logger().warn(
                "Lidar scan: no usable validated reachability samples for scan arc; "
                "trying generated horizontal semicircle with path validation.")

    if log and bool(getattr(cfg, "local_close_first", True)):
        poses = _local_close_scan_fallback(node, cur, log=log)
        if len(poses) >= int(getattr(cfg, "semicircle_min_points", 3)):
            if publish_preview:
                markers_mod.publish_lidar_scan_preview(node, poses, valid=False)
            return poses

    center_xy = np.array(center[:2], dtype=float)
    z = float(center[2]) + float(getattr(cfg, "semicircle_z_offset_m", 0.0))

    def make_pose(a, r):
        offset_xy = r * (math.cos(a) * radial + math.sin(a) * tangent)
        xyz = [float(center_xy[0] + offset_xy[0]), float(center_xy[1] + offset_xy[1]), z]
        quat = list(ref_quat)
        if bool(getattr(cfg, "semicircle_face_target", True)):
            desired = [center[0] - xyz[0], center[1] - xyz[1], center[2] - xyz[2]]
            quat = yaw_only_align_local_axis(ref_quat, desired, local_axis=local_axis)
        return xyz + list(quat)

    radius_candidates = [
        max(0.02, float(r))
        for r in getattr(cfg, "semicircle_radius_candidates_m", [radius])
    ]
    if radius not in radius_candidates:
        radius_candidates.append(radius)
    radius_candidates = list(dict.fromkeys(radius_candidates))
    if log and bool(getattr(cfg, "semicircle_adaptive_scan", True)):
        node.get_logger().info(
            "Lidar scan: adaptive radius candidates close-first="
            f"{[round(r, 3) for r in radius_candidates]}")

    poses = []
    for a in angles:
        poses.append(make_pose(float(a), radius))

    if log:
        if bool(getattr(cfg, "semicircle_adaptive_scan", True)):
            candidate_rows = [
                [make_pose(float(a), r) for r in radius_candidates]
                for a in angles
            ]
            poses = _filter_adaptive_cartesian_scan_sequence(
                node, candidate_rows, log=log)
        else:
            poses = _filter_cartesian_scan_sequence(node, poses, log=log)
        if len(poses) < 2:
            poses = _local_close_scan_fallback(node, cur, log=log)
            if len(poses) < 2:
                return []

    if publish_preview:
        markers_mod.publish_lidar_scan_preview(node, poses, valid=False)
    if log:
        node.get_logger().info(
            f"Lidar scan: generated {len(poses)} semicircle points around {source} "
            f"center={[round(v, 3) for v in center]} radius={radius:.2f}m "
            f"z={z:.3f}m angle_mode={angle_mode} "
            f"angles={[round(math.degrees(float(a)), 1) for a in angles]}")
    return poses


def _semicircle_from_reachability(node, center, ref_quat, source, n, radius, arc,
                                  local_axis, radial, tangent, desired_angles,
                                  *, publish_preview: bool = True,
                                  log: bool = True):
    samples = list(getattr(node, "_latest_reachability_samples", []) or [])
    if not samples:
        return []
    cfg = node.cfg.lidar_scan
    cur = node.get_end_effector_pose()
    if cur is None or len(cur) < 7:
        return []
    center_xy = np.array(center[:2], dtype=float)
    candidates = []
    max_radius_error = float(getattr(
        cfg, "semicircle_reachability_max_radius_error_m", 0.12))
    target_z = float(center[2]) + float(getattr(cfg, "semicircle_z_offset_m", 0.0))
    z_tol = float(getattr(cfg, "semicircle_z_tolerance_m", 0.02))
    for s in samples:
        if not bool(s.get("valid_direct", False)):
            continue
        if s.get("display_delta_deg", s.get("delta_deg", 999.0)) > float(getattr(
                node.cfg.planner, "safe_zone_verified_interp_max_delta_deg", 80.0)):
            continue
        xyz = np.array(s["xyz"], dtype=float)
        z_err = abs(float(xyz[2]) - target_z)
        if z_err > z_tol:
            continue
        vec = xyz[:2] - center_xy
        r = float(np.linalg.norm(vec))
        if r < 1e-4 or abs(r - radius) > max_radius_error:
            continue
        x_comp = float(np.dot(vec, radial))
        y_comp = float(np.dot(vec, tangent))
        angle = math.atan2(y_comp, x_comp)
        nearest_desired = min(
            (float(a) for a in desired_angles),
            key=lambda a: abs(_angle_near_reference(angle, a) - a),
        )
        angle = _angle_near_reference(angle, nearest_desired)
        if (angle < min(desired_angles) - 1e-6
                or angle > max(desired_angles) + 1e-6):
            continue
        candidates.append((angle, r, z_err, s))

    if len(candidates) < 2:
        if log:
            node.get_logger().warn(
                f"Lidar scan: only {len(candidates)} validated reachability samples "
                f"near horizontal scan plane z={target_z:.3f}±{z_tol:.3f}m")
        return []

    chosen = []
    used = set()
    for a in desired_angles:
        best = None
        best_score = float("inf")
        for idx, (angle, r, z_err, sample) in enumerate(candidates):
            if idx in used:
                continue
            score = abs(angle - float(a)) + 0.8 * abs(r - radius) + 2.0 * z_err
            if score < best_score:
                best_score = score
                best = (idx, sample)
        if best is None:
            continue
        used.add(best[0])
        sample = best[1]
        pose = list(sample.get("pose") or [])
        if len(pose) < 7:
            pose = list(sample["xyz"]) + list(ref_quat)
        # Keep the exact sampled orientation/joints: these are what made the
        # point reachability-valid. Re-orienting here would create a new IK problem.
        chosen.append({"pose": pose, "joints": list(sample["joints"])})

    chosen = _filter_reachable_scan_sequence(node, chosen, log=log)
    if len(chosen) < 2:
        return []
    if publish_preview:
        markers_mod.publish_lidar_scan_preview(node, chosen, valid=True)
    if log:
        node.get_logger().info(
            f"Lidar scan: selected {len(chosen)} validated reachability scan points "
            f"around {source} center={[round(v, 3) for v in center]} "
            f"radius={radius:.2f}m z={target_z:.3f}±{z_tol:.3f}m")
        node.get_logger().info(
            "Lidar scan points: "
            + " | ".join(
                f"{i + 1}:[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
                for i, p in enumerate(_target_pose(t)[:3] for t in chosen)
            )
        )
    return chosen


def _local_close_scan_fallback(node, cur_pose, *, log: bool = True):
    cfg = node.cfg.lidar_scan
    if not bool(getattr(cfg, "local_close_fallback_enabled", True)):
        return []
    if cur_pose is None or len(cur_pose) < 7:
        return []

    axis = np.array(getattr(cfg, "local_close_axis", [0.0, 1.0, 0.0])[:3], dtype=float)
    axis[2] = 0.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        axis /= norm

    offsets = [float(v) for v in getattr(cfg, "local_close_offsets_m", [0.0, 0.04, 0.08])]
    base = [float(v) for v in cur_pose[:3]]
    quat = list(cur_pose[3:7])
    candidates = []
    for off in offsets:
        xyz = [
            base[0] + float(axis[0]) * off,
            base[1] + float(axis[1]) * off,
            base[2],
        ]
        candidates.append(xyz + quat)

    poses = _filter_cartesian_scan_sequence(node, candidates, log=log)
    if log:
        if poses:
            node.get_logger().warn(
                f"Lidar scan: using local close fallback sweep ({len(poses)} points) "
                "with current tool orientation.")
        else:
            node.get_logger().warn("Lidar scan: local close fallback found no valid points.")
    return poses


def _cartesian_scan_candidate(node, prev_joints, pose, label: str,
                              *, require_horizontal: bool,
                              log_rejection: bool):
    plan = motions_mod._plan_cartesian_states_from_joints(
        node, prev_joints, pose, label, plan_cfg=PLAN_CFG_SCAN_PREFLIGHT)
    if plan is None:
        return None
    states, curobo_dt = plan
    states = _unwrap_states_near_previous(prev_joints, states)
    if not _scan_joint_motion_ok(
            node, prev_joints, states, label, log_rejection=log_rejection):
        return None
    if not motions_mod._manual_cartesian_path_inside_safe_zone(node, states, label):
        return None
    if require_horizontal and not _states_stay_horizontal(
            node, states, target_z=float(pose[2]), label=label,
            log_rejection=log_rejection):
        return None
    max_total, max_step = _scan_joint_motion_metrics(prev_joints, states)
    return {
        "pose": pose,
        "states": states,
        "dt": curobo_dt,
        "end_joints": list(states[-1]),
        "score": max_total + 0.5 * max_step + 0.0005 * len(states),
    }


def _filter_adaptive_cartesian_scan_sequence(node, candidate_rows, *, log: bool = True):
    """Choose a smooth reachable scan sweep; radius/angle samples may be skipped."""
    if node.current_joint_positions is None:
        return []
    cfg = node.cfg.lidar_scan
    max_failed = max(1, int(getattr(cfg, "semicircle_max_failed_candidates", 3)))
    min_points = max(2, int(getattr(cfg, "semicircle_min_points", 3)))

    def evaluate(rows, direction_label):
        kept = []
        prev_joints = list(node.current_joint_positions)
        failed = 0
        total_score = 0.0
        for i, row in enumerate(rows, 1):
            best = None
            for j, pose in enumerate(row, 1):
                cand = _cartesian_scan_candidate(
                    node,
                    prev_joints,
                    pose,
                    f"SCAN_PREFLIGHT_{direction_label}_{i}_{j}",
                    require_horizontal=bool(kept),
                    log_rejection=False,
                )
                if cand is not None:
                    best = cand
                    break
            if best is None:
                failed += 1
                if not kept and failed >= max_failed:
                    break
                continue
            kept.append({
                "pose": best["pose"],
                "states": best["states"],
                "dt": best["dt"],
            })
            prev_joints = best["end_joints"]
            total_score += best["score"]
            failed = 0
        return kept, total_score

    fwd, fwd_score = evaluate(candidate_rows, "FWD")
    rev, rev_score = evaluate(list(reversed(candidate_rows)), "REV")
    if len(rev) > len(fwd) or (len(rev) == len(fwd) and rev_score < fwd_score):
        kept, direction, score = rev, "reverse", rev_score
    else:
        kept, direction, score = fwd, "forward", fwd_score

    if len(kept) < min_points:
        if log:
            node.get_logger().warn(
                f"Lidar scan: only {len(kept)} adaptive scan point(s) passed "
                f"(need {min_points}); aborting instead of forcing a bad arc.")
        return []

    if log:
        node.get_logger().info(
            f"Lidar scan: adaptive {direction} sweep kept {len(kept)}/"
            f"{len(candidate_rows)} angle slots score={score:.3f}")
    return kept


def _filter_cartesian_scan_sequence(node, poses, *, log: bool = True):
    """Keep generated center-facing poses whose scan sweep stays in the scan box/plane.

    The first kept pose is the pre-recording move-to-start, so it may move out of
    the horizontal plane. The horizontal constraint applies to the recorded sweep
    between scan poses.
    """
    if node.current_joint_positions is None:
        return []
    kept = []
    prev_joints = list(node.current_joint_positions)
    failed_candidates = 0
    max_failed = max(1, int(getattr(
        node.cfg.lidar_scan, "semicircle_max_failed_candidates", 3)))
    for i, pose in enumerate(poses, 1):
        label = f"SCAN_PREFLIGHT_{i}"
        cur_pose = node.get_end_effector_pose()
        if not kept and cur_pose is not None and len(cur_pose) >= 3:
            if math.dist([float(v) for v in cur_pose[:3]], [float(v) for v in pose[:3]]) < 0.005:
                kept.append(pose)
                failed_candidates = 0
                continue
        plan = motions_mod._plan_cartesian_states_from_joints(
            node, prev_joints, pose, label)
        if plan is None:
            failed_candidates += 1
            if log:
                node.get_logger().warn(
                    f"Lidar scan: dropping generated point {i} — no Cartesian plan")
            if not kept and failed_candidates >= max_failed:
                if log:
                    node.get_logger().warn(
                        "Lidar scan: aborting preflight early — first scan points "
                        "need IK/joint jumps on this arc side")
                return []
            continue
        states, _ = plan
        states = _unwrap_states_near_previous(prev_joints, states)
        if not _scan_joint_motion_ok(node, prev_joints, states, label, log_rejection=log):
            failed_candidates += 1
            if log:
                node.get_logger().warn(
                    f"Lidar scan: dropping generated point {i} — joint motion jump too large")
            if not kept and failed_candidates >= max_failed:
                if log:
                    node.get_logger().warn(
                        "Lidar scan: aborting preflight early — first scan points "
                        "need large joint jumps on this arc side")
                return []
            continue
        if not motions_mod._manual_cartesian_path_inside_safe_zone(node, states, label):
            failed_candidates += 1
            if log:
                node.get_logger().warn(
                    f"Lidar scan: dropping generated point {i} — path leaves safe-zone box")
            if not kept and failed_candidates >= max_failed:
                if log:
                    node.get_logger().warn(
                        "Lidar scan: aborting preflight early — first scan points "
                        "leave the safe-zone box on this arc side")
                return []
            continue
        if kept:
            if not _states_stay_horizontal(
                    node, states, target_z=float(pose[2]), label=label, log_rejection=log):
                failed_candidates += 1
                if log:
                    node.get_logger().warn(
                        f"Lidar scan: dropping generated point {i} — path leaves horizontal scan plane")
                continue
        kept.append(pose)
        failed_candidates = 0
        prev_joints = list(states[-1])

    if log:
        node.get_logger().info(
            f"Lidar scan: preflight kept {len(kept)}/{len(poses)} generated scan points")
    return kept


def publish_lidar_scan_preview(node):
    cfg = node.cfg.lidar_scan
    if not bool(getattr(cfg, "semicircle_preview_enabled", True)):
        return
    if not bool(getattr(cfg, "use_semicircle", True)):
        return
    if getattr(node, "motion_phase", "IDLE") != "IDLE":
        return
    try:
        _semicircle_scan_poses(node, publish_preview=True, log=False)
    except Exception as e:
        node.get_logger().debug(f"Lidar scan preview skipped: {e}")


def _target_pose(target):
    return target["pose"] if isinstance(target, dict) else target


def _target_joints(target):
    return target.get("joints") if isinstance(target, dict) else None


def _target_states(target):
    return target.get("states") if isinstance(target, dict) else None


def _target_dt(target):
    return target.get("dt") if isinstance(target, dict) else None


def _filter_reachable_scan_sequence(node, targets, *, log: bool = True):
    if node.current_joint_positions is None:
        return []
    planner = node.cfg.planner
    cap = float(getattr(planner, "safe_zone_verified_interp_max_delta_deg", 80.0))
    step = float(getattr(planner, "safe_zone_verified_interp_step_deg", 1.0))
    filtered = []
    prev = list(node.current_joint_positions)
    for target in targets:
        joints = _target_joints(target)
        if joints is None:
            filtered.append(target)
            continue
        pose = _target_pose(target)
        ok = motions_mod.validate_joint_interpolations_batch(
            node, prev, [joints], max_delta_deg=cap, step_deg=step)
        horizontal_ok = _joint_path_stays_horizontal(
            node, prev, joints, target_z=pose[2], label="Lidar scan",
            step_deg=step, log_rejection=False)
        if ok and ok[0] and horizontal_ok:
            filtered.append(target)
            prev = motions_mod.nearest_joint_config(prev, list(joints))
        else:
            if log:
                node.get_logger().warn(
                    f"Lidar scan: dropping scan point "
                    f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}] — "
                    "not valid from previous scan posture or horizontal scan plane")
    return filtered


def _unwrap_states_near_previous(start_joints, states):
    unwrapped = []
    prev = list(start_joints)
    for state in states:
        s = motions_mod.nearest_joint_config(prev, list(state))
        unwrapped.append(s)
        prev = s
    return unwrapped


def _scan_joint_motion_ok(node, start_joints, states, label: str,
                          *, log_rejection: bool = True) -> bool:
    if not states:
        return False
    cfg = node.cfg.lidar_scan
    total_cap = math.radians(float(getattr(cfg, "semicircle_max_joint_delta_deg", 75.0)))
    step_cap = math.radians(float(getattr(cfg, "semicircle_max_joint_step_deg", 8.0)))
    max_total, max_step = _scan_joint_motion_metrics(start_joints, states)
    max_step_idx = _scan_joint_max_step_index(start_joints, states)
    if max_total > total_cap:
        if log_rejection:
            node.get_logger().warn(
                f"{label}: rejected scan segment because max joint change "
                f"{math.degrees(max_total):.1f}deg exceeds "
                f"{math.degrees(total_cap):.1f}deg")
        return False
    if max_step > step_cap:
        if log_rejection:
            node.get_logger().warn(
                f"{label}: rejected scan segment because joint step "
                f"{math.degrees(max_step):.1f}deg at sample {max_step_idx}/{len(states) - 1} "
                f"exceeds {math.degrees(step_cap):.1f}deg")
        return False
    return True


def _scan_joint_motion_metrics(start_joints, states):
    prev = list(start_joints)
    max_total = 0.0
    max_step = 0.0
    for state in states:
        total = max(abs(float(s) - float(c)) for s, c in zip(state, start_joints))
        step = max(abs(float(s) - float(p)) for s, p in zip(state, prev))
        if total > max_total:
            max_total = total
        if step > max_step:
            max_step = step
        prev = state
    return max_total, max_step


def _scan_joint_max_step_index(start_joints, states):
    prev = list(start_joints)
    max_step = 0.0
    max_step_idx = 0
    for i, state in enumerate(states):
        step = max(abs(float(s) - float(p)) for s, p in zip(state, prev))
        if step > max_step:
            max_step = step
            max_step_idx = i
        prev = state
    return max_step_idx


def _joint_interp_states(start_joints, target_joints, *, step_deg: float):
    target = motions_mod.nearest_joint_config(start_joints, list(target_joints))
    max_delta = max(abs(t - c) for c, t in zip(start_joints, target))
    steps = max(12, int(math.ceil(math.degrees(max_delta) / max(float(step_deg), 0.25))))
    states = []
    for i in range(steps + 1):
        alpha = i / steps
        smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
        states.append([c + smooth * (t - c) for c, t in zip(start_joints, target)])
    return states


def _joint_path_stays_horizontal(node, start_joints, target_joints, *,
                                target_z: float, label: str,
                                step_deg: float,
                                log_rejection: bool = True) -> bool:
    states = _joint_interp_states(start_joints, target_joints, step_deg=step_deg)
    return _states_stay_horizontal(
        node, states, target_z=target_z, label=label, log_rejection=log_rejection)


def _states_stay_horizontal(node, states, *, target_z: float, label: str,
                            log_rejection: bool = True) -> bool:
    cfg = node.cfg.lidar_scan
    z_tol = float(getattr(cfg, "semicircle_path_z_tolerance_m", 0.035))
    points = fk_mod.forward_kinematics_batch(node, states)
    if len(points) != len(states):
        if log_rejection:
            node.get_logger().warn(f"{label}: horizontal scan check unavailable")
        return False
    zs = [float(p.z) for p in points]
    min_z = min(zs)
    max_z = max(zs)
    if min_z < target_z - z_tol or max_z > target_z + z_tol:
        if log_rejection:
            node.get_logger().warn(
                f"{label}: rejected scan segment because TCP leaves horizontal plane "
                f"z={target_z:.3f}±{z_tol:.3f}m "
                f"(observed {min_z:.3f}..{max_z:.3f}m)")
        return False
    return True


def _execute_scan_pose(node, pose, label, speed_factor, *, require_horizontal: bool = True):
    if node.current_joint_positions is None:
        return False
    plan = motions_mod._plan_cartesian_states_from_joints(
        node, list(node.current_joint_positions), pose, label)
    if plan is None:
        return False
    states, curobo_dt = plan
    start_joints = list(node.current_joint_positions)
    states = _unwrap_states_near_previous(start_joints, states)
    if not _scan_joint_motion_ok(node, start_joints, states, label):
        return False
    if not motions_mod._manual_cartesian_path_inside_safe_zone(node, states, label):
        return False
    if require_horizontal and not _states_stay_horizontal(
            node, states, target_z=float(pose[2]), label=label):
        return False
    return motions_mod._publish_cartesian_states(
        node,
        states,
        curobo_dt,
        pose,
        label,
        motion_type="manual",
        speed_factor=speed_factor,
    )


def _execute_scan_target(node, target, label, speed_factor, *, require_horizontal: bool = True):
    cached_states = _target_states(target)
    if cached_states is not None:
        if node.current_joint_positions is None:
            return False
        pose = _target_pose(target)
        start_joints = list(node.current_joint_positions)
        states = _unwrap_states_near_previous(start_joints, cached_states)
        if not _scan_joint_motion_ok(node, start_joints, states, label):
            return False
        if not motions_mod._manual_cartesian_path_inside_safe_zone(node, states, label):
            return False
        if require_horizontal and not _states_stay_horizontal(
                node, states, target_z=float(pose[2]), label=label):
            return False
        return motions_mod._publish_cartesian_states(
            node,
            states,
            float(_target_dt(target) or 0.02),
            pose,
            label,
            motion_type="manual",
            speed_factor=speed_factor,
        )

    joints = _target_joints(target)
    if joints is not None:
        if node.current_joint_positions is None:
            return False
        step = float(getattr(
            node.cfg.planner, "safe_zone_verified_interp_step_deg", 1.0))
        pose = _target_pose(target)
        if require_horizontal and not _joint_path_stays_horizontal(
                node, list(node.current_joint_positions), joints,
                target_z=float(pose[2]), label=label, step_deg=step):
            return False
        return motions_mod.execute_known_joint_goal(
            node,
            joints,
            label=label,
            motion_type="manual",
            speed_factor=speed_factor,
        )
    return _execute_scan_pose(
        node,
        _target_pose(target),
        label,
        speed_factor,
        require_horizontal=require_horizontal,
    )


def _execute_scan_joint(node, joints, label, speed_factor):
    return motions_mod.plan_execute_js(
        node,
        joints,
        label=label,
        motion_type="scan",
        speed_factor=speed_factor,
    )


def run_lidar_scan(node):
    """
    Sweep the arm through a semicircle around the selected scan center while
    recording /livox/lidar and /livox/imu to a ROS 2 bag file.

    With cfg.lidar_scan.use_semicircle=True, the scan center is the first queued
    goal, latest goal, best tracked goal, or current TCP. Set use_semicircle=False
    to use cfg.lidar_scan.scan_waypoints as legacy joint-space waypoints.
    """
    cfg = node.cfg.lidar_scan
    use_semicircle = bool(getattr(cfg, "use_semicircle", True))
    targets = _semicircle_scan_poses(node) if use_semicircle else cfg.scan_waypoints

    if not targets:
        node.get_logger().error("Lidar scan: no scan targets available")
        return

    bag_dir = os.path.expanduser(cfg.bag_dir)
    os.makedirs(bag_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = os.path.join(bag_dir, f"lidar_scan_{timestamp}")

    node.get_logger().info(
        f"Lidar scan: {len(targets)} {'poses' if use_semicircle else 'waypoints'}, "
        f"recording to {bag_path}"
    )
    node.motion_phase = "LIDAR_SCAN"
    bag_proc = None

    try:
        # Move to scan start before recording so motion artefacts are not captured
        node.get_logger().info("Lidar scan: moving to start position")
        if use_semicircle:
            ok = _execute_scan_target(
                node, targets[0], "SCAN_START", cfg.speed_factor,
                require_horizontal=False)
        else:
            ok = _execute_scan_joint(node, targets[0], "SCAN_START", cfg.speed_factor)
        if not ok or node.stop_requested:
            node.get_logger().warn("Lidar scan: could not reach start position, aborting")
            return

        # Start ros2 bag record
        bag_proc = subprocess.Popen(
            ["ros2", "bag", "record", "-o", bag_path,
             "/livox/lidar", "/livox/imu"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        node.get_logger().info("Lidar scan: bag recording started")
        time.sleep(1.0)  # allow recorder to initialise before moving

        # Sweep through remaining waypoints
        for i, target in enumerate(targets[1:], 1):
            if node.stop_requested:
                node.get_logger().info("Lidar scan: interrupted by stop request")
                break
            node.get_logger().info(
                f"Lidar scan: moving to point {i}/{len(targets) - 1}"
            )
            if use_semicircle:
                ok = _execute_scan_target(
                    node, target, f"SCAN_{i}", cfg.speed_factor,
                    require_horizontal=True)
                if not ok:
                    node.get_logger().warn(f"Lidar scan: skipped point {i}; move failed")
            else:
                _execute_scan_joint(node, target, f"SCAN_{i}", cfg.speed_factor)
            time.sleep(0.3)  # brief dwell at each waypoint for full LiDAR sweep

        node.get_logger().info(f"Lidar scan complete — bag saved at {bag_path}")

    except Exception as e:
        node.get_logger().error(f"Lidar scan error: {e}")

    finally:
        if bag_proc is not None:
            bag_proc.terminate()
            try:
                bag_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                bag_proc.kill()
        if not node.stop_requested:
            node.motion_phase = "LIDAR_SCAN_HOME"
            node.get_logger().info("Lidar scan: returning HOME")
            motions_mod.move_to_home_position(node)
        node.motion_phase = "IDLE"
