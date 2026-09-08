import time
import math
import numpy as np

from . import fk as fk_mod
from . import markers as markers_mod
from . import motions as motions_mod
from .config import PLAN_CFG_SCAN_PREFLIGHT
from .goals import bounded_yaw_align_local_axis, _log_forearm_flange_clearance

def _current_tcp_pose(node, *, attempts: int = 3, sleep_s: float = 0.05):
    """Return the best available current TCP pose, allowing brief FK/cache hiccups."""
    for i in range(max(1, attempts)):
        cur = node.get_end_effector_pose()
        if cur is not None and len(cur) >= 7:
            return [float(v) for v in cur[:7]]
        cached = getattr(getattr(node, "_motion_mgr", None), "_last_ee_pose", None)
        if cached is not None and len(cached) >= 7:
            return [float(v) for v in cached[:7]]
        cached = getattr(node, "_last_ee_pose", None)
        if cached is not None and len(cached) >= 7:
            return [float(v) for v in cached[:7]]
        if i + 1 < attempts:
            time.sleep(max(0.0, float(sleep_s)))
    return None


def _scan_center(node):
    cur = _current_tcp_pose(node)
    if cur is not None and len(cur) >= 7:
        return [float(v) for v in cur[:3]], list(cur[3:7]), "current TCP"
    goals = (
        node.goal_poses.snapshot()
        if hasattr(node.goal_poses, "snapshot")
        else [g for g in node.goal_poses]
    )
    if goals:
        return list(goals[0][:3]), list(goals[0][3:7]) if len(goals[0]) >= 7 else None, "queued goal"
    latest = getattr(node, "latest_goal_pose", None)
    if latest is not None and len(latest) >= 3:
        quat = list(latest[3:7]) if len(latest) >= 7 else None
        return [float(v) for v in latest[:3]], quat, "latest goal"
    best = getattr(node, "best_goal_xyz", None)
    if best is not None and len(best) >= 3:
        cur = _current_tcp_pose(node)
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
    cur = _current_tcp_pose(node)
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


def _semicircle_scan_poses(node, *, publish_preview: bool = True, log: bool = True,
                            stop_check=None):
    cfg = node.cfg.lidar_scan
    center, ref_quat, source = _scan_center(node)
    if center is None:
        if log:
            node.get_logger().error("Lidar scan: no scan center available")
        return []

    cur = _current_tcp_pose(node)
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

    configured_yaw_limit = float(getattr(
        cfg, "semicircle_face_target_max_yaw_deg", 10.0))

    def make_pose(a, r, yaw_limit_deg=None):
        offset_xy = r * (math.cos(a) * radial + math.sin(a) * tangent)
        xyz = [float(center_xy[0] + offset_xy[0]), float(center_xy[1] + offset_xy[1]), z]
        quat = list(ref_quat)
        if bool(getattr(cfg, "semicircle_face_target", True)):
            desired = [center[0] - xyz[0], center[1] - xyz[1], center[2] - xyz[2]]
            yaw_limit = (
                configured_yaw_limit
                if yaw_limit_deg is None
                else float(yaw_limit_deg)
            )
            quat, _, _ = bounded_yaw_align_local_axis(
                ref_quat,
                desired,
                local_axis=local_axis,
                max_delta_deg=yaw_limit,
            )
        return xyz + list(quat)

    radius_candidates = [
        max(0.02, float(r))
        for r in getattr(cfg, "semicircle_radius_candidates_m", [radius])
    ]
    if radius not in radius_candidates:
        radius_candidates.append(radius)
    radius_candidates = list(dict.fromkeys(radius_candidates))
    yaw_candidates = list(dict.fromkeys([
        max(0.0, configured_yaw_limit),
        max(0.0, configured_yaw_limit * 0.5),
        0.0,
    ]))
    candidate_specs = [
        (candidate_radius, candidate_yaw)
        for candidate_radius in radius_candidates
        for candidate_yaw in yaw_candidates
    ]

    def candidate_specs_for_angle(angle_deg):
        if not bool(getattr(cfg, "semicircle_fast_yaw_order", True)):
            return list(candidate_specs)
        angle = abs(float(angle_deg)) % 360.0
        if angle > 180.0:
            angle = 360.0 - angle
        max_yaw = max(yaw_candidates)
        half_yaw = max_yaw * 0.5
        if angle <= 15.0:
            preferred_yaws = [half_yaw, 0.0, max_yaw]
        elif angle < 75.0:
            preferred_yaws = [0.0, half_yaw, max_yaw]
        else:
            preferred_yaws = [max_yaw, half_yaw, 0.0]
        preferred_yaws = list(dict.fromkeys(preferred_yaws))
        return [
            (candidate_radius, candidate_yaw)
            for candidate_radius in radius_candidates
            for candidate_yaw in preferred_yaws
        ]

    def candidate_row_for_angle(angle_deg):
        specs = candidate_specs_for_angle(angle_deg)
        return (
            [make_pose(math.radians(float(angle_deg)), r, yaw) for r, yaw in specs],
            [r for r, _yaw in specs],
            [yaw for _r, yaw in specs],
        )
    if log and bool(getattr(cfg, "semicircle_adaptive_scan", True)):
        node.get_logger().info(
            "Lidar scan: adaptive radius candidates close-first="
            f"{[round(r, 3) for r in radius_candidates]}")

    poses = []
    for a in angles:
        poses.append(make_pose(float(a), radius))

    used_local_fallback = False
    if log:
        if bool(getattr(cfg, "semicircle_adaptive_scan", True)):
            candidate_data = [
                candidate_row_for_angle(math.degrees(float(a)))
                for a in angles
            ]
            candidate_rows = [data[0] for data in candidate_data]
            poses = _filter_adaptive_cartesian_scan_sequence(
                node,
                candidate_rows,
                slot_angles_deg=[math.degrees(float(a)) for a in angles],
                radius_candidate_rows=[data[1] for data in candidate_data],
                yaw_candidate_rows=[data[2] for data in candidate_data],
                bridge_row_factory=candidate_row_for_angle,
                stop_check=stop_check,
                log=log,
            )
        else:
            poses = _filter_cartesian_scan_sequence(node, poses, log=log)
        if len(poses) < 2:
            poses = _local_close_scan_fallback(node, cur, log=log)
            if len(poses) < 2:
                return []
            used_local_fallback = True

    if publish_preview:
        markers_mod.publish_lidar_scan_preview(node, poses, valid=False)
    if log and not used_local_fallback:
        retained_angles = [
            round(float(t["angle_deg"]), 1)
            for t in poses
            if isinstance(t, dict) and "angle_deg" in t
        ]
        retained_radii = [
            round(float(t["radius_m"]), 3)
            for t in poses
            if isinstance(t, dict) and "radius_m" in t
        ]
        retained_yaws = [
            round(float(t["yaw_limit_deg"]), 1)
            for t in poses
            if isinstance(t, dict) and "yaw_limit_deg" in t
        ]
        node.get_logger().info(
            f"Lidar scan: generated {len(poses)} semicircle points around {source} "
            f"center={[round(v, 3) for v in center]} "
            f"radii={retained_radii or [round(radius, 3)]}m "
            f"yaw_limits={retained_yaws or [round(configured_yaw_limit, 1)]}deg "
            f"z={z:.3f}m angle_mode={angle_mode} "
            f"angles={retained_angles or [round(math.degrees(float(a)), 1) for a in angles]}")
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
        return None, "PLAN_OR_IK_FAIL"
    states, curobo_dt = plan
    states = _unwrap_states_near_previous(prev_joints, states)
    if not _scan_joint_motion_ok(
            node, prev_joints, states, label, log_rejection=log_rejection):
        max_total, max_step = _scan_joint_motion_metrics(prev_joints, states)
        total_cap = math.radians(float(getattr(
            node.cfg.lidar_scan, "semicircle_max_joint_delta_deg", 75.0)))
        reason = "JOINT_DELTA" if max_total > total_cap else "JOINT_STEP"
        return None, reason
    clamp_mm, _, _ = _log_forearm_flange_clearance(
        node, states, label, log_result=False)
    clamp_threshold_mm = float(getattr(
        node.cfg.planner, "clamp_safety_threshold_mm", 45.0))
    if not math.isfinite(clamp_mm) or clamp_mm < clamp_threshold_mm:
        clearance_text = "UNKNOWN" if not math.isfinite(clamp_mm) else f"{clamp_mm:.1f}mm"
        return None, f"CLAMP_CLEARANCE_{clearance_text}"
    if not motions_mod._manual_cartesian_path_inside_safe_zone(node, states, label):
        return None, "SAFE_ZONE"
    if require_horizontal and not _states_stay_horizontal(
            node, states, target_z=float(pose[2]), label=label,
            log_rejection=log_rejection):
        return None, "Z_PLANE"
    max_total, max_step = _scan_joint_motion_metrics(prev_joints, states)
    return {
        "pose": pose,
        "states": states,
        "dt": curobo_dt,
        "end_joints": list(states[-1]),
        "score": max_total + 0.5 * max_step + 0.0005 * len(states),
    }, "OK"


def _filter_adaptive_cartesian_scan_sequence(
        node, candidate_rows, *, slot_angles_deg=None,
        radius_candidates_m=None, yaw_candidates_deg=None,
        radius_candidate_rows=None, yaw_candidate_rows=None,
        bridge_row_factory=None,
        stop_check=None,
        log: bool = True):
    """Choose a smooth reachable scan sweep; radius/angle samples may be skipped.

    stop_check, if given, is polled before each angle slot's (expensive, IK/
    collision preflight-based) test_row() call -- e.g. a callable that reports
    whether a goal has already been (re)detected, making the rest of this sweep
    moot. Without it, a full sweep (every angle slot times every radius/yaw
    candidate) always runs to completion even when the goal it was searching
    for shows up seconds into the computation, wasting the remaining time.
    """
    if node.current_joint_positions is None:
        return []
    cfg = node.cfg.lidar_scan
    max_failed = max(1, int(getattr(cfg, "semicircle_max_failed_candidates", 3)))
    min_points = max(2, int(getattr(cfg, "semicircle_min_points", 3)))
    slot_angles_deg = list(slot_angles_deg or range(len(candidate_rows)))
    radius_candidates_m = list(radius_candidates_m or range(
        max((len(row) for row in candidate_rows), default=0)))
    yaw_candidates_deg = list(yaw_candidates_deg or [0.0] * len(radius_candidates_m))
    radius_candidate_rows = list(radius_candidate_rows or [
        list(radius_candidates_m) for _row in candidate_rows
    ])
    yaw_candidate_rows = list(yaw_candidate_rows or [
        list(yaw_candidates_deg) for _row in candidate_rows
    ])
    original_angles = {round(float(a), 6) for a in slot_angles_deg}
    bridge_step_deg = max(1.0, float(getattr(
        cfg, "semicircle_joint_bridge_step_deg", 10.0)))

    def evaluate(indexed_rows, direction_label):
        kept = []
        diagnostics = []
        prev_joints = list(node.current_joint_positions)
        failed = 0
        total_score = 0.0
        stopped_at_joint_boundary = False
        def test_row(row, angle_deg, label_suffix, row_radii, row_yaws):
            best = None
            rejected = []
            for j, pose in enumerate(row, 1):
                cand, reason = _cartesian_scan_candidate(
                    node,
                    prev_joints,
                    pose,
                    f"SCAN_PREFLIGHT_{direction_label}_{label_suffix}_{j}",
                    require_horizontal=bool(kept),
                    log_rejection=False,
                )
                if cand is not None:
                    best = cand
                    best["angle_deg"] = float(angle_deg)
                    best["radius_m"] = float(row_radii[j - 1])
                    best["yaw_limit_deg"] = float(row_yaws[j - 1])
                    break
                rejected.append(
                    f"{float(row_radii[j - 1]):.2f}m/"
                    f"yaw{float(row_yaws[j - 1]):.0f}:{reason}")
            return best, rejected

        def keep_candidate(best, rejected, *, is_bridge):
            nonlocal prev_joints, total_score, failed
            kept.append({
                "pose": best["pose"],
                "states": best["states"],
                "dt": best["dt"],
                "angle_deg": best["angle_deg"],
                "radius_m": best["radius_m"],
                "yaw_limit_deg": best["yaw_limit_deg"],
                "is_bridge": bool(is_bridge),
            })
            diagnostics.append(
                (best["angle_deg"], best["radius_m"], rejected))
            prev_joints = best["end_joints"]
            total_score += best["score"]
            failed = 0

        for sequence_i, (slot_i, row) in enumerate(indexed_rows, 1):
            if stop_check is not None and stop_check():
                diagnostics.append(
                    (float(slot_angles_deg[slot_i]), None, ["ABORTED_GOAL_FOUND"]))
                break
            angle_deg = float(slot_angles_deg[slot_i])
            best, rejected = test_row(
                row,
                angle_deg,
                str(sequence_i),
                radius_candidate_rows[slot_i],
                yaw_candidate_rows[slot_i],
            )

            # A coarse 30-degree arc step can make the IK solver jump branches.
            # Insert 10-degree poses only after such a failure, then retry the
            # requested slot from the newly continued joint posture.
            joint_discontinuity = bool(rejected) and all(
                reason.endswith(("JOINT_DELTA", "JOINT_STEP"))
                for reason in rejected
            )
            if (
                best is None
                and joint_discontinuity
                and kept
                and bridge_row_factory is not None
            ):
                previous_angle = float(kept[-1]["angle_deg"])
                gap = angle_deg - previous_angle
                bridge_count = max(0, int(math.ceil(abs(gap) / bridge_step_deg)) - 1)
                bridge_ok = True
                for bridge_i in range(1, bridge_count + 1):
                    bridge_angle = previous_angle + math.copysign(
                        min(bridge_step_deg * bridge_i, abs(gap)), gap)
                    bridge_data = bridge_row_factory(bridge_angle)
                    if isinstance(bridge_data, tuple) and len(bridge_data) == 3:
                        bridge_row, bridge_radii, bridge_yaws = bridge_data
                    else:
                        bridge_row = bridge_data
                        bridge_radii = radius_candidates_m
                        bridge_yaws = yaw_candidates_deg
                    bridge, bridge_rejected = test_row(
                        bridge_row,
                        bridge_angle,
                        f"{sequence_i}_BRIDGE_{bridge_i}",
                        bridge_radii,
                        bridge_yaws,
                    )
                    if bridge is None:
                        diagnostics.append(
                            (bridge_angle, None, bridge_rejected + ["BRIDGE_FAILED"]))
                        bridge_ok = False
                        stopped_at_joint_boundary = True
                        break
                    keep_candidate(bridge, bridge_rejected, is_bridge=True)
                if bridge_ok:
                    best, rejected = test_row(
                        row,
                        angle_deg,
                        f"{sequence_i}_RETRY",
                        radius_candidate_rows[slot_i],
                        yaw_candidate_rows[slot_i],
                    )

            if best is None:
                diagnostics.append(
                    (angle_deg, None, rejected))
                failed += 1
                # Do not skip across an unreachable IK boundary and later accept
                # an isolated angle. That would no longer be a continuous arc.
                if stopped_at_joint_boundary:
                    break
                if not kept and failed >= max_failed:
                    break
                continue
            keep_candidate(best, rejected, is_bridge=False)
        tested_slots = {d[0] for d in diagnostics}
        for slot_i, _row in indexed_rows:
            angle = float(slot_angles_deg[slot_i])
            if angle not in tested_slots:
                reason = (
                    "NOT_TESTED_ARC_DISCONTINUITY"
                    if stopped_at_joint_boundary
                    else "NOT_TESTED_EARLY_ABORT"
                )
                diagnostics.append((angle, None, [reason]))
        return kept, total_score, diagnostics

    indexed_rows = list(enumerate(candidate_rows))

    preferred = str(getattr(cfg, "semicircle_preflight_direction", "reverse")).strip().lower()
    if preferred not in ("forward", "reverse", "both"):
        preferred = "reverse"

    if preferred == "both":
        fwd, fwd_score, fwd_diag = evaluate(indexed_rows, "FWD")
        rev, rev_score, rev_diag = evaluate(list(reversed(indexed_rows)), "REV")
        if len(rev) > len(fwd) or (len(rev) == len(fwd) and rev_score < fwd_score):
            kept, direction, score, diagnostics = rev, "reverse", rev_score, rev_diag
        else:
            kept, direction, score, diagnostics = fwd, "forward", fwd_score, fwd_diag
    else:
        primary_rows = list(reversed(indexed_rows)) if preferred == "reverse" else indexed_rows
        kept, score, diagnostics = evaluate(primary_rows, preferred[:3].upper())
        direction = preferred
        if (
            len(kept) < min_points
            and bool(getattr(cfg, "semicircle_preflight_fallback_opposite", True))
        ):
            opposite = "forward" if preferred == "reverse" else "reverse"
            opposite_rows = indexed_rows if opposite == "forward" else list(reversed(indexed_rows))
            alt, alt_score, alt_diag = evaluate(opposite_rows, opposite[:3].upper())
            if len(alt) > len(kept) or (len(alt) == len(kept) and alt_score < score):
                kept, direction, score, diagnostics = alt, opposite, alt_score, alt_diag

    if log:
        for angle, selected_radius, rejected in sorted(diagnostics, key=lambda item: item[0]):
            if selected_radius is not None:
                rejected_text = f" prior_rejected={','.join(rejected)}" if rejected else ""
                result = (
                    "BRIDGE_KEPT"
                    if round(float(angle), 6) not in original_angles
                    else "KEPT"
                )
                node.get_logger().info(
                    f"[LIDAR_SLOT] angle={angle:.1f}deg result={result} "
                    f"radius={selected_radius:.2f}m "
                    f"yaw_limit={next((float(t['yaw_limit_deg']) for t in kept if abs(float(t['angle_deg']) - float(angle)) < 1e-6), 0.0):.1f}deg"
                    f"{rejected_text}")
            else:
                node.get_logger().warn(
                    f"[LIDAR_SLOT] angle={angle:.1f}deg result=REJECTED "
                    f"reasons={','.join(rejected) or 'UNKNOWN'}")

    if len(kept) < min_points:
        if log:
            node.get_logger().warn(
                f"Lidar scan: only {len(kept)} adaptive scan point(s) passed "
                f"(need {min_points}); aborting instead of forcing a bad arc.")
        return []

    if log:
        original_kept = sum(
            1 for target in kept if not bool(target.get("is_bridge", False)))
        bridge_kept = len(kept) - original_kept
        bridge_text = f" + {bridge_kept} bridge" if bridge_kept else ""
        node.get_logger().info(
            f"Lidar scan: adaptive {direction} sweep kept {original_kept}/"
            f"{len(candidate_rows)} angle slots{bridge_text} score={score:.3f}")
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
        poses = _semicircle_scan_poses(node, publish_preview=True, log=False)
        if poses and all(isinstance(t, dict) and _target_states(t) is not None for t in poses):
            node._latest_lidar_scan_targets = poses
            node._latest_lidar_scan_joint_seed = (
                list(node.current_joint_positions)
                if node.current_joint_positions is not None else None
            )
            node._latest_lidar_scan_targets_time = time.time()
    except Exception as e:
        node.get_logger().debug(f"Lidar scan preview skipped: {e}")


def _cached_scan_targets(node):
    cfg = node.cfg.lidar_scan
    if not bool(getattr(cfg, "semicircle_preview_cache_enabled", True)):
        return None
    targets = getattr(node, "_latest_lidar_scan_targets", None)
    seed = getattr(node, "_latest_lidar_scan_joint_seed", None)
    stamp = float(getattr(node, "_latest_lidar_scan_targets_time", 0.0) or 0.0)
    if not targets or seed is None or node.current_joint_positions is None:
        return None
    max_age = float(getattr(cfg, "semicircle_preview_cache_max_age_s", 8.0))
    if time.time() - stamp > max_age:
        return None
    max_delta = math.radians(float(getattr(
        cfg, "semicircle_preview_cache_max_joint_delta_deg", 3.0)))
    cur = list(node.current_joint_positions)
    if max(abs(float(a) - float(b)) for a, b in zip(cur, seed)) > max_delta:
        return None
    return targets


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
    clamp_mm, _, _ = _log_forearm_flange_clearance(
        node, states, label, log_result=True)
    clamp_threshold_mm = float(getattr(
        node.cfg.planner, "clamp_safety_threshold_mm", 45.0))
    if not math.isfinite(clamp_mm) or clamp_mm < clamp_threshold_mm:
        node.get_logger().error(
            f"[LIDAR_CLAMP_GUARD] {label}: rejected before motion "
            f"(clearance={clamp_mm:.1f}mm, required={clamp_threshold_mm:.1f}mm)")
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
        clamp_mm, _, _ = _log_forearm_flange_clearance(
            node, states, label, log_result=True)
        clamp_threshold_mm = float(getattr(
            node.cfg.planner, "clamp_safety_threshold_mm", 45.0))
        if not math.isfinite(clamp_mm) or clamp_mm < clamp_threshold_mm:
            node.get_logger().error(
                f"[LIDAR_CLAMP_GUARD] {label}: rejected before motion "
                f"(clearance={clamp_mm:.1f}mm, required={clamp_threshold_mm:.1f}mm)")
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


def run_lidar_scan(node, *, stop_when_goal_found: bool = False,
                   return_home: bool = True):
    """
    Sweep the arm through a semicircle around the selected scan center while
    recording /livox/lidar and /livox/imu to a ROS 2 bag file.

    With cfg.lidar_scan.use_semicircle=True, the scan center is the first queued
    goal, latest goal, best tracked goal, or current TCP. Set use_semicircle=False
    to use cfg.lidar_scan.scan_waypoints as legacy joint-space waypoints.
    """
    cfg = node.cfg.lidar_scan
    goal_found = False

    def _goal_available():
        return bool(stop_when_goal_found and len(node.goal_poses) > 0)
    use_semicircle = bool(getattr(cfg, "use_semicircle", True))
    targets = _cached_scan_targets(node) if use_semicircle else None
    if targets:
        node.get_logger().info(
            f"Lidar scan: using cached validated preview ({len(targets)} poses)")
    else:
        targets = (
            _semicircle_scan_poses(node, stop_check=_goal_available)
            if use_semicircle else cfg.scan_waypoints)
        if use_semicircle and targets and all(
                isinstance(t, dict) and _target_states(t) is not None for t in targets):
            node._latest_lidar_scan_targets = targets
            node._latest_lidar_scan_joint_seed = (
                list(node.current_joint_positions)
                if node.current_joint_positions is not None else None
            )
            node._latest_lidar_scan_targets_time = time.time()

    if not targets:
        if _goal_available():
            node.get_logger().info(
                "[AUTO_HARVEST] Goal found during scan target planning; "
                "skipping sweep for stationary reacquisition")
            return True
        node.get_logger().error("Lidar scan: no scan targets available")
        return False

    node.get_logger().info(
        f"Lidar scan: {len(targets)} "
        f"{'poses' if use_semicircle else 'waypoints'}"
    )
    node.motion_phase = "LIDAR_SCAN"

    try:
        # Move to the first validated scan pose before beginning the sweep.
        node.get_logger().info("Lidar scan: moving to start position")
        if use_semicircle:
            ok = _execute_scan_target(
                node, targets[0], "SCAN_START", cfg.speed_factor,
                require_horizontal=False)
        else:
            ok = _execute_scan_joint(node, targets[0], "SCAN_START", cfg.speed_factor)
        if not ok or node.stop_requested:
            if _goal_available():
                goal_found = True
                node.get_logger().info(
                    "[AUTO_HARVEST] Goal found while moving to scan start; "
                    "holding discovery pose")
                return True
            node.get_logger().warn("Lidar scan: could not reach start position, aborting")
            return False

        if _goal_available():
            goal_found = True
            node.get_logger().info(
                "[AUTO_HARVEST] Goal found at scan start; stopping LiDAR sweep")

        # Sweep through remaining waypoints
        for i, target in enumerate(targets[1:], 1):
            if goal_found or _goal_available():
                goal_found = True
                node.get_logger().info(
                    f"[AUTO_HARVEST] Goal found after scan point {max(0, i - 1)}; "
                    "stopping sweep for stationary reacquisition")
                break
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
                if _goal_available():
                    goal_found = True
                    node.get_logger().info(
                        f"[AUTO_HARVEST] Goal found during scan segment {i}; "
                        "stopping sweep for stationary reacquisition")
                    break
                if not ok:
                    node.get_logger().warn(f"Lidar scan: skipped point {i}; move failed")
            else:
                _execute_scan_joint(node, target, f"SCAN_{i}", cfg.speed_factor)
            time.sleep(0.3)  # brief dwell at each waypoint for full LiDAR sweep

        node.get_logger().info(
            "Lidar scan stopped on detected goal"
            if goal_found else "Lidar scan motion complete")

    except Exception as e:
        node.get_logger().error(f"Lidar scan error: {e}")

    finally:
        if not node.stop_requested and return_home and not goal_found:
            node.motion_phase = "LIDAR_SCAN_HOME"
            node.get_logger().info("Lidar scan: returning HOME")
            motions_mod.move_to_home_position(node)
        node.motion_phase = "IDLE"
    return goal_found
