"""Pollination mode: find the best view of a flower, close in, hold.

Not a harvest. Nothing is grasped and nothing is planned against the date
pipeline's grasp offsets. The sequence is:

    swap to the flower YOLO engine
      -> sweep the existing semicircle arc, scoring each pose by how large the
         flower's segmentation mask is
      -> drive back to the winning pose
      -> close in along the viewing ray to a fixed standoff
      -> hold
      -> retreat, restore the date engine, go home

Why mask area and not confidence: confidence saturates near 1.0 and jitters
frame to frame, so it cannot rank two poses that both clearly see the flower.
Apparent size varies smoothly with view angle, so the largest mask is the pose
looking most squarely at the flower with the least occlusion. Vision publishes
it on /vision/flower_visibility (see vision/node.py).

Why the full sweep instead of stopping at the first good pose: "most visible"
is a maximum, and a maximum is not knowable until every candidate has been
seen. Stopping early takes the first acceptable view, which is a different and
weaker guarantee.
"""

import math
import time

import rclpy

from std_msgs.msg import String

from . import lidar_scan as lidar_scan_mod
from . import motions as motions_mod
from . import pollen_dispenser
from .goals import (_direct_ik_move, align_local_axis_to_vector,
                    goal_is_in_robot_workspace, quat_rotate_vec)
from .vision.config import DEFAULT_WEIGHTS, MODELS_DIR


def _flower_sample(node):
    """Latest [area_px, n, conf, x, y, z] from vision, or None if never seen."""
    data = getattr(node, "latest_flower_visibility", None)
    if data is None or len(data) < 6:
        return None
    return [float(v) for v in data[:6]]


def _switch_model(node, filename, label):
    """Ask vision to load a different engine. It restarts, so this is not free."""
    import os
    path = filename if os.path.isabs(filename) else os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        node.get_logger().error(
            f"[POLLINATION] {label} model not found: {path}")
        return False
    msg = String()
    msg.data = f"model path={path}"
    node._refresh_camera_pub.publish(msg)
    node.get_logger().info(
        f"[POLLINATION] requested {label} model {os.path.basename(path)}; "
        "vision is restarting")
    return True


def _wait_for_vision(node, timeout_s, quiet_s=2.0):
    """Block until vision is publishing again on the NEW engine.

    Two phases, and the first one matters. The dying vision process keeps
    publishing for a moment after the restart is requested, so simply waiting
    for "any message" is satisfied in ~0.1s by a stale frame from the OLD
    model. Seen 2026-09-15: "vision back after 0.1s" against a restart that
    really takes 22-24s, and PREFIX then read the date model and found nothing.

    So wait for the topic to go SILENT first -- that is the old process
    actually exiting -- and only then treat the next message as the new engine.
    """
    t0 = time.time()

    # Phase 1: silence. The old process is gone when nothing has arrived for
    # quiet_s.
    last_seen = time.time()
    node.latest_flower_visibility = None
    while time.time() - t0 < timeout_s:
        if node.stop_requested:
            # Say so. Returning False silently here made a stale stop flag look
            # like an instant, unexplained abort: the model swap was requested
            # and the finally restored it in the same second, with no reason
            # logged anywhere (seen 2026-09-15 16:41:27).
            node.get_logger().warn(
                "[POLLINATION] stop requested while waiting for vision")
            return False
        if getattr(node, "latest_flower_visibility", None) is not None:
            node.latest_flower_visibility = None
            last_seen = time.time()
        elif time.time() - last_seen >= quiet_s:
            break
        time.sleep(0.05)
    else:
        node.get_logger().error(
            f"[POLLINATION] vision never went quiet within {timeout_s:.0f}s; "
            "the restart may not have happened")
        return False

    # Phase 2: the new process publishing.
    while time.time() - t0 < timeout_s:
        if node.stop_requested:
            node.get_logger().warn(
                "[POLLINATION] stop requested while waiting for vision")
            return False
        if getattr(node, "latest_flower_visibility", None) is not None:
            node.get_logger().info(
                f"[POLLINATION] vision back after {time.time() - t0:.1f}s")
            return True
        time.sleep(0.05)
    node.get_logger().error(
        f"[POLLINATION] vision did not publish within {timeout_s:.0f}s")
    return False


def _score_pose_here(node, cfg, label):
    """Settle, then take the largest mask area seen over a few frames.

    Returns (area, xyz, conf, ok, frames_seen, frames_sampled).

    Max rather than mean: a single frame where the flower is momentarily
    occluded (by a leaf moving, or a dropped detection) should not penalise a
    pose that genuinely has the best view.
    """
    time.sleep(max(0.0, float(cfg.settle_s)))
    best_area, best_xyz, best_conf, seen = 0.0, None, 0.0, 0
    n = max(1, int(cfg.samples_per_pose))
    for _ in range(n):
        s = _flower_sample(node)
        if s is not None and s[0] > 0.0:
            seen += 1
            if s[0] > best_area:
                best_area, best_conf = s[0], s[2]
                best_xyz = [s[3], s[4], s[5]]
        time.sleep(float(getattr(cfg, "sample_interval_s", 0.15)))
    # Consistency is the reliability signal, not confidence -- see the table in
    # config.py next to min_confidence. A flower held across a burst is real; a
    # high score on 3 frames of 12 is a flicker.
    _need_frames = max(1, int(math.ceil(
        n * float(getattr(cfg, "min_frame_fraction", 0.5)))))
    ok = (best_area >= float(cfg.min_mask_area_px)
          and seen >= _need_frames
          and best_conf >= float(cfg.min_confidence))
    node.get_logger().info(
        f"[POLLINATION_POSE] {label} area={best_area:.0f}px "
        f"conf={best_conf:.2f} frames_with_flower={seen}/{n} "
        f"(need {_need_frames}) {'CANDIDATE' if ok else 'rejected'}")
    # Raw measurement alongside the verdict. min_mask_area_px is a guess until
    # a real flower has been swept, so a run that rejects everything has to say
    # what it actually saw or there is nothing to tune against.
    return best_area, best_xyz, best_conf, ok, seen, n


def _tip_offset(node, cfg):
    """Distance from TCP to gripper_tip along the pointing axis, from TF."""
    try:
        tf = node.tf_buffer.lookup_transform(
            "tool0", "gripper_tip", rclpy.time.Time())
        t = tf.transform.translation
        d = math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z)
        if d > 1e-4:
            return d
    except Exception:
        pass
    return float(getattr(cfg, "gripper_tip_offset_m", 0.221))


def _slerp(q0, q1, t):
    """Shortest-arc interpolation between two [x,y,z,w] quaternions."""
    a = [float(v) for v in q0]
    b = [float(v) for v in q1]
    dot = sum(a[i] * b[i] for i in range(4))
    if dot < 0.0:
        b = [-v for v in b]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = [a[i] + t * (b[i] - a[i]) for i in range(4)]
    else:
        th = math.acos(dot)
        s0 = math.sin((1.0 - t) * th) / math.sin(th)
        s1 = math.sin(t * th) / math.sin(th)
        out = [s0 * a[i] + s1 * b[i] for i in range(4)]
    n = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / n for v in out]


def _rotate_towards(node, cur, target_quat, swing_deg):
    """Rotate in place toward target_quat, backing off if the IK branch flips.

    A tool rotation is not proportional to joint travel: 44deg of swing cost
    149deg of joint change on 2026-09-15 17:45 because the solver crossed to a
    different wrist branch, and the move was refused against the 60.2deg cap.
    A partial rotation usually stays on the current branch, and even a partial
    turn toward the flower beats none -- the translation carries the remainder.

    num_steps is forced because _direct_ik_move sizes its interpolation from
    Cartesian distance, and a rotation in place travels 0mm: it would fall to
    the 10-step minimum and trip a joint-velocity fault.
    """
    for frac in (1.0, 0.6, 0.3):
        q = (list(target_quat) if frac >= 1.0
             else _slerp(cur[3:7], target_quat, frac))
        if _direct_ik_move(node, [*cur[:3], *q], label="POLLINATION_AIM",
                           motion_type="approach", store_trajectory=False,
                           num_steps=60):
            node.get_logger().info(
                f"[POLLINATION_AIM] rotated {swing_deg * frac:.0f}deg toward "
                f"the flower"
                + ("" if frac >= 1.0
                   else f" ({frac * 100:.0f}% of the {swing_deg:.0f}deg needed "
                        "— a full turn flipped the IK branch)"))
            return True
    return False


def _aim_at(flower_xyz, pos, quat, max_deg):
    """Swing the tool's local Z onto the flower. Returns (quat, want, capped).

    `want` is the swing the flower actually needs, in degrees, computed before
    the cap is applied -- align_local_axis_to_vector clamps internally and
    returns the clamped value, so it cannot tell you whether it ran out of
    travel. Without that, a capped aim looks identical to a satisfied one and
    the residual shows up as the gripper sitting off to one side.
    """
    d = [float(flower_xyz[i]) - float(pos[i]) for i in range(3)]
    dn = math.sqrt(sum(v * v for v in d))
    if dn < 1e-9:
        return list(quat), 0.0, False
    cur = quat_rotate_vec(list(quat), (0.0, 0.0, 1.0))
    cn = math.sqrt(sum(float(v) * float(v) for v in cur))
    want = 0.0
    if cn > 1e-9:
        dot = sum(float(cur[i]) / cn * d[i] / dn for i in range(3))
        want = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    aimed, _ = align_local_axis_to_vector(
        list(quat), d, local_axis=(0.0, 0.0, 1.0), max_angle_deg=float(max_deg))
    return (list(aimed) if aimed is not None else list(quat),
            want, want > float(max_deg) + 0.5)


def _standoff_pose(node, flower_xyz, view_pose, standoff_m, level=True,
                   z_offset=0.0, aim=False, aim_max_deg=25.0, lateral=0.0,
                   log=False):
    """A point `standoff_m` from the flower on the chosen viewing azimuth.

    With `level` (the default), the offset is taken in the horizontal plane and
    the target sits at the FLOWER's height -- the gripper ends up in front of
    it. Backing off along the full 3D ray instead inherits the sweep's upward
    slant and finishes well below the flower, aimed up at it.

    Either way the viewing pose's orientation is kept: that orientation is what
    made the flower most visible, and the arc's reachability filter validated
    it. Only the position moves.
    """
    vx = float(view_pose[0]) - float(flower_xyz[0])
    vy = float(view_pose[1]) - float(flower_xyz[1])
    vz = float(view_pose[2]) - float(flower_xyz[2])
    if level:
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            return None
        scale = float(standoff_m) / norm
        _p = [
            float(flower_xyz[0]) + vx * scale,
            float(flower_xyz[1]) + vy * scale,
            float(flower_xyz[2]) + float(z_offset),
        ]
        if aim:
            _q, _want, _capped = _aim_at(
                flower_xyz, _p, view_pose[3:7], aim_max_deg)
            if log and node is not None and _capped:
                node.get_logger().warn(
                    f"[POLLINATION_AIM] needs {_want:.0f}deg to point at the "
                    f"flower but is CAPPED at {aim_max_deg:.0f}deg — the "
                    f"gripper will sit ~{_want - aim_max_deg:.0f}deg off to "
                    "one side. Raise aim_max_deg.")
            elif log and node is not None:
                node.get_logger().info(
                    f"[POLLINATION_AIM] swung {_want:.0f}deg onto the flower "
                    f"(cap {aim_max_deg:.0f}deg)")
        else:
            _q = [float(v) for v in view_pose[3:7]]
        # Tool-frame lateral trim, applied after the aim so it is perpendicular
        # to the final pointing axis rather than to the arc pose's.
        _lat = float(lateral)
        if abs(_lat) > 1e-6:
            _x = quat_rotate_vec(_q, (1.0, 0.0, 0.0))
            _p = [_p[i] + _lat * float(_x[i]) for i in range(3)]
        return [*_p, *_q]
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm < 1e-6:
        return None
    scale = float(standoff_m) / norm
    _p = [
        float(flower_xyz[0]) + vx * scale,
        float(flower_xyz[1]) + vy * scale,
        float(flower_xyz[2]) + vz * scale,
    ]
    # Aim here too. This branch used to return the arc pose's orientation
    # untouched, so a ray approach pointed wherever the sweep had been looking
    # rather than at the flower -- and because the quaternion was unchanged the
    # pre-rotation reported "orientation already correct" and skipped.
    _q = (_aim_at(flower_xyz, _p, view_pose[3:7], aim_max_deg)[0] if aim
          else [float(v) for v in view_pose[3:7]])
    if abs(lateral) > 1e-6:
        _x = quat_rotate_vec(_q, (1.0, 0.0, 0.0))
        _p = [_p[i] + float(lateral) * float(_x[i]) for i in range(3)]
    return [*_p, *_q]


def _goto_arc_pose(node, targets, idx, label, speed_factor):
    """Fly to one arc pose, planning fresh from wherever the arm actually is.

    Deliberately NOT _execute_scan_target: that replays states cached for a
    start at the PREVIOUS arc pose. Now that the arc is centred on the flower
    rather than on the TCP, the arm begins off-arc, so the very first cached
    segment mismatches -- and because each cached segment assumes its
    predecessor flew, one rejection cascades through the rest. Seen 2026-09-15:
    ARC_1 was refused for leaving the scan plane, then ARC_2..6 were all refused
    with "joint step ~32deg at sample 0", and only 1 of 7 poses ever flew.

    require_horizontal=False because that check enforces a planar sweep for
    LiDAR data quality, not safety. Entering the arc from an off-arc start
    necessarily dips out of plane (46mm on the run above). Clearance, safe-zone
    and joint-step limits all still apply -- those are the real guards.
    """
    return lidar_scan_mod._execute_scan_pose(
        node, lidar_scan_mod._target_pose(targets[idx]), label, speed_factor,
        require_horizontal=False)


def _plan_standoff(node, cfg, flower_xyz, view_pose, log=False, level_only=False,
                   standoff_override=None):
    """Best reachable standoff from this viewing pose, or (None, "").

    Walks level-at-flower-height first, shortening if the far side of a distant
    flower puts the point outside the arm's reach, then falls back to the
    slanted viewing ray. Returns (pose, description).
    """
    level = bool(getattr(cfg, "approach_at_flower_height", True))
    reach = float(getattr(cfg, "max_reach_m", 1.25))
    so = float(standoff_override if standoff_override is not None
               else cfg.standoff_m)
    # The TCP has to sit a further tip-offset back so the TIP lands at standoff.
    tip = _tip_offset(node, cfg)
    tries = ([(so, True), (so * 1.4, True), (so * 0.7, True)]
             if level else [])
    # The slanted ray lands the gripper well below the flower -- the very
    # behaviour the level approach exists to replace -- so it is opt-in.
    if not level_only and bool(getattr(cfg, "allow_ray_fallback", False)):
        tries.append((so, False))
    for _so, _lv in tries:
        t = _standoff_pose(
            node, flower_xyz, view_pose, _so + tip, level=_lv,
            z_offset=float(getattr(cfg, "approach_z_offset_m", 0.0)),
            aim=bool(getattr(cfg, "aim_tool_at_flower", True)),
            aim_max_deg=float(getattr(cfg, "aim_max_deg", 25.0)),
            lateral=float(getattr(cfg, "approach_lateral_offset_m", 0.0)),
            log=log)
        if t is None:
            continue
        d = math.dist([0.0, 0.0, 0.0], t[:3])
        if d > reach:
            if log:
                node.get_logger().info(
                    f"[POLLINATION] {'level' if _lv else 'ray'} standoff "
                    f"{_so * 100:.0f}cm would sit {d:.2f}m from base "
                    f"(limit {reach:.2f}m); trying closer")
            continue
        return t, (f"{'level with' if _lv else 'along the ray to'} the flower, "
                   f"tip {_so * 100:.0f}cm out (TCP {(_so + tip) * 100:.0f}cm)")
    return None, ""


def _walk_arc_to(node, targets, from_idx, to_idx, speed_factor):
    """Step along the arc from from_idx to to_idx, planning each leg fresh."""
    if from_idx == to_idx:
        return True
    step = -1 if to_idx < from_idx else 1
    node.get_logger().info(
        f"[POLLINATION] retracing arc {from_idx + 1} -> {to_idx + 1} "
        f"({abs(to_idx - from_idx)} step(s))")
    for j in range(from_idx + step, to_idx + step, step):
        label = f"POLLINATION_BACK_{j + 1}"
        if not _goto_arc_pose(node, targets, j, label, speed_factor):
            node.get_logger().error(f"[POLLINATION] {label}: leg rejected")
            return False
    return True


def _search_for_flower(node, cfg, scan_cfg, speed_factor):
    """Sweep the arm around its own position hunting for a flower.

    Runs only when the start pose sees nothing. This is a SEARCH, not a
    ranking: it stops at the first pose with a confirmed, reachable sighting,
    because one good fix is all the real sweep needs to re-centre itself. The
    arc is deliberately wider than the scoring sweep -- it is covering ground,
    not comparing views.

    Returns the flower position in base_link, or None.
    """
    _saved = (scan_cfg.semicircle_points,
              scan_cfg.semicircle_start_deg,
              scan_cfg.semicircle_end_deg)
    try:
        scan_cfg.semicircle_points = int(getattr(cfg, "search_poses", 7))
        _half = float(getattr(cfg, "search_arc_deg", 180.0)) / 2.0
        scan_cfg.semicircle_start_deg = 90.0 - _half
        scan_cfg.semicircle_end_deg = 90.0 + _half

        # No centre override: the flower's position is the unknown here, so the
        # arc can only be built around the arm itself.
        targets = lidar_scan_mod._semicircle_scan_poses(node)
        if not targets:
            node.get_logger().warn(
                "[POLLINATION_SEARCH] no reachable search arc from here")
            return None

        node.get_logger().info(
            f"[POLLINATION_SEARCH] sweeping {len(targets)} poses over "
            f"{scan_cfg.semicircle_start_deg:.0f}..{scan_cfg.semicircle_end_deg:.0f}deg "
            "looking for the flower")

        for i in range(len(targets)):
            if node.stop_requested:
                return None
            label = f"POLLINATION_SEARCH_{i + 1}/{len(targets)}"
            if not _goto_arc_pose(node, targets, i, label, speed_factor):
                node.get_logger().warn(
                    f"[POLLINATION_SEARCH] {label}: move rejected, skipping")
                continue
            area, xyz, conf, ok, _, _ = _score_pose_here(node, cfg, label)
            if not (ok and xyz is not None):
                continue
            # A sighting the arm cannot orbit is not a usable centre, so keep
            # looking rather than committing to it.
            ws_ok, ws_why = goal_is_in_robot_workspace(node, xyz)
            if not ws_ok:
                node.get_logger().info(
                    f"[POLLINATION_SEARCH] {label}: flower at "
                    f"[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}] is outside the "
                    f"workspace ({ws_why}); continuing the search")
                continue
            node.get_logger().info(
                f"[POLLINATION_SEARCH] found it at {label}: "
                f"[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}] "
                f"({area:.0f}px conf={conf:.2f}) — re-centring the sweep")
            return xyz

        node.get_logger().warn(
            "[POLLINATION_SEARCH] swept the whole search arc without a usable "
            "flower")
        return None
    finally:
        (scan_cfg.semicircle_points,
         scan_cfg.semicircle_start_deg,
         scan_cfg.semicircle_end_deg) = _saved


def run_pollination(node):
    """Full pollination cycle. Returns True if the flower was visited."""
    cfg = node.cfg.pollination
    if not bool(getattr(cfg, "enabled", True)):
        node.get_logger().warn("[POLLINATION] disabled in config")
        return False

    scan_cfg = node.cfg.lidar_scan
    saved = {
        "semicircle_points": scan_cfg.semicircle_points,
        "semicircle_arc_deg": scan_cfg.semicircle_arc_deg,
        "semicircle_radius_m": scan_cfg.semicircle_radius_m,
        "semicircle_start_deg": scan_cfg.semicircle_start_deg,
        "semicircle_end_deg": scan_cfg.semicircle_end_deg,
        "semicircle_radius_candidates_m": list(
            scan_cfg.semicircle_radius_candidates_m),
        "semicircle_z_offset_m": scan_cfg.semicircle_z_offset_m,
    }
    # Nothing tracks the live engine path, so fall back to the configured date
    # model. Restoring the wrong engine would leave harvest running on the
    # flower model, which is worse than a redundant reload.
    prior_weights = getattr(node, "_active_vision_weights", None) or DEFAULT_WEIGHTS
    # Whatever mode vision was in when we took over, put it back. Pollination is
    # a standalone action started from an idle system, so leaving vision paused
    # on the way out silently kills date detection -- the harvest then sits
    # waiting for goals that never arrive and looks hung. "paused" was copied
    # from the harvest FINAL pattern, where the harvest loop owns vision and
    # re-pauses it deliberately; nothing owns it here.
    prior_mode = str(getattr(node, "_vision_mode_state", None) or "full").strip()
    if prior_mode not in ("full", "reacquire", "reacquire_fast"):
        prior_mode = "full"
    visited = False

    try:
        node.motion_phase = "POLLINATION"
        # Pressing Pollinate is a deliberate new action, so clear any stop left
        # over from a previous one -- the same thing home, dropoff, manual goal
        # and execute-batch all do before they start. Without it a stale flag
        # aborted the run inside the first wait, before the arm ever moved.
        if node.stop_requested:
            node.get_logger().info(
                "[POLLINATION] clearing a stop left over from a previous action")
            node.stop_requested = False
        # Say up front whether this run will dispense. The spray only happens
        # ~60s in, at the flower, so "is the blower armed?" should be answerable
        # before the arm moves rather than discovered when it fires.
        if bool(getattr(cfg, "spray_enabled", False)):
            _g = float(getattr(cfg, "spray_grams", 5.0))
            try:
                _secs = pollen_dispenser.grams_to_seconds(_g)
                _cycle = f"{_secs:.2f}s blow, " \
                         f"~{_secs + pollen_dispenser.FIRMWARE_OVERHEAD_S:.1f}s cycle"
            except ValueError as _exc:
                _cycle = f"INVALID DOSE: {_exc}"
            node.get_logger().warn(
                f"[POLLINATION_SPRAY] ARMED — will dispense ~{_g:.1f}g via "
                f"{getattr(cfg, 'spray_port', '/dev/ttyUSB0')} ({_cycle}) "
                f"inside the {float(cfg.dwell_s):.0f}s hold; "
                f"spray_required={bool(getattr(cfg, 'spray_required', False))}")
        else:
            node.get_logger().info(
                "[POLLINATION_SPRAY] disabled — dry run, the arm will visit "
                "the flower and hold without dispensing")

        if not _switch_model(node, cfg.flower_weights, "flower"):
            return False
        if not _wait_for_vision(node, 40.0):
            return False
        node.set_vision_mode("full")

        # The arc generator reads these off lidar_scan. Borrow its geometry
        # rather than duplicating the reachability/clearance logic, and put the
        # values back in the finally so a later LiDAR scan is unaffected.
        scan_cfg.semicircle_points = int(cfg.arc_poses)
        scan_cfg.semicircle_arc_deg = float(cfg.arc_deg)
        scan_cfg.semicircle_radius_m = float(cfg.arc_radius_m)
        # With semicircle_use_base_angles on -- which is the default -- the span
        # comes from start/end degrees and semicircle_arc_deg is ignored. The
        # first run asked for 120deg and got the full [180..0] sweep because of
        # this. Centre the requested arc on 90deg (straight ahead in base frame)
        # so arc_deg means what it says.
        _half = float(cfg.arc_deg) / 2.0
        scan_cfg.semicircle_start_deg = 90.0 - _half
        scan_cfg.semicircle_end_deg = 90.0 + _half

        # Find the flower BEFORE building the arc, so the sweep orbits it.
        #
        # _scan_center() returns the current TCP unconditionally -- correct for
        # a LiDAR scan of a tree, wrong here. Runs on 2026-09-15 swept arcs
        # centred 43-53cm BELOW the flower: every pose looked upward from the
        # same side rather than around it, and the close-in leg then had to
        # climb half a metre, dropping out of straight-line Cartesian into
        # joint-space with 25-51mm of lateral deviation.
        pre_area, pre_xyz, pre_conf, pre_ok, _, _ = _score_pose_here(
            node, cfg, "POLLINATION_PREFIX")
        center_override = None
        if pre_ok and pre_xyz is not None:
            # Check the centre before spending IK on an arc around it. Each
            # preflight slot costs ~2s and there are a dozen of them, so an
            # unreachable centre burns ~25s to learn nothing. Seen 2026-09-15 as
            # a wall of SCAN_PREFLIGHT_REV_*: IK_FAIL.
            # goal_is_in_robot_workspace allows forward to 1.6m, which is past
            # what the arm can actually touch. A fix at 1.61m produced a sweep
            # that could never be approached: the aim needed a branch flip
            # (149deg) and the close-in 163deg, both refused, and the run
            # sprayed from wherever it happened to be standing.
            _ws_ok, _ws_why = goal_is_in_robot_workspace(node, pre_xyz)
            _reach = float(getattr(cfg, "max_reach_m", 1.25))
            _fix_d = math.dist([0.0, 0.0, 0.0], pre_xyz)
            if _ws_ok and _fix_d > _reach:
                node.get_logger().warn(
                    f"[POLLINATION] flower fix is {_fix_d:.2f}m from base, past "
                    f"the arm's {_reach:.2f}m — the sweep can orbit it but the "
                    "close-in will not reach. Move the base nearer the plant.")
            if _ws_ok:
                center_override = pre_xyz
                node.get_logger().info(
                    f"[POLLINATION] centring the arc on the flower at "
                    f"[{pre_xyz[0]:.3f},{pre_xyz[1]:.3f},{pre_xyz[2]:.3f}] "
                    f"(area={pre_area:.0f}px conf={pre_conf:.2f}, "
                    f"{math.dist([0.0, 0.0, 0.0], pre_xyz):.2f}m from base)")
            else:
                node.get_logger().warn(
                    f"[POLLINATION] detected flower at "
                    f"[{pre_xyz[0]:.3f},{pre_xyz[1]:.3f},{pre_xyz[2]:.3f}] is "
                    f"outside the workspace ({_ws_why}); arc will centre on the "
                    "TCP instead")
        elif bool(getattr(cfg, "search_when_not_visible", True)):
            # Hunt for it rather than handing the problem back to the operator.
            node.get_logger().warn(
                "[POLLINATION] no flower from the start pose "
                f"(saw {pre_area:.0f}px, conf {pre_conf:.2f}) — searching")
            _found = _search_for_flower(
                node, cfg, scan_cfg,
                float(getattr(cfg, "arc_speed_factor", scan_cfg.speed_factor)))
            if _found is not None:
                center_override = _found
                node.get_logger().info(
                    f"[POLLINATION] centring the arc on the flower found by "
                    f"search at [{_found[0]:.3f},{_found[1]:.3f},{_found[2]:.3f}] "
                    f"({math.dist([0.0, 0.0, 0.0], _found):.2f}m from base)")
            elif bool(getattr(cfg, "require_prefix_detection", True)):
                node.get_logger().error(
                    "[POLLINATION] the search found no flower either — move the "
                    "arm nearer the plant and run again.")
                return False
            else:
                node.get_logger().warn(
                    "[POLLINATION] search found nothing; falling back to an arc "
                    "around the current TCP")
        elif bool(getattr(cfg, "require_prefix_detection", True)):
            node.get_logger().error(
                "[POLLINATION] no flower visible from the start pose "
                f"(saw {pre_area:.0f}px, conf {pre_conf:.2f}) — aim the arm at "
                "the flower and run again. Sweeping from here would circle the "
                "TCP instead of the flower, so most poses would see nothing.")
            return False
        else:
            node.get_logger().warn(
                "[POLLINATION] no flower visible from the start pose; falling "
                "back to an arc around the current TCP. Aim the arm at the "
                "flower before starting for a sweep that actually orbits it.")

        # Orbit at the distance the arm is already standing off at. lidar_scan
        # tries its radius candidates close-first, so left alone it picks 0.22m
        # -- a tight orbit that sweeps the arm through the leaves around the
        # flower on every pose. Where the arm is now is, by definition, a
        # distance it reached without hitting anything.
        #
        # Candidates are ordered safe-first with the tighter ones kept as
        # fallbacks: a flower far from the base cannot be orbited widely (the
        # far side of the arc runs past the arm's reach), so allow it to pull in
        # rather than fail outright.
        if (center_override is not None
                and bool(getattr(cfg, "arc_radius_from_current", True))):
            _cur = lidar_scan_mod._current_tcp_pose(node)
            if _cur is not None:
                # Drop the sweep plane back to the arm's starting height. The
                # flower stays the orbit centre in XY so the arc still circles
                # it; only the height changes, out of the leaves.
                if bool(getattr(cfg, "arc_at_current_height", True)):
                    _dz = float(_cur[2]) - float(center_override[2])
                    scan_cfg.semicircle_z_offset_m = _dz
                    node.get_logger().info(
                        f"[POLLINATION] sweeping at the arm's height "
                        f"z={_cur[2]:.3f} ({abs(_dz) * 100:.0f}cm "
                        f"{'below' if _dz < 0 else 'above'} the flower), "
                        "not in the canopy")
                # Horizontal radius: the arc is a level circle around the
                # flower's XY, so the 3D distance would overstate it by the
                # height difference and pull the orbit in too tight.
                _r_now = math.dist(_cur[:2], center_override[:2])
                # getattr with defaults throughout: node.cfg is built at
                # startup while this module is imported lazily on the first
                # button press, so a node started before a config change runs
                # new code against an old Pollination object. That skew should
                # degrade to defaults, not abort the run with AttributeError
                # (seen 2026-09-15 on arc_radius_min_m).
                _r_arc = min(max(_r_now, float(getattr(
                                 cfg, "arc_radius_min_m", 0.25))),
                             float(getattr(cfg, "arc_radius_max_m", 0.60)))
                scan_cfg.semicircle_radius_m = _r_arc
                scan_cfg.semicircle_radius_candidates_m = list(
                    dict.fromkeys([round(_r_arc, 3),
                                   float(getattr(cfg, "arc_radius_m", 0.35)),
                                   0.22]))
                node.get_logger().info(
                    f"[POLLINATION] arm is {_r_now * 100:.0f}cm from the flower; "
                    f"sweeping at {_r_arc * 100:.0f}cm "
                    f"(fallbacks {scan_cfg.semicircle_radius_candidates_m[1:]})")

        targets = lidar_scan_mod._semicircle_scan_poses(
            node, center_override=center_override)
        if not targets and center_override is not None:
            # A flower-centred arc can be geometrically fine yet unreachable --
            # the arm has to orbit a point it can only just touch. Falling back
            # to the TCP-centred arc gives a worse viewing geometry but still
            # sweeps, which beats abandoning the visit.
            node.get_logger().warn(
                "[POLLINATION] no reachable arc around the flower; retrying "
                "centred on the TCP")
            targets = lidar_scan_mod._semicircle_scan_poses(node)
        if not targets:
            node.get_logger().error(
                "[POLLINATION] no reachable arc poses; nothing to sweep")
            return False
        # Radius is not ours to state: semicircle_adaptive_scan picks from its
        # own candidate list, so arc_radius_m is an upper preference, not the
        # value flown. The generator logs the radii it actually chose.
        node.get_logger().info(
            f"[POLLINATION] sweeping {len(targets)} poses over "
            f"{scan_cfg.semicircle_start_deg:.0f}..{scan_cfg.semicircle_end_deg:.0f}deg")

        _arc_speed = float(getattr(cfg, "arc_speed_factor",
                                   scan_cfg.speed_factor))
        _max_drift = float(getattr(cfg, "max_center_drift_m", 0.20))
        cands = []
        drifted = []
        seen_any = []
        last_idx = -1
        _forced_best = None
        for i, target in enumerate(targets):
            if node.stop_requested:
                node.get_logger().warn("[POLLINATION] stop requested during sweep")
                return False
            label = f"POLLINATION_ARC_{i + 1}/{len(targets)}"
            if not _goto_arc_pose(node, targets, i, label, _arc_speed):
                node.get_logger().warn(f"[POLLINATION] {label}: move rejected, skipping")
                continue
            last_idx = i
            area, xyz, conf, ok, _seen, _n = _score_pose_here(node, cfg, label)
            seen_any.append((label, area, conf))
            if ok and xyz is not None:
                # Is this even the flower we came for? The arc is centred on
                # the PREFIX fix, so a sighting far from that centre is either
                # another bloom or a bad depth reading of this one -- neither is
                # a position to drive to. Selecting purely on area let a point
                # 48cm away win and put the arm 50cm below the target.
                if center_override is not None:
                    _drift = math.dist(xyz, center_override)
                    if _drift > _max_drift:
                        node.get_logger().info(
                            f"[POLLINATION] {label}: sighting is "
                            f"{_drift * 100:.0f}cm from the arc centre "
                            f"(limit {_max_drift * 100:.0f}cm) — not this "
                            "flower, or a bad depth read; skipping")
                        drifted.append({"area": area, "xyz": xyz, "conf": conf,
                                        "target": target, "label": label,
                                        "index": i})
                        continue
                cands.append({"area": area, "xyz": xyz, "conf": conf,
                              "target": target, "label": label, "index": i})

                # Good enough: the flower was held in EVERY frame of the burst
                # and the model is confident about it. Stop here rather than
                # sweeping on and having to come back -- the retrace is the step
                # that keeps failing on joint limits, and the arm is already
                # standing where it needs to be.
                _stop_conf = float(getattr(cfg, "stop_on_confidence", 0.4))
                if (bool(getattr(cfg, "stop_on_full_frames", True))
                        and _seen >= _n and conf >= _stop_conf):
                    _t, _ = _plan_standoff(
                        node, cfg, xyz,
                        lidar_scan_mod._target_pose(target))
                    if _t is not None:
                        node.get_logger().info(
                            f"[POLLINATION] {label} holds the flower in every "
                            f"frame ({_seen}/{_n}) at conf {conf:.2f} "
                            f"(>= {_stop_conf:.2f}) with a reachable standoff "
                            "— stopping the sweep here")
                        # And it IS the winner. Breaking out only ended the
                        # loop; selection then still ran on area and picked a
                        # different pose, so the arm retraced AWAY from the one
                        # it had just stopped at (2026-09-15 17:20: stopped at
                        # ARC_6 conf 0.80, then drove back to ARC_5 conf 0.23).
                        # The whole point of stopping is to be already there.
                        _forced_best = cands[-1]
                        break
                    node.get_logger().info(
                        f"[POLLINATION] {label} is a full-frame view but has "
                        "no reachable standoff; continuing the sweep")

        # The drift gate anchors on the PREFIX fix. If that fix was itself the
        # bad measurement, the gate rejects every real sighting instead -- seen
        # 2026-09-15 18:02, where five independent 6/6 views all landed 47-66cm
        # from the centre and within 19cm of EACH OTHER. Five agreeing
        # measurements outvote one; re-anchor on their consensus.
        if not cands and len(drifted) >= 2:
            _med = [sorted(c["xyz"][k] for c in drifted)[len(drifted) // 2]
                    for k in range(3)]
            _tight = [c for c in drifted
                      if math.dist(c["xyz"], _med) <= _max_drift]
            if len(_tight) >= 2:
                node.get_logger().warn(
                    f"[POLLINATION] every pose was rejected against the PREFIX "
                    f"centre, but {len(_tight)} of {len(drifted)} agree with "
                    f"each other at [{_med[0]:.3f},{_med[1]:.3f},{_med[2]:.3f}] "
                    "— the centre was the bad fix; using their consensus")
                cands = _tight
                center_override = _med

        # Largest mask first, but only among poses with a reachable standoff.
        # On 2026-09-15 the biggest view (40160px at 150deg) sat on the far side
        # of a flower 1.28m out, so every standoff from it was past the arm's
        # reach -- it held at the arc pose, 38cm below the flower, and never
        # went near it. Five of the seven poses were reachable; the winner was
        # one of the two that were not.
        # Two passes. A pose whose only reachable standoff is the slanted ray
        # puts the gripper ~20cm under the flower, so prefer ANY pose that
        # supports a level one -- even a smaller view -- before settling for
        # that. On 2026-09-15 the largest view was accepted on its ray fallback
        # and the arm finished 21cm low.
        best = None
        if _forced_best is not None:
            best = _forced_best
        _ordered = [] if best is not None else sorted(
            cands, key=lambda c: c["area"], reverse=True)
        for _lvl_only in (True, False):
            for _c in _ordered:
                _t, _ = _plan_standoff(
                    node, cfg, _c["xyz"],
                    lidar_scan_mod._target_pose(_c["target"]),
                    level_only=_lvl_only)
                if _t is not None:
                    best = _c
                    break
            if best is not None:
                if not _lvl_only:
                    node.get_logger().warn(
                        "[POLLINATION] no pose supports a level standoff; "
                        "falling back to the slanted ray, which sits below "
                        "the flower")
                break
            node.get_logger().info(
                "[POLLINATION] no pose has a reachable level standoff"
                + ("; allowing the slanted ray" if _lvl_only else ""))
        if best is None and cands:
            best = max(cands, key=lambda c: c["area"])
            node.get_logger().warn(
                "[POLLINATION] no pose has a reachable standoff; taking the "
                f"largest view ({best['label']}) and holding at the arc pose")

        if best is None:
            _top = max(seen_any, key=lambda r: r[1]) if seen_any else None
            node.get_logger().warn(
                "[POLLINATION] swept the whole arc without a usable flower view "
                f"(need area>={cfg.min_mask_area_px:.0f}px "
                f"conf>={cfg.min_confidence:.2f}); best seen was "
                + (f"{_top[1]:.0f}px conf={_top[2]:.2f} at {_top[0]}"
                   if _top else "nothing at all")
                + " -- lower min_mask_area_px if that looks like a real flower")
            return False

        node.get_logger().info(
            f"[POLLINATION_BEST] {best['label']} area={best['area']:.0f}px "
            f"conf={best['conf']:.2f} flower=["
            f"{best['xyz'][0]:.3f},{best['xyz'][1]:.3f},{best['xyz'][2]:.3f}]")

        # Back to the winning pose, one arc step at a time.
        #
        # Replaying the winner's cached scan segment does not work: those states
        # were generated for a start at the PREVIOUS arc pose, so replaying them
        # from the far end of the sweep begins with a joint jump. Observed
        # 2026-09-15 as "rejected scan segment because joint step 43.4deg at
        # sample 0/99 exceeds 8.0deg", which killed the run after a good sweep.
        #
        # Planning one direct move to the winner does not work either: the arm
        # is at the far end of the arc, and the straight Cartesian chord between
        # the two ends cuts through the arc centre -- which is the plant.
        #
        # So retrace the arc. Each leg is one 30deg step that the forward sweep
        # already flew, and _execute_scan_pose plans it fresh from wherever the
        # arm actually is, so there is no cached-state mismatch.
        if not _walk_arc_to(node, targets, last_idx, best["index"], _arc_speed):
            # Do not throw away a good sweep over a leg the joint limits refused.
            # The arm is standing at a pose that also saw the flower -- on
            # 2026-09-15 17:33 the retrace wanted 200.6deg for one 20deg arc
            # step (an IK branch flip) and the whole run was abandoned after
            # seven successful poses. Spray from where we are instead.
            _here = next((c for c in cands if c["index"] == last_idx), None)
            if _here is None:
                node.get_logger().error(
                    "[POLLINATION] could not return to the best pose, and the "
                    "pose we ended on saw nothing usable")
                return False
            node.get_logger().warn(
                f"[POLLINATION] could not return to {best['label']}; using "
                f"{_here['label']} instead ({_here['area']:.0f}px vs "
                f"{best['area']:.0f}px) — the arm is already standing there")
            best = _here

        # Re-read the flower from the winning pose. Its position was measured
        # on the way past; standing here now gives a fresher, closer fix.
        # Only let the re-fix move the target if it clears the same gate the
        # sweep used. A weak detection here would otherwise drag the standoff
        # point toward a spurious blob, and the sweep's measurement is already
        # good enough to visit.
        area, xyz, conf, _ok, _, _ = _score_pose_here(
            node, cfg, "POLLINATION_REFIX")
        # Only let the re-fix move the target if it is a BETTER measurement than
        # the sweep's winner, not merely a passing one. On 2026-09-15 a re-fix
        # at 40187px / 4-of-6 frames overrode a best view of 51696px / 6-of-6
        # and pulled the target 6.6cm lower. Same flower, three measurements
        # spanning 10.8cm -- so which one is used is not a detail.
        _refix_drift = (math.dist(xyz, center_override)
                        if (xyz is not None and center_override is not None)
                        else 0.0)
        _best_drift = (math.dist(best["xyz"], center_override)
                       if center_override is not None else 0.0)
        if (_ok and xyz is not None and center_override is not None
                and _best_drift > _max_drift >= _refix_drift):
            # Agreement beats pixel count. Two independent fixes landing near
            # the arc centre outrank one larger mask sitting far from it.
            flower_xyz = xyz
            node.get_logger().warn(
                f"[POLLINATION] the sweep's winner is {_best_drift * 100:.0f}cm "
                f"from the arc centre but the re-fix is {_refix_drift * 100:.0f}cm; "
                "taking the re-fix — the winner was not this flower")
        elif _ok and xyz is not None and area >= float(best["area"]):
            flower_xyz = xyz
            node.get_logger().info(
                f"[POLLINATION] re-fix is the stronger view ({area:.0f}px vs "
                f"{best['area']:.0f}px); using it")
        else:
            flower_xyz = best["xyz"]
            node.get_logger().info(
                f"[POLLINATION] keeping the sweep's position "
                f"[{best['xyz'][0]:.3f},{best['xyz'][1]:.3f},"
                f"{best['xyz'][2]:.3f}] ({best['area']:.0f}px); re-fix was "
                f"{area:.0f}px"
                + (f", dz={(xyz[2] - best['xyz'][2]) * 100:+.0f}cm"
                   if xyz is not None else ""))

        view_pose = lidar_scan_mod._target_pose(best["target"])

        # With the arc centred on the flower every sweep pose already sits at
        # the arc radius from it. If standoff_m is larger than that, "closing
        # in" would drive the arm BACKWARDS, so hold instead.
        #
        # Measured in the plane when the approach is level: the arc runs ~40cm
        # below the flower, so the 3D distance is dominated by that height gap
        # and would read "already close enough" while the gripper is still a
        # long way out horizontally -- and still under the flower.
        # Same ladder the selection used, so the pose that was chosen for being
        # reachable is the one actually flown. Do NOT skip on horizontal
        # distance: the sweep runs ~40cm below the flower by design, so the arm
        # can be 24cm away in plan view and still be well under it -- the climb
        # IS the approach.
        approach, _how = _plan_standoff(
            node, cfg, flower_xyz, view_pose, log=True)

        if approach is not None and math.dist(view_pose[:3], approach[:3]) < 0.02:
            node.get_logger().info(
                "[POLLINATION] already in front of the flower; no close-in needed")
            approach = None
        elif approach is None:
            # Cannot get closer, so at least face it -- the viewing angle that
            # won the sweep is already good, and where the nozzle POINTS is what
            # decides whether the spray lands.
            node.get_logger().warn(
                "[POLLINATION] no reachable standoff; holding at the arc pose "
                "and turning to face the flower")
            _cur = lidar_scan_mod._current_tcp_pose(node)
            if _cur is not None:
                _q, _want, _capped = _aim_at(
                    flower_xyz, _cur[:3], _cur[3:7],
                    float(getattr(cfg, "aim_max_deg", 45.0)))
                if _want >= 1.0:
                    _rotate_towards(node, _cur, _q, _want)
        if approach is not None:
            node.get_logger().info(
                f"[POLLINATION_APPROACH] closing {_how}, flower "
                f"[{flower_xyz[0]:.3f},{flower_xyz[1]:.3f},{flower_xyz[2]:.3f}] "
                f"-> [{approach[0]:.3f},{approach[1]:.3f},{approach[2]:.3f}] "
                f"(dz={(approach[2] - flower_xyz[2]) * 100:+.0f}cm, "
                f"moving {math.dist(view_pose[:3], approach[:3]) * 100:.0f}cm)")
            # Rotate first, then translate. Two reasons.
            #
            # The aim rides along with the approach, so a ~13cm move cost
            # 64.5deg of joint travel and was refused against the 60.2deg cap --
            # the wrist swing, not the distance, was most of it.
            #
            # And backing the standoff off to shorten the move does the
            # opposite: the arm already sits closer to the flower than the
            # standoff point, so a larger standoff is a LONGER move. The three
            # retries on 2026-09-15 17:24 went 64.5 -> 65.2 -> 66.8deg while the
            # travel grew 32 -> 34 -> 38cm. Scaling the wrong way.
            #
            # Rotating in place first leaves a near-pure translation, so each
            # leg is well inside the cap -- and if the translation is still
            # refused we are already pointing at the flower, which is what
            # actually matters for spraying.
            # Rotate to the orientation that will be correct AT THE TARGET,
            # then translate. Aiming from the CURRENT pose and carrying that
            # orientation through the move leaves it stale by however much the
            # bearing changes on the way: on 2026-09-15 17:29 the arm aimed from
            # the arc pose, translated 34cm, and ended up facing above the
            # flower -- a 34cm move at 0.47m subtends ~36deg.
            #
            # _standoff_pose already aims from the target position, so
            # approach[3:7] is the orientation we want; adopting it before
            # moving makes the translation a pure translation and leaves the
            # gripper pointing at the flower once it arrives.
            _cur = lidar_scan_mod._current_tcp_pose(node)
            if _cur is not None:
                _dot = abs(sum(float(_cur[3 + i]) * float(approach[3 + i])
                               for i in range(4)))
                _swing = math.degrees(2.0 * math.acos(max(0.0, min(1.0, _dot))))
                if _swing < 1.0:
                    node.get_logger().info(
                        "[POLLINATION_AIM] orientation already correct for the "
                        "target")
                elif not _rotate_towards(node, _cur, approach[3:7], _swing):
                    node.get_logger().warn(
                        "[POLLINATION_AIM] could not rotate at all; the "
                        "approach will carry the rotation instead")

            # Pure translation now: approach already holds the target aim.
            if not _direct_ik_move(node, approach, label="POLLINATION_APPROACH",
                                   motion_type="approach",
                                   store_trajectory=False):
                node.get_logger().warn(
                    "[POLLINATION] close-in refused; holding here — spraying "
                    "from this distance")

        node.get_logger().info(
            f"[POLLINATION_DWELL] at the flower"
            + (f", holding {cfg.dwell_s:.0f}s after the spray"
               if float(cfg.dwell_s) > 0.0 else ", leaving as soon as it sprays"))
        t0 = time.time()

        if bool(getattr(cfg, "spray_enabled", False)):
            # Blocking on purpose. The arm must not move while the blower is
            # running, and the firmware owns the whole on/settle/blow/off
            # cycle, so there is nothing useful to do until it answers.
            node.get_logger().warn(
                "[POLLINATION_SPRAY] firing now — arm is holding at the flower")
            sprayed, detail = pollen_dispenser.dispense(
                node.get_logger(),
                grams=float(getattr(cfg, "spray_grams", 5.0)),
                port=str(getattr(cfg, "spray_port", "/dev/ttyUSB0")),
                baud=int(getattr(cfg, "spray_baud", 9600)))
            if sprayed:
                node.get_logger().info(
                    f"[POLLINATION_SPRAY] dispensed "
                    f"~{float(getattr(cfg, 'spray_grams', 5.0)):.1f}g "
                    f"({detail})")
            else:
                node.get_logger().error(f"[POLLINATION_SPRAY] failed: {detail}")
                if bool(getattr(cfg, "spray_required", False)):
                    node.get_logger().error(
                        "[POLLINATION] spray_required is set; ending the visit")
                    return False

        # Whatever the spray consumed counts toward the hold, so dwell_s=0
        # simply falls through here the moment the blower reports OK.
        while time.time() - t0 < float(cfg.dwell_s):
            if node.stop_requested:
                node.get_logger().warn("[POLLINATION] stop requested during dwell")
                break
            time.sleep(0.2)
        visited = True
        node.get_logger().info(
            f"[POLLINATION_DONE] held {time.time() - t0:.1f}s")

        # Retreat along the same ray before anything else moves, so the wrist
        # does not sweep sideways while still inside the flower.
        #
        # Split like the approach: a combined move back to the arc pose bundles
        # the un-aiming rotation with the translation and gets refused on the
        # 60.2deg cap (63.3deg on 2026-09-15 17:38), which left the arm parked
        # at the flower in a pose HOME could then not plan out of -- ending in
        # an emergency stop. Backing straight out first is the part that
        # matters; the orientation can follow once clear.
        if approach is not None:
            _cur = lidar_scan_mod._current_tcp_pose(node)
            _q_now = list(_cur[3:7]) if _cur is not None else list(view_pose[3:7])
            if not _direct_ik_move(
                    node, [*view_pose[:3], *_q_now],
                    label="POLLINATION_RETREAT", motion_type="approach",
                    store_trajectory=False):
                node.get_logger().warn(
                    "[POLLINATION_RETREAT] straight back-out refused; trying "
                    "the full arc pose in one move")
                _direct_ik_move(
                    node, list(view_pose[:7]), label="POLLINATION_RETREAT",
                    motion_type="approach", store_trajectory=False)
            else:
                # Clear of the flower now, so restoring the arc pose's
                # orientation is safe and cheap.
                _direct_ik_move(
                    node, list(view_pose[:7]),
                    label="POLLINATION_RETREAT_AIM", motion_type="approach",
                    store_trajectory=False, num_steps=60)
        return visited

    except Exception as exc:
        node.get_logger().error(f"[POLLINATION] aborted: {exc}")
        return visited
    finally:
        for key, value in saved.items():
            setattr(scan_cfg, key, value)

        # HOME first, then the model swap. The restore is ~20s of vision
        # restart during which nothing moves, and leaving the arm parked deep in
        # the canopy for that long is the wrong order: get it out to a known
        # pose while the geometry is still fresh, then do the slow bookkeeping.
        if bool(getattr(cfg, "return_home_after", True)) and not node.stop_requested:
            node.motion_phase = "HOME"
            motions_mod.move_to_home_position(node)

        if bool(getattr(cfg, "restore_weights_after", True)) and prior_weights:
            node.get_logger().info(
                "[POLLINATION] restoring the date model — vision restarts, "
                "~20s, Pollinate stays busy until it finishes")
            _switch_model(node, prior_weights, "date")
            _wait_for_vision(node, 40.0)
        node.set_vision_mode(prior_mode)
        node.get_logger().info(
            f"[POLLINATION] vision restored to '{prior_mode}'")
        node.motion_phase = "IDLE"
