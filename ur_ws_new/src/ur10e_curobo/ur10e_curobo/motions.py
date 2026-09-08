# ruff: noqa
import itertools, math, time, torch
from typing import List
from curobo.types.math import Pose
from curobo.types.robot import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .config import (PLAN_CFG_DEFAULT, PLAN_CFG_JS, PLAN_CFG_JS_GRAPH,
                     PLAN_CFG_JS_NO_FINETUNE, VOXEL_CONFIG)
from .utils import build_trajectory, wait_until_xyz
from .fk import forward_kinematics, forward_kinematics_batch, pose_from_joints


def interpolated_positions(result):
    interp = result.get_interpolated_plan()
    if isinstance(interp, JointState):
        interp = interp.position
    if not isinstance(interp, torch.Tensor):
        interp = torch.tensor(interp, dtype=torch.float32)
    return interp.to("cpu").tolist()


def get_curobo_dt(result) -> float:
    """Get cuRobo's interpolation dt from a MotionGenResult."""
    return getattr(result, 'interpolation_dt', 0.02)


def _safe_zone_bounds(node):
    if not bool(getattr(node, "safe_zone_enabled", False)):
        return None
    lo = getattr(node, "safe_zone_min", None)
    hi = getattr(node, "safe_zone_max", None)
    if lo is None or hi is None or len(lo) < 3 or len(hi) < 3:
        return None
    return (
        [min(float(lo[i]), float(hi[i])) for i in range(3)],
        [max(float(lo[i]), float(hi[i])) for i in range(3)],
    )


def _point_inside_safe_zone_bounds(node, xyz, margin: float = 0.0) -> bool:
    bounds = _safe_zone_bounds(node)
    if bounds is None:
        return True
    lo, hi = bounds
    return all(lo[i] + margin <= float(xyz[i]) <= hi[i] - margin for i in range(3))


def _manual_cartesian_path_inside_safe_zone(node, states, label: str) -> bool:
    """Reject manual Cartesian trajectories whose TCP samples leave the saved safe-zone box."""
    if not states or _safe_zone_bounds(node) is None:
        return True
    margin = float(getattr(node.cfg.planner, "safe_zone_path_margin_m", 0.0))
    points = forward_kinematics_batch(node, states)
    if not points:
        node.get_logger().warn(
            f"{label}: safe-zone path check unavailable; rejecting manual Cartesian path.")
        return False
    for i, p in enumerate(points):
        xyz = [p.x, p.y, p.z]
        if not _point_inside_safe_zone_bounds(node, xyz, margin):
            node.get_logger().warn(
                f"{label}: rejected Cartesian path leaves safe-zone box at "
                f"waypoint {i}/{len(points) - 1}: "
                f"[{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]")
            return False
    return True


def _plan_cartesian_states_from_joints(node, start_joints, pose, label: str, plan_cfg=None):
    """Plan Cartesian motion from explicit start joints and return dense states + dt.

    ``plan_cfg`` overrides the planner config (default PLAN_CFG_DEFAULT); preflight callers
    pass a cheap config so failed candidates give up fast."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        start = JointState.from_position(
            torch.tensor([start_joints], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        goal = Pose.from_list(pose)
    except Exception as e:
        node.get_logger().warn(f"{label}: failed to create Cartesian plan inputs: {e}")
        return None

    lock = getattr(node, '_planning_lock', None)
    yolo = getattr(node, 'yolo_thread', None)
    yolo_lock = getattr(yolo, 'inference_lock', None)
    if yolo_lock:
        yolo_lock.acquire()
    try:
        if lock:
            lock.acquire()
        try:
            res = node.motion_gen.plan_single(start, goal, plan_cfg or PLAN_CFG_DEFAULT)
        except Exception as e:
            msg = str(e)
            if "CUDA error" in msg or "illegal memory access" in msg:
                node._cuda_faulted = True
            node.get_logger().warn(f"{label}: Cartesian plan exception: {e}")
            return None
        finally:
            if lock:
                lock.release()
    finally:
        if yolo_lock:
            yolo_lock.release()

    if res is None or not res.success:
        status = getattr(res, 'status', 'unknown') if res is not None else 'None'
        node.get_logger().warn(f"{label}: Cartesian plan failed. status={status}")
        return None
    states = interpolated_positions(res)
    if not states:
        node.get_logger().warn(f"{label}: Cartesian plan produced no states")
        return None
    return states, get_curobo_dt(res)


def _publish_cartesian_states(node, states, curobo_dt: float, target_pose,
                              label: str, motion_type: str, speed_factor: float) -> bool:
    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = (speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier
             * max(speed_factor, 1e-6))
    dt = curobo_dt / max(scale, 1e-6)
    dt = min(max(dt, curobo_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * planner.global_speed_multiplier,
        max_acc=planner.max_joint_acceleration,
        ramp_points=0,
    )
    if not traj.points:
        node.get_logger().warn(f"{label}: planned route produced an empty trajectory.")
        return False
    node.get_logger().warn(f"{label}: publishing combined Cartesian route ({len(states)} samples)")
    node.trajectory_pub.publish(traj)
    return wait_until_xyz(
        node,
        target_pose[:3],
        tol=0.008,
        timeout=15.0,
        target_quat=target_pose[3:] if len(target_pose) >= 7 else None,
    )


def publish_stop_trajectory(node):
    if node.current_joint_positions is None:
        return
    try:
        stop = JointTrajectory(); stop.joint_names = node.joint_order
        pt = JointTrajectoryPoint(); pt.positions = list(node.current_joint_positions)
        pt.velocities = [0.0]*len(node.joint_order); pt.accelerations = [0.0]*len(node.joint_order)
        pt.time_from_start.nanosec = 1_000_000; stop.points = [pt]
        node.trajectory_pub.publish(stop)
    except Exception:
        pass  # handle already destroyed during shutdown


# Executes a single pose in Cartesian space.
#input: pose: List of 7 elements [x,y,z,qw,qx,qy,qz]

#output: Publishes /joint_trajectory_controller/joint_trajectory.
def execute_single_pose(node, pose: list, motion_type: str = "default",
                        speed_factor: float = 1.0):
    if node.current_joint_positions is None:
        node.get_logger().warn("No joint state; cannot execute pose.")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = JointState.from_position(
        torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )
    goal = Pose.from_list(pose)
    lock = getattr(node, '_planning_lock', None)
    if lock: lock.acquire()
    try:
        res = node.motion_gen.plan_single(start, goal, PLAN_CFG_DEFAULT)
    finally:
        if lock: lock.release()
    if not res.success:
        node.get_logger().warn("Plan failed for single pose.")
        return False

    states = interpolated_positions(res)
    curobo_dt = get_curobo_dt(res)

    # Verify trajectory against latest depth data before execution
    if (VOXEL_CONFIG.get("verify_before_execute", True) and
        hasattr(node, 'voxel_obstacles') and node.voxel_obstacles is not None):

        # Extract goal position for exclusion zone (we WANT to reach the target)
        goal_position = pose[:3]  # [x, y, z] from input pose

        max_attempts = VOXEL_CONFIG.get("max_replan_attempts", 2)
        for attempt in range(max_attempts):
            is_safe, collision_idx = node.voxel_obstacles.verify_trajectory_collision(
                states,
                exclude_position=goal_position,  # Skip collision check near target
            )

            if is_safe:
                break

            node.get_logger().warn(
                f"Collision detected at waypoint {collision_idx}/{len(states)} "
                f"(attempt {attempt + 1}/{max_attempts})"
            )

            # Replan with updated obstacles
            if lock: lock.acquire()
            try:
                res = node.motion_gen.plan_single(start, goal, PLAN_CFG_DEFAULT)
            finally:
                if lock: lock.release()
            if not res.success:
                node.get_logger().error("Replan failed after collision detection")
                return False
            states = interpolated_positions(res)
        else:
            node.get_logger().error(f"Collision persists after {max_attempts} replans")
            return False

    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = (speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier
             * max(speed_factor, 1e-6))

    dt = curobo_dt / max(scale, 1e-6)
    dt = min(max(dt, curobo_dt), planner.max_dt)

    if motion_type == "manual" and not _manual_cartesian_path_inside_safe_zone(
            node, states, "MANUAL_CART"):
        return False

    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * planner.global_speed_multiplier,
        max_acc=planner.max_joint_acceleration,
        ramp_points=0,
    )
    if not traj.points:
        node.get_logger().warn("Single-pose plan produced an empty trajectory.")
        return False

    node.trajectory_pub.publish(traj)
    return wait_until_xyz(
        node,
        pose[:3],
        tol=0.008,
        timeout=15.0,
        target_quat=pose[3:] if len(pose) >= 7 else None,
    )


def execute_goal_via_local_staging(node, pose: list, label: str = "GOAL_STAGE",
                                   motion_type: str = "manual",
                                   speed_factor: float = 1.0) -> bool:
    """Recover a difficult goal by first moving to a standoff near that goal.

    This is deliberately goal-local: it tries to move above the target, then
    descend, rather than returning to a generic HOME/side-home posture.
    """
    if len(pose) < 7:
        return False
    cur = node.get_end_effector_pose()
    if cur is None or len(cur) < 7:
        return False
    current_joints = node.current_joint_positions
    if current_joints is None or len(current_joints) != len(node.joint_order):
        return False
    start_joints = list(current_joints)

    planner = node.cfg.planner
    offsets = list(getattr(planner, "goal_recovery_stage_offsets_m", [0.20, 0.12, 0.30]))
    max_attempts = max(
        1, int(getattr(planner, "goal_recovery_stage_max_attempts", 3)))
    quat_dot = abs(sum(a * b for a, b in zip(pose[3:7], cur[3:7])))
    orientations = [("FREEORI", list(cur[3:7]))]
    if quat_dot < 0.999:
        marker = ("MARKERORI", list(pose[3:7]))
        free = ("FREEORI", list(cur[3:7]))
        if bool(getattr(planner, "goal_recovery_stage_marker_first", True)):
            orientations = [marker, free]
        else:
            orientations = [free, marker]

    attempts = 0
    for dz in offsets:
        for ori_label, quat in orientations:
            if getattr(node, "stop_requested", False):
                return False
            stage_z = max(float(pose[2]) + float(dz), float(pose[2]))
            stage_pose = [float(pose[0]), float(pose[1]), stage_z] + quat
            final_pose = [float(pose[0]), float(pose[1]), float(pose[2])] + quat
            margin = float(getattr(planner, "safe_zone_path_margin_m", 0.0))
            if (
                not _point_inside_safe_zone_bounds(node, stage_pose[:3], margin)
                or not _point_inside_safe_zone_bounds(node, final_pose[:3], margin)
            ):
                node.get_logger().warn(
                    f"{label}: skipping local standoff z+{float(dz):.2f}m; "
                    f"stage/final would leave safe-zone box.")
                continue
            if attempts >= max_attempts:
                node.get_logger().warn(
                    f"{label}: local staging attempt limit reached "
                    f"({max_attempts}); skipping remaining standoffs.")
                return False
            attempts += 1
            node.get_logger().warn(
                f"{label}: trying local standoff {ori_label} "
                f"z+{float(dz):.2f}m -> "
                f"[{stage_pose[0]:.3f}, {stage_pose[1]:.3f}, {stage_pose[2]:.3f}]")

            stage_plan = _plan_cartesian_states_from_joints(
                node, start_joints, stage_pose, f"{label}_STANDOFF")
            if stage_plan is None:
                continue
            stage_states, stage_dt = stage_plan
            if motion_type == "manual" and not _manual_cartesian_path_inside_safe_zone(
                    node, stage_states, f"{label}_STANDOFF"):
                continue

            descent_plan = _plan_cartesian_states_from_joints(
                node, stage_states[-1], final_pose, f"{label}_DESCENT")
            if descent_plan is None:
                continue
            descent_states, descent_dt = descent_plan
            if motion_type == "manual" and not _manual_cartesian_path_inside_safe_zone(
                    node, descent_states, f"{label}_DESCENT"):
                continue

            node.get_logger().warn(
                f"{label}: validated standoff + descent; executing combined route.")
            combined = stage_states + descent_states[1:]
            if _publish_cartesian_states(
                    node,
                    combined,
                    max(stage_dt, descent_dt),
                    final_pose,
                    label,
                    motion_type,
                    speed_factor):
                return True
    return False


def estimate_nearest_ik_delta_deg(node, pose: list) -> float:
    """Return nearest IK max-joint delta in degrees, or inf if no quick IK solution."""
    if node.current_joint_positions is None or len(pose) < 7:
        return float("inf")
    cur_pose = node.get_end_effector_pose()
    if cur_pose is not None and len(cur_pose) >= 7:
        pos_err = math.dist(cur_pose[:3], pose[:3])
        quat_dot = abs(sum(a * b for a, b in zip(cur_pose[3:7], pose[3:7])))
        if pos_err < 0.01 and quat_dot > 0.999:
            return 0.0
    start_js = list(node.current_joint_positions)
    lock = getattr(node, '_planning_lock', None)
    if lock:
        lock.acquire()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        goal_pose = Pose(
            position=torch.tensor([pose[:3]], dtype=torch.float32, device=device),
            quaternion=torch.tensor([pose[3:7]], dtype=torch.float32, device=device),
        )
        seed = torch.tensor([start_js], dtype=torch.float32, device=device).unsqueeze(0)
        retract = torch.tensor([start_js], dtype=torch.float32, device=device)
        n = max(1, int(getattr(node.cfg.planner, "shortest_ik_seeds", 8)))
        solver_seeds = int(getattr(node.motion_gen.ik_solver, "num_seeds", n))
        ik_result = node.motion_gen.ik_solver.solve_single(
            goal_pose,
            seed_config=seed,
            retract_config=retract,
            return_seeds=min(n, solver_seeds),
        )
        sols = ik_result.js_solution.position
        succ = ik_result.success
        if sols.ndim == 3:
            sols = sols[0]
        if succ.ndim == 2:
            succ = succ[0]
        best = float("inf")
        for i in range(min(len(sols), len(succ))):
            if not bool(succ[i].item()):
                continue
            goal_js = nearest_joint_config(start_js, sols[i].detach().cpu().tolist())
            md = max(abs(g - c) for g, c in zip(goal_js, start_js))
            best = min(best, math.degrees(md))
        return best
    except Exception as e:
        msg = str(e)
        if "CUDA error" in msg or "illegal memory access" in msg:
            node._cuda_faulted = True
        try:
            node.get_logger().warn(f"IK reachability estimate failed: {e}")
        except Exception:
            pass
        return float("inf")
    finally:
        if lock:
            lock.release()


def execute_pose_shortest(node, pose: list, label: str = "MOVE",
                          motion_type: str = "default",
                          speed_factor: float = 1.0):
    """Move to a Cartesian pose via the IK solution NEAREST the current joints.

    A plain Cartesian plan (``execute_single_pose``) lets cuRobo pick whatever
    goal IK branch it likes, which can be a wrist-flip / elbow-over config far
    from the start — so the arm takes a long arc to reach a Cartesian-close
    goal. Here we solve several IK solutions biased toward the current joints,
    snap each to the nearest 2*pi branch, and run the collision-aware joint-space
    planner to the nearest ones (nearest first), so the executed motion is short.

    Outside the safe zone, the Cartesian fallback is intentionally disabled: a
    long unconstrained arc is unsafe for tight paths. With safe-zone walls active,
    the Cartesian planner remains a bounded last resort after short IK paths fail.
    """
    if node.current_joint_positions is None:
        for _ in range(20):
            time.sleep(0.05)
            if node.current_joint_positions is not None:
                break
    if node.current_joint_positions is None:
        node.get_logger().warn(f"{label}: no joint state; cannot move.")
        return False

    if getattr(node, "safe_zone_enabled", False):
        node.get_logger().info(
            f"{label}: safe zone active — trying nearest-IK short path first "
            f"(Cartesian planner is fallback).")

    start_js = list(node.current_joint_positions)
    candidates = []  # (max_joint_delta_from_start, goal_js) — collision-aware short moves
    if not getattr(node, "_cuda_faulted", False):
        lock = getattr(node, '_planning_lock', None)
        if lock:
            lock.acquire()
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            goal_pose = Pose(
                position=torch.tensor(
                    [pose[:3]], dtype=torch.float32, device=device),
                quaternion=torch.tensor(
                    [pose[3:7]], dtype=torch.float32, device=device),
            )
            seed = torch.tensor(
                [start_js], dtype=torch.float32, device=device).unsqueeze(0)
            retract = torch.tensor(
                [start_js], dtype=torch.float32, device=device)
            # Ask for several IK solutions: if the nearest branch is hard for trajopt,
            # we try the next-nearest before ever resorting to an unconstrained Cartesian
            # plan (which can pick a wrist-flip/elbow-over branch and swing the long way).
            _n = max(1, int(getattr(node.cfg.planner, "shortest_ik_seeds", 8)))
            _solver_seeds = int(getattr(node.motion_gen.ik_solver, "num_seeds", _n))
            ik_result = node.motion_gen.ik_solver.solve_single(
                goal_pose, seed_config=seed, retract_config=retract,
                return_seeds=min(_n, _solver_seeds))
            sols = ik_result.js_solution.position
            succ = ik_result.success
            if sols.ndim == 3:
                sols = sols[0]
            if succ.ndim == 2:
                succ = succ[0]
            for _i in range(min(len(sols), len(succ))):
                if not bool(succ[_i].item()):
                    continue
                # Snap to the same 2*pi branch as current so interpolation doesn't
                # travel a full revolution, then rank by how far the arm must move.
                _gj = nearest_joint_config(start_js, sols[_i].detach().cpu().tolist())
                _md = max(abs(g - c) for g, c in zip(_gj, start_js))
                candidates.append((_md, _gj))
            # Extra: an explicit single current-seeded solve. The multi-seed batch uses
            # mostly random seeds and can miss the near branch; seeding once from the
            # current config is the best shot at the genuinely-nearest solution.
            try:
                ik1 = node.motion_gen.ik_solver.solve_single(
                    goal_pose, seed_config=seed, retract_config=retract)
                _s1 = ik1.js_solution.position
                _ok1 = ik1.success
                if _s1.ndim == 3:
                    _s1 = _s1[0]
                if _ok1.ndim == 2:
                    _ok1 = _ok1[0]
                if len(_ok1) and bool(_ok1[0].item()):
                    _gj1 = nearest_joint_config(
                        start_js, _s1[0].detach().cpu().tolist())
                    _md1 = max(abs(g - c) for g, c in zip(_gj1, start_js))
                    candidates.append((_md1, _gj1))
                    node.get_logger().warn(
                        f"{label}: single current-seeded IK delta="
                        f"{round(math.degrees(_md1), 1)}deg")
            except Exception:
                pass
        except Exception as e:
            msg = str(e)
            if "CUDA error" in msg or "illegal memory access" in msg:
                node._cuda_faulted = True
            node.get_logger().warn(
                f"{label}: IK exception ({e}); cannot plan a short move.")
        finally:
            if lock:
                lock.release()

    # One-shot diagnostic: when the goal looks close but IK is far, show whether the goal
    # pose actually matches the current EE pose (frame check) and which joint is far.
    try:
        _cur_ee = node.get_end_effector_pose()
        _near = min(candidates, key=lambda t: t[0]) if candidates else None
        _pjd = ([round(math.degrees(g - c), 1) for g, c in zip(_near[1], start_js)]
                if _near else None)
        node.get_logger().warn(
            f"{label} DBG: goal_xyz={[round(p, 3) for p in pose[:3]]} "
            f"cur_ee_xyz={[round(v, 3) for v in (_cur_ee[:3] if _cur_ee else [])]} | "
            f"goal_quat={[round(p, 3) for p in pose[3:7]]} "
            f"cur_ee_quat={[round(v, 3) for v in (_cur_ee[3:7] if _cur_ee else [])]} | "
            f"n_ik={len(candidates)} "
            f"nearest_max_delta_deg={round(math.degrees(_near[0]), 1) if _near else None} "
            f"per_joint_delta_deg={_pjd}")
    except Exception as _e:
        node.get_logger().warn(f"{label} DBG failed: {_e}")

    if not candidates:
        node.get_logger().warn(
            f"{label}: no IK solution found — skipping move "
            f"(Cartesian fallback disabled: too risky for tight paths).")
        return False

    # De-duplicate near-identical IK solutions, then try the collision-aware joint plan
    # to each — nearest first — so the executed motion stays short. If every near branch
    # fails the move is skipped (no Cartesian fallback — see docstring).
    _seen = set()
    _ordered = []
    for _md, _gj in sorted(candidates, key=lambda t: t[0]):
        _key = tuple(round(v, 3) for v in _gj)
        if _key in _seen:
            continue
        _seen.add(_key)
        _ordered.append((_md, _gj))
    _planner = node.cfg.planner
    _max_plan_delta_deg = float(getattr(
        _planner, "shortest_ik_plan_max_delta_deg", 80.0))
    if _ordered and math.degrees(_ordered[0][0]) > _max_plan_delta_deg:
        node.get_logger().warn(
            f"{label}: nearest IK branch is "
            f"{math.degrees(_ordered[0][0]):.1f}deg away "
            f"(cap {_max_plan_delta_deg:.0f}deg); skipping expensive short-path "
            f"trajopt and using recovery/fallback.")
        return False
    _max_try = max(1, int(getattr(node.cfg.planner, "shortest_ik_max_tries", 3)))
    if (
        getattr(node, "safe_zone_enabled", False)
        and bool(getattr(_planner, "safe_zone_verified_interp_first", True))
        and _ordered
    ):
        _interp_cap = float(getattr(
            _planner, "safe_zone_verified_interp_max_delta_deg", 80.0))
        if math.degrees(_ordered[0][0]) <= _interp_cap:
            _speed_map = {
                "home": _planner.speed_home,
                "dropoff": _planner.speed_dropoff,
                "predropoff": _planner.speed_predropoff,
            }
            _scale = (
                _speed_map.get(motion_type, 1.0)
                * _planner.global_speed_multiplier
                * max(speed_factor, 1e-6)
            )
            _step_deg = float(getattr(
                _planner, "safe_zone_verified_interp_step_deg", 1.0))
            if _execute_verified_joint_interpolation(
                    node,
                    _ordered[0][1],
                    _scale,
                    label=f"{label}_SHORT_INTERP",
                    max_delta_deg=_interp_cap,
                    step_deg=_step_deg):
                return True

    for _rank, (_md, _gj) in enumerate(_ordered[:_max_try]):
        if getattr(node, 'stop_requested', False):
            return False
        _lbl = label if _rank == 0 else f"{label}_alt{_rank}"
        if plan_execute_js(node, _gj, label=_lbl, motion_type=motion_type,
                           speed_factor=speed_factor):
            return True

    # Every near IK branch was unreachable for trajopt (the optimizer didn't converge —
    # which doesn't mean a straight line is in collision). Rather than a long Cartesian arc
    # (disabled — unsafe) or skipping, do a guarded DIRECT interpolation straight to the
    # NEAREST IK config: the shortest possible move, capped in joint travel and forearm/
    # flange clearance-checked. (No world-obstacle avoidance — hence the small-move cap.)
    if getattr(node, 'stop_requested', False):
        return False
    _speed_map = {"home": _planner.speed_home, "dropoff": _planner.speed_dropoff,
                  "predropoff": _planner.speed_predropoff}
    _scale = (_speed_map.get(motion_type, 1.0) * _planner.global_speed_multiplier
              * max(speed_factor, 1e-6))
    _cap = float(getattr(_planner, "direct_interp_max_delta_deg", 25.0))
    node.get_logger().warn(
        f"{label}: {len(_ordered)} near IK branch(es) failed trajopt — trying guarded "
        f"direct interpolation to nearest IK config (cap {_cap:.0f}deg).")
    if _execute_home_direct_fallback(node, _ordered[0][1], _scale,
                                     label=f"{label}_DIRECT", max_delta_deg=_cap):
        return True
    # In-zone planned move: when the safe-zone walls are active, a Cartesian plan is bounded
    # by them (cuRobo can't route the arm out of the trusted box), so a longer planned path
    # is safe here even though it's globally disabled. Last resort before giving up.
    if getattr(node, "safe_zone_enabled", False):
        node.get_logger().warn(
            f"{label}: no short path — trying in-zone Cartesian plan "
            f"(safe-zone walls bound the arm).")
        if execute_single_pose(node, pose, motion_type=motion_type,
                               speed_factor=speed_factor):
            return True
    node.get_logger().warn(
        f"{label}: no short path and no in-zone plan — skipping move.")
    return False


# Plans and executes a joint-space motion to reach a specified set of joint angles.
# Input: target_joints: List of joint angles in radians.
        # label: A string label for logging purposes.
        # dt: Time step for trajectory interpolation.
        
# Publishes /joint_trajectory_controller/joint_trajectory.
def _wait_for_joint_target(node, target_joints, timeout: float = 15.0, tol: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and getattr(node, 'running', True):
        if getattr(node, 'stop_requested', False):
            return False
        current = node.current_joint_positions
        if current is not None:
            max_err = max(abs(c - t) for c, t in zip(current, target_joints))
            if max_err < tol:
                return True
        time.sleep(0.05)
    return False


def _execute_home_direct_fallback(node, target_joints, scale: float, *,
                                  label: str = "HOME_FALLBACK",
                                  max_delta_deg: float = None) -> bool:
    """Short, clearance-checked S-curve straight to ``target_joints`` when trajopt can't
    converge. Interpolates directly in joint space (no world-obstacle avoidance), so it is
    only allowed for SMALL moves — capped by ``max_delta_deg`` (default: HOME's config) —
    and gated by a forearm/flange clearance check. Keeps the motion short instead of a
    long planned arc. Returns False (caller decides) if over the cap or clearance fails."""
    current = node.current_joint_positions
    if current is None or len(current) != len(target_joints):
        return False

    planner = node.cfg.planner
    max_delta = max(abs(t - c) for c, t in zip(current, target_joints))
    _cap = (max_delta_deg if max_delta_deg is not None
            else getattr(planner, "home_direct_fallback_max_delta_deg", 20.0))
    limit = math.radians(float(_cap))
    if max_delta > limit:
        node.get_logger().warn(
            f"[{label}] rejected: max joint change "
            f"{math.degrees(max_delta):.1f}deg exceeds {math.degrees(limit):.1f}deg")
        return False

    steps = max(12, int(math.ceil(max_delta / math.radians(1.0))))
    states = []
    for i in range(steps + 1):
        alpha = i / steps
        smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
        states.append([c + smooth * (t - c) for c, t in zip(current, target_joints)])

    planning_lock = getattr(node, '_planning_lock', None)
    yolo = getattr(node, 'yolo_thread', None)
    yolo_lock = getattr(yolo, 'inference_lock', None)
    if yolo_lock:
        yolo_lock.acquire()
    try:
        if planning_lock:
            planning_lock.acquire()
        try:
            from .goals import _log_forearm_flange_clearance
            min_mm, _, destination_mm = _log_forearm_flange_clearance(
                node, states, label)
        finally:
            if planning_lock:
                planning_lock.release()
    except Exception as e:
        node.get_logger().error(f"[{label}] clearance check failed: {e}")
        return False
    finally:
        if yolo_lock:
            yolo_lock.release()

    required_mm = float(getattr(planner, "clamp_safety_threshold_mm", 35.0))
    if min_mm < required_mm or destination_mm < required_mm:
        node.get_logger().error(
            f"[{label}] rejected: clearance min={min_mm:.1f}mm "
            f"destination={destination_mm:.1f}mm required={required_mm:.1f}mm")
        return False

    dt = min(max(planner.base_dt / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * 0.5,
        max_acc=planner.max_joint_acceleration * 0.5,
        ramp_points=0,
        include_acc=False,
    )
    if not traj.points:
        return False

    node.get_logger().warn(
        f"[{label}] publishing guarded direct move "
        f"(max_delta={math.degrees(max_delta):.1f}deg, points={len(states)})")
    node.trajectory_pub.publish(traj)
    reached = _wait_for_joint_target(node, target_joints)
    blend_motion(node)
    if not reached:
        node.get_logger().warn(f"[{label}] trajectory did not reach target")
    return reached


def _execute_verified_joint_interpolation(node, target_joints, scale: float, *,
                                          label: str,
                                          max_delta_deg: float,
                                          step_deg: float = 1.0) -> bool:
    """Execute a short joint interpolation only after cuRobo constraint validation."""
    current = node.current_joint_positions
    if current is None or len(current) != len(target_joints):
        return False

    max_delta = max(abs(t - c) for c, t in zip(current, target_joints))
    if max_delta > math.radians(float(max_delta_deg)):
        node.get_logger().warn(
            f"[{label}] verified interpolation skipped: max joint change "
            f"{math.degrees(max_delta):.1f}deg exceeds {float(max_delta_deg):.1f}deg")
        return False

    steps = max(
        12,
        int(math.ceil(math.degrees(max_delta) / max(float(step_deg), 0.25))))
    states = []
    for i in range(steps + 1):
        alpha = i / steps
        smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
        states.append([c + smooth * (t - c) for c, t in zip(current, target_joints)])

    planning_lock = getattr(node, '_planning_lock', None)
    yolo = getattr(node, 'yolo_thread', None)
    yolo_lock = getattr(yolo, 'inference_lock', None)
    if yolo_lock:
        yolo_lock.acquire()
    try:
        if planning_lock:
            planning_lock.acquire()
        try:
            if not _curobo_path_constraints_valid(node, states, label):
                return False
            from .goals import _log_forearm_flange_clearance
            min_mm, _, destination_mm = _log_forearm_flange_clearance(
                node, states, label)
        finally:
            if planning_lock:
                planning_lock.release()
    except Exception as e:
        node.get_logger().warn(f"[{label}] verified interpolation check failed: {e}")
        return False
    finally:
        if yolo_lock:
            yolo_lock.release()

    planner = node.cfg.planner
    required_mm = float(getattr(planner, "clamp_safety_threshold_mm", 35.0))
    if min_mm < required_mm or destination_mm < required_mm:
        node.get_logger().warn(
            f"[{label}] verified interpolation rejected: clearance "
            f"min={min_mm:.1f}mm destination={destination_mm:.1f}mm "
            f"required={required_mm:.1f}mm")
        return False

    dt = min(max(planner.base_dt / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * 0.5,
        max_acc=planner.max_joint_acceleration * 0.5,
        ramp_points=0,
        include_acc=False,
    )
    if not traj.points:
        return False

    node.get_logger().warn(
        f"[{label}] publishing verified short joint interpolation "
        f"(max_delta={math.degrees(max_delta):.1f}deg, samples={len(states)})")
    node.trajectory_pub.publish(traj)
    reached = _wait_for_joint_target(node, target_joints)
    blend_motion(node)
    if not reached:
        node.get_logger().warn(f"[{label}] trajectory did not reach target")
    return reached


def validate_joint_interpolations_batch(node, start_joints, target_joints_list,
                                        *, max_delta_deg: float,
                                        step_deg: float = 1.0):
    """Return per-target booleans for short direct joint paths against cuRobo constraints."""
    if not target_joints_list:
        return []
    start = list(start_joints)
    cap = math.radians(float(max_delta_deg))
    step = max(float(step_deg), 0.25)
    all_states = []
    slices = []
    for target in target_joints_list:
        target = nearest_joint_config(start, list(target))
        max_delta = max(abs(t - c) for c, t in zip(start, target))
        if max_delta > cap:
            slices.append(None)
            continue
        steps = max(12, int(math.ceil(math.degrees(max_delta) / step)))
        begin = len(all_states)
        for i in range(steps + 1):
            alpha = i / steps
            smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
            all_states.append([c + smooth * (t - c) for c, t in zip(start, target)])
        slices.append((begin, len(all_states)))

    if not all_states:
        return [False for _ in target_joints_list]

    planning_lock = getattr(node, '_planning_lock', None)
    yolo = getattr(node, 'yolo_thread', None)
    yolo_lock = getattr(yolo, 'inference_lock', None)
    if yolo_lock:
        yolo_lock.acquire()
    try:
        if planning_lock:
            planning_lock.acquire()
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            js = JointState.from_position(
                torch.tensor(all_states, dtype=torch.float32, device=device),
                joint_names=node.joint_order,
            )
            metrics = node.motion_gen.check_constraints(js)
        finally:
            if planning_lock:
                planning_lock.release()
    except Exception as e:
        node.get_logger().warn(f"Reachability cloud validation failed: {e}")
        return [False for _ in target_joints_list]
    finally:
        if yolo_lock:
            yolo_lock.release()

    feasible = getattr(metrics, "feasible", None)
    if feasible is not None:
        try:
            mask = torch.as_tensor(feasible).detach().reshape(-1).bool().cpu()
        except Exception:
            mask = None
    else:
        mask = None
    if mask is None:
        constraint = getattr(metrics, "constraint", None)
        if constraint is not None:
            try:
                mask = (torch.as_tensor(constraint).detach().reshape(-1) <= 0.0).cpu()
            except Exception:
                mask = None
    if mask is None:
        return [False for _ in target_joints_list]

    result = []
    for item in slices:
        if item is None:
            result.append(False)
            continue
        begin, end = item
        result.append(bool(mask[begin:end].all().item()))
    return result


def execute_known_joint_goal(node, target_joints, *, label: str = "GOAL_KNOWN",
                             motion_type: str = "manual",
                             speed_factor: float = 1.0) -> bool:
    """Execute a known reachable joint sample without re-solving IK from its pose."""
    if node.current_joint_positions is None:
        return False
    planner = node.cfg.planner
    target = nearest_joint_config(node.current_joint_positions, list(target_joints))
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = (
        speed_map.get(motion_type, 1.0)
        * planner.global_speed_multiplier
        * max(speed_factor, 1e-6)
    )
    cap_deg = float(getattr(planner, "safe_zone_verified_interp_max_delta_deg", 80.0))
    step_deg = float(getattr(planner, "safe_zone_verified_interp_step_deg", 1.0))
    return _execute_verified_joint_interpolation(
        node,
        target,
        scale,
        label=label,
        max_delta_deg=cap_deg,
        step_deg=step_deg,
    )


def _constraint_is_feasible(metrics) -> bool:
    feasible = getattr(metrics, "feasible", None)
    if feasible is not None:
        try:
            return bool(torch.as_tensor(feasible).detach().bool().all().item())
        except Exception:
            return bool(feasible)
    constraint = getattr(metrics, "constraint", None)
    if constraint is not None:
        try:
            return bool((torch.as_tensor(constraint).detach() <= 0.0).all().item())
        except Exception:
            return False
    return False


def _first_infeasible_index(metrics):
    feasible = getattr(metrics, "feasible", None)
    if feasible is None:
        return None
    try:
        mask = torch.as_tensor(feasible).detach().reshape(-1).bool()
        bad = (~mask).nonzero(as_tuple=False)
        if bad.numel():
            return int(bad[0].item())
    except Exception:
        return None
    return None


def _curobo_path_constraints_valid(node, states, label: str, *, log_failure: bool = True) -> bool:
    """Validate a dense joint path against cuRobo's active joint/world/self constraints."""
    if not states:
        return False
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        js = JointState.from_position(
            torch.tensor(states, dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        metrics = node.motion_gen.check_constraints(js)
    except Exception as e:
        node.get_logger().warn(f"[{label}] cuRobo path validation failed: {e}")
        return False

    if _constraint_is_feasible(metrics):
        return True

    bad_idx = _first_infeasible_index(metrics)
    extra = ""
    constraint = getattr(metrics, "constraint", None)
    if bad_idx is not None and constraint is not None:
        try:
            flat = torch.as_tensor(constraint).detach().reshape(-1)
            extra = f" constraint={float(flat[bad_idx].item()):.4g}"
        except Exception:
            pass
    if log_failure:
        node.get_logger().warn(
            f"[{label}] rejected: cuRobo constraints fail"
            f"{'' if bad_idx is None else f' at sample {bad_idx}/{len(states) - 1}'}"
            f"{extra}")
    return False


def _execute_home_verified_interpolation(node, target_joints, scale: float) -> bool:
    """HOME-only recovery for valid endpoints when JS trajopt refuses the move.

    Unlike _execute_home_direct_fallback, this is not blind: every dense waypoint is
    checked against cuRobo's active constraints before publishing.
    """
    current = node.current_joint_positions
    if current is None or len(current) != len(target_joints):
        return False

    planner = node.cfg.planner
    max_delta = max(abs(t - c) for c, t in zip(current, target_joints))
    cap_deg = float(getattr(planner, "home_verified_interp_max_delta_deg", 185.0))
    if max_delta > math.radians(cap_deg):
        node.get_logger().warn(
            f"[HOME_VERIFIED_INTERP] rejected: max joint change "
            f"{math.degrees(max_delta):.1f}deg exceeds {cap_deg:.1f}deg")
        return False

    step_deg = max(0.25, float(getattr(planner, "home_verified_interp_step_deg", 1.0)))
    steps = max(12, int(math.ceil(math.degrees(max_delta) / step_deg)))
    states = []
    for i in range(steps + 1):
        alpha = i / steps
        smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
        states.append([c + smooth * (t - c) for c, t in zip(current, target_joints)])

    planning_lock = getattr(node, '_planning_lock', None)
    yolo = getattr(node, 'yolo_thread', None)
    yolo_lock = getattr(yolo, 'inference_lock', None)
    if yolo_lock:
        yolo_lock.acquire()
    try:
        if planning_lock:
            planning_lock.acquire()
        try:
            if not _curobo_path_constraints_valid(node, states, "HOME_VERIFIED_INTERP"):
                return False
            from .goals import _log_forearm_flange_clearance
            min_mm, _, destination_mm = _log_forearm_flange_clearance(
                node, states, "HOME_VERIFIED_INTERP")
        finally:
            if planning_lock:
                planning_lock.release()
    except Exception as e:
        node.get_logger().error(f"[HOME_VERIFIED_INTERP] clearance check failed: {e}")
        return False
    finally:
        if yolo_lock:
            yolo_lock.release()

    required_mm = float(getattr(planner, "clamp_safety_threshold_mm", 35.0))
    if min_mm < required_mm or destination_mm < required_mm:
        node.get_logger().error(
            f"[HOME_VERIFIED_INTERP] rejected: clearance min={min_mm:.1f}mm "
            f"destination={destination_mm:.1f}mm required={required_mm:.1f}mm")
        return False

    dt = min(max(planner.base_dt / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * 0.5,
        max_acc=planner.max_joint_acceleration * 0.5,
        ramp_points=0,
        include_acc=False,
    )
    if not traj.points:
        return False

    node.get_logger().warn(
        f"[HOME_VERIFIED_INTERP] publishing cuRobo-verified HOME interpolation "
        f"(max_delta={math.degrees(max_delta):.1f}deg, samples={len(states)})")
    node.trajectory_pub.publish(traj)
    reached = _wait_for_joint_target(node, target_joints)
    blend_motion(node)
    if not reached:
        node.get_logger().warn("[HOME_VERIFIED_INTERP] trajectory did not reach HOME")
    return reached


def _densify_joint_segment(start, goal, step_deg: float, *, include_start: bool):
    max_delta = max(abs(g - s) for s, g in zip(start, goal))
    steps = max(1, int(math.ceil(math.degrees(max_delta) / max(step_deg, 0.25))))
    states = [list(start)] if include_start else []
    for i in range(1, steps + 1):
        alpha = i / steps
        smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
        states.append([s + smooth * (g - s) for s, g in zip(start, goal)])
    return states


def _densify_joint_waypoints(waypoints, step_deg: float):
    states = []
    for i, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        states.extend(_densify_joint_segment(a, b, step_deg, include_start=(i == 0)))
    return states


def _route_states_from_joint_order(current, target, order, step_deg: float):
    waypoints = [list(current)]
    q = list(current)
    for idx in order:
        q = list(q)
        q[idx] = target[idx]
        if max(abs(a - b) for a, b in zip(q, waypoints[-1])) > 1e-4:
            waypoints.append(q)
    if max(abs(a - b) for a, b in zip(waypoints[-1], target)) > 1e-4:
        waypoints.append(list(target))
    return _densify_joint_waypoints(waypoints, step_deg)


def _home_preset_route_candidates(node, current, target, step_deg: float):
    names = (
        "home_left_joints",
        "home_right_joints",
        "home_left_low_joints",
        "home_right_low_joints",
        "predropoff_joints",
        "dropoff_joints",
    )
    for name in names:
        preset = getattr(node, name, None)
        if preset is None:
            continue
        mid_from_current = nearest_joint_config(current, list(preset))
        final_from_mid = nearest_joint_config(mid_from_current, target)
        yield f"preset:{name.removesuffix('_joints')}", _densify_joint_waypoints(
            [current, mid_from_current, final_from_mid], step_deg)


def _joint_order_candidates(limit: int):
    preferred = [
        (5, 4, 3, 2, 1, 0),  # wrist unwind before moving the big links
        (3, 4, 5, 2, 1, 0),
        (4, 5, 3, 2, 1, 0),
        (5, 3, 4, 0, 1, 2),
        (0, 1, 2, 3, 4, 5),
        (1, 2, 0, 3, 4, 5),
        (2, 1, 0, 3, 4, 5),
        (0, 2, 1, 5, 4, 3),
    ]
    seen = set()
    for order in preferred:
        if order in seen:
            continue
        seen.add(order)
        yield order
    if len(seen) >= limit:
        return
    for order in itertools.permutations(range(6)):
        if order in seen:
            continue
        seen.add(order)
        yield order
        if len(seen) >= limit:
            return


def _execute_home_verified_route(node, target_joints, scale: float) -> bool:
    """Find and execute a multi-leg HOME route whose dense samples validate in cuRobo."""
    current = node.current_joint_positions
    if current is None or len(current) != len(target_joints):
        return False

    planner = node.cfg.planner
    step_deg = max(0.25, float(getattr(planner, "home_verified_interp_step_deg", 1.0)))
    max_candidates = max(1, int(getattr(
        planner, "home_verified_route_max_candidates", 120)))

    def candidates():
        yield from _home_preset_route_candidates(node, list(current), list(target_joints), step_deg)
        remaining = max(0, max_candidates - 6)
        for order in _joint_order_candidates(remaining):
            yield "joint_order:" + ",".join(str(i) for i in order), _route_states_from_joint_order(
                current, target_joints, order, step_deg)

    planning_lock = getattr(node, '_planning_lock', None)
    yolo = getattr(node, 'yolo_thread', None)
    yolo_lock = getattr(yolo, 'inference_lock', None)
    if yolo_lock:
        yolo_lock.acquire()
    try:
        if planning_lock:
            planning_lock.acquire()
        try:
            for label, states in candidates():
                if getattr(node, "stop_requested", False):
                    return False
                if not states:
                    continue
                if not _curobo_path_constraints_valid(
                        node, states, f"HOME_ROUTE {label}", log_failure=False):
                    continue
                from .goals import _log_forearm_flange_clearance
                min_mm, _, destination_mm = _log_forearm_flange_clearance(
                    node, states, f"HOME_ROUTE {label}")
                required_mm = float(getattr(planner, "clamp_safety_threshold_mm", 35.0))
                if min_mm < required_mm or destination_mm < required_mm:
                    node.get_logger().warn(
                        f"[HOME_ROUTE {label}] rejected: clearance min={min_mm:.1f}mm "
                        f"destination={destination_mm:.1f}mm required={required_mm:.1f}mm")
                    continue
                route_label = label
                route_states = states
                break
            else:
                node.get_logger().warn(
                    f"[HOME_ROUTE] no validated route found in {max_candidates} candidates")
                return False
        finally:
            if planning_lock:
                planning_lock.release()
    except Exception as e:
        node.get_logger().error(f"[HOME_ROUTE] validation failed: {e}")
        return False
    finally:
        if yolo_lock:
            yolo_lock.release()

    dt = min(max(planner.base_dt / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order,
        route_states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * 0.5,
        max_acc=planner.max_joint_acceleration * 0.5,
        ramp_points=0,
        include_acc=False,
    )
    if not traj.points:
        return False

    node.get_logger().warn(
        f"[HOME_ROUTE] publishing validated route via {route_label} "
        f"({len(route_states)} samples)")
    node.trajectory_pub.publish(traj)
    reached = _wait_for_joint_target(node, route_states[-1])
    blend_motion(node)
    if not reached:
        node.get_logger().warn("[HOME_ROUTE] trajectory did not reach HOME")
    return reached


def _execute_home_cartesian_pose_recovery(node, target_joints, scale: float) -> bool:
    """Use Cartesian planning to escape a JS branch, then finish to exact HOME joints."""
    current = node.current_joint_positions
    if current is None or len(current) != len(target_joints):
        return False

    home_pose = pose_from_joints(node, target_joints)
    if home_pose is None:
        node.get_logger().warn("[HOME_CART] could not compute HOME FK pose")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        start = JointState.from_position(
            torch.tensor([current], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        goal = Pose.from_list(home_pose)
    except Exception as e:
        node.get_logger().warn(f"[HOME_CART] failed to create planning inputs: {e}")
        return False

    lock = getattr(node, '_planning_lock', None)
    yolo = getattr(node, 'yolo_thread', None)
    yolo_lock = getattr(yolo, 'inference_lock', None)
    if yolo_lock:
        yolo_lock.acquire()
    try:
        if lock:
            lock.acquire()
        try:
            res = node.motion_gen.plan_single(start, goal, PLAN_CFG_DEFAULT)
        except Exception as e:
            msg = str(e)
            if "CUDA error" in msg or "illegal memory access" in msg:
                node._cuda_faulted = True
            node.get_logger().warn(f"[HOME_CART] Cartesian recovery exception: {e}")
            return False
        finally:
            if lock:
                lock.release()
    finally:
        if yolo_lock:
            yolo_lock.release()

    if res is None or not res.success:
        status = getattr(res, 'status', 'unknown') if res is not None else 'None'
        node.get_logger().warn(f"[HOME_CART] Cartesian recovery failed. status={status}")
        return False

    states = interpolated_positions(res)
    if not states:
        return False

    try:
        from .goals import _log_forearm_flange_clearance
        min_mm, _, destination_mm = _log_forearm_flange_clearance(
            node, states, "HOME_CART")
        required_mm = float(getattr(
            node.cfg.planner, "clamp_safety_threshold_mm", 35.0))
        if min_mm < required_mm or destination_mm < required_mm:
            node.get_logger().warn(
                f"[HOME_CART] rejected: clearance min={min_mm:.1f}mm "
                f"destination={destination_mm:.1f}mm required={required_mm:.1f}mm")
            return False
    except Exception as e:
        node.get_logger().warn(f"[HOME_CART] clearance check failed: {e}")
        return False

    planner = node.cfg.planner
    dt = min(max(get_curobo_dt(res) / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * 0.5,
        max_acc=planner.max_joint_acceleration * 0.5,
        ramp_points=0,
        include_acc=False,
    )
    if not traj.points:
        return False

    node.get_logger().warn(
        f"[HOME_CART] publishing Cartesian recovery to HOME pose "
        f"({len(states)} samples)")
    node.trajectory_pub.publish(traj)
    if not wait_until_xyz(
            node,
            home_pose[:3],
            tol=0.012,
            timeout=20.0,
            target_quat=home_pose[3:7]):
        node.get_logger().warn("[HOME_CART] did not reach HOME pose")
        return False

    # Finish with a normal, non-recursive JS plan to the exact HOME preset.
    time.sleep(0.2)
    finish_target = list(target_joints)
    if node.current_joint_positions is not None:
        finish_target = nearest_joint_config(node.current_joint_positions, finish_target)
    if _wait_for_joint_target(node, finish_target, timeout=0.1):
        return True
    if node.current_joint_positions is not None:
        finish_delta = max(
            abs(t - c) for c, t in zip(node.current_joint_positions, finish_target))
        finish_cap_deg = float(getattr(
            planner, "home_cart_finish_max_delta_deg",
            getattr(planner, "home_direct_fallback_max_delta_deg", 40.0)))
        if finish_delta > math.radians(finish_cap_deg):
            node.get_logger().warn(
                f"[HOME_CART] reached HOME TCP pose; exact HOME joint branch is "
                f"{math.degrees(finish_delta):.1f}deg away; rejecting TCP-only HOME.")
            return False
    if plan_execute_js(
            node,
            finish_target,
            label="HOME_CART_FINISH",
            motion_type="home"):
        return True
    if _execute_home_verified_interpolation(node, finish_target, scale):
        return True
    if _execute_home_verified_route(node, finish_target, scale):
        return True
    if _execute_home_direct_fallback(
        node,
        finish_target,
        scale,
        label="HOME_CART_FINISH_FALLBACK",
        max_delta_deg=getattr(planner, "home_direct_fallback_max_delta_deg", 40.0),
    ):
        return True
    node.get_logger().warn(
        "[HOME_CART] reached HOME TCP pose but exact HOME joint branch is unreachable; "
        "rejecting TCP-only HOME.")
    return False


def _extract_graph_states(node, res, target_joints, densify_deg: float = 1.5):
    """Pull the collision-free GRAPH path out of a failed (TRAJOPT_FAIL) result and densify it.

    The graph planner validates each edge as collision-free, so a straight line between two
    adjacent graph nodes is collision-free too — we can safely interpolate between them. Trajopt
    only failed to *smooth* the path, not to find it. Returns a dense list of joint states ending
    at ``target_joints``, or None if the result carries no usable graph path. Reads GPU tensors
    to host here (cheap) so the caller can reset the planner before executing."""
    gp = getattr(res, "graph_plan", None) if res is not None else None
    if gp is None:
        return None
    try:
        import numpy as np
        pos = np.asarray(gp.position.detach().cpu().numpy(), dtype=float)
    except Exception:
        return None
    if pos.ndim == 3:
        pos = pos[0]
    if pos.ndim != 2 or pos.shape[0] < 2:
        return None
    # Reorder columns to node.joint_order if the graph reports its own joint names.
    names = list(getattr(gp, "joint_names", None) or node.joint_order)
    if names != list(node.joint_order) and set(names) == set(node.joint_order):
        idx = [names.index(j) for j in node.joint_order]
        pos = pos[:, idx]
    waypoints = pos.tolist()
    # The path must actually reach HOME, else it's not a usable recovery.
    end_err = math.degrees(max(abs(a - b) for a, b in zip(waypoints[-1], target_joints)))
    if end_err > 5.0:
        node.get_logger().warn(
            f"Graph path ends {end_err:.1f}deg from HOME — not using it.")
        return None
    step = math.radians(densify_deg)
    states = [list(waypoints[0])]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        seg = max(abs(y - x) for x, y in zip(a, b))
        n = max(1, int(math.ceil(seg / step)))
        for k in range(1, n + 1):
            alpha = k / float(n)
            states.append([x + alpha * (y - x) for x, y in zip(a, b)])
    # Snap the final state exactly onto HOME.
    states.append(list(target_joints))
    return states


def _execute_graph_states(node, states, scale: float, label: str) -> bool:
    """Time-parameterize and execute a (collision-free) graph path that trajopt couldn't smooth.
    Recovery move — capped velocity/accel, not time-optimal."""
    planner = node.cfg.planner
    node.get_logger().warn(
        f"[{label}] trajopt failed but graph found a collision-free path — "
        f"executing it directly ({len(states)} pts).")
    dt = min(max(planner.base_dt / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * 0.5,
        max_acc=planner.max_joint_acceleration * 0.5,
        ramp_points=0,
        include_acc=False,
    )
    if not traj.points:
        return False
    node.trajectory_pub.publish(traj)
    reached = _wait_for_joint_target(node, states[-1])
    blend_motion(node)
    if not reached:
        node.get_logger().warn(f"[{label}] graph path did not reach target")
    return reached


def _execute_home_staged(node, target_joints, chunk_deg: float = 45.0) -> bool:
    """Recover a large HOME reconfiguration that single-shot trajopt can't solve.

    A 170+deg joint move has a straight-line trajopt seed that sweeps through obstacles, so
    the optimizer can't make it collision-free (TRAJOPT_FAIL). Split it into collision-checked
    ``plan_single_js`` sub-segments small enough for trajopt to converge, and chain them. Each
    segment plans from the robot's *actual* current state, so this is robust. Avoids the graph
    planner (corrupts plan_single_js buffers here) and the unsafe blind interpolation.
    Returns True only if every segment plans and executes."""
    current = node.current_joint_positions
    if current is None or len(current) != len(target_joints):
        return False
    max_delta_deg = math.degrees(max(abs(t - c) for c, t in zip(current, target_joints)))
    n = max(2, int(math.ceil(max_delta_deg / max(chunk_deg, 1.0))))
    node.get_logger().info(
        f"HOME: staging {max_delta_deg:.0f}deg move in {n} collision-checked segments.")
    origin = list(current)
    for i in range(1, n + 1):
        if getattr(node, "stop_requested", False):
            return False
        alpha = i / float(n)
        waypoint = [c + alpha * (t - c) for c, t in zip(origin, target_joints)]
        # label != "HOME" so a failed segment returns False instead of recursing into recovery.
        if not plan_execute_js(node, waypoint, label=f"HOME_STAGE{i}/{n}",
                               motion_type="home"):
            node.get_logger().warn(
                f"HOME staged: segment {i}/{n} failed — aborting staged home.")
            return False
    node.get_logger().info("HOME: staged move complete.")
    return True


def plan_execute_js(
    node,
    target_joints: List[float],
    label: str,
    motion_type: str = "default",
    speed_factor: float = 1.0,
):
    if node.current_joint_positions is None:
        # Wait briefly for joint state callback to fire (can be delayed after blocking ops)
        for _ in range(10):
            time.sleep(0.1)
            if node.current_joint_positions is not None:
                break
    if node.current_joint_positions is None:
        node.get_logger().warn(f"No joint state; skipping {label} move.")
        return False

    # Attempt CUDA recovery if previously faulted
    if getattr(node, "_cuda_faulted", False):
        from .goals import try_cuda_recovery
        try_cuda_recovery(node)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 1. Build joint states
    # ------------------------------
    try:
        start = JointState.from_position(
            torch.tensor([node.current_joint_positions], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )

        goal_js = JointState.from_position(
            torch.tensor([target_joints], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
    except Exception as e:
        msg = str(e)
        if "CUDA error" in msg or "illegal memory access" in msg:
            node._cuda_faulted = True
        node.get_logger().warn(f"Failed to create tensors for {label}: {e}")
        return False

    # ------------------------------
    # 2. Speed scaling (applies global multiplier)
    # ------------------------------
    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier
    scale *= max(speed_factor, 1e-6)

    if label == "HOME":
        home_tol = math.radians(float(
            getattr(planner, "home_reached_tolerance_deg", 3.0)))
        max_err = max(
            abs(c - t) for c, t in zip(node.current_joint_positions, target_joints))
        if max_err < home_tol:
            node.get_logger().info(
                f"HOME: already at target (max error={math.degrees(max_err):.1f}deg)")
            return True

    # ------------------------------
    # 3. cuRobo plan (hold lock to prevent concurrent CUDA ops)
    # ------------------------------
    lock = getattr(node, '_planning_lock', None)
    res = None
    graph_states = None
    _planning_t0 = time.perf_counter()
    _record_motion = getattr(node, "_record_motion_plan_event", None)
    home_graph_first = (
        label == "HOME" and bool(getattr(node, "safe_zone_enabled", False))
    )
    _yolo = getattr(node, 'yolo_thread', None)
    _yolo_lock = getattr(_yolo, 'inference_lock', None)
    if _yolo_lock: _yolo_lock.acquire()
    try:
        if home_graph_first:
            node.get_logger().info(
                "HOME: safe-zone walls active — using graph planner first")
        else:
            if lock: lock.acquire()
            try:
                res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_JS)
            except Exception as e:
                msg = str(e)
                if "CUDA error" in msg or "illegal memory access" in msg:
                    node._cuda_faulted = True
                node.get_logger().warn(f"Joint-space plan to {label} exception: {e}")
                return False
            finally:
                if lock: lock.release()

        if (not home_graph_first) and (res is None or not res.success):
            status = getattr(res, 'status', 'unknown') if res is not None else 'None'
            # The no-finetune pass can recover both a failed smoothing pass and
            # some plain trajopt failures without enabling graph-buffer resizing.
            if res is not None and (
                "FINETUNE" in str(status) or "TRAJOPT" in str(status)
            ):
                node.get_logger().info(
                    f"Joint-space plan to {label}: {status}, retrying without finetune")
                if lock: lock.acquire()
                try:
                    res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_JS_NO_FINETUNE)
                except Exception as e:
                    node.get_logger().warn(f"Joint-space retry to {label} exception: {e}")
                    return False
                finally:
                    if lock: lock.release()
        # Graph-planner recovery for HOME: when trajopt can't seed a large/contorted
        # reconfiguration, the graph planner finds a collision-free global seed. It resizes
        # internal buffers that corrupt later plan_single_js calls, so reset right after.
        # If trajopt then fails to SMOOTH that seed (TRAJOPT_FAIL while used_graph), we
        # salvage the raw collision-free graph path and execute it directly (see below).
        if (res is None or not res.success) and label == "HOME":
            status = getattr(res, 'status', 'unknown') if res is not None else 'None'
            if home_graph_first:
                node.get_logger().info(
                    "Joint-space plan to HOME: running graph planner")
            else:
                node.get_logger().info(
                    f"Joint-space plan to {label}: {status}, retrying with graph planner")
            if lock: lock.acquire()
            try:
                # One-shot start-state diagnostic: is the current pose itself the problem?
                cur_deg = [round(math.degrees(v), 1)
                           for v in node.current_joint_positions]
                try:
                    valid, sstatus = node.motion_gen.check_start_state(start)
                    node.get_logger().warn(
                        f"HOME start-state check: valid={valid} status={sstatus} "
                        f"| current_joints_deg={cur_deg}")
                except Exception as e:
                    node.get_logger().warn(
                        f"HOME check_start_state failed: {e} | current_joints_deg={cur_deg}")
                try:
                    gvalid, gstatus = node.motion_gen.check_start_state(goal_js)
                    node.get_logger().warn(
                        f"HOME GOAL-state check: valid={gvalid} status={gstatus}")
                except Exception as e:
                    node.get_logger().warn(f"HOME goal-state check failed: {e}")
                try:
                    wm = node.motion_gen.world_model
                    names = [c.name for c in (wm.cuboid or [])]
                    nvox = 0
                    try:
                        for vg in (wm.voxel or []):
                            nvox += 1
                    except Exception:
                        pass
                    node.get_logger().warn(
                        f"HOME world: {len(names)} cuboids={names} | voxel_grids={nvox}")
                except Exception as e:
                    node.get_logger().warn(f"HOME world inspect failed: {e}")
                res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_JS_GRAPH)
                if res is None or not res.success:
                    graph_states = _extract_graph_states(node, res, target_joints)
            except Exception as e:
                node.get_logger().warn(f"Graph retry to {label} exception: {e}")
                res = None
            finally:
                try:
                    node.motion_gen.reset(reset_seed=True)
                except Exception as e:
                    node.get_logger().warn(f"motion_gen.reset after graph failed: {e}")
                if lock: lock.release()
        if res is None or not res.success:
            status = getattr(res, 'status', 'unknown') if res is not None else 'None'
            valid = getattr(res, 'valid_query', None) if res is not None else None
            # Diagnostic: flag joints outside cuRobo's usable range (URDF ±2π minus
            # position_limit_clip in ur10e.yml). goal OOB ⇒ the saved posture has a
            # joint too near ±360° for cuRobo; valid_query=False ⇒ start/goal invalid
            # (limits or collision); True ⇒ endpoints OK but no path found.
            try:
                _JLIM = 6.283185307 - 0.1
                names = getattr(node, "joint_order", None) or list(range(len(target_joints)))
                _sj = list(node.current_joint_positions) if node.current_joint_positions is not None else []
                start_oob = [f"{names[i]}={_sj[i]:.3f}" for i in range(len(_sj)) if abs(_sj[i]) > _JLIM]
                goal_oob = [f"{names[i]}={target_joints[i]:.3f}"
                            for i in range(len(target_joints)) if abs(target_joints[i]) > _JLIM]
                node.get_logger().warn(
                    f"[{label}] joint-limit check (usable ±{_JLIM:.3f} rad) — "
                    f"start OOB: {start_oob or 'none'}; goal OOB: {goal_oob or 'none'}")
            except Exception:
                pass
            node.get_logger().warn(
                f"Joint-space plan to {label} failed. status={status} valid_query={valid} "
                "(valid_query=False ⇒ start/goal out of limits or in collision; "
                "True ⇒ endpoints OK but no collision-free path found)")
    finally:
        if _yolo_lock: _yolo_lock.release()

    if res is None or not res.success:
        if callable(_record_motion):
            _record_motion(
                stage="PLAN", label=label, motion_type=motion_type,
                planner="curobo.plan_single_js", result="FAILED",
                status=str(getattr(res, 'status', 'None')),
                valid_query=str(getattr(res, 'valid_query', None)),
                planning_ms=round(
                    (time.perf_counter() - _planning_t0) * 1000.0, 1))
        if label == "HOME":
            # The graph path is collision-free even when trajopt couldn't smooth it.
            if graph_states is not None and _execute_graph_states(
                    node, graph_states, scale, label):
                return True
            # Big reconfigurations (170+deg) defeat single-shot trajopt; stage them in
            # collision-checked chunks before resorting to the blind short-move fallback.
            if _execute_home_staged(node, target_joints):
                return True
            if _execute_home_verified_interpolation(node, target_joints, scale):
                return True
            if _execute_home_verified_route(node, target_joints, scale):
                return True
            if bool(getattr(planner, "home_cartesian_pose_recovery", False)):
                if _execute_home_cartesian_pose_recovery(node, target_joints, scale):
                    return True
            else:
                node.get_logger().warn(
                    "[HOME_CART] skipped: TCP-only HOME recovery is disabled; "
                    "exact HOME joint branch is required.")
            return _execute_home_direct_fallback(node, target_joints, scale)
        return False

    # ------------------------------
    # 4. Interpolate (older cuRobo API)
    # ------------------------------
    # No args allowed
    states = interpolated_positions(res)
    _raw_start_error = (
        max(abs(c - s) for c, s in zip(node.current_joint_positions, states[0]))
        if states else 0.0)
    states = _prepend_measured_start_bridge(
        states, list(node.current_joint_positions), label)
    if _raw_start_error > math.radians(0.25):
        node.get_logger().info(
            f"{label}: inserted measured-start bridge "
            f"({_raw_start_error*57.3:.1f}deg → ≤1.0deg steps)")
    curobo_dt = get_curobo_dt(res)

    if callable(_record_motion):
        _record_motion(
            stage="PLAN", label=label, motion_type=motion_type,
            planner="curobo.plan_single_js", result="SUCCESS",
            status=str(getattr(res, 'status', 'SUCCESS')),
            valid_query=str(getattr(res, 'valid_query', None)),
            planning_ms=round(
                (time.perf_counter() - _planning_t0) * 1000.0, 1),
            raw_trajectory_samples=len(states),
            measured_start_bridge_deg=round(math.degrees(_raw_start_error), 2))

    dt = curobo_dt / max(scale, 1e-6)
    dt = min(max(dt, planner.min_dt), planner.max_dt)

    traj = build_trajectory(
        node.joint_order,
        states,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=math.pi,  # UR10e physical joint limit; prevents clipping cuRobo's natural velocities
    )

    if getattr(node.cfg.planner, "log_phase_timings", False):
        node.get_logger().info(f"Moving to {label} (dt={dt:.3f})")
    if traj.points:
        node.trajectory_pub.publish(traj)
        if callable(_record_motion):
            _record_motion(
                stage="EXECUTE", label=label, motion_type=motion_type,
                planner="curobo.plan_single_js", result="PUBLISHED",
                trajectory_samples=len(states), dt_s=round(float(dt), 5),
                trajectory_duration_s=round(
                    float(dt) * max(len(states) - 1, 0), 3),
                speed_scale=round(float(scale), 3))
    else:
        node.get_logger().warn(f"Skipping publish for {label} — empty trajectory (stop requested?)")
        return False

    # ------------------------------
    # 8. Wait for the robot, then blend to avoid abrupt stop
    # ------------------------------
    target_joints = states[-1]
    fk = forward_kinematics(node, target_joints)
    if fk:
        wait_until_xyz(node, [fk.x, fk.y, fk.z])
    # Secondary joint-space check: if EE arrived instantly (different IK branch),
    # wait until joints actually converge to the target configuration.
    reached = _wait_for_joint_target(node, target_joints)
    blend_motion(node)
    if not reached:
        node.get_logger().warn(f"Joint-space move to {label} did not reach target")
    if callable(_record_motion):
        _record_motion(
            stage="ENDPOINT", label=label, motion_type=motion_type,
            result="REACHED" if reached else "TIMEOUT")
    return reached



def nearest_joint_config(current: List[float], target: List[float]) -> List[float]:
    """Return the joint angles equivalent to `target` that are closest to `current`.

    For each joint, picks the value in {target[i] + k*2π | k ∈ ℤ} that minimises
    |current[i] - target[i]|. This prevents cuRobo from routing through a 360°
    detour when the stored config and the current config have drifted by ≈2π.
    """
    out = []
    for c, t in zip(current, target):
        diff = c - t
        k = round(diff / (2 * math.pi))
        out.append(t + k * 2 * math.pi)
    return out


def _clear_voxels(node):
    """Clear voxel obstacle world so depth-camera noise doesn't block return paths."""
    vo = getattr(node, 'voxel_obstacles', None)
    if vo is not None:
        try:
            vo.clear()
        except Exception:
            pass


def _exit_safe_zone(node, why):
    """Keep safe-zone walls active.

    Older recovery code temporarily cleared the safe-zone walls for HOME/dropoff/posture
    moves. That allowed valid TCP paths whose links could still leave the trusted box.
    The safety rule is now simple: programmatic moves never clear the walls; if a target
    requires leaving the box, the plan should fail and the operator can adjust the box or
    explicitly disable it from the UI.
    """
    if not getattr(node, "safe_zone_enabled", False):
        return
    node.get_logger().info(
        f"Safe zone remains ENABLED for {why}; walls are never cleared automatically.")


def _restore_safe_zone(node, why):
    """Re-enable safe-zone walls after a recovery move that temporarily cleared them."""
    cb = getattr(node, "_safe_zone_on_change", None)
    node.safe_zone_enabled = True
    if cb is not None:
        try:
            cb()
        except Exception as e:
            node.get_logger().warn(f"Safe-zone restore failed after {why}: {e}")
            return False
    node.get_logger().info(f"Safe zone RESTORED after {why}.")
    return True


def move_to_nearest_good_posture(node, *, label: str = "GOAL_RECOVERY",
                                 goal_pose: list = None) -> bool:
    """Move to a stable posture before retrying a failed goal.

    If a goal pose is supplied, prefer the good posture whose TCP is closest to
    that goal; otherwise fall back to closest-in-joints from the current state.
    """
    current = node.current_joint_positions
    if current is None or len(current) != len(node.joint_order):
        node.get_logger().warn(f"{label}: no joint state; cannot recover posture.")
        return False

    presets = [
        ("HOME", getattr(node, "home_joints", None)),
        ("HOME_LEFT", getattr(node, "home_left_joints", None)),
        ("HOME_RIGHT", getattr(node, "home_right_joints", None)),
        ("HOME_LEFT_LOW", getattr(node, "home_left_low_joints", None)),
        ("HOME_RIGHT_LOW", getattr(node, "home_right_low_joints", None)),
    ]
    candidates = []
    for name, joints in presets:
        if joints is None:
            continue
        target = nearest_joint_config(list(current), list(joints))
        max_delta = max(abs(t - c) for c, t in zip(current, target))
        l2_delta = math.sqrt(sum((t - c) ** 2 for c, t in zip(current, target)))
        target_dist = float("inf")
        if goal_pose is not None and len(goal_pose) >= 3:
            fk_pose = pose_from_joints(node, target)
            if fk_pose is not None:
                target_dist = math.dist(fk_pose[:3], goal_pose[:3])
        candidates.append((target_dist, max_delta, l2_delta, name, target))

    if not candidates:
        node.get_logger().warn(f"{label}: no good posture presets configured.")
        return False

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    restore_safe_zone = bool(getattr(node, "safe_zone_enabled", False))
    _exit_safe_zone(node, label)
    for target_dist, max_delta, _, name, target in candidates:
        if getattr(node, "stop_requested", False):
            if restore_safe_zone:
                _restore_safe_zone(node, label)
            return False
        dist_txt = (
            f", target_dist={target_dist:.2f}m"
            if math.isfinite(target_dist) else "")
        node.get_logger().warn(
            f"{label}: trying nearest good posture {name} "
            f"(max_delta={math.degrees(max_delta):.1f}deg{dist_txt})")
        if name == "HOME":
            if move_to_home_position(node):
                if restore_safe_zone:
                    _restore_safe_zone(node, label)
                return True
        elif plan_execute_js(
                node,
                target,
                label=f"{label}_{name}",
                motion_type="home",
                speed_factor=0.7):
            if restore_safe_zone:
                _restore_safe_zone(node, label)
            return True
    node.get_logger().warn(f"{label}: no good posture could be reached.")
    if restore_safe_zone:
        _restore_safe_zone(node, label)
    return False


def move_to_home_position(node):
    _clear_voxels(node)
    _exit_safe_zone(node, "HOME")
    target = node.home_joints
    if node.current_joint_positions is not None:
        target = nearest_joint_config(node.current_joint_positions, target)
    return plan_execute_js(node, target, label="HOME", motion_type="home")


def move_to_dropoff_position(node):
    _clear_voxels(node)
    _exit_safe_zone(node, "DROP-OFF")
    target = node.dropoff_joints
    if node.current_joint_positions is not None:
        target = nearest_joint_config(node.current_joint_positions, target)
    return plan_execute_js(node, target, label="DROP-OFF", motion_type="dropoff")


def _prepend_measured_start_bridge(states, measured_start, label,
                                   max_step_deg: float = 1.0):
    """Ensure a planned joint trajectory begins at the measured robot posture.

    Some cuRobo interpolation results begin after the mathematical start state.
    Publishing that first sample directly can make the UR controller see a
    several-degree discontinuity and reject the trajectory.  Insert a short,
    linear bridge with at most ``max_step_deg`` per joint; it follows the first
    local segment of the already collision-checked joint-space plan.
    """
    states = [list(q) for q in states]
    if not states or measured_start is None:
        return states
    start_error = max(
        abs(float(c) - float(s))
        for c, s in zip(measured_start, states[0]))
    if start_error <= math.radians(0.25):
        states[0] = list(measured_start)
        return states

    steps = max(2, int(math.ceil(
        start_error / math.radians(max_step_deg))))
    first = states[0]
    bridge = []
    for i in range(steps):
        alpha = i / float(steps)
        bridge.append([
            float(c) + alpha * (float(s) - float(c))
            for c, s in zip(measured_start, first)
        ])
    return bridge + states


def preplan_js(node, target_joints: List[float], start_joints: List[float],
               label: str = "PREPLAN", motion_type: str = "default"):
    """Plan a joint-space trajectory without executing it.
    Returns the built JointTrajectory message, or None on failure."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = JointState.from_position(
        torch.tensor([start_joints], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )
    goal_js = JointState.from_position(
        torch.tensor([target_joints], dtype=torch.float32, device=device),
        joint_names=node.joint_order,
    )
    lock = getattr(node, '_planning_lock', None)
    _yolo = getattr(node, 'yolo_thread', None)
    _yolo_lock = getattr(_yolo, 'inference_lock', None)
    if _yolo_lock: _yolo_lock.acquire()
    try:
        if lock: lock.acquire()
        try:
            res = node.motion_gen.plan_single_js(start, goal_js, PLAN_CFG_JS)
        finally:
            if lock: lock.release()
    finally:
        if _yolo_lock: _yolo_lock.release()
    if not res.success:
        # ── Diagnostic: why did cuRobo refuse this joint-space plan? ──────────
        # status      : cuRobo's failure reason (IK/collision/trajopt/limits)
        # valid_query : False ⇒ the start OR goal itself is invalid (out of joint
        #               limits or in collision at the endpoint); True ⇒ endpoints
        #               are fine but no collision-free/limit-respecting PATH found.
        _status = getattr(res, "status", None)
        _valid = getattr(res, "valid_query", None)
        try:
            # URDF joint limit is ±2π; ur10e.yml clips 0.1 rad off each end
            # (position_limit_clip), so cuRobo's usable range is ±(2π − 0.1).
            _JLIM = 6.283185307 - 0.1
            names = getattr(node, "joint_order", None) or list(range(len(target_joints)))

            def _oob(q):
                return [f"{names[i]}={q[i]:.3f}" for i in range(len(q)) if abs(q[i]) > _JLIM]

            start_oob, goal_oob = _oob(start_joints), _oob(target_joints)
            node.get_logger().warn(
                f"[{label}] joint-limit check (usable ±{_JLIM:.3f} rad) — "
                f"start OOB: {start_oob or 'none'}; goal OOB: {goal_oob or 'none'}")
        except Exception:
            pass
        node.get_logger().warn(
            f"[{label}] cuRobo plan_single_js FAILED — status={_status} valid_query={_valid} "
            "(valid_query=False ⇒ goal/start out of limits or in collision; "
            "True ⇒ endpoints OK but no collision-free path found)")
        return None
    states = interpolated_positions(res)
    _raw_start_error = (
        max(abs(c - s) for c, s in zip(start_joints, states[0]))
        if states else 0.0)
    states = _prepend_measured_start_bridge(states, start_joints, label)
    if _raw_start_error > math.radians(0.25):
        node.get_logger().info(
            f"[PREPLAN] {label}: inserted measured-start bridge "
            f"({_raw_start_error*57.3:.1f}deg → ≤1.0deg steps)")
    curobo_dt = get_curobo_dt(res)
    planner = node.cfg.planner
    speed_map = {
        "home": planner.speed_home,
        "dropoff": planner.speed_dropoff,
        "predropoff": planner.speed_predropoff,
    }
    scale = speed_map.get(motion_type, 1.0) * planner.global_speed_multiplier
    dt = curobo_dt / max(scale, 1e-6)
    dt = min(max(dt, planner.min_dt), planner.max_dt)
    traj = build_trajectory(
        node.joint_order, states, dt=dt,
        stop_flag=lambda: node.stop_requested,
    )
    return traj, states


def execute_preplan(node, traj, states, label: str) -> bool:
    """Publish a pre-planned JointTrajectory and wait for completion.
    Drop-in replacement for the publish+wait portion of plan_execute_js."""
    if not traj or not traj.points:
        node.get_logger().warn(f"[PREPLAN] {label}: empty trajectory, skipping")
        return False
    current = node.current_joint_positions
    if current is None or not states:
        node.get_logger().warn(f"[PREPLAN] {label}: missing current/start joint state")
        return False
    start_error = max(abs(c - s) for c, s in zip(current, states[0]))
    if start_error > 0.10:
        node.get_logger().warn(
            f"[PREPLAN] {label}: stale start state "
            f"({math.degrees(start_error):.1f}deg mismatch); replanning required")
        return False
    if not getattr(node.cfg.planner, "concise_console_logs", False):
        node.get_logger().info(
            f"[PREPLAN] {label}: publishing pre-planned trajectory "
            f"({len(traj.points)} pts)")
    node.trajectory_pub.publish(traj)

    target_joints = states[-1]
    fk = forward_kinematics(node, target_joints)
    if fk:
        wait_until_xyz(node, [fk.x, fk.y, fk.z])

    _joint_tol = 0.05
    _deadline = time.time() + 15.0
    converged = False
    while time.time() < _deadline and getattr(node, 'running', True):
        if getattr(node, 'stop_requested', False):
            break
        cur = node.current_joint_positions
        if cur is not None:
            if max(abs(c - t) for c, t in zip(cur, target_joints)) < _joint_tol:
                converged = True
                break
        time.sleep(0.05)
    blend_motion(node)
    if not converged:
        node.get_logger().warn(f"[PREPLAN] {label}: trajectory did not reach target")
    return converged


def move_to_predropoff_position(node):
    _exit_safe_zone(node, "preDROP-OFF")
    target = node.predropoff_joints
    if node.current_joint_positions is not None:
        target = nearest_joint_config(node.current_joint_positions, target)
    plan_execute_js(node, target, label="preDROP-OFF", motion_type="predropoff")


def rotate_wrist(node, degrees: float,
                 rotate_time: float = 1.2,
                 hold_time: float = 0.1,
                 return_time: float = 1.2):

    if node.current_joint_positions is None:
        node.get_logger().error("No joint state; cannot rotate wrist.")
        return

    rad = math.radians(degrees)

    start = node.current_joint_positions.copy()
    peak = start.copy()
    peak[5] += rad  # wrist_3_joint

    traj = JointTrajectory()
    traj.joint_names = node.joint_order

    def pt(q, t):
        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start.sec = int(t)
        p.time_from_start.nanosec = int((t - int(t)) * 1e9)
        return p

    t1 = rotate_time
    t2 = t1 + hold_time
    t3 = t2 + return_time

    traj.points = [
        pt(start, 0.0),    # start
        pt(peak, t1),      # rotate
        pt(peak, t2),      # short hold
        pt(start, t3)      # return
    ]

    node.trajectory_pub.publish(traj)




def move_backward(node, delta: float):
    if node.current_joint_positions is None:
        node.get_logger().error("No joint state; cannot move.")
        return
    
    q = node.current_joint_positions.copy(); q[1] -= delta
    traj = JointTrajectory(); traj.joint_names = node.joint_order
    p = JointTrajectoryPoint(); p.positions = q; p.time_from_start.sec = 1
    traj.points.append(p); node.trajectory_pub.publish(traj)

def blend_motion(node, pause=0.1):
    # maintains smoothness, avoids jerk
    if node.current_joint_positions is None:
        time.sleep(pause)
        return

    planner = node.cfg.planner
    traj = build_trajectory(
        node.joint_order,
        [node.current_joint_positions],
        dt=0.02,
        stop_flag=lambda: node.stop_requested,
    )
    if traj.points:
        node.trajectory_pub.publish(traj)
    time.sleep(pause)
