# ruff: noqa
import time, math, re, torch
import threading
import copy
import numpy as np
from geometry_msgs.msg import Pose as ROSPose, PoseStamped, PointStamped
from curobo.types.math import Pose
from curobo.types.robot import JointState


def try_cuda_recovery(node) -> bool:
    """Attempt to recover from a CUDA fault by resetting the device state.
    Returns True if CUDA is usable again, False if still broken."""
    if not getattr(node, "_cuda_faulted", False):
        return True
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Test with a small tensor operation
        t = torch.tensor([1.0], device="cuda")
        _ = t + t
        del t
        torch.cuda.empty_cache()
        node._cuda_faulted = False
        node.get_logger().info("CUDA recovery successful — GPU planning re-enabled")
        return True
    except Exception as e:
        node.get_logger().warn(f"CUDA recovery failed: {e}")
        return False


class ThreadSafeGoalList:
    """Thread-safe wrapper for goal_poses list to prevent race conditions between ROS callbacks and main thread."""
    def __init__(self):
        self._lock = threading.Lock()
        self._goals = []

    def append(self, goal):
        with self._lock:
            self._goals.append(goal)

    def insert(self, index, goal):
        with self._lock:
            self._goals.insert(index, goal)

    def pop(self, index=0):
        with self._lock:
            if self._goals:
                return self._goals.pop(index)
            return None

    def clear(self):
        with self._lock:
            self._goals.clear()

    def __len__(self):
        with self._lock:
            return len(self._goals)

    def __bool__(self):
        with self._lock:
            return bool(self._goals)

    def __iter__(self):
        with self._lock:
            return iter(list(self._goals))

    def any_within_distance(self, pos, threshold):
        """Check if any goal is within threshold distance of pos[:3]."""
        with self._lock:
            return any(
                isinstance(e, (list, tuple))
                and len(e) >= 3
                and math.dist(pos[:3], e[:3]) < threshold
                for e in self._goals
            )

    def peek(self, index=0):
        """Return goal at index without removing it, or None if out of range."""
        with self._lock:
            if index < len(self._goals):
                return copy.deepcopy(self._goals[index])
            return None

    def snapshot(self):
        """Return a copy of the current goals, preserving non-pose queue items."""
        with self._lock:
            return copy.deepcopy(self._goals)

    def sort(self, key=None, reverse=False):
        """Sort goals in place with optional key function."""
        with self._lock:
            self._goals.sort(key=key, reverse=reverse)
from .motions import execute_single_pose as _exec
from .motions import publish_stop_trajectory
from .config import (
    LOW_Z_THRESH, LATERAL_THRESH, PLAN_CFG_DEFAULT, PLAN_CFG_SCAN_PREFLIGHT,
    X_FORWARD_Y_LATERAL,
)
from .utils import build_trajectory, wait_until_xyz, ease_out_tail, ease_in_head
from .markers import publish_goal_marker, publish_planned_path, clear_path_markers
from .motions import interpolated_positions, get_curobo_dt
from .motions import move_to_dropoff_position, move_to_home_position
from .motions import blend_motion, preplan_js, execute_preplan, plan_execute_js, nearest_joint_config
from .utils import compute_visibility_approach
from .fk import forward_kinematics, forward_kinematics_batch, pose_from_joints
from .grasp_learner import GraspRecord
from . import gripper as gripper_mod
from . import markers as markers_mod

def pose_to_vec7(p: ROSPose):
    return [p.position.x, p.position.y, p.position.z, p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z]


def log_path_deviation(node, label: str):
    """Compare actual EE position to the planned path stored by plan_and_send.
    Logs: distance to planned endpoint + max deviation from nearest planned waypoint."""
    if not getattr(node.cfg.planner, "log_phase_timings", False):
        return
    planned = getattr(node, '_planned_cartesian_path', None)
    if not planned:
        return
    actual_pose = node.get_end_effector_pose()
    if not actual_pose:
        return
    ax, ay, az = actual_pose[0], actual_pose[1], actual_pose[2]

    # Distance to planned final point
    fp = planned[-1]
    end_dist = math.sqrt((ax - fp.x)**2 + (ay - fp.y)**2 + (az - fp.z)**2)

    # Distance to nearest planned waypoint
    min_dist = float('inf')
    min_idx = 0
    for i, p in enumerate(planned):
        d = math.sqrt((ax - p.x)**2 + (ay - p.y)**2 + (az - p.z)**2)
        if d < min_dist:
            min_dist = d
            min_idx = i

    node.get_logger().info(
        f"[PATH_DEV] {label}: end_err={end_dist*100:.1f}cm | "
        f"nearest_wp={min_idx}/{len(planned)} dist={min_dist*100:.1f}cm"
    )


def notify_grasp_attempt(node, position_xyz):
    """
    Publish grasp attempt notification to vision system for fruit tracking.
    This increments the attempt count for the fruit at the given position.
    """
    if not hasattr(node, "_grasp_attempt_pub"):
        node._grasp_attempt_pub = node.create_publisher(PointStamped, "/fruit_grasp_attempt", 10)

    msg = PointStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.point.x = float(position_xyz[0])
    msg.point.y = float(position_xyz[1])
    msg.point.z = float(position_xyz[2])
    node._grasp_attempt_pub.publish(msg)


def lock_target(node, position_xyz):
    """
    Lock vision system onto a specific target position.
    Vision will track this target instead of switching to a "better" one.
    Call this when starting approach to a date.
    """
    if not hasattr(node, "_target_lock_pub"):
        node._target_lock_pub = node.create_publisher(PointStamped, "/target_lock", 10)

    msg = PointStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.point.x = float(position_xyz[0])
    msg.point.y = float(position_xyz[1])
    msg.point.z = float(position_xyz[2])
    node._target_lock_pub.publish(msg)
    if getattr(node.cfg.planner, "log_cycle_start", False):
        node.get_logger().info(f"Target lock sent: [{position_xyz[0]:.3f}, {position_xyz[1]:.3f}, {position_xyz[2]:.3f}]")


def _set_goal_rejection(node, reason: str, detail: str = ""):
    setter = getattr(node, "set_goal_rejection", None)
    if callable(setter):
        setter(reason, detail)


def unlock_target(node):
    """
    Release target lock, allowing vision to select the best target again.
    Call this after grasp attempt completes (success or final failure).
    """
    if not hasattr(node, "_target_lock_pub"):
        node._target_lock_pub = node.create_publisher(PointStamped, "/target_lock", 10)

    msg = PointStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "base_link"
    msg.point.x = msg.point.y = msg.point.z = 0.0  # Zero = unlock signal
    node._target_lock_pub.publish(msg)
    if getattr(node.cfg.planner, "log_cycle_start", False):
        node.get_logger().info("Target lock released")


def quat_multiply(a, b):
    """Multiply two quaternions [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ]


def quat_normalize(q):
    n = math.sqrt(sum(x * x for x in q))
    return [x / n for x in q] if n > 1e-9 else [1.0, 0.0, 0.0, 0.0]


def quat_apply_local_x_pitch(q, pitch_deg):
    half = math.radians(float(pitch_deg)) / 2.0
    q_pitch = [math.cos(half), math.sin(half), 0.0, 0.0]
    return quat_normalize(quat_multiply(list(q), q_pitch))


def quat_apply_local_z_roll(q, roll_deg):
    """Rotate the finger triangle around the tool's local insertion (+Z) axis."""
    half = math.radians(float(roll_deg)) / 2.0
    q_roll = [math.cos(half), 0.0, 0.0, math.sin(half)]
    return quat_normalize(quat_multiply(list(q), q_roll))


def quat_rotate_vec(q, v):
    qv = [0.0, v[0], v[1], v[2]]
    qi = [q[0], -q[1], -q[2], -q[3]]
    return quat_multiply(quat_multiply(q, qv), qi)[1:]


def closure_center_corrected_tcp(desired_grasp_xyz, orientation_wxyz,
                                 closure_offset_tcp):
    """Return TCP XYZ that places the physical closure centre at the grasp point."""
    if len(closure_offset_tcp) != 3:
        raise ValueError("closure_center_offset_tcp_m must contain exactly 3 values")
    offset_base = quat_rotate_vec(
        quat_normalize(list(orientation_wxyz)), list(closure_offset_tcp))
    tcp_xyz = [
        float(desired_grasp_xyz[i]) - float(offset_base[i]) for i in range(3)
    ]
    return tcp_xyz, offset_base


def yaw_delta_for_local_axis(current_quat, desired_dir, local_axis=(0.0, 0.0, 1.0)):
    """Return world-yaw delta so the selected local axis faces desired_dir in XY."""
    if current_quat is None or desired_dir is None:
        return None

    desired = np.array([desired_dir[0], desired_dir[1], 0.0], dtype=float)
    desired_norm = np.linalg.norm(desired)
    if desired_norm < 1e-9:
        return None
    desired /= desired_norm

    current_axis = np.array(quat_rotate_vec(current_quat, local_axis), dtype=float)
    current_xy = np.array([current_axis[0], current_axis[1], 0.0], dtype=float)
    current_norm = np.linalg.norm(current_xy)
    if current_norm < 1e-9:
        return None
    current_xy /= current_norm

    cross_z = current_xy[0] * desired[1] - current_xy[1] * desired[0]
    dot = float(np.clip(np.dot(current_xy, desired), -1.0, 1.0))
    return math.atan2(cross_z, dot)


def yaw_only_align_local_axis(current_quat, desired_dir, local_axis=(0.0, 0.0, 1.0)):
    """Keep pitch/roll and yaw only so the selected local axis faces desired_dir in XY."""
    delta = yaw_delta_for_local_axis(current_quat, desired_dir, local_axis=local_axis)
    if delta is None:
        return list(current_quat) if current_quat is not None else current_quat
    q_yaw = [math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0)]
    return quat_normalize(quat_multiply(q_yaw, list(current_quat)))


def bounded_yaw_align_local_axis(
        current_quat, desired_dir, *, local_axis=(0.0, 0.0, 1.0),
        max_delta_deg=35.0):
    """Yaw toward desired_dir without changing tool pitch/roll."""
    raw_delta = yaw_delta_for_local_axis(
        current_quat, desired_dir, local_axis=local_axis)
    if raw_delta is None:
        return list(current_quat), 0.0, 0.0
    limit = math.radians(abs(float(max_delta_deg)))
    applied_delta = max(-limit, min(limit, raw_delta))
    q_yaw = [
        math.cos(applied_delta / 2.0), 0.0, 0.0,
        math.sin(applied_delta / 2.0),
    ]
    aligned = quat_normalize(quat_multiply(q_yaw, list(current_quat)))
    remaining = yaw_delta_for_local_axis(
        aligned, desired_dir, local_axis=local_axis)
    return aligned, raw_delta, (remaining if remaining is not None else 0.0)


def align_local_axis_to_vector(current_quat, desired_dir, local_axis=(0.0, 0.0, 1.0),
                               max_angle_deg=25.0):
    """Minimal 3D swing so local_axis points at desired_dir; capped to avoid branch flips."""
    if current_quat is None or desired_dir is None:
        return current_quat, 0.0

    desired = np.array(desired_dir, dtype=float)
    desired_norm = np.linalg.norm(desired)
    if desired_norm < 1e-9:
        return list(current_quat), 0.0
    desired /= desired_norm

    current_axis = np.array(quat_rotate_vec(current_quat, local_axis), dtype=float)
    current_norm = np.linalg.norm(current_axis)
    if current_norm < 1e-9:
        return list(current_quat), 0.0
    current_axis /= current_norm

    axis = np.cross(current_axis, desired)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(current_axis, desired), -1.0, 1.0))
    if axis_norm < 1e-9:
        return list(current_quat), 0.0

    angle = math.atan2(axis_norm, dot)
    max_angle = math.radians(max_angle_deg)
    angle = max(-max_angle, min(max_angle, angle))
    axis /= axis_norm
    half = angle / 2.0
    q_swing = [math.cos(half), *(math.sin(half) * axis).tolist()]
    return quat_normalize(quat_multiply(q_swing, list(current_quat))), angle


def bounded_target_approach_direction(nominal_dir, vision_outward_dir,
                                      max_delta_deg=25.0):
    """Return a bounded inward direction from collision-filtered vision data.

    Vision publishes an outward fruit/surface direction. The insertion direction
    is its opposite. The caller supplies the required vertical policy (level for
    MID/HIGH or +12 degrees for LOW); only the horizontal heading comes from
    vision so noisy surface tilt cannot create an unsafe vertical approach.
    """
    nominal = np.asarray(nominal_dir, dtype=float)
    nominal /= max(float(np.linalg.norm(nominal)), 1e-9)
    if vision_outward_dir is None:
        return nominal, 0.0, 0.0, False
    outward = np.asarray(vision_outward_dir, dtype=float)
    proposed_xy = -outward[:2]
    xy_norm = float(np.linalg.norm(proposed_xy))
    if xy_norm < 0.15:
        return nominal, 0.0, 0.0, False
    nominal_xy_mag = math.hypot(float(nominal[0]), float(nominal[1]))
    proposed = np.array([
        proposed_xy[0] / xy_norm * nominal_xy_mag,
        proposed_xy[1] / xy_norm * nominal_xy_mag,
        nominal[2],
    ], dtype=float)
    proposed /= max(float(np.linalg.norm(proposed)), 1e-9)
    dot = float(np.clip(np.dot(nominal, proposed), -1.0, 1.0))
    requested = math.acos(dot)
    limit = math.radians(abs(float(max_delta_deg)))
    applied = min(requested, limit)
    if requested < 1e-8:
        return nominal, 0.0, 0.0, True
    # Normalized interpolation is sufficient here because corrections are <=25°.
    ratio = applied / requested
    selected = (1.0 - ratio) * nominal + ratio * proposed
    selected /= max(float(np.linalg.norm(selected)), 1e-9)
    return selected, requested, applied, True


def select_mid_high_corridor_direction(node, fruit_outward_dir,
                                       class_label="MID_HIGH"):
    """Choose a fixed horizontal corridor from selected-date direction only."""
    planner = node.cfg.planner
    nominal = np.array(
        [1.0, 0.0, 0.0] if X_FORWARD_Y_LATERAL
        else [0.0, -1.0, 0.0], dtype=float)
    if not bool(getattr(planner, "mid_high_corridor_approach_enabled", True)):
        return nominal, "STRAIGHT"
    angle_deg = abs(float(getattr(
        planner, "mid_high_corridor_side_angle_deg", 20.0)))
    inner_angle_deg = abs(float(getattr(
        planner, "mid_high_corridor_inner_angle_deg", 10.0)))
    wide_angle_deg = abs(float(getattr(
        planner, "mid_high_corridor_wide_angle_deg", 30.0)))
    outer_angle_deg = abs(float(getattr(
        planner, "mid_high_corridor_outer_angle_deg", 45.0)))
    # World-Z rotation changes only the horizontal approach heading.
    def rotated(delta):
        c, s = math.cos(delta), math.sin(delta)
        return np.array([
            c * nominal[0] - s * nominal[1],
            s * nominal[0] + c * nominal[1], 0.0])
    candidates = [
        (f"FROM_RIGHT_{outer_angle_deg:g}", rotated(-math.radians(outer_angle_deg))),
        (f"FROM_RIGHT_{wide_angle_deg:g}", rotated(-math.radians(wide_angle_deg))),
        (f"FROM_RIGHT_{angle_deg:g}", rotated(-math.radians(angle_deg))),
        (f"FROM_RIGHT_{inner_angle_deg:g}", rotated(-math.radians(inner_angle_deg))),
        ("STRAIGHT", nominal.copy()),
        (f"FROM_LEFT_{inner_angle_deg:g}", rotated(math.radians(inner_angle_deg))),
        (f"FROM_LEFT_{angle_deg:g}", rotated(math.radians(angle_deg))),
        (f"FROM_LEFT_{wide_angle_deg:g}", rotated(math.radians(wide_angle_deg))),
        (f"FROM_LEFT_{outer_angle_deg:g}", rotated(math.radians(outer_angle_deg))),
    ]
    axis_enabled = bool(getattr(
        planner, "corridor_date_axis_enabled", True))
    axis_stable = bool(getattr(node, "fruit_major_axis_stable", False))
    axis_confidence = float(getattr(
        node, "fruit_major_axis_confidence", 0.0))
    axis_min_confidence = float(getattr(
        planner, "corridor_date_axis_min_confidence", 0.20))
    axis_angle_deg = math.degrees(float(getattr(
        node, "fruit_major_axis_angle", 0.0)))
    axis_valid = (
        axis_enabled and axis_stable
        and axis_confidence >= axis_min_confidence
        and math.isfinite(axis_angle_deg))
    if axis_valid:
        # Image major-axis deviation is zero for a vertical date. Positive
        # tilt selects an origin on the physical right; negative selects left.
        # Clamp to the widest tested corridor and change translation heading
        # only; this is not a tool-roll command.
        bounded_axis_deg = float(np.clip(
            axis_angle_deg, -outer_angle_deg, outer_angle_deg))
        inward = rotated(-math.radians(bounded_axis_deg))
        valid = True
        direction_source = "selected_date_axis"
        # Promoted from debug. This axis steers the entire corridor choice, and a
        # near-round date gives an almost arbitrary ellipse major axis: observed
        # a ~55deg swing on a stationary fruit that flipped FROM_RIGHT_10 into
        # FROM_LEFT_45 (branch_cost 22deg -> 67deg, straight 6cm approach -> a
        # curved 18.6cm one, final TCP 24mm above the date) and missed. Without
        # these numbers in the log there is no way to tell a good axis from a
        # flipped one after the fact.
        node.get_logger().info(
            f"[APPROACH_AXIS_SOURCE] class={class_label} "
            f"axis={axis_angle_deg:+.1f}deg applied={bounded_axis_deg:+.1f}deg "
            f"confidence={axis_confidence:.2f} stable={axis_stable} "
            "tool_roll=UNCHANGED")
    elif fruit_outward_dir is None:
        inward = nominal.copy()
        valid = False
        direction_source = "nominal_fallback"
    else:
        outward = np.asarray(fruit_outward_dir, dtype=float)
        inward_xy = -outward[:2]
        horizontal = float(np.linalg.norm(inward_xy))
        minimum = float(getattr(
            planner, "mid_high_direction_min_horizontal", 0.15))
        valid = horizontal >= minimum
        inward = (np.array([inward_xy[0] / horizontal,
                            inward_xy[1] / horizontal, 0.0])
                  if valid else nominal.copy())
        direction_source = (
            "selected_date_direction" if valid else "nominal_fallback")
    evaluated = []
    excluded = set(getattr(node, "_corridor_exclusions", set()))
    for label, direction in candidates:
        score = float(np.dot(direction, inward))
        error_deg = math.degrees(math.acos(float(np.clip(score, -1.0, 1.0))))
        evaluated.append({
            "label": label, "direction": direction,
            "score": score, "error_deg": error_deg})
        if bool(getattr(planner, "log_corridor_candidates", False)):
            node.get_logger().info(
                f"[APPROACH_CANDIDATE] class={class_label} side={label} "
                f"dir=[{direction[0]:+.3f},{direction[1]:+.3f},+0.000] "
                f"direction_match={score:.3f} "
                f"angular_error={error_deg:.1f}deg")
    available = [item for item in evaluated if item["label"] not in excluded]
    forced_label = getattr(node, "_corridor_forced_label", None)
    if forced_label is not None:
        forced = [item for item in available if item["label"] == forced_label]
        if forced:
            available = forced
    if bool(getattr(node, "_corridor_tip_fallback_active", False)):
        # Centre fallback is a bounded reachability recovery, not another full
        # nine-angle search. Try straight first, then the two nearest headings;
        # prefer the +/-10-degree side that best matches the frozen date axis.
        by_label = {item["label"]: item for item in available}
        left_10 = f"FROM_LEFT_{inner_angle_deg:g}"
        right_10 = f"FROM_RIGHT_{inner_angle_deg:g}"
        near = [label for label in (left_10, right_10) if label in by_label]
        near.sort(key=lambda label: by_label[label]["score"], reverse=True)
        fallback_order = ["STRAIGHT", *near]
        available = [by_label[label] for label in fallback_order
                     if label in by_label]
    if not available:
        node.get_logger().error(
            f"[APPROACH_SIDE_SELECTED] class={class_label} "
            "no corridors remain after Cartesian failures")
        return nominal, "NONE"
    best = (available[0] if bool(getattr(
        node, "_corridor_tip_fallback_active", False)) or forced_label is not None
        else max(available, key=lambda item: item["score"]))
    # Polarity is ambiguous only when the MIRRORED corridor is genuinely
    # competitive, not merely because the axis is valid. A fitted ellipse gives
    # an undirected major axis, so LEFT/RIGHT can both be geometrically valid --
    # but when the chosen heading matches the inward direction far better than
    # its mirror (e.g. 0.997 vs 0.814, a 31deg margin), the polarity is decided
    # and preflighting the mirror is ~1s of wasted IK per goal. Only treat it as
    # a genuine pair when the two are within corridor_polarity_margin of each
    # other, which is the case this pair evaluation exists to resolve.
    _mirror_label = None
    if isinstance(best.get("label"), str):
        if best["label"].startswith("FROM_LEFT_"):
            _mirror_label = best["label"].replace("FROM_LEFT_", "FROM_RIGHT_", 1)
        elif best["label"].startswith("FROM_RIGHT_"):
            _mirror_label = best["label"].replace("FROM_RIGHT_", "FROM_LEFT_", 1)
    _mirror_item = next(
        (item for item in evaluated if item["label"] == _mirror_label), None)
    _polarity_margin = float(getattr(planner, "corridor_polarity_margin", 0.10))
    _polarity_ambiguous = bool(
        axis_valid and _mirror_item is not None
        and (float(best["score"]) - float(_mirror_item["score"])) < _polarity_margin)
    if axis_valid and not _polarity_ambiguous and _mirror_item is not None:
        node.get_logger().debug(
            f"[CORRIDOR_POLARITY] decided: {best['label']} "
            f"match={float(best['score']):.3f} vs mirror {_mirror_label} "
            f"match={float(_mirror_item['score']):.3f} "
            f"(margin>={_polarity_margin:.2f}); skipping mirror preflight")
    node._corridor_axis_polarity_ambiguous = _polarity_ambiguous
    node.get_logger().info(
        f"[APPROACH_SIDE_SELECTED] class={class_label} side={best['label']} "
        f"source={direction_source} "
        f"inward=[{inward[0]:+.3f},{inward[1]:+.3f},+0.000] "
        f"direction_match={best['score']:.3f} "
        f"angular_error={best['error_deg']:.1f}deg "
        "orientation=FIXED insertion=LEVEL neighbours=IGNORED")
    return best["direction"], best["label"]


def side_low_wrist3_orientation(node, base_quat, desired_dir, max_delta_deg=45.0):
    """Use FK of a nearby wrist_3 adjustment so IK stays on the side-home branch."""
    delta = yaw_delta_for_local_axis(base_quat, desired_dir, local_axis=(0.0, 0.0, 1.0))
    if delta is None or node.current_joint_positions is None:
        return yaw_only_align_local_axis(base_quat, desired_dir), 0.0

    limit = math.radians(max_delta_deg)
    delta = max(-limit, min(limit, delta))
    q = list(node.current_joint_positions)
    q[5] += delta
    fk_pose = pose_from_joints(node, q)
    if fk_pose is None:
        return yaw_only_align_local_axis(base_quat, desired_dir), delta
    return fk_pose[3:], delta


def image_lateral_side(node, cx_norm, cy_norm, *, force_very_low_center=False):
    """Classify fruit side from bunch-relative position when available, else image position."""
    if not bool(getattr(node.cfg.planner, "side_approach_enabled", False)):
        return "CENTER"
    is_low = cy_norm > 0.60
    rel_y = getattr(node, "fruit_bunch_rel_y", None)
    lower_band = getattr(node.cfg.planner, "bunch_lower_center_band", 0.80)
    if rel_y is not None and rel_y >= lower_band:
        return "CENTER"
    if force_very_low_center:
        return "CENTER"

    left_thresh = float(getattr(
        node.cfg.planner,
        "low_left_thresh" if is_low else "mid_high_left_thresh",
        0.32 if is_low else 0.20,
    ))
    right_thresh = float(getattr(
        node.cfg.planner,
        "low_right_thresh" if is_low else "mid_high_right_thresh",
        0.68 if is_low else 0.80,
    ))

    rel_x = getattr(node, "fruit_bunch_rel_x", None)
    if rel_x is not None:
        if rel_x < left_thresh:
            return "LEFT"
        if rel_x > right_thresh:
            return "RIGHT"
        return "CENTER"

    if cx_norm < left_thresh:
        return "LEFT"
    if cx_norm > right_thresh:
        return "RIGHT"
    return "CENTER"


def is_bunch_lower_boundary(node):
    rel_y = getattr(node, "fruit_bunch_rel_y", None)
    lower_band = getattr(node.cfg.planner, "bunch_lower_center_band", 0.80)
    return rel_y is not None and rel_y >= lower_band


def _planner_value(planner, semantic_name, legacy_name=None, default=0.0):
    semantic_exists = hasattr(planner, semantic_name)
    legacy_exists = legacy_name is not None and hasattr(planner, legacy_name)
    if semantic_exists:
        semantic_value = getattr(planner, semantic_name)
        if legacy_exists:
            legacy_value = getattr(planner, legacy_name)
            # Backward compatibility: if an old ROS param/config override changed
            # the legacy field while the new semantic field is still at its default,
            # honor the legacy override.
            if semantic_value == default and legacy_value != default:
                return legacy_value
        return semantic_value
    if legacy_exists:
        return getattr(planner, legacy_name)
    return default


def low_side_standoff_offsets(node, is_left_side):
    """Return XYZ offsets using the configured depth/lateral axes."""
    side = "left" if is_left_side else "right"
    default_x = 0.12 if is_left_side else 0.08
    default_y = 0.065 if is_left_side else 0.050
    default_z = -0.055 if is_left_side else -0.025
    planner = node.cfg.planner
    lateral_mag = _planner_value(
        planner, f"low_{side}_standoff_lateral",
        f"low_{side}_standoff_x", default_x)
    depth_off = _planner_value(
        planner, f"low_{side}_standoff_depth",
        f"low_{side}_standoff_y", default_y)
    z_off = getattr(node.cfg.planner, f"low_{side}_standoff_z", default_z)
    lateral_off = lateral_mag * (1 if is_left_side else -1)
    if X_FORWARD_Y_LATERAL:
        return depth_off, lateral_off, z_off
    return lateral_off, depth_off, z_off


def lateral_value(x, y):
    return y if X_FORWARD_Y_LATERAL else x


def add_axis_offsets(x, y, *, depth=0.0, lateral=0.0):
    if X_FORWARD_Y_LATERAL:
        return x - depth, y + lateral
    return x + lateral, y + depth


def goal_is_in_robot_workspace(node, xyz):
    """Validate a vision goal in semantic forward/lateral robot coordinates."""
    if xyz is None or len(xyz) < 3 or not all(math.isfinite(float(v)) for v in xyz[:3]):
        return False, "non-finite XYZ"
    x, y, z = (float(v) for v in xyz[:3])
    forward = x if X_FORWARD_Y_LATERAL else -y
    lateral = y if X_FORWARD_Y_LATERAL else x
    planner = node.cfg.planner
    f_min = float(getattr(planner, "goal_workspace_forward_min_m", 0.35))
    f_max = float(getattr(planner, "goal_workspace_forward_max_m", 1.60))
    l_min = float(getattr(planner, "goal_workspace_lateral_min_m", -0.80))
    l_max = float(getattr(planner, "goal_workspace_lateral_max_m", 0.90))
    z_max = float(getattr(planner, "goal_workspace_z_max_m", 1.40))
    valid = (
        f_min <= forward <= f_max
        and l_min <= lateral <= l_max
        # Dates below base_link are valid in the harvesting workspace. Keep an
        # upper-Z guard, but do not reject a goal solely because Z is negative.
        and z <= z_max
    )
    reason = (
        f"forward={forward:.3f}m [{f_min:.2f},{f_max:.2f}], "
        f"lateral={lateral:.3f}m [{l_min:.2f},{l_max:.2f}], "
        f"z={z:.3f}m [no lower limit,{z_max:.2f}]"
    )
    return valid, reason


def _warn_rejected_goal_throttled(node, xyz, reason):
    now = time.time()
    if now - float(getattr(node, "_last_workspace_reject_log_s", 0.0)) >= 1.0:
        node._last_workspace_reject_log_s = now
        node.get_logger().warn(
            "Rejected vision goal outside robot workspace: "
            f"xyz=[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}] | {reason}")


def trunk_lateral(node, fallback=0.16):
    """Return the trunk coordinate along the configured lateral axis."""
    xyz = getattr(node, "trunk_xyz", None)
    if xyz is not None:
        return float(xyz[1] if X_FORWARD_Y_LATERAL else xyz[0])
    return float(fallback)


def compute_dynamic_side_home(node, fruit_xyz, is_left_side, reference_joints, desired_dir=None):
    """Return a date-clamped side-home pose using the fixed side-home FK as reference."""
    ref_pose = pose_from_joints(node, reference_joints)
    if ref_pose is None or len(ref_pose) < 7:
        return None, None

    planner = node.cfg.planner
    fx, fy, fz = [float(v) for v in fruit_xyz]
    mx = float(_planner_value(
        planner, "side_home_date_margin_lateral",
        "side_home_date_margin_x", 0.02))
    my = float(_planner_value(
        planner, "side_home_date_margin_depth",
        "side_home_date_margin_y", 0.08))
    mz = float(getattr(planner, "side_home_max_z_drop", 0.03))

    rx, ry, rz = [float(v) for v in ref_pose[:3]]
    lateral_off = -mx if is_left_side else mx
    if X_FORWARD_Y_LATERAL:
        dyn_x = min(rx, fx - my)
        dyn_y = fy + lateral_off
        dyn_y = min(ry, dyn_y) if is_left_side else max(ry, dyn_y)
    else:
        dyn_x = fx + lateral_off
        dyn_x = min(rx, dyn_x) if is_left_side else max(rx, dyn_x)
        dyn_y = max(ry, fy + my)
    dyn_z = max(rz, fz - mz)

    if desired_dir is not None and getattr(planner, "side_home_align_low_orientation", True):
        quat = yaw_only_align_local_axis(list(ref_pose[3:]), desired_dir)
    elif getattr(planner, "side_home_use_reference_orientation", True):
        quat = list(ref_pose[3:])
    else:
        cur_pose = node.get_end_effector_pose()
        quat = list(cur_pose[3:]) if cur_pose and len(cur_pose) >= 7 else list(ref_pose[3:])

    dynamic_pose = [dyn_x, dyn_y, dyn_z, *quat]
    clamps = {
        "ref": [rx, ry, rz],
        "dyn": [dyn_x, dyn_y, dyn_z],
        "x": abs(dyn_x - rx) > 1e-6,
        "y": abs(dyn_y - ry) > 1e-6,
        "z": abs(dyn_z - rz) > 1e-6,
    }
    return dynamic_pose, clamps


def quat_dot(q1, q2):
    """Dot product of two quaternions (measures similarity)."""
    return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]


def quat_flip_z(q):
    """Rotate quaternion by 180° around Z axis (flip gripper orientation)."""
    # q_z180 = (0, 0, 0, 1) represents 180° rotation around Z
    # q_new = q * q_z180
    w, x, y, z = q
    # Multiply by (0, 0, 0, 1):
    return [-z, y, -x, w]


def quat_slerp(q0, q1, t):
    """
    Spherical linear interpolation between quaternions.
    t=0 returns q0, t=1 returns q1.
    """
    dot = quat_dot(q0, q1)

    # If dot < 0, negate one quat to take shorter path
    if dot < 0:
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]]
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = [q0[i] + t * (q1[i] - q0[i]) for i in range(4)]
        # Normalize
        n = math.sqrt(sum(x*x for x in result))
        return [x/n for x in result]

    # Standard slerp
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return [s0 * q0[i] + s1 * q1[i] for i in range(4)]


def minimize_rotation_orientation(current_quat, target_quat, blend_weight=0.50):
    """
    Blend current and target orientation, prioritizing current.

    Since gripper can grasp from either direction (180° apart),
    first pick the closer one, then blend towards current.

    Args:
        current_quat: [qw, qx, qy, qz] current end-effector orientation
        target_quat: [qw, qx, qy, qz] target orientation from vision
        blend_weight: 0.0 = keep current, 1.0 = use target fully (default 0.3)

    Returns:
        [qw, qx, qy, qz] blended orientation
    """
    if current_quat is None or target_quat is None:
        return target_quat

    # Compute similarity (dot product) with target
    dot_orig = abs(quat_dot(current_quat, target_quat))

    # Compute flipped version (180° around Z)
    flipped = quat_flip_z(target_quat)
    dot_flip = abs(quat_dot(current_quat, flipped))

    # Higher dot = more similar = less rotation needed
    if dot_flip > dot_orig:
        best_target = flipped
    else:
        best_target = list(target_quat)

    # Blend: slerp from current towards best_target
    # blend_weight=0.3 means 70% current, 30% target
    return quat_slerp(list(current_quat), best_target, blend_weight)


def _jacobian_ik(node, target_xyz, start_js, max_iters=10, tol=0.003):
    """Solve position-only IK via damped least-squares Jacobian.

    For small displacements (~15cm), converges in 2-3 iterations.
    Guaranteed to stay near current joint configuration.

    Returns: goal joint positions (list) or None
    """
    from .fk import _get_kin_model

    try:
        use_cuda = torch.cuda.is_available() and not getattr(node, "_cuda_faulted", False)
        device = torch.device("cuda" if use_cuda else "cpu")
        kin = _get_kin_model(node)
        q = np.array(start_js, dtype=np.float64)
        target = np.array(target_xyz, dtype=np.float64)
        eps = 1e-4
        damping = 1e-3

        for _ in range(max_iters):
            q_t = torch.tensor([q.tolist()], dtype=torch.float32, device=device)
            with torch.no_grad():
                ee_pos, _, _, _, _, _, _ = kin.forward(q_t)
            current_xyz = ee_pos[0].cpu().numpy()

            error = target - current_xyz
            if np.linalg.norm(error) < tol:
                return q.tolist()

            # Numerical Jacobian (3x6)
            J = np.zeros((3, len(q)))
            for j in range(len(q)):
                q_pert = q.copy()
                q_pert[j] += eps
                q_pt = torch.tensor([q_pert.tolist()], dtype=torch.float32, device=device)
                with torch.no_grad():
                    ee_pert, _, _, _, _, _, _ = kin.forward(q_pt)
                J[:, j] = (ee_pert[0].cpu().numpy() - current_xyz) / eps

            # Damped least-squares: dq = J^T (J J^T + λ²I)^-1 * error
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            dq = np.clip(dq, -0.1, 0.1)
            q = q + dq

        # Check final error after max iterations (allow 2x convergence tol as fallback)
        q_t = torch.tensor([q.tolist()], dtype=torch.float32, device=device)
        with torch.no_grad():
            ee_pos, _, _, _, _, _, _ = kin.forward(q_t)
        final_err = np.linalg.norm(target - ee_pos[0].cpu().numpy())
        if final_err < tol * 2:
            return q.tolist()
        return None
    except Exception as e:
        msg = str(e)
        if "CUDA error" in msg or "illegal memory access" in msg:
            node._cuda_faulted = True
            try_cuda_recovery(node)
        node.get_logger().warn(f"[DIRECT] Jacobian IK failed: {e}")
        return None


def _direct_ik_move(node, target_pose_list, label="FINAL", motion_type="final",
                    store_trajectory=True, num_steps=None, goal_js_override=None,
                    require_cartesian=False):
    """Use Jacobian IK to find goal joints, then interpolate directly.
    Bypasses trajectory optimization — goes straight to the IK solution."""
    _direct_t0 = time.perf_counter()
    _record_motion = getattr(node, "_record_motion_plan_event", None)
    if node.current_joint_positions is None:
        for _ in range(20):
            time.sleep(0.05)
            if node.current_joint_positions is not None:
                break
        if node.current_joint_positions is None:
            node.get_logger().warn(f"[DIRECT] {label}: no joint state"); return False

    start_js = list(node.current_joint_positions)
    _verbose_direct = getattr(node.cfg.planner, "log_phase_timings", False)
    _preselected_branch = goal_js_override is not None

    # 1) cuRobo native IK (position + orientation), fallback to Jacobian IK (position-only)
    best_js = list(goal_js_override) if goal_js_override is not None else None
    if best_js is not None and _verbose_direct:
        node.get_logger().info(f"[DIRECT] {label}: using preflight-selected IK branch")
    if best_js is None and getattr(node, "_cuda_faulted", False):
        node.get_logger().warn(f"[DIRECT] {label}: skipping cuRobo IK (CUDA previously faulted)")
    elif best_js is None:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # goal_pose: position [1,3], quaternion [1,4]
            pos = torch.tensor([target_pose_list[:3]], dtype=torch.float32, device=device)
            quat = torch.tensor([target_pose_list[3:]], dtype=torch.float32, device=device)
            goal_pose = Pose(position=pos, quaternion=quat)
            # seed_config: (n_seeds, 1, dof)
            seed = torch.tensor([start_js], dtype=torch.float32, device=device).unsqueeze(0)
            # retract_config: (1, dof) — pull solution towards current joints
            retract = torch.tensor([start_js], dtype=torch.float32, device=device)
            ik_result = node.motion_gen.ik_solver.solve_single(
                goal_pose, seed_config=seed, retract_config=retract
            )
            if ik_result.success.item():
                best_js = ik_result.js_solution.position.squeeze().cpu().tolist()
                if _verbose_direct:
                    node.get_logger().info(f"[DIRECT] {label}: cuRobo IK solved (pos_err={ik_result.position_error.item():.4f}, rot_err={ik_result.rotation_error.item():.4f})")
        except Exception as e:
            msg = str(e)
            if "CUDA error" in msg or "illegal memory access" in msg:
                node._cuda_faulted = True
                try_cuda_recovery(node)
            if _verbose_direct:
                node.get_logger().info(f"[DIRECT] {label}: cuRobo IK exception: {e}")
    if best_js is None:
        if _verbose_direct:
            node.get_logger().info(f"[DIRECT] {label}: cuRobo IK failed, trying Jacobian fallback")
        try:
            best_js = _jacobian_ik(node, target_pose_list[:3], start_js)
        except Exception as e:
            node.get_logger().warn(f"[DIRECT] {label}: Jacobian fallback exception: {e}")
            best_js = None
    if best_js is None:
        node.get_logger().warn(f"[DIRECT] {label}: all IK solvers failed"); return False

    # Normalise IK solution to the same 2π branch as current joints.
    # cuRobo IK can return an equivalent config that is ±2π away, which would
    # make the interpolation travel a full revolution instead of staying put.
    best_js = nearest_joint_config(start_js, best_js)

    _cur_ee = node.get_end_effector_pose()
    _target_dist = None
    if _cur_ee:
        _target_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(_cur_ee[:3], target_pose_list[:3])))

    # If the IK solution is on a different kinematic branch (large joint delta even
    # after 2π normalisation), retry with perturbed seeds to find the nearest branch.
    # This prevents the arm taking a long arc when a shorter path exists.
    planner = node.cfg.planner
    _total_delta = sum(abs(g - c) for g, c in zip(best_js, start_js))
    _initial_best_delta = max(abs(g - c) for g, c in zip(best_js, start_js))
    _RETRY_THRESH_RAD = 0.70  # ~40° total — above this, search for a closer branch
    _MAX_DIRECT_DELTA_RAD = (
        math.radians(float(getattr(
            planner, "very_low_ik_max_joint_delta_deg", 75.0)))
        if _preselected_branch else 1.05
    )
    _retry_min_dist = getattr(planner, "direct_branch_retry_min_dist", 0.15)
    _retry_seed_count = max(0, int(getattr(planner, "direct_branch_retry_seeds", 4)))
    _close_move = _target_dist is not None and _target_dist < _retry_min_dist
    _must_retry_for_safety = _initial_best_delta > _MAX_DIRECT_DELTA_RAD
    _allow_branch_retry = (
        not _preselected_branch and
        _retry_seed_count > 0 and
        (not _close_move or _must_retry_for_safety)
    )
    if _total_delta > _RETRY_THRESH_RAD and _allow_branch_retry and not getattr(node, "_cuda_faulted", False):
        try:
            import random as _random
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pos  = torch.tensor([target_pose_list[:3]], dtype=torch.float32, device=device)
            quat = torch.tensor([target_pose_list[3:]], dtype=torch.float32, device=device)
            goal_pose = Pose(position=pos, quaternion=quat)
            retract   = torch.tensor([start_js], dtype=torch.float32, device=device)

            # Build a batch of seeds: current config + small random perturbations.
            _seeds = [start_js]
            for _ in range(_retry_seed_count):
                _perturb = [j + _random.uniform(-0.3, 0.3) for j in start_js]
                _seeds.append(_perturb)

            _best_alt = best_js
            _best_total = _total_delta
            for _s in _seeds:
                _seed_t = torch.tensor([_s], dtype=torch.float32, device=device).unsqueeze(0)
                _r = node.motion_gen.ik_solver.solve_single(goal_pose, seed_config=_seed_t, retract_config=retract)
                if _r.success.item():
                    _candidate = nearest_joint_config(start_js, _r.js_solution.position.squeeze().cpu().tolist())
                    _d = sum(abs(g - c) for g, c in zip(_candidate, start_js))
                    if _d < _best_total:
                        _best_total = _d
                        _best_alt = _candidate
            if _best_alt is not best_js:
                if _verbose_direct:
                    node.get_logger().info(
                        f"[DIRECT] {label}: branch retry found shorter path "
                        f"({_total_delta*57.3:.1f}° → {_best_total*57.3:.1f}° total)")
                best_js = _best_alt
        except Exception as _e:
            node.get_logger().warn(f"[DIRECT] {label}: branch retry exception: {_e}")
    elif _total_delta > _RETRY_THRESH_RAD and not _allow_branch_retry:
        _dist_msg = "unknown" if _target_dist is None else f"{_target_dist:.3f}m"
        if _verbose_direct:
            node.get_logger().info(
                f"[DIRECT] {label}: branch retry skipped "
                f"(dist={_dist_msg}, min={_retry_min_dist:.3f}m, "
                f"delta={_initial_best_delta*57.3:.1f}deg, seeds={_retry_seed_count})")

    best_delta = max(abs(g - c) for g, c in zip(best_js, start_js))

    # Scale num_steps with Cartesian distance (1 step per 5mm, clamped 10-60)
    if num_steps is None:
        if _target_dist is not None:
            num_steps = max(10, min(60, int(_target_dist / 0.005)))
        else:
            num_steps = 30

    if _verbose_direct:
        node.get_logger().info(
            f"[DIRECT] {label}: max_joint_delta={best_delta*57.3:.1f}deg, {num_steps}-step interpolation"
        )

    # Safety: if IK solution is too far, fall back
    if best_delta > _MAX_DIRECT_DELTA_RAD:
        _set_goal_rejection(
            node,
            "IK jump too large",
            f"{best_delta*57.3:.1f}deg")
        node.get_logger().warn(f"[DIRECT] {label}: IK too far ({best_delta*57.3:.1f}deg)"); return False

    # 3) Cartesian IK waypoints → per-segment S-curve interpolation
    #
    # Direct joint-space interpolation from approach to final can arc through
    # dangerous wrist configurations (lower-arm / tool-flange clamping) on
    # left+low fruits.  Instead, solve IK at N evenly-spaced Cartesian positions
    # along the straight EE line — each seeded from the previous solution so the
    # arm stays on the same kinematic branch throughout.
    _N_CART = max(0, int(getattr(
        planner,
        "direct_approach_cart_waypoints" if require_cartesian
        else "direct_final_cart_waypoints",
        3 if require_cartesian else 2)))
    _states_built = False
    states = []
    _use_cartesian_ik = (label == "FINAL" or require_cartesian) and _N_CART > 0
    if _use_cartesian_ik and _cur_ee is not None and not getattr(node, "_cuda_faulted", False):
        # Hold the YOLO inference lock across this IK burst. plan_and_send()
        # already does this for plan_single -- YOLO TRT and cuRobo kernels
        # contending on the shared Jetson GPU is both slower and the documented
        # source of device-side asserts -- but this path solved unlocked, so the
        # per-waypoint chain (plus its branch retries) raced vision inference at
        # ~7Hz. This is FINAL's planner and the straight approach entry's, and
        # FINAL alone measured ~940ms per cycle. Released before the publish and
        # wait below, so vision is never starved while the arm is moving.
        _yolo_dm = getattr(node, 'yolo_thread', None)
        _yolo_dm_lock = getattr(_yolo_dm, 'inference_lock', None)
        if _yolo_dm_lock:
            _yolo_dm_lock.acquire()
        try:
            _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _start_quat = list(_cur_ee[3:]) if len(_cur_ee) > 3 else [1.0, 0.0, 0.0, 0.0]
            _target_quat = list(target_pose_list[3:])
            _wp_js = [start_js]
            _prev_js = start_js
            _cart_ok = True
            for _k in range(1, _N_CART + 1):
                _a = _k / (_N_CART + 1)
                _wp_xyz = [_cur_ee[i] + _a * (target_pose_list[i] - _cur_ee[i]) for i in range(3)]
                # Slerp orientation from current to target so wrist rotates gradually —
                # using fixed target orientation for all waypoints forces an immediate wrist
                # snap at the first IK point, which can create a tool-flange/lower-arm
                # proximity violation in the interpolated states between waypoints.
                _wp_quat = quat_slerp(_start_quat, _target_quat, _a)
                _pos_t   = torch.tensor([_wp_xyz],  dtype=torch.float32, device=_dev)
                _quat_t  = torch.tensor([_wp_quat], dtype=torch.float32, device=_dev)
                _seed_t  = torch.tensor([_prev_js], dtype=torch.float32, device=_dev).unsqueeze(0)
                _ret_t   = torch.tensor([_prev_js], dtype=torch.float32, device=_dev)
                _r = node.motion_gen.ik_solver.solve_single(
                    Pose(position=_pos_t, quaternion=_quat_t),
                    seed_config=_seed_t, retract_config=_ret_t)
                # A single seeded solve can miss a valid nearby branch. Retry
                # deterministically around the previous waypoint before
                # declaring the straight Cartesian corridor infeasible.
                if not _r.success.item():
                    _retry_offsets = (0.04, -0.04, 0.08, -0.08)
                    _best_retry = None
                    _best_retry_delta = float("inf")
                    for _off in _retry_offsets:
                        for _joint_idx in (0, 2, 4, 5):
                            _retry_seed = list(_prev_js)
                            _retry_seed[_joint_idx] += _off
                            _retry_seed_t = torch.tensor(
                                [_retry_seed], dtype=torch.float32,
                                device=_dev).unsqueeze(0)
                            _rr = node.motion_gen.ik_solver.solve_single(
                                Pose(position=_pos_t, quaternion=_quat_t),
                                seed_config=_retry_seed_t,
                                retract_config=_ret_t)
                            if _rr.success.item():
                                _rr_js = nearest_joint_config(
                                    _prev_js,
                                    _rr.js_solution.position.squeeze().cpu().tolist())
                                _rr_delta = max(abs(a - b) for a, b in zip(
                                    _rr_js, _prev_js))
                                if _rr_delta < _best_retry_delta:
                                    _best_retry = _rr
                                    _best_retry_delta = _rr_delta
                    if _best_retry is not None:
                        _r = _best_retry
                        node.get_logger().info(
                            f"[DIRECT] {label}: Cartesian waypoint {_k} "
                            f"recovered with alternate seed "
                            f"(max_delta={_best_retry_delta*57.3:.1f}deg)")
                if _r.success.item():
                    _wj = nearest_joint_config(
                        _prev_js, _r.js_solution.position.squeeze().cpu().tolist())
                    # Reject if this waypoint jumps to a different kinematic branch.
                    # nearest_joint_config only fixes ±2π wrapping — a branch jump
                    # still shows up as a large per-joint delta, which causes a jerk
                    # in the trajectory and a zig-zag in the RViz path visualization.
                    _wp_delta = max(abs(a - b) for a, b in zip(_wj, _prev_js))
                    if _wp_delta > 0.52:  # ~30° — branch jumped, abort Cartesian path
                        _cart_ok = False
                        node.get_logger().warn(
                            f"[DIRECT] {label}: Cartesian waypoint {_k} jumped "
                            f"{_wp_delta*57.3:.1f}deg — aborting (branch switch)")
                        break
                    _wp_js.append(_wj)
                    _prev_js = _wj
                else:
                    _cart_ok = False
                    node.get_logger().warn(
                        f"[DIRECT] {label}: Cartesian IK failed at step {_k}/{_N_CART}")
                    break
            if _cart_ok:
                # Re-solve final endpoint seeded from last intermediate waypoint
                _pos_final = torch.tensor([target_pose_list[:3]], dtype=torch.float32, device=_dev)
                _quat_final = torch.tensor([_target_quat], dtype=torch.float32, device=_dev)
                _seed_final = torch.tensor([_prev_js], dtype=torch.float32, device=_dev).unsqueeze(0)
                _ret_final  = torch.tensor([_prev_js], dtype=torch.float32, device=_dev)
                _r_final = node.motion_gen.ik_solver.solve_single(
                    Pose(position=_pos_final, quaternion=_quat_final),
                    seed_config=_seed_final, retract_config=_ret_final)
                if _r_final.success.item():
                    _final_js = nearest_joint_config(
                        _prev_js, _r_final.js_solution.position.squeeze().cpu().tolist())
                    _final_delta = max(abs(g - c) for g, c in zip(_final_js, _prev_js))
                    if _verbose_direct:
                        node.get_logger().info(
                            f"[DIRECT] {label}: final-seg re-solve delta={_final_delta*57.3:.1f}deg")
                    _wp_js.append(_final_js)
                else:
                    _final_delta = max(abs(g - c) for g, c in zip(best_js, _prev_js))
                    node.get_logger().warn(
                        f"[DIRECT] {label}: final-seg re-solve failed, using best_js "
                        f"(delta={_final_delta*57.3:.1f}deg)")
                    if _final_delta > 1.05:
                        node.get_logger().warn(
                            f"[DIRECT] {label}: final segment too large ({_final_delta*57.3:.1f}deg) — aborting")
                        _cart_ok = False
                    else:
                        _wp_js.append(best_js)
            if _cart_ok:
                # Linear interpolation — build_trajectory central-difference gives
                # one smooth continuous velocity profile across all waypoints.
                # Minimum 10 steps per segment for smooth central-difference velocities.
                _steps_per = max(10, num_steps // max(len(_wp_js) - 1, 1))
                for _si in range(len(_wp_js) - 1):
                    _s0, _s1 = _wp_js[_si], _wp_js[_si + 1]
                    for _i in range(_steps_per + 1):
                        if _i == 0 and _si > 0:
                            continue  # avoid duplicate at segment boundary
                        _t = _i / _steps_per
                        states.append([_s0[j] + _t * (_s1[j] - _s0[j]) for j in range(len(_s0))])
                _states_built = True
                if _verbose_direct:
                    node.get_logger().info(
                        f"[DIRECT] {label}: Cartesian IK path — "
                        f"{len(_wp_js)} waypoints, {len(states)} states")
        except Exception as _ce:
            node.get_logger().warn(f"[DIRECT] {label}: Cartesian IK path failed: {_ce}")
        finally:
            if _yolo_dm_lock:
                _yolo_dm_lock.release()

    if not _states_built:
        if require_cartesian:
            if callable(_record_motion):
                _record_motion(
                    stage="PLAN", label=label, motion_type=motion_type,
                    planner="direct_cartesian_ik", result="FAILED",
                    reason="CARTESIAN_CORRIDOR_UNAVAILABLE",
                    planning_ms=round((time.perf_counter() - _direct_t0) * 1000.0, 1),
                    cartesian_waypoints=int(_N_CART),
                    target_xyz_m=[float(v) for v in target_pose_list[:3]])
            node.get_logger().error(
                f"[DIRECT] {label}: required straight Cartesian corridor unavailable; "
                "rejecting instead of using a sideways joint-space path")
            return False
        if label == "FINAL":
            if bool(getattr(planner, "strict_final_cartesian_only", True)):
                node.get_logger().error(
                    "[DIRECT] FINAL: straight Cartesian insertion unavailable; "
                    "joint fallback DISABLED — rejecting grasp")
                return False
            # Sometimes an intermediate Cartesian waypoint jumps IK branch even
            # though the final endpoint IK is close. In that case a guarded
            # joint interpolation is safer than falling back to slow trajopt.
            _fallback_limit = math.radians(float(
                getattr(planner, "direct_final_joint_fallback_max_delta_deg", 25.0)))
            if best_delta <= _fallback_limit:
                node.get_logger().warn(
                    f"[DIRECT] {label}: Cartesian waypoint branch switch; "
                    f"using guarded joint fallback (max_delta={best_delta*57.3:.1f}deg)")
            else:
                # For large endpoint moves, defer to plan_and_send for a smooth cuRobo trajectory.
                if _verbose_direct:
                    node.get_logger().info(f"[DIRECT] {label}: Cartesian path unavailable — deferring to plan_and_send")
                return False
        # For short moves (DEPTH_CORRECT etc.), joint-space interpolation is fine.
        if _verbose_direct:
            node.get_logger().info(f"[DIRECT] {label}: fallback to joint-space interpolation")
        for i in range(num_steps + 1):
            alpha = i / num_steps
            t_smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
            wp = [s + t_smooth * (g - s) for s, g in zip(start_js, best_js)]
            states.append(wp)

    # 4) Build slow, smooth trajectory
    planner = node.cfg.planner
    base_dt = getattr(planner, "base_dt", 0.02)
    global_scale = max(getattr(node, "speed_scale", 1.0), 1e-6)
    type_scale = getattr(planner, f"speed_{motion_type}", getattr(planner, "speed_final", 1.0))
    scale = global_scale * type_scale
    dt = min(max(base_dt / max(scale, 1e-6), planner.min_dt), planner.max_dt)
    vel = min(0.05 * scale, 0.15)

    traj = build_trajectory(node.joint_order, states, vel=vel, dt=dt,
                            stop_flag=lambda: node.stop_requested,
                            max_vel=planner.max_joint_velocity * 0.5,
                            max_acc=planner.max_joint_acceleration * 0.3,
                            ramp_points=0, include_acc=False)
    if node.stop_requested:
        node.stop_requested = False; return False

    # 5) Clamp check. Reject the complete motion when any sampled waypoint is
    # below the safety threshold. A truncated approach leaves the arm at an
    # unintended pose and makes the subsequent FINAL path less predictable.
    _clamp_mm, _safe_cutoff, _dest_mm = _log_forearm_flange_clearance(node, states, label)
    _clamp_threshold = float(getattr(node.cfg.planner, "clamp_safety_threshold_mm", 35.0))
    _this_traj_truncated = False  # local flag — avoids leaking into subsequent calls
    _is_approach = label.startswith("APPROACH")
    _is_final = label.startswith("FINAL")
    if _is_final and _clamp_mm < _clamp_threshold:
        _set_goal_rejection(
            node,
            f"clamp clearance below {_clamp_threshold:.0f} mm",
            f"min={_clamp_mm:.1f}mm")
        node.get_logger().error(
            f"[CLAMP] {label}: unsafe FINAL clearance "
            f"(min={_clamp_mm:.1f}mm, destination={_dest_mm:.1f}mm, "
            f"required={_clamp_threshold:.1f}mm) — skipping goal")
        return False
    if _is_approach and _clamp_mm < _clamp_threshold:
        _set_goal_rejection(
            node,
            f"clamp clearance below {_clamp_threshold:.0f} mm",
            f"min={_clamp_mm:.1f}mm")
        node.get_logger().error(
            f"[CLAMP] {label}: rejecting complete APPROACH "
            f"(min={_clamp_mm:.1f}mm, destination={_dest_mm:.1f}mm, "
            f"required={_clamp_threshold:.1f}mm)")
        node._approach_clamp_rejected = True
        return False

    cart_path = forward_kinematics_batch(node, states)
    _lateral_deviation_m = 0.0
    if len(cart_path) >= 2:
        _p0 = np.asarray([cart_path[0].x, cart_path[0].y, cart_path[0].z], dtype=float)
        _p1 = np.asarray([cart_path[-1].x, cart_path[-1].y, cart_path[-1].z], dtype=float)
        _line = _p1 - _p0
        _line_norm = float(np.linalg.norm(_line))
        if _line_norm > 1e-9:
            _axis = _line / _line_norm
            for _p in cart_path:
                _delta = np.asarray([_p.x, _p.y, _p.z], dtype=float) - _p0
                _perp = _delta - float(np.dot(_delta, _axis)) * _axis
                _lateral_deviation_m = max(
                    _lateral_deviation_m, float(np.linalg.norm(_perp)))
    node.get_logger().info(
        f"[PATH_LATERAL] {label}: max_deviation="
        f"{_lateral_deviation_m * 1000.0:.1f}mm "
        f"cartesian_required={require_cartesian}")
    node._planned_cartesian_path = cart_path  # keep PATH_DEV in sync with this motion
    publish_planned_path(node, states, label, cartesian_points=cart_path)
    node.trajectory_pub.publish(traj)
    if callable(_record_motion):
        _record_motion(
            stage="PLAN", label=label, motion_type=motion_type,
            planner="direct_cartesian_ik" if _states_built else "direct_joint_ik",
            result="SUCCESS",
            planning_ms=round((time.perf_counter() - _direct_t0) * 1000.0, 1),
            trajectory_samples=len(states), dt_s=round(float(dt), 5),
            trajectory_duration_s=round(float(dt) * max(len(states) - 1, 0), 3),
            max_joint_delta_deg=round(math.degrees(float(best_delta)), 2),
            cartesian_waypoints=int(_N_CART if _states_built else 0),
            lateral_deviation_mm=round(_lateral_deviation_m * 1000.0, 1),
            clamp_clearance_mm=round(float(_clamp_mm), 1),
            clamp_required_mm=round(float(_clamp_threshold), 1),
            target_xyz_m=[float(v) for v in target_pose_list[:3]])

    # Wait target: only use FK of the truncated endpoint when THIS trajectory was
    # truncated.  Using the node flag (_approach_truncated) instead caused the flag
    # to leak from a previous APPROACH into a later FINAL call, mis-directing the wait.
    # A LIST, deliberately, and published on the node. wait_until_xyz reads
    # `math.dist(cur[:3], target_xyz)` fresh on every loop iteration, so
    # mutating this list in place retargets the move already in flight without
    # touching that function. _final_inflight_apply uses it.
    _wait_xyz = list(target_pose_list[:3])
    if _is_final:
        node._final_live_wait_xyz = _wait_xyz
        node._final_retarget_xyz = None
    if _this_traj_truncated:
        _fk_end = forward_kinematics(node, states[-1])
        if _fk_end:
            # In place: node._final_live_wait_xyz holds a reference to this
            # same list, and rebinding here would orphan it -- the retarget
            # would then mutate a list nothing reads.
            _wait_xyz[0], _wait_xyz[1], _wait_xyz[2] = (
                _fk_end.x, _fk_end.y, _fk_end.z)

    # Wait for motion to finish (with orientation + velocity checks), then blend
    target_quat = target_pose_list[3:] if len(target_pose_list) > 3 else None
    _endpoint_tol = (
        float(getattr(planner, "final_endpoint_tolerance", 0.004))
        if _is_final else 0.008
    )
    reached = wait_until_xyz(
        node, _wait_xyz, tol=_endpoint_tol, timeout=8.0,
        target_quat=target_quat)
    blend_motion(node)
    _execution_ms = (time.perf_counter() - _direct_t0) * 1000.0

    # Abort goal if robot has stalled twice — something is obstructing or IK is wrong
    if not reached and getattr(node, '_goal_stall_count', 0) >= 2:
        node.get_logger().warn(
            f"[DIRECT] {label}: stall count={node._goal_stall_count} — aborting goal.")
        return False

    # Reject stall-acceptance far from target — but skip this check when the trajectory
    # was intentionally truncated: the arm is supposed to stop short of the original target.
    cur_pose = node.get_end_effector_pose()
    if _this_traj_truncated:
        # Arm stopped at the safe cutoff — this is a successful partial approach
        if store_trajectory:
            if not hasattr(node, 'stored_trajectory_states'):
                node.stored_trajectory_states = []
            node.stored_trajectory_states.extend(states)
        return True
    if cur_pose:
        # Measure against the goal the arm was actually driving to. An in-flight
        # retarget moves it, and comparing to the ORIGINAL target then reports
        # the size of the correction as endpoint error: observed 2026-09-13 as
        # "endpoint error=8.2mm exceeds 4.0mm -- treating as failure" after two
        # correct retargets totalling 11.1mm, which sent a good grasp back to
        # HOME to retry another corridor.
        #
        # Accuracy is still enforced to 4mm, just against the right point. The
        # retarget is quality-gated and capped at 25mm, so this cannot excuse an
        # arbitrary miss.
        _endpoint_ref = getattr(node, "_final_retarget_xyz", None) or target_pose_list[:3]
        final_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(cur_pose[:3], _endpoint_ref)))
        if _is_final and final_dist > _endpoint_tol:
            if callable(_record_motion):
                _record_motion(
                    stage="ENDPOINT", label=label, motion_type=motion_type,
                    result="FAILED", reason="ENDPOINT_ERROR",
                    endpoint_error_mm=round(final_dist * 1000.0, 1),
                    tolerance_mm=round(_endpoint_tol * 1000.0, 1),
                    elapsed_ms=round(_execution_ms, 1))
            node.get_logger().warn(
                f"[DIRECT] {label}: endpoint error={final_dist*1000:.1f}mm "
                f"exceeds {_endpoint_tol*1000:.1f}mm — treating as failure")
            return False
        if final_dist > 0.015:
            node.get_logger().warn(
                f"[DIRECT] {label}: stalled {final_dist*100:.1f}cm from target — treating as failure")
            return False

    if store_trajectory:
        if not hasattr(node, 'stored_trajectory_states'):
            node.stored_trajectory_states = []
        node.stored_trajectory_states.extend(states)

    if callable(_record_motion):
        _endpoint_error = None
        if cur_pose:
            _endpoint_error = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(
                    cur_pose[:3], target_pose_list[:3]))) * 1000.0
        _record_motion(
            stage="ENDPOINT", label=label, motion_type=motion_type,
            result="REACHED" if reached else "SETTLED",
            endpoint_error_mm=(round(_endpoint_error, 1)
                               if _endpoint_error is not None else None),
            tolerance_mm=round(_endpoint_tol * 1000.0, 1),
            elapsed_ms=round(_execution_ms, 1))

    return True


def _preflight_approach_final_chain(node, start_js, approach_pose, final_pose,
                                    waypoint_count=2):
    """Validate APPROACH and fixed-orientation straight FINAL before motion.

    Holds the YOLO inference lock across the whole IK burst. plan_and_send()
    already does this for plan_single -- YOLO TRT and cuRobo kernels contending
    on the shared Jetson GPU is both slower and the source of device-side
    asserts -- but this preflight was solving unlocked, so its chain of solves
    raced vision inference at ~7Hz. Taking the lock once for the burst costs at
    most one in-flight inference (~140ms) and then runs uncontended, instead of
    every solve fighting for the GPU.
    """
    node._corridor_preflight_approach_js = None
    node._corridor_preflight_cost_deg = float("inf")
    if start_js is None or getattr(node, "_cuda_faulted", False):
        return False, "NO_IK_STATE"
    _yolo = getattr(node, 'yolo_thread', None)
    _yolo_lock = getattr(_yolo, 'inference_lock', None)
    # Split the cost: time spent waiting for the YOLO lock vs time actually
    # inside cuRobo, and per solve. The chain runs 1 + (waypoint_count + 1)
    # solves; a single solve_single() was measured at ~22ms on this hardware,
    # so ~90ms is expected, while the field logs show ~900ms with vision ALREADY
    # paused. That rules out GPU contention and points at the CUDA graph not
    # being reused for this call shape. These numbers say which it is, and must
    # be answered before any batching refactor -- batching 4 fast solves saves
    # little, batching 4 slow non-graphed solves saves a lot.
    _pf_lock_t0 = time.time()
    if _yolo_lock:
        _yolo_lock.acquire()
    _pf_lock_ms = (time.time() - _pf_lock_t0) * 1000.0
    _pf_solve_ms = []
    _pf_split_ms = []   # (prep, call, sync) ms per solve
    _pf_cpu_ms = []     # thread CPU ms inside the call
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def solve(pose, seed):
            # cuRobo's IK solver already searches ik_solver.num_seeds (32 by
            # default) candidates in parallel inside ONE solve_single() call,
            # regularized toward retract_config (== seed here) so it already
            # favors the branch closest to the current joint state -- that's
            # what the old manual 5-seed Python retry loop was trying to
            # approximate by hand, at ~6x the cost of one plain call (measured
            # ~127ms sequential-loop vs ~22ms single call on this hardware),
            # since each of its 5 calls *also* internally ran the same 32-seed
            # search. Kept to a single plain-shape call so it matches the
            # solve state MotionGen.warmup() already CUDA-graphs -- passing a
            # custom num_seeds/return_seeds here breaks the graph on this
            # cuRobo/CUDA build ("changing goal type, cuda graph reset not
            # available"). Trade-off vs. the old code: this no longer retries
            # with hand-nudged wrist seeds when the first result isn't
            # continuous enough -- it may reject a few corridors the manual
            # nudge would have rescued, but never accepts a worse branch than
            # the manual loop would have, and the continuity/branch-switch
            # checks below still run on whatever this returns.
            _solve_t0 = time.time()
            _prep_t0 = _solve_t0
            pos = torch.tensor([pose[:3]], dtype=torch.float32, device=dev)
            quat = torch.tensor([pose[3:]], dtype=torch.float32, device=dev)
            retract_t = torch.tensor([seed], dtype=torch.float32, device=dev)
            goal_pose = Pose(position=pos, quaternion=quat)
            seed_t = torch.tensor(
                [seed], dtype=torch.float32, device=dev).unsqueeze(0)
            # Split the cost three ways. A standalone benchmark of this exact
            # call -- production collision world, a real VERY_LOW pose, both
            # call shapes, even with the vision stack running -- measures 14ms,
            # while the field measures ~250ms. So the time is NOT in the solver,
            # and lumping it into one number cannot say where it is:
            #   prep  = tensor/Pose construction (pure CPU, holds the GIL)
            #   call  = solve_single itself (async GPU launch)
            #   sync  = .item()/.cpu() -- where the thread blocks, and where any
            #           contention or descheduling in this busy multi-threaded
            #           node actually shows up
            # Wall clock AND this thread's CPU time across the call. A
            # standalone benchmark of this exact call measures 14ms -- with the
            # production world, a real VERY_LOW pose, both call shapes, an
            # update_world() after warmup, and even with the vision stack
            # running. Production measures 144-396ms for the same line. The only
            # difference left is that production runs it inside a busy
            # multi-threaded ROS node, so the question is whether the thread is
            # BUSY for those 250ms or merely not scheduled:
            #   cpu ~= wall  -> the solve genuinely costs that much here
            #   cpu <<  wall -> the thread is blocked/descheduled (GIL, executor
            #                   contention) and cuRobo is not the problem
            _call_t0 = time.time()
            _call_c0 = time.thread_time()
            result = node.motion_gen.ik_solver.solve_single(
                goal_pose, seed_config=seed_t, retract_config=retract_t)
            _call_cpu_ms = (time.thread_time() - _call_c0) * 1000.0
            _sync_t0 = time.time()
            if not result.success.item():
                _now = time.time()
                _pf_solve_ms.append((_now - _solve_t0) * 1000.0)
                _pf_split_ms.append((
                    (_call_t0 - _prep_t0) * 1000.0,
                    (_sync_t0 - _call_t0) * 1000.0,
                    (_now - _sync_t0) * 1000.0))
                _pf_cpu_ms.append(_call_cpu_ms)
                return None
            _out = nearest_joint_config(
                seed, result.js_solution.position.squeeze().cpu().tolist())
            _now = time.time()
            _pf_solve_ms.append((_now - _solve_t0) * 1000.0)
            _pf_split_ms.append((
                (_call_t0 - _prep_t0) * 1000.0,
                (_sync_t0 - _call_t0) * 1000.0,
                (_now - _sync_t0) * 1000.0))
            _pf_cpu_ms.append(_call_cpu_ms)
            return _out

        approach_js = solve(approach_pose, start_js)
        if approach_js is None:
            return False, "APPROACH_IK"
        _approach_deltas = [
            abs(a - b) for a, b in zip(approach_js, start_js)]
        approach_delta = max(_approach_deltas)
        _worst_joint = _approach_deltas.index(approach_delta)
        chain_max_delta = approach_delta
        max_approach_delta = math.radians(float(getattr(
            node.cfg.planner,
            "corridor_preflight_approach_max_joint_delta_deg", 75.0)))
        if approach_delta > max_approach_delta:
            # Report the magnitude, not just the verdict. How far over the limit
            # decides the remedy: a few degrees over means a nearby staging
            # posture makes this corridor reachable, whereas a near-180deg flip
            # means the arm is simply on the wrong side and no amount of
            # corridor retrying will help.
            return False, (
                f"APPROACH_BRANCH_SWITCH(j{_worst_joint}="
                f"{math.degrees(approach_delta):.0f}deg"
                f">{math.degrees(max_approach_delta):.0f}deg)")

        previous = approach_js
        for index in range(1, waypoint_count + 2):
            alpha = index / (waypoint_count + 1)
            xyz = [
                approach_pose[i] + alpha * (final_pose[i] - approach_pose[i])
                for i in range(3)
            ]
            candidate = solve([*xyz, *final_pose[3:]], previous)
            if candidate is None:
                return False, f"FINAL_WAYPOINT_{index}"
            _step_deltas = [
                abs(a - b) for a, b in zip(candidate, previous)]
            _step_max = max(_step_deltas)
            if _step_max > math.radians(30.0):
                return False, (
                    f"FINAL_BRANCH_SWITCH_{index}"
                    f"(j{_step_deltas.index(_step_max)}="
                    f"{math.degrees(_step_max):.0f}deg>30deg)")
            chain_max_delta = max(
                chain_max_delta,
                max(abs(a - b) for a, b in zip(candidate, previous)))
            previous = candidate
        # Execute the exact branch that passed this chain. Re-solving APPROACH
        # later can select a different branch and produce a multi-metre detour.
        node._corridor_preflight_approach_js = list(approach_js)
        node._corridor_preflight_cost_deg = math.degrees(chain_max_delta)
        return True, "OK"
    except Exception as exc:
        node.get_logger().warn(f"[CORRIDOR_PREFLIGHT] IK exception: {exc}")
        return False, "IK_EXCEPTION"
    finally:
        # Publish the breakdown on every exit path, including rejections.
        node._corridor_preflight_lock_ms = _pf_lock_ms
        node._corridor_preflight_solve_ms = list(_pf_solve_ms)
        node._corridor_preflight_split_ms = list(_pf_split_ms)
        node._corridor_preflight_cpu_ms = list(_pf_cpu_ms)
        if _yolo_lock:
            _yolo_lock.release()


def _log_forearm_flange_clearance(node, states: list, label: str, log_result: bool = True):
    """Check forearm_link / tool0 clearance across sampled waypoints.

    Returns (min_dist_mm, safe_cutoff_idx, destination_mm):
      min_dist_mm     -- global minimum surface distance found (mm); negative infinity on error
      safe_cutoff_idx -- index of last sampled waypoint that is >= threshold; len(states)
                         if all waypoints are safe, or 0 if even the first is unsafe.
                         Callers can truncate states[:safe_cutoff_idx+1] to stay in the
                         safe zone rather than rejecting the whole trajectory.
      destination_mm  -- surface distance at the final waypoint.
    """
    try:
        if not states:
            return float('-inf'), 0, float('-inf')
        from .fk import _fk_lock, _get_kin_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kin = _get_kin_model(node)

        kc = kin.kinematics_config
        idx_map = kc.link_sphere_idx_map.cpu()
        name_map = kc.link_name_to_idx_map

        fa_link_idx = name_map.get('forearm_link')
        t0_link_idx = name_map.get('tool0')
        if fa_link_idx is None or t0_link_idx is None:
            node.get_logger().error(f"[CLAMP] {label}: link missing from link map")
            return float('-inf'), 0, float('-inf')

        fa_sph_list = torch.where(idx_map == fa_link_idx)[0].tolist()
        t0_sph_list = torch.where(idx_map == t0_link_idx)[0].tolist()
        if not fa_sph_list or not t0_sph_list:
            node.get_logger().error(f"[CLAMP] {label}: no spheres found")
            return float('-inf'), 0, float('-inf')

        threshold_mm = float(getattr(node.cfg.planner, "clamp_safety_threshold_mm", 35.0))

        # Short direct motions are checked at every waypoint. Longer trajectories
        # are sampled evenly to keep the safety check bounded.
        n_samples = min(60, len(states))
        step = max(1, len(states) // n_samples)
        sample_indices = list(range(0, len(states), step))
        if sample_indices[-1] != len(states) - 1:
            sample_indices.append(len(states) - 1)

        danger_mm = float(getattr(
            node.cfg.planner, "clamp_safety_threshold_mm", 45.0))

        min_dist_mm = float('inf')
        destination_mm = float('inf')   # clearance at the last sampled waypoint (destination)
        worst_idx = sample_indices[-1]
        worst_fa = fa_sph_list[-1]
        worst_t0 = t0_sph_list[0]

        # Last sample before the first point below the configured safety threshold.
        safe_cutoff_idx = len(states)
        first_unsafe_found = False

        q = torch.tensor(
            [states[idx] for idx in sample_indices],
            dtype=torch.float32,
            device=device,
        )
        with _fk_lock:
            with torch.no_grad():
                ls = kin.forward(q)[6]
            if ls.ndim == 2:
                ls = ls.unsqueeze(0)
            if ls.ndim != 3 or ls.shape[0] != len(sample_indices):
                # Defensive fallback for malformed output from a CUDA kernel or
                # a model implementation with stale internal batch buffers.
                malformed_shape = tuple(ls.shape)
                sphere_frames = []
                with torch.no_grad():
                    for sample_q in q:
                        sample_ls = kin.forward(sample_q.unsqueeze(0))[6]
                        if sample_ls.ndim == 2:
                            sample_ls = sample_ls.unsqueeze(0)
                        if sample_ls.ndim != 3 or sample_ls.shape[0] < 1:
                            raise ValueError(
                                f"single-waypoint sphere shape {tuple(sample_ls.shape)}")
                        sphere_frames.append(sample_ls[0].clone())
                ls = torch.stack(sphere_frames, dim=0)
                if log_result:
                    node.get_logger().warn(
                        f"[CLAMP] {label}: recovered malformed FK sphere batch "
                        f"{malformed_shape} using {len(sample_indices)} single-waypoint checks")
            else:
                ls = ls.clone()
        max_sphere_idx = max(fa_sph_list + t0_sph_list)
        if max_sphere_idx >= ls.shape[1]:
            raise ValueError(
                f"sphere index {max_sphere_idx} outside FK sphere count {ls.shape[1]}")

        fa_sph = ls[:, fa_sph_list, :]
        t0_sph = ls[:, t0_sph_list, :]
        fa_xyz = fa_sph[:, :, None, :3]
        t0_xyz = t0_sph[:, None, :, :3]
        center_d = torch.norm(fa_xyz - t0_xyz, dim=3)
        surf_d = (
            center_d
            - fa_sph[:, :, None, 3]
            - t0_sph[:, None, :, 3]
        )
        sample_min_mm = surf_d.amin(dim=(1, 2)).detach().cpu().tolist()
        if len(sample_min_mm) != len(sample_indices):
            raise ValueError(
                f"clearance batch length {len(sample_min_mm)} "
                f"!= sample count {len(sample_indices)}")

        for pos, idx in enumerate(sample_indices):
            d_mm = float(sample_min_mm[pos]) * 1000.0

            if d_mm < min_dist_mm:
                min_dist_mm = d_mm
                worst_idx = idx
                pair_flat = surf_d[pos].reshape(-1).argmin().item()
                fa_pos, t0_pos = divmod(pair_flat, len(t0_sph_list))
                if fa_pos < len(fa_sph_list) and t0_pos < len(t0_sph_list):
                    worst_fa = fa_sph_list[fa_pos]
                    worst_t0 = t0_sph_list[t0_pos]

            if idx == sample_indices[-1]:
                destination_mm = d_mm

            if d_mm >= threshold_mm and not first_unsafe_found:
                safe_cutoff_idx = idx
            elif d_mm < threshold_mm and not first_unsafe_found:
                first_unsafe_found = True
                safe_cutoff_idx = sample_indices[pos - 1] if pos > 0 else 0

        tight_state = states[min(max(worst_idx, 0), len(states) - 1)]
        tight_joints = [round(float(v), 4) for v in tight_state]
        suffix = (
            f"at waypoint {worst_idx}/{len(states)} "
            f"joints={tight_joints} "
            f"(observed clamp <=41.8mm | threshold={threshold_mm:.0f}mm | "
            f"fa_sph={worst_fa}, t0_sph={worst_t0})"
        )
        if log_result:
            if min_dist_mm < danger_mm:
                node.get_logger().error(
                    f"[CLAMP] {label}: DANGER {min_dist_mm:.1f}mm — will trigger UR clamping error! {suffix}")
            elif min_dist_mm < threshold_mm:
                node.get_logger().warn(
                    f"[CLAMP] {label}: WARNING {min_dist_mm:.1f}mm — below safety threshold {suffix}")
            elif not getattr(node.cfg.planner, "concise_console_logs", False):
                node.get_logger().info(
                    f"[CLAMP] {label}: {min_dist_mm:.1f}mm OK {suffix}")
        return min_dist_mm, safe_cutoff_idx, destination_mm
    except Exception as e:
        if log_result:
            node.get_logger().error(
                f"[CLAMP] {label}: clearance check failed "
                f"({type(e).__name__}): {e}")
        return float('-inf'), 0, float('-inf')


def _select_safe_approach_candidate(node, approach_pose, start_js, goal_xyz=None):
    """Preflight several IK branches and return the lowest-cost safe one.

    When goal_xyz is given, each orientation candidate is also validated at the
    ALIGNMENT staging pose (the outer standoff point plan_and_execute() steps
    back to before the straight Cartesian entry -- see approach_alignment_extra_m
    below). Previously only the APPROACH pose itself was swept for clearance;
    ALIGNMENT was a separate point discovered only by actually attempting a full
    trajectory to it, so a clamp failure there wasted a full motion attempt and
    then a full goal retry with no way to pick a different orientation. Rejecting
    the same orientations here that would fail ALIGNMENT too means the caller can
    reuse the validated alignment_pose/alignment_goal_js directly.
    """
    if getattr(node, "_cuda_faulted", False):
        return None

    planner = node.cfg.planner
    threshold_mm = max(
        float(getattr(planner, "clamp_safety_threshold_mm", 45.0)),
        float(getattr(planner, "very_low_preflight_min_clearance_mm", 50.0)),
    )
    _alignment_enabled = bool(getattr(planner, "approach_alignment_enabled", True))
    _alignment_extra = max(0.0, float(getattr(planner, "approach_alignment_extra_m", 0.08)))
    _check_alignment = bool(_alignment_enabled and _alignment_extra > 0.005 and goal_xyz is not None)
    max_ratio = float(getattr(planner, "approach_max_path_ratio", 2.5))
    max_delta = math.radians(float(
        getattr(planner, "very_low_ik_max_joint_delta_deg", 75.0)))
    requested = max(1, int(getattr(planner, "very_low_ik_return_seeds", 8)))
    solver_seeds = int(getattr(node.motion_gen.ik_solver, "num_seeds", requested))
    return_seeds = min(requested, solver_seeds)
    # No sticky pitch: every goal starts from the configured preferred pitch and
    # re-sweeps -5 -> -10 -> -20 -> -30 from scratch, landing on the gentlest safe
    # negative tilt rather than inheriting a larger tilt that won on a previous goal.
    preferred_pitch = float(getattr(
        planner, "very_low_preflight_preferred_pitch_deg", -5.0))
    preferred_wrist = float(getattr(
        node, "_very_low_preferred_wrist_deg",
        getattr(planner, "very_low_preflight_preferred_wrist_deg", 0.0)))
    wrist_angles = [preferred_wrist, 0.0, -20.0, 20.0]
    wrist_angles = list(dict.fromkeys(wrist_angles))
    pitch_step = abs(float(getattr(
        planner, "very_low_preflight_pitch_step_deg", 10.0)))
    pitch_steps = max(0, int(getattr(
        planner, "very_low_preflight_pitch_steps", 3)))
    # Strict negative-pitch-first, gentlest tilt first: lead with the (negative)
    # preferred pitch, then sweep negatives by increasing magnitude (-10, -20, -30),
    # then 0, then positives as a last-resort tail. This stops at the smallest negative
    # pitch that clears the clearance bar instead of jumping straight to a large tilt.
    # A positive preferred is dropped from the seed so positive pitch can never lead.
    negative_offsets = [-pitch_step * i for i in range(1, pitch_steps + 1)]
    positive_offsets = [pitch_step * i for i in range(1, pitch_steps + 1)]
    seed_pitches = [preferred_pitch] if preferred_pitch <= 0.0 else []
    pitch_offsets = [*seed_pitches, *negative_offsets, 0.0, *positive_offsets]
    pitch_offsets = list(dict.fromkeys(pitch_offsets))
    early_accept_mm = max(threshold_mm, float(getattr(
        planner, "very_low_preflight_early_accept_mm", threshold_mm)))
    candidates = []
    best_observed_clearance = float('-inf')
    orientations_tested = 0
    early_stop = False
    # Diagnostic counters so a -inf failure says WHY: IK never solved this pose,
    # vs. IK solved but every branch needed a bigger reconfiguration than
    # max_delta allows, vs. branches were close enough but self-collision
    # (forearm/flange) clearance was too tight.
    ik_fail_count = 0
    delta_filtered_count = 0
    clearance_filtered_count = 0
    smallest_rejected_delta = float('inf')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = torch.tensor([start_js], dtype=torch.float32, device=device).unsqueeze(0)
    retract = torch.tensor([start_js], dtype=torch.float32, device=device)
    start_pose = node.get_end_effector_pose()

    # First sweep pitch with no wrist rotation. Pitch changes the actual
    # forearm/flange separation; wrist variants are a slower fallback.
    orientation_order = [(pitch_deg, preferred_wrist) for pitch_deg in pitch_offsets]
    orientation_order.extend(
        (pitch_deg, wrist_deg)
        for wrist_deg in wrist_angles
        if wrist_deg != preferred_wrist
        for pitch_deg in pitch_offsets
    )

    for pitch_deg, wrist_deg in orientation_order:
            pitched_quat = quat_apply_local_x_pitch(
                list(approach_pose[3:]), pitch_deg)
            orientations_tested += 1
            orientation_candidate_start = len(candidates)
            half = math.radians(wrist_deg) / 2.0
            q_wrist = [math.cos(half), 0.0, 0.0, math.sin(half)]
            quat = quat_normalize(quat_multiply(pitched_quat, q_wrist))
            pose = [*approach_pose[:3], *quat]
            pos_t = torch.tensor([pose[:3]], dtype=torch.float32, device=device)
            quat_t = torch.tensor([pose[3:]], dtype=torch.float32, device=device)

            try:
                result = node.motion_gen.ik_solver.solve_single(
                    Pose(position=pos_t, quaternion=quat_t),
                    seed_config=seed,
                    retract_config=retract,
                    return_seeds=return_seeds,
                )
            except Exception as exc:
                node.get_logger().warn(
                    f"[PREFLIGHT] pitch={pitch_deg:+.0f}deg "
                    f"wrist={wrist_deg:+.0f}deg IK exception: {exc}")
                continue

            solutions = result.js_solution.position
            success = result.success
            if solutions.ndim == 3:
                solutions = solutions[0]
            if success.ndim == 2:
                success = success[0]

            for branch_idx in range(min(len(solutions), len(success))):
                if not bool(success[branch_idx].item()):
                    ik_fail_count += 1
                    continue
                goal_js = nearest_joint_config(
                    start_js, solutions[branch_idx].detach().cpu().tolist())
                deltas = [abs(g - s) for g, s in zip(goal_js, start_js)]
                if max(deltas) > max_delta:
                    delta_filtered_count += 1
                    smallest_rejected_delta = min(
                        smallest_rejected_delta, math.degrees(max(deltas)))
                    continue

                states = []
                for idx in range(31):
                    alpha = idx / 30.0
                    smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
                    states.append([
                        s + smooth * (g - s) for s, g in zip(start_js, goal_js)
                    ])

                clearance, _, destination = _log_forearm_flange_clearance(
                    node, states, "PREFLIGHT", log_result=False)
                best_observed_clearance = max(
                    best_observed_clearance, min(clearance, destination))
                if clearance < threshold_mm or destination < threshold_mm:
                    clearance_filtered_count += 1
                    continue

                cart_path = forward_kinematics_batch(node, states)
                if len(cart_path) < 2:
                    continue
                path_len = sum(
                    math.dist(
                        (cart_path[i - 1].x, cart_path[i - 1].y, cart_path[i - 1].z),
                        (cart_path[i].x, cart_path[i].y, cart_path[i].z),
                    )
                    for i in range(1, len(cart_path))
                )
                if start_pose is not None:
                    straight = math.dist(start_pose[:3], pose[:3])
                else:
                    straight = math.dist(
                        (cart_path[0].x, cart_path[0].y, cart_path[0].z),
                        (cart_path[-1].x, cart_path[-1].y, cart_path[-1].z),
                    )
                ratio = path_len / max(straight, 0.001)
                if straight > 0.02 and ratio > max_ratio:
                    continue

                # Also validate the ALIGNMENT staging pose (same orientation, stepped
                # back approach_alignment_extra_m along goal-approach axis) at this
                # candidate before accepting it -- otherwise a good APPROACH orientation
                # can still fail the later real ALIGNMENT move with no way to recover.
                alignment_pose = None
                alignment_goal_js = None
                alignment_clearance = float('inf')
                if _check_alignment:
                    _toward = np.asarray(goal_xyz, dtype=float) - np.asarray(pose[:3], dtype=float)
                    _toward_norm = float(np.linalg.norm(_toward))
                    if _toward_norm > 0.020:
                        _align_xyz = (
                            np.asarray(pose[:3], dtype=float)
                            - _alignment_extra * (_toward / _toward_norm))
                        alignment_pose = [
                            float(_align_xyz[0]), float(_align_xyz[1]),
                            float(_align_xyz[2]), *quat]
                        _align_pos_t = torch.tensor(
                            [alignment_pose[:3]], dtype=torch.float32, device=device)
                        _align_quat_t = torch.tensor(
                            [alignment_pose[3:]], dtype=torch.float32, device=device)
                        try:
                            _align_result = node.motion_gen.ik_solver.solve_single(
                                Pose(position=_align_pos_t, quaternion=_align_quat_t),
                                seed_config=seed, retract_config=retract,
                                return_seeds=return_seeds)
                        except Exception:
                            continue
                        _align_solutions = _align_result.js_solution.position
                        _align_success = _align_result.success
                        if _align_solutions.ndim == 3:
                            _align_solutions = _align_solutions[0]
                        if _align_success.ndim == 2:
                            _align_success = _align_success[0]
                        if not bool(_align_success[0].item()):
                            continue
                        alignment_goal_js = nearest_joint_config(
                            start_js, _align_solutions[0].detach().cpu().tolist())
                        _align_deltas = [
                            abs(g - s) for g, s in zip(alignment_goal_js, start_js)]
                        if max(_align_deltas) > max_delta:
                            continue
                        _align_states = []
                        for idx in range(31):
                            alpha = idx / 30.0
                            smooth = (1.0 - math.cos(math.pi * alpha)) / 2.0
                            _align_states.append([
                                s + smooth * (g - s)
                                for s, g in zip(start_js, alignment_goal_js)
                            ])
                        alignment_clearance, _, _align_dest = _log_forearm_flange_clearance(
                            node, _align_states, "PREFLIGHT", log_result=False)
                        alignment_clearance = min(alignment_clearance, _align_dest)
                        if alignment_clearance < threshold_mm:
                            clearance_filtered_count += 1
                            continue

                joint_cost = sum(deltas) + 1.5 * max(deltas)
                angle_cost = math.radians(
                    abs(wrist_deg) + abs(pitch_deg)) * 0.15
                effective_clearance = min(clearance, alignment_clearance)
                clearance_bonus = min(effective_clearance, 100.0) * 0.001
                score = joint_cost + angle_cost - clearance_bonus
                candidates.append({
                    "score": score,
                    "pose": pose,
                    "goal_js": goal_js,
                    "pitch_deg": pitch_deg,
                    "wrist_deg": wrist_deg,
                    "clearance": effective_clearance,
                    "path_len": path_len,
                    "ratio": ratio,
                    "branch": branch_idx,
                    "alignment_pose": alignment_pose,
                    "alignment_goal_js": alignment_goal_js,
                })
                if effective_clearance >= early_accept_mm:
                    break

            new_candidates = candidates[orientation_candidate_start:]
            if (new_candidates and
                    max(item["clearance"] for item in new_candidates) >= early_accept_mm):
                early_stop = True
                break

    if not candidates:
        _set_goal_rejection(
            node,
            f"clamp clearance below {threshold_mm:.0f} mm",
            f"best={best_observed_clearance:.1f}mm")
        # best=-inf means clearance was never even computed for a single branch:
        # every candidate was rejected earlier, at the IK-solve or joint-delta
        # filter. Report which one so this isn't a guessing game next time.
        if best_observed_clearance == float('-inf'):
            _reject_reason = (
                f"IK never solved ({ik_fail_count} branches failed)"
                if delta_filtered_count == 0 else
                f"nearest reachable branch needed "
                f"{smallest_rejected_delta:.0f}deg reconfiguration "
                f"(cap {math.degrees(max_delta):.0f}deg); "
                f"{delta_filtered_count} branch(es) filtered by delta, "
                f"{ik_fail_count} failed IK outright")
        else:
            _reject_reason = (
                f"best branch clearance {best_observed_clearance:.1f}mm "
                f"below required {threshold_mm:.1f}mm "
                f"({clearance_filtered_count} branch(es) filtered by clearance)")
        node.get_logger().error(
            f"[PREFLIGHT] No safe IK branch found for VERY LOW approach "
            f"(best={best_observed_clearance:.1f}mm, required={threshold_mm:.1f}mm) — "
            f"{_reject_reason}")
        return None

    max_clearance = max(item["clearance"] for item in candidates)
    clearance_window = max(0.0, float(
        getattr(planner, "very_low_clearance_window_mm", 8.0)))
    safest_candidates = [
        item for item in candidates
        if item["clearance"] >= max_clearance - clearance_window
    ]
    best = min(safest_candidates, key=lambda item: item["score"])
    # Pitch is not sticky — each goal re-sweeps from the configured preferred pitch.
    node._very_low_preferred_wrist_deg = best["wrist_deg"]
    if not getattr(node.cfg.planner, "concise_console_logs", False):
        node.get_logger().info(
            f"[PREFLIGHT] selected pitch_offset={best['pitch_deg']:+.0f}deg "
            f"wrist={best['wrist_deg']:+.0f}deg "
            f"branch={best['branch']} clearance={best['clearance']:.1f}mm "
            f"path={best['path_len']*100:.1f}cm ratio={best['ratio']:.2f}x "
            f"from {len(candidates)} safe candidates "
            f"after {orientations_tested} orientations "
            f"(best clearance={max_clearance:.1f}mm)")
    return best


def straight_cartesian_entry_states(node, start_js, start_xyz, start_quat,
                                    target_pose_list, dt, label="ENTRY",
                                    duration_s=None):
    """Joint states for a straight-line Cartesian entry, for appending to a plan.

    Same construction _direct_ik_move() uses for its required-Cartesian leg:
    IK at evenly spaced points along the straight EE line, each seeded from the
    previous solution so the arm stays on one kinematic branch, orientation
    slerped so the wrist rotates gradually rather than snapping. The difference
    is that it starts from a GIVEN joint config (a planned trajectory's
    endpoint) rather than the arm's live pose, so it can be concatenated onto
    that plan and executed as one continuous motion.

    Density is chosen so the leg takes duration_s at the caller's dt. With a
    uniform dt, sample spacing IS speed -- so this keeps the near-fruit entry
    slow even when the staging leg it is appended to runs fast.

    Returns the states AFTER the junction (start_js is not repeated), or None
    if the straight corridor is not achievable.
    """
    planner = node.cfg.planner
    n_wp = max(1, int(getattr(planner, "direct_approach_cart_waypoints", 3)))
    if duration_s is None:
        duration_s = float(getattr(planner, "approach_entry_duration_s", 1.5))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_quat = list(target_pose_list[3:])

    wp_js = [list(start_js)]
    prev_js = list(start_js)
    for k in range(1, n_wp + 2):        # intermediate points, then the endpoint
        a = k / (n_wp + 1)
        wp_xyz = [start_xyz[i] + a * (target_pose_list[i] - start_xyz[i])
                  for i in range(3)]
        wp_quat = quat_slerp(list(start_quat), target_quat, min(a, 1.0))
        try:
            r = node.motion_gen.ik_solver.solve_single(
                Pose(position=torch.tensor([wp_xyz], dtype=torch.float32, device=dev),
                     quaternion=torch.tensor([wp_quat], dtype=torch.float32, device=dev)),
                seed_config=torch.tensor(
                    [prev_js], dtype=torch.float32, device=dev).unsqueeze(0),
                retract_config=torch.tensor([prev_js], dtype=torch.float32, device=dev))
        except Exception as exc:
            node.get_logger().warn(f"[ENTRY] {label}: IK exception at {k}: {exc}")
            return None
        if not bool(r.success.item()):
            node.get_logger().warn(
                f"[ENTRY] {label}: straight-line IK failed at point {k}/{n_wp + 1}")
            return None
        js = nearest_joint_config(
            prev_js, r.js_solution.position.squeeze().cpu().tolist())
        if max(abs(a2 - b2) for a2, b2 in zip(js, prev_js)) > 0.52:   # ~30deg
            node.get_logger().warn(
                f"[ENTRY] {label}: branch switch at point {k}; corridor unusable")
            return None
        wp_js.append(js)
        prev_js = js

    segments = len(wp_js) - 1
    total_steps = max(segments * 4, int(round(max(duration_s, 0.05) / max(dt, 1e-4))))
    steps_per = max(4, total_steps // segments)
    out = []
    for si in range(segments):
        s0, s1 = wp_js[si], wp_js[si + 1]
        for i in range(1, steps_per + 1):      # skip i=0: junction not repeated
            t = i / steps_per
            out.append([s0[j] + t * (s1[j] - s0[j]) for j in range(len(s0))])
    node.get_logger().info(
        f"[ENTRY] {label}: straight leg {len(wp_js)} IK waypoints -> "
        f"{len(out)} states, {len(out) * dt:.2f}s at dt={dt * 1000:.0f}ms")
    return out


def plan_and_send(node, start_state, goal_pose: Pose, label: str, motion_type: str = "default", goal_xyz: list = None, store_trajectory: bool = False, extend_fn=None) -> bool:
    """Plan to goal_pose and publish it as one trajectory.

    extend_fn, when given, is called as extend_fn(last_planned_js, dt) after
    planning and returns extra joint states to append into the SAME trajectory,
    so a follow-on leg runs continuously instead of being published separately
    (separate messages each begin and end at zero velocity, which forces a full
    stop between them). Returning an empty/None result rejects the whole move.
    """

    # 1) Plan with cuRobo — hold YOLO inference_lock so YOLO and cuRobo
    # never run CUDA kernels concurrently on the shared Jetson GPU.
    plan_cfg = PLAN_CFG_DEFAULT
    lock = getattr(node, '_planning_lock', None)
    _yolo = getattr(node, 'yolo_thread', None)
    _yolo_lock = getattr(_yolo, 'inference_lock', None)
    _plan_t0 = time.perf_counter()
    if _yolo_lock: _yolo_lock.acquire()
    if lock: lock.acquire()
    try:
        res = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
    finally:
        if lock: lock.release()
        if _yolo_lock: _yolo_lock.release()
    _planning_ms = (time.perf_counter() - _plan_t0) * 1000.0
    _record_motion = getattr(node, "_record_motion_plan_event", None)
    if not res.success:
        status = getattr(res, 'status', 'unknown')
        if callable(_record_motion):
            _record_motion(
                stage="PLAN", label=label, motion_type=motion_type,
                planner="curobo.plan_single", result="FAILED",
                status=str(status), planning_ms=round(_planning_ms, 1),
                goal_xyz_m=list(goal_xyz) if goal_xyz is not None else None)
        node.get_logger().warn(f"Plan failed for {label}. status={status}")
        # Flush any deferred async CUDA errors before the next planning call
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        return False

    if callable(_record_motion):
        _record_motion(
            stage="PLAN", label=label, motion_type=motion_type,
            planner="curobo.plan_single", result="SUCCESS",
            status=str(getattr(res, 'status', 'SUCCESS')),
            planning_ms=round(_planning_ms, 1),
            goal_xyz_m=list(goal_xyz) if goal_xyz is not None else None)

    states = interpolated_positions(res)
    curobo_dt = get_curobo_dt(res)

    def _max_joint_step(waypoints):
        if len(waypoints) < 2:
            return 0.0, -1, -1
        best = 0.0
        best_joint = -1
        best_idx = -1
        for i in range(len(waypoints) - 1):
            for j, (a, b) in enumerate(zip(waypoints[i], waypoints[i + 1])):
                d = abs(float(b) - float(a))
                if d > best:
                    best = d
                    best_joint = j
                    best_idx = i + 1
        return best, best_joint, best_idx

    def _replan_once(reason):
        if lock: lock.acquire()
        try:
            replanned = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
        finally:
            if lock: lock.release()
        if not replanned.success:
            node.get_logger().warn(f"Replan failed for {label} after {reason}")
            return None
        return replanned, interpolated_positions(replanned)

    # Safety: reject trajectories with joint wraparound (>300 deg on any joint), retry up to 3x
    for _plan_attempt in range(3):
        wraparound = False
        if len(states) > 1:
            for j in range(len(states[0])):
                total_rot = sum(abs(states[i+1][j] - states[i][j]) for i in range(len(states)-1))
                if total_rot > math.radians(300):
                    node.get_logger().warn(
                        f"Joint {j} wraparound: {math.degrees(total_rot):.0f}deg for {label} — replanning ({_plan_attempt+1}/3)"
                    )
                    wraparound = True
                    break
        if not wraparound:
            break
        replanned = _replan_once("joint wraparound")
        if replanned is None:
            return False
        res, replanned_states = replanned
        states = replanned_states
        curobo_dt = get_curobo_dt(res)
    else:
        node.get_logger().warn(f"All replans had joint wraparound for {label} — rejecting")
        return False

    if label.startswith("APPROACH"):
        _step, _joint, _idx = _max_joint_step(states)
        _limit = math.radians(float(getattr(node.cfg.planner, "approach_max_joint_step_deg", 12.0)))
        if _step > _limit:
            node.get_logger().warn(
                f"[JOINT] {label}: abrupt step {math.degrees(_step):.1f}deg "
                f"at waypoint {_idx} joint{_joint}; replanning")
            replanned = _replan_once("abrupt approach joint step")
            if replanned is not None:
                replanned_res, replanned_states = replanned
                _step2, _joint2, _idx2 = _max_joint_step(replanned_states)
                node.get_logger().info(
                    f"[JOINT] {label} replan: max_step={math.degrees(_step2):.1f}deg "
                    f"at waypoint {_idx2} joint{_joint2}")
                if _step2 < _step:
                    res = replanned_res
                    curobo_dt = get_curobo_dt(res)
                    states = replanned_states
                    _step, _joint, _idx = _step2, _joint2, _idx2
            if _step > _limit:
                node.get_logger().warn(
                    f"[JOINT] {label}: still snappy after replan "
                    f"({math.degrees(_step):.1f}deg > {math.degrees(_limit):.1f}deg); rejecting")
                return False
        else:
            node.get_logger().info(
                f"[JOINT] {label}: max_step={math.degrees(_step):.1f}deg "
                f"at waypoint {_idx} joint{_joint}")

    # Early return if trajectory is trivial (already at goal)
    if len(states) <= 2:
        node.get_logger().info(f"{label}: already at goal ({len(states)} waypoints), skipping motion")
        if store_trajectory:
            if not hasattr(node, 'stored_trajectory_states'):
                node.stored_trajectory_states = []
            node.stored_trajectory_states.extend(states)
        return True


    # 1c) Path-length sanity check using subsampled single-waypoint FK (batch=1, planner-safe)
    def _sampled_path_len(waypoints, step=20):
        """Compute approximate Cartesian path length by FK on every `step`-th waypoint."""
        indices = list(range(0, len(waypoints), step))
        if indices[-1] != len(waypoints) - 1:
            indices.append(len(waypoints) - 1)
        pts = []
        for idx in indices:
            p = forward_kinematics(node, waypoints[idx])
            if p is None:
                return None, None  # FK failed
            pts.append(p)
        total = 0.0
        for i in range(1, len(pts)):
            total += math.sqrt((pts[i].x - pts[i-1].x)**2 + (pts[i].y - pts[i-1].y)**2 + (pts[i].z - pts[i-1].z)**2)
        straight = math.sqrt((pts[-1].x - pts[0].x)**2 + (pts[-1].y - pts[0].y)**2 + (pts[-1].z - pts[0].z)**2)
        return total, straight

    _path_len_diag = None
    _straight_diag = None
    _path_ratio_diag = None
    if goal_xyz is not None and len(states) >= 2:
        path_len, straight = _sampled_path_len(states)
        if path_len is not None and straight is not None:
            ratio = path_len / max(straight, 0.001)
            node._last_plan_ratio = ratio  # expose to caller for retry decisions
            node.get_logger().info(
                f"[PATH] {label}: path_len={path_len*100:.1f}cm, straight={straight*100:.1f}cm, ratio={ratio:.1f}x"
            )
            # If path is roundabout (>1.8x straight line), try 1 more plan and pick shorter
            best_states = states
            best_path_len = path_len
            if ratio > 1.8 and straight > 0.02:
                if lock: lock.acquire()
                try:
                    res2 = node.motion_gen.plan_single(start_state, goal_pose, plan_cfg)
                finally:
                    if lock: lock.release()
                if res2.success:
                    s2 = interpolated_positions(res2)
                    skip = False
                    for j in range(len(s2[0])):
                        tot = sum(abs(s2[i+1][j] - s2[i][j]) for i in range(len(s2)-1))
                        if tot > math.radians(300):
                            skip = True
                            break
                    if not skip:
                        pl2, _ = _sampled_path_len(s2)
                        if pl2 is not None:
                            node.get_logger().info(
                                f"[PATH] {label} replan: path_len={pl2*100:.1f}cm ({pl2/max(straight,0.001):.1f}x)"
                            )
                            if pl2 < best_path_len:
                                best_path_len = pl2
                                best_states = s2
                if best_path_len < path_len:
                    node.get_logger().info(
                        f"[PATH] {label}: picked shorter path {best_path_len*100:.1f}cm (was {path_len*100:.1f}cm)"
                    )
            states = best_states
            final_ratio = best_path_len / max(straight, 0.001)
            _path_len_diag = best_path_len
            _straight_diag = straight
            _path_ratio_diag = final_ratio
            node._last_plan_ratio = final_ratio
            if label.startswith("APPROACH"):
                max_ratio = float(getattr(
                    node.cfg.planner, "approach_max_path_ratio", 2.5))
                if straight > 0.02 and final_ratio > max_ratio:
                    _set_goal_rejection(
                        node,
                        "approach path too roundabout",
                        f"{final_ratio:.1f}x > {max_ratio:.1f}x")
                    node.get_logger().error(
                        f"[PATH] {label}: rejecting roundabout approach "
                        f"({best_path_len*100:.1f}cm, {final_ratio:.1f}x > {max_ratio:.1f}x)")
                    node._approach_safety_rejected = True
                    return False

    # 1d) Batched FK + trim overshoot (single GPU call — done AFTER all replanning)
    cart_path = forward_kinematics_batch(node, states)

    if goal_xyz is not None and cart_path:
        # Find closest-to-goal waypoint in the last 40% of trajectory
        n = len(cart_path)
        search_start = max(0, int(n * 0.6))
        min_dist = float('inf')
        trim_idx = n - 1
        for i in range(search_start, n):
            p = cart_path[i]
            d = math.sqrt((p.x - goal_xyz[0])**2 + (p.y - goal_xyz[1])**2 + (p.z - goal_xyz[2])**2)
            if d < min_dist:
                min_dist = d
                trim_idx = i
        # Only trim if the closest point is actually near the goal (<5cm),
        # there are overshoot waypoints after it, and we retain enough waypoints
        original_len = len(states)
        min_keep = max(10, int(original_len * 0.2))  # keep at least 20% or 10 waypoints
        if min_dist < 0.05 and trim_idx < original_len - 2 and (trim_idx + 1) >= min_keep:
            states = states[:trim_idx + 1]
            cart_path = cart_path[:trim_idx + 1]
            node.get_logger().info(
                f"[TRIM] {label}: trimmed {original_len} → {len(states)} waypoints "
                f"(removed {original_len - len(states)} overshoot, closest dist={min_dist*100:.1f}cm)"
            )

    # 2) Speed scaling — computed here, BEFORE the optional continuation leg
    #    below, because that leg sizes its own waypoint density against this dt
    #    (uniform dt means density sets speed: denser samples = slower motion
    #    over the same distance).
    base_dt = getattr(node.cfg.planner, "base_dt", 0.02)   # e.g. 0.02 s
    planner = node.cfg.planner
    global_scale = max(getattr(node, "speed_scale", 1.0), 1e-6)
    if motion_type == "alignment":
        type_scale = getattr(planner, "speed_alignment", 1.0)
    elif motion_type == "approach":
        type_scale = getattr(planner, "speed_approach", 1.0)
    elif motion_type == "final":
        type_scale = getattr(planner, "speed_final", 1.0)
    elif motion_type in ["home", "dropoff", "predropoff"]:
        type_scale = getattr(planner, f"speed_{motion_type}", 1.0)
    else:
        type_scale = getattr(planner, "speed_home", 1.0)  # default
    scale = global_scale * type_scale
    # Never go faster than cuRobo's own interpolation timing: the positions were
    # planned for curobo_dt intervals; compressing them produces velocities and
    # accelerations that exceed what the robot can physically follow → jerks.
    # Scale > 1 ("go faster") has no effect — cuRobo already runs at max speed.
    raw_dt = base_dt / max(scale, 1e-6)
    dt = min(max(raw_dt, curobo_dt), planner.max_dt)
    base_vel = 0.08        # slightly gentler than 0.1
    vel = min(base_vel * scale, 0.25)   # hard cap for safety

    # 2b) Optional continuation leg, appended into THIS trajectory.
    #
    # build_trajectory() zeroes only the first and last waypoint, so anything
    # appended here flows through continuously -- no stop, no zero-velocity
    # handoff. Publishing the two legs as separate messages (the old behaviour)
    # structurally forced a full stop between them, since each message begins
    # and ends at rest. The extension is generated from the PLANNED endpoint
    # (states[-1]), not the arm's live pose, so the junction is continuous in
    # joint space. It runs after the overshoot trim above (so the trim still
    # keys off this leg's own goal_xyz) and before the clamp check below (so
    # clearance is verified across the whole merged path).
    if extend_fn is not None:
        try:
            _extra = extend_fn(list(states[-1]), float(dt))
        except Exception as _ex:
            node.get_logger().error(f"[{label}] continuation leg raised: {_ex}")
            return False
        if not _extra:
            node.get_logger().error(
                f"[{label}] continuation leg unavailable; rejecting rather than "
                "publishing a staging move that stops short of the target")
            return False
        states = list(states) + list(_extra)
        cart_path = forward_kinematics_batch(node, states)

    node._planned_cartesian_path = cart_path
    node._planned_label = label

    _clamp_mm, _safe_cutoff, _dest_mm = _log_forearm_flange_clearance(node, states, label)
    _clamp_threshold = float(getattr(node.cfg.planner, "clamp_safety_threshold_mm", 35.0))
    _is_approach = label.startswith("APPROACH")
    _is_final = label.startswith("FINAL")
    if _is_final and _clamp_mm < _clamp_threshold:
        _set_goal_rejection(
            node,
            f"clamp clearance below {_clamp_threshold:.0f} mm",
            f"min={_clamp_mm:.1f}mm")
        node.get_logger().error(
            f"[CLAMP] {label}: rejecting cuRobo FINAL "
            f"(min={_clamp_mm:.1f}mm, destination={_dest_mm:.1f}mm, "
            f"required={_clamp_threshold:.1f}mm)")
        return False
    if _is_approach and _clamp_mm < _clamp_threshold:
        _set_goal_rejection(
            node,
            f"clamp clearance below {_clamp_threshold:.0f} mm",
            f"min={_clamp_mm:.1f}mm")
        node.get_logger().error(
            f"[CLAMP] {label}: rejecting complete cuRobo APPROACH "
            f"(min={_clamp_mm:.1f}mm, destination={_dest_mm:.1f}mm, "
            f"required={_clamp_threshold:.1f}mm)")
        node._approach_clamp_rejected = True
        return False

    # 1d) Visualize planned path in RViz (blue line) — reuse pre-computed points
    publish_planned_path(node, states, label, cartesian_points=cart_path)

    # 4) Build trajectory — constant speed, except FINAL's tail below.
    # FINAL runs at dt=max_dt (12.5Hz) and hands off into GRASP/REVERSE, which
    # run at a much higher rate (REVERSE's own tail is already eased — see
    # execute_partial_reverse). Without easing, FINAL's own forced-zero-velocity
    # endpoint (build_trajectory always zeroes the last waypoint) is preceded by
    # waypoints still moving at the full approach rate -- an abrupt one-dt brake
    # right at the handoff. Ease the last few waypoints the same way REVERSE's
    # tail already is, so FINAL itself decelerates into that handoff.
    if motion_type == "final":
        _final_tail = max(4, int(getattr(planner, "final_decel_tail_points", 8)))
        states = ease_out_tail(states, _final_tail)
    # ALIGNMENT is typically the first real motion of a new goal, right after a
    # HOME/return move that runs at min_dt (100Hz, motions.py Formula A) via a
    # completely separate trajectory message. ALIGNMENT's own dt here is much
    # larger (speed_alignment, ~40ms/25Hz), and build_trajectory only forces
    # waypoint[0] to zero velocity -- waypoint[1] can already be at full cruising
    # speed, a one-dt jump from rest. Ease the head the same way FINAL's tail is
    # eased, so the first motion of a cycle also ramps up instead of snapping.
    elif motion_type == "alignment":
        _alignment_head = max(4, int(getattr(planner, "alignment_ease_head_points", 8)))
        states = ease_in_head(states, _alignment_head)
    traj = build_trajectory(
        node.joint_order,
        states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * planner.global_speed_multiplier,
        max_acc=planner.max_joint_acceleration,
        ramp_points=0,
    )

    if node.stop_requested:
        node.get_logger().warn(f"Stop before sending {label} trajectory.")
        node.stop_requested = False
        return False

    node.trajectory_pub.publish(traj)

    # Planning is finished and the arm is now executing, so the GPU is free.
    # Stream vision through the approach travel; reacquire then reads a
    # measurement that is already fresh on arrival instead of parking the arm
    # while it waits for one. lock_target() was called before the pause, so
    # vision stays on this target and cannot drift to a different "best".
    if (label.startswith("APPROACH")
            and bool(getattr(node.cfg.planner, "approach_vision_streaming", False))
            and hasattr(node, "set_vision_mode")):
        node.set_vision_mode("reacquire_fast")
        node.get_logger().info(
            "[APPROACH_VISION] streaming during travel (GPU free; planning done)")

    if callable(_record_motion):
        _record_motion(
            stage="EXECUTE", label=label, motion_type=motion_type,
            planner="curobo.plan_single", result="PUBLISHED",
            trajectory_samples=len(states), dt_s=round(float(dt), 5),
            trajectory_duration_s=round(float(dt) * max(len(states) - 1, 0), 3),
            path_length_m=(round(float(_path_len_diag), 4)
                           if _path_len_diag is not None else None),
            straight_distance_m=(round(float(_straight_diag), 4)
                                 if _straight_diag is not None else None),
            path_ratio=(round(float(_path_ratio_diag), 3)
                        if _path_ratio_diag is not None else None),
            clamp_clearance_mm=round(float(_clamp_mm), 1),
            clamp_required_mm=round(float(_clamp_threshold), 1),
            speed_scale=round(float(scale), 3),
            goal_xyz_m=list(goal_xyz) if goal_xyz is not None else None)

    # Store trajectory for later reversal if requested
    if store_trajectory:
        if not hasattr(node, 'stored_trajectory_states'):
            node.stored_trajectory_states = []
        node.stored_trajectory_states.extend(states)

    return True


def execute_reversed_trajectory(node, motion_type: str = "predropoff"):
    """
    Execute the stored trajectory in reverse order to return along the same collision-free path.
    """
    if not hasattr(node, 'stored_trajectory_states') or not node.stored_trajectory_states:
        node.get_logger().warn("No stored trajectory to reverse; falling back to normal motion.")
        return False

    # Reverse the stored states
    reversed_states = list(reversed(node.stored_trajectory_states))

    # Save for pre-planning dropoff from the end position
    node._last_reversed_states = reversed_states

    # Clear stored trajectory after use
    node.stored_trajectory_states = []

    planner = node.cfg.planner

    # First: very slow motion from current position to first reversed waypoint
    if node.current_joint_positions is not None:
        current_pos = list(node.current_joint_positions)
        first_waypoint = reversed_states[0]
        max_diff = max(abs(c - f) for c, f in zip(current_pos, first_waypoint))

        if max_diff > 0.005:  # Any significant difference - do slow initial motion
            node.get_logger().info(f"Slow initial motion to first waypoint (max_diff={max_diff:.4f} rad)")

            # Create slow trajectory from current to first waypoint
            initial_traj = build_trajectory(
                node.joint_order,
                [current_pos, first_waypoint],
                vel=0.02,   # Very slow
                dt=0.04,    # Long time steps
                stop_flag=lambda: node.stop_requested,
                max_vel=0.3,   # Very low max velocity
                max_acc=0.2,   # Very low acceleration
                ramp_points=0,
            )
            node.trajectory_pub.publish(initial_traj)

            # Wait for this slow motion to complete
            time.sleep(0.3)
            timeout_start = time.time()
            while time.time() - timeout_start < 10.0:
                if not is_robot_moving(node, velocity_threshold=0.005):
                    break
                time.sleep(0.05)
            time.sleep(0.1)  # Small settle time

    # Now execute the main reversed trajectory
    base_dt = getattr(planner, "base_dt", 0.02)
    scale = getattr(planner, "speed_predropoff", 1.0) * getattr(planner, "global_speed_multiplier", 1.0)

    dt = base_dt / max(scale, 1e-6)
    dt = min(max(dt, getattr(planner, "min_dt", 0.012)), getattr(planner, "max_dt", 0.03))

    base_vel = 0.08
    vel = min(base_vel * scale, getattr(planner, "max_traj_velocity", 0.25))

    traj = build_trajectory(
        node.joint_order,
        reversed_states,
        vel=vel,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=getattr(planner, "max_joint_velocity", 2.0) * getattr(planner, "global_speed_multiplier", 1.0),
        max_acc=getattr(planner, "max_joint_acceleration", 1.0),
        ramp_points=0,
    )

    if node.stop_requested:
        node.get_logger().warn("Stop before sending reversed trajectory.")
        node.stop_requested = False
        return False

    node.get_logger().info(f"Executing reversed trajectory with {len(reversed_states)} waypoints")
    node.trajectory_pub.publish(traj)
    _record_motion = getattr(node, "_record_motion_plan_event", None)
    if callable(_record_motion):
        _record_motion(
            stage="EXECUTE", label="REVERSE_FULL", motion_type=motion_type,
            planner="stored_trajectory_reverse", result="PUBLISHED",
            trajectory_samples=len(reversed_states), dt_s=round(float(dt), 5),
            trajectory_duration_s=round(
                float(dt) * max(len(reversed_states) - 1, 0), 3),
            speed_scale=round(float(scale), 3))
    return True


def _prepare_partial_reverse_states(node, clearance_m: float):
    """Select the reverse prefix and its predicted endpoint without moving."""
    stored = getattr(node, 'stored_trajectory_states', None)
    if not stored:
        return None, 0
    reversed_states = list(reversed(stored))

    # Use FK to find how many waypoints = clearance_m of Cartesian distance
    grasp_fk = forward_kinematics(node, reversed_states[0])
    if not grasp_fk:
        node.get_logger().warn("FK failed for partial reverse; using full reverse.")
        partial = reversed_states
    else:
        gx, gy, gz = grasp_fk.x, grasp_fk.y, grasp_fk.z
        partial = [reversed_states[0]]
        _clearance_reached = False
        reverse_cart_path = forward_kinematics_batch(node, reversed_states)
        for wp_idx, wp in enumerate(reversed_states[1:], start=1):
            partial.append(wp)
            fk = (
                reverse_cart_path[wp_idx]
                if wp_idx < len(reverse_cart_path)
                else forward_kinematics(node, wp)
            )
            if fk:
                dist = math.sqrt((fk.x - gx)**2 + (fk.y - gy)**2 + (fk.z - gz)**2)
                if dist >= clearance_m:
                    _clearance_reached = True
                    break
        # If all waypoints were consumed without reaching clearance, the reverse
        # goes all the way back to the approach start (home area). cuRobo's plan
        # can front-load wrist rotation at the beginning of the approach; reversed,
        # this becomes a concentrated wrist snap at the END of the reverse — the jerk.
        # In this case truncate the last 15% of waypoints (the jerky wrist portion)
        # and let cuRobo re-plan the remainder smoothly after the reverse.
        if not _clearance_reached:
            keep = max(10, int(len(partial) * 0.85))
            partial = partial[:keep]
            if getattr(node.cfg.planner, "log_phase_timings", False):
                node.get_logger().info(
                    f"Partial reverse: all waypoints consumed — truncating to {keep} "
                    f"(dropping last 15% to avoid wrist-snap near home)"
                )
    return partial, len(reversed_states)


def execute_partial_reverse(node, clearance_m: float = 0.35,
                            prepared_states=None,
                            original_state_count: int = 0):
    """
    Reverse only enough of the stored trajectory to pull back `clearance_m` from
    the grasp position. ``prepared_states`` allows drop-off planning to run from
    the predicted endpoint concurrently with this physical reverse.
    """
    if prepared_states is None:
        partial, original_state_count = _prepare_partial_reverse_states(
            node, clearance_m)
    else:
        partial = [list(q) for q in prepared_states]
    if not partial:
        node.get_logger().warn("No stored trajectory for partial reverse.")
        return False
    node.stored_trajectory_states = []

    # Wait for the arm to fully stop before publishing the reverse trajectory.
    # A fixed sleep isn't reliable — poll velocities until settled or timeout.
    _vel_tol = 0.05   # rad/s — joint considered stopped below this
    _settle_timeout = 1.0
    _t0 = time.time()
    while (time.time() - _t0) < _settle_timeout:
        vels = node.current_joint_velocities
        if vels is not None and all(abs(v) < _vel_tol for v in vels):
            break
        time.sleep(0.02)
    else:
        node.get_logger().warn("Partial reverse: arm did not settle within 1s — publishing anyway")

    # Prepend current position to bridge any gap (e.g. after depth correction nudge)
    # as part of the single reverse trajectory — no separate bridge + stop needed,
    # which avoids the stop→restart jerk at the transition.
    if node.current_joint_positions is not None:
        current_pos = list(node.current_joint_positions)
        max_diff = max(abs(c - f) for c, f in zip(current_pos, partial[0]))
        if max_diff > 0.002:
            if getattr(node.cfg.planner, "log_phase_timings", False):
                node.get_logger().info(
                    f"Partial reverse: prepending current pos ({max_diff*57.3:.2f}° gap)"
                )
            partial = [current_pos] + partial

    if getattr(node.cfg.planner, "log_phase_timings", False):
        node.get_logger().info(
            f"Partial reverse: {len(partial)}/{original_state_count} waypoints "
            f"({clearance_m*100:.0f}cm clearance)"
        )

    planner = node.cfg.planner
    base_dt = getattr(planner, "base_dt", 0.02)
    # Reverse uses a slow dedicated scale — do NOT use speed_predropoff (full speed).
    # global_speed_multiplier is intentionally NOT applied to max_vel here to avoid
    # 10 rad/s peaks that cause jerk at the start of the reverse motion.
    dt = getattr(planner, "min_dt", 0.012) * float(
        getattr(planner, "reverse_dt_multiplier", 1.7))

    # Decouple reverse duration from how densely the approach happened to be
    # sampled. This replays the STORED approach path sample-for-sample, so its
    # duration is len(partial)*dt -- meaning any change to approach sampling
    # silently rescales the reverse. That bit us directly: densifying the entry
    # leg to hold its own 1.5s (148 states over the same 80mm instead of ~62)
    # made the reverse *longer* even though dt had been halved. Resample to a
    # target wall-clock duration instead, preserving the path exactly (first and
    # last states are always kept) and only changing sample spacing.
    _target_s = float(getattr(planner, "reverse_duration_s", 0.0))
    if _target_s > 0.0 and len(partial) > 2:
        _want = max(8, int(round(_target_s / max(dt, 1e-4))))
        if _want < len(partial):
            _src = len(partial) - 1
            _idx = [round(i * _src / (_want - 1)) for i in range(_want)]
            _seen = set()
            _keep = [i for i in _idx if not (i in _seen or _seen.add(i))]
            if _keep[-1] != _src:
                _keep.append(_src)
            node.get_logger().info(
                f"[REVERSE] resampled {len(partial)} -> {len(_keep)} states "
                f"({len(partial) * dt:.2f}s -> {len(_keep) * dt:.2f}s at dt={dt * 1000:.0f}ms)")
            partial = [partial[i] for i in _keep]

    # Shape the final section into a real ease-out. Duplicate endpoint samples do
    # not decelerate: central differences merely become zero after the first
    # duplicate, leaving an abrupt one-dt velocity drop. ease_out_tail() remaps
    # progress along the existing final path with f(u)=u+u²-u³, whose slope
    # starts at 1 and reaches 0 at the endpoint (utils.py).
    DECEL_TAIL = max(
        8, int(getattr(planner, "reverse_decel_tail_points", 14)))
    partial = ease_out_tail(partial, DECEL_TAIL)
    # The eased tail already reaches zero slope. Do not append stationary
    # duplicate points: DROP-OFF is pre-planned concurrently and is dispatched
    # as soon as this endpoint is confirmed.

    reverse_velocity_scale = float(
        getattr(planner, "reverse_velocity_scale", 0.35))
    reverse_acceleration_scale = float(
        getattr(planner, "reverse_acceleration_scale", 0.35))
    traj = build_trajectory(
        node.joint_order,
        partial,
        dt=dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=getattr(planner, "max_joint_velocity", 2.0) * reverse_velocity_scale,
        max_acc=getattr(planner, "max_joint_acceleration", 1.0) * reverse_acceleration_scale,
        ramp_points=0,
    )

    node.trajectory_pub.publish(traj)
    _record_motion = getattr(node, "_record_motion_plan_event", None)
    if callable(_record_motion):
        _record_motion(
            stage="EXECUTE", label="REVERSE_PARTIAL",
            motion_type="reverse", planner="stored_trajectory_reverse",
            result="PUBLISHED", trajectory_samples=len(partial),
            dt_s=round(float(dt), 5),
            trajectory_duration_s=round(
                float(dt) * max(len(partial) - 1, 0), 3),
            requested_clearance_m=round(float(clearance_m), 3),
            velocity_scale=round(reverse_velocity_scale, 3),
            acceleration_scale=round(reverse_acceleration_scale, 3))

    # Wait for the commanded reverse endpoint, not merely for a moment of zero
    # velocity.  The controller may still be idle for a short time after publish;
    # the old velocity-only check could therefore return before motion even began
    # and let the "after reverse" camera capture duplicate the before image.
    final_reverse_joints = list(partial[-1])
    duration_msg = traj.points[-1].time_from_start
    expected_duration = (
        float(duration_msg.sec) + float(duration_msg.nanosec) * 1.0e-9)
    reverse_wait_timeout = max(2.0, expected_duration + 2.0)
    time.sleep(min(
        getattr(planner, "reverse_initial_wait", 0.15),
        max(0.02, expected_duration * 0.25),
    ))
    timeout_start = time.time()
    endpoint_reached = False
    closest_joint_error = float("inf")
    while time.time() - timeout_start < reverse_wait_timeout:
        current = node.current_joint_positions
        if current is not None and len(current) >= len(final_reverse_joints):
            joint_error = max(
                abs(float(c) - float(t))
                for c, t in zip(current, final_reverse_joints)
            )
            closest_joint_error = min(closest_joint_error, joint_error)
            if (
                joint_error <= 0.02
                and not is_robot_moving(node, velocity_threshold=0.005)
            ):
                endpoint_reached = True
                break
        time.sleep(0.05)
    if not endpoint_reached:
        node.get_logger().warn(
            "Partial reverse endpoint wait timed out "
            f"(closest joint error={closest_joint_error * 57.3:.2f}deg); "
            "after-reverse snapshot will be marked unreliable"
        )
    time.sleep(getattr(planner, "reverse_final_settle", 0.05))
    return endpoint_reached


def reacquire_goal_pose(node, seed_xyz, candidate_seeds=None, timeout=5.5, radius=0.04, z_tolerance=0.05, depth_settle_s=2.0, stable_needed=None, restore_mode="full", planner_busy=False):
    """Fast reacquire across multiple candidate seeds.

    Checks vision against all candidates. Returns first stable match.
    With multiple candidates, accepts after just 1 stable reading.

    Args:
        seed_xyz: primary seed [x,y,z]
        candidate_seeds: list of [x,y,z,...] alternate candidates (optional)
        timeout: max wait time including settle period
        depth_settle_s: how long to wait for depth to stabilise (reduce for small nudges)
        restore_mode: vision mode to restore on exit ("full" or "paused").
                      Pass "paused" for slip check so YOLO stays off during dropoff planning.
        planner_busy: True if cuRobo may be planning concurrently, in which case
                      vision keeps throttling YOLO to every 3rd frame to leave it
                      GPU headroom. False (default) when the arm is stopped and
                      nothing is planning, so YOLO runs every frame and reacquire
                      gets ~3x the samples inside the same timeout. Only the slip
                      check needs True: _bg_preplan_dropoff is still running then.
    """
    # Switch vision to lightweight mode: no heatmap, no trunk, no viz
    if hasattr(node, 'set_vision_mode'):
        node.set_vision_mode("reacquire" if planner_busy else "reacquire_fast")

    try:
        return _reacquire_goal_pose_impl(
            node, seed_xyz, candidate_seeds, timeout, radius, z_tolerance, depth_settle_s,
            stable_needed=stable_needed)
    finally:
        # Restore to requested mode (full for normal reacquire, paused for slip check)
        if hasattr(node, 'set_vision_mode'):
            node.set_vision_mode(restore_mode)
            if restore_mode == "paused":
                time.sleep(0.12)  # wait for any in-flight YOLO inference to finish


def _apply_final_inflight_correction(node, delta, report):
    """Retarget the FINAL move already in flight. Returns True if published.

    Regenerates the remaining straight leg from where the arm IS now to the
    corrected TCP pose and publishes it, replacing the active trajectory, then
    mutates the live wait target in place so wait_until_xyz follows.

    Nothing here stops the arm. The correction is bounded by the caller to
    [min_delta, max_delta], so the new straight line stays close to the one the
    corridor preflight already cleared.
    """
    planner = node.cfg.planner
    tcp = getattr(node, "_final_live_wait_xyz", None)
    quat = getattr(node, "_final_live_quat", None)
    if tcp is None or quat is None:
        return False
    js = node.current_joint_positions
    if js is None:
        return False
    cur = node.get_end_effector_pose()
    if not cur or len(cur) < 7:
        return False

    corrected = [float(tcp[i]) + float(delta[i]) for i in range(3)]
    ok, why = goal_is_in_robot_workspace(node, corrected)
    if not ok:
        node.get_logger().warn(f"[FINAL_INFLIGHT] retarget outside workspace: {why}")
        return False

    # Remaining distance decides how long the new leg should take, so the
    # correction does not change the approach speed near the fruit.
    _remaining = math.dist(cur[:3], corrected)
    _dt = float(min(max(planner.base_dt / max(
        planner.speed_scale * planner.speed_final, 1e-6),
        planner.min_dt), planner.max_dt))
    _dur = max(0.25, _remaining / max(0.024, 1e-6))   # ~2.4cm/s, the FINAL rate

    states = straight_cartesian_entry_states(
        node, list(js), list(cur[:3]), list(cur[3:7]),
        [*corrected, *quat], _dt, label="FINAL_RETARGET", duration_s=_dur)
    if not states:
        node.get_logger().warn(
            "[FINAL_INFLIGHT] retarget IK failed; keeping the original path")
        return False

    # Same clearance gate the original FINAL had to pass.
    _clamp_mm, _cut, _dest = _log_forearm_flange_clearance(
        node, states, "FINAL_RETARGET", log_result=False)
    _thr = float(getattr(planner, "clamp_safety_threshold_mm", 35.0))
    if _clamp_mm < _thr:
        node.get_logger().warn(
            f"[FINAL_INFLIGHT] retarget clearance {_clamp_mm:.1f}mm < {_thr:.0f}mm; "
            "keeping the original path")
        return False

    traj = build_trajectory(
        node.joint_order, states, vel=min(0.05 * 0.5, 0.15), dt=_dt,
        stop_flag=lambda: node.stop_requested,
        max_vel=planner.max_joint_velocity * 0.5,
        max_acc=planner.max_joint_acceleration * 0.3,
        ramp_points=0, include_acc=False)
    node.trajectory_pub.publish(traj)
    # In place: wait_until_xyz is holding a reference to this same list.
    tcp[0], tcp[1], tcp[2] = corrected
    # And the endpoint check must judge against this, not the original target.
    node._final_retarget_xyz = list(corrected)
    report["applied"] += 1
    node.get_logger().info(
        f"[FINAL_INFLIGHT] retargeted "
        f"{math.dist([0,0,0], delta)*1000:.1f}mm -> "
        f"[{corrected[0]:.3f},{corrected[1]:.3f},{corrected[2]:.3f}] "
        f"({_remaining*1000:.0f}mm remaining, clearance {_clamp_mm:.0f}mm)")
    return True


def _final_inflight_monitor(node, target_xyz, stop_evt, report):
    """Watch vision during the FINAL move. Observation-only.

    Runs on its own thread while the arm covers the last few centimetres. Polls
    the same quality-gated candidate list reacquire uses, and records how far
    the best match sits from the target the arm is currently driving to, plus
    how far through the move that happened.

    This is the measurement nobody has: the camera has never been on during the
    FINAL leg, so it is unknown whether the date stays detectable as the
    off-axis camera converges on it and the fingers enter frame. `report`
    accumulates the answer; nothing here commands motion or alters the target.
    """
    planner = node.cfg.planner
    min_d = float(getattr(planner, "final_inflight_min_delta_m", 0.002))
    max_d = float(getattr(planner, "final_inflight_max_delta_m", 0.025))
    t0 = time.time()
    seen_seq = -1
    while not stop_evt.is_set():
        time.sleep(0.05)
        seq = int(getattr(node, "all_fruits_seq", 0) or 0)
        if seq == seen_seq:
            continue
        seen_seq = seq
        report["frames"] += 1
        best, best_d = None, float("inf")
        for cand in (getattr(node, "all_fruit_poses", None) or []):
            if len(cand) < 3:
                continue
            ok, why = _reacquire_quality_ok(node, cand)
            if not ok:
                report["rejected"] += 1
                report["last_reject"] = why
                continue
            d = math.dist(cand[:3], target_xyz[:3])
            if d < best_d:
                best, best_d = list(cand[:3]), d
        if best is None:
            report["no_candidate"] += 1
            continue
        report["matched"] += 1
        if best_d < min_d:
            report["below_min"] += 1
            continue
        if best_d > max_d:
            report["above_max"] += 1
            continue
        # A usable correction.
        report["usable"] += 1
        report["latest"] = best
        report["latest_d"] = best_d
        report["latest_t"] = time.time() - t0

        if not bool(getattr(planner, "final_inflight_apply", False)):
            continue
        # Bounded: a capped number of retargets, spaced out, and the running
        # total held under max_d so repeated small nudges cannot walk the goal
        # somewhere the corridor preflight never cleared.
        if report["applied"] >= int(getattr(
                planner, "final_inflight_max_updates", 2)):
            continue
        if (time.time() - report["last_apply_t"]
                < float(getattr(planner, "final_inflight_min_interval_s", 0.25))):
            continue
        delta = [best[i] - target_xyz[i] for i in range(3)]
        if math.dist([0.0, 0.0, 0.0], delta) + report["applied_total"] > max_d:
            report["above_max"] += 1
            continue
        if _apply_final_inflight_correction(node, delta, report):
            report["last_apply_t"] = time.time()
            report["applied_total"] += math.dist([0.0, 0.0, 0.0], delta)
            # Corrections are measured against the goal we are now driving to.
            target_xyz = [target_xyz[i] + delta[i] for i in range(3)]


def _reacquire_candidate_quality(node, candidate):
    """Quality for the fruit at `candidate`, or None if unknown.

    /vision/all_fruit_quality is published in the same score-sorted order as
    /vision/all_fruit_poses, but the two arrive as separate messages and
    node.latest_goal_pose comes from a third topic on its own timer. Rather
    than trust index alignment, match by position: find the nearest published
    fruit and use its quality. Returns None when nothing is close enough,
    which callers must treat as "unknown", not "bad".
    """
    quality = getattr(node, "all_fruit_quality", None)
    poses = getattr(node, "all_fruit_poses", None)
    if not quality or not poses:
        return None
    best_i, best_d = None, float("inf")
    for i, p in enumerate(poses):
        if i >= len(quality):
            break
        d = math.dist(candidate, p)
        if d < best_d:
            best_i, best_d = i, d
    # 2cm: same fruit seen by both topics, not a neighbour.
    if best_i is None or best_d > 0.02:
        return None
    return quality[best_i]


def _reacquire_quality_ok(node, candidate):
    """(ok, reason). Unknown quality passes -- never block on missing data."""
    planner = node.cfg.planner
    if not bool(getattr(planner, "reacquire_quality_gate", True)):
        return True, ""
    q = _reacquire_candidate_quality(node, candidate)
    if q is None:
        return True, ""
    vis_ratio, z_std, confidence, _score, edge_margin = q
    min_vis = float(getattr(planner, "reacquire_min_vis_ratio", 0.25))
    max_std = float(getattr(planner, "reacquire_max_z_std_m", 0.025))
    min_conf = float(getattr(planner, "reacquire_min_confidence", 0.20))
    min_edge = float(getattr(planner, "reacquire_min_edge_margin", 0.0))
    if vis_ratio < min_vis:
        return False, f"vis_ratio {vis_ratio:.2f}<{min_vis:.2f}"
    if z_std > max_std:
        return False, f"z_std {z_std*1000:.0f}mm>{max_std*1000:.0f}mm"
    if confidence < min_conf:
        return False, f"conf {confidence:.2f}<{min_conf:.2f}"
    if edge_margin < min_edge:
        return False, f"bbox clipped (edge {edge_margin:.3f}<{min_edge:.3f})"
    return True, ""


def _reacquire_goal_pose_impl(node, seed_xyz, candidate_seeds=None, timeout=5.5, radius=0.04, z_tolerance=0.05, depth_settle_s=2.0, stable_needed=None):
    import numpy as _np
    _verbose_reacq = getattr(node.cfg.planner, "log_phase_timings", False)
    # Build seed list: primary first, then candidates
    seeds = [seed_xyz[:3]]
    if candidate_seeds:
        for c in candidate_seeds:
            xyz = c[:3] if len(c) > 3 else c
            if all(math.dist(xyz, s) > 0.03 for s in seeds):
                seeds.append(list(xyz))

    if stable_needed is None:
        stable_needed = 5      # readings that must match before accepting
    Z_STABLE_THRESH = 0.012    # max std of Z across stable readings (12 mm)

    stable_count = 0
    matched_seed = None
    start = time.time()
    prev_pose_tuple = None
    recent_zs = []             # Z values of the last stable_needed matched readings

    # Let depth settle before starting to match — ZED stereo depth needs a few
    # frames to stabilize after the arm stops moving at the approach position.
    DEPTH_SETTLE_S = depth_settle_s
    search_s = max(timeout - DEPTH_SETTLE_S, 1.0)
    if _verbose_reacq:
        node.get_logger().info(
            f"[REACQ] Settling {DEPTH_SETTLE_S:.1f}s | "
            f"seeds={len(seeds)} primary=[{seeds[0][0]:.3f},{seeds[0][1]:.3f},{seeds[0][2]:.3f}] | "
            f"search budget={search_s:.1f}s")
    while time.time() - start < DEPTH_SETTLE_S:
        time.sleep(0.05)
    if _verbose_reacq:
        node.get_logger().info(f"[REACQ] Settle done — searching ({search_s:.1f}s remaining)")

    _last_progress_log = 0.0   # throttle per-frame progress to once/sec

    # Perception sample in effect when reacquire started. Vision is PAUSED
    # during the approach, so node.latest_goal_pose still holds the very sample
    # that produced the seed. Matching against it "succeeds" instantly with
    # drift=(0,0,0) and adds no information -- observed in the field as
    # "ACCEPTED (full 3D) after 0.0s | drift=(0,-0,0)mm". Refuse to match until
    # vision has published a genuinely new sample. If none arrives we time out
    # and fall back to the seed, which is the correct outcome: reacquire has
    # nothing to say, and must not pretend otherwise.
    # Watch BOTH the goal stamp and the all-fruit-poses counter. The goal topic
    # alone was never a workable signal here: it is published only from
    # _process_best_target, which the vision node skips in reacquire mode, so it
    # is silent for the entire duration of this function and this gate always
    # expired -- burning its full _fresh_wait_s every cycle and then matching
    # "the latest available" anyway, which is exactly what it existed to prevent.
    # all_fruits_seq increments on every processed frame in every mode.
    _entry_sample_ns = int(getattr(node, "latest_goal_sample_ns", 0) or 0)
    _entry_fruits_seq = int(getattr(node, "all_fruits_seq", 0) or 0)
    _fresh_seen = False
    _fresh_gate_expired = False
    # Bound on how long the freshness gate may hold us up.
    _fresh_wait_s = min(0.6, max(0.0, float(timeout) * 0.4))

    _grace_extended = False
    # Absolute ceiling. `timeout` is extended by the grace period below, so it is
    # not by itself a guarantee of progress. This one is never extended:
    # reacquire is an optimisation and must never be able to stall a harvest.
    _hard_deadline = start + float(timeout) + 2.0
    while time.time() - start < timeout and time.time() < _hard_deadline:
        # Grace period: if we have at least one reading and are about to time
        # out, extend by 1s to wait for the next one.
        #
        # This check MUST run before the `continue` paths below. It used to sit
        # after them, so it was only ever evaluated on an iteration where a new
        # distinct matching pose arrived -- which is exactly the thing that is
        # not happening when we are stuck at stable=1. Observed in the field:
        # "TIMEOUT after 0.9s | stable=1/2" with the fruit 4mm from the seed,
        # the grace never having fired.
        if (not _grace_extended and stable_count >= 1
                and time.time() - start >= timeout - 0.05):
            timeout += 1.0
            _grace_extended = True
            if _verbose_reacq:
                node.get_logger().info(
                    f"[REACQ] Grace +1s: stable={stable_count}/{stable_needed} "
                    "— waiting for one more reading")

        # Freshness gate, BOUNDED. Its job is to stop us instantly matching the
        # very sample that produced the seed (vision is paused during approach,
        # so latest_goal_pose still holds it) -- not to block indefinitely. In
        # throttled "reacquire" mode the vision node runs YOLO every 3rd frame and
        # has just come back from a pause, so a genuinely new sample can take a
        # while. After _fresh_wait_s, proceed with whatever is published and say
        # so, rather than burning the whole budget waiting.
        if (not _fresh_seen
                and (int(getattr(node, "latest_goal_sample_ns", 0) or 0)
                     != _entry_sample_ns
                     or int(getattr(node, "all_fruits_seq", 0) or 0)
                     != _entry_fruits_seq)):
            _fresh_seen = True
            if _verbose_reacq:
                node.get_logger().info(
                    f"[REACQ] fresh perception sample after "
                    f"{time.time() - start - DEPTH_SETTLE_S:.2f}s")
        if not _fresh_seen:
            if time.time() - start < _fresh_wait_s:
                time.sleep(0.005)
                continue
            if not _fresh_gate_expired:
                _fresh_gate_expired = True
                node.get_logger().warn(
                    f"[REACQ] no new perception sample within "
                    f"{_fresh_wait_s:.1f}s (no detections, or vision throttled); "
                    "matching against the latest available")

        # Check best fruit first, then fall back to all visible fruits.
        # This allows reacquire to find the target even when it isn't best-ranked
        # (e.g. another fruit is closer during the approach phase).
        #
        # BUT only if latest_goal_pose has actually refreshed since we entered.
        # In reacquire mode the vision node skips _process_best_target, so
        # /external_goal_pose is never republished, and the publish timer drops
        # the goal after 0.25s of staleness. latest_goal_pose therefore still
        # holds the very sample that produced the seed. Listing it first meant
        # reacquire matched the seed against itself and "succeeded" with
        # drift=(0,0,0)mm, learning nothing -- the exact outcome the freshness
        # gate above was written to prevent. Making the gate work was not enough
        # while the stale value stayed in the candidate list.
        #
        # all_fruit_poses is published every processed frame in every mode, so
        # it carries the genuinely new measurements.
        pose = None
        _goal_pose_is_fresh = (
            int(getattr(node, "latest_goal_sample_ns", 0) or 0) != _entry_sample_ns)
        if node.latest_goal_pose and _goal_pose_is_fresh:
            pose = node.latest_goal_pose[:3]

        # If best fruit doesn't match any seed, scan all visible fruits
        candidate_poses = [pose] if pose is not None else []
        all_visible = getattr(node, 'all_fruit_poses', [])
        for fp in all_visible:
            if not any(fp == p for p in candidate_poses):
                candidate_poses.append(fp)

        best_seed = None
        best_dist = float('inf')
        pose = None
        _used_xy_only = False
        _closest_miss = None   # for diagnostics
        _closest_miss_d = float('inf')
        for candidate in candidate_poses:
            if candidate is None:
                continue
            # Quality gate first: a detection can sit well inside the match
            # radius and still be junk (half the date out of frame, depth
            # scattered across the bunch behind it). Accepting it moves the
            # grasp goal onto a bad measurement, which is worse than timing out
            # and keeping the seed.
            _q_ok, _q_reason = _reacquire_quality_ok(node, candidate)
            if not _q_ok:
                _now_q = time.time()
                if (not hasattr(_reacquire_goal_pose_impl, "_last_q_log")
                        or _now_q - _reacquire_goal_pose_impl._last_q_log > 1.0):
                    _reacquire_goal_pose_impl._last_q_log = _now_q
                    node.get_logger().info(
                        f"[REACQ] rejected on quality: {_q_reason} "
                        f"at [{candidate[0]:.3f},{candidate[1]:.3f},{candidate[2]:.3f}]")
                continue
            cx, cy, cz = candidate
            for s in seeds:
                d_xy = math.hypot(cx - s[0], cy - s[1])
                d_z = abs(cz - s[2])
                d = math.dist(candidate, s)
                # XY-only match: only when we have enough readings AND they are
                # genuinely noisy (high Z std). Do NOT open this gate early just
                # because stable_count is low — that allows neighbouring fruits
                # at 4-5cm to be accepted before we have evidence of depth noise.
                depth_unstable = (
                    len(recent_zs) >= stable_needed and
                    float(_np.std(recent_zs)) > Z_STABLE_THRESH
                )
                xy_only_match = depth_unstable and d_xy <= 0.05
                full_match = d_xy <= radius and d_z <= z_tolerance
                # One-dim-close: if either dxy<2cm or dz<2cm the fruit is very close in that
                # dimension — accept it but fall back to original seed xyz (safest position).
                one_dim_close = (d_xy < 0.02 or d_z < 0.02) and (d_xy < radius or d_z < z_tolerance)
                if (full_match or xy_only_match or one_dim_close) and d_xy < best_dist:
                    best_dist = d_xy
                    best_seed = s
                    _used_xy_only = xy_only_match and not full_match
                    _match_type = "one-dim-close" if (one_dim_close and not full_match and not xy_only_match) else ("XY-only" if _used_xy_only else "full")
                    node.get_logger().debug(
                        f"[REACQ] match: [{cx:.3f},{cy:.3f},{cz:.3f}] → seed [{s[0]:.3f},{s[1]:.3f},{s[2]:.3f}] "
                        f"dxy={d_xy*100:.1f}cm dz={d_z*100:.1f}cm ({_match_type})")
                    if one_dim_close and not full_match and not xy_only_match:
                        # Use whichever dimension is close from candidate, keep seed for the other
                        nx = cx if d_xy < 0.02 else s[0]
                        ny = cy if d_xy < 0.02 else s[1]
                        nz = cz if d_z < 0.02 else s[2]
                        pose = [nx, ny, nz]
                    elif _used_xy_only:
                        # Use detected XY but keep seed Z — depth unreliable
                        pose = [cx, cy, s[2]]
                    else:
                        pose = candidate
                elif d < _closest_miss_d:
                    _closest_miss_d = d
                    _closest_miss = (candidate, s, d_xy, d_z)

        if best_seed is None or pose is None:
            # Log once per second so we can see why reacquire keeps missing
            _now = time.time()
            if (_verbose_reacq and
                    (not hasattr(_reacquire_goal_pose_impl, '_last_miss_log') or
                     _now - _reacquire_goal_pose_impl._last_miss_log > 1.0)):
                _reacquire_goal_pose_impl._last_miss_log = _now
                if _closest_miss is not None:
                    _cm, _cs, _dxy, _dz = _closest_miss
                    node.get_logger().warn(
                        f"[REACQ] No match — closest candidate "
                        f"[{_cm[0]:.3f},{_cm[1]:.3f},{_cm[2]:.3f}] "
                        f"seed [{_cs[0]:.3f},{_cs[1]:.3f},{_cs[2]:.3f}] "
                        f"d_xy={_dxy*100:.1f}cm d_z={_dz*100:.1f}cm "
                        f"(limit: xy={radius*100:.0f}cm z={z_tolerance*100:.0f}cm)"
                    )
                else:
                    node.get_logger().warn("[REACQ] No candidates visible at all")
            time.sleep(0.001)
            continue

        pose_tuple = tuple(pose)
        if pose_tuple == prev_pose_tuple:
            time.sleep(0.001)
            continue
        prev_pose_tuple = pose_tuple

        x, y, z = pose

        # Stability check: count how many distinct readings land near the seed.
        # Depth noise is typically 5-20mm so a tight consecutive-delta gate (old 3mm)
        # would reset on every noisy frame. Instead just count distinct detections.
        if matched_seed != best_seed:
            stable_count = 0
            matched_seed = best_seed
            recent_zs.clear()

        stable_count += 1
        recent_zs.append(z)
        if len(recent_zs) > stable_needed:
            recent_zs.pop(0)

        z_std = float(_np.std(recent_zs)) if len(recent_zs) >= 2 else float('nan')

        # Throttled progress log — once per second
        _now = time.time()
        if _verbose_reacq and _now - _last_progress_log >= 1.0:
            _last_progress_log = _now
            elapsed = _now - start - DEPTH_SETTLE_S
            node.get_logger().info(
                f"[REACQ] t={elapsed:.1f}s | stable={stable_count}/{stable_needed} "
                f"Z_std={z_std*1000:.1f}mm | pos=[{x:.3f},{y:.3f},{z:.3f}] | "
                f"visible={len(candidate_poses)} fruits")

        if stable_count >= stable_needed:
            if z_std > Z_STABLE_THRESH:
                # Depth still fluctuating — keep collecting, don't reset count
                if _verbose_reacq:
                    node.get_logger().info(
                        f"[REACQ] Waiting for depth: Z_std={z_std*1000:.1f}mm > {Z_STABLE_THRESH*1000:.0f}mm "
                        f"(stable_count={stable_count})")
                time.sleep(0.001)
                continue
            match_type = "XY-only (depth unstable)" if _used_xy_only else "full 3D"
            dx = (x - matched_seed[0]) * 1000
            dy = (y - matched_seed[1]) * 1000
            dz = (z - matched_seed[2]) * 1000
            drift_xy = math.hypot(dx, dy)
            # With stable_needed=1 there is only one reading, so no Z spread
            # exists to report. Print n/a rather than "nan".
            _zstd_txt = (
                "n/a" if z_std != z_std else f"{z_std*1000:.1f}mm")
            _accept_msg = (
                f"[REACQ] ACCEPTED ({match_type}) after {time.time()-start-DEPTH_SETTLE_S:.1f}s | "
                f"count={stable_count} Z_std={_zstd_txt} | "
                f"[{x:.3f},{y:.3f},{z:.3f}] | seed=[{matched_seed[0]:.3f},{matched_seed[1]:.3f},{matched_seed[2]:.3f}] | "
                f"drift=({dx:.0f},{dy:.0f},{dz:.0f})mm XY={drift_xy:.0f}mm"
            )
            if drift_xy > 25:
                node.get_logger().warn(_accept_msg + " ← LARGE DRIFT, verify correct fruit")
            else:
                node.get_logger().info(_accept_msg)
            return (x, y, z)

        time.sleep(0.001)

    # Timeout — soft-accept if we have ≥3 stable readings with good Z
    elapsed = time.time() - start - DEPTH_SETTLE_S
    if stable_count >= 3 and matched_seed is not None and len(recent_zs) >= 2:
        z_std = float(_np.std(recent_zs))
        if z_std <= Z_STABLE_THRESH and prev_pose_tuple is not None:
            x, y, z = prev_pose_tuple
            dx = (x - matched_seed[0]) * 1000
            dy = (y - matched_seed[1]) * 1000
            dz = (z - matched_seed[2]) * 1000
            drift_xy = math.hypot(dx, dy)
            node.get_logger().warn(
                f"[REACQ] SOFT-ACCEPT at timeout | stable={stable_count}/{stable_needed} "
                f"Z_std={z_std*1000:.1f}mm | [{x:.3f},{y:.3f},{z:.3f}] | "
                f"drift=({dx:.0f},{dy:.0f},{dz:.0f})mm XY={drift_xy:.0f}mm")
            return (x, y, z)

    # Hard timeout — nothing usable
    if _closest_miss is not None:
        _cm, _cs, _dxy, _dz = _closest_miss
        node.get_logger().warn(
            f"[REACQ] TIMEOUT after {elapsed:.1f}s | stable={stable_count}/{stable_needed} | "
            f"closest was [{_cm[0]:.3f},{_cm[1]:.3f},{_cm[2]:.3f}] "
            f"dxy={_dxy*100:.1f}cm dz={_dz*100:.1f}cm (need xy<{radius*100:.0f}cm z<{z_tolerance*100:.0f}cm)")
    else:
        node.get_logger().warn(
            f"[REACQ] TIMEOUT after {elapsed:.1f}s | stable={stable_count}/{stable_needed} | "
            f"no fruits visible in {len(seeds)} seed windows")
    return None



def subscribe_to_goal_pose(node):
    """Subscribe to /external_goal_pose and accept one stable goal.

    Multi-goal queuing is handled only by subscribe_multi_goals().
    """

    # Wait until robot stops before subscribing
    if is_robot_moving(node):
        _wait_timer = [None]
        def _wait_and_subscribe():
            if not is_robot_moving(node):
                if _wait_timer[0] is not None:
                    _wait_timer[0].cancel()
                    _wait_timer[0] = None
                subscribe_to_goal_pose(node)
        _wait_timer[0] = node.create_timer(0.5, _wait_and_subscribe)
        return

    # Cancel any existing idle timer first (prevents multiple timers from stacking)
    if hasattr(node, 'idle_timer'):
        try:
            node.idle_timer.cancel()
            del node.idle_timer
        except Exception:
            pass

    # Reset all tracking state for fresh cycle
    node.reset_goal_tracking()

    # Invalidate cached goal - always wait for fresh pose from vision
    node.latest_goal_pose = None
    node.latest_goal_time = 0

    # Destroy previous subscription if it exists
    if hasattr(node, 'goal_pose_sub'):
        node.destroy_subscription(node.goal_pose_sub)
        del node.goal_pose_sub

    cur = node.get_end_effector_pose()
    current_orientation = cur[3:] if cur else [1.0, 0.0, 0.0, 0.0]
    node.goal_poses.clear()
    getattr(node, "_reachability_goal_metadata", {}).clear()
    node.candidate_goals = []  # list of [x,y,z,qw,qx,qy,qz]

    # Track position stability before accepting
    goal_history = {
        "poses": [], "stable_count": 0,
        "accepted": False, "accept_time": 0.0,
        "start_time": time.time(), "latest_quat": current_orientation,
        "last_stamp_ns": None,
    }

    def _destroy_sub():
        try:
            if hasattr(node, 'goal_pose_sub'):
                node.destroy_subscription(node.goal_pose_sub)
                del node.goal_pose_sub
        except Exception:
            pass

    def _goal_cb(msg: PoseStamped):
        nonlocal goal_history

        # /external_goal_pose is published faster than perception runs.  Count
        # each stamped perception sample once; repeated timer publications of
        # the same sample must not satisfy the stability requirement.
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if stamp_ns != 0:
            if stamp_ns == goal_history["last_stamp_ns"]:
                return
            goal_history["last_stamp_ns"] = stamp_ns

        new_xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        new_quat = [msg.pose.orientation.w, msg.pose.orientation.x,
                     msg.pose.orientation.y, msg.pose.orientation.z]
        _workspace_ok, _workspace_reason = goal_is_in_robot_workspace(node, new_xyz)
        if not _workspace_ok:
            _warn_rejected_goal_throttled(node, new_xyz, _workspace_reason)
            return

        # ---- ALWAYS STORE LATEST GOAL POSE ----
        node.latest_goal_pose = [*new_xyz, *new_quat]
        node.latest_goal_time = time.time()

        # Phase 1: waiting for first stable goal
        if goal_history["accepted"]:
            return

        # Check for position jump (vision switched to new fruit)
        if goal_history["poses"]:
            last_xyz = goal_history["poses"][-1]
            jump = math.dist(new_xyz, last_xyz)
            if jump > 0.10:  # >10cm = new fruit, reset
                goal_history["poses"].clear()
                goal_history["stable_count"] = 0

        goal_history["poses"].append(new_xyz)
        goal_history["latest_quat"] = new_quat

        min_settle_s = max(0.0, float(getattr(node.cfg.planner, "subscribe_goal_min_settle_s", 0.40)))
        max_wait_s = max(min_settle_s, float(getattr(node.cfg.planner, "subscribe_goal_max_wait_s", 1.20)))
        stable_tol = float(getattr(node.cfg.planner, "subscribe_goal_stable_tol", 0.020))
        median_window = max(2, int(getattr(node.cfg.planner, "subscribe_goal_median_window", 5)))
        elapsed = time.time() - goal_history["start_time"]
        stable_pair = False

        # Need at least 2 consistent readings before accepting
        if len(goal_history["poses"]) >= 2:
            recent = goal_history["poses"][-2:]
            stable_pair = math.dist(recent[0], recent[1]) < stable_tol
            if stable_pair:
                goal_history["stable_count"] += 1
            else:
                goal_history["stable_count"] = 0

        accept_mode = None
        accepted_xyz = new_xyz
        if elapsed >= min_settle_s and stable_pair:
            accept_mode = "stable"
        elif elapsed >= max_wait_s and len(goal_history["poses"]) >= 2:
            recent_np = np.array(goal_history["poses"][-median_window:], dtype=float)
            accepted_xyz = np.median(recent_np, axis=0).tolist()
            accept_mode = "median"

        if accept_mode is not None:
            goal_history["accepted"] = True
            goal_history["accept_time"] = time.time()
            node.goal_received = True
            new_xyz = accepted_xyz
            new_quat = goal_history["latest_quat"]

            # Stop idle timer
            if hasattr(node, 'idle_timer'):
                try:
                    node.idle_timer.cancel()
                    del node.idle_timer
                except Exception:
                    pass

            publish_stop_trajectory(node)

            node.goal_seed_xy = [new_xyz[0], new_xyz[1]]
            node.best_goal_xyz = new_xyz
            node.best_goal_score = float("inf")

            # Single subscribe stores exactly one goal. Use Sub Multi for
            # score-ordered multi-goal execution.
            g = [*new_xyz, *new_quat]
            node.candidate_goals = [g]
            node.goal_poses.append(g)
            publish_goal_marker(node, new_xyz)
            node.get_logger().info(
                "Detected goal accepted: "
                f"xyz=[{new_xyz[0]:.6f}, {new_xyz[1]:.6f}, {new_xyz[2]:.6f}] m, "
                f"quat_wxyz=[{new_quat[0]:.6f}, {new_quat[1]:.6f}, "
                f"{new_quat[2]:.6f}, {new_quat[3]:.6f}], "
                f"accept={accept_mode}")
            _depth_cached = getattr(node, "_latest_depth_diagnostics", None)
            if (_depth_cached is not None and
                    time.time() - _depth_cached[0] <= 1.0):
                _dv = _depth_cached[1]
                node.get_logger().debug(
                    "[GOAL_DEPTH] "
                    f"cam=[{_dv[0]:.3f},{_dv[1]:.3f},{_dv[2]:.3f}]m "
                    f"z_p5/p50/p95={_dv[3]:.3f}/{_dv[4]:.3f}/{_dv[5]:.3f}m "
                    f"near/far={int(_dv[6])}/{int(_dv[7])} "
                    f"outside_near={int(_dv[8])}/{int(_dv[9])} "
                    + (
                        f"outside_p5/p50={_dv[10]:.3f}/{_dv[11]:.3f}m "
                        if len(_dv) >= 12 else ""
                    ) +
                    f"base=[{new_xyz[0]:.3f},{new_xyz[1]:.3f},{new_xyz[2]:.3f}]m"
                )
            # Height + lateral classification from image-space bbox position when available,
            # falling back to 3D coordinate comparison with the trunk.
            # cy_norm > 0.60 = bottom 40% of image → LOW
            # LOW uses wider side bands; MID/HIGH uses stricter side bands so
            # mildly off-center fruit still uses center approach.
            img_norm = getattr(node, 'fruit_image_norm', None)
            if img_norm is not None:
                cx_norm, cy_norm = img_norm
                is_low = cy_norm > 0.60
                _very_low_thresh = getattr(node.cfg.planner, "very_low_center_cy_thresh", 0.90)
                is_very_low_center = (is_low and cy_norm >= _very_low_thresh) or is_bunch_lower_boundary(node)
                lateral_type = image_lateral_side(
                    node, cx_norm, cy_norm, force_very_low_center=is_very_low_center)
                height_type = "VERY LOW" if is_very_low_center else ("LOW" if is_low else "MID/HIGH")
                rel_x = getattr(node, "fruit_bunch_rel_x", None)
                rel_y = getattr(node, "fruit_bunch_rel_y", None)
                if rel_x is not None and rel_y is not None:
                    bunch_txt = f", bunch rx={rel_x:.2f} ry={rel_y:.2f}"
                else:
                    bunch_txt = ", bunch rx/ry=None"
                node.get_logger().info(
                    f"Primary goal accepted: {height_type} | {lateral_type} "
                    f"(img cx={cx_norm:.2f} cy={cy_norm:.2f}{bunch_txt}, "
                    f"z={new_xyz[2]:.2f}m, accept={accept_mode})")
            else:
                is_low = new_xyz[2] < LOW_Z_THRESH
                height_type = "LOW" if is_low else "MID/HIGH"
                trunk_lat = trunk_lateral(node)
                fruit_lat = lateral_value(new_xyz[0], new_xyz[1])
                lateral_dist = abs(fruit_lat - trunk_lat)
                if not bool(getattr(node.cfg.planner, "side_approach_enabled", False)):
                    lateral_type = "CENTER"
                elif lateral_dist > LATERAL_THRESH:
                    lateral_type = "LEFT" if fruit_lat > trunk_lat else "RIGHT"
                else:
                    lateral_type = "CENTER"
                node.get_logger().info(
                    f"Primary goal accepted: {height_type} | {lateral_type} "
                    f"(z={new_xyz[2]:.2f}m, fruit_lateral={fruit_lat:.3f}, "
                    f"trunk_lateral={trunk_lat:.3f}, accept={accept_mode})")

            node.latest_goal_classification = f"{height_type} | {lateral_type}"
            node.goal_lateral_side = lateral_type  # authoritative; reused at planning to avoid re-classification
            try:
                node.obstacles.update_pose("fruit_obstacle", new_xyz)
            except Exception as _e:
                node.get_logger().warn(f"obstacle update_pose failed (CUDA faulted?): {_e}")

            node.get_logger().info("Collected 1 single goal")
            _destroy_sub()
            return

    # Subscribe to /external_goal_pose with VOLATILE QoS
    # VOLATILE = don't receive old buffered messages, only fresh ones
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    fresh_qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )
    node.goal_pose_sub = node.create_subscription(
        PoseStamped,
        '/external_goal_pose',
        _goal_cb,
        fresh_qos,
    )

    # Keep the arm stationary while waiting for a valid vision goal. Previously
    # a timer moved the TCP through four idle poses; that is undesirable in the
    # field and can also change the camera view during depth stabilization.


def subscribe_multi_goals(node, max_goals=3, timeout=10.0):
    """Queue top-scored visible fruits, with /external_goal_pose collection as fallback.

    Vision publishes /vision/all_fruit_poses in score order, so goal multi normally
    queues the first distinct entries from that list for sequential execution.
    If no list is available yet, it falls back to the older live subscription path:
    each goal must be stable (2 readings <2cm) and distinct from accepted goals.
    Goals are queued in node.goal_poses for sequential execution.
    """
    DISTINCT_DIST = 0.05  # 5cm apart to count as a separate goal

    # Wait until robot stops before subscribing
    if is_robot_moving(node):
        _wait_timer = [None]
        def _wait_and_subscribe_multi():
            if not is_robot_moving(node):
                if _wait_timer[0] is not None:
                    _wait_timer[0].cancel()
                    _wait_timer[0] = None
                subscribe_multi_goals(node, max_goals, timeout)
        _wait_timer[0] = node.create_timer(0.5, _wait_and_subscribe_multi)
        return

    # Cancel any existing idle timer
    if hasattr(node, 'idle_timer'):
        try:
            node.idle_timer.cancel()
            del node.idle_timer
        except Exception:
            pass

    # Reset all tracking state
    node.reset_goal_tracking()
    node.latest_goal_pose = None
    node.latest_goal_time = 0

    # Destroy previous subscription
    if hasattr(node, 'goal_pose_sub'):
        node.destroy_subscription(node.goal_pose_sub)
        del node.goal_pose_sub

    node.goal_poses.clear()
    getattr(node, "_reachability_goal_metadata", {}).clear()
    node.candidate_goals = []

    # Clear vision exclusions at start
    from std_msgs.msg import Float32MultiArray
    def _publish_exclusions(positions):
        """Publish excluded positions to vision node so it picks different targets."""
        msg = Float32MultiArray()
        for pos in positions:
            msg.data.extend([float(pos[0]), float(pos[1]), float(pos[2])])
        node.exclude_pub.publish(msg)

    _publish_exclusions([])  # clear any previous exclusions

    cur = node.get_end_effector_pose()
    current_orientation = cur[3:] if cur else [1.0, 0.0, 0.0, 0.0]

    visible = list(getattr(node, 'all_fruit_poses', []) or [])
    if visible:
        queued_xyz = []
        for xyz in visible:
            if len(xyz) < 3 or any(v is None for v in xyz[:3]):
                continue
            xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            _workspace_ok, _workspace_reason = goal_is_in_robot_workspace(node, xyz)
            if not _workspace_ok:
                _warn_rejected_goal_throttled(node, xyz, _workspace_reason)
                continue
            if any(math.dist(xyz, prev) < DISTINCT_DIST for prev in queued_xyz):
                continue
            queued_xyz.append(xyz)
            if len(queued_xyz) >= max_goals:
                break

        if queued_xyz:
            for rank, xyz in enumerate(queued_xyz, start=1):
                g = [*xyz, *current_orientation]
                node.goal_poses.append(g)
                node.candidate_goals.append(g)
                publish_goal_marker(node, xyz)
                goal_type = "LOW" if xyz[2] < LOW_Z_THRESH else "MID/HIGH"
                node.get_logger().info(
                    f"Multi top-score goal #{rank}/{max_goals}: {goal_type} "
                    f"(z={xyz[2]:.2f}m) [{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]")

            first_xyz = queued_xyz[0]
            node.best_goal_xyz = first_xyz
            node.goal_seed_xy = [first_xyz[0], first_xyz[1]]
            node.goal_received = True
            node.get_logger().info(f"Subscribe multi: queued {len(queued_xyz)}/{max_goals} top-scored goals")
            return

    state = {
        "poses": [],
        "stable_count": 0,
        "accepted_goals": [],  # list of [x,y,z] for distinctness check
        "start_time": time.time(),
        "pending_xyz": None,
        "pending_quat": None,
    }

    def _destroy_sub():
        try:
            if hasattr(node, 'goal_pose_sub'):
                node.destroy_subscription(node.goal_pose_sub)
                del node.goal_pose_sub
        except Exception:
            pass

    def _cancel_timeout():
        if hasattr(node, '_multi_timeout_timer'):
            try:
                node._multi_timeout_timer.cancel()
                del node._multi_timeout_timer
            except Exception:
                pass

    def _finish():
        if state.get("finished"):
            return
        state["finished"] = True
        _destroy_sub()
        _cancel_timeout()
        _publish_exclusions([])  # clear exclusions so vision returns to normal
        n = len(state["accepted_goals"])
        node.get_logger().info(f"Subscribe multi: collected {n}/{max_goals} goals in {time.time() - state['start_time']:.1f}s")
        if n > 0:
            node.goal_received = True

    def _goal_cb(msg: PoseStamped):
        nonlocal state

        # Check timeout
        if time.time() - state["start_time"] > timeout:
            _finish()
            return

        # Already have enough goals
        if len(state["accepted_goals"]) >= max_goals:
            _finish()
            return

        new_xyz = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        new_quat = [msg.pose.orientation.w, msg.pose.orientation.x,
                     msg.pose.orientation.y, msg.pose.orientation.z]
        _workspace_ok, _workspace_reason = goal_is_in_robot_workspace(node, new_xyz)
        if not _workspace_ok:
            _warn_rejected_goal_throttled(node, new_xyz, _workspace_reason)
            return

        # Always store latest
        node.latest_goal_pose = [*new_xyz, *new_quat]
        node.latest_goal_time = time.time()

        # Check for position jump (vision switched to new fruit)
        if state["poses"]:
            last_xyz = state["poses"][-1]
            jump = math.dist(new_xyz, last_xyz)
            if jump > 0.10:  # >10cm = new fruit, reset stability
                state["poses"].clear()
                state["stable_count"] = 0

        state["poses"].append(new_xyz)
        state["pending_xyz"] = new_xyz
        state["pending_quat"] = new_quat

        # Need at least 2 consistent readings
        if len(state["poses"]) >= 2:
            recent = state["poses"][-2:]
            if math.dist(recent[0], recent[1]) < 0.02:
                state["stable_count"] += 1
            else:
                state["stable_count"] = 0

        # Accept after 2 stable readings
        if state["stable_count"] >= 2:
            xyz = state["pending_xyz"]
            quat = state["pending_quat"]

            # Check distinctness from all previously accepted goals
            is_distinct = all(
                math.dist(xyz, prev) > DISTINCT_DIST
                for prev in state["accepted_goals"]
            )

            if not is_distinct:
                # Don't reset stability — just keep waiting for vision to switch targets
                return

            # Accept this goal
            g = [*xyz, *quat]
            state["accepted_goals"].append(xyz)
            node.goal_poses.append(g)
            node.candidate_goals.append(g)
            publish_goal_marker(node, xyz)

            # Tell vision to exclude this position so it picks next-best
            _publish_exclusions(state["accepted_goals"])

            goal_type = "LOW" if xyz[2] < LOW_Z_THRESH else "MID/HIGH"
            n = len(state["accepted_goals"])
            node.get_logger().info(
                f"Multi goal #{n}/{max_goals}: {goal_type} "
                f"(z={xyz[2]:.2f}m) [{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]")

            if n == 1:
                node.best_goal_xyz = xyz
                node.goal_seed_xy = [xyz[0], xyz[1]]

            # Check if we have enough
            if n >= max_goals:
                _finish()
                return

            # Reset stability for next goal
            state["poses"].clear()
            state["stable_count"] = 0

    # Subscribe with VOLATILE QoS
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
    fresh_qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )
    node.goal_pose_sub = node.create_subscription(
        PoseStamped,
        '/external_goal_pose',
        _goal_cb,
        fresh_qos,
    )

    # Auto-finish after timeout
    def _timeout_cb():
        node.get_logger().info("Subscribe multi: timeout reached")
        _finish()

    node._multi_timeout_timer = node.create_timer(timeout, _timeout_cb)


def is_robot_moving(node, velocity_threshold: float = 0.001) -> bool:
    """Check if robot is moving based on joint velocities.

    Returns True (assume moving) if velocity data is unavailable - safer default.
    """
    v = getattr(node, 'current_joint_velocities', None)
    if v is None or len(v) == 0:
        # No velocity data available - assume robot IS moving (safer default)
        return True
    return any(abs(x) > velocity_threshold for x in v)

def blend_approach_direction(node, x, y, z, vis_ratio=1.0, z_std=0.01):
    prev_dir = getattr(node, "_prev_blend_dir", None)
    prev_dot = None
    d_prev = None
    fruit_dir = getattr(node, "fruit_direction", None)

    # Only trust the vision direction if it actually belongs to THIS goal.
    # /datefruit_direction carries the axis of whatever fruit vision currently
    # considers best, and _direction_cb stores it as a bare vector with no
    # association to a position -- so when several goals are queued from one
    # detection pass, goals 2..N would silently reuse goal 1's axis and pick a
    # corridor from the wrong fruit's orientation (observed: three goals 18cm
    # apart all approached with inward=[-0.002,-1.000,0.000], and a misleading
    # direction_match=1.000 because the stale axis trivially matches itself).
    # date_tip_point is the per-fruit 3D position published with that axis, so
    # reuse the association test the tool-axis aiming already applies. Failing
    # it falls through to the camera-ray direction below, which is derived from
    # this goal's own geometry.
    if fruit_dir is not None and bool(getattr(
            node.cfg.planner, "corridor_axis_require_goal_match", True)):
        _tip = getattr(node, "date_tip_point", None)
        _tip_age = time.time() - float(getattr(node, "date_tip_time", 0.0) or 0.0)
        # Deliberately tighter than tool_axis_tip_max_goal_distance_m (80mm):
        # that limit answers "is the tip close enough to aim with", whereas this
        # answers "is this the SAME fruit". subscribe_multi_goals treats
        # detections >=5cm apart as separate fruits, so use the same standard --
        # at 80mm a neighbouring date 54mm away would still lend its axis.
        _max_gap = float(getattr(
            node.cfg.planner, "corridor_axis_max_goal_distance_m", 0.05))
        _max_age = float(getattr(
            node.cfg.planner, "corridor_axis_max_age_s", 2.0))
        _gap = math.dist(list(_tip), [x, y, z]) if _tip is not None else float("inf")
        if _tip is None or _gap > _max_gap or _tip_age > _max_age:
            _why = ("no tip" if _tip is None
                    else (f"tip {_gap*1000:.0f}mm from goal (limit {_max_gap*1000:.0f}mm)"
                          if _gap > _max_gap else f"tip {_tip_age:.1f}s old"))
            node.get_logger().info(
                f"[CORRIDOR_AXIS] vision axis not associated with this goal "
                f"({_why}); using goal-local camera-ray direction instead")
            fruit_dir = None

    if fruit_dir is not None:
        d_dir = np.array(fruit_dir, dtype=float)
        n_dir = np.linalg.norm(d_dir)
        if n_dir > 1e-9:
            d_dir /= n_dir
        else:
            d_dir = np.array([0.0, -1.0, 0.0], dtype=float)
        # Visibility blending is currently disabled below. Do not enter the
        # TF/spin fallback when vision already supplied a frozen direction.
        d_vis = d_dir.copy()

        # Align with previous blended direction to avoid 180 flips.
        if prev_dir is not None:
            d_prev = np.array(prev_dir, dtype=float)
            n_prev = np.linalg.norm(d_prev)
            if n_prev > 1e-9:
                d_prev /= n_prev
                prev_dot = float(np.dot(d_dir, d_prev))
                if prev_dot < 0.0:
                    d_dir = -d_dir
                    prev_dot = -prev_dot
            else:
                d_prev = None

        dir_conf = 1.0
    else:
        d_vis_pt = compute_visibility_approach(node, x, y, z, dist=0.05)
        d_vis = -np.array(
            [x - d_vis_pt[0], y - d_vis_pt[1], z - d_vis_pt[2]])
        _d_vis_norm = float(np.linalg.norm(d_vis))
        if _d_vis_norm > 1e-9:
            d_vis /= _d_vis_norm
        else:
            d_vis = np.array([0.0, -1.0, 0.0], dtype=float)
        d_dir = d_vis
        dir_conf = 0.0

    # vis_conf based on visibility quality (higher vis_ratio + lower z_std = more confident)
    # Note: Currently disabled (vis_conf=0) to rely purely on fruit_direction from vision
    vis_conf = 0.0  # np.clip(vis_ratio * np.exp(-z_std / 0.02), 0.0, 1.0)

    d = dir_conf * d_dir + vis_conf * d_vis
    n_blend = np.linalg.norm(d)
    d_norm = d / n_blend if n_blend > 1e-9 else d_vis.copy()

    # Determine if this is a new direction or same as last time
    _prev = getattr(node, "_prev_blend_dir", None)
    if _prev is None:
        _status = "NEW (first reading)"
    elif fruit_dir is None:
        _status = "FALLBACK (no fruit_direction from vision — using camera-ray)"
    else:
        _delta = float(np.linalg.norm(d_norm - np.array(_prev, dtype=float)))
        if _delta < 0.05:
            _status = f"SAME (delta={_delta:.3f})"
        else:
            _status = f"NEW (delta={_delta:.3f})"

    node._prev_blend_dir = d_norm.tolist()

    node.get_logger().debug(
        f"[DIR] {_status} | "
        f"source={'fruit_direction' if fruit_dir is not None else 'camera-ray'} | "
        f"d_blend=[{d_norm[0]:.3f},{d_norm[1]:.3f},{d_norm[2]:.3f}]"
        + (f" | prev_dot={prev_dot:.3f}" if prev_dot is not None else "")
    )

    return d_norm



# Main goal-execution pipeline — runs through all saved goals and performs motion + gripper actions in sequence.
def plan_and_execute(node):

    # Auto-harvest uses this explicit result to count only cycles that reached
    # the normal end of grasp, reverse, drop-off, and return handling.
    node._last_completed_harvest_cycles = 0

    def _valid_joint_positions(wait_s: float = 1.0):
        deadline = time.time() + wait_s
        expected = len(getattr(node, "joint_order", []) or [])
        while time.time() <= deadline:
            joints = getattr(node, "current_joint_positions", None)
            if (joints is not None and
                    (expected == 0 or len(joints) == expected) and
                    all(j is not None and math.isfinite(float(j)) for j in joints)):
                return list(joints)
            time.sleep(0.05)
        return None

    _start_joints = _valid_joint_positions()
    if _start_joints is None:
        node.get_logger().warn("No joint state yet."); return
    if not node.goal_poses:
        node.get_logger().warn("No stored goals."); return
    if not node.robot_running:
        node.get_logger().error("Robot program OFF; may fail.")

    # Corridor exclusions are valid only while retrying a target inside this
    # execution batch. Never carry rejected angles into a later GUI run.
    node._corridor_exclusions = set()
    node._corridor_retry_goal = None
    node._corridor_tip_fallback_active = False
    node._corridor_forced_label = None
    node._corridor_pair_eval = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _yolo = getattr(node, 'yolo_thread', None)  # local inference_lock only (same process)
    # Vision mode helpers — these publish to /vision/mode to control the separate vision process.
    # _vision_pause() waits for the vision node to confirm it has paused and drained
    # any in-flight YOLO inference (see wait_for_vision_paused). Without this handoff,
    # YOLO TRT and cuRobo CUDA kernels overlap on the shared GPU and trigger
    # device-side asserts. If no ack arrives (vision not running or ack dropped), fall
    # back to a fixed 120ms delay: ROS2 topic latency (~20ms) + worst-case YOLO
    # inference at 640px on Jetson (~50ms) + margin.
    def _vision_pause():
        node.set_vision_mode("paused")
        if not node.wait_for_vision_paused(timeout=0.30):
            time.sleep(0.12)
    def _vision_resume(): node.set_vision_mode("full")

    def _check_stop():
        """Check if stop was requested; if so, halt robot and clear goals."""
        if getattr(node, "stop_requested", False) or getattr(node, "_stop_was_requested", False):
            node.get_logger().warn("STOP requested — aborting immediately.")
            publish_stop_trajectory(node)
            node.goal_poses.clear()
            # Latch before consuming: clearing these two flags is what let an
            # AUTO-HARVEST batch resume after a stop, because the outer loop
            # re-reads stop_requested and sees False. The latch survives until a
            # new batch is deliberately started.
            node._auto_harvest_abort = True
            node.stop_requested = False
            node._stop_was_requested = False
            unlock_target(node)
            return True
        return False

    _cycle_idx = 0
    _run_t0 = time.time()
    while node.goal_poses and getattr(node, 'running', True):
        if _check_stop():
            break
        
        _start_joints = _valid_joint_positions()
        if _start_joints is None:
            node.get_logger().warn("Incomplete joint state before goal; skipping remaining goals.")
            break

        start = JointState.from_position(
            torch.tensor([_start_joints], dtype=torch.float32, device=device),
            joint_names=node.joint_order,
        )
        
        
        # 0. Get next goal (thread-safe pop returns None if empty)
        goal = node.goal_poses.pop(0)
        if goal is None:
            node.get_logger().warn("Goal queue empty during pop; skipping.")
            continue
        _cycle_idx += 1
        _cycle_t0 = time.time()
        _timing = {}
        _begin_motion_log = getattr(node, "_begin_motion_cycle_log", None)
        if callable(_begin_motion_log):
            _begin_motion_log(_cycle_idx)
        _log_cycle_start = getattr(node.cfg.planner, "log_cycle_start", False)
        _log_phase_timings = getattr(node.cfg.planner, "log_phase_timings", False)
        if getattr(node, "_slip_retry_count", 0) <= 0:
            node._slip_retry_approach = None
            node._slip_retry_fruit_radius = None
        x,y,z = goal[:3]

        _tip_anchored_final_grasp = None
        _tip_anchor = None
        _tip_candidate = getattr(node, "date_tip_point", None)
        _tip_age = time.time() - float(getattr(node, "date_tip_time", 0.0))
        _tip_goal_match = (
            math.dist(list(_tip_candidate), [x, y, z])
            if _tip_candidate is not None else float("inf"))
        _tip_max_goal_distance = float(getattr(
            node.cfg.planner, "tool_axis_tip_max_goal_distance_m", 0.015))
        _tip_axis_stable = bool(getattr(
            node, "fruit_major_axis_stable", False))
        _tip_selection = getattr(node, "safe_grasp_candidate", None)
        _tip_candidate_unambiguous = bool(
            isinstance(_tip_selection, dict)
            and _tip_selection.get("available", False))
        _tip_require_safe_candidate = bool(getattr(
            node.cfg.planner, "tool_axis_tip_require_safe_candidate", True))
        _tip_aim_enabled = bool(getattr(
            node.cfg.planner, "tool_axis_tip_aim_enabled", True))
        _retry_goal = getattr(node, "_corridor_retry_goal", None)
        _same_corridor_goal = bool(
            _retry_goal is not None
            and math.dist(list(goal[:3]), list(_retry_goal)) <= 0.005)
        _frozen_retry_tip = (
            getattr(node, "_corridor_retry_tip_anchor", None)
            if _same_corridor_goal else None)
        # Age of the frozen snapshot, measured from the VISION timestamp of the
        # reading that was frozen -- so it counts both how stale that reading
        # already was and how long corridor retrying has taken since. Missing
        # timestamp (0.0) yields a huge age and is therefore rejected, which is
        # the safe direction: fall back to the calibrated corridor grasp.
        _frozen_tip_max_age = float(getattr(
            node.cfg.planner, "tool_axis_tip_frozen_max_age_s", 2.0))
        _frozen_tip_age = time.time() - float(
            getattr(node, "_corridor_retry_tip_time", 0.0) or 0.0)
        _frozen_tip_fresh = (
            _frozen_retry_tip is not None
            and _frozen_tip_age <= _frozen_tip_max_age)
        if (_frozen_retry_tip is not None and not _frozen_tip_fresh):
            node.get_logger().warn(
                f"[TOOL_AXIS_AIM] frozen snapshot {_frozen_tip_age:.1f}s old "
                f"(limit {_frozen_tip_max_age:.1f}s); discarding it and "
                "aiming at the detected date centre instead")
            _frozen_retry_tip = None
            node._corridor_retry_tip_anchor = None
            node._corridor_retry_tip_time = 0.0
        node._active_tool_axis_result = "NO_TIP"
        node._active_tool_axis_reason = "NO_TIP_AVAILABLE"
        node._active_tool_axis_tip_age_s = (
            round(float(_tip_age), 3) if math.isfinite(_tip_age) else None)
        node._active_tool_axis_goal_match_mm = (
            round(float(_tip_goal_match) * 1000.0, 1)
            if math.isfinite(_tip_goal_match) else None)
        if (
            _tip_aim_enabled
            and _tip_candidate is not None
            and bool(getattr(node, "date_tip_stable", False))
            and _tip_axis_stable
            and (not _tip_require_safe_candidate or _tip_candidate_unambiguous)
            and _tip_age <= float(getattr(
                node.cfg.planner, "tool_axis_tip_max_age_s", 0.50))
            and _tip_goal_match <= _tip_max_goal_distance
        ):
            _tip_anchor = np.asarray(_tip_candidate, dtype=float)
            node._active_tool_axis_result = "DATE_TIP_ACCEPTED"
            node._active_tool_axis_reason = "STABLE"
        elif (_tip_aim_enabled and _frozen_retry_tip is not None
              and math.dist(list(_frozen_retry_tip), [x, y, z])
              <= _tip_max_goal_distance):
            # Corridor preflight retries intentionally keep the same frozen
            # perception snapshot. Do not call that snapshot stale and claim a
            # calibrated fallback while subsequently restoring and using it.
            #
            # The distance guard above is NOT optional, even though this is the
            # "same goal" path: the branch used to restore the snapshot
            # unconditionally, so a tip sitting well off the goal was accepted
            # and pulled the grasp point with it. Observed in a batch run: a tip
            # 46.5mm off moved goal 1's grasp 42.8mm, landing 13.5mm from goal 2
            # -- two queued goals (52.6mm apart, so legitimately distinct) ended
            # up grasping the same date. Aim correction must never exceed what
            # keeps goals separable.
            _tip_anchor = np.asarray(_frozen_retry_tip, dtype=float)
            _tip_goal_match = math.dist(_tip_anchor.tolist(), [x, y, z])
            node._active_tool_axis_result = "DATE_TIP_FROZEN_RETRY"
            node._active_tool_axis_reason = "SAME_GOAL_SNAPSHOT"
            node._active_tool_axis_goal_match_mm = round(
                float(_tip_goal_match) * 1000.0, 1)
            node.get_logger().info(
                "[TOOL_AXIS_AIM] source=FROZEN_RETRY "
                f"goal_match={_tip_goal_match * 1000.0:.1f}mm")
        elif _tip_aim_enabled and _tip_candidate is not None:
            _tip_reasons = []
            if not bool(getattr(node, "date_tip_stable", False)):
                _tip_reasons.append("TIP_UNSTABLE")
            if not _tip_axis_stable:
                _tip_reasons.append("DATE_AXIS_UNSTABLE")
            if (_tip_require_safe_candidate and
                    not _tip_candidate_unambiguous):
                _tip_reasons.append("YAW_AMBIGUOUS")
            if _tip_age > float(getattr(
                    node.cfg.planner, "tool_axis_tip_max_age_s", 0.50)):
                _tip_reasons.append("TIP_STALE")
            if _tip_goal_match > _tip_max_goal_distance:
                _tip_reasons.append(
                    f"GOAL_MISMATCH_{_tip_goal_match * 1000.0:.1f}MM")
            node.get_logger().warn(
                "[TOOL_AXIS_AIM_REJECTED] "
                f"reason={'+'.join(_tip_reasons) if _tip_reasons else 'DISABLED'} "
                f"limit={_tip_max_goal_distance * 1000.0:.1f}mm "
                "fallback=CALIBRATED_CORRIDOR")
            node._active_tool_axis_result = "CALIBRATED_CORRIDOR"
            node._active_tool_axis_reason = "+".join(
                _tip_reasons) if _tip_reasons else "DISABLED"
        if not _same_corridor_goal:
            node._corridor_exclusions = set()
            node._corridor_roll_disabled = set()
            node._corridor_retry_goal = list(goal[:3])
            node._corridor_retry_tip_anchor = None
            node._corridor_retry_tip_time = 0.0
            node._corridor_preflight_cache = {}
            node._corridor_preflight_cache_start = None
            node._corridor_tip_fallback_active = False
            node._corridor_forced_label = None
            node._corridor_pair_eval = None
        if _tip_anchor is not None:
            node._corridor_retry_tip_anchor = _tip_anchor.tolist()
            # Stamp the snapshot with the vision time of the reading behind it,
            # so later retries can tell how old it has become.
            node._corridor_retry_tip_time = float(
                getattr(node, "date_tip_time", 0.0) or 0.0)
        elif _same_corridor_goal and _frozen_retry_tip is not None:
            # Silent restore path. This used to resurrect the snapshot with no
            # distance OR age check -- even right after the chain above logged
            # "fallback=CALIBRATED_CORRIDOR", which made that message untrue.
            # Apply the same two gates the FROZEN_RETRY branch applies.
            _saved_match = math.dist(list(_frozen_retry_tip), [x, y, z])
            if _saved_match <= _tip_max_goal_distance:
                _tip_anchor = np.asarray(_frozen_retry_tip, dtype=float)
                _tip_goal_match = _saved_match
                node.get_logger().info(
                    "[TOOL_AXIS_AIM] restored frozen snapshot "
                    f"({_frozen_tip_age:.1f}s old, "
                    f"goal_match={_saved_match * 1000.0:.1f}mm)")
        if bool(getattr(node, "_corridor_tip_fallback_active", False)):
            _tip_anchor = None
            node.get_logger().debug(
                "[TOOL_AXIS_AIM] mode=CENTRE_FALLBACK "
                "tip_anchor=DISABLED_FOR_THIS_GOAL")

        # Freeze the gripper geometry for this target. AUTO uses the selected
        # date's stable image major axis: 0deg is vertical, +/-90deg horizontal.
        _configured_grasp_mode = str(getattr(
            node.cfg.gripper, "grasp_mode", "NORMAL")).strip().upper()
        if (_same_corridor_goal
                and getattr(node, "_corridor_retry_grasp_mode", None)):
            _goal_grasp_mode = node._corridor_retry_grasp_mode
        elif _configured_grasp_mode == "AUTO":
            _axis_deg_raw = math.degrees(float(getattr(
                node, "fruit_major_axis_angle", 0.0)))
            _axis_from_vertical = abs(
                ((_axis_deg_raw + 90.0) % 180.0) - 90.0)
            _axis_confidence = float(getattr(
                node, "fruit_major_axis_confidence", 0.0))
            _axis_stable = bool(getattr(
                node, "fruit_major_axis_stable", False))
            _axis_threshold = float(getattr(
                node.cfg.gripper,
                "auto_envelop_axis_from_vertical_deg", 45.0))
            _confidence_threshold = float(getattr(
                node.cfg.gripper,
                "auto_envelop_min_axis_confidence", 0.30))
            _goal_grasp_mode = (
                "ENVELOP"
                if (_axis_stable
                    and _axis_confidence >= _confidence_threshold
                    and _axis_from_vertical >= _axis_threshold)
                else "NORMAL")
            node.get_logger().debug(
                f"[GRASP_MODE_AUTO] selected={_goal_grasp_mode} "
                f"axis_from_vertical={_axis_from_vertical:.1f}deg "
                f"threshold={_axis_threshold:.1f}deg "
                f"confidence={_axis_confidence:.2f}/"
                f"{_confidence_threshold:.2f} stable={_axis_stable} "
                "frozen=THIS_GOAL")
        else:
            _goal_grasp_mode = (
                _configured_grasp_mode
                if _configured_grasp_mode in ("NORMAL", "ENVELOP")
                else "NORMAL")
        if not _same_corridor_goal:
            _grasp_style = (
                "wrap" if _goal_grasp_mode == "ENVELOP" else "pinch")
            node.get_logger().info(
                f"[GRASP_MODE] selected={_goal_grasp_mode} "
                f"style={_grasp_style} source={_configured_grasp_mode}")
        node.active_goal_grasp_mode = _goal_grasp_mode
        node._corridor_retry_grasp_mode = _goal_grasp_mode
        _active_corridor = None
        # Diagnostic timers only (no behavior change): measure where the time
        # actually goes on a corridor mirror-pair trip -- pose computation vs.
        # the IK/cartesian preflight itself -- before deciding how to speed it up.
        _pose_compute_ms = None
        if _log_cycle_start:
            _img_norm_for_log = getattr(node, "fruit_image_norm", None)
            _bunch_rel_x_for_log = getattr(node, "fruit_bunch_rel_x", None)
            _bunch_rel_y_for_log = getattr(node, "fruit_bunch_rel_y", None)
            node.get_logger().info(
                f"[CYCLE {_cycle_idx}] START | goal=[{x:.3f},{y:.3f},{z:.3f}] "
                f"queue_remaining={len(node.goal_poses)} retry={getattr(node, '_slip_retry_count', 0)} "
                + (f"img=[{_img_norm_for_log[0]:.2f},{_img_norm_for_log[1]:.2f}] " if _img_norm_for_log is not None else "")
                + (f"bunch_rel_x={_bunch_rel_x_for_log:.2f}" if _bunch_rel_x_for_log is not None else "bunch_rel_x=NA")
                + (f" bunch_rel_y={_bunch_rel_y_for_log:.2f}" if _bunch_rel_y_for_log is not None else " bunch_rel_y=NA")
            )

        # Clear previous trajectory markers from RViz
        clear_path_markers(node)

        # A preflight-only corridor retry has not moved the robot and deliberately
        # keeps this exact target locked with vision paused. Do not republish the
        # lock or repeat the GPU pause handshake/candidate evaluation.
        _reuse_preflight_lock = bool(
            _same_corridor_goal
            and getattr(node, "_corridor_retry_keep_lock", False))
        node._corridor_retry_keep_lock = False
        if not _reuse_preflight_lock:
            # Lock vision onto this target (prevents switching to a different
            # "best" during approach), then free the GPU for cuRobo.
            lock_target(node, goal[:3])
            _vision_pause()

        # Clear stored trajectory for partial reverse after grasp
        node.stored_trajectory_states = []

        # Adaptive gripper open: deferred until approach motion starts (opens during arm travel)
        fruit_radius = getattr(node, 'latest_fruit_radius', None)
        gripper_opened = False

        # Height-based approach strategy. Prefer the same image-space
        # classification used when accepting the goal, because depth/z can sit
        # near the threshold and flip a visually MID/HIGH fruit into LOW.
        _img_norm_height = getattr(node, 'fruit_image_norm', None)
        is_very_low_center = False
        if _img_norm_height is not None:
            _, _cy_norm = _img_norm_height
            is_low = _cy_norm > 0.60
            _very_low_thresh = getattr(node.cfg.planner, "very_low_center_cy_thresh", 0.90)
            is_very_low_center = (is_low and _cy_norm >= _very_low_thresh) or is_bunch_lower_boundary(node)
            _height_source = f"img cy={_cy_norm:.2f}"
        else:
            is_low = z < LOW_Z_THRESH
            _height_source = f"z_thresh={LOW_Z_THRESH:.2f}"

        standoff = _planner_value(
            node.cfg.planner, "mid_center_approach_depth_offset",
            "mid_center_approach_y_offset", 0.10)
        d_blend = blend_approach_direction(node, x, y, z)
        side_approach_enabled = bool(getattr(
            node.cfg.planner, "side_approach_enabled", False))

        if is_low:
            # Low-hanging: original master_new approach (no d_blend for position)
            ax = x
            ay = y
            az = z
            _approach_height_label = "VERY LOW" if is_very_low_center else "LOW"
            if _log_cycle_start:
                node.get_logger().info(
                    f"{_approach_height_label} approach ({_height_source}, z={z:.2f}): "
                    f"fruit=[{x:.3f},{y:.3f},{z:.3f}]")
        else:
            # Mid/high: standoff directly behind fruit along the configured depth axis.
            # Right-side fruits get a larger standoff to improve approach angle
            _is_right = (
                lateral_value(x, y) < trunk_lateral(node)
                if side_approach_enabled else False)
            _midhi_standoff = 0.12 if _is_right else standoff
            _midhi_final_depth = _planner_value(
                node.cfg.planner, "mid_center_final_depth_offset",
                "mid_center_final_y_offset", -0.015)
            _midhi_final_z = float(getattr(
                node.cfg.planner, "mid_center_final_z_offset", 0.030))
            _midhi_final_x, _midhi_final_y = add_axis_offsets(
                x, y, depth=-_midhi_final_depth)
            _midhi_final_grasp = np.array(
                [_midhi_final_x, _midhi_final_y, z + _midhi_final_z])
            _midhi_insert_dir, _midhi_side = select_mid_high_corridor_direction(
                node, d_blend)
            _active_corridor = _midhi_side
            # Level insertion: APPROACH follows the clearest explicit corridor.
            _midhi_approach = (
                _midhi_final_grasp - _midhi_standoff * _midhi_insert_dir)
            ax, ay, az = _midhi_approach.tolist()
            node.get_logger().info(
                f"[APPROACH_ORIGIN] class=MID_HIGH from={_midhi_side} "
                f"origin=[{ax:.3f},{ay:.3f},{az:.3f}] "
                f"final_grasp=[{_midhi_final_grasp[0]:.3f},"
                f"{_midhi_final_grasp[1]:.3f},{_midhi_final_grasp[2]:.3f}] "
                f"travel_dir=[{_midhi_insert_dir[0]:+.3f},"
                f"{_midhi_insert_dir[1]:+.3f},{_midhi_insert_dir[2]:+.3f}]")
            if _log_cycle_start:
                node.get_logger().info(
                    f"MID/HIGH approach ({_height_source}, z={z:.2f}): "
                    f"fruit=[{x:.3f},{y:.3f},{z:.3f}] standoff=[{ax:.3f},{ay:.3f},{az:.3f}] "
                    f"depth_standoff={_midhi_standoff:+.3f} "
                    f"final_z_offset={_midhi_final_z:+.3f} "
                    "insertion=LEVEL right=" + str(_is_right))

        # 1. Plan approach - different strategy based on height and lateral position
        # Get current orientation and minimize rotation
        cur_pose = node.get_end_effector_pose()
        cur_quat = cur_pose[3:] if cur_pose else None
        target_quat = goal[3:]

        # Side HOME: use predefined home_left / home_right based on fruit vs trunk position
        is_side_approach = False
        side_home_reacquired = False
        trunk_lat = trunk_lateral(node)
        fruit_lat = lateral_value(x, y)
        _low_side_home_min_z = float(
            getattr(node.cfg.planner, "low_side_home_min_goal_z_m", 0.55))
        _small_tree_low_target = bool(is_low and z < _low_side_home_min_z)
        if _small_tree_low_target and not is_very_low_center:
            is_very_low_center = True
            if _log_cycle_start:
                node.get_logger().info(
                    f"Small-tree low target z={z:.2f}m < {_low_side_home_min_z:.2f}m — "
                    "using target-local very-low approach instead of fixed side-low HOME")
        _img_norm = getattr(node, 'fruit_image_norm', None)
        _img_side = None
        _force_center_low = _small_tree_low_target
        if cur_pose is not None:
            lateral_dist = abs(fruit_lat - trunk_lat)
            if not side_approach_enabled:
                _img_side = "CENTER"
            # If image-based classification said CENTER, respect it — skip side home
            # even if 3D coordinate check would say lateral. If it said LEFT/RIGHT,
            # use that side so preview and execution do not disagree.
            _img_lateral = True  # default: trust 3D
            if _img_norm is not None:
                _cx, _cy = _img_norm
                _force_center_low = is_very_low_center or _small_tree_low_target
                # Reuse the lateral classification from acceptance — avoids threshold boundary
                # flips between acceptance and execution (the fruit's side doesn't change).
                _img_side = getattr(node, 'goal_lateral_side', None)
                if _img_side is None:
                    _img_side = image_lateral_side(
                        node, _cx, _cy, force_very_low_center=_force_center_low)
                _img_lateral = _img_side in ("LEFT", "RIGHT")
            if not side_approach_enabled:
                _img_side = "CENTER"
                _img_lateral = False
            if _log_cycle_start:
                node.get_logger().info(
                    f"Fruit lateral distance: {lateral_dist:.2f}m, "
                    f"trunk_lateral={trunk_lat:.3f}"
                    + (f", img_side={_img_side}" if _img_norm is not None else "")
                    + (", very_low_center=True" if _force_center_low else ""))
            if not side_approach_enabled:
                if _log_cycle_start:
                    node.get_logger().info(
                        "Side approach disabled by config — using center/local approach")
                _img_lateral = False
            if _force_center_low:
                if _log_cycle_start:
                    node.get_logger().info("Very low fruit — using center HOME, skipping side-low HOME")
                _img_lateral = False
            # Front-zone override: if the fruit's image cx is within the front band the
            # fruit is facing the robot directly — use center approach regardless of
            # bunch_rx lateral classification.  bunch_rx=0.12 can say LEFT while
            # cx=0.34 is still in front, making a side home unnecessary and causing
            # long roundabout approach paths.
            if _img_lateral and _img_norm is not None:
                _cx_raw = _img_norm[0]
                _fz_min = float(getattr(node.cfg.planner, "front_zone_cx_min", 0.25))
                _fz_max = float(getattr(node.cfg.planner, "front_zone_cx_max", 0.75))
                if _fz_min <= _cx_raw <= _fz_max:
                    _img_lateral = False
                    node.get_logger().info(
                        f"Front-zone override: cx={_cx_raw:.2f} in [{_fz_min:.2f},{_fz_max:.2f}] "
                        f"— using center approach instead of {_img_side}")
            if (side_approach_enabled and not _force_center_low and
                    ((_img_norm is not None and _img_lateral) or
                     (_img_norm is None and lateral_dist > LATERAL_THRESH))):
                _side_is_left = (
                    (_img_side == "LEFT")
                    if _img_side in ("LEFT", "RIGHT")
                    else (fruit_lat > trunk_lat)
                )
                if _side_is_left:
                    side_joints = node.home_left_low_joints if is_low else node.home_left_joints
                    side_label = "HOME_LEFT_LOW" if is_low else "HOME_LEFT"
                else:
                    if not is_low:
                        # MID/HIGH RIGHT: skip side HOME, use center approach
                        if _log_cycle_start:
                            node.get_logger().info("MID/HIGH RIGHT: skipping side HOME, using center approach")
                        side_joints = None
                        side_label = None
                    else:
                        side_joints = node.home_right_low_joints
                        side_label = "HOME_RIGHT_LOW"
                if side_joints is not None:
                    if _log_cycle_start:
                        node.get_logger().info(
                            f"{side_label}: fruit_lateral={fruit_lat:.2f}, "
                            f"trunk_lateral={trunk_lat:.3f}")
                    _side_start_joints = _valid_joint_positions()
                    if _side_start_joints is not None:
                        side_joints = nearest_joint_config(_side_start_joints, side_joints)

                    plan_execute_js(node, side_joints, label=side_label, motion_type="home", speed_factor=0.5)
                    is_side_approach = True
                # Update start state and cur_pose after side HOME
                cur_pose = node.get_end_effector_pose()
                cur_quat = cur_pose[3:] if cur_pose else cur_quat
                _post_side_joints = _valid_joint_positions()
                if _post_side_joints is None:
                    node.get_logger().warn("Incomplete joint state after side HOME; skipping goal.")
                    unlock_target(node)
                    continue
                start = JointState.from_position(
                    torch.tensor([_post_side_joints], dtype=torch.float32, device=device),
                    joint_names=node.joint_order,
                )
                # Camera is fixed — its surface normal won't change with arm position.
                # Use the geometric direction from the current EE (at side home) toward the fruit.
                if cur_pose is not None:
                    ee_to_fruit = np.array([x - cur_pose[0], y - cur_pose[1], z - cur_pose[2]], dtype=float)
                    n = np.linalg.norm(ee_to_fruit)
                    if n > 1e-6:
                        d_blend = ee_to_fruit / n
                        if _log_cycle_start:
                            node.get_logger().debug(
                                f"[DIR] NEW (geometric from EE at {side_label}) | "
                                f"d_blend=[{d_blend[0]:.3f},{d_blend[1]:.3f},{d_blend[2]:.3f}]")
                if is_side_approach and node.cfg.planner.reacquire_after_approach:
                    seed = [x, y, z]
                    node.motion_phase = "REACQUIRE"
                    node.reacquire_result = "SEARCH"
                    node._publish_goal_info()
                    if _log_cycle_start:
                        node.get_logger().info(
                            f"[REACQ_SIDE_HOME] Searching from {side_label} before approach")
                    reacq = reacquire_goal_pose(
                        node,
                        seed_xyz=seed,
                        candidate_seeds=[],
                        timeout=float(getattr(
                            node.cfg.planner, "reacquire_timeout_s", 1.0)),
                        radius=0.03,
                        z_tolerance=0.010,  # see the after-approach call site
                        depth_settle_s=float(getattr(
                            node.cfg.planner, "reacquire_depth_settle_s", 0.10)),
                        stable_needed=int(getattr(
                            node.cfg.planner, "reacquire_stable_frames", 2)),
                        restore_mode="paused",
                    )
                    if reacq:
                        x, y, z = reacq
                        goal[:3] = [x, y, z]
                        lock_target(node, [x, y, z])
                        publish_goal_marker(node, [x, y, z])
                        node.reacquire_result = "OK"
                        side_home_reacquired = True
                        node.get_logger().info(
                            f"[REACQ_SIDE_HOME] Refined: [{x:.3f}, {y:.3f}, {z:.3f}]")
                    else:
                        x, y, z = seed
                        node.reacquire_result = "SEED"
                        side_home_reacquired = True
                        node.get_logger().info(
                            f"[REACQ_SIDE_HOME] No match — using original seed: "
                            f"[{x:.3f}, {y:.3f}, {z:.3f}]")

                    if is_low:
                        ax, ay, az = x, y, z
                    else:
                        fruit_lat = lateral_value(x, y)
                        _is_right = fruit_lat < trunk_lateral(node)
                        _midhi_standoff = 0.12 if _is_right else standoff
                        _midhi_final_depth = _planner_value(
                            node.cfg.planner, "mid_center_final_depth_offset",
                            "mid_center_final_y_offset", -0.015)
                        _midhi_final_z = float(getattr(
                            node.cfg.planner, "mid_center_final_z_offset", 0.030))
                        _midhi_final_x, _midhi_final_y = add_axis_offsets(
                            x, y, depth=-_midhi_final_depth)
                        ax, ay = add_axis_offsets(
                            _midhi_final_x, _midhi_final_y,
                            depth=_midhi_standoff)
                        az = z + _midhi_final_z
            else:
                if _log_cycle_start:
                    node.get_logger().info("Fruit near center; using default HOME without side move.")

        fruit_lat = lateral_value(x, y)
        _height_log = "VERY_LOW" if is_very_low_center else ("LOW" if is_low else "MID_HIGH")
        if _log_cycle_start:
            node.get_logger().debug(
                f"[DECISION] height={_height_log} source={_height_source} "
                f"side={_img_side if _img_norm is not None else '3D'} "
                f"side_home={is_side_approach} side_home_reacq={side_home_reacquired} "
                f"trunk_lateral={trunk_lat:.3f} fruit_lateral={fruit_lat:.3f}")

        # Skip approach if EE is already close to the goal
        skip_approach = False
        if cur_pose is not None:
            ee_dist = math.sqrt((cur_pose[0] - x)**2 + (cur_pose[1] - y)**2 + (cur_pose[2] - z)**2)
            if _log_cycle_start:
                node.get_logger().info(f"EE-to-goal distance: {ee_dist*100:.1f}cm")

        # 2-finger mode: detect between-branches scenario from vision depth analysis
        between_branches = getattr(node, 'fruit_between_branches', False)
        gap_angle = getattr(node, 'fruit_gap_angle', 0.0)
        low_side_dir = None
        final_orientation_override = None

        _roll_default = float(getattr(
            node.cfg.planner, "low_center_tool_roll_default_deg", -60.0))
        _center_tool_roll_deg = _roll_default
        _center_tool_roll_source = "fallback"
        _roll_score = float(getattr(node, "fruit_contact_roll_score", 0.0))
        _roll_min_score = float(getattr(
            node.cfg.planner, "low_center_tool_roll_min_score", 0.60))
        _dynamic_roll = bool(getattr(
            node.cfg.planner, "dynamic_low_center_tool_roll", True))
        _vision_roll_deg = None
        if _dynamic_roll and between_branches:
            _vision_roll_deg = math.degrees(float(gap_angle))
            _center_tool_roll_source = "vision_gap"
        elif (
            _dynamic_roll
            and bool(getattr(node, "fruit_contact_roll_valid", False))
            and _roll_score >= _roll_min_score
        ):
            _vision_roll_deg = math.degrees(float(getattr(
                node, "fruit_contact_roll", 0.0)))
            _center_tool_roll_source = "vision_contacts"
        if _vision_roll_deg is not None:
            # Three-finger geometry repeats every 120 degrees. Select the
            # vision-equivalent angle closest to the calibrated fallback.
            _center_tool_roll_deg = _roll_default + (
                (_vision_roll_deg - _roll_default + 60.0) % 120.0 - 60.0)

        if _dynamic_roll and is_low and not is_side_approach:
            node.get_logger().debug(
                f"[TOOL_ROLL] selected={_center_tool_roll_deg:+.1f}deg "
                f"source={_center_tool_roll_source} "
                f"vision_valid={bool(getattr(node, 'fruit_contact_roll_valid', False))} "
                f"score={_roll_score:.3f} threshold={_roll_min_score:.3f} "
                f"stable={bool(getattr(node, 'fruit_contact_roll_stable', False))} "
                f"samples={int(getattr(node, 'fruit_contact_roll_samples', 0))} "
                f"spread={float(getattr(node, 'fruit_contact_roll_spread_deg', float('inf'))):.1f}deg "
                f"fallback={_roll_default:+.1f}deg")

        if between_branches:
            node.get_logger().info(
                f"Between-branches detected! gap_angle={math.degrees(gap_angle):.1f} deg — using 2-finger mode")
            if not _dynamic_roll:
                # Original two-finger behavior: align the active finger pair with
                # the detected branch gap before insertion-axis alignment.
                half = gap_angle / 2.0
                q_roll = [math.cos(half), 0.0, 0.0, math.sin(half)]
                target_quat = quat_multiply(list(target_quat), q_roll)
            node.gripper_controller.frozen_fingers = {1}  # freeze center finger
        else:
            node.gripper_controller.frozen_fingers = set()  # all 3 fingers active

        def _apply_center_yaw(orientation_in, approach_xyz):
            if not bool(getattr(
                    node.cfg.planner, "low_center_yaw_align_enabled", True)):
                return list(orientation_in)
            desired_xy = [
                x - float(approach_xyz[0]),
                y - float(approach_xyz[1]),
                0.0,
            ]
            aligned, raw_delta, remaining = bounded_yaw_align_local_axis(
                orientation_in,
                desired_xy,
                local_axis=(0.0, 0.0, 1.0),
                max_delta_deg=float(getattr(
                    node.cfg.planner, "low_center_yaw_align_max_deg", 35.0)),
            )
            forward = quat_rotate_vec(aligned, (0.0, 0.0, 1.0))
            node.get_logger().debug(
                f"[CENTER_YAW] requested={math.degrees(raw_delta):+.1f}deg "
                f"applied={math.degrees(raw_delta - remaining):+.1f}deg "
                f"remaining={math.degrees(remaining):+.1f}deg "
                f"tool_forward=[{forward[0]:.3f},{forward[1]:.3f},{forward[2]:.3f}] "
                f"date_dir_xy=[{desired_xy[0]:.3f},{desired_xy[1]:.3f}]"
            )
            return aligned

        def _apply_date_axis_at_approach(orientation_in):
            """Rotate finger layout at APPROACH, then hold it through FINAL."""
            enabled = bool(getattr(
                node.cfg.planner, "approach_date_axis_enabled", True))
            angle = float(getattr(node, "fruit_major_axis_angle", 0.0))
            confidence = float(getattr(
                node, "fruit_major_axis_confidence", 0.0))
            stable = bool(getattr(node, "fruit_major_axis_stable", False))
            samples = int(getattr(node, "fruit_major_axis_samples", 0))
            spread_deg = float(getattr(
                node, "fruit_major_axis_spread_deg", float("inf")))
            min_conf = float(getattr(
                node.cfg.planner, "approach_date_axis_min_confidence", 0.20))
            max_deg = float(getattr(
                node.cfg.planner, "approach_date_axis_max_deg", 35.0))
            requested_deg = math.degrees(angle)
            applied_deg = float(np.clip(requested_deg, -max_deg, max_deg))
            if not enabled or confidence < min_conf or not stable:
                node.get_logger().debug(
                    f"[APPROACH_DATE_AXIS] applied=+0.0deg "
                    f"requested={requested_deg:+.1f}deg confidence={confidence:.2f} "
                    f"threshold={min_conf:.2f} stable={stable} samples={samples} "
                    f"spread={spread_deg:.1f}deg decision=KEEP_DEFAULT")
                return list(orientation_in)
            node.get_logger().debug(
                f"[APPROACH_DATE_AXIS] requested={requested_deg:+.1f}deg "
                f"applied={applied_deg:+.1f}deg confidence={confidence:.2f} "
                f"stable={stable} samples={samples} spread={spread_deg:.1f}deg "
                f"limit={max_deg:.1f}deg decision=APPLY_AT_APPROACH_HOLD_TO_FINAL")
            return quat_apply_local_z_roll(orientation_in, applied_deg)

        def _apply_candidate_roll_at_approach(orientation_in):
            """Apply a small scored finger-layout correction before motion."""
            node._approach_finger_roll_applied_deg = 0.0
            if not bool(getattr(
                    node.cfg.planner,
                    "approach_candidate_roll_enabled", True)):
                selection = getattr(node, "safe_grasp_candidate", None)
                requested_deg = (
                    float(selection.get("delta_deg", 0.0))
                    if selection else 0.0)
                node.get_logger().debug(
                    f"[APPROACH_FINGER_ROLL] requested={requested_deg:+.1f}deg "
                    "applied=+0.0deg decision=LOG_ONLY_DISABLED "
                    "orientation=CALIBRATED_FIXED")
                return list(orientation_in)
            if _active_corridor in getattr(
                    node, "_corridor_roll_disabled", set()):
                node.get_logger().debug(
                    f"[APPROACH_FINGER_ROLL] corridor={_active_corridor} "
                    "applied=+0.0deg decision=IK_FALLBACK_NO_ROLL")
                return list(orientation_in)
            selection = getattr(node, "safe_grasp_candidate", None)
            if not selection:
                node.get_logger().debug(
                    "[APPROACH_FINGER_ROLL] applied=+0.0deg decision=NO_CANDIDATE")
                return list(orientation_in)
            target_match = math.dist(
                [x, y, z],
                list(selection.get("target_xyz", [0.0, 0.0, 0.0])))
            requested_deg = float(selection.get("delta_deg", 0.0))
            score = float(selection.get("score", 0.0))
            minimum_score = float(selection.get("minimum_score", 0.0))
            match_limit = float(getattr(
                node.cfg.planner,
                "approach_candidate_roll_target_match_m", 0.05))
            min_score = float(getattr(
                node.cfg.planner,
                "approach_candidate_roll_min_score", 0.85))
            min_finger_score = float(getattr(
                node.cfg.planner,
                "approach_candidate_roll_min_finger_score", 0.75))
            max_deg = float(getattr(
                node.cfg.planner,
                "approach_candidate_roll_max_deg", 15.0))
            if (target_match > match_limit or score < min_score
                    or minimum_score < min_finger_score):
                node.get_logger().debug(
                    f"[APPROACH_FINGER_ROLL] requested={requested_deg:+.1f}deg "
                    "applied=+0.0deg decision=QUALITY_REJECT "
                    f"match={target_match*1000:.1f}mm "
                    f"Q={score:.3f} minF={minimum_score:.3f}")
                return list(orientation_in)
            applied_deg = float(np.clip(requested_deg, -max_deg, max_deg))
            node._approach_finger_roll_applied_deg = applied_deg
            node.get_logger().debug(
                f"[APPROACH_FINGER_ROLL] requested={requested_deg:+.1f}deg "
                f"applied={applied_deg:+.1f}deg limit={max_deg:.1f}deg "
                f"match={target_match*1000:.1f}mm Q={score:.3f} "
                f"minF={minimum_score:.3f} "
                "phase=APPROACH final=ORIENTATION_LOCKED")
            return quat_apply_local_z_roll(orientation_in, applied_deg)

        if skip_approach:
            node.reacquire_result = ""  # reset at start of each attempt
            pass  # jump straight to reacquire + final below

        elif (getattr(node, '_slip_retry_count', 0) > 0 and
              getattr(node, '_slip_retry_approach', None) is not None):
            # Slip retry: recompute approach XYZ from current fruit position (fruit may have shifted
            # after slip reacquire). For side-low, recompute orientation from current pose so the
            # gripper faces the fruit correctly rather than reusing stale stored orientation.
            _stored_approach = node._slip_retry_approach
            if is_low and is_side_approach:
                orientation = list(cur_quat) if cur_quat is not None else _stored_approach[3:]
            else:
                orientation = _stored_approach[3:]
            node._slip_retry_approach = None
            node._slip_retry_fruit_radius = None
            # fruit_radius already set from node.latest_fruit_radius at line 1847
            # — same source as the first attempt, so gripper opens to the same width
            # Recompute standoff using current x, y, z (same formula as is_low / MID/HIGH)
            if is_low:
                fruit_lat = lateral_value(x, y)
                _is_lat = abs(fruit_lat - trunk_lat) > LATERAL_THRESH
                if _is_lat and is_side_approach:
                    _x_off, _y_off, _z_off = low_side_standoff_offsets(
                        node, fruit_lat > trunk_lat)
                    low_side_dir = [-_x_off, -_y_off, 0.0]
                    _base_quat = list(cur_quat) if cur_quat is not None else _stored_approach[3:]
                    orientation, _yaw_delta = side_low_wrist3_orientation(node, _base_quat, low_side_dir)
                    node.get_logger().info(
                        f"Side-low wrist yaw correction: {_yaw_delta*57.3:+.1f}deg")
                    approach = [ax + _x_off, ay + _y_off, az + _z_off, *orientation]
                else:
                    _depth_off = (
                        _planner_value(
                            node.cfg.planner, "very_low_center_approach_depth_offset",
                            "very_low_center_approach_y_offset", 0.08)
                        if is_very_low_center
                        else _planner_value(
                            node.cfg.planner, "low_center_approach_depth_offset",
                            "low_center_approach_y_offset", 0.07)
                    )
                    _insert_pitch = math.radians(float(getattr(
                        node.cfg.planner, "low_center_insertion_pitch_deg", 12.0)))
                    _horizontal = math.cos(_insert_pitch)
                    _insert_dir = (
                        [_horizontal, 0.0, math.sin(_insert_pitch)]
                        if X_FORWARD_Y_LATERAL
                        else [0.0, -_horizontal, math.sin(_insert_pitch)])
                    _final_depth = _planner_value(
                        node.cfg.planner, "low_center_final_depth_offset",
                        "low_center_final_y_offset", 0.004)
                    _final_z = min(
                        getattr(node.cfg.planner, "low_center_final_z_offset", 0.025),
                        0.02)
                    _final_x, _final_y = add_axis_offsets(
                        x, y, depth=-_final_depth)
                    approach = [
                        _final_x - _depth_off * _insert_dir[0],
                        _final_y - _depth_off * _insert_dir[1],
                        z + _final_z - _depth_off * _insert_dir[2],
                        *orientation,
                    ]
                    orientation, _center_swing = align_local_axis_to_vector(
                        orientation, _insert_dir, local_axis=(0.0, 0.0, 1.0),
                        max_angle_deg=float(getattr(
                            node.cfg.planner, "low_center_forward_align_max_deg", 45.0)))
                    orientation = _apply_center_yaw(
                        orientation, approach[:3])
                    orientation = _apply_date_axis_at_approach(orientation)
                    if _dynamic_roll:
                        orientation = quat_apply_local_z_roll(
                            orientation, _center_tool_roll_deg)
                    approach[3:] = orientation
                    node.get_logger().info(
                        f"LOW/CENTER slip-retry insertion: pitch={math.degrees(_insert_pitch):.1f}deg "
                        f"standoff={_depth_off*1000:.0f}mm"
                        + (f" tool_roll={_center_tool_roll_deg:+.1f}deg "
                           f"source={_center_tool_roll_source} score={_roll_score:.2f}"
                           if _dynamic_roll else ""))
            else:
                if not is_side_approach:
                    _pitch_deg = getattr(node.cfg.planner, "mid_center_approach_pitch_deg", 0.0)
                    if abs(_pitch_deg) > 1e-6:
                        final_orientation_override = list(orientation)
                        orientation = quat_apply_local_x_pitch(orientation, _pitch_deg)
                approach = [ax, ay, az, *orientation]
            node.get_logger().info(
                f"Slip retry: stored approach was {[round(v,3) for v in _stored_approach[:3]]}, "
                f"recomputed from current fruit → {[round(v,3) for v in approach[:3]]} "
                f"fruit_radius={fruit_radius}")

        elif is_low:
            _pose_compute_t0 = time.time()
            fruit_lat = lateral_value(x, y)
            is_low_lateral = (
                side_approach_enabled and
                abs(fruit_lat - trunk_lat) > LATERAL_THRESH)
            if is_low_lateral and is_side_approach:
                # Side approach: combine configured depth and lateral standoffs.
                # Use one lateral direction for both APPROACH and FINAL so the wrist is already
                # facing the fruit before the short grasp move begins.
                _x_offset, _y_offset, _z_offset = low_side_standoff_offsets(
                    node, fruit_lat > trunk_lat)
                low_side_dir = [-_x_offset, -_y_offset, 0.0]
                _base_quat = list(cur_quat) if cur_quat is not None else list(target_quat)
                orientation, _yaw_delta = side_low_wrist3_orientation(node, _base_quat, low_side_dir)
                node.get_logger().info(
                    f"Side-low wrist yaw correction: {_yaw_delta*57.3:+.1f}deg")
                approach = [ax + _x_offset, ay + _y_offset, az + _z_offset, *orientation]
            else:
                side_blend = 0.10 if is_low_lateral else 0.25
                orientation = (list(cur_quat) if cur_quat is not None
                               else list(target_quat))
                _depth_offset = (
                    _planner_value(
                        node.cfg.planner, "very_low_center_approach_depth_offset",
                        "very_low_center_approach_y_offset", 0.08)
                    if is_very_low_center
                    else _planner_value(
                        node.cfg.planner, "low_center_approach_depth_offset",
                        "low_center_approach_y_offset", 0.07)
                )
                _insert_pitch = math.radians(float(getattr(
                    node.cfg.planner, "low_center_insertion_pitch_deg", 12.0)))
                _horizontal = math.cos(_insert_pitch)
                _center_nominal_dir = np.array(
                    [_horizontal, 0.0, math.sin(_insert_pitch)]
                    if X_FORWARD_Y_LATERAL else
                    [0.0, -_horizontal, math.sin(_insert_pitch)], dtype=float)
                _low_class = "VERY_LOW" if is_very_low_center else "LOW"
                _low_horizontal_dir, _low_corridor = (
                    select_mid_high_corridor_direction(
                        node, d_blend, class_label=_low_class))
                _active_corridor = _low_corridor
                # Preserve LOW's calibrated upward pitch while using the
                # selected date to choose only the horizontal corridor.
                _center_insert_dir = np.array([
                    _horizontal * _low_horizontal_dir[0],
                    _horizontal * _low_horizontal_dir[1],
                    math.sin(_insert_pitch),
                ], dtype=float)
                node.get_logger().info(
                    f"[APPROACH_ANGLE] class={_low_class} mode=CORRIDOR "
                    f"side={_low_corridor} pitch={math.degrees(_insert_pitch):.1f}deg "
                    f"nominal=[{_center_nominal_dir[0]:+.3f},"
                    f"{_center_nominal_dir[1]:+.3f},{_center_nominal_dir[2]:+.3f}] "
                    f"selected=[{_center_insert_dir[0]:+.3f},"
                    f"{_center_insert_dir[1]:+.3f},{_center_insert_dir[2]:+.3f}]")
                _center_final_depth = _planner_value(
                    node.cfg.planner, "low_center_final_depth_offset",
                    "low_center_final_y_offset", 0.004)
                _center_final_z = getattr(
                    node.cfg.planner, "low_center_final_z_offset", 0.025)
                _center_final_x, _center_final_y = add_axis_offsets(
                    x, y, depth=-_center_final_depth)
                approach = [
                    _center_final_x - _depth_offset * _center_insert_dir[0],
                    _center_final_y - _depth_offset * _center_insert_dir[1],
                    z + _center_final_z - _depth_offset * _center_insert_dir[2],
                    *orientation,
                ]
                node.get_logger().info(
                    f"[APPROACH_ORIGIN] class={_low_class} from={_low_corridor} "
                    f"origin=[{approach[0]:.3f},{approach[1]:.3f},"
                    f"{approach[2]:.3f}] "
                    f"final_grasp=[{_center_final_x:.3f},"
                    f"{_center_final_y:.3f},{z + _center_final_z:.3f}] "
                    f"travel_dir=[{_center_insert_dir[0]:+.3f},"
                    f"{_center_insert_dir[1]:+.3f},"
                    f"{_center_insert_dir[2]:+.3f}]")
                # Face the selected corridor before insertion. FINAL later
                # locks the orientation physically reached at APPROACH.
                orientation, _center_swing = align_local_axis_to_vector(
                    orientation,
                    _center_insert_dir,
                    local_axis=(0.0, 0.0, 1.0),
                    max_angle_deg=float(getattr(
                        node.cfg.planner,
                        "low_center_forward_align_max_deg", 45.0)),
                )
                orientation = _apply_candidate_roll_at_approach(orientation)
                if _tip_anchor is not None:
                    _tip_depth = abs(float(_center_final_depth))
                    _tip_standoff = float(_depth_offset)
                    _tip_anchored_final_grasp = (
                        _tip_anchor - _tip_depth * _center_insert_dir)
                    _tip_approach_closure = (
                        _tip_anchor
                        - (_tip_depth + _tip_standoff) * _center_insert_dir)
                    _tip_offset_tcp = list(getattr(
                        node.cfg.planner, "closure_center_offset_tcp_m",
                        [-0.004553, 0.000100, -0.016223]))
                    _tip_approach_tcp, _ = closure_center_corrected_tcp(
                        _tip_approach_closure, orientation, _tip_offset_tcp)
                    approach[:3] = list(_tip_approach_tcp)
                    node.get_logger().debug(
                        f"[TOOL_AXIS_AIM] class={_low_class} "
                        "source=DATE_TIP_3D geometry=COLLINEAR "
                        f"tip=[{_tip_anchor[0]:.3f},{_tip_anchor[1]:.3f},"
                        f"{_tip_anchor[2]:.3f}] goal_match={_tip_goal_match*1000:.1f}mm "
                        f"approach_closure=[{_tip_approach_closure[0]:.3f},"
                        f"{_tip_approach_closure[1]:.3f},{_tip_approach_closure[2]:.3f}] "
                        f"final_closure=[{_tip_anchored_final_grasp[0]:.3f},"
                        f"{_tip_anchored_final_grasp[1]:.3f},"
                        f"{_tip_anchored_final_grasp[2]:.3f}]")
                approach[3:] = orientation
                _x_offset = approach[0] - ax
                _y_offset = approach[1] - ay
                _z_offset = approach[2] - az
                _vlc_pitch = math.degrees(_insert_pitch)
                if _log_cycle_start:
                    _forward_axis = quat_rotate_vec(
                        orientation, (0.0, 0.0, 1.0))
                    node.get_logger().info(
                        "CENTER approach orientation: "
                        f"swing={math.degrees(_center_swing):.1f}deg "
                        f"local+Z_world=[{_forward_axis[0]:.3f},"
                        f"{_forward_axis[1]:.3f},{_forward_axis[2]:.3f}] "
                        f"insert_dir=[{_center_insert_dir[0]:.3f},"
                        f"{_center_insert_dir[1]:.3f},{_center_insert_dir[2]:.3f}] "
                        f"path_pitch={_vlc_pitch:.1f}deg "
                        f"standoff={_depth_offset*1000:.0f}mm "
                        "rotation_phase=APPROACH final=TRANSLATION_ONLY")
            if _log_cycle_start:
                node.get_logger().info(
                    f"{'VERY LOW' if is_very_low_center else 'LOW'} approach pose: {approach[:3]}, is_side={is_side_approach}, "
                    f"is_low_lateral={is_low_lateral}, x_offset={_x_offset:+.3f}, "
                    f"y_offset={_y_offset:+.3f}, z_offset={_z_offset:+.3f}"
                    + (f", pitch={_vlc_pitch:+.1f}deg" if is_very_low_center else "")
                    + (f", side_dir=[{low_side_dir[0]:.3f},{low_side_dir[1]:.3f},{low_side_dir[2]:.3f}]" if low_side_dir is not None else ""))
            _pose_compute_ms = (time.time() - _pose_compute_t0) * 1000.0
        else:
            side_blend =0.0
            orientation = minimize_rotation_orientation(cur_quat, target_quat, blend_weight=side_blend)
            _pitch_deg = 0.0
            if not is_side_approach:
                _pitch_deg = getattr(node.cfg.planner, "mid_center_approach_pitch_deg", 0.0)
                if abs(_pitch_deg) > 1e-6:
                    final_orientation_override = list(orientation)
                    orientation = quat_apply_local_x_pitch(orientation, _pitch_deg)
            orientation, _midhi_swing = align_local_axis_to_vector(
                orientation,
                _midhi_insert_dir,
                local_axis=(0.0, 0.0, 1.0),
                max_angle_deg=float(getattr(
                    node.cfg.planner,
                    "low_center_forward_align_max_deg", 45.0)),
            )
            if not is_side_approach:
                orientation = _apply_candidate_roll_at_approach(orientation)
            if _tip_anchor is not None:
                _tip_depth = abs(float(_midhi_final_depth))
                _tip_standoff = float(_midhi_standoff)
                _tip_anchored_final_grasp = (
                    _tip_anchor - _tip_depth * _midhi_insert_dir)
                _tip_approach_closure = (
                    _tip_anchor
                    - (_tip_depth + _tip_standoff) * _midhi_insert_dir)
                _tip_offset_tcp = list(getattr(
                    node.cfg.planner, "closure_center_offset_tcp_m",
                    [-0.004553, 0.000100, -0.016223]))
                _tip_approach_tcp, _ = closure_center_corrected_tcp(
                    _tip_approach_closure, orientation, _tip_offset_tcp)
                ax, ay, az = _tip_approach_tcp
                node.get_logger().debug(
                    "[TOOL_AXIS_AIM] class=MID_HIGH source=DATE_TIP_3D "
                    "geometry=COLLINEAR "
                    f"tip=[{_tip_anchor[0]:.3f},{_tip_anchor[1]:.3f},"
                    f"{_tip_anchor[2]:.3f}] goal_match={_tip_goal_match*1000:.1f}mm "
                    f"approach_closure=[{_tip_approach_closure[0]:.3f},"
                    f"{_tip_approach_closure[1]:.3f},{_tip_approach_closure[2]:.3f}] "
                    f"final_closure=[{_tip_anchored_final_grasp[0]:.3f},"
                    f"{_tip_anchored_final_grasp[1]:.3f},"
                    f"{_tip_anchored_final_grasp[2]:.3f}]")
            approach = [ax, ay, az, *orientation]
            if _log_cycle_start:
                _forward_axis = quat_rotate_vec(
                    orientation, (0.0, 0.0, 1.0))
                node.get_logger().info(
                    f"MID/HIGH approach pose: {approach[:3]}, is_side={is_side_approach}, blend={side_blend:.2f}, "
                    f"pitch={_pitch_deg:+.1f}deg insertion=LEVEL "
                    f"orientation_swing={math.degrees(_midhi_swing):.1f}deg "
                    f"tool_forward=[{_forward_axis[0]:+.3f},{_forward_axis[1]:+.3f},{_forward_axis[2]:+.3f}] "
                    "rotation_phase=APPROACH final=TRANSLATION_ONLY")

        def _final_offsets(is_slip_retry=False):
            if is_low and is_side_approach:
                _side = "left" if lateral_value(x, y) > trunk_lat else "right"
                _default_z = 0.020 if _side == "left" else 0.010
                return (
                    _planner_value(
                        node.cfg.planner, "low_side_final_depth_offset",
                        "low_side_final_y_offset", 0.0),
                    getattr(node.cfg.planner, f"low_{_side}_final_z_offset", _default_z),
                )
            if is_low:
                _z = getattr(node.cfg.planner, "low_center_final_z_offset", 0.025)
                if is_slip_retry:
                    _z = min(_z, 0.02)
                return (
                    _planner_value(
                        node.cfg.planner, "low_center_final_depth_offset",
                        "low_center_final_y_offset", 0.004),
                    _z,
                )
            return (
                _planner_value(
                    node.cfg.planner, "mid_center_final_depth_offset",
                    "mid_center_final_y_offset", 0.008),
                (getattr(node.cfg.planner, "mid_center_slip_final_z_offset", 0.020)
                 if is_slip_retry
                else getattr(node.cfg.planner, "mid_center_final_z_offset", 0.030)),
            )

        def _apply_mode_final_adjustment(position, pose_orientation):
            """Apply the selected grasp mode's FINAL-only depth and world-Z trim."""
            mode = str(getattr(
                node, "active_goal_grasp_mode", "NORMAL")).strip().upper()
            prefix = "envelop" if mode == "ENVELOP" else "normal"
            depth_extra = float(getattr(
                node.cfg.gripper, f"{prefix}_depth_extra_m", 0.0))
            z_extra = float(getattr(
                node.cfg.gripper, f"{prefix}_z_extra_m", 0.0))
            tool_axis = quat_rotate_vec(
                pose_orientation, (0.0, 0.0, 1.0))
            adjusted = [
                float(position[i]) + depth_extra * float(tool_axis[i])
                for i in range(3)
            ]
            adjusted[2] += z_extra
            return adjusted, mode, depth_extra, z_extra, tool_axis

        # Reject an unusable corridor before preview/confirmation and before
        # any robot motion. Validate the exact approach quaternion followed by
        # the closure-centre-corrected, fixed-orientation straight FINAL chain.
        if (not skip_approach and not is_side_approach
                and _active_corridor not in (None, "NONE")):
            _pf_depth, _pf_z = _final_offsets(is_slip_retry=False)
            _pf_gx, _pf_gy = add_axis_offsets(x, y, depth=-_pf_depth)
            _pf_grasp = (
                list(_tip_anchored_final_grasp)
                if _tip_anchored_final_grasp is not None
                else [_pf_gx, _pf_gy, z + _pf_z])
            _pf_grasp, _, _, _, _ = _apply_mode_final_adjustment(
                _pf_grasp, orientation)
            _pf_closure_offset = list(getattr(
                node.cfg.planner, "closure_center_offset_tcp_m",
                [-0.004553, 0.000100, -0.016223]))
            _pf_tcp, _ = closure_center_corrected_tcp(
                _pf_grasp, orientation, _pf_closure_offset)
            _pf_final = [*_pf_tcp, *orientation]
            _pf_start = _valid_joint_positions()
            _pf_ik_t0 = time.time()
            # The preflight result is a pure function of (start joints, approach
            # pose, final pose) -- the arm does not move between corridor tests.
            # The mirror-polarity evaluation tests A, then its mirror B, then
            # re-selects A, and used to re-solve A from scratch: measured in the
            # field as FROM_RIGHT_10 costing 1035ms and then 869ms again for a
            # byte-identical input and an identical branch_cost=20.1deg result.
            # Memoised per goal; the cache is dropped when the goal changes.
            _pf_cache = getattr(node, "_corridor_preflight_cache", None)
            if _pf_cache is None:
                _pf_cache = {}
                node._corridor_preflight_cache = _pf_cache
            # Start joints are NOT part of the key. Rounding them into a key
            # does not work on real hardware: the reported joint values jitter
            # in the low digits even while the arm is stationary, so every
            # lookup missed. Corridor evaluation is explicitly "without robot
            # motion", so instead hold the start pose the cache was built
            # against and invalidate wholesale if the arm has actually moved.
            _pf_cache_start = getattr(node, "_corridor_preflight_cache_start", None)
            if (_pf_cache_start is None
                    or len(_pf_cache_start) != len(_pf_start)
                    or max(abs(a - b) for a, b in
                           zip(_pf_cache_start, _pf_start)) > 0.002):
                _pf_cache.clear()
                node._corridor_preflight_cache_start = list(_pf_start)
            _pf_key = (
                str(_active_corridor),
                tuple(round(float(v), 5) for v in approach),
                tuple(round(float(v), 5) for v in _pf_final),
            )
            _pf_hit = _pf_cache.get(_pf_key)
            if _pf_hit is not None:
                _pf_ok, _pf_reason, _pf_hit_js, _pf_hit_cost = _pf_hit
                # Restore the two side effects the solver would have set, or the
                # caller would execute a stale/absent IK branch.
                node._corridor_preflight_approach_js = (
                    list(_pf_hit_js) if _pf_hit_js is not None else None)
                node._corridor_preflight_cost_deg = _pf_hit_cost
                _pf_reason = f"{_pf_reason}|CACHED"
            else:
                _pf_ok, _pf_reason = _preflight_approach_final_chain(
                    node, _pf_start, approach, _pf_final,
                    waypoint_count=max(1, int(getattr(
                        node.cfg.planner, "direct_final_cart_waypoints", 2))))
                _pf_cached_js = getattr(
                    node, "_corridor_preflight_approach_js", None)
                _pf_cache[_pf_key] = (
                    _pf_ok, _pf_reason,
                    list(_pf_cached_js) if _pf_cached_js is not None else None,
                    float(getattr(
                        node, "_corridor_preflight_cost_deg", float("inf"))),
                )
            _pf_ik_ms = (time.time() - _pf_ik_t0) * 1000.0
            _pose_compute_ms_str = (
                f"{_pose_compute_ms:.0f}" if _pose_compute_ms is not None else "NA")
            # Per-corridor PASS/REJECT with reason and branch cost. This is the
            # line that answers "would corridors 3..9 fail the same way", so
            # promote it to info when corridor logging is enabled instead of
            # leaving it at debug where it is invisible in field logs.
            _pf_log = (
                node.get_logger().info
                if getattr(node.cfg.planner, "log_corridor_candidates", False)
                else node.get_logger().debug)
            _pf_log(
                f"[CORRIDOR_PREFLIGHT] candidate={_active_corridor} "
                f"result={'PASS' if _pf_ok else 'REJECT'} reason={_pf_reason} "
                f"branch_cost={getattr(node, '_corridor_preflight_cost_deg', float('inf')):.1f}deg "
                f"pose_compute_ms={_pose_compute_ms_str} "
                f"ik_preflight_ms={_pf_ik_ms:.0f} "
                f"lock_wait_ms={getattr(node, '_corridor_preflight_lock_ms', 0.0):.0f} "
                f"solves_ms={[round(v) for v in getattr(node, '_corridor_preflight_solve_ms', [])]} "
                f"split_prep/call/sync={[tuple(round(x) for x in t) for t in getattr(node, '_corridor_preflight_split_ms', [])]} "
                f"call_cpu_ms={[round(v) for v in getattr(node, '_corridor_preflight_cpu_ms', [])]} "
                "stage=BEFORE_CONFIRM motion=NONE")

            # A fitted ellipse provides an undirected major axis: LEFT and
            # RIGHT signs are both geometrically valid. Preflight the mirrored
            # corridor before confirmation and select the safe branch with the
            # smaller maximum joint change from the current HOME branch.
            _pair = getattr(node, "_corridor_pair_eval", None)
            _is_axis_pair = bool(getattr(
                node, "_corridor_axis_polarity_ambiguous", False))
            _mirror = None
            if isinstance(_active_corridor, str):
                if _active_corridor.startswith("FROM_LEFT_"):
                    _mirror = _active_corridor.replace(
                        "FROM_LEFT_", "FROM_RIGHT_", 1)
                elif _active_corridor.startswith("FROM_RIGHT_"):
                    _mirror = _active_corridor.replace(
                        "FROM_RIGHT_", "FROM_LEFT_", 1)
            if _is_axis_pair and _mirror is not None:
                if _pair is None:
                    node._corridor_pair_eval = {
                        "first_label": _active_corridor,
                        "first_ok": bool(_pf_ok),
                        "first_reason": _pf_reason,
                        "first_cost": float(getattr(
                            node, "_corridor_preflight_cost_deg", float("inf"))),
                        "mirror_label": _mirror,
                        "done": False,
                    }
                    node._corridor_forced_label = _mirror
                    node.get_logger().debug(
                        "[CORRIDOR_PAIR] "
                        f"first={_active_corridor} "
                        f"result={'PASS' if _pf_ok else 'REJECT'}; "
                        f"testing_mirror={_mirror} motion=NONE")
                    node.goal_poses.insert(0, goal)
                    # Keep the lock and leave vision paused, exactly as the
                    # bounded corridor-retry paths below do. The arm does not
                    # move for a mirror test and no fresh detection is consumed,
                    # so resuming vision only to re-pause it next iteration adds
                    # a handshake per hop and puts YOLO/ZED depth back on the
                    # GPU while cuRobo is solving.
                    node._corridor_retry_keep_lock = True
                    continue
                if (not _pair.get("done", False)
                        and _active_corridor == _pair.get("mirror_label")):
                    _mirror_cost = float(getattr(
                        node, "_corridor_preflight_cost_deg", float("inf")))
                    _first_ok = bool(_pair.get("first_ok", False))
                    _first_cost = float(_pair.get("first_cost", float("inf")))
                    if _pf_ok and (not _first_ok or _mirror_cost < _first_cost):
                        _selected = _active_corridor
                        _selected_cost = _mirror_cost
                        _pair["done"] = True
                        node._corridor_forced_label = None
                    elif _first_ok:
                        _selected = str(_pair["first_label"])
                        _selected_cost = _first_cost
                        _pair["done"] = True
                        node._corridor_forced_label = _selected
                        node.get_logger().debug(
                            "[CORRIDOR_PAIR] "
                            f"first={_pair['first_label']}:{_first_cost:.1f}deg "
                            f"mirror={_active_corridor}:"
                            f"{'PASS' if _pf_ok else 'REJECT'}:"
                            f"{_mirror_cost:.1f}deg selected={_selected}; "
                            "reloading selected branch motion=NONE")
                        node.goal_poses.insert(0, goal)
                        # Same convention as above: reloading the already
                        # selected branch involves no robot motion.
                        node._corridor_retry_keep_lock = True
                        continue
                    else:
                        # Both failed; fall through to the existing bounded
                        # corridor retry logic below.
                        _selected = "NONE"
                        _selected_cost = float("inf")
                        _pair["done"] = True
                        node._corridor_forced_label = None
                    node.get_logger().debug(
                        "[CORRIDOR_PAIR] "
                        f"first={_pair['first_label']}:"
                        f"{'PASS' if _first_ok else 'REJECT'}:"
                        f"{_first_cost:.1f}deg mirror={_active_corridor}:"
                        f"{'PASS' if _pf_ok else 'REJECT'}:"
                        f"{_mirror_cost:.1f}deg selected={_selected} "
                        f"selected_cost={_selected_cost:.1f}deg motion=NONE")
            if not _pf_ok:
                _roll_applied = abs(float(getattr(
                    node, "_approach_finger_roll_applied_deg", 0.0))) > 0.1
                _roll_disabled = getattr(
                    node, "_corridor_roll_disabled", set())
                if _roll_applied and _active_corridor not in _roll_disabled:
                    _roll_disabled.add(_active_corridor)
                    node._corridor_roll_disabled = _roll_disabled
                    node.get_logger().warn(
                        f"[APPROACH_FINGER_ROLL] corridor={_active_corridor} "
                        f"rolled_preflight={_pf_reason}; retrying same corridor "
                        "with zero finger roll before excluding it")
                    node.goal_poses.insert(0, goal)
                    # Same target, no motion: retain the target lock and paused
                    # perception snapshot instead of paying another vision
                    # handshake/candidate evaluation on the retry.
                    node._corridor_retry_keep_lock = True
                    continue
                node._corridor_exclusions.add(_active_corridor)
                _tip_failure_limit = max(1, int(getattr(
                    node.cfg.planner,
                    "tool_axis_tip_fallback_after_corridors", 2)))
                if (_tip_anchor is not None
                        and len(node._corridor_exclusions)
                        >= _tip_failure_limit):
                    node._corridor_tip_fallback_active = True
                    node._corridor_exclusions = set()
                    node._corridor_roll_disabled = set()
                    node.get_logger().warn(
                        "[TOOL_AXIS_AIM_FALLBACK] "
                        f"tip-aligned preflight failed for {_tip_failure_limit} "
                        "corridors; retrying once from detected date centre "
                        "with calibrated closure offsets")
                    node.goal_poses.insert(0, goal)
                    # Retry the same frozen target without restarting vision.
                    node._corridor_retry_keep_lock = True
                    continue
                _centre_fallback = bool(getattr(
                    node, "_corridor_tip_fallback_active", False))
                _pf_limit = 3 if _centre_fallback else 9
                _pf_remaining = _pf_limit - len(node._corridor_exclusions)
                if _pf_remaining > 0:
                    node.get_logger().warn(
                        f"[CORRIDOR_RETRY] rejected={_active_corridor} "
                        f"reason=PREFLIGHT_{_pf_reason} "
                        f"remaining={_pf_remaining}; trying next-best "
                        f"{'bounded centre-fallback' if _centre_fallback else ''} corridor "
                        "without robot motion")
                    node.goal_poses.insert(0, goal)
                    node._corridor_retry_keep_lock = True
                else:
                    node.get_logger().warn(
                        ("Bounded centre fallback exhausted after 3 corridors; "
                         "rejecting unreachable goal without further search."
                         if _centre_fallback else
                         "All candidate corridors failed pre-motion IK validation."))
                    node._corridor_exclusions = set()
                    node._corridor_retry_goal = None
                    node._corridor_tip_fallback_active = False
                if _pf_remaining <= 0:
                    _vision_resume()
                    unlock_target(node)
                continue

            # Freeze the exact FINAL TCP predicted for this goal after corridor
            # and mode selection. The operator may cancel the preview, place
            # the robot manually, and measure manual-minus-predicted correction.
            node.final_tcp_teach_prediction = {
                "tcp_pose": list(_pf_final),
                "goal_xyz": [float(x), float(y), float(z)],
                "grasp_mode": str(getattr(
                    node, "active_goal_grasp_mode", "NORMAL")).upper(),
                "target_class": (
                    "VERY_LOW_CENTER" if is_very_low_center else
                    "LOW_CENTER" if is_low else "MID_HIGH_CENTER"),
                "corridor": str(_active_corridor),
                "created_time": time.time(),
            }
            node._active_motion_corridor = str(_active_corridor or "UNSPECIFIED")
            node._active_motion_target_class = str(
                node.final_tcp_teach_prediction["target_class"])
            node.final_tcp_teach_suggestion = None
            node.get_logger().debug(
                "[TEACH_FINAL_PREDICTED] "
                f"mode={node.final_tcp_teach_prediction['grasp_mode']} "
                f"class={node.final_tcp_teach_prediction['target_class']} "
                f"corridor={_active_corridor} "
                f"tcp=[{_pf_final[0]:.3f},{_pf_final[1]:.3f},"
                f"{_pf_final[2]:.3f}] ready=MEASURE_AFTER_CANCEL_AND_MANUAL_PLACE")
            node._publish_goal_info()

        # === DEBUG PLAN PREVIEW (RViz visualization) ===
        if node.cfg.planner.debug_plan_preview:
            preview_steps = []
            # 1. Current position (HOME or side HOME)
            if is_side_approach:
                side_label = (
                    "HOME_LEFT"
                    if lateral_value(x, y) > trunk_lateral(node)
                    else "HOME_RIGHT"
                )
                side_js = node.home_left_joints if "LEFT" in side_label else node.home_right_joints
                preview_steps.append({"label": side_label, "joints": side_js})
            else:
                preview_steps.append({"label": "HOME", "joints": node.home_joints})
            # 2. Approach
            if not skip_approach:
                preview_steps.append({"label": "APPROACH", "position": approach[:3]})
            # 3. Final
            _preview_depth_offset, _preview_z_offset = _final_offsets(
                is_slip_retry=False)
            _preview_x, _preview_y = add_axis_offsets(
                x, y, depth=-_preview_depth_offset)
            _preview_final_position = (
                list(_tip_anchored_final_grasp)
                if _tip_anchored_final_grasp is not None
                else [_preview_x, _preview_y, z + _preview_z_offset])
            _preview_final_position, _, _, _, _ = \
                _apply_mode_final_adjustment(
                    _preview_final_position, orientation)
            preview_steps.append({
                "label": "FINAL",
                "position": _preview_final_position,
            })
            # 4. Dropoff
            preview_steps.append({"label": "DROPOFF", "joints": node.dropoff_joints})
            # 5. Return HOME
            preview_steps.append({"label": "HOME", "joints": node.home_joints})

            markers_mod.publish_plan_preview(node, preview_steps)

            # Show EXACTLY the classification the approach will execute, so the preview
            # can never disagree with the planned motion (previously it recomputed both
            # labels from live vision and could flip across threshold boundaries between
            # acceptance and execution):
            #   height  -> the is_very_low_center / is_low decided above for planning
            #   lateral -> the stored goal_lateral_side that planning reuses (see ~L2823),
            #              not a fresh image_lateral_side() that can boundary-flip.
            if is_very_low_center:
                height_label = "VERY LOW"
            elif is_low:
                height_label = "LOW"
            else:
                height_label = "MID/HIGH"
            lateral_label = (
                "CENTER"
                if not side_approach_enabled
                else getattr(node, 'goal_lateral_side', None))
            if lateral_label is None:
                _img_norm = getattr(node, 'fruit_image_norm', None)
                if _img_norm is not None:
                    _cx, _cy = _img_norm
                    lateral_label = image_lateral_side(
                        node, _cx, _cy, force_very_low_center=is_very_low_center)
                else:
                    trunk_lat_val = trunk_lateral(node)
                    fruit_lat_val = lateral_value(x, y)
                    lateral_dist = abs(fruit_lat_val - trunk_lat_val)
                    lateral_label = (
                        ("LEFT" if fruit_lat_val > trunk_lat_val else "RIGHT")
                        if lateral_dist > LATERAL_THRESH
                        else "CENTER"
                    )
            node.get_logger().info(
                f"PLAN PREVIEW: {height_label} | {lateral_label} | "
                f"side={is_side_approach} | goal=[{x:.3f},{y:.3f},{z:.3f}] | "
                f"Waiting for confirm (GUI 'confirm'/'cancel' or keyboard ENTER/k)")

            # Wait for confirmation via event (set by keyboard thread or GUI command)
            node.plan_confirmed = None
            node.plan_confirm_event.clear()
            node.plan_waiting = True
            node.plan_confirm_event.wait()  # blocks until set
            node.plan_waiting = False
            markers_mod.clear_plan_preview(node)
            if not node.plan_confirmed:
                node.get_logger().info("Plan cancelled by user.")
                unlock_target(node)
                continue

        node._goal_stall_count = 0  # reset stall counter for each new goal

        _t_approach = time.time()
        if not skip_approach:
            node.reacquire_result = ""  # reset at start of each attempt
            node.motion_phase = "APPROACH"
            _preflight_candidate = None
            if is_very_low_center and not is_side_approach:
                _preflight_start = _valid_joint_positions()
                if _preflight_start is None:
                    node.get_logger().warn(
                        "No complete joint state for VERY LOW preflight; skipping goal")
                    _vision_resume()
                    unlock_target(node)
                    continue
                _preflight_candidate = _select_safe_approach_candidate(
                    node, approach, _preflight_start, goal_xyz=[x, y, z])
                if _preflight_candidate is None:
                    move_to_home_position(node)
                    if _check_stop(): break
                    _vision_resume()
                    unlock_target(node)
                    continue
                approach = list(_preflight_candidate["pose"])
                orientation = list(approach[3:])

            # HOME -> APPROACH is otherwise a joint interpolation. Its endpoint can
            # be labelled STRAIGHT while the TCP still swings sideways through the
            # fruit volume. Stage farther behind the selected approach first; only
            # the outer HOME -> ALIGNMENT leg may curve. The near-fruit
            # ALIGNMENT -> APPROACH leg is required to follow Cartesian waypoints.
            _alignment_cartesian_done = False
            _alignment_enabled = bool(getattr(
                node.cfg.planner, "approach_alignment_enabled", True))
            if _alignment_enabled and not is_side_approach:
                _toward_goal = np.asarray([x, y, z], dtype=float) - np.asarray(
                    approach[:3], dtype=float)
                _toward_norm = float(np.linalg.norm(_toward_goal))
                _alignment_extra = max(0.0, float(getattr(
                    node.cfg.planner, "approach_alignment_extra_m", 0.08)))
                if _toward_norm > 0.020 and _alignment_extra > 0.005:
                    _toward_axis = _toward_goal / _toward_norm
                    _alignment_xyz = (
                        np.asarray(approach[:3], dtype=float)
                        - _alignment_extra * _toward_axis)
                    _alignment_pose = [
                        float(_alignment_xyz[0]), float(_alignment_xyz[1]),
                        float(_alignment_xyz[2]), *list(approach[3:])]

                    # Cheap preflight: verify the near-fruit ALIGNMENT->APPROACH
                    # Cartesian entry leg (executed below via _direct_ik_move,
                    # require_cartesian=True) is actually reachable BEFORE paying
                    # for the real outer HOME->ALIGNMENT motion. Previously this
                    # was only discovered by physically driving out to ALIGNMENT
                    # (real motion, tens of cm) and back home again on failure --
                    # a full round trip for a fact knowable from geometry alone,
                    # since neither pose depends on how the arm got to ALIGNMENT.
                    # Mirrors _direct_ik_move's own "IK too far" check/threshold
                    # exactly (goals.py, _MAX_DIRECT_DELTA_RAD = 1.05rad when no
                    # preselected branch) so a corridor rejected here would have
                    # been rejected there too, just after a wasted 10s+ round trip.
                    if _preflight_candidate is None:
                        try:
                            _align_seed_js = _valid_joint_positions()
                            if _align_seed_js is not None:
                                _a_pos = torch.tensor(
                                    [_alignment_pose[:3]], dtype=torch.float32, device=device)
                                _a_quat = torch.tensor(
                                    [_alignment_pose[3:]], dtype=torch.float32, device=device)
                                _a_seed = torch.tensor(
                                    [_align_seed_js], dtype=torch.float32, device=device).unsqueeze(0)
                                _a_retract = torch.tensor(
                                    [_align_seed_js], dtype=torch.float32, device=device)
                                _a_result = node.motion_gen.ik_solver.solve_single(
                                    Pose(position=_a_pos, quaternion=_a_quat),
                                    seed_config=_a_seed, retract_config=_a_retract)
                                if bool(_a_result.success.item()):
                                    _align_js = nearest_joint_config(
                                        _align_seed_js,
                                        _a_result.js_solution.position.squeeze().cpu().tolist())
                                    _app_pos = torch.tensor(
                                        [approach[:3]], dtype=torch.float32, device=device)
                                    _app_quat = torch.tensor(
                                        [approach[3:]], dtype=torch.float32, device=device)
                                    _app_seed = torch.tensor(
                                        [_align_js], dtype=torch.float32, device=device).unsqueeze(0)
                                    _app_retract = torch.tensor(
                                        [_align_js], dtype=torch.float32, device=device)
                                    _app_result = node.motion_gen.ik_solver.solve_single(
                                        Pose(position=_app_pos, quaternion=_app_quat),
                                        seed_config=_app_seed, retract_config=_app_retract)
                                    if bool(_app_result.success.item()):
                                        _app_from_align_js = nearest_joint_config(
                                            _align_js,
                                            _app_result.js_solution.position.squeeze().cpu().tolist())
                                        _entry_delta = max(
                                            abs(a - b) for a, b in
                                            zip(_app_from_align_js, _align_js))
                                        if _entry_delta > 1.05:
                                            node.get_logger().warn(
                                                "[APPROACH_ALIGNMENT] entry-leg preflight: "
                                                f"ALIGNMENT->APPROACH needs {math.degrees(_entry_delta):.1f}deg "
                                                "(cap 60deg); rejecting corridor before outer motion")
                                            _vision_resume()
                                            unlock_target(node)
                                            continue
                        except Exception as _exc:
                            node.get_logger().debug(
                                f"[APPROACH_ALIGNMENT] entry-leg preflight skipped: {_exc}")

                    node.get_logger().info(
                        "[APPROACH_ALIGNMENT] "
                        f"outer=[{_alignment_pose[0]:.3f},"
                        f"{_alignment_pose[1]:.3f},{_alignment_pose[2]:.3f}] "
                        f"approach=[{approach[0]:.3f},{approach[1]:.3f},"
                        f"{approach[2]:.3f}] straight_leg="
                        f"{_alignment_extra * 1000.0:.0f}mm")
                    _alignment_start_joints = _valid_joint_positions()
                    if _alignment_start_joints is None:
                        _alignment_ok = False
                    else:
                        _alignment_start = JointState.from_position(
                            torch.tensor(
                                [_alignment_start_joints], dtype=torch.float32,
                                device=device),
                            joint_names=node.joint_order)
                        # Staging leg and the straight Cartesian entry are planned
                        # separately but published as ONE trajectory. Two separate
                        # messages structurally forced a full stop between them --
                        # build_trajectory() zeroes the first and last waypoint of
                        # every message, so the arm had to brake to rest at the
                        # staging pose and start again. Appending the entry leg here
                        # keeps a single continuous motion through the staging pose:
                        # no mid-way stop, no zero-velocity handoff to jerk against,
                        # and one planning round instead of two. The safety property
                        # is unchanged -- the last leg is still generated as straight
                        # Cartesian IK waypoints, just concatenated instead of sent
                        # on its own. Entry density is sized to hold that leg at
                        # approach_entry_duration_s even though the staging leg now
                        # runs fast (uniform dt => spacing is speed).
                        def _entry_leg(_last_js, _dt, _pose=list(_alignment_pose),
                                       _target=list(approach)):
                            _extra = straight_cartesian_entry_states(
                                node, _last_js, _pose[:3], _pose[3:], _target,
                                _dt, label="APPROACH_CARTESIAN")
                            if _extra:
                                # Reverse replays only the near-fruit leg, as before.
                                if not hasattr(node, 'stored_trajectory_states'):
                                    node.stored_trajectory_states = []
                                node.stored_trajectory_states.extend(_extra)
                            return _extra

                        _alignment_ok = plan_and_send(
                            node, _alignment_start,
                            Pose.from_list(_alignment_pose),
                            label="APPROACH_STAGED",
                            motion_type="alignment",
                            goal_xyz=_alignment_pose[:3],
                            store_trajectory=False,
                            extend_fn=_entry_leg)
                    if not _alignment_ok:
                        node.get_logger().error(
                            "[APPROACH_STAGED] staged approach failed (staging plan or "
                            "straight Cartesian entry); rejecting goal instead of "
                            "substituting a side-swing path")
                        move_to_home_position(node)
                        if _check_stop(): break
                        _vision_resume()
                        unlock_target(node)
                        continue
                    # One continuous motion now ends at the approach pose, so wait
                    # for that (not the staging pose) before FINAL takes over.
                    _approach_ok = wait_until_xyz(node, approach[:3], tol=0.008)
                    if _check_stop(): break
                    if not _approach_ok:
                        node.get_logger().error(
                            "[APPROACH_STAGED] approach pose not reached; rejecting goal")
                        move_to_home_position(node)
                        if _check_stop(): break
                        _vision_resume()
                        unlock_target(node)
                        continue
                    _alignment_cartesian_done = True

            _ap_dist = math.sqrt(sum((cur_pose[i] - approach[i])**2 for i in range(3))) if cur_pose else 999.0
            _skip_approach_wait = False
            if _log_phase_timings:
                node.get_logger().info(f"EE-to-approach distance: {_ap_dist*100:.1f}cm")
            if _alignment_cartesian_done:
                pass
            elif _preflight_candidate is not None:
                node._approach_clamp_rejected = False
                node._approach_safety_rejected = False
                node._approach_truncated = False
                _approach_ok = _direct_ik_move(
                    node,
                    approach,
                    label="APPROACH_PREFLIGHT",
                    motion_type="approach",
                    store_trajectory=True,
                    goal_js_override=_preflight_candidate["goal_js"],
                )
            elif _ap_dist < 0.20:
                if _log_phase_timings:
                    node.get_logger().info("Approach standoff close — using direct IK (skipping cuRobo plan)")
                node._approach_clamp_rejected = False
                node._approach_safety_rejected = False
                node._approach_truncated = False
                _corridor_approach_js = (
                    getattr(node, "_corridor_preflight_approach_js", None)
                    if (not is_side_approach
                        and _active_corridor not in (None, "NONE"))
                    else None)
                _approach_ok = _direct_ik_move(
                    node, approach, label="APPROACH",
                    motion_type="approach", store_trajectory=True,
                    goal_js_override=_corridor_approach_js)
                if not _approach_ok and not getattr(node, '_approach_clamp_rejected', False):
                    # IK failed (branch mismatch) — cuRobo handles branch switching via TRAJOPT.
                    # Path will be a detour but gets the arm to the correct approach position.
                    # Skip this retry if the failure was a clamping rejection — replanning gives
                    # the same tight configuration and wastes several seconds.
                    if _log_phase_timings:
                        node.get_logger().info(
                            f"IK failed — cuRobo plan to approach {[round(v,3) for v in approach[:3]]}"
                        )
                    _ik_fail_joints = _valid_joint_positions()
                    if _ik_fail_joints is None:
                        node.get_logger().warn("Incomplete joint state before approach fallback; skipping goal.")
                        unlock_target(node)
                        continue
                    _start_after_ik_fail = JointState.from_position(
                        torch.tensor([_ik_fail_joints], dtype=torch.float32, device=device),
                        joint_names=node.joint_order,
                    )
                    _approach_ok = plan_and_send(
                        node, _start_after_ik_fail,
                        Pose.from_list(approach),
                        label="APPROACH",
                        motion_type="approach",
                        goal_xyz=approach[:3],
                        store_trajectory=True,
                    )
                if (not _approach_ok and
                        not getattr(node, '_approach_clamp_rejected', False) and
                        not getattr(node, '_approach_safety_rejected', False)):
                    node.get_logger().warn("All approach attempts failed — re-homing then skipping to FINAL")
                    move_to_home_position(node)
                    if _check_stop(): break
                    _approach_ok = True
                    _skip_approach_wait = True
            else:
                node._approach_clamp_rejected = False
                node._approach_safety_rejected = False
                node._approach_truncated = False
                _approach_ok = plan_and_send(node, start, Pose.from_list(approach), label="APPROACH", motion_type="approach", goal_xyz=approach[:3], store_trajectory=True)
                if not _approach_ok and not getattr(node, '_approach_clamp_rejected', False):
                    # Fallback: direct IK move (different solver, handles cases plan_single can't).
                    # Skip if the failure was a clamping rejection — same config, same result.
                    node.get_logger().warn("APPROACH plan failed — retrying with direct IK...")
                    _approach_ok = _direct_ik_move(node, approach, label="APPROACH_IK",
                                                   motion_type="approach", store_trajectory=True)
            if _preflight_candidate is not None and not _approach_ok:
                node.get_logger().error(
                    "[PREFLIGHT] Validated branch failed during execution; "
                    "re-homing and skipping goal")
                move_to_home_position(node)
                if _check_stop(): break
                _vision_resume()
                unlock_target(node)
                continue
            if not _approach_ok:
                # A clamping/safety rejection means the chosen IK branch folds the
                # forearm toward the tool flange (self-clamp). Wrist-3 roll spins the
                # tool about its own approach axis and CANNOT change that forearm/flange
                # clearance. The pitch + IK-branch preflight does: it sweeps approach
                # pitch AND alternate IK branches, scoring forearm/flange clearance (and
                # also covers wrist variants). Reuse it here instead of wrist-only roll.
                if (getattr(node, '_approach_clamp_rejected', False) or
                        getattr(node, '_approach_safety_rejected', False)):
                    _safe_start = _valid_joint_positions()
                    _safe_candidate = (
                        _select_safe_approach_candidate(
                            node, approach, _safe_start, goal_xyz=[x, y, z])
                        if _safe_start is not None else None)
                    if _safe_candidate is not None:
                        node._approach_clamp_rejected = False
                        node._approach_safety_rejected = False
                        node._approach_truncated = False
                        approach = list(_safe_candidate["pose"])
                        orientation = list(approach[3:])
                        _approach_ok = _direct_ik_move(
                            node,
                            approach,
                            label="APPROACH_PREFLIGHT",
                            motion_type="approach",
                            store_trajectory=True,
                            goal_js_override=_safe_candidate["goal_js"],
                        )
                    if not _approach_ok:
                        node.get_logger().warn(
                            "APPROACH clamp-unsafe and no safe pitch/branch found — "
                            "re-homing and skipping goal")
                        move_to_home_position(node)
                        if _check_stop(): break
                        _vision_resume()
                        unlock_target(node)
                        continue
                else:
                    node.get_logger().warn("All APPROACH attempts failed — re-homing then skipping to FINAL")
                if not _approach_ok:
                    move_to_home_position(node)
                    if _check_stop(): break
                    _approach_ok = True
                    _skip_approach_wait = True
            # Truncated approach: arm stopped before standoff — skip the standoff wait
            # to avoid the full 10-second timeout waiting for a position the arm won't reach.
            if getattr(node, '_approach_truncated', False):
                _skip_approach_wait = True
            # Open gripper during approach motion (arm is already moving)
            if not gripper_opened:
                gripper_mod.control_gripper(node, "OPEN", fruit_radius=fruit_radius)
                gripper_opened = True
            if not _skip_approach_wait:
                _approach_reached = wait_until_xyz(
                    node, approach[:3], tol=0.008)
                if _check_stop(): break
                if not _approach_reached:
                    # Distinguish: arm never moved vs arm moved but stalled near target.
                    _cur_after = node.get_end_effector_pose()
                    _dist_to_approach = math.dist(_cur_after[:3], approach[:3]) if _cur_after else float('inf')
                    _truncated = getattr(node, '_approach_truncated', False)
                    if _dist_to_approach > 0.10 and not _truncated:
                        if _active_corridor not in (None, "NONE"):
                            node._corridor_exclusions.add(_active_corridor)
                            remaining = 9 - len(node._corridor_exclusions)
                        else:
                            remaining = 0
                        if remaining > 0:
                            node.get_logger().warn(
                                f"[CORRIDOR_RETRY] rejected={_active_corridor} "
                                f"reason=APPROACH_EXECUTION_FAILED "
                                f"distance={_dist_to_approach*100:.1f}cm "
                                f"remaining={remaining}; returning HOME and "
                                "trying next-best corridor")
                            node.goal_poses.insert(0, goal)
                        else:
                            node.get_logger().warn(
                                f"Approach did not execute (arm "
                                f"{_dist_to_approach*100:.0f}cm from target) "
                                "and no candidate corridors remain.")
                        _vision_resume()
                        unlock_target(node)
                        move_to_home_position(node)
                        if remaining > 0:
                            continue
                        break
                    if _log_phase_timings or _truncated:
                        node.get_logger().info(
                            f"{'[CLAMP-TRUNCATED] ' if _truncated else ''}Approach stopped {_dist_to_approach*100:.1f}cm from standoff — proceeding to FINAL from here.")
                log_path_deviation(node, "APPROACH")
                # wait_until_xyz can return as soon as the TCP enters tolerance,
                # slightly before the controller consumes the trajectory's
                # zero-velocity endpoint. Publishing a one-point blend/hold here
                # preempts that deceleration and creates a visible end jerk.
                time.sleep(float(getattr(
                    node.cfg.planner, "approach_endpoint_settle_s", 0.25)))

            # Wait for joint state to be available after approach
            _post_approach_joints = _valid_joint_positions()
            if _post_approach_joints is None:
                node.get_logger().warn("No joint state after approach; skipping goal.")
                unlock_target(node)
                continue

            start = JointState.from_position(
                torch.tensor([_post_approach_joints], dtype=torch.float32, device=device),
                joint_names=node.joint_order,
            )
        _timing["approach"] = time.time() - _t_approach
        if _log_phase_timings:
            node.get_logger().info(
                f"[TIMING] approach={_timing['approach']:.2f}s "
                f"skipped={skip_approach} gripper_opened={gripper_opened}")

        # Lock the orientation that the robot physically reached at APPROACH.
        # FINAL must be a translation-only insertion; asking it to correct even
        # a small residual quaternion error makes one fingertip sweep into the
        # fruit before the others. Preserve the +date-axis layout already reached.
        _measured_approach_orientation = None
        if (
            not skip_approach
            and bool(getattr(
                node.cfg.planner,
                "final_lock_measured_approach_orientation", True))
        ):
            _measured_approach_pose = node.get_end_effector_pose()
            if _measured_approach_pose and len(_measured_approach_pose) >= 7:
                _measured_approach_orientation = quat_normalize(
                    list(_measured_approach_pose[3:7]))
                _commanded_approach_orientation = quat_normalize(
                    list(approach[3:7]))
                _qdot = min(1.0, abs(float(np.dot(
                    _measured_approach_orientation,
                    _commanded_approach_orientation))))
                _orient_error_deg = math.degrees(2.0 * math.acos(_qdot))
                orientation = list(_measured_approach_orientation)
                node.get_logger().debug(
                    f"[ORIENTATION_LOCK] source=MEASURED_APPROACH "
                    f"command_error={_orient_error_deg:.2f}deg "
                    "FINAL=TRANSLATION_ONLY")

        # Ensure gripper is open before final approach (fallback for skip_approach case)
        if not gripper_opened:
            gripper_mod.control_gripper(node, "OPEN", fruit_radius=fruit_radius)
            gripper_opened = True

        if _check_stop(): break

        # 2. Soft reacquire — re-detection for accurate final position.
        # Side-home targets reacquire before approach from the side-home camera view,
        # so don't repeat it after moving to the side approach standoff.
        _t_reacq = time.time()
        _skip_reacq_low_center = False
        # A required reacquisition is always performed from the actual approach
        # pose.  A prior side-HOME reacquisition is useful for approach planning,
        # but it is not a substitute for confirming the fruit immediately before
        # final insertion.
        _require_final_reacquire = bool(getattr(
            node.cfg.planner, "require_reacquire_before_grasp", False))
        _do_reacq_after_approach = (
            bool(node.cfg.planner.reacquire_after_approach)
            and (_require_final_reacquire or not side_home_reacquired)
        )
        if _do_reacq_after_approach:
            _vision_resume()
        seed = [x, y, z]
        node.motion_phase = "REACQUIRE"
        if _do_reacq_after_approach:
            node.reacquire_result = "SEARCH"
            node._publish_goal_info()
            reacq = reacquire_goal_pose(
                node,
                seed_xyz=seed,
                candidate_seeds=[],   # tight to seed only — no roaming
                timeout=float(getattr(
                    node.cfg.planner, "reacquire_timeout_s", 1.0)),
                radius=0.03,          # 3cm — tighter than default 4cm
                # 10mm, not 2mm. The old gate sat below this function's own
                # noise floor: it accepts Z_STABLE_THRESH=12mm of scatter when
                # calling a reading "stable", then demanded 2mm agreement with
                # the seed. Field logs showed the date at dz=4mm and dz=8mm from
                # the seed -- good measurements, both outside a 2mm gate.
                z_tolerance=0.010,
                depth_settle_s=float(getattr(
                    node.cfg.planner, "reacquire_depth_settle_s", 0.10)),
                stable_needed=int(getattr(
                    node.cfg.planner, "reacquire_stable_frames", 2)),
            )
            if reacq:
                x, y, z = reacq
                publish_goal_marker(node, [x, y, z])
                node.get_logger().info(f"[REACQ] Refined: [{x:.3f}, {y:.3f}, {z:.3f}]")
                node.reacquire_result = "OK"
            else:
                x, y, z = seed
                if _log_cycle_start:
                    node.get_logger().info(f"[REACQ] No match — using original seed: [{x:.3f}, {y:.3f}, {z:.3f}]")
                node.reacquire_result = "SEED"
        else:
            if side_home_reacquired:
                if _log_cycle_start:
                    node.get_logger().info(
                        f"[REACQ] Skipped after approach — already reacquired at side HOME: "
                        f"[{x:.3f}, {y:.3f}, {z:.3f}]")
            else:
                node.reacquire_result = "SEED"
                if _log_cycle_start:
                    node.get_logger().info(
                        f"[REACQ] Skipped — using original seed: "
                        f"[{x:.3f}, {y:.3f}, {z:.3f}]")
        # Pause YOLO again — final move and grasp need full GPU for IK.
        _vision_pause()
        node._publish_goal_info()
        _timing["reacquire"] = time.time() - _t_reacq
        if _log_phase_timings:
            node.get_logger().info(
                f"[TIMING] reacquire={_timing['reacquire']:.2f}s "
                f"result={node.reacquire_result or 'NA'} searched={_do_reacq_after_approach} "
                f"fast_timeout={getattr(node.cfg.planner, 'reacquire_timeout_s', 1.0):.1f}s "
                f"side_home_reacq={side_home_reacquired}")

        if (
            _do_reacq_after_approach
            and node.reacquire_result != "OK"
            and _require_final_reacquire
        ):
            node.get_logger().warn(
                "[REACQ] Required close-view confirmation failed — "
                "skipping grasp instead of closing at the seed target.")
            unlock_target(node)
            continue
            
        if _check_stop(): break

        # If reacquired position drifted too far from seed, fall back to seed.
        # Large drift means a neighbouring fruit was matched or depth was unreliable.
        _drift_xy = math.hypot(x - seed[0], y - seed[1]) * 1000  # mm
        _drift_z  = abs(z - seed[2]) * 1000                       # mm
        if _drift_xy > 35.0 or _drift_z > 20.0:
            node.get_logger().warn(
                f"[REACQ] Large drift (XY={_drift_xy:.0f}mm Z={_drift_z:.0f}mm) — reverting to original seed "
                f"[{seed[0]:.3f},{seed[1]:.3f},{seed[2]:.3f}]")
            x, y, z = seed[0], seed[1], seed[2]

        # 3. Final slow precise grasp — IK + direct joint interpolation (no cuRobo trajectory)
        # Reuse approach orientation — approach and final must be consistent to avoid wrist flips.
        # A large orientation change between standoff and final forces IK through singularities.
        if _log_cycle_start:
            node.get_logger().info(f"FINAL orientation: reusing approach orientation (is_low={is_low})")
        _is_slip_retry = getattr(node, "_slip_retry_count", 0) > 0
        depth_offset, z_offset = _final_offsets(is_slip_retry=_is_slip_retry)
        fruit_radius = getattr(node, 'latest_fruit_radius', None) or 0.035
        # Recompute approach direction from current EE → reacquired fruit (not stale state_manager).
        # This ensures the approach vector is accurate after the arm has settled at standoff.
        _cur_ee = node.get_end_effector_pose()
        approach_dir = None
        if _cur_ee is not None:
            _dx = x - _cur_ee[0]; _dy = y - _cur_ee[1]; _dz = z - _cur_ee[2]
            _dist = math.sqrt(_dx**2 + _dy**2 + _dz**2)
            if _dist > 0.01:
                approach_dir = [_dx/_dist, _dy/_dist, _dz/_dist]
                if _log_cycle_start:
                    node.get_logger().info(
                        f"FINAL approach_dir recomputed from EE→fruit: "
                        f"[{approach_dir[0]:.3f},{approach_dir[1]:.3f},{approach_dir[2]:.3f}] dist={_dist:.3f}m"
                    )
        if approach_dir is None:
            _sm_dir = getattr(getattr(node, 'state_manager', None), 'fruit_direction', None)
            approach_dir = list(_sm_dir) if _sm_dir is not None else None
        _approach_orientation_for_final = list(orientation)
        if (final_orientation_override is not None
                and _measured_approach_orientation is None):
            orientation = list(final_orientation_override)
            if _log_cycle_start:
                node.get_logger().info(
                    "FINAL orientation: using unpitched MID/HIGH center orientation for insertion")
        _used_side_low_tilt = False
        if is_low and is_side_approach and approach_dir is not None:
            _front_tilt_cap = getattr(node.cfg.planner, "low_side_final_front_tilt_deg", 10.0)
            orientation, _front_tilt = align_local_axis_to_vector(
                orientation, approach_dir, local_axis=(0.0, 0.0, 1.0), max_angle_deg=_front_tilt_cap)
            _used_side_low_tilt = True
            if _log_cycle_start:
                node.get_logger().info(
                    f"FINAL orientation: side-low local +Z/front aligned toward fruit "
                    f"(tilt={_front_tilt*57.3:.1f}deg cap={_front_tilt_cap:.1f}deg)")

        # Deprecated FINAL-only approach yaw: disabled by default. Date-axis
        # orientation is now established at APPROACH and held through FINAL.
        # Face local +Z into the fruit using the
        # opposite of vision's outward fruit/surface direction. This changes
        # the insertion heading, not the three-finger roll about local +Z.
        # The direct FINAL interpolation blends from the unchanged APPROACH
        # orientation, and all existing IK/path fallback remains active.
        _pre_final_approach_yaw_orientation = list(orientation)
        _final_approach_yaw_applied = False
        _fruit_dir_for_yaw = getattr(node, "fruit_direction", None)
        if (
            bool(getattr(node.cfg.planner, "final_approach_yaw_enabled", True))
            and not is_side_approach
            and _fruit_dir_for_yaw is not None
        ):
            _fd = np.asarray(_fruit_dir_for_yaw, dtype=float)
            _fd_norm = float(np.linalg.norm(_fd))
            _desired_final_xy = [-float(_fd[0]), -float(_fd[1]), 0.0]
            _desired_xy_norm = math.hypot(
                _desired_final_xy[0], _desired_final_xy[1])
            if _fd_norm > 1e-6 and _desired_xy_norm > 0.20:
                _yaw_orientation, _yaw_raw, _yaw_remaining = (
                    bounded_yaw_align_local_axis(
                        orientation,
                        _desired_final_xy,
                        local_axis=(0.0, 0.0, 1.0),
                        max_delta_deg=float(getattr(
                            node.cfg.planner,
                            "final_approach_yaw_max_deg", 30.0)),
                    ))
                _yaw_applied = _yaw_raw - _yaw_remaining
                orientation = list(_yaw_orientation)
                _final_approach_yaw_applied = abs(_yaw_applied) > math.radians(0.1)
                _final_forward = quat_rotate_vec(
                    orientation, (0.0, 0.0, 1.0))
                node.get_logger().debug(
                    f"[FINAL_APPROACH_YAW] source=fruit_direction "
                    f"requested={math.degrees(_yaw_raw):+.1f}deg "
                    f"applied={math.degrees(_yaw_applied):+.1f}deg "
                    f"remaining={math.degrees(_yaw_remaining):+.1f}deg "
                    f"desired_xy=[{_desired_final_xy[0]:+.3f},"
                    f"{_desired_final_xy[1]:+.3f}] "
                    f"tool_forward=[{_final_forward[0]:+.3f},"
                    f"{_final_forward[1]:+.3f},{_final_forward[2]:+.3f}] "
                    "tool_roll=UNCHANGED")
            else:
                node.get_logger().warn(
                    "[FINAL_APPROACH_YAW] skipped: fruit direction has "
                    "insufficient horizontal component")

        if _tip_anchored_final_grasp is not None:
            gx, gy, gz = [float(v) for v in _tip_anchored_final_grasp]
        else:
            gx, gy = add_axis_offsets(x, y, depth=-depth_offset)
            gz = z + z_offset
        desired_grasp_xyz = [gx, gy, gz]
        (desired_grasp_xyz, _grasp_mode, _mode_depth_extra,
         _mode_z_extra, _mode_axis) = _apply_mode_final_adjustment(
            desired_grasp_xyz, orientation)
        gx, gy, gz = desired_grasp_xyz
        node.get_logger().debug(
            f"[GRIPPER_MODE] mode={_grasp_mode} "
            f"depth_extra={_mode_depth_extra*1000:+.1f}mm "
            f"z_extra={_mode_z_extra*1000:+.1f}mm "
            f"tool_axis=[{_mode_axis[0]:+.3f},"
            f"{_mode_axis[1]:+.3f},{_mode_axis[2]:+.3f}] "
            f"closure_target=[{gx:.3f},{gy:.3f},{gz:.3f}]")
        closure_offset_tcp = list(getattr(
            node.cfg.planner, "closure_center_offset_tcp_m",
            [-0.004553, -0.012559, -0.016223]))
        _baseline_final_orientation = list(orientation)
        _baseline_tcp_xyz, _baseline_offset_base = closure_center_corrected_tcp(
            desired_grasp_xyz, _baseline_final_orientation, closure_offset_tcp)
        _baseline_final_target = [
            *_baseline_tcp_xyz, *_baseline_final_orientation]
        _phase4_yaw_applied = False
        _selection = getattr(node, "safe_grasp_candidate", None)
        if (
            bool(getattr(
                node.cfg.planner, "phase4_safe_final_yaw_enabled", True))
            and _selection
            and bool(_selection.get("available", False))
        ):
            _candidate_match = math.dist(
                [x, y, z], list(_selection.get("target_xyz", [0.0, 0.0, 0.0])))
            _candidate_delta_deg = float(_selection.get("delta_deg", 0.0))
            _max_candidate_delta = float(getattr(
                node.cfg.planner, "phase4_safe_final_yaw_max_deg", 35.0))
            _candidate_match_limit = float(getattr(
                node.cfg.planner, "phase4_candidate_target_match_m", 0.05))
            if (
                _candidate_match <= _candidate_match_limit
                and abs(_candidate_delta_deg) <= _max_candidate_delta
            ):
                _candidate_orientation = quat_apply_local_z_roll(
                    _baseline_final_orientation, _candidate_delta_deg)
                _candidate_tcp_xyz, _candidate_offset_base = (
                    closure_center_corrected_tcp(
                        desired_grasp_xyz, _candidate_orientation,
                        closure_offset_tcp))
                _candidate_target = [
                    *_candidate_tcp_xyz, *_candidate_orientation]
                _candidate_ik_valid = False
                _candidate_ik_delta_deg = float("inf")
                _candidate_start_js = _valid_joint_positions()
                if (
                    _candidate_start_js is not None
                    and not getattr(node, "_cuda_faulted", False)
                ):
                    try:
                        _dev = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu")
                        _candidate_pose = Pose(
                            position=torch.tensor(
                                [_candidate_target[:3]], dtype=torch.float32,
                                device=_dev),
                            quaternion=torch.tensor(
                                [_candidate_target[3:]], dtype=torch.float32,
                                device=_dev),
                        )
                        _seed_t = torch.tensor(
                            [_candidate_start_js], dtype=torch.float32,
                            device=_dev).unsqueeze(0)
                        _retract_t = torch.tensor(
                            [_candidate_start_js], dtype=torch.float32,
                            device=_dev)
                        _ik = node.motion_gen.ik_solver.solve_single(
                            _candidate_pose, seed_config=_seed_t,
                            retract_config=_retract_t)
                        if _ik.success.item():
                            _candidate_js = nearest_joint_config(
                                _candidate_start_js,
                                _ik.js_solution.position.squeeze().cpu().tolist())
                            _candidate_ik_delta_deg = math.degrees(max(
                                abs(a - b) for a, b in zip(
                                    _candidate_js, _candidate_start_js)))
                            _candidate_ik_valid = _candidate_ik_delta_deg <= 60.0
                    except Exception as _candidate_ik_error:
                        node.get_logger().warn(
                            f"[PHASE4_YAW] IK preflight exception: "
                            f"{_candidate_ik_error}")
                node.get_logger().debug(
                    f"[PHASE4_YAW] safe_psi={_selection.get('psi_deg', 0.0):.1f}deg "
                    f"local_delta={_candidate_delta_deg:+.1f}deg "
                    f"target_match={_candidate_match*1000:.1f}mm "
                    f"score={_selection.get('score', 0.0):.3f} "
                    f"min={_selection.get('minimum_score', 0.0):.2f} "
                    f"loss={_selection.get('score_loss', 0.0):.3f} "
                    f"ik_valid={_candidate_ik_valid} "
                    f"ik_max_delta={_candidate_ik_delta_deg:.1f}deg")
                if _candidate_ik_valid:
                    orientation = _candidate_orientation
                    tcp_xyz = _candidate_tcp_xyz
                    closure_offset_base = _candidate_offset_base
                    final_target = _candidate_target
                    _phase4_yaw_applied = True
                else:
                    node.get_logger().warn(
                        "[PHASE4_YAW] candidate rejected by IK preflight; "
                        "using unchanged FINAL orientation")
            else:
                node.get_logger().warn(
                    f"[PHASE4_YAW] frozen candidate rejected: "
                    f"target_match={_candidate_match*1000:.1f}mm/"
                    f"{_candidate_match_limit*1000:.0f}mm "
                    f"delta={_candidate_delta_deg:+.1f}deg/"
                    f"{_max_candidate_delta:.1f}deg")
        if not _phase4_yaw_applied:
            orientation = _baseline_final_orientation
            tcp_xyz = _baseline_tcp_xyz
            closure_offset_base = _baseline_offset_base
            final_target = _baseline_final_target
        _goal_forward = x if X_FORWARD_Y_LATERAL else -y
        _target_forward = tcp_xyz[0] if X_FORWARD_Y_LATERAL else -tcp_xyz[1]
        _forward_delta_mm = (_target_forward - _goal_forward) * 1000.0
        node.get_logger().debug(
            f"[CLOSURE_CENTER] desired=[{gx:.3f},{gy:.3f},{gz:.3f}] "
            f"offset_tcp=[{closure_offset_tcp[0]:+.4f},{closure_offset_tcp[1]:+.4f},"
            f"{closure_offset_tcp[2]:+.4f}] "
            f"offset_base=[{closure_offset_base[0]:+.4f},{closure_offset_base[1]:+.4f},"
            f"{closure_offset_base[2]:+.4f}] "
            f"tcp=[{tcp_xyz[0]:.3f},{tcp_xyz[1]:.3f},{tcp_xyz[2]:.3f}]")
        if _log_cycle_start:
            node.get_logger().info(
                f"FINAL target offsets: depth=-{depth_offset:.3f} "
                f"z=+{z_offset:.3f} "
                f"grasp=[{gx:.3f},{gy:.3f},{gz:.3f}] "
                f"tcp=[{tcp_xyz[0]:.3f},{tcp_xyz[1]:.3f},{tcp_xyz[2]:.3f}]")
        else:
            node.get_logger().debug(
                f"[FINAL_TARGET] goal=[{x:.3f},{y:.3f},{z:.3f}] "
                f"grasp=[{gx:.3f},{gy:.3f},{gz:.3f}] "
                f"tcp=[{tcp_xyz[0]:.3f},{tcp_xyz[1]:.3f},{tcp_xyz[2]:.3f}] "
                f"forward_delta={_forward_delta_mm:+.0f}mm")
        node.motion_phase = "FINAL"
        # Vision was historically kept "paused" through the FINAL move, on the
        # grounds that cuRobo needs the GPU. It does not: by this point planning
        # is finished and the arm is replaying a trajectory computed a moment
        # ago, so nothing is competing. That pause is the reason the last ~7cm
        # is dead-reckoned.
        #
        # With observe enabled, vision runs through the leg and records what it
        # sees. It does NOT change the target -- see _final_inflight_monitor.
        _t_final = time.time()
        _inflight_report = {
            "frames": 0, "matched": 0, "rejected": 0, "no_candidate": 0,
            "below_min": 0, "above_max": 0, "usable": 0,
            "latest": None, "latest_d": 0.0, "latest_t": 0.0,
            "last_reject": "", "applied": 0, "applied_total": 0.0,
            "last_apply_t": 0.0,
        }
        _inflight_stop = threading.Event()
        _inflight_thread = None
        if bool(getattr(node.cfg.planner, "final_inflight_observe", False)):
            node._final_live_quat = list(final_target[3:7])
            node.set_vision_mode("reacquire_fast")
            # Compare against the FRUIT goal (x,y,z), not final_target.
            # final_target is the TCP pose, which sits ~28mm from the fruit by
            # design (grasp depth/z offsets plus the closure-centre correction).
            # Measuring fruit detections against it made every candidate read
            # as >25mm off, so the first run reported above_max=10 usable=0 --
            # an artefact of the wrong reference, not a real miss.
            _inflight_thread = threading.Thread(
                target=_final_inflight_monitor,
                args=(node, [float(x), float(y), float(z)], _inflight_stop,
                      _inflight_report),
                daemon=True)
            _inflight_thread.start()
        try:
            final_ok = _direct_ik_move(node, final_target, label="FINAL",
                                       motion_type="final", store_trajectory=True)
        finally:
            _inflight_stop.set()
            if _inflight_thread is not None:
                _inflight_thread.join(timeout=0.5)
                # Freeze the last crop BEFORE re-pausing vision. This is the
                # model input for a ready-to-close judgement: the date between
                # the fingertips at the moment the arm has finished closing in
                # and is about to grip. Taken here it is a few hundred ms old;
                # the recorder otherwise saves whatever survives until the
                # outcome fires, which is after the close AND the reverse --
                # measured at 2.9s stale, by which time the gripper has moved
                # and the image no longer describes the decision.
                try:
                    _cc = getattr(node, "_latest_grasp_crop", None)
                    node._grasp_crop_at_close = (
                        _cc.copy() if _cc is not None else None)
                    node._grasp_crop_at_close_meta = list(
                        getattr(node, "_latest_grasp_crop_meta", None) or [])
                    node._grasp_crop_at_close_age = round(
                        time.time() - float(getattr(
                            node, "_latest_grasp_crop_t", 0.0)), 3)
                except Exception:
                    node._grasp_crop_at_close = None
                # Restore the pause the rest of the sequence expects.
                node.set_vision_mode("paused")
                node.wait_for_vision_paused(timeout=0.30)
                node._last_inflight_report = dict(_inflight_report)
                _r = _inflight_report
                _latest = (
                    f" nearest_usable={_r['latest_d']*1000:.1f}mm"
                    f"@{_r['latest_t']:.2f}s"
                    if _r["latest"] is not None else " nearest_usable=NONE")
                node.get_logger().info(
                    "[FINAL_INFLIGHT] "
                    f"frames={_r['frames']} matched={_r['matched']} "
                    f"no_candidate={_r['no_candidate']} "
                    f"rejected_quality={_r['rejected']} "
                    f"below_min={_r['below_min']} above_max={_r['above_max']} "
                    f"usable={_r['usable']}{_latest} "
                    f"applied={_r['applied']} "
                    f"applied_total={_r['applied_total']*1000:.1f}mm "
                    f"apply={bool(getattr(node.cfg.planner, 'final_inflight_apply', False))}"
                    + (f" last_reject={_r['last_reject']}"
                       if _r["last_reject"] else ""))
        if not final_ok and _phase4_yaw_applied:
            node.get_logger().warn(
                "[PHASE4_YAW] candidate execution IK/path failed; retrying "
                "unchanged FINAL orientation with recomputed TCP")
            final_target = _baseline_final_target
            orientation = _baseline_final_orientation
            tcp_xyz = _baseline_tcp_xyz
            closure_offset_base = _baseline_offset_base
            _phase4_yaw_applied = False
            final_ok = _direct_ik_move(
                node, final_target, label="FINAL",
                motion_type="final", store_trajectory=True)
        if not final_ok and _final_approach_yaw_applied:
            _yaw_fallback_tcp, _yaw_fallback_offset = (
                closure_center_corrected_tcp(
                    desired_grasp_xyz,
                    _pre_final_approach_yaw_orientation,
                    closure_offset_tcp))
            _yaw_fallback_target = [
                *_yaw_fallback_tcp, *_pre_final_approach_yaw_orientation]
            node.get_logger().warn(
                "[FINAL_APPROACH_YAW] adjusted FINAL failed IK/path; "
                "retrying unchanged APPROACH orientation")
            final_ok = _direct_ik_move(
                node, _yaw_fallback_target, label="FINAL_YAW_FALLBACK",
                motion_type="final", store_trajectory=True)
            if final_ok:
                orientation = list(_pre_final_approach_yaw_orientation)
                tcp_xyz = list(_yaw_fallback_tcp)
                closure_offset_base = list(_yaw_fallback_offset)
                final_target = list(_yaw_fallback_target)
                _final_approach_yaw_applied = False
        if not final_ok:
            if _used_side_low_tilt:
                _fallback_tcp_xyz, _fallback_offset_base = closure_center_corrected_tcp(
                    desired_grasp_xyz, _approach_orientation_for_final,
                    closure_offset_tcp)
                approach_orientation_target = [
                    *_fallback_tcp_xyz, *_approach_orientation_for_final]
                node.get_logger().warn(
                    "FINAL tilted IK failed — retrying with original approach "
                    "orientation and its recomputed closure-centre TCP")
                final_ok = _direct_ik_move(
                    node, approach_orientation_target, label="FINAL_APPROACH_ORIENT",
                    motion_type="final", store_trajectory=True)
                if final_ok:
                    final_target = approach_orientation_target
        if (not final_ok and not bool(getattr(
                node.cfg.planner, "strict_final_cartesian_only", True))):
            # Fallback: full cuRobo plan_and_send (handles branch changes, no IK restriction)
            node.get_logger().warn("FINAL IK failed — trying plan_and_send as fallback...")
            _final_joints = _valid_joint_positions()
            if _final_joints is not None:
                _final_start = JointState.from_position(
                    torch.tensor([_final_joints], dtype=torch.float32, device=device),
                    joint_names=node.joint_order,
                )
                final_ok = plan_and_send(node, _final_start, Pose.from_list(final_target),
                    label="FINAL_PLAN", motion_type="final", goal_xyz=final_target[:3], store_trajectory=True)
                if final_ok:
                    final_ok = wait_until_xyz(
                        node, final_target[:3],
                        tol=float(getattr(
                            node.cfg.planner, "final_endpoint_tolerance", 0.004)))
        if not final_ok:
            if _active_corridor not in (None, "NONE"):
                node._corridor_exclusions.add(_active_corridor)
                remaining = 9 - len(node._corridor_exclusions)
                if remaining > 0:
                    node.get_logger().warn(
                        f"[CORRIDOR_RETRY] rejected={_active_corridor} "
                        f"reason=STRAIGHT_FINAL_IK remaining={remaining}; "
                        "returning HOME and trying next-best corridor")
                    node.motion_phase = "HOME"
                    move_to_home_position(node)
                    node.goal_poses.insert(0, goal)
                    unlock_target(node)
                    _vision_resume()
                    continue
            node.get_logger().warn(
                "All candidate corridors failed straight Cartesian FINAL — "
                "skipping grasp without closing.")
            node._corridor_exclusions = set()
            node._corridor_retry_goal = None
            unlock_target(node)
            continue
        log_path_deviation(node, "FINAL")
        _timing["final"] = time.time() - _t_final
        if _log_phase_timings:
            node.get_logger().info(
                f"[TIMING] final={_timing['final']:.2f}s ok={final_ok} "
                f"target=[{final_target[0]:.3f},{final_target[1]:.3f},{final_target[2]:.3f}]")

        # If stop was requested but arm is already at the final target, proceed with grasp.
        # E-stop during the wait does not mean the arm failed — check actual EE position.
        if getattr(node, '_stop_was_requested', False):
            _cur_ee = node.get_end_effector_pose()
            _dist_to_final = math.dist(_cur_ee[:3], final_target[:3]) if _cur_ee else float('inf')
            if _dist_to_final < 0.015:  # within 15mm — arm is at the goal, safe to grasp
                node.get_logger().info(
                    f"Stop was requested but arm is at final target ({_dist_to_final*100:.1f}cm) — proceeding with grasp.")
                node._stop_was_requested = False
            else:
                node.get_logger().warn(
                    f"Stop requested and arm is {_dist_to_final*100:.1f}cm from final target — aborting.")
                break

        if _check_stop(): break

        if approach_dir is not None:
            _cur_final = node.get_end_effector_pose()
            if _cur_final:
                _past = sum((_cur_final[i] - final_target[i]) * approach_dir[i] for i in range(3))
                _overshoot_thresh = getattr(node.cfg.planner, "final_overshoot_threshold", 0.004)
                if _past > _overshoot_thresh:
                    _max_backoff = getattr(node.cfg.planner, "final_overshoot_max_backoff", 0.012)
                    _backoff = min(_past, _max_backoff)
                    _corrected_final = [
                        _cur_final[i] - _backoff * approach_dir[i] for i in range(3)
                    ] + list(_cur_final[3:])
                    node.get_logger().info(
                        f"[FINAL_OVERSHOOT] past target by {_past*1000:.1f}mm — "
                        f"pulling back {_backoff*1000:.1f}mm before close")
                    _direct_ik_move(node, _corrected_final, label="FINAL_BACKOFF",
                                    motion_type="final", store_trajectory=True)

        _measured_final = node.get_end_effector_pose()
        if _measured_final:
            # Report against the goal the arm was actually driving to. An
            # in-flight retarget moves it, and printing the original target here
            # showed an error that is really the size of the correction -- seen
            # as "error=5.2mm" on a move the endpoint check (which does use the
            # retargeted goal) passed at 3.2mm. Two references in adjacent lines
            # reads like a near-miss when nothing is wrong.
            _reached_ref = getattr(node, "_final_retarget_xyz", None) or final_target[:3]
            _retarget_note = (
                f" retargeted_from=[{final_target[0]:.3f},"
                f"{final_target[1]:.3f},{final_target[2]:.3f}]"
                if getattr(node, "_final_retarget_xyz", None) else "")
            _final_error_mm = math.dist(
                _measured_final[:3], _reached_ref) * 1000.0
            node.get_logger().info(
                f"[FINAL_REACHED] target=[{_reached_ref[0]:.3f},{_reached_ref[1]:.3f},{_reached_ref[2]:.3f}] "
                f"actual=[{_measured_final[0]:.3f},{_measured_final[1]:.3f},{_measured_final[2]:.3f}] "
                f"error={_final_error_mm:.1f}mm{_retarget_note}")
        else:
            node.get_logger().warn(
                "[FINAL_REACHED] Actual TCP unavailable before gripper close")

        # Observation only: briefly resume live detection at the actual FINAL pose
        # so the vision overlay can verify the physical red fingertips around the
        # selected date. This stage never changes the target or commands motion.
        _verify_s = max(0.0, float(getattr(
            node.cfg.planner, "final_visual_verification_s", 0.7)))
        if _verify_s > 0.0:
            node.latest_fingertip_verification = None
            node.latest_fingertip_verification_time = 0.0
            _vision_resume()
            time.sleep(_verify_s)
            _vision_pause()
            _verification = getattr(node, "latest_fingertip_verification", None)

            # Phase 5 is deliberately observation-only. Convert the measured
            # image residual into a bounded image-plane suggestion for sign and
            # scale calibration; do not map it to robot axes or command motion.
            if bool(getattr(node.cfg.planner, "phase5_center_logging_enabled", True)):
                _match = re.search(
                    r"^(FIT|NOT_FIT) tips=3 containment=([0-9.]+) "
                    r"dx=([+-]?[0-9.]+)px dy=([+-]?[0-9.]+)px$",
                    _verification or "")
                if _match:
                    _state = _match.group(1)
                    _containment = float(_match.group(2))
                    _dx_px = float(_match.group(3))
                    _dy_px = float(_match.group(4))
                    _px_per_mm = max(0.1, float(getattr(
                        node.cfg.planner, "phase5_center_px_per_mm", 4.0)))
                    _limit_mm = max(0.0, float(getattr(
                        node.cfg.planner, "phase5_center_max_correction_mm", 3.0)))
                    _deadband_px = max(0.0, float(getattr(
                        node.cfg.planner, "phase5_center_deadband_px", 4.0)))
                    _du_mm = float(np.clip(_dx_px / _px_per_mm, -_limit_mm, _limit_mm))
                    _dv_mm = float(np.clip(_dy_px / _px_per_mm, -_limit_mm, _limit_mm))
                    _centered = (
                        abs(_dx_px) <= _deadband_px
                        and abs(_dy_px) <= _deadband_px
                        and _state == "FIT")
                    _result = "CENTERED" if _centered else "CORRECTION_SUGGESTED"
                    node.get_logger().debug(
                        f"[PHASE5_CENTER] mode=OBSERVE_ONLY result={_result} tips=3 "
                        f"containment={_containment:.2f} error_px=[{_dx_px:+.1f},{_dy_px:+.1f}] "
                        f"suggest_image_mm=[u:{_du_mm:+.1f},v:{_dv_mm:+.1f}] "
                        f"scale={_px_per_mm:.1f}px/mm clamp={_limit_mm:.1f}mm "
                        "robot_correction=NOT_APPLIED camera_to_tool_mapping=UNVALIDATED")
                else:
                    node.get_logger().debug(
                        "[PHASE5_CENTER] mode=OBSERVE_ONLY result=INCONCLUSIVE "
                        f"robot_correction=NOT_APPLIED reason={_verification or 'NO_RESULT'}")

        # Prepare the reverse prefix and kick off DROP-OFF planning BEFORE the
        # gripper closes, not just before the reverse motion. The plan depends
        # only on stored_trajectory_states (set during approach) and FK -- nothing
        # the grasp produces -- so it can start ~2.5s earlier and use the gripper
        # close window as well as the reverse. Previously it only had the ~2.5s
        # reverse to finish in, and a cuRobo plan contending with YOLO for the GPU
        # often overran that, leaving the arm standing still on the join() below
        # (the pause after reverse).
        # Reverse along the stored approach path.
        # Side-approach fruits need more clearance to clear the bunch before dropoff planning.
        _reverse_clearance = (
            float(getattr(node.cfg.planner, "side_home_partial_reverse_m", 0.30))
            if is_side_approach
            else float(getattr(node.cfg.planner, "center_partial_reverse_m", 0.20))
        )

        # Select the reverse endpoint before motion and plan DROP-OFF from that
        # exact joint state while the arm is physically reversing. This removes
        # the old stationary post-reverse planning pause.
        _prepared_reverse, _reverse_state_count = _prepare_partial_reverse_states(
            node, _reverse_clearance)
        import threading as _threading
        _dropoff_preplan_result = [None]
        _dropoff_plan_start = (
            list(node.home_joints) if is_side_approach
            else (list(_prepared_reverse[-1]) if _prepared_reverse else None)
        )

        def _bg_preplan_dropoff():
            try:
                if _dropoff_plan_start is None:
                    return
                _tgt = nearest_joint_config(_dropoff_plan_start, node.dropoff_joints)
                result = preplan_js(
                    node, _tgt, _dropoff_plan_start, "DROPOFF", "dropoff")
                _dropoff_preplan_result[0] = result
                if result is not None and not getattr(
                        node.cfg.planner, "concise_console_logs", False):
                    node.get_logger().info("[PREPLAN] Dropoff pre-plan ready during reverse")
                elif result is None:
                    node.get_logger().warn(
                        "[PREPLAN] Dropoff pre-plan returned no trajectory")
            except Exception as _e:
                node.get_logger().warn(f"[PREPLAN] Dropoff pre-plan failed: {_e}")

        _preplan_thread = _threading.Thread(
            target=_bg_preplan_dropoff, daemon=True)
        _preplan_thread.start()

        _t_grasp = time.time()
        node.control_gripper("CLOSE")
        if _check_stop():
            break
        # Notify vision system about grasp attempt for fruit tracking
        notify_grasp_attempt(node, final_target[:3])

        # Evaluate grasp using force profile (when did contact start during closure)
        gc = node.gripper_controller
        first_contact = getattr(gc, 'closure_first_contact_step', gc.steps)
        stopped_early = getattr(gc, 'closure_stopped_early', False)
        closure_step = getattr(gc, 'closure_step_stopped', gc.steps)
        deltas = getattr(gc, 'closure_deltas', [0.0, 0.0, 0.0])

        learner = getattr(node, 'grasp_learner', None)
        if learner and learner.enabled and learner.has_enough_data():
            prediction = learner.predict_grasp_success(first_contact, gc.steps, stopped_early)
            action = learner.suggest_action(first_contact, gc.steps, stopped_early)
            node.get_logger().info(f"[LEARNER] prediction={prediction}, action={action}, first_contact={first_contact}/{gc.steps}")
        else:
            # Sensor-driven: target contact at ~40% closure (step 4/10).
            # Too late  (>60%) = fruit at entrance → move forward.
            # Too early (<20%) = gripper overshot fruit → pull back.
            # Stopped early counts as proper regardless of contact step.
            contact_ratio = first_contact / max(gc.steps, 1)
            # Force confirmation: if ≥2 fingers show meaningful contact force,
            # the grip is physically real — don't disturb it even if timing is off.
            _force_confirmed = sum(1 for d in deltas if d > 1.5) >= 2
            if stopped_early or (0.20 <= contact_ratio <= 0.60) or _force_confirmed:
                prediction = "PROPER"
                action = "PROCEED"
                if _force_confirmed and not (stopped_early or (0.20 <= contact_ratio <= 0.60)):
                    node.get_logger().info(
                        f"Grip timing off ({contact_ratio:.0%}) but force confirms contact "
                        f"(deltas={[f'{d:.2f}' for d in deltas]}N) — skipping regrip"
                    )
            else:
                prediction = "NO_CONTACT"
                action = "REGRIP"

        if action == "REGRIP" and node.cfg.planner.regrip_after_slip:
            node.get_logger().warn(f"Weak grip ({prediction}) — re-gripping...")
            gripper_mod.control_gripper(node, "OPEN", fruit_radius=fruit_radius); time.sleep(0.1)
            cur = node.get_end_effector_pose()
            if cur:
                f0, f1, f2 = deltas[0], deltas[1], deltas[2]  # left, center, right

                # Lateral correction: imbalance between left (F0) and right (F2).
                # Keep this in semantic axes; add_axis_offsets maps it to the
                # old robot axes: X lateral, Y depth.
                lateral_imbalance = f0 - f2
                lateral_correction = -float(lateral_imbalance) * 0.008  # ~8mm per 1N imbalance
                lateral_correction = max(-0.02, min(0.02, lateral_correction))  # clamp ±20mm

                # Depth correction: target contact_ratio = 0.40 (step 4/10).
                # Negative = move forward (toward fruit), positive = pull back (away from fruit).
                # Both directions allowed: late contact → forward, early contact → back.
                contact_ratio = first_contact / max(gc.steps, 1)
                forward_correction = -(contact_ratio - 0.40) * 0.04  # 16mm range each direction
                forward_correction = max(-0.025, min(0.015, forward_correction))  # clamp ±

                # Vertical correction: if center (F1) much weaker than sides → fruit is below center
                center_vs_sides = f1 - (f0 + f2) / 2.0
                vertical_correction = -float(center_vs_sides) * 0.005  # small, ±5mm max
                vertical_correction = max(-0.01, min(0.01, vertical_correction))

                node.get_logger().info(
                    f"[REGRIP] F=[{f0:.2f},{f1:.2f},{f2:.2f}]N "
                    f"contact={first_contact}/{gc.steps} ({contact_ratio:.0%}) | "
                    f"corrections: lateral={lateral_correction*1000:+.1f}mm "
                    f"forward={forward_correction*1000:+.1f}mm "
                    f"vertical={vertical_correction*1000:+.1f}mm"
                )

                target_x, target_y = add_axis_offsets(
                    cur[0],
                    cur[1],
                    depth=forward_correction,
                    lateral=lateral_correction,
                )
                closer_target = [
                    target_x,
                    target_y,
                    cur[2] + vertical_correction,
                    *cur[3:]
                ]
                # Use _direct_ik_move — stays on the same kinematic branch.
                # exec_pose (full cuRobo planner) can plan a wild arc for a tiny correction.
                _direct_ik_move(node, closer_target, label="REGRIP",
                                motion_type="final", store_trajectory=True)
            node.control_gripper("CLOSE"); time.sleep(0.1)
            # Re-read after re-grip
            first_contact = getattr(gc, 'closure_first_contact_step', gc.steps)
            stopped_early = getattr(gc, 'closure_stopped_early', False)
            closure_step = getattr(gc, 'closure_step_stopped', gc.steps)
            deltas = getattr(gc, 'closure_deltas', [0.0, 0.0, 0.0])
            if learner and learner.enabled and learner.has_enough_data():
                prediction = learner.predict_grasp_success(first_contact, gc.steps, stopped_early)
            else:
                prediction = "PROPER" if (stopped_early or first_contact < gc.steps - 2) else "NO_CONTACT"
            node.get_logger().info(f"Re-grip result: {prediction}, first_contact={first_contact}/{gc.steps}")

        # Depth correction: nudge gripper forward (gripper closed) to bring contact
        # ratio closer to ideal 0.40. Only applied when grip is accepted (no regrip
        # needed) but fruit is shallower than ideal (ratio > 0.40).
        # Gripper stays closed — this deepens the cup around the fruit without releasing.
        contact_ratio_now = first_contact / max(gc.steps, 1)
        _depth_correction = -(contact_ratio_now - 0.40) * 0.04  # same formula as REGRIP
        _depth_correction = max(-0.020, min(0.0, _depth_correction))  # forward only, ≤20mm
        if abs(_depth_correction) > 0.003:  # only move if correction > 3mm
            _cur_for_depth = node.get_end_effector_pose()
            if _cur_for_depth:
                _depth_x, _depth_y = add_axis_offsets(
                    _cur_for_depth[0],
                    _cur_for_depth[1],
                    depth=_depth_correction,
                )
                _depth_target = [
                    _depth_x,
                    _depth_y,
                    _cur_for_depth[2],
                    *_cur_for_depth[3:]
                ]
                node.get_logger().info(
                    f"[DEPTH] contact_ratio={contact_ratio_now:.0%} → nudging "
                    f"{_depth_correction*1000:+.1f}mm forward to deepen grip"
                )
                _direct_ik_move(node, _depth_target, label="DEPTH_CORRECT",
                                motion_type="final", store_trajectory=True)
        _timing["grasp"] = time.time() - _t_grasp
        if _log_phase_timings:
            node.get_logger().info(
                f"[TIMING] grasp={_timing['grasp']:.2f}s prediction={prediction} action={action} "
                f"contact={first_contact}/{gc.steps} deltas=[{deltas[0]:.2f},{deltas[1]:.2f},{deltas[2]:.2f}]N")

        # Log grasp attempt for learning (success filled in by user feedback later)
        if learner:
            node.pending_grasp_record = GraspRecord(
                fruit_x=x, fruit_y=y, fruit_z=z,
                first_contact_step=first_contact,
                stopped_early=stopped_early,
                closure_step=closure_step,
                total_steps=gc.steps,
                delta_f0=deltas[0], delta_f1=deltas[1], delta_f2=deltas[2],
            )

        # Store approach + fruit_radius for potential slip retry
        node._slip_retry_approach = approach
        node._slip_retry_fruit_radius = fruit_radius

        # Keep one in-memory frame while the closed gripper is beside the bunch.
        # A second in-memory frame is captured after reverse for classification;
        # neither frame is written to disk.
        _grasp_pair_attempt_id = time.strftime("%Y%m%d_%H%M%S") + (
            f"_{int((time.time() % 1.0) * 1000):03d}")
        _grasp_pair_before_frame = None
        _grasp_pair_timeout = float(getattr(
            node.cfg.planner, "grasp_pair_capture_timeout_s", 0.25))
        _post_reverse_verify = bool(getattr(
            node.cfg.planner, "post_reverse_verification_enabled", False))
        if _post_reverse_verify:
            try:
                _grasp_pair_before_frame = node.capture_grasp_pair_frame(
                    _grasp_pair_attempt_id, "before_reverse",
                    timeout=_grasp_pair_timeout)
                # kept on the node so the episode recorder can persist it
                node._grasp_pair_before_frame = _grasp_pair_before_frame
            except Exception as _e:
                node.get_logger().warn(
                    f"[GRASP_PAIR] {_grasp_pair_attempt_id} before_reverse failed: {_e}")

        # 4. Drop-off and return
        if _check_stop():
            break
        # Wrist rotation to detach fruit from stem
        if _log_cycle_start:
            node.get_logger().info("Post-grip wrist rotation DISABLED for testing")
        # rotate_wrist(node, degrees=90, rotate_time=0.6, hold_time=0.1, return_time=1.5)
        # time.sleep(2.4)

        # Attempt CUDA recovery before dropoff/home planning
        if getattr(node, "_cuda_faulted", False):
            try_cuda_recovery(node)


        _t_reverse = time.time()
        node.motion_phase = "REVERSING"
        _reverse_endpoint_reached = execute_partial_reverse(
            node, clearance_m=_reverse_clearance,
            prepared_states=_prepared_reverse,
            original_state_count=_reverse_state_count)
        _grasp_pair_after_frame = None
        if _post_reverse_verify and _reverse_endpoint_reached:
            try:
                _grasp_pair_after_frame = node.capture_grasp_pair_frame(
                    _grasp_pair_attempt_id, "after_reverse",
                    timeout=_grasp_pair_timeout)
                # kept on the node so the episode recorder can persist it
                node._grasp_pair_after_frame = _grasp_pair_after_frame
            except Exception as _e:
                node.get_logger().warn(
                    f"[GRASP_PAIR] {_grasp_pair_attempt_id} "
                    f"after_reverse failed: {_e}")
        elif _post_reverse_verify:
            node.get_logger().warn(
                f"[GRASP_PAIR] {_grasp_pair_attempt_id} after_reverse NOT captured: "
                "reverse endpoint was not confirmed")

        # First temporal classifier: logging only.  No result from this block is
        # connected to retry, dropoff, or any other robot command.
        if (_grasp_pair_before_frame is not None and
                _grasp_pair_after_frame is not None):
            try:
                from ur10e_curobo.visual_grasp_verifier import classify_grasp_pair
                _temporal = classify_grasp_pair(
                    _grasp_pair_before_frame,
                    _grasp_pair_after_frame,
                )
                node.temporal_grasp_result = _temporal.label
                node.get_logger().debug(
                    f"[TEMPORAL_GRASP] {_temporal.label} | "
                    f"after_score={_temporal.after_score:.3f} "
                    f"appearance_corr={_temporal.appearance_correlation:.3f} | "
                    f"{_temporal.reason} (logging only; no automatic retry)"
                )
            except Exception as _e:
                node.temporal_grasp_result = "UNCERTAIN"
                node.get_logger().warn(
                    f"[TEMPORAL_GRASP] UNCERTAIN | classifier failed: {_e} "
                    "(logging only)")

        # No post-reverse force sampling in direct-handoff mode. Contact was
        # already classified at close; pausing here made the arm visibly wait
        # before DROP-OFF. Keep the field unknown so it cannot trigger a retry.
        node.fruit_held_after_reverse = None

        # Logging-only camera verification.  This is deliberately independent
        # of force classification and cannot trigger retry or robot motion.
        _skip_redundant_visual = bool(getattr(
            node.cfg.planner,
            "skip_redundant_visual_grasp_after_temporal", True))
        if (_post_reverse_verify
                and not (_skip_redundant_visual
                         and _grasp_pair_after_frame is not None)):
            try:
                _visual = node.verify_visual_grasp(frame_count=5, timeout=0.8)
                node.visual_grasp_result = _visual.label
                if not getattr(node.cfg.planner, "concise_console_logs", False):
                    node.get_logger().debug(
                        f"[VISUAL_GRASP] {_visual.label} | score={_visual.score:.3f} "
                        f"red_pixels={_visual.red_pixels} | {_visual.reason} "
                        "(logging only; no automatic retry)"
                    )
            except Exception as _e:
                node.visual_grasp_result = "UNCERTAIN"
                node.get_logger().warn(
                    f"[VISUAL_GRASP] UNCERTAIN | verifier failed: {_e} "
                    "(logging only)")
        _timing["reverse"] = time.time() - _t_reverse
        if _log_phase_timings:
            node.get_logger().info(
                f"[TIMING] reverse={_timing['reverse']:.2f}s "
                "post_reverse_hold=0.00s")

        _MAX_SLIP_RETRIES = int(getattr(
            node.cfg.planner, "max_force_grasp_retries", 2))
        _slip_retry_count = getattr(node, "_slip_retry_count", 0)
        _force_miss = (
            bool(getattr(node.cfg.planner, "force_retry_after_reverse", True))
            and getattr(node, "fruit_held_after_reverse", None) is False)
        _slip_detected = _force_miss
        if _force_miss:
            node.get_logger().warn(
                "[FORCE_RETRY] Post-reverse force indicates an empty/failed grasp")
        elif node.cfg.planner.slip_check_reacquire:
            node.get_logger().info(f"Slip check: querying depth at grasp=[{x:.3f},{y:.3f},{z:.3f}]")
            _slip_reacq = reacquire_goal_pose(
                node,
                seed_xyz=[x, y, z],
                candidate_seeds=[],
                timeout=3.0,
                radius=0.04,
                z_tolerance=0.15,
                depth_settle_s=2.0,
                stable_needed=3,
                restore_mode="paused",
                # _bg_preplan_dropoff (started right after the grasp, joined
                # below) may still be planning on the GPU. Keep YOLO throttled.
                planner_busy=True,
            )
            _slip_detected = _slip_reacq is not None
            node.get_logger().info(
                f"Slip check: {'SLIP at ' + str([round(v,3) for v in _slip_reacq]) if _slip_detected else 'OK — no fruit at grasp position'}")
        else:
            if _log_cycle_start:
                node.get_logger().info("Slip check disabled.")

        if _slip_detected and _slip_retry_count < _MAX_SLIP_RETRIES:
            _preplan_thread.join(timeout=float(getattr(
                node.cfg.planner, "dropoff_preplan_wait_s", 15.0)))
            if _preplan_thread.is_alive():
                node.get_logger().error(
                    "[PREPLAN] Dropoff planner did not finish before slip retry; "
                    "aborting automatic execution")
                node.motion_phase = "ERROR"
                _vision_resume()
                unlock_target(node)
                break
            node._slip_retry_count = _slip_retry_count + 1
            _retry_reason = "FORCE MISS" if _force_miss else "SLIP DETECTED"
            node.get_logger().warn(
                f"{_retry_reason}: target=[{x:.3f},{y:.3f},{z:.3f}] "
                f"(retry {node._slip_retry_count}/{_MAX_SLIP_RETRIES}) — "
                "retrying from reverse position")
            unlock_target(node)
            # Re-insert original x,y,z — approach was computed for this seed and will be reused
            _retry_goal = [x, y, z] + list(goal[3:])
            node.goal_poses.append(_retry_goal)
            # _slip_retry_approach already set above — reused in next iteration for approach step
            continue
        else:
            node._slip_retry_count = 0
            if _slip_detected:
                node.get_logger().warn(
                    "Grasp failure detected but max retries reached — proceeding to dropoff")

        # Flush any async CUDA errors that accumulated during FINAL IK/planning.
        # They surface at the next CUDA op — force them here so DROP-OFF gets a clean state.
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            node._cuda_faulted = True
        if getattr(node, "_cuda_faulted", False):
            try_cuda_recovery(node)

        # Reset fruit_obstacle sphere to below-floor park position — it was placed at the
        # fruit centroid when the goal arrived; after gripping the arm is at that same position,
        # so cuRobo sees the arm intersecting the sphere → INVALID_START_STATE_WORLD_COLLISION.
        try:
            node.obstacles.update_pose("fruit_obstacle", [0.0, 0.0, -10.0])
        except Exception:
            pass

        # For side-approach fruits, go to center HOME before dropoff.
        # The arm is lateral to the trunk — direct dropoff risks clipping the trunk.
        # Pre-plan (above) was computed from home_joints so it's valid after this move.
        if is_side_approach:
            node.get_logger().info(f"[DROPOFF] Side approach ({side_label}): going to center HOME to clear trunk")
            node.motion_phase = "HOME"
            if not move_to_home_position(node):
                node.get_logger().error(
                    "[DROPOFF] Could not reach center HOME; aborting automatic dropoff")
                node.motion_phase = "ERROR"
                _vision_resume()
                unlock_target(node)
                break

        # Do not race a second planner against the background pre-plan. The old
        # 3-second timeout expired before cuRobo returned, so its valid result was
        # ignored while another DROP-OFF plan started behind the same planning lock.
        _preplan_wait_s = float(getattr(
            node.cfg.planner, "dropoff_preplan_wait_s", 15.0))
        _preplan_thread.join(timeout=_preplan_wait_s)
        if _preplan_thread.is_alive():
            node.get_logger().error(
                f"[PREPLAN] Dropoff planning still running after "
                f"{_preplan_wait_s:.1f}s; aborting automatic dropoff")
            node.motion_phase = "ERROR"
            _vision_resume()
            unlock_target(node)
            break

        # Try direct dropoff — use pre-planned trajectory if available, else re-plan.
        # Only go to center HOME first if dropoff planning fails (trunk in path).
        _t_dropoff = time.time()
        _dropoff_direct = True
        node.motion_phase = "DROPOFF"
        _preplan_ok = False

        # Pre-plan the return HOME while the arm is physically doing DROP-OFF,
        # the same way DROP-OFF itself is pre-planned during the reverse. HOME
        # was costing a fresh ~1.05s plan with the arm standing still, even
        # though both ends are effectively fixed: the start is the drop-off
        # endpoint and the target is the constant home_joints. Dropping off takes
        # ~2.3s, which is ample. If it isn't ready (or fails) we simply fall back
        # to planning HOME the old way, so this can only save time.
        _home_preplan_result = [None]
        _home_plan_start = (
            list(_dropoff_preplan_result[0][1][-1])
            if _dropoff_preplan_result[0] is not None
            else list(node.dropoff_joints))

        def _bg_preplan_home():
            try:
                _tgt = nearest_joint_config(_home_plan_start, node.home_joints)
                _home_preplan_result[0] = preplan_js(
                    node, _tgt, _home_plan_start, "HOME", "home")
            except Exception as _e:
                node.get_logger().warn(f"[PREPLAN] HOME pre-plan failed: {_e}")

        _home_thread = _threading.Thread(target=_bg_preplan_home, daemon=True)
        _home_thread.start()

        if _dropoff_preplan_result[0] is not None:
            _pre_traj, _pre_states = _dropoff_preplan_result[0]
            _preplan_ok = execute_preplan(node, _pre_traj, _pre_states, "DROPOFF")
        _dropoff_ok = _preplan_ok
        if not _dropoff_ok:
            _dropoff_ok = move_to_dropoff_position(node)
        if not _dropoff_ok:
            _dropoff_direct = False
            node.get_logger().info("Direct dropoff failed — going to center HOME first to clear trunk")
            node.motion_phase = "HOME"
            _home_ok = move_to_home_position(node)
            if _home_ok:
                node.motion_phase = "DROPOFF"
                _dropoff_ok = move_to_dropoff_position(node)
        if not _dropoff_ok:
            node.get_logger().error(
                "[DROPOFF] Recovery failed; keeping gripper closed and ending "
                "automatic execution so manual HOME remains available")
            node.motion_phase = "ERROR"
            _vision_resume()
            unlock_target(node)
            break
        # Reset 2-finger mode before dropoff open (all fingers active for release)
        node.gripper_controller.frozen_fingers = set()
        node.control_gripper("OPEN")
        _timing["dropoff"] = time.time() - _t_dropoff
        if _log_phase_timings:
            node.get_logger().info(
                f"[TIMING] dropoff={_timing['dropoff']:.2f}s direct={_dropoff_direct}")

        if _check_stop(): 
            break

        # Smart return: if there are more goals, try direct approach instead of going home first
        _t_return_home = time.time()

        def _return_home():
            """Use the HOME trajectory pre-planned during DROP-OFF when it is
            ready; otherwise plan it now exactly as before."""
            _pre = None
            if _home_thread.is_alive():
                _home_thread.join(timeout=float(getattr(
                    node.cfg.planner, "dropoff_preplan_wait_s", 15.0)))
            if not _home_thread.is_alive():
                _pre = _home_preplan_result[0]
            if _pre is not None:
                _traj, _states = _pre
                if execute_preplan(node, _traj, _states, "HOME"):
                    return True
                node.get_logger().warn(
                    "[PREPLAN] HOME pre-plan did not execute; replanning")
            return move_to_home_position(node)
        if node.goal_poses:
            # Peek at next goal (don't pop it yet)
            next_goal = node.goal_poses.peek(0)
            if next_goal is not None:
                nx, ny, nz = next_goal[:3]
                next_quat = next_goal[3:]
                next_is_low = nz < LOW_Z_THRESH

                next_standoff = 0.12
                if next_is_low:
                    next_depth = _planner_value(
                        node.cfg.planner, "low_center_approach_depth_offset",
                        "low_center_approach_y_offset", 0.07)
                    next_z_offset = getattr(
                        node.cfg.planner, "low_center_approach_z_offset", -0.07)
                    nax, nay = add_axis_offsets(nx, ny, depth=next_depth)
                    naz = nz + next_z_offset
                else:
                    nax, nay = add_axis_offsets(nx, ny, depth=next_standoff)
                    naz = nz

                # Try planning from current (dropoff) position to next approach
                cur_joints = _valid_joint_positions()
                if cur_joints is not None:
                    next_start = JointState.from_position(
                        torch.tensor([cur_joints], dtype=torch.float32, device=device),
                        joint_names=node.joint_order,
                    )
                    cur_pose = node.get_end_effector_pose()
                    cur_quat = cur_pose[3:] if cur_pose else [1.0, 0.0, 0.0, 0.0]
                    next_orient = minimize_rotation_orientation(cur_quat, next_quat)
                    next_approach = [nax, nay, naz, *next_orient]

                    # Test plan (don't execute yet, just check feasibility).
                    # The resulting trajectory is ALWAYS discarded: this asks only
                    # "could the arm get near the next goal without going HOME
                    # first", and the real approach the main loop plans afterwards
                    # goes to a different pose (corridor selection, tool-axis
                    # aiming and the staging offset all still to be applied). So
                    # use the cheap reachability config rather than the polished
                    # one -- PLAN_CFG_SCAN_PREFLIGHT drops the finetune smoothing
                    # pass, which is the expensive part and is pure waste for a
                    # yes/no answer. Hold the YOLO lock for it as every other
                    # cuRobo call on this shared GPU does.
                    plan_cfg = PLAN_CFG_SCAN_PREFLIGHT
                    lock = getattr(node, '_planning_lock', None)
                    _yolo_t = getattr(node, 'yolo_thread', None)
                    _yolo_l = getattr(_yolo_t, 'inference_lock', None)
                    if _yolo_l: _yolo_l.acquire()
                    if lock: lock.acquire()
                    try:
                        test_res = node.motion_gen.plan_single(
                            next_start, Pose.from_list(next_approach), plan_cfg)
                    finally:
                        if lock: lock.release()
                        if _yolo_l: _yolo_l.release()

                    if test_res.success:
                        node.get_logger().info(
                            f"Direct path to next goal approach is COLLISION-FREE — skipping HOME")
                        # Don't go home, the main loop will handle approach planning
                    else:
                        node.get_logger().info(
                            f"Direct path to next goal BLOCKED — going HOME first")
                        node.motion_phase = "HOME"
                        _return_home()
                else:
                    node.motion_phase = "HOME"
                    _return_home()
            else:
                node.motion_phase = "HOME"
                _return_home()
        else:
            node.motion_phase = "HOME"
            _return_home()

        # Back at home — resume YOLO so it can detect the next fruit.
        _timing["return_home"] = time.time() - _t_return_home
        if _log_phase_timings:
            node.get_logger().info(f"[TIMING] return_home={_timing['return_home']:.2f}s")
        _vision_resume()

        # Wait for grasp feedback from RViz GUI (Y/N keys) or terminal
        if node.cfg.grasp.learning_enabled and hasattr(node, 'pending_grasp_record') and node.pending_grasp_record is not None:
            node._grasp_feedback = None  # reset
            node.get_logger().info("Waiting for grasp feedback (Y=success / N=fail)...")
            t0 = time.time()
            while node._grasp_feedback is None and (time.time() - t0) < 10.0:
                time.sleep(0.1)
            if node._grasp_feedback is not None:
                record = node.pending_grasp_record
                record.success = node._grasp_feedback
                node.grasp_learner.log_attempt(record)
                node.get_logger().info(f"Grasp logged: {'SUCCESS' if record.success else 'FAIL'}")
            else:
                node.get_logger().info("Grasp feedback timeout — skipping.")
            node._grasp_feedback = None
            node.pending_grasp_record = None

        _cycle_total = time.time() - _cycle_t0
        _timing_summary = " ".join(
            f"{k}={v:.2f}s" for k, v in _timing.items()
        )
        node.get_logger().info(
            f"[CYCLE {_cycle_idx}] DONE total={_cycle_total:.2f}s "
            f"force_signal={getattr(node, 'fruit_held_after_reverse', None)} "
            f"reacq={node.reacquire_result or 'NA'} {_timing_summary}")
        _append_cycle_log = getattr(node, "_append_harvest_log", None)
        if callable(_append_cycle_log):
            _append_cycle_log({
                "event": "CYCLE_COMPLETE",
                "cycle": _cycle_idx,
                "goal_xyz_m": [float(x), float(y), float(z)],
                "grasp_mode": str(getattr(
                    node, "active_goal_grasp_mode", "NORMAL")),
                "target_class": str(getattr(
                    node, "_active_motion_target_class", "UNKNOWN")),
                "selected_corridor": str(getattr(
                    node, "_active_motion_corridor", "UNSPECIFIED")),
                "tool_axis_result": str(getattr(
                    node, "_active_tool_axis_result", "UNKNOWN")),
                "tool_axis_reason": str(getattr(
                    node, "_active_tool_axis_reason", "UNKNOWN")),
                "tool_axis_tip_age_s": getattr(
                    node, "_active_tool_axis_tip_age_s", None),
                "tool_axis_goal_match_mm": getattr(
                    node, "_active_tool_axis_goal_match_mm", None),
                "grasp_prediction": str(prediction),
                "grasp_action": str(action),
                "first_contact_step": int(first_contact),
                "total_close_steps": int(gc.steps),
                "force_deltas": [float(value) for value in deltas[:3]],
                "fruit_held_after_reverse": getattr(
                    node, "fruit_held_after_reverse", None),
                "reacquire_result": str(node.reacquire_result or "NA"),
                "temporal_grasp_result": str(getattr(
                    node, "temporal_grasp_result", "NA")),
                "visual_grasp_result": str(getattr(
                    node, "visual_grasp_result", "NA")),
                "phase_timing_s": {
                    str(name): round(float(seconds), 3)
                    for name, seconds in _timing.items()
                },
                "motion_planning": (
                    node._motion_cycle_summary()
                    if callable(getattr(node, "_motion_cycle_summary", None))
                    else {}
                ),
                "total_s": round(float(_cycle_total), 3),
                "status": "COMPLETED",
            })

        node._last_completed_harvest_cycles += 1

        # Release target lock and reset tracking state for next goal
        unlock_target(node)
        node.reset_goal_tracking()
        node._corridor_forced_label = None
        node._corridor_pair_eval = None

        if len(node.goal_poses) == 0:
            # Cancel idle timer when cycle completes
            if hasattr(node, 'idle_timer'):
                try:
                    node.idle_timer.cancel()
                    del node.idle_timer
                except Exception:
                    pass
            node.get_logger().info("All goals completed; returned HOME.")
        else:
            node.get_logger().info("Preparing for next goal...")

    # Always resume YOLO when the grasp loop exits — break, stop, or normal completion.
    _vision_resume()
    # HOME describes the last commanded motion, not the post-cycle state. Return
    # to IDLE so safe runtime settings are not permanently rejected after Done.
    if getattr(node, "motion_phase", "") != "ERROR":
        node.motion_phase = "IDLE"
    node.get_logger().info(f"Done. run_time={time.time() - _run_t0:.2f}s cycles={_cycle_idx}")


def quaternion_from_approach(node, direction_xyz=None, pitch_deg=None, world=False):
    import numpy as np, math
    if pitch_deg is not None:
        cur = node.get_end_effector_pose()
        if cur: qw, qx, qy, qz = cur[3], cur[4], cur[5], cur[6]
        else:   qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        half = math.radians(pitch_deg)/2.0
        cw, cx, cy, cz = math.cos(half), math.sin(half), 0.0, 0.0
        if world:
            nw = cw*qw - cx*qx - cy*qy - cz*qz
            nx = cw*qx + cx*qw + cy*qz - cz*qy
            ny = cw*qy - cx*qz + cy*qw + cz*qx
            nz = cw*qz + cx*qy - cy*qx + cz*qw
        else:
            nw = qw*cw - qx*cx - qy*cy - qz*cz
            nx = qw*cx + qx*cw + qy*cz - qz*cy
            ny = qw*cy - qx*cz + qy*cw + qz*cx
            nz = qw*cz + qx*cy - qy*cx + qz*cw
        return [nw, nx, ny, nz]
    d = np.array(direction_xyz or (0.0, 0.0, -1.0), dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9: return [1.0, 0.0, 0.0, 0.0]
    d /= n
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.cross(up, d); x_axis = x_axis/np.linalg.norm(x_axis) if np.linalg.norm(x_axis) >= 1e-6 else np.array([1.0,0.0,0.0])
    y_axis = np.cross(d, x_axis); y_axis /= np.linalg.norm(y_axis)
    z_axis = d
    Rm = np.array([[x_axis[0], y_axis[0], z_axis[0]],[x_axis[1], y_axis[1], z_axis[1]],[x_axis[2], y_axis[2], z_axis[2]]], dtype=float)
    t = Rm[0,0] + Rm[1,1] + Rm[2,2]
    if t>0.0:
        s = math.sqrt(t+1.0)*2.0; qw = 0.25*s
        qx = (Rm[2,1]-Rm[1,2])/s; qy = (Rm[0,2]-Rm[2,0])/s; qz = (Rm[1,0]-Rm[0,1])/s
    else:
        if Rm[0,0]>Rm[1,1] and Rm[0,0]>Rm[2,2]:
            s = math.sqrt(1.0+Rm[0,0]-Rm[1,1]-Rm[2,2])*2.0; qw = (Rm[2,1]-Rm[1,2])/s; qx = 0.25*s
            qy = (Rm[0,1]+Rm[1,0])/s; qz = (Rm[0,2]+Rm[2,0])/s
        elif Rm[1,1]>Rm[2,2]:
            s = math.sqrt(1.0+Rm[1,1]-Rm[0,0]-Rm[2,2])*2.0; qw = (Rm[0,2]-Rm[2,0])/s
            qx = (Rm[0,1]+Rm[1,0])/s; qy = 0.25*s; qz = (Rm[1,2]+Rm[2,1])/s
        else:
            s = math.sqrt(1.0+Rm[2,2]-Rm[0,0]-Rm[1,1])*2.0; qw = (Rm[1,0]-Rm[0,1])/s
            qx = (Rm[0,2]+Rm[2,0])/s; qy = (Rm[1,2]+Rm[2,1])/s; qz = 0.25*s
    return [qw, qx, qy, qz]
