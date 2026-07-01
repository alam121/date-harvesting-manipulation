# ruff: noqa
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import math
import numpy as np
import rclpy
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from .fk import forward_kinematics_batch


def publish_planned_path(node, joint_states, label="planned", cartesian_points=None):
    """Publish planned trajectory as a line strip marker (blue).

    If cartesian_points is provided, uses those directly (avoids recomputing FK).
    """
    points = cartesian_points if cartesian_points else forward_kinematics_batch(node, joint_states)

    if not points:
        return

    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "planned_path"
    m.id = hash(label) % 10000  # Different ID per label
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.008  # Thinner than actual path
    m.color.a = 0.8
    m.color.b = 1.0  # Blue for planned
    m.color.r = 0.2
    m.points = points
    m.lifetime.sec = 0  # Persist until cleared
    node.path_marker_pub.publish(m)
    if getattr(node.cfg.planner, "log_path_publish", False):
        node.get_logger().info(f"Published planned path ({len(points)} points) for {label}")


def clear_path_markers(node):
    """Delete all planned path markers in RViz."""
    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "planned_path"
    m.action = Marker.DELETEALL
    node.path_marker_pub.publish(m)
    # Also clear the live robot path trail
    node.path_points.clear()
    m2 = Marker()
    m2.header.frame_id = "base_link"
    m2.header.stamp = node.get_clock().now().to_msg()
    m2.ns = "robot_path"
    m2.action = Marker.DELETEALL
    node.path_marker_pub.publish(m2)


def clear_goal_markers(node):
    """Delete all queued-goal markers (spheres, approach arrows, labels) in RViz."""
    stamp = node.get_clock().now().to_msg()
    for ns in ("goal_positions", "goal_approach", "goal_labels"):
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = ns
        m.action = Marker.DELETEALL
        node.goal_marker_pub.publish(m)


def publish_goal_marker(node, position, rank=None):
    stamp = node.get_clock().now().to_msg()
    goal_pos = np.array(position, dtype=float)

    # Try to render heatmap as the goal marker
    heatmap_data = getattr(node, '_latest_heatmap_data', None)
    if heatmap_data and len(heatmap_data) >= 4:
        # Parse flat array: [x0,y0,z0,s0, x1,y1,z1,s1, ...]
        n_pts = len(heatmap_data) // 4
        pts = np.array(heatmap_data[:n_pts * 4]).reshape(n_pts, 4)

        # Compute offset so heatmap centers on goal position
        centroid = pts[:, :3].mean(axis=0)
        offset = goal_pos - centroid

        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = "goal_positions"
        m.id = len(node.goal_poses)
        m.type = Marker.SPHERE_LIST
        m.action = Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.008
        m.pose.orientation.w = 1.0

        for i in range(n_pts):
            p = Point()
            p.x = float(pts[i, 0] + offset[0])
            p.y = float(pts[i, 1] + offset[1])
            p.z = float(pts[i, 2] + offset[2])
            m.points.append(p)
            s = float(pts[i, 3])
            c = ColorRGBA()
            c.a = 0.9
            if s < 0.5:
                c.r, c.g, c.b = 0.0, s * 2.0, 1.0 - s * 2.0
            else:
                c.r, c.g, c.b = (s - 0.5) * 2.0, 1.0 - (s - 0.5) * 2.0, 0.0
            m.colors.append(c)
        node.goal_marker_pub.publish(m)

        # Arrow for approach direction
        fruit_dir = getattr(node, 'fruit_direction', None)
        if fruit_dir is not None:
            arrow = Marker()
            arrow.header.frame_id = "base_link"
            arrow.header.stamp = stamp
            arrow.ns = "goal_approach"
            arrow.id = 0
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.008
            arrow.scale.y = 0.015
            arrow.scale.z = 0.02
            arrow.color.r = 1.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0
            start = Point(x=position[0], y=position[1], z=position[2])
            end = Point(
                x=position[0] + fruit_dir[0] * 0.15,
                y=position[1] + fruit_dir[1] * 0.15,
                z=position[2] + fruit_dir[2] * 0.15,
            )
            arrow.points.append(start)
            arrow.points.append(end)
            node.goal_marker_pub.publish(arrow)
    else:
        # Fallback: solid red sphere
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = "goal_positions"
        m.id = len(node.goal_poses)
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = position
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.a, m.color.r = 1.0, 1.0
        node.goal_marker_pub.publish(m)

    if rank is not None:
        t = Marker()
        t.header.frame_id = "base_link"
        t.header.stamp = stamp
        t.ns = "goal_labels"
        t.id = 1000 + len(node.goal_poses)
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = position[0]
        t.pose.position.y = position[1]
        t.pose.position.z = position[2] + 0.06
        t.scale.z = 0.05
        t.color.a = t.color.r = t.color.g = t.color.b = 1.0
        t.text = str(rank)
        node.goal_marker_pub.publish(t)


def publish_plan_preview(node, steps):
    """Publish plan preview markers in RViz.

    steps: list of dicts with keys:
        label: str          — e.g. "HOME", "HOME_RIGHT", "APPROACH", "FINAL", "DROPOFF"
        position: [x,y,z]   — Cartesian position (or None to compute from joints)
        joints: [j1..j6]    — joint positions (used if position is None)
        color: (r,g,b,a)    — marker color
    """
    from .fk import forward_kinematics
    stamp = node.get_clock().now().to_msg()
    prev_pt = None

    # Colors per stage
    COLORS = {
        "HOME":       (0.2, 0.8, 0.2, 1.0),   # green
        "HOME_LEFT":  (0.2, 0.8, 0.8, 1.0),   # cyan
        "HOME_RIGHT": (0.2, 0.8, 0.8, 1.0),   # cyan
        "APPROACH":   (0.0, 0.5, 1.0, 1.0),   # blue
        "FINAL":      (1.0, 0.3, 0.0, 1.0),   # orange
        "DROPOFF":    (0.8, 0.0, 0.8, 1.0),   # purple
    }

    for i, step in enumerate(steps):
        label = step["label"]
        pos = step.get("position")
        joints = step.get("joints")
        r, g, b, a = step.get("color") or COLORS.get(label, (1.0, 1.0, 1.0, 1.0))

        # Resolve position
        if pos is None and joints is not None:
            fk_pt = forward_kinematics(node, joints)
            if fk_pt:
                pos = [fk_pt.x, fk_pt.y, fk_pt.z]
        if pos is None:
            node.get_logger().warn(f"Plan preview: skipping step '{label}' — position is None")
            continue

        # Sphere marker for this waypoint
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = "plan_preview"
        m.id = i * 2
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = pos[2]
        m.pose.orientation.w = 1.0
        sz = 0.06 if label in ("APPROACH", "FINAL") else 0.04
        m.scale.x = m.scale.y = m.scale.z = sz
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
        m.lifetime.sec = 30
        node.goal_marker_pub.publish(m)

        # Text label above sphere
        t = Marker()
        t.header.frame_id = "base_link"
        t.header.stamp = stamp
        t.ns = "plan_preview"
        t.id = i * 2 + 1
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = pos[0]
        t.pose.position.y = pos[1]
        t.pose.position.z = pos[2] + 0.06
        t.scale.z = 0.04
        t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = f"{i+1}. {label}"
        t.lifetime.sec = 30
        node.goal_marker_pub.publish(t)

        # Connecting line to previous waypoint
        cur_pt = Point(x=pos[0], y=pos[1], z=pos[2])
        if prev_pt is not None:
            line = Marker()
            line.header.frame_id = "base_link"
            line.header.stamp = stamp
            line.ns = "plan_preview_lines"
            line.id = i
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.005
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 1.0, 1.0, 0.5
            line.points.append(prev_pt)
            line.points.append(cur_pt)
            line.lifetime.sec = 30
            node.goal_marker_pub.publish(line)
        prev_pt = cur_pt

    labels = [s["label"] for s in steps]
    if getattr(node.cfg.planner, "log_path_publish", False):
        node.get_logger().info(f"Published plan preview with {len(steps)} steps in RViz: {labels}")


def clear_plan_preview(node):
    """Clear all plan preview markers from RViz."""
    stamp = node.get_clock().now().to_msg()
    for ns in ("plan_preview", "plan_preview_lines"):
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = ns
        m.action = Marker.DELETEALL
        node.goal_marker_pub.publish(m)


def publish_lidar_scan_preview(node, targets, *, valid: bool = True):
    """Publish a short-lived LiDAR scan arc preview without touching plan_preview."""
    stamp = node.get_clock().now().to_msg()
    for ns in (
        "lidar_scan_preview",
        "lidar_scan_preview_lines",
        "lidar_scan_preview_points",
        "lidar_scan_preview_labels",
    ):
        delete = Marker()
        delete.header.frame_id = "base_link"
        delete.header.stamp = stamp
        delete.ns = ns
        delete.action = Marker.DELETEALL
        node.goal_marker_pub.publish(delete)

    color = (0.1, 1.0, 0.35, 0.95) if valid else (0.1, 0.9, 1.0, 0.75)
    arc_line = Marker()
    arc_line.header.frame_id = "base_link"
    arc_line.header.stamp = stamp
    arc_line.ns = "lidar_scan_preview_lines"
    arc_line.id = 0
    arc_line.type = Marker.LINE_STRIP
    arc_line.action = Marker.ADD
    arc_line.pose.orientation.w = 1.0
    arc_line.scale.x = 0.008
    arc_line.color.r, arc_line.color.g, arc_line.color.b, arc_line.color.a = color
    arc_line.lifetime.sec = 2

    for i, target in enumerate(targets):
        pose = target.get("pose") if isinstance(target, dict) else target
        if pose is None or len(pose) < 3:
            continue
        pos = [float(pose[0]), float(pose[1]), float(pose[2])]

        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = "lidar_scan_preview_points"
        m.id = i
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = pos[2]
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.045
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime.sec = 2
        node.goal_marker_pub.publish(m)

        t = Marker()
        t.header.frame_id = "base_link"
        t.header.stamp = stamp
        t.ns = "lidar_scan_preview_labels"
        t.id = i
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x = pos[0]
        t.pose.position.y = pos[1]
        t.pose.position.z = pos[2] + 0.105
        t.scale.z = 0.050
        t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = f"S{i + 1}"
        t.lifetime.sec = 2
        node.goal_marker_pub.publish(t)

        arc_line.points.append(Point(x=pos[0], y=pos[1], z=pos[2]))

    if len(arc_line.points) >= 2:
        node.goal_marker_pub.publish(arc_line)


def publish_reachability_cloud(node, points, deltas_deg, directions=None):
    """Publish current-posture reachability samples as an RViz colored point cloud."""
    pub = getattr(node, "reachability_marker_pub", None)
    if pub is None:
        return

    stamp = node.get_clock().now().to_msg()
    if not points:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = "reachability_cloud"
        m.action = Marker.DELETEALL
        pub.publish(m)
        return

    planner = node.cfg.planner
    green_cap = float(getattr(
        planner, "safe_zone_verified_interp_max_delta_deg", 80.0))
    yellow_cap = float(getattr(
        planner, "shortest_ik_plan_max_delta_deg", 80.0))
    max_cap = float(getattr(
        planner, "reachability_cloud_max_delta_deg",
        getattr(planner, "goal_reachability_skip_delta_deg", 100.0)))
    size = float(getattr(planner, "reachability_cloud_point_size_m", 0.025))
    period = float(getattr(planner, "reachability_cloud_period_s", 1.0))

    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = stamp
    m.ns = "reachability_cloud"
    m.id = 0
    m.type = Marker.SPHERE_LIST
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = size
    m.lifetime.sec = max(1, int(math.ceil(period * 2.5)))

    # Reachable points lying on/near the scan arc get a darker green so the operator can see
    # which green samples the lidar sweep will actually pass through. Lazy import avoids the
    # markers<->lidar_scan circular import.
    arc_desc = None
    arc_tol = 0.04
    try:
        if bool(getattr(node.cfg.lidar_scan, "arc_highlight_enabled", True)):
            from . import lidar_scan as _ls
            arc_desc = _ls.arc_descriptor(node)
            arc_tol = float(getattr(node.cfg.lidar_scan, "arc_highlight_tol_m", 0.04))
    except Exception:
        arc_desc = None

    for p, delta in zip(points, deltas_deg):
        m.points.append(p)
        c = ColorRGBA()
        c.a = 0.38
        if delta <= green_cap:
            if arc_desc is not None and _ls.point_on_arc(arc_desc, p.x, p.y, p.z, arc_tol):
                c.r, c.g, c.b = 0.0, 0.40, 0.0   # dark green: on/near the scan arc
                c.a = 0.90
            else:
                c.r, c.g, c.b = 0.18, 0.70, 0.24
                c.a = 0.46
        elif delta <= yellow_cap:
            c.r, c.g, c.b = 0.90, 0.72, 0.16
        else:
            t = min(1.0, max(0.0, (delta - yellow_cap) / max(max_cap - yellow_cap, 1.0)))
            c.r, c.g, c.b = 0.90, 0.36 * (1.0 - t), 0.10
            c.a = 0.28
        m.colors.append(c)
    pub.publish(m)

    d = Marker()
    d.header.frame_id = "base_link"
    d.header.stamp = stamp
    d.ns = "reachability_directions"
    d.id = 0
    d.type = Marker.LINE_LIST
    d.action = Marker.ADD if directions else Marker.DELETE
    d.pose.orientation.w = 1.0
    d.scale.x = 0.004
    d.color.r, d.color.g, d.color.b, d.color.a = 0.20, 0.58, 0.72, 0.45
    d.lifetime.sec = m.lifetime.sec
    length = float(getattr(planner, "reachability_direction_length_m", 0.08))
    if directions:
        for p, direction in zip(points, directions):
            if direction is None:
                continue
            norm = math.sqrt(sum(float(v) * float(v) for v in direction))
            if norm < 1e-6:
                continue
            d.points.append(p)
            e = Point()
            e.x = p.x + float(direction[0]) / norm * length
            e.y = p.y + float(direction[1]) / norm * length
            e.z = p.z + float(direction[2]) / norm * length
            d.points.append(e)
    pub.publish(d)

    t = Marker()
    t.header.frame_id = "base_link"
    t.header.stamp = stamp
    t.ns = "reachability_cloud"
    t.id = 1
    t.type = Marker.TEXT_VIEW_FACING
    t.action = Marker.ADD
    t.pose.orientation.w = 1.0
    t.pose.position.x = min(p.x for p in points)
    t.pose.position.y = min(p.y for p in points)
    t.pose.position.z = max(p.z for p in points) + 0.08
    t.scale.z = 0.035
    t.color.r = t.color.g = t.color.b = t.color.a = 1.0
    t.text = (
        f"Reachability from current posture: green <= {green_cap:.0f}deg, "
        f"yellow <= {yellow_cap:.0f}deg")
    t.lifetime.sec = m.lifetime.sec
    pub.publish(t)


def publish_path_marker(node):

    m = Marker(); m.header.frame_id = "base_link"; m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "robot_path"; m.id = 0
    m.type = Marker.LINE_STRIP; m.action = Marker.ADD
    m.scale.x = 0.01; m.color.a = 1.0; m.color.g = 1.0
    m.points = list(node.path_points)  # snapshot — prevents concurrent clear() from freeing Points during C++ serialization
    node.path_marker_pub.publish(m)


def track_robot_path(node):

    try:
        tf = node.tf_buffer.lookup_transform(
            "base_link", "gripper_tip", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
        )
        p = Point(
            x=float(tf.transform.translation.x),
            y=float(tf.transform.translation.y),
            z=float(tf.transform.translation.z),
        )
        if not node.path_points or (p.x != node.path_points[-1].x or p.y != node.path_points[-1].y or p.z != node.path_points[-1].z):
            node.path_points.append(p); publish_path_marker(node)
            if not node.tf_printed:
                node.tf_printed = True
                node.get_logger().debug("Tracked robot path point. TF READY.")

            
    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        if not node.tf_warning_printed:
            node.get_logger().warn(f"Waiting for TF. Path trace TF failed: {e}")
            node.tf_warning_printed = True
