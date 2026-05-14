# ruff: noqa
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
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


def publish_path_marker(node):
    
    m = Marker(); m.header.frame_id = "base_link"; m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "robot_path"; m.id = 0
    m.type = Marker.LINE_STRIP; m.action = Marker.ADD
    m.scale.x = 0.01; m.color.a = 1.0; m.color.g = 1.0
    m.points = node.path_points
    node.path_marker_pub.publish(m)


def track_robot_path(node):

    try:
        tf = node.tf_buffer.lookup_transform(
            "base_link", "gripper_tip", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
        )
        p = Point()
        p.x = float(tf.transform.translation.x)
        p.y = float(tf.transform.translation.y)
        p.z = float(tf.transform.translation.z)
        if not node.path_points or (p.x != node.path_points[-1].x or p.y != node.path_points[-1].y or p.z != node.path_points[-1].z):
            node.path_points.append(p); publish_path_marker(node)
            if not node.tf_printed:
                node.tf_printed = True
                node.get_logger().debug("Tracked robot path point. TF READY.")

            
    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        if not node.tf_warning_printed:
            node.get_logger().warn(f"Waiting for TF. Path trace TF failed: {e}")
            node.tf_warning_printed = True
