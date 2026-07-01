"""Self-contained RViz interactive marker for queuing Cartesian goals.

Adds a draggable 6-DOF marker to the RViz viewport (base_link frame). A
right-click context menu drops the marker's current pose into the goal queue,
snapshots the robot's live TCP pose, snaps the marker back onto the TCP, runs
the queue, or clears it. This is the 3D-viewport counterpart to the panel's
"Add Current Pos as Goal" / Execute / Clear buttons — both feed the same
``node.goal_poses`` queue, so poses added either way run together in order.

Unlike the MoveIt motion-planning marker (whose feedback StateManager already
listens to), this server is owned by our node and always present, so it does
not depend on a MoveIt display being loaded in the RViz config.
"""

import math
import threading

from geometry_msgs.msg import Pose
from interactive_markers import InteractiveMarkerServer, MenuHandler
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    Marker,
)

from . import markers as markers_mod
from .goals import pose_to_vec7

_SQRT_HALF = 1.0 / math.sqrt(2.0)
_MARKER_NAME = "goal_marker"


def _axis_controls(int_marker):
    """Append the standard 6-DOF move/rotate ring controls."""
    # (qx, qy, qz) per axis; w is _SQRT_HALF for all (basic_controls convention).
    axes = (
        ("x", _SQRT_HALF, 0.0, 0.0),
        ("y", 0.0, 0.0, _SQRT_HALF),
        ("z", 0.0, _SQRT_HALF, 0.0),
    )
    for name, qx, qy, qz in axes:
        for mode, suffix in (
            (InteractiveMarkerControl.ROTATE_AXIS, "rotate"),
            (InteractiveMarkerControl.MOVE_AXIS, "move"),
        ):
            control = InteractiveMarkerControl()
            control.orientation.w = _SQRT_HALF
            control.orientation.x = qx
            control.orientation.y = qy
            control.orientation.z = qz
            control.name = f"{suffix}_{name}"
            control.interaction_mode = mode
            int_marker.controls.append(control)


def _make_marker(pose: Pose) -> InteractiveMarker:
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "base_link"
    int_marker.name = _MARKER_NAME
    int_marker.description = "Goal (right-click for menu)"
    int_marker.scale = 0.18
    int_marker.pose = pose

    sphere = Marker()
    sphere.type = Marker.SPHERE
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.05
    sphere.color.r = 0.10
    sphere.color.g = 0.80
    sphere.color.b = 0.25
    sphere.color.a = 0.85

    # A MENU control hosts the visible sphere and provides the right-click menu.
    menu_control = InteractiveMarkerControl()
    menu_control.interaction_mode = InteractiveMarkerControl.MENU
    menu_control.always_visible = True
    menu_control.markers.append(sphere)
    int_marker.controls.append(menu_control)

    _axis_controls(int_marker)
    return int_marker


def _identity_pose() -> Pose:
    pose = Pose()
    pose.position.x = 0.4
    pose.position.z = 0.4
    pose.orientation.w = 1.0
    return pose


def setup_goal_marker(node):
    """Create the interactive goal marker and wire its context menu to the queue."""
    server = InteractiveMarkerServer(node, _MARKER_NAME)
    menu = MenuHandler()

    def _append_goal(vec7, source):
        if node.goal_poses.any_within_distance(vec7[:3], 0.01):
            node.get_logger().info(
                f"Goal from {source} already queued (within 1cm) — skipping")
            return
        node.goal_poses.append(vec7)
        markers_mod.publish_goal_marker(node, vec7[:3])
        node.get_logger().info(
            f"Queued goal #{len(node.goal_poses)} from {source}: "
            f"[{vec7[0]:.3f}, {vec7[1]:.3f}, {vec7[2]:.3f}]")

    def _on_add_marker(feedback):
        _append_goal(pose_to_vec7(feedback.pose), "marker")

    def _on_add_tcp(feedback):
        cur = node.get_end_effector_pose()
        if not cur or len(cur) < 7:
            node.get_logger().warn(
                "Cannot add current robot pose: end-effector pose unavailable")
            return
        _append_goal([float(v) for v in cur[:7]], "robot TCP")

    def _on_snap_to_tcp(feedback):
        cur = node.get_end_effector_pose()
        if not cur or len(cur) < 7:
            node.get_logger().warn(
                "Cannot snap marker: end-effector pose unavailable")
            return
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(cur[0]), float(cur[1]), float(cur[2]))
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (
            float(cur[3]), float(cur[4]), float(cur[5]), float(cur[6]))
        server.setPose(_MARKER_NAME, pose)
        server.applyChanges()

    def _on_execute(feedback):
        # Plain sequential moves — no grasp behavior.
        def run():
            if not node._motion_lock.acquire(blocking=False):
                node.get_logger().warn(
                    "Motion already in progress, ignoring Execute from marker menu")
                return
            try:
                node._execute_goals_no_grasp()
            finally:
                node._motion_lock.release()
        threading.Thread(target=run, daemon=True).start()

    def _on_clear(feedback):
        node.goal_poses.clear()
        getattr(node, "_reachability_goal_metadata", {}).clear()
        markers_mod.clear_goal_markers(node)
        markers_mod.clear_path_markers(node)
        markers_mod.clear_plan_preview(node)
        node.get_logger().info("Goal queue cleared (marker menu)")

    menu.insert("Add marker pose as Goal", callback=_on_add_marker)
    menu.insert("Add current robot pose as Goal", callback=_on_add_tcp)
    menu.insert("Snap marker to robot TCP", callback=_on_snap_to_tcp)
    menu.insert("Execute queue (moves, no grasp)", callback=_on_execute)
    menu.insert("Clear queue", callback=_on_clear)

    int_marker = _make_marker(_identity_pose())
    server.insert(int_marker)
    menu.apply(server, _MARKER_NAME)
    server.applyChanges()

    # Keep references alive (servers stop publishing if garbage-collected).
    node._goal_marker_server = server
    node._goal_marker_menu = menu
    node.get_logger().info(
        "Interactive goal marker ready in RViz "
        "(drag it, right-click for Add/Execute/Clear).")
    return server
