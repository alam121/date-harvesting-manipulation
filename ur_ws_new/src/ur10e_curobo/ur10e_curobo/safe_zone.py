"""Draggable RViz safe-zone box — the volume the arm is allowed to work in.

Because the camera is eye-in-hand and forward-facing, it cannot see behind/to the
sides of the arm, so obstacles there can never be perceived. Instead of trusting
perception everywhere, we define a trusted box up front and keep the arm inside it.

Two draggable corner handles (min / max) define an axis-aligned box in base_link;
the box is shown as a translucent cube. A right-click menu enables/disables
enforcement and snaps the box around the robot. State lives on the node:

    node.safe_zone_min / node.safe_zone_max : list[3]  (base_link metres)
    node.safe_zone_enabled                  : bool
    node.in_safe_zone(xyz)                  : bool helper

Stage 1 here: definition, visualization, goal rejection. The planner-side keep-out
walls (whole-arm) call ``node._safe_zone_on_change`` whenever the box changes — wire
that in the motion executor (Stage 2).
"""

import math

from geometry_msgs.msg import Point
from interactive_markers import InteractiveMarkerServer, MenuHandler
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    Marker,
)

_SERVER = "safe_zone"
_SQRT_HALF = 1.0 / math.sqrt(2.0)

# Sensible default box in base_link (metres) — tweak by dragging, then Enable.
DEFAULT_MIN = [0.10, -0.60, 0.10]
DEFAULT_MAX = [0.70, 0.20, 0.85]


def _corner_marker(name: str, pos, color) -> InteractiveMarker:
    """A draggable corner handle: a sphere with X/Y/Z move arrows."""
    im = InteractiveMarker()
    im.header.frame_id = "base_link"
    im.name = name
    im.description = ""
    im.scale = 0.15
    im.pose.position.x, im.pose.position.y, im.pose.position.z = (
        float(pos[0]), float(pos[1]), float(pos[2]))
    im.pose.orientation.w = 1.0

    sphere = Marker()
    sphere.type = Marker.SPHERE
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.05
    sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = color
    vis = InteractiveMarkerControl()
    vis.interaction_mode = InteractiveMarkerControl.MOVE_3D
    vis.always_visible = True
    vis.markers.append(sphere)
    im.controls.append(vis)

    # Per-axis move arrows for precise dragging with a normal mouse.
    for axis, qx, qy, qz in (("x", _SQRT_HALF, 0.0, 0.0),
                             ("y", 0.0, 0.0, _SQRT_HALF),
                             ("z", 0.0, _SQRT_HALF, 0.0)):
        ctl = InteractiveMarkerControl()
        ctl.orientation.w = _SQRT_HALF
        ctl.orientation.x = qx
        ctl.orientation.y = qy
        ctl.orientation.z = qz
        ctl.name = f"move_{axis}"
        ctl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        im.controls.append(ctl)
    return im


def setup_safe_zone(node):
    node.safe_zone_min = list(getattr(node, "safe_zone_min", DEFAULT_MIN))
    node.safe_zone_max = list(getattr(node, "safe_zone_max", DEFAULT_MAX))
    node.safe_zone_enabled = bool(getattr(node, "safe_zone_enabled", False))

    box_pub = node.create_publisher(Marker, "/safe_zone_box", 1)
    server = InteractiveMarkerServer(node, _SERVER)
    menu = MenuHandler()

    def _fire_change():
        cb = getattr(node, "_safe_zone_on_change", None)
        if cb is not None:
            try:
                cb()
            except Exception as e:  # pragma: no cover
                node.get_logger().warn(f"safe-zone on_change failed: {e}")

    def publish_box():
        # Wireframe outline (LINE_LIST of the 12 edges), NOT a solid cube — a solid
        # translucent cube occludes/intercepts clicks so you can't grab the goal marker
        # inside it. The outline leaves the interior clear.
        lo, hi = node.safe_zone_min, node.safe_zone_max
        xs, ys, zs = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
        corners = [(x, y, z) for x in xs for y in ys for z in zs]
        edges = [(0, 1), (2, 3), (4, 5), (6, 7),   # along z
                 (0, 2), (1, 3), (4, 6), (5, 7),   # along y
                 (0, 4), (1, 5), (2, 6), (3, 7)]    # along x
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = node.get_clock().now().to_msg()
        m.ns = "safe_zone"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.008  # line width
        if node.safe_zone_enabled:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.10, 0.90, 0.25, 0.95
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 0.70, 0.70, 0.70, 0.70
        for a, b in edges:
            for idx in (a, b):
                p = Point()
                p.x, p.y, p.z = (float(corners[idx][0]), float(corners[idx][1]),
                                 float(corners[idx][2]))
                m.points.append(p)
        box_pub.publish(m)

    def _read_corners():
        mn = server.get(f"{_SERVER}_min")
        mx = server.get(f"{_SERVER}_max")
        if mn is None or mx is None:
            return False
        node.safe_zone_min = [mn.pose.position.x, mn.pose.position.y, mn.pose.position.z]
        node.safe_zone_max = [mx.pose.position.x, mx.pose.position.y, mx.pose.position.z]
        return True

    # Sync box <- corner handles on a timer (robust across interactive_markers API
    # versions; avoids relying on per-drag feedback callbacks).
    _state = {"last": None}

    def _sync():
        if not _read_corners():
            return
        key = (tuple(round(v, 4) for v in node.safe_zone_min),
               tuple(round(v, 4) for v in node.safe_zone_max))
        if key != _state["last"]:
            # Live-update only the visual box while dragging. Walls are (re)applied on the
            # discrete Enable/Snap actions to avoid hammering cuRobo's update_world.
            _state["last"] = key
            publish_box()

    node.create_timer(0.2, _sync)

    def _on_enable(_fb):
        node.safe_zone_enabled = True
        _read_corners()
        node.get_logger().info(
            f"Safe zone ENABLED: min={[round(v, 2) for v in node.safe_zone_min]} "
            f"max={[round(v, 2) for v in node.safe_zone_max]}")
        publish_box()
        _fire_change()

    def _on_disable(_fb):
        node.safe_zone_enabled = False
        node.get_logger().info("Safe zone DISABLED")
        publish_box()
        _fire_change()

    def _on_snap(_fb):
        # Center a default-sized box around the robot's current TCP.
        cur = node.get_end_effector_pose()
        if not cur or len(cur) < 3:
            node.get_logger().warn("Safe zone snap: TCP pose unavailable")
            return
        half = [0.30, 0.40, 0.375]
        node.safe_zone_min = [cur[i] - half[i] for i in range(3)]
        node.safe_zone_max = [cur[i] + half[i] for i in range(3)]
        for which, pos in (("min", node.safe_zone_min), ("max", node.safe_zone_max)):
            im = server.get(f"{_SERVER}_{which}")
            if im is not None:
                im.pose.position.x, im.pose.position.y, im.pose.position.z = (
                    float(pos[0]), float(pos[1]), float(pos[2]))
                server.insert(im)
        server.applyChanges()
        publish_box()
        _fire_change()

    menu.insert("Enable safe zone", callback=_on_enable)
    menu.insert("Disable safe zone", callback=_on_disable)
    menu.insert("Snap box around robot", callback=_on_snap)

    for which, color in (("min", (0.20, 0.55, 1.0, 0.9)),
                         ("max", (1.0, 0.55, 0.10, 0.9))):
        im = _corner_marker(f"{_SERVER}_{which}",
                            node.safe_zone_min if which == "min" else node.safe_zone_max,
                            color)
        server.insert(im)
        menu.apply(server, im.name)
    server.applyChanges()
    publish_box()

    def in_safe_zone(xyz) -> bool:
        if not node.safe_zone_enabled:
            return True
        lo, hi = node.safe_zone_min, node.safe_zone_max
        return all(min(lo[i], hi[i]) <= xyz[i] <= max(lo[i], hi[i]) for i in range(3))

    node.in_safe_zone = in_safe_zone

    # Keep references alive (servers stop publishing if garbage-collected).
    node._safe_zone_server = server
    node._safe_zone_menu = menu
    node.get_logger().info(
        "Safe-zone box ready in RViz (drag the two corner handles; right-click to "
        "Enable/Disable/Snap). Disabled by default.")
    return server
