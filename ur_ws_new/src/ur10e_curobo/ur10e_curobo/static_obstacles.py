from dataclasses import dataclass
from typing import Iterable, Sequence

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from .config import STATIC_OBSTACLES


@dataclass(frozen=True)
class StaticObstacleSpec:
    marker_id: int
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    color: tuple[float, float, float, float]  # r, g, b, a
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    marker_type: int = Marker.CUBE
    ns: str = "environment"
    points: tuple[tuple[float, float, float], ...] = ()
    mesh_resource: str = ""


# Derive from STATIC_OBSTACLES in config.py (single source of truth)
# Note: pose format is [x, y, z, qw, qx, qy, qz], marker expects (qx, qy, qz, qw)
def _obs_to_spec(i: int, obs: dict) -> StaticObstacleSpec:
    obs_type = obs.get("type", "cuboid")
    if obs_type == "cylinder":
        diameter = float(obs["radius"]) * 2.0
        scale = (diameter, diameter, float(obs["height"]))
        marker_type = Marker.CYLINDER
    elif obs_type == "mesh":
        return StaticObstacleSpec(
            marker_id=i,
            position=(float(obs["pose"][0]), float(obs["pose"][1]), float(obs["pose"][2])),
            scale=(float(obs["scale"][0]), float(obs["scale"][1]), float(obs["scale"][2])),
            color=obs["color"],
            orientation=(float(obs["pose"][4]), float(obs["pose"][5]), float(obs["pose"][6]), float(obs["pose"][3])),
            marker_type=Marker.MESH_RESOURCE,
            mesh_resource=str(obs["mesh_resource"]),
        )
    elif obs_type == "cuboid_outline":
        x, y, z = (float(obs["pose"][0]), float(obs["pose"][1]), float(obs["pose"][2]))
        sx, sy, sz = (float(obs["dims"][0]), float(obs["dims"][1]), float(obs["dims"][2]))
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        corners = {
            "000": (x - hx, y - hy, z - hz),
            "100": (x + hx, y - hy, z - hz),
            "110": (x + hx, y + hy, z - hz),
            "010": (x - hx, y + hy, z - hz),
            "001": (x - hx, y - hy, z + hz),
            "101": (x + hx, y - hy, z + hz),
            "111": (x + hx, y + hy, z + hz),
            "011": (x - hx, y + hy, z + hz),
        }
        edges = [
            ("000", "100"), ("100", "110"), ("110", "010"), ("010", "000"),
            ("001", "101"), ("101", "111"), ("111", "011"), ("011", "001"),
            ("000", "001"), ("100", "101"), ("110", "111"), ("010", "011"),
        ]
        points = tuple(p for edge in edges for p in (corners[edge[0]], corners[edge[1]]))
        return StaticObstacleSpec(
            marker_id=i,
            position=(0.0, 0.0, 0.0),
            scale=(float(obs.get("line_width", 0.012)), 0.0, 0.0),
            color=obs["color"],
            orientation=(0.0, 0.0, 0.0, 1.0),
            marker_type=Marker.LINE_LIST,
            points=points,
        )
    else:
        scale = (float(obs["dims"][0]), float(obs["dims"][1]), float(obs["dims"][2]))
        marker_type = Marker.CUBE
    return StaticObstacleSpec(
        marker_id=i,
        position=(float(obs["pose"][0]), float(obs["pose"][1]), float(obs["pose"][2])),
        scale=scale,
        color=obs["color"],
        orientation=(float(obs["pose"][4]), float(obs["pose"][5]), float(obs["pose"][6]), float(obs["pose"][3])),
        marker_type=marker_type,
    )

DEFAULT_STATIC_OBSTACLES: Sequence[StaticObstacleSpec] = tuple(
    _obs_to_spec(i, obs) for i, obs in enumerate(STATIC_OBSTACLES)
)


def build_static_markers(
    stamp,
    *,
    frame_id: str = "base_link",
    specs: Iterable[StaticObstacleSpec] = DEFAULT_STATIC_OBSTACLES,
) -> list[Marker]:
    markers: list[Marker] = []
    for spec in specs:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = spec.ns
        marker.id = spec.marker_id
        marker.type = spec.marker_type
        marker.action = Marker.ADD

        marker.pose.position.x = spec.position[0]
        marker.pose.position.y = spec.position[1]
        marker.pose.position.z = spec.position[2]
        marker.pose.orientation.x = spec.orientation[0]
        marker.pose.orientation.y = spec.orientation[1]
        marker.pose.orientation.z = spec.orientation[2]
        marker.pose.orientation.w = spec.orientation[3]

        marker.scale.x = spec.scale[0]
        marker.scale.y = spec.scale[1]
        marker.scale.z = spec.scale[2]

        marker.color.r = spec.color[0]
        marker.color.g = spec.color[1]
        marker.color.b = spec.color[2]
        marker.color.a = spec.color[3]

        for x, y, z in spec.points:
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            marker.points.append(p)

        if spec.mesh_resource:
            marker.mesh_resource = spec.mesh_resource
            marker.mesh_use_embedded_materials = False

        markers.append(marker)
    return markers


def publish_static_obstacles(
    publisher,
    clock,
    *,
    frame_id: str = "base_link",
    specs: Iterable[StaticObstacleSpec] = DEFAULT_STATIC_OBSTACLES,
) -> None:
    stamp = clock.now().to_msg()
    for marker in build_static_markers(stamp, frame_id=frame_id, specs=specs):
        publisher.publish(marker)
