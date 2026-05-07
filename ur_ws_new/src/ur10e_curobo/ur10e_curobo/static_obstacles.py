from dataclasses import dataclass
from typing import Iterable, Sequence

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


# Derive from STATIC_OBSTACLES in config.py (single source of truth)
# Note: pose format is [x, y, z, qw, qx, qy, qz], marker expects (qx, qy, qz, qw)
def _obs_to_spec(i: int, obs: dict) -> StaticObstacleSpec:
    obs_type = obs.get("type", "cuboid")
    if obs_type == "cylinder":
        diameter = float(obs["radius"]) * 2.0
        scale = (diameter, diameter, float(obs["height"]))
        marker_type = Marker.CYLINDER
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
