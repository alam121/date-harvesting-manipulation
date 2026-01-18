from dataclasses import dataclass
from typing import Iterable, Sequence

from visualization_msgs.msg import Marker


@dataclass(frozen=True)
class StaticObstacleSpec:
    marker_id: int
    position: tuple[float, float, float]
    scale: tuple[float, float, float]
    color: tuple[float, float, float, float]  # r, g, b, a
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    marker_type: int = Marker.CUBE
    ns: str = "environment"


DEFAULT_STATIC_OBSTACLES: Sequence[StaticObstacleSpec] = (
    StaticObstacleSpec(
        marker_id=0,
        position=(0.0, 0.0, -0.1),
        scale=(5.0, 5.0, 0.2),
        color=(1.0, 0.0, 0.0, 1.0),
    ),
    StaticObstacleSpec(
        marker_id=1,
        position=(0.25, -0.80, 0.5),
        scale=(0.02, 0.02, 1.0),
        color=(0.0, 1.0, 0.0, 1.0),
    ),
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
