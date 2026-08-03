"""Camera calibration profile helpers.

Profiles make the active camera mode explicit so ZED X Mini RGBD and
ZED X One RGB + ZED X Mini depth do not share one ambiguous hand-eye file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROFILE_DIR = Path(__file__).resolve().parent / "calibration_profiles"

PROFILE_BY_MODE = {
    "zedx_mini": "zedx_mini_rgbd",
    "zed_mini": "zed_one_rgb_zedx_mini_depth",
    "lidar": "zed_one_rgb_zedx_mini_depth",
    "stereo": "zed_one_rgb_zedx_mini_depth",
}


def profile_name_for_mode(mode: str) -> str:
    return PROFILE_BY_MODE.get(mode, "zedx_mini_rgbd")


def contextual_profile_name(
    mode: str,
    robot_profile: Optional[str] = None,
    environment: Optional[str] = None,
) -> str:
    """Return the robot/environment-specific calibration profile name."""
    robot = (robot_profile or os.getenv(
        "UR10E_ROBOT_PROFILE", "old")).strip().lower()
    env = (environment or os.getenv(
        "UR10E_ENVIRONMENT", "outdoor")).strip().lower()
    if robot not in ("new", "old"):
        raise ValueError(f"unsupported robot profile: {robot!r}")
    if env not in ("lab", "outdoor"):
        raise ValueError(f"unsupported environment: {env!r}")
    return f"{env}_{robot}_{profile_name_for_mode(mode)}"


def profile_path_for_name(name: str) -> Path:
    return PROFILE_DIR / f"{name}.yaml"


def active_profile_path(mode: Optional[str] = None) -> Path:
    override = os.getenv("UR10E_CAMERA_PROFILE_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    name = os.getenv("UR10E_CAMERA_PROFILE", "").strip()
    if not name:
        camera_mode = mode or os.getenv("UR10E_CAMERA_MODE", "zed_mini")
        contextual_name = contextual_profile_name(camera_mode)
        # Existing installations remain usable until their first contextual
        # calibration is saved. Once present, the contextual profile wins.
        name = (
            contextual_name
            if profile_path_for_name(contextual_name).exists()
            else profile_name_for_mode(camera_mode)
        )
    return profile_path_for_name(name)


def load_camera_profile(mode: Optional[str] = None) -> Dict[str, Any]:
    path = active_profile_path(mode)
    if not path.exists():
        return {
            "camera_profile": path.stem,
            "profile_path": str(path),
        }
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("camera_profile", path.stem)
    data["profile_path"] = str(path)
    return data


def save_camera_profile(profile: Dict[str, Any], path: Optional[Path] = None) -> Path:
    out = path or active_profile_path(profile.get("mode"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(profile, sort_keys=False))
    return out


def frame_from_profile(profile: Dict[str, Any], key: str, default: str) -> str:
    value = profile.get(key)
    return str(value).strip() if value else default


def matrix_from_profile(profile: Dict[str, Any], key: str, default):
    value = profile.get(key)
    if isinstance(value, dict):
        value = value.get("matrix")
    if value is None:
        return default
    return value
