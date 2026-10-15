"""Render a NuRec scene in Isaac Sim from the recorded camera poses (for perception checks).

Each output image isaac_NNNNN.png matches SCENE/images/left/NNNNN.jpg: same pose (from the
metric COLMAP model), same intrinsics (stereo_calibration.json).

  unset CONDA_SHLVL
  ~/miniconda3/envs/env_isaaclab/bin/python isaac/nurec/render_nurec_views.py SCENE OUT \
      --nurec_pose X Y Z QW QX QY QZ [--stride 20] --headless --enable_cameras
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("scene", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--nurec_pose", type=float, nargs=7, default=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
parser.add_argument(
    "--asset",
    type=Path,
    default=None,
    help="Render a different NuRec USDZ (e.g. an older map) from this scene's camera poses.",
)
parser.add_argument("--asset_pose", type=float, nargs=7, default=None, help="World pose of --asset (its --nurec_pose).")
parser.add_argument("--stride", type=int, default=20, help="Render every Nth registered left image.")
parser.add_argument(
    "--exposure_ev",
    type=float,
    default=None,
    help="USD camera exposure in EV (UsdGeom.Camera 'exposure'); negative darkens.",
)
parser.add_argument(
    "--rtx",
    nargs="*",
    default=[],
    metavar="KEY=VALUE",
    help="Carb render settings, e.g. /rtx/post/histogram/enabled=false (auto-exposure off).",
)
parser.add_argument("--settle_steps", type=int, default=20, help="Render steps per view before capture.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import carb
import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_frame import quat_to_rot, read_images, rot_to_quat  # noqa: E402

calibration = json.loads((args.scene / "stereo_calibration.json").read_text())
left = calibration["left"]
width, height = calibration["width"], calibration["height"]

# Isaac renders with the principal point at the image centre, so render a canvas centred on the
# real principal point and crop back to the recorded image.
canvas_w = 2 * int(np.ceil(max(left["cx"], width - left["cx"])))
canvas_h = 2 * int(np.ceil(max(left["cy"], height - left["cy"])))
crop_x = int(round(canvas_w / 2 - left["cx"]))
crop_y = int(round(canvas_h / 2 - left["cy"]))
# RTX mis-renders the sub-millimetre focal length PinholeCameraCfg.from_intrinsic_matrix authors
# (the view came out 1.28x zoomed); the same field of view with a 24 mm lens renders correctly.
FOCAL_MM = 24.0


asset_pose = args.asset_pose or args.nurec_pose


@configclass
class ViewSceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    nurec = AssetBaseCfg(
        prim_path="/World/NuRec",
        spawn=sim_utils.UsdFileCfg(usd_path=str(args.asset or args.scene / "scene_nurec.usdz")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(asset_pose[:3]), rot=tuple(asset_pose[3:])),
    )
    camera = CameraCfg(
        prim_path="/World/RecordedCamera",
        height=canvas_h,
        width=canvas_w,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_MM,
            horizontal_aperture=FOCAL_MM * canvas_w / left["fx"],
            vertical_aperture=FOCAL_MM * canvas_h / left["fy"],
            clipping_range=(0.05, 50.0),
        ),
    )


def apply_rtx_settings():
    settings = carb.settings.get_settings()
    for item in args.rtx:
        key, value = item.split("=", 1)
        lowered = value.lower()
        if lowered in ("true", "false"):
            settings.set_bool(key, lowered == "true")
        elif lowered.lstrip("-").replace(".", "", 1).isdigit():
            settings.set(key, float(value) if "." in value else int(value))
        else:
            settings.set_string(key, value)
        print(f"[INFO] {key} = {settings.get(key)}", flush=True)


def main():
    apply_rtx_settings()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device))
    scene = InteractiveScene(ViewSceneCfg(num_envs=1, env_spacing=1.0))
    sim.reset()
    camera = scene["camera"]
    if args.exposure_ev is not None:
        from pxr import UsdGeom

        UsdGeom.Camera(sim.stage.GetPrimAtPath("/World/RecordedCamera")).CreateExposureAttr().Set(args.exposure_ev)

    r_world_nurec = quat_to_rot(*args.nurec_pose[3:])
    p_world_nurec = np.array(args.nurec_pose[:3])
    views = sorted(
        (image for image in read_images(args.scene / "sparse_metric_txt" / "images.txt")
         if image["name"].startswith("left/")),
        key=lambda image: image["name"],
    )[:: args.stride]

    args.out.mkdir(parents=True, exist_ok=True)
    for view in views:
        # COLMAP stores the optical camera from nurec-world transform; Isaac's "ros" convention is optical too.
        r_world_camera = r_world_nurec @ view["R"].T
        p_world_camera = r_world_nurec @ (-view["R"].T @ view["t"]) + p_world_nurec
        camera.set_world_poses(
            torch.tensor(np.array([p_world_camera]), dtype=torch.float32, device=sim.device),
            torch.tensor(np.array([rot_to_quat(r_world_camera)]), dtype=torch.float32, device=sim.device),
            convention="ros",
        )
        for _ in range(args.settle_steps):
            sim.step()
            scene.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"][0, crop_y : crop_y + height, crop_x : crop_x + width, :3].cpu().numpy()
        index = Path(view["name"]).stem
        cv2.imwrite(str(args.out / f"isaac_{index}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[INFO] rendered {view['name']}", flush=True)

    print(f"[DONE] {len(views)} views -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang for minutes after headless rendering; the images are already on disk.
    os._exit(0)
