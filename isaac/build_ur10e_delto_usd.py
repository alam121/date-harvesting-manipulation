"""Build the Isaac UR10e + Delto DG-3F articulation from the real robot's /robot_description.

The Isaac Lab asset ur10_tesollo.usd is a UR10 (shoulder 0.127 m up, not the UR10e's 0.181 m):
at the lab home its tool0 sits 7 cm below where cuRobo, which plans for the real UR10e, puts it.
This converts the URDF the robot itself published (calibrated UR10e kinematics, the Delto mount,
recorded by isaac/nurec/record_stereo.sh) and renames links and joints to the old asset's names
(gripper_f1_tip_link, gripper_f1m1_joint, ...), so the scene and test scripts take it unchanged.

  unset CONDA_SHLVL
  ~/miniconda3/envs/env_isaaclab/bin/python isaac/build_ur10e_delto_usd.py \\
      ~/nurec_data/bags/palm3/robot_description.urdf --headless
"""

import argparse
import re
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path.home() / "isaac_ws" / "ur10e_delto"
PACKAGES = {
    "ur_description": REPO / "ur_ws_new/install/ur_description/share/ur_description",
    "delto_description": REPO / "ur_ws_new/src/DELTO_ROS2/delto_description",
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("urdf", type=Path, help="robot_description.urdf (replay_bag_joints.py saves it from a bag).")
parser.add_argument("--out", type=Path, default=OUT_DIR / "ur10e_delto.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def isaac_names(urdf: str) -> str:
    """Real Delto names -> the names the Isaac scripts use."""
    urdf = re.sub(r'"F([123])M([1-4])"', r'"gripper_f\1m\2_joint"', urdf)
    urdf = re.sub(r'"F([123])_(0[1-4]|TIP)"', lambda m: f'"gripper_f{m[1]}_{m[2].lower()}_link"', urdf)
    urdf = urdf.replace('"delto_base_link"', '"gripper_base_link"')
    for package, path in PACKAGES.items():
        urdf = urdf.replace(f"package://{package}/", f"file://{path}/")
    return urdf


def main():
    source = args.urdf.read_text()
    # ros2_control blocks are for the driver; the converter only needs links and joints.
    source = re.sub(r"<ros2_control.*?</ros2_control>", "", source, flags=re.S)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    urdf = args.out.with_suffix(".urdf")
    urdf.write_text(isaac_names(source))
    missing = sorted({m for m in re.findall(r'filename="file://([^"]+)"', urdf.read_text()) if not Path(m).exists()})
    if missing:
        raise SystemExit("missing meshes:\n  " + "\n  ".join(missing))

    converter = UrdfConverter(UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(args.out.parent),
        usd_file_name=args.out.name,
        fix_base=True,
        root_link_name="world",
        # Keep tool0, flange, the camera frames: they are fixed links the scripts look up.
        merge_fixed_joints=False,
        force_usd_conversion=True,
        make_instanceable=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1200.0, damping=120.0)),
    ))
    print(f"[BUILD] {urdf.name} -> {converter.usd_path}")


if __name__ == "__main__":
    main()
    app.close()
