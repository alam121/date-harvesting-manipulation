"""Live Isaac Lab backend for the date-palm ROS hardware contract."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="UR10e/Tesollo Isaac hardware backend")
parser.add_argument("--command_host", default="127.0.0.1")
parser.add_argument("--command_port", type=int, default=55000)
parser.add_argument("--state_host", default="127.0.0.1")
parser.add_argument("--state_port", type=int, default=55001)
parser.add_argument("--debug_bridge", action="store_true")
parser.add_argument("--nurec_usdz", default="", help="NuRec USDZ from isaac/nurec/reconstruct.sh.")
parser.add_argument(
    "--nurec_pose",
    type=float,
    nargs=7,
    default=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    help="World pose of the reconstruction frame (first left camera, ROS body axes) when recording began.",
)
parser.add_argument(
    "--wrist_stream_port",
    type=int,
    default=0,
    help="Stream the simulated wrist ZED X Mini as JPEG over localhost UDP to this port (0 = off).",
)
parser.add_argument("--wrist_stream_hz", type=float, default=10.0)
parser.add_argument(
    "--wrist_camera_extrinsic",
    type=float,
    nargs=7,
    default=None,
    metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
    help="tool0 -> zed_mini_left_camera_frame (OpenCV optical axes), as printed by nurec_pose_from_bag.py.",
)
parser.add_argument(
    "--nurec_dates",
    default="",
    help="dates_physics.json from isaac/nurec/split_dates.py: detachable photo-textured dates "
    "(use the matching split/tree_nurec.usdz as --nurec_usdz).",
)
parser.add_argument("--date_break_force", type=float, default=5.0, help="Pull [N] that detaches a date.")
parser.add_argument(
    "--date_break_torque",
    type=float,
    default=1.0e6,
    help="Joint break torque; effectively off by default (uncalibrated, see date_pull_test.py).",
)
parser.add_argument("--date_mass", type=float, default=0.01, help="Mass of one date [kg].")
parser.add_argument(
    "--nurec_bunch",
    action="store_true",
    help="Dynamic bunch (isaac/nurec/nurec_bunch.py) from the --nurec_dates split directory; the tree "
    "defaults to its split/tree_nobunch_nurec.usdz.",
)
parser.add_argument(
    "--home_joints",
    type=float,
    nargs=6,
    default=None,
    help="Start arm pose (default: outdoor hardware home). Must match the bridge's UR10E_ENVIRONMENT.",
)
parser.add_argument(
    "--vision_port",
    type=int,
    default=55002,
    help="NuRec mode: date/trunk detections from isaac/nurec/sim_vision.py arrive on this port.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms


# UR10e + Delto from the robot's own /robot_description (isaac/build_ur10e_delto_usd.py). The Isaac
# Lab ur10_tesollo.usd is a UR10: its tool0 sits up to 7 cm off what cuRobo plans for the UR10e.
COMBINED_USD = str(Path.home() / "isaac_ws/ur10e_delto/ur10e_delto.usd")
if not Path(COMBINED_USD).exists():
    raise SystemExit(f"{COMBINED_USD} missing: build it with isaac/build_ur10e_delto_usd.py")
ROBOT_BASE_WORLD = (0.0, 0.0, 0.30)
TARGET_WORLD = (0.62, -0.04, 0.82)
TRUNK_WORLD = (0.82, -0.04, 0.62)
CAMERA_WORLD = (0.70, 0.0, 1.20)
CAMERA_QUAT_ROS = (0.0, 1.0, 0.0, 0.0)
TARGET_RADIUS = 0.024
# ZED X Mini left, rectified, streamed at 640x360 (fx 499.46 at 1280x720, /vision/nurec calibration).
WRIST_WIDTH, WRIST_HEIGHT, WRIST_FX = 640, 360, 249.73
# RTX mis-renders the sub-millimetre focal lengths from PinholeCameraCfg.from_intrinsic_matrix;
# the same field of view with a 24 mm lens renders correctly.
WRIST_FOCAL_MM = 24.0
# NuRec mode: newest wrist RGB-D frame + camera pose in base_link for isaac/nurec/sim_vision.py.
WRIST_FRAME_PATH = Path("/dev/shm/isaac_wrist_frame.npz")
VISION_STALE_S = 1.0

ARM_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_NAMES_ROS = [
    "gripper_f1m1_joint",
    "gripper_f1m2_joint",
    "gripper_f1m3_joint",
    "gripper_f1m4_joint",
    "gripper_f2m1_joint",
    "gripper_f2m2_joint",
    "gripper_f2m3_joint",
    "gripper_f2m4_joint",
    "gripper_f3m1_joint",
    "gripper_f3m2_joint",
    "gripper_f3m3_joint",
    "gripper_f3m4_joint",
]
HARDWARE_HOME = [
    5.108725070953369,
    -1.794300218621725,
    2.4091363588916224,
    1.6457602220722656,
    4.382841110229492,
    -0.23138553300966436,
]
GRIPPER_OPEN = [
    -0.0942,
    -0.1500,
    1.0660,
    -0.5062,
    -1.6318,
    0.1309,
    1.6753,
    -0.4887,
    0.3333,
    0.2234,
    1.0673,
    -0.4311,
]


@configclass
class HardwareSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    stem = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DateStem",
        spawn=sim_utils.CylinderCfg(
            radius=0.006,
            height=0.12,
            axis="X",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.09, 0.035)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.76, -0.04, 0.96)),
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetDate",
        spawn=sim_utils.SphereCfg(
            radius=TARGET_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=4.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.04, 0.02)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TARGET_WORLD),
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/PerceptionCamera",
        update_period=0.05,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_WORLD,
            rot=CAMERA_QUAT_ROS,
            convention="ros",
        ),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=COMBINED_USD,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_WORLD,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1200.0,
                damping=120.0,
                effort_limit_sim=10000.0,
                velocity_limit_sim=10.0,
            )
        },
    )
    fingertips = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*gripper_f[123]_tip_link",
        update_period=0.0,
        history_length=1,
        force_threshold=0.05,
    )


def _wrist_camera_cfg():
    # Take the mount from the robot's live TF (recorded in the bag), not a calibration
    # profile file: the Jetson's active hand-eye can differ from the copies in this repo.
    if args_cli.wrist_camera_extrinsic is None:
        raise SystemExit("--wrist_stream_port needs --wrist_camera_extrinsic (see isaac/nurec/nurec_pose_from_bag.py)")
    x, y, z, qw, qx, qy, qz = args_cli.wrist_camera_extrinsic
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tool0/WristCamera",
        update_period=1.0 / args_cli.wrist_stream_hz,
        height=WRIST_HEIGHT,
        width=WRIST_WIDTH,
        data_types=["rgb", "distance_to_image_plane"] if args_cli.nurec_dates else ["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=WRIST_FOCAL_MM,
            horizontal_aperture=WRIST_FOCAL_MM * WRIST_WIDTH / WRIST_FX,
            vertical_aperture=WRIST_FOCAL_MM * WRIST_HEIGHT / WRIST_FX,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(x, y, z),
            rot=(qw, qx, qy, qz),
            convention="ros",
        ),
    )


class UdpLink:
    def __init__(self):
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.command_socket.bind((args_cli.command_host, args_cli.command_port))
        self.command_socket.setblocking(False)
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_address = (args_cli.state_host, args_cli.state_port)
        self.last_command_time = 0.0

    def receive(self):
        latest = None
        while True:
            try:
                packet, _ = self.command_socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                latest = json.loads(packet.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if latest is not None:
            self.last_command_time = time.monotonic()
        return latest

    def send(self, payload):
        self.state_socket.sendto(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            self.state_address,
        )

    def close(self):
        self.command_socket.close()
        self.state_socket.close()


class VisionLink:
    """Latest detections from sim_vision.py (NuRec mode)."""

    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", port))
        self.socket.setblocking(False)
        self.latest, self.time = None, 0.0

    def fresh(self):
        while True:
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                self.latest, self.time = json.loads(packet.decode("utf-8")), time.monotonic()
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return self.latest if time.monotonic() - self.time < VISION_STALE_S else None

    def close(self):
        self.socket.close()


def _write_wrist_frame(camera, robot, tool, frame_id):
    """Hand the newest wrist RGB-D frame and its base_link pose to sim_vision.py (atomic replace)."""
    output = camera.data.output
    if output.get("rgb") is None or output.get("distance_to_image_plane") is None:
        return
    # The pose from tool0 and the mount: the camera sensor's own pose under the articulation
    # link is not populated (NaN quaternion).
    tool_pos_b, tool_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w[:1], robot.data.root_quat_w[:1],
        robot.data.body_pos_w[:1, tool], robot.data.body_quat_w[:1, tool],
    )
    mount = torch.tensor([args_cli.wrist_camera_extrinsic], device=tool_pos_b.device, dtype=torch.float32)
    pos_b, quat_b = combine_frame_transforms(tool_pos_b, tool_quat_b, mount[:, :3], mount[:, 3:])
    tmp = WRIST_FRAME_PATH.with_name("isaac_wrist_frame.tmp.npz")
    np.savez(
        tmp,
        rgb=output["rgb"][0, ..., :3].cpu().numpy(),
        depth=output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32),
        K=camera.data.intrinsic_matrices[0].cpu().numpy(),
        cam_pos_b=pos_b[0].cpu().numpy(),
        cam_quat_b=quat_b[0].cpu().numpy(),
        frame_id=frame_id,
    )
    tmp.replace(WRIST_FRAME_PATH)


def _skeleton_trunk_b(split_dir):
    """Trunk point for the planner's pole obstacle: the bunch stalk root, 4 cm further back.

    The wrist camera cannot range the trunk in this scene (NuRec splats write no depth).
    """
    path = split_dir / "bunch_skeleton.json"
    if not path.exists():
        return None
    root = np.array(json.loads(path.read_text())["nodes"][0]["position"])
    w, x, y, z = args_cli.nurec_pose[3:]
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    point = rotation @ root + np.array(args_cli.nurec_pose[:3]) - np.array(ROBOT_BASE_WORLD)
    away = np.array([point[0], point[1], 0.0])
    return (point + 0.04 * away / np.linalg.norm(away)).tolist()


def _detect_target(camera, target):
    output = camera.data.output
    rgb = output.get("rgb") if output is not None else None
    depth = output.get("distance_to_image_plane") if output is not None else None
    if rgb is None or depth is None or camera.data.intrinsic_matrices is None:
        return target.data.root_pos_w[0], [0.5] * 4, "isaac-oracle-fallback", "camera not ready"

    rgb0 = rgb[0, ..., :3].float()
    if rgb0.max() > 1.0:
        rgb0 = rgb0 / 255.0
    depth0 = depth[0]
    values = depth0[..., 0]
    mask = (
        (rgb0[..., 0] > 0.32)
        & (rgb0[..., 0] > 2.2 * rgb0[..., 1])
        & (rgb0[..., 0] > 3.0 * rgb0[..., 2])
        & torch.isfinite(values)
        & (values > 0.0)
    )
    count = int(mask.sum().detach().cpu())
    if count < 20:
        return target.data.root_pos_w[0], [0.5] * 4, "isaac-oracle-fallback", f"red pixels={count}"

    # This is a fixed world camera. Isaac's raw camera prim pose can briefly be
    # identity/zero even though the configured offset is already rendering, so
    # use the known calibrated world pose for RGB-D unprojection.
    camera_pos_w = torch.tensor(CAMERA_WORLD, device=depth.device, dtype=torch.float32)
    camera_quat_w_ros = torch.tensor(CAMERA_QUAT_ROS, device=depth.device, dtype=torch.float32)
    points_w = create_pointcloud_from_depth(
        intrinsic_matrix=camera.data.intrinsic_matrices[0],
        depth=depth0,
        keep_invalid=True,
        position=camera_pos_w,
        orientation=camera_quat_w_ros,
        device=depth.device,
    )
    mask_flat = mask.transpose(0, 1).reshape(-1)
    workspace = (
        (points_w[:, 0] > 0.35)
        & (points_w[:, 0] < 0.95)
        & (points_w[:, 1] > -0.35)
        & (points_w[:, 1] < 0.35)
        & (points_w[:, 2] > 0.65)
        & (points_w[:, 2] < 1.00)
    )
    mask_flat = mask_flat & workspace
    points = points_w[mask_flat]
    points = points[torch.all(torch.isfinite(points), dim=1)]
    if points.shape[0] < 20:
        return target.data.root_pos_w[0], [0.5] * 4, "isaac-oracle-fallback", "invalid RGB-D points"

    # A median is stable when a few red robot/material pixels survive the broad
    # workspace gate. The target center Z is known from the hanging-fruit plane.
    center = points.median(dim=0).values
    center[2] = target.data.root_pos_w[0, 2]
    rows, cols = torch.where(mask)
    height, width = mask.shape
    x0 = float(cols.min().detach().cpu()) / width
    y0 = float(rows.min().detach().cpu()) / height
    x1 = float(cols.max().detach().cpu()) / width
    y1 = float(rows.max().detach().cpu()) / height
    return center, [x0, y0, x1, y1], "isaac-rgbd-red", f"red pixels={count}"


def run(sim, scene):
    nurec_mode = bool(args_cli.nurec_dates)
    robot = scene["robot"]
    target = None if nurec_mode else scene["target"]
    camera = None if nurec_mode else scene["camera"]
    fingertips = scene["fingertips"]
    link = UdpLink()
    vision = VisionLink(args_cli.vision_port) if nurec_mode else None
    trunk_b_last = _skeleton_trunk_b(Path(args_cli.nurec_dates).parent) if nurec_mode else None
    tool = robot.body_names.index("tool0")
    tip_ids = [robot.body_names.index(f"gripper_f{f}_tip_link") for f in (1, 2, 3)]
    dates_view = None
    if nurec_mode and args_cli.debug_bridge:
        from isaacsim.core.prims import RigidPrim

        dates_view = RigidPrim(prim_paths_expr="/World/Dates/date_[0-9][0-9]")
        dates_view.initialize()
        dates_rest_w = None
    wrist_frames = 0
    wrist_camera = scene["wrist_camera"] if (args_cli.wrist_stream_port or nurec_mode) else None
    wrist_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args_cli.wrist_stream_port else None
    wrist_every = max(1, round(1.0 / (args_cli.wrist_stream_hz * sim.get_physics_dt())))

    arm_ids = [robot.joint_names.index(name) for name in ARM_NAMES]
    gripper_ids_articulation = [i for i, name in enumerate(robot.joint_names) if name.startswith("gripper_")]
    gripper_names_articulation = [robot.joint_names[i] for i in gripper_ids_articulation]
    gripper_ros_to_articulation = [GRIPPER_NAMES_ROS.index(name) for name in gripper_names_articulation]

    home = args_cli.home_joints or HARDWARE_HOME
    arm_target = torch.tensor(home, device=sim.device, dtype=torch.float32).unsqueeze(0)
    arm_velocity = torch.zeros_like(arm_target)
    arm_mode = "target"
    arm_state_command_time = 0.0
    gripper_target_ros = torch.tensor(GRIPPER_OPEN, device=sim.device, dtype=torch.float32)
    gripper_target = gripper_target_ros[gripper_ros_to_articulation].unsqueeze(0)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, arm_ids] = arm_target
    joint_pos[:, gripper_ids_articulation] = gripper_target
    if nurec_mode:
        # The real Delto's joint readings carry per-unit zero offsets: the controller's gripper
        # profiles command M4 down to -0.68 and M3 up to 2.60 rad, past the URDF's -0.227 / 2.531
        # limits, and the real hand follows. Clamped there, the open hand was ~1.5 cm narrower
        # per finger and struck dates on insertion. Let the sim fingers follow the commands.
        limits = robot.data.joint_pos_limits.clone()
        limits[:, gripper_ids_articulation, 0] = torch.clamp(limits[:, gripper_ids_articulation, 0], max=-1.0)
        limits[:, gripper_ids_articulation, 1] = torch.clamp(limits[:, gripper_ids_articulation, 1], min=2.8)
        robot.write_joint_position_limit_to_sim(limits)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
    robot.set_joint_position_target(gripper_target, joint_ids=gripper_ids_articulation)
    scene.write_data_to_sim()

    sim_dt = sim.get_physics_dt()
    step = 0
    detection = (target.data.root_pos_w[0], [0.5] * 4, "starting", "warming camera") if target is not None else None
    print("[ISAAC BRIDGE] Live PhysX backend ready")
    print("[ISAAC BRIDGE] Arm joints:", ARM_NAMES)
    print("[ISAAC BRIDGE] Gripper articulation order:", gripper_names_articulation)

    try:
        while simulation_app.is_running():
            command = link.receive()
            if command is not None:
                arm = command.get("arm_target", [])
                velocity = command.get("arm_velocity", [])
                grip = command.get("gripper_target", [])
                if len(arm) == len(ARM_NAMES):
                    arm_target = torch.tensor(arm, device=sim.device, dtype=torch.float32).unsqueeze(0)
                    arm_mode = str(command.get("arm_mode", "target"))
                    arm_state_command_time = time.monotonic()
                if len(velocity) == len(ARM_NAMES):
                    arm_velocity = torch.tensor(
                        velocity, device=sim.device, dtype=torch.float32
                    ).unsqueeze(0)
                if len(grip) == len(GRIPPER_NAMES_ROS):
                    grip_ros = torch.tensor(grip, device=sim.device, dtype=torch.float32)
                    gripper_target = grip_ros[gripper_ros_to_articulation].unsqueeze(0)

            applied_arm_target = arm_target
            if arm_mode == "state":
                # The physical UR is authoritative in digital-twin mode. Use
                # measured velocity to bridge the gap between 50 Hz samples.
                command_age = max(time.monotonic() - arm_state_command_time, 0.0)
                extrapolation = min(command_age, 0.03)
                applied_arm_target = arm_target + arm_velocity * extrapolation
                robot.write_joint_state_to_sim(
                    applied_arm_target, arm_velocity, joint_ids=arm_ids
                )

            robot.set_joint_position_target(applied_arm_target, joint_ids=arm_ids)
            robot.set_joint_position_target(gripper_target, joint_ids=gripper_ids_articulation)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            step += 1

            if wrist_camera is not None and step % wrist_every == 0:
                if nurec_mode:
                    wrist_frames += 1
                    _write_wrist_frame(wrist_camera, robot, tool, wrist_frames)
                rgb = wrist_camera.data.output["rgb"]
                if rgb is not None and wrist_socket is not None:
                    frame = cv2.cvtColor(rgb[0, ..., :3].cpu().numpy(), cv2.COLOR_RGB2BGR)
                    # One JPEG per UDP datagram: lower quality until the frame fits.
                    for quality in (85, 70, 55, 40):
                        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                        if ok and len(jpeg) < 65000:
                            wrist_socket.sendto(jpeg.tobytes(), ("127.0.0.1", args_cli.wrist_stream_port))
                            break

            if not nurec_mode and step % 3 == 0:
                detection = _detect_target(camera, target)
            if step % 2 != 0:
                continue

            if nurec_mode:
                # Detections are already in base_link (sim_vision.py used the camera's base pose).
                seen = vision.fresh()
                fruits = seen["fruits"] if seen else []
                if seen and seen.get("trunk"):
                    trunk_b_last = seen["trunk"]  # keep the last trunk fix, like the pole obstacle
                perception = {
                    "fruits_b": [f[:3] for f in fruits],
                    "fruit_radius": fruits[0][4] if fruits else None,
                    "trunk_xyz_b": trunk_b_last,
                    "bbox": seen["bbox"] if seen else [0.5] * 4,
                    "perception_source": "isaac-wrist-yolo",
                    "perception_reason": seen["reason"] if seen else "sim_vision.py not running",
                }
            else:
                detected_w, bbox, perception_source, perception_reason = detection
                root_w = robot.data.root_pos_w[0]
                perception = {
                    "fruit_xyz_b": (detected_w - root_w).detach().cpu().tolist(),
                    "trunk_xyz_b": (torch.tensor(TRUNK_WORLD, device=sim.device) - root_w).detach().cpu().tolist(),
                    "bbox": bbox,
                    "perception_source": perception_source,
                    "perception_reason": perception_reason,
                }
            arm_pos = robot.data.joint_pos[0, arm_ids]
            arm_vel = robot.data.joint_vel[0, arm_ids]
            grip_art = robot.data.joint_pos[0, gripper_ids_articulation]
            grip_by_name = {name: float(grip_art[i].detach().cpu()) for i, name in enumerate(gripper_names_articulation)}
            grip_ros = [grip_by_name[name] for name in GRIPPER_NAMES_ROS]
            # Joint torque per finger joint: the bridge reports it as the Delto motor current,
            # which the controller's non-tactile contact check compares against its close baseline.
            torque_art = robot.data.applied_torque[0, gripper_ids_articulation].detach().cpu().tolist()
            torque_by_name = dict(zip(gripper_names_articulation, torque_art))
            grip_effort = [abs(float(torque_by_name[name])) for name in GRIPPER_NAMES_ROS]

            net_forces = fingertips.data.net_forces_w
            if net_forces is None:
                forces = [0.0, 0.0, 0.0]
            else:
                norms = torch.linalg.norm(net_forces[0], dim=-1).detach().cpu().tolist()
                forces = (norms + [0.0, 0.0, 0.0])[:3]

            link.send({
                "arm_pos": arm_pos.detach().cpu().tolist(),
                "arm_vel": arm_vel.detach().cpu().tolist(),
                "gripper_pos": grip_ros,
                "gripper_effort": grip_effort,
                "forces": forces,
                **perception,
            })

            if args_cli.debug_bridge and step % 100 == 0:
                age = time.monotonic() - link.last_command_time if link.last_command_time else -1.0
                print(
                    f"[ISAAC BRIDGE] step={step} command_age={age:.2f}s "
                    f"arm_mode={arm_mode} "
                    f"forces={[round(f, 2) for f in forces]} "
                    f"source={perception['perception_source']} reason={perception['perception_reason']}"
                )
            if dates_view is not None and step % 10 == 0:
                # Grasp diagnosis (--debug_bridge): every 10 steps while the hand is near a date,
                # fingertip centre vs that date and how far the date has left its rest pose.
                root = robot.data.root_pos_w[0]
                dates_w = dates_view.get_world_poses()[0]
                if dates_rest_w is None:
                    dates_rest_w = dates_w.clone()
                tips_b = (robot.data.body_pos_w[0, tip_ids] - root).mean(0)
                gaps = torch.linalg.norm(dates_w - root - tips_b, dim=1)
                k = int(torch.argmin(gaps))
                if step % 100 == 0 or float(gaps[k]) < 0.12:
                    moved = float(torch.linalg.norm(dates_w[k] - dates_rest_w[k]))
                    tip_forces = fingertips.data.net_forces_w
                    tip_force = [round(float(v), 2) for v in torch.linalg.norm(tip_forces[0], dim=-1)] if tip_forces is not None else []
                    print(f"[ISAAC GRASP] step={step} tips_b={[round(v, 3) for v in tips_b.tolist()]} "
                          f"nearest={Path(dates_view.prim_paths[k]).name} "
                          f"date_b={[round(v, 3) for v in (dates_w[k] - root).tolist()]} gap={float(gaps[k]) * 100:.1f}cm "
                          f"moved={moved * 100:.1f}cm tip_force={tip_force} grip_M3={[round(v, 2) for v in grip_ros[2::4]]}",
                          flush=True)
    finally:
        link.close()
        if vision is not None:
            vision.close()
        if wrist_socket is not None:
            wrist_socket.close()


def main():
    # Render every 4th physics step (30 Hz; the wrist camera samples at 10 Hz): rendering the RTX
    # viewport every 120 Hz step ran the NuRec scene at 0.25x real time, so the arm lagged
    # cuRobo's wall-clock trajectories.
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, render_interval=4)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = HardwareSceneCfg(num_envs=1, env_spacing=2.0)
    split_dir = Path(args_cli.nurec_dates).parent if args_cli.nurec_dates else None
    if args_cli.nurec_bunch and not args_cli.nurec_usdz:
        args_cli.nurec_usdz = str(split_dir / "tree_nobunch_nurec.usdz")
    if args_cli.nurec_dates:
        # The palm replaces the red test date, its stem and the fixed perception camera: dates are
        # found on the wrist camera by isaac/nurec/sim_vision.py, as on the robot.
        scene_cfg.target = None
        scene_cfg.stem = None
        scene_cfg.camera = None
        # The UR controller compensates gravity; plain PD joints would sag ~0.1 rad at the
        # shoulder (~10 cm at the date). Real Delto fingertips push ~10-20 N; the arm's 1200
        # stiffness on the fingers would crush and fling a 10 g date (date_pick_test.py).
        scene_cfg.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        scene_cfg.robot.actuators = {
            "arm": ImplicitActuatorCfg(joint_names_expr=ARM_NAMES, stiffness=1200.0, damping=120.0,
                                       effort_limit_sim=10000.0, velocity_limit_sim=10.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper_.*"], stiffness=20.0, damping=1.0,
                                           effort_limit_sim=1.0, velocity_limit_sim=3.0),
        }
    if args_cli.nurec_usdz:
        # The reconstruction supplies the visuals; the ground plane stays only as a collider.
        scene_cfg.ground.spawn.visible = False
        scene_cfg.nurec = AssetBaseCfg(
            prim_path="/World/NuRec",
            spawn=sim_utils.UsdFileCfg(usd_path=args_cli.nurec_usdz),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(args_cli.nurec_pose[:3]),
                rot=tuple(args_cli.nurec_pose[3:]),
            ),
        )
    if args_cli.wrist_stream_port or args_cli.nurec_dates:
        scene_cfg.wrist_camera = _wrist_camera_cfg()
    scene = InteractiveScene(scene_cfg)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "nurec"))
    from robot_looks import colour_robot

    colour_robot(sim.stage)
    if args_cli.nurec_bunch:
        from nurec_bunch import add_bunch
        from pxr import UsdPhysics

        bunch = add_bunch(sim.stage, split_dir, args_cli.nurec_pose, date_mass=args_cli.date_mass,
                          break_force=args_cli.date_break_force)
        # The hand touches only dates; trapped strands dragged the cluster and flung the date.
        hand = UsdPhysics.CollisionGroup.Define(sim.stage, "/World/RobotCollisionGroup")
        hand.GetCollidersCollectionAPI().GetIncludesRel().AddTarget("/World/envs/env_0/Robot")
        hand.CreateFilteredGroupsRel().AddTarget("/World/Bunch/collision_group")
        print(f"[INFO] dynamic bunch: {len(bunch['date_segment'])} dates on swinging strands")
    elif args_cli.nurec_dates:
        from nurec_dates import add_dates

        dates = add_dates(
            sim.stage,
            Path(args_cli.nurec_dates),
            args_cli.nurec_pose,
            mass=args_cli.date_mass,
            break_force=args_cli.date_break_force,
            break_torque=args_cli.date_break_torque,
        )
        print(f"[INFO] {len(dates)} detachable dates (break force {args_cli.date_break_force} N)")
    sim.set_camera_view(eye=(1.5, 1.2, 1.25), target=(0.55, 0.0, 0.75))
    sim.reset()
    print("[INFO] Isaac date-palm hardware scene setup complete")
    run(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
