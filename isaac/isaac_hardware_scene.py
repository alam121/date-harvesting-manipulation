"""Live Isaac Lab backend for the date-palm ROS hardware contract."""

from __future__ import annotations

import argparse
import json
import socket
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="UR10e/Tesollo Isaac hardware backend")
parser.add_argument("--command_host", default="127.0.0.1")
parser.add_argument("--command_port", type=int, default=55000)
parser.add_argument("--state_host", default="127.0.0.1")
parser.add_argument("--state_port", type=int, default=55001)
parser.add_argument("--debug_bridge", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.utils import configclass


COMBINED_USD = "/home/alam/isaac_ws/IsaacLab/source/isaaclab_assets/data/Robots/UR10_Tesollo/ur10_tesollo.usd"
ROBOT_BASE_WORLD = (0.0, 0.0, 0.30)
TARGET_WORLD = (0.62, -0.04, 0.82)
TRUNK_WORLD = (0.82, -0.04, 0.62)
CAMERA_WORLD = (0.70, 0.0, 1.20)
CAMERA_QUAT_ROS = (0.0, 1.0, 0.0, 0.0)
TARGET_RADIUS = 0.024

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
    robot = scene["robot"]
    target = scene["target"]
    camera = scene["camera"]
    fingertips = scene["fingertips"]
    link = UdpLink()

    arm_ids = [robot.joint_names.index(name) for name in ARM_NAMES]
    gripper_ids_articulation = [i for i, name in enumerate(robot.joint_names) if name.startswith("gripper_")]
    gripper_names_articulation = [robot.joint_names[i] for i in gripper_ids_articulation]
    gripper_ros_to_articulation = [GRIPPER_NAMES_ROS.index(name) for name in gripper_names_articulation]

    arm_target = torch.tensor(HARDWARE_HOME, device=sim.device, dtype=torch.float32).unsqueeze(0)
    arm_velocity = torch.zeros_like(arm_target)
    arm_mode = "target"
    arm_state_command_time = 0.0
    gripper_target_ros = torch.tensor(GRIPPER_OPEN, device=sim.device, dtype=torch.float32)
    gripper_target = gripper_target_ros[gripper_ros_to_articulation].unsqueeze(0)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, arm_ids] = arm_target
    joint_pos[:, gripper_ids_articulation] = gripper_target
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
    robot.set_joint_position_target(gripper_target, joint_ids=gripper_ids_articulation)
    scene.write_data_to_sim()

    sim_dt = sim.get_physics_dt()
    step = 0
    detection = (target.data.root_pos_w[0], [0.5] * 4, "starting", "warming camera")
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

            if step % 3 == 0:
                detection = _detect_target(camera, target)
            if step % 2 != 0:
                continue

            detected_w, bbox, perception_source, perception_reason = detection
            root_w = robot.data.root_pos_w[0]
            fruit_b = detected_w - root_w
            trunk_b = torch.tensor(TRUNK_WORLD, device=sim.device) - root_w
            arm_pos = robot.data.joint_pos[0, arm_ids]
            arm_vel = robot.data.joint_vel[0, arm_ids]
            grip_art = robot.data.joint_pos[0, gripper_ids_articulation]
            grip_by_name = {name: float(grip_art[i].detach().cpu()) for i, name in enumerate(gripper_names_articulation)}
            grip_ros = [grip_by_name[name] for name in GRIPPER_NAMES_ROS]

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
                "forces": forces,
                "fruit_xyz_b": fruit_b.detach().cpu().tolist(),
                "trunk_xyz_b": trunk_b.detach().cpu().tolist(),
                "bbox": bbox,
                "perception_source": perception_source,
                "perception_reason": perception_reason,
            })

            if args_cli.debug_bridge and step % 100 == 0:
                age = time.monotonic() - link.last_command_time if link.last_command_time else -1.0
                print(
                    f"[ISAAC BRIDGE] step={step} command_age={age:.2f}s "
                    f"arm_mode={arm_mode} "
                    f"fruit_b={fruit_b.detach().cpu().tolist()} forces={forces} "
                    f"source={perception_source} reason={perception_reason}"
                )
    finally:
        link.close()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(HardwareSceneCfg(num_envs=1, env_spacing=2.0))
    sim.set_camera_view(eye=(1.5, 1.2, 1.25), target=(0.55, 0.0, 0.75))
    sim.reset()
    print("[INFO] Isaac date-palm hardware scene setup complete")
    run(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
