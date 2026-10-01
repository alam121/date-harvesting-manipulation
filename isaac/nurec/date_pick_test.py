"""Scripted date picking with the UR10e + Delto gripper in the NuRec scene (no planner).

Differential IK drives tool0 so that the fingertip grasp point (calibrated by closing the hand
in free air) reaches a date along a horizontal approach from the robot; the hand closes
gradually, pulls back, and the script reports whether the date detached, stayed in the hand,
the gripper contact forces, and any neighbouring dates knocked off.

  single date:   ... date_pick_test.py MANIFEST --nurec_pose ... --date 6 --headless
  whole bunch:   ... date_pick_test.py MANIFEST --nurec_pose ... --harvest_all --headless
                 (front dates first; each picked date is carried aside and released)
  video frames:  add --record DIR --wrist_camera_extrinsic ... (loads the tree; overview + wrist
                 frames at 30 fps sim time, events.jsonl), then compose_harvest_video.py DIR

  unset CONDA_SHLVL; interpreter: ~/miniconda3/envs/env_isaaclab/bin/python
"""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("manifest", type=Path)
parser.add_argument("--nurec_pose", type=float, nargs=7, required=True)
parser.add_argument("--date", type=int, default=-1, help="Date id; default: the one closest to the robot base.")
parser.add_argument("--harvest_all", action="store_true", help="Pick every date, nearest the base first.")
parser.add_argument("--break_force", type=float, default=5.0)
parser.add_argument("--mass", type=float, default=0.01)
parser.add_argument("--standoff", type=float, default=0.12, help="Pre-grasp distance before the date [m].")
parser.add_argument("--pull", type=float, default=None,
                    help="Retreat after closing [m] (default 0.15; 0.25 with --bunch, the strand stretches first).")
parser.add_argument("--finger_torque", type=float, default=1.0, help="Gripper joint effort limit [N m].")
parser.add_argument("--roll_offset", type=float, default=0.0,
                    help="Extra tool roll about the approach axis on top of the upright-camera roll [deg].")
parser.add_argument("--close_seconds", type=float, default=1.5, help="Time to close the hand.")
parser.add_argument("--record", type=Path, default=None, help="Save overview + wrist frames here (loads the tree).")
parser.add_argument("--wrist_camera_extrinsic", type=float, nargs=7, default=None,
                    help="tool0 -> camera (OpenCV optical), as printed by nurec_pose_from_bag.py (for --record).")
parser.add_argument("--tree", type=Path, default=None, help="Tree USDZ for --record (default: next to manifest).")
parser.add_argument("--bunch", action="store_true",
                    help="Dynamic bunch (nurec_bunch.py): dates hang on swinging strands, tree without strands.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record:
    args.enable_cameras = True
    if args.wrist_camera_extrinsic is None:
        raise SystemExit("--record needs --wrist_camera_extrinsic")
if args.pull is None:
    args.pull = 0.25 if args.bunch else 0.15
app = AppLauncher(args).app

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_inv, subtract_frame_transforms
from isaacsim.core.prims import RigidPrim
from pxr import UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nurec_bunch import add_bunch  # noqa: E402
from nurec_dates import add_dates  # noqa: E402
from robot_looks import colour_robot  # noqa: E402

# UR10e + Delto built from the robot's /robot_description (isaac/build_ur10e_delto_usd.py).
ROBOT_USD = str(Path.home() / "isaac_ws/ur10e_delto/ur10e_delto.usd")
ROBOT_BASE_WORLD = (0.0, 0.0, 0.30)  # as in isaac_hardware_scene.py (--nurec_pose is relative to it)
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# Recorded palm3 start pose: the camera looks at the bunch, a reachable starting configuration.
ARM_START = [-1.56780, -0.77998, 1.50989, 2.21126, -1.39554, -1.00800]
# Delto M1..M12 (f1m1..f1m4, f2m1.., f3m1..), from ur10e_curobo/gripper_profiles.py (old gripper).
GRIPPER_ROS = [f"gripper_f{f}m{m}_joint" for f in (1, 2, 3) for m in (1, 2, 3, 4)]
GRIPPER_OPEN = [-0.0942, -0.1500, 1.0660, -0.5062, -1.6318, 0.1309, 1.6753, -0.4887, 0.3333, 0.2234, 1.0673, -0.4311]
GRIPPER_CLOSED = [-0.0942, -0.1500, 2.5660, -0.5062, -1.6318, 0.1309, 2.3753, -0.4887, 0.3333, 0.2234, 2.5673, -0.4311]
DT = 1.0 / 120.0
RENDER_EVERY = 4  # physics steps per recorded frame -> 30 fps sim time
# Wrist ZED X Mini at 640x360 (as isaac_hardware_scene.py): 24 mm lens, same field of view as fx.
WRIST_W, WRIST_H, WRIST_FX, FOCAL_MM = 640, 360, 249.73, 24.0


@configclass
class PickSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(visible=args.record is None))
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        # The UR controller compensates gravity; plain PD joints would sag ~0.1 rad at the shoulder
        # (~10 cm at the date), which IK from measured joints never recovers.
        spawn=sim_utils.UsdFileCfg(usd_path=ROBOT_USD, activate_contact_sensors=True,
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=ROBOT_BASE_WORLD, rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=ARM, stiffness=1200.0, damping=120.0,
                                       effort_limit_sim=10000.0, velocity_limit_sim=10.0),
            # Real Delto fingertips push ~10-20 N; the scene's 1200 stiffness would crush the date.
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper_.*"], stiffness=20.0, damping=1.0,
                                           effort_limit_sim=args.finger_torque, velocity_limit_sim=3.0),
        },
    )
    # Every gripper link: the date is often held by the middle phalanges, not only the tips.
    contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper_.*",
                                update_period=0.0, history_length=1, force_threshold=0.05)


def look_rotation(forward: np.ndarray) -> np.ndarray:
    """Rotation whose z axis is `forward` (approach) and x axis is as close to world down as possible."""
    z = forward / np.linalg.norm(forward)
    x = np.array([0.0, 0.0, -1.0]) - z * z[2]
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def upright_rotation(forward: np.ndarray, camera_in_tool: np.ndarray) -> np.ndarray:
    """Tool rotation approaching along `forward`, rolled so the wrist camera image is upright.

    look_rotation fixes the tool x axis, but the camera is mounted at an angle on the flange, so
    its image came out rolled (differently per pick). Choose the roll about the approach axis that
    points the camera's image-down axis (optical y) most nearly at world down.
    """
    base = look_rotation(forward)
    best, best_score = base, np.inf
    for angle in np.linspace(-np.pi, np.pi, 721):
        c, s = np.cos(angle), np.sin(angle)
        rotation = base @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        score = (rotation @ camera_in_tool)[2, 1]
        if score < best_score:
            best, best_score = rotation, score
    return best


def recording_cameras(cfg, bunch_centre: np.ndarray):
    x, y, z, qw, qx, qy, qz = args.wrist_camera_extrinsic
    cfg.wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tool0/WristCamera", height=WRIST_H, width=WRIST_W, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=FOCAL_MM, horizontal_aperture=FOCAL_MM * WRIST_W / WRIST_FX,
                                         vertical_aperture=FOCAL_MM * WRIST_H / WRIST_FX, clipping_range=(0.05, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=(x, y, z), rot=(qw, qx, qy, qz), convention="ros"))
    cfg.overview = CameraCfg(
        prim_path="/World/OverviewCamera", height=540, width=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=36.0, clipping_range=(0.05, 50.0)))
    tree = args.tree or args.manifest.parent / ("tree_nobunch_nurec.usdz" if args.bunch else "tree_nurec.usdz")
    cfg.tree = AssetBaseCfg(prim_path="/World/Tree", spawn=sim_utils.UsdFileCfg(usd_path=str(tree)),
                            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(args.nurec_pose[:3]),
                                                                    rot=tuple(args.nurec_pose[3:])))


def date_world_positions() -> np.ndarray:
    manifest = json.loads(args.manifest.read_text())
    w, x, y, z = args.nurec_pose[3:]
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.array([rot @ np.array(d["centre"]) + np.array(args.nurec_pose[:3]) for d in manifest["dates"]])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device, render_interval=RENDER_EVERY))
    cfg = PickSceneCfg(num_envs=1, env_spacing=2.0)
    bunch_centre = np.median(date_world_positions(), 0)
    if args.record:
        recording_cameras(cfg, bunch_centre)
        args.record.mkdir(parents=True, exist_ok=True)
    scene = InteractiveScene(cfg)
    colour_robot(sim.stage)
    bunch = None
    if args.bunch:
        bunch = add_bunch(sim.stage, args.manifest.parent, args.nurec_pose, date_mass=args.mass,
                  break_force=args.break_force, visual=args.record is not None)
        # The hand touches only dates: closing on a date it also trapped the neighbouring strands,
        # dragged the cluster along and the date was flung out when they sprang back. The strands
        # still bend with the date they hold.
        hand = UsdPhysics.CollisionGroup.Define(sim.stage, "/World/RobotCollisionGroup")
        hand.GetCollidersCollectionAPI().GetIncludesRel().AddTarget("/World/envs/env_0/Robot")
        hand.CreateFilteredGroupsRel().AddTarget("/World/Bunch/collision_group")
    else:
        add_dates(sim.stage, args.manifest, args.nurec_pose, mass=args.mass, break_force=args.break_force,
                  visual_mode="mesh" if args.record else "none")
    sim.reset()

    robot, contacts = scene["robot"], scene["contacts"]
    dates = RigidPrim(prim_paths_expr="/World/Dates/date_[0-9][0-9]")
    dates.initialize()
    names = [Path(p).name for p in dates.prim_paths]
    dev = sim.device
    arm_ids = [robot.joint_names.index(n) for n in ARM]
    grip_ids = [robot.joint_names.index(n) for n in GRIPPER_ROS]
    tool = robot.body_names.index("tool0")
    tip_ids = [robot.body_names.index(f"gripper_f{f}_tip_link") for f in (1, 2, 3)]
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", ik_method="dls"), num_envs=1, device=dev)
    base = np.array(ROBOT_BASE_WORLD)

    open_q = torch.tensor([GRIPPER_OPEN], device=dev)
    closed_q = torch.tensor([GRIPPER_CLOSED], device=dev)
    arm_target = torch.tensor([ARM_START], device=dev)
    grip_target = open_q.clone()
    state = robot.data.default_joint_pos.clone()
    state[:, arm_ids], state[:, grip_ids] = arm_target, grip_target
    robot.write_joint_state_to_sim(state, torch.zeros_like(state))

    frame_count = 0
    events = open(args.record / "events.jsonl", "w") if args.record else None
    status = {"text": "", "picked": 0, "total": len(names)}

    def log_event(text: str):
        status["text"] = text
        print(f"[PICK] {text}", flush=True)
        if events:
            events.write(json.dumps({"frame": frame_count, **status}) + "\n")
            events.flush()

    camera_in_tool = None
    if args.wrist_camera_extrinsic:
        qw, qx, qy, qz = args.wrist_camera_extrinsic[3:]
        camera_in_tool = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                                   [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                                   [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    if args.record:
        # Follow camera: from the side, ~0.45 m off the hand, easing after it so the fingers closing
        # on the date fill the frame. (Kept close: seen from further off, the arm sits behind white
        # haze in the reconstruction around where the real robot stood.)
        toward = bunch_centre - base
        toward[2] = 0.0
        toward /= np.linalg.norm(toward)
        side = np.cross(toward, [0.0, 0.0, 1.0])
        follow_offset = 0.4 * side - 0.15 * toward + np.array([0.0, 0.0, 0.1])
        follow = {"focus": None}

        def update_follow_camera():
            hand = (grasp_point_world() if grasp_offset is not None else robot.data.body_pos_w[:, tool])[0].cpu().numpy()
            focus = hand if follow["focus"] is None else follow["focus"] + 0.06 * (hand - follow["focus"])
            follow["focus"] = focus
            scene["overview"].set_world_poses_from_view(
                torch.tensor([focus + follow_offset], device=dev, dtype=torch.float32),
                torch.tensor([focus], device=dev, dtype=torch.float32))

    def step(n: int):
        nonlocal frame_count
        for i in range(n):
            robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
            robot.set_joint_position_target(grip_target, joint_ids=grip_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(DT)
            step.count += 1
            if args.record and step.count % RENDER_EVERY == 0:
                update_follow_camera()
                for camera in ("overview", "wrist"):
                    rgb = scene[camera].data.output["rgb"][0, ..., :3].cpu().numpy()
                    cv2.imwrite(str(args.record / f"{camera}_{frame_count:06d}.jpg"),
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_count += 1
    step.count = 0
    grasp_offset = None

    def grasp_point_world():
        # Calibrated fingertip meeting point, carried by tool0.
        return robot.data.body_pos_w[:, tool] + quat_apply(robot.data.body_quat_w[:, tool], grasp_offset)

    def move_to(target_pos_w, target_rot_w, seconds=3.0, tol=0.005):
        """Drive the grasp point to target_pos_w with tool orientation target_rot_w (3x3, world)."""
        nonlocal arm_target
        goal = torch.tensor(target_pos_w, dtype=torch.float32, device=dev)
        quat_w = quat_from_matrix(torch.tensor(target_rot_w, dtype=torch.float32, device=dev)).unsqueeze(0)
        for _ in range(int(seconds / DT / 4)):
            tool_pos_w = goal.unsqueeze(0) - quat_apply(quat_w, grasp_offset)
            goal_pos_b, goal_quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, tool_pos_w, quat_w)
            pos_b, quat_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                      robot.data.body_pos_w[:, tool], robot.data.body_quat_w[:, tool])
            ik.set_command(torch.cat([goal_pos_b, goal_quat_b], 1))
            jacobian = robot.root_physx_view.get_jacobians()[:, tool - 1, :, arm_ids]
            arm_target = ik.compute(pos_b, quat_b, jacobian, robot.data.joint_pos[:, arm_ids])
            step(4)
            if torch.linalg.norm(grasp_point_world()[0] - goal) < tol:
                break
        return float(torch.linalg.norm(grasp_point_world()[0] - goal))

    def set_gripper(target_q, seconds):
        # Close/open gradually like the real Delto: a finger jumping to "closed" strikes the 10 g
        # date and launches it.
        nonlocal grip_target
        start = grip_target.clone()
        ramp = max(1, int(seconds / DT))
        for i in range(1, ramp + 1):
            grip_target = start + (target_q - start) * (i / ramp)
            step(1)

    def date_positions():
        pos, _ = dates.get_world_poses()
        return pos.cpu().numpy()

    # Calibrate the grasp point: close in free air at the start pose, average the fingertips.
    step(120)
    set_gripper(closed_q, 1.0)
    step(120)
    tips_w = robot.data.body_pos_w[0, tip_ids]
    tool_pos, tool_quat = robot.data.body_pos_w[:, tool], robot.data.body_quat_w[:, tool]
    grasp_offset = quat_apply(quat_inv(tool_quat), tips_w.mean(0, keepdim=True) - tool_pos)
    print(f"[PICK] grasp point {grasp_offset[0].cpu().numpy().round(3)} m in tool0")
    set_gripper(open_q, 1.0)
    step(60)

    original = date_positions()
    # Attached = still at its stem: on the fixed tree that is "not moved"; on the dynamic bunch the
    # date swings with its strand, so compare its distance to the strand section it hangs on.
    if bunch:
        sections = RigidPrim(prim_paths_expr="/World/Bunch/section_[0-9][0-9][0-9]")
        sections.initialize()
        holder = [sections.prim_paths.index(bunch["date_segment"][int(n.split("_")[1])]) for n in names]

        def stem_gaps():
            section_pos = sections.get_world_poses()[0].cpu().numpy()[holder]
            return np.linalg.norm(date_positions() - section_pos, axis=1)
    else:
        def stem_gaps():
            return np.linalg.norm(date_positions() - original, axis=1)
    rest_gap = stem_gaps()

    def attached():
        return np.abs(stem_gaps() - rest_gap) < 0.015
    order = np.argsort(np.linalg.norm(original - base, axis=1))
    if not args.harvest_all:
        wanted = args.date if args.date >= 0 else int(names[order[0]].split("_")[1])
        order = [names.index(f"date_{wanted:02d}")]

    results = {}
    for index in order:
        name = names[index]
        current = date_positions()
        attached_before = attached()
        if not attached_before[index]:
            results[name] = "knocked off earlier"
            log_event(f"{name}: skipped (knocked off earlier)")
            continue
        target = current[index]
        approach = target - base
        approach[2] = 0.0
        approach /= np.linalg.norm(approach)
        # Rolled so the wrist camera stays upright (given its extrinsic); else the plain approach frame.
        rotation = look_rotation(approach) if camera_in_tool is None else upright_rotation(approach, camera_in_tool)
        c, s = np.cos(np.radians(args.roll_offset)), np.sin(np.radians(args.roll_offset))
        rotation = rotation @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        log_event(f"picking {name} ({np.linalg.norm(target - base):.2f} m from the base)")

        err = move_to(target - args.standoff * approach, rotation, seconds=4.0)
        if err > 0.03:
            results[name] = f"unreachable ({err*100:.0f} cm short)"
            log_event(f"{name}: {results[name]}")
            continue
        if args.bunch:
            # The bunch swings: let it settle, then re-read where the date is now (visual servoing).
            step(60)
            target = date_positions()[index]
        move_to(target, rotation, seconds=3.0)
        set_gripper(closed_q, args.close_seconds)
        step(60)
        held_force = float(contacts.data.net_forces_w[0].norm(dim=-1).sum())
        pull_goal = target - args.pull * approach
        for fraction in np.linspace(0.0, 1.0, 30):
            move_to(target + (pull_goal - target) * fraction, rotation, seconds=0.2, tol=0.002)
        step(60)

        after = date_positions()
        in_hand = float(np.linalg.norm(after[index] - grasp_point_world()[0].cpu().numpy()))
        attached_after = attached()
        detached = not attached_after[index]
        knocked = [names[i] for i in range(len(names)) if i != index
                   and attached_before[i] and not attached_after[i]]
        if detached and in_hand < 0.03:
            results[name] = f"picked (hand contact {held_force:.0f} N)"
            status["picked"] += 1
            log_event(f"{name}: PICKED - {status['picked']} picked so far")
            # Carry it aside and release it, then come back for the next one.
            drop = base + 0.55 * (target - base)
            drop[2] = target[2] - 0.1
            move_to(drop, rotation, seconds=3.0, tol=0.02)
            set_gripper(open_q, 1.0)
            step(90)
        else:
            results[name] = f"detached but dropped ({in_hand*100:.0f} cm from the hand)" if detached else "not detached"
            log_event(f"{name}: {results[name]}")
            set_gripper(open_q, 1.0)
        if knocked:
            log_event(f"{name}: knocked off {', '.join(knocked)}")

    print("[PICK] summary:")
    for name, result in results.items():
        print(f"[PICK]   {name}: {result}")
    print(f"[PICK] picked {status['picked']} of {len(names)} dates")
    if events:
        log_event(f"done: {status['picked']} of {len(names)} dates picked")
        events.close()
        print(f"[PICK] {frame_count} frames -> {args.record}")


if __name__ == "__main__":
    main()
    os._exit(0)
