#!/usr/bin/env python3
"""Launch UR10e nodes in a single terminator window with panes."""
import sys
import subprocess
import os
import re
import ast
from pathlib import Path

WS = str(Path(__file__).resolve().parent)
REPO_ROOT = str(Path(WS).parent)
ROS_DOMAIN_ID = os.environ.get("ROS_DOMAIN_ID", "6")
ROS_DOMAIN_CMD = f"export ROS_DOMAIN_ID={ROS_DOMAIN_ID}"
FASTDDS_CFG = f"export FASTRTPS_DEFAULT_PROFILES_FILE={WS}/fastdds_config.xml"
SOURCE = f"{ROS_DOMAIN_CMD} && {FASTDDS_CFG} && source {WS}/install/setup.bash"
CONFIG_PATH = os.path.expanduser("~/.config/terminator/config")
UNET_SCRIPT = str(Path(REPO_ROOT) / "bin" / "unet.sh")
ROBOT_IP = "192.168.1.190"
ROBOT_PC_IP = "192.168.1.101"
SCRIPT_SENDER_PORT = "50002"
ROBOT_CONFIG_SOURCE = (
    Path(WS) / "src" / "ur10e_curobo" / "ur10e_curobo" / "config.py"
)
CAMERA_PROFILE_DIR = (
    Path(WS) / "src" / "ur10e_curobo" / "ur10e_curobo"
    / "vision" / "calibration_profiles"
)
PRESETS = {
    "harvest": ["main", "vision", "zedx_mini"],
    "field": ["main", "vision", "zedx_mini"],
}


def expand_presets(args):
    expanded = []
    for arg in args:
        preset = PRESETS.get(arg)
        if preset is None:
            expanded.append(arg)
            continue
        for item in preset:
            if item not in expanded:
                expanded.append(item)
    return expanded


def load_robot_profile() -> str:
    """Read ROBOT_PROFILE without importing config.py and initializing CUDA."""
    tree = ast.parse(ROBOT_CONFIG_SOURCE.read_text(), filename=str(ROBOT_CONFIG_SOURCE))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "ROBOT_PROFILE"
               for target in node.targets):
            try:
                profile = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                profile = os.environ.get("UR10E_ROBOT_PROFILE", "old")
            if profile not in ("new", "old"):
                raise ValueError(f"ROBOT_PROFILE must be 'new' or 'old', got {profile!r}")
            return profile
    raise RuntimeError(f"ROBOT_PROFILE not found in {ROBOT_CONFIG_SOURCE}")


DEFAULT_ROBOT_PROFILE = load_robot_profile()


def kinematics_file_for(robot_profile: str) -> Path:
    return (
        Path(WS) / "install" / "ur_description" / "share" / "ur_description"
        / "config" / "ur10e"
        / ("default_kinematics.yaml" if robot_profile == "new" else "old_kinematics.yaml")
    )

RVIZ_CONFIG = str(Path(WS) / "install" / "rviz_ur10e_panel" / "share" / "rviz_ur10e_panel" / "rviz" / "view_robot.rviz")


def camera_mode_for(use_lidar=False, use_zed_mini=False, use_zedx_mini_only=False):
    if use_zedx_mini_only:
        return "zedx_mini"
    if use_zed_mini:
        return "zed_mini"
    if use_lidar:
        return "lidar"
    return "stereo"


def camera_profile_env(camera_mode: str) -> str:
    profiles = {
        "zedx_mini": (
            "zedx_mini_rgbd",
            "zed_mini_left_camera_frame",
            "zed_mini_left_camera_frame",
        ),
        "zed_mini": (
            "zed_one_rgb_zedx_mini_depth",
            "zed2_left_camera_frame",
            "zed_mini_left_camera_frame",
        ),
        "lidar": (
            "zed_one_rgb_zedx_mini_depth",
            "zed2_left_camera_frame",
            "zed_mini_left_camera_frame",
        ),
        "stereo": (
            "zed_one_rgb_zedx_mini_depth",
            "zed2_left_camera_frame",
            "zed_mini_left_camera_frame",
        ),
    }
    profile, rgb_frame, depth_frame = profiles.get(camera_mode, profiles["zedx_mini"])
    profile_path = CAMERA_PROFILE_DIR / f"{profile}.yaml"
    return (
        f"export UR10E_CAMERA_MODE={camera_mode} && "
        f"export UR10E_CAMERA_PROFILE={profile} && "
        f"export UR10E_CAMERA_PROFILE_PATH={profile_path} && "
        f"export UR10E_CAM_FRAME={rgb_frame} && "
        f"export UR10E_ZEDMINI_CAM_FRAME={depth_frame}"
    )

def get_commands(
    fake_hardware=False,
    use_panel=False,
    use_lidar=False,
    use_zed_mini=False,
    use_zedx_mini_only=False,
    use_vision=False,
    hand_eye_target="charuco",
    hand_eye_resolution="qhdplus",
    robot_profile=None,
    gripper_profile="old",
    gripper_enabled=True,
):
    robot_profile = robot_profile or DEFAULT_ROBOT_PROFILE
    kinematics_file = kinematics_file_for(robot_profile)
    camera_mode = camera_mode_for(use_lidar, use_zed_mini, use_zedx_mini_only)
    camera_env = camera_profile_env(camera_mode)
    source_with_camera = f"{SOURCE} && {camera_env}"
    profile_source = (
        f"{source_with_camera} && export UR10E_ROBOT_PROFILE={robot_profile} "
        f"&& export UR10E_GRIPPER_PROFILE={gripper_profile} "
        f"&& export UR10E_GRIPPER_ENABLED={'true' if gripper_enabled else 'false'}"
    )
    hw = "true" if fake_hardware else "false"
    vision = "true" if use_vision else "false"
    # Skip unet.sh for fake hardware since no real robot
    ur_prefix = "" if fake_hardware else f"{UNET_SCRIPT} && "
    # When panel is used, disable default rviz and launch rviz with our panel config
    launch_rviz = "false" if use_panel else "true"
    if use_zedx_mini_only:
        vision_flags = " --use_zedx_mini_only"
    elif use_zed_mini:
        vision_flags = " --use_zed_mini"
    elif use_lidar:
        vision_flags = " --use_lidar"
    else:
        vision_flags = ""
    cmds = {
        "ur": (
            f'{ur_prefix}{SOURCE} && ros2 launch ur_bringup ur_control.launch.py '
            f'ur_type:=ur10e robot_ip:={ROBOT_IP} reverse_ip:={ROBOT_PC_IP} '
            f'script_sender_port:={SCRIPT_SENDER_PORT} '
            f'robot_profile:={robot_profile} '
            f'kinematics_params_file:={kinematics_file} '
            f'use_fake_hardware:={hw} launch_gripper:={"true" if gripper_enabled else "false"} '
            f'launch_rviz:={launch_rviz}'
        ),
        "main": (
            f'sleep 15 && {profile_source} && ros2 run ur10e_curobo main --ros-args '
            f'-p planner.use_fake_hardware:={hw} '
            f'-p perception.enabled:={vision}'
        ),
        "vision": f'sleep 12 && {source_with_camera} && ros2 run ur10e_curobo vision{vision_flags}',
        "teleop": f'sleep 6 && {source_with_camera} && ros2 run ur10e_curobo teleop',
        "gui": f'sleep 8 && {profile_source} && ros2 run ur10e_curobo gui',
        "rqt": f'sleep 8 && {source_with_camera} && rqt --force-discover --standalone rqt_ur10e_panel',
        "calibrate": f'sleep 6 && {profile_source} && ros2 run ur10e_curobo calibrate',
        "hand_eye": (
            f'sleep 4 && {profile_source} && '
            f'ros2 run ur10e_curobo hand_eye '
            f'--target {hand_eye_target} --resolution {hand_eye_resolution} '
            f'--camera-mode {camera_mode}'
        ),
        "extrinsic": (
            f'sleep 4 && {SOURCE} && {camera_profile_env("zed_mini")} && '
            f'ros2 run ur10e_curobo zed_extrinsic'
        ),
        "lidar": (
            f'sleep 8 && {source_with_camera} && '
            # Static TF: zed2_left_camera_frame (parent) → livox_frame (child)
            f'ros2 run tf2_ros static_transform_publisher '
            f'--x -0.667081 --y -0.305516 --z -0.177982 '
            f'--qx 0.495951 --qy -0.468885 --qz 0.427989 --qw 0.592457 '
            f'--frame-id zed2_left_camera_frame --child-frame-id livox_frame & '
            f'ros2 launch livox_ros2_driver livox_lidar_launch.py'
        ),
    }
    if use_panel:
        cmds["rviz_panel"] = f'sleep 5 && {source_with_camera} && rviz2 -d {RVIZ_CONFIG}'
    return cmds

TITLES = {
    "ur": "UR Bringup",
    "main": "Main",
    "vision": "Vision",
    "teleop": "Teleop",
    "gui": "GUI",
    "rqt": "RQT Panel",
    "rviz_panel": "RViz+Panel",
    "calibrate": "Calibrate",
    "hand_eye": "Hand-Eye",
    "extrinsic": "ZED Extrinsic",
    "lidar": "LiDAR",
}

def make_command(cmd):
    return f'bash -c "{cmd}; exec bash"'

def build_layout(
    nodes,
    fake_hardware=False,
    use_panel=False,
    use_lidar=False,
    use_zed_mini=False,
    use_zedx_mini_only=False,
    use_vision=False,
    hand_eye_target="charuco",
    hand_eye_resolution="qhdplus",
    robot_profile=None,
    gripper_profile="old",
    gripper_enabled=True,
):
    COMMANDS = get_commands(
        fake_hardware, use_panel, use_lidar, use_zed_mini, use_zedx_mini_only, use_vision,
        hand_eye_target, hand_eye_resolution, robot_profile, gripper_profile,
        gripper_enabled,
    )
    """Build the dynamic layout section."""
    all_nodes = ["ur"] + nodes
    n = len(all_nodes)

    lines = []
    lines.append("  [[dynamic]]")
    lines.append("    [[[child0]]]")
    lines.append("      type = Window")
    lines.append('      parent = ""')
    lines.append("      order = 0")
    lines.append("      position = 100:100")
    lines.append("      maximised = True")
    lines.append("      fullscreen = False")
    lines.append("      size = 1920, 1080")

    if n == 1:
        lines.append("    [[[term_ur]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child0")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES['ur']}")
        lines.append(f"      command = '{make_command(COMMANDS['ur'])}'")

    elif n == 2:
        lines.append("    [[[child1]]]")
        lines.append("      type = HPaned")
        lines.append("      parent = child0")
        lines.append("      order = 0")
        lines.append("      position = 900")
        lines.append("      ratio = 0.5")
        lines.append("    [[[term_ur]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child1")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES['ur']}")
        lines.append(f"      command = '{make_command(COMMANDS['ur'])}'")
        lines.append(f"    [[[term_{nodes[0]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child1")
        lines.append("      order = 1")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[0]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[0]])}'")

    elif n == 3:
        lines.append("    [[[child1]]]")
        lines.append("      type = VPaned")
        lines.append("      parent = child0")
        lines.append("      order = 0")
        lines.append("      position = 500")
        lines.append("      ratio = 0.5")
        lines.append("    [[[child2]]]")
        lines.append("      type = HPaned")
        lines.append("      parent = child1")
        lines.append("      order = 0")
        lines.append("      position = 900")
        lines.append("      ratio = 0.5")
        lines.append("    [[[term_ur]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child2")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES['ur']}")
        lines.append(f"      command = '{make_command(COMMANDS['ur'])}'")
        lines.append(f"    [[[term_{nodes[0]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child2")
        lines.append("      order = 1")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[0]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[0]])}'")
        lines.append(f"    [[[term_{nodes[1]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child1")
        lines.append("      order = 1")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[1]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[1]])}'")

    elif n == 4:
        lines.append("    [[[child1]]]")
        lines.append("      type = VPaned")
        lines.append("      parent = child0")
        lines.append("      order = 0")
        lines.append("      position = 500")
        lines.append("      ratio = 0.5")
        lines.append("    [[[child2]]]")
        lines.append("      type = HPaned")
        lines.append("      parent = child1")
        lines.append("      order = 0")
        lines.append("      position = 900")
        lines.append("      ratio = 0.5")
        lines.append("    [[[child3]]]")
        lines.append("      type = HPaned")
        lines.append("      parent = child1")
        lines.append("      order = 1")
        lines.append("      position = 900")
        lines.append("      ratio = 0.5")
        lines.append("    [[[term_ur]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child2")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES['ur']}")
        lines.append(f"      command = '{make_command(COMMANDS['ur'])}'")
        lines.append(f"    [[[term_{nodes[0]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child2")
        lines.append("      order = 1")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[0]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[0]])}'")
        lines.append(f"    [[[term_{nodes[1]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child3")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[1]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[1]])}'")
        lines.append(f"    [[[term_{nodes[2]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child3")
        lines.append("      order = 1")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[2]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[2]])}'")

    else:  # n >= 5
        lines.append("    [[[child1]]]")
        lines.append("      type = VPaned")
        lines.append("      parent = child0")
        lines.append("      order = 0")
        lines.append("      position = 500")
        lines.append("      ratio = 0.5")
        lines.append("    [[[child2]]]")
        lines.append("      type = HPaned")
        lines.append("      parent = child1")
        lines.append("      order = 0")
        lines.append("      position = 900")
        lines.append("      ratio = 0.5")
        lines.append("    [[[child3]]]")
        lines.append("      type = HPaned")
        lines.append("      parent = child1")
        lines.append("      order = 1")
        lines.append("      position = 600")
        lines.append("      ratio = 0.33")
        lines.append("    [[[child4]]]")
        lines.append("      type = HPaned")
        lines.append("      parent = child3")
        lines.append("      order = 1")
        lines.append("      position = 600")
        lines.append("      ratio = 0.5")
        lines.append("    [[[term_ur]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child2")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES['ur']}")
        lines.append(f"      command = '{make_command(COMMANDS['ur'])}'")
        lines.append(f"    [[[term_{nodes[0]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child2")
        lines.append("      order = 1")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[0]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[0]])}'")
        lines.append(f"    [[[term_{nodes[1]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child3")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[1]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[1]])}'")
        lines.append(f"    [[[term_{nodes[2]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child4")
        lines.append("      order = 0")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[2]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[2]])}'")
        lines.append(f"    [[[term_{nodes[3]}]]]")
        lines.append("      type = Terminal")
        lines.append("      parent = child4")
        lines.append("      order = 1")
        lines.append("      profile = default")
        lines.append(f"      title = {TITLES[nodes[3]]}")
        lines.append(f"      command = '{make_command(COMMANDS[nodes[3]])}'")

    return '\n'.join(lines)

def update_config(
    nodes,
    fake_hardware=False,
    use_panel=False,
    use_lidar=False,
    use_zed_mini=False,
    use_zedx_mini_only=False,
    use_vision=False,
    hand_eye_target="charuco",
    hand_eye_resolution="qhdplus",
    robot_profile=None,
    gripper_profile="old",
    gripper_enabled=True,
):
    """Update terminator config with dynamic layout."""
    with open(CONFIG_PATH, 'r') as f:
        lines = f.readlines()

    # Find and remove existing [[dynamic]] section
    new_lines = []
    in_dynamic = False
    for line in lines:
        # Check if we're entering [[dynamic]] section (double bracket, not triple)
        if re.match(r'^\s*\[\[dynamic\]\]\s*$', line):
            in_dynamic = True
            continue
        # Check if we're exiting to a new [[section]] (double bracket only) or [plugins]
        if in_dynamic:
            # Match [[name]] but NOT [[[name]]] - use negative lookahead for third bracket
            if re.match(r'^\s*\[\[(?!\[)[a-zA-Z]', line) or line.strip() == '[plugins]':
                in_dynamic = False
            else:
                continue  # Skip lines inside [[dynamic]]
        new_lines.append(line)

    # Build new layout
    new_layout = build_layout(
        nodes, fake_hardware, use_panel, use_lidar, use_zed_mini, use_zedx_mini_only, use_vision,
        hand_eye_target, hand_eye_resolution, robot_profile, gripper_profile,
        gripper_enabled,
    )

    # Find [plugins] and insert before it
    final_lines = []
    for line in new_lines:
        if line.strip() == '[plugins]':
            final_lines.append(new_layout + '\n')
        final_lines.append(line)

    with open(CONFIG_PATH, 'w') as f:
        f.writelines(final_lines)

def main():
    valid = ["bringup", "ur", "main", "vision", "teleop", "gui", "rqt", "calibrate", "hand_eye", "extrinsic"]
    args = expand_presets(sys.argv[1:])

    # Extract modifier flags
    fake_hardware = "fake" in args
    robot_profile = DEFAULT_ROBOT_PROFILE
    if any(a in args for a in ("robot_new", "new_robot")):
        robot_profile = "new"
    elif any(a in args for a in ("robot_old", "old_robot")):
        robot_profile = "old"
    gripper_profile = "old"
    if any(a in args for a in ("gripper_new", "new_gripper")):
        gripper_profile = "new"
    elif any(a in args for a in ("gripper_old", "old_gripper")):
        gripper_profile = "old"
    gripper_enabled = not any(a in args for a in ("gripper_off", "no_gripper", "without_gripper"))
    if any(a in args for a in ("gripper_on", "use_gripper", "with_gripper")):
        gripper_enabled = True
    use_lidar     = "lidar" in args
    use_zed_mini  = "zed_mini" in args
    use_zedx_mini_only = "zedx_mini" in args
    hand_eye_target = "chessboard" if "chessboard" in args else "charuco"
    if any(a in args for a in ("hd1080", "1080", "1080p")):
        hand_eye_resolution = "hd1080"
    elif "4k" in args:
        hand_eye_resolution = "4k"
    else:
        hand_eye_resolution = "qhdplus"
    args = [
        a for a in args
        if a not in (
            "fake", "lidar", "zed_mini", "zedx_mini", "charuco", "aruco", "chessboard",
            "qhdplus", "qhd+", "4k", "hd1080", "1080", "1080p",
            "robot_old", "robot_new", "old_robot", "new_robot",
            "gripper_old", "gripper_new", "old_gripper", "new_gripper",
            "gripper_on", "gripper_off", "use_gripper", "no_gripper",
            "with_gripper", "without_gripper",
        )
    ]

    bringup_only = any(arg in ("bringup", "ur") for arg in args)
    nodes = [arg for arg in args if arg in valid and arg not in ("bringup", "ur")]

    if not nodes and not bringup_only:
        print("Usage: launch_ur10e [fake] [robot_old|robot_new] [gripper_old|gripper_new] [gripper_on|gripper_off] [harvest|field] [lidar|zed_mini|zedx_mini] [charuco|chessboard] [qhdplus|4k|hd1080] [bringup|main] [vision] [teleop] [gui] [calibrate] [hand_eye] [extrinsic]")
        print()
        print("Presets:")
        print("  harvest  - Real robot + main + vision with ZED X Mini RGBD")
        print("  field    - Alias for harvest")
        print()
        print("Options:")
        print("  fake      - Use fake/simulated hardware (no real robot)")
        print("  robot_old - Use old UR10e mounting/kinematics profile")
        print("  robot_new - Use new UR10e mounting/kinematics profile")
        print("  gripper_old - Use old calibrated gripper open/close postures")
        print("  gripper_new - Use DG-3F-M native paired-motor open/close postures")
        print("  gripper_off - Do not launch/control the Delto gripper")
        print("  lidar     - Use Livox Mid-70 LiDAR for depth (adds LiDAR pane, passes --use_lidar to vision)")
        print("  zed_mini  - Use ZED X Mini for depth + ZED One Mono for detection (passes --use_zed_mini to vision)")
        print("  zedx_mini - Use ZED X Mini for both RGB detection and depth")
        print("  charuco   - Use ChArUco target for hand-eye calibration (default)")
        print("  chessboard- Use plain chessboard target for hand-eye calibration")
        print("  qhdplus   - Hand-eye camera at QHD+ with HDR (default)")
        print("  4k        - Hand-eye camera at HD4K without HDR")
        print("  hd1080    - Hand-eye camera at HD1080 (required for ZED X Mini)")
        print("  bringup   - UR driver bringup only")
        print("  main      - Main control node (RViz includes control panel)")
        print("  vision    - Vision node")
        print("  teleop    - Teleop node")
        print("  gui       - GUI node (standalone PyQt)")
        print("  rqt       - RQT UR10e panel (auto-added with main)")
        print("  calibrate - Grasp force calibration tool")
        print("  hand_eye  - ChArUco hand-eye camera calibration")
        print("              Uses the selected camera mode: zedx_mini updates Mini RGBD profile; zed_mini updates ZED One RGB profile")
        print("  extrinsic - Calibrate ZED X Mini depth frame to ZED X One RGB frame")
        print()
        print("Examples:")
        print("  launch_ur10e_gui                     # Dialog launcher for robot/camera/depth choices")
        print("  launch_ur10e bringup                 # Real robot bringup only")
        print("  launch_ur10e main                    # Real robot + main + RViz with panel")
        print("  launch_ur10e main vision             # Real robot + main + vision (ZED stereo)")
        print("  launch_ur10e main vision zed_mini    # Real robot + main + vision (ZED Mini depth)")
        print("  launch_ur10e main vision zedx_mini   # Real robot + main + vision (ZED X Mini RGBD)")
        print("  launch_ur10e harvest                 # Same as: main vision zedx_mini")
        print("  launch_harvest                       # Short wrapper for the harvest preset")
        print("  launch_ur10e main vision lidar       # Real robot + main + vision + LiDAR depth")
        print("  launch_ur10e fake main               # Fake hardware + main")
        print("  launch_ur10e robot_new gripper_new fake main # New robot/gripper profiles in fake mode")
        print("  launch_ur10e gripper_off main       # Robot control without Delto gripper")
        print("  launch_ur10e main teleop             # Real robot + main + teleop")
        print("  launch_ur10e calibrate               # Grasp force calibration only")
        print("  launch_ur10e hand_eye                # Hand-eye calibration")
        print("  launch_ur10e main hand_eye           # Main + RViz + ChArUco hand-eye")
        print("  launch_ur10e main hand_eye chessboard # Main + RViz + chessboard hand-eye")
        print("  launch_ur10e main hand_eye charuco 4k # ChArUco hand-eye at 4K")
        print("  launch_ur10e zedx_mini main hand_eye hd1080 # ZED X Mini hand-eye at 1080p")
        print("  launch_ur10e extrinsic               # ZED One ↔ ZED X Mini extrinsic calibration")
        sys.exit(1)

    # When main is specified, use panel-integrated RViz
    use_panel = "main" in nodes
    use_vision = "vision" in nodes

    # Reorder: main, vision, teleop, gui, rqt, calibrate, hand_eye, extrinsic
    ordered = []
    for n in ["main", "vision", "teleop", "gui", "rqt", "calibrate", "hand_eye", "extrinsic"]:
        if n in nodes:
            ordered.append(n)

    # LiDAR pane comes before vision (driver must be up first)
    if use_lidar and "vision" in ordered:
        ordered.insert(ordered.index("vision"), "lidar")

    # Add rviz_panel pane (separate RViz with integrated control panel)
    if use_panel:
        ordered.append("rviz_panel")

    hw_mode = "FAKE hardware" if fake_hardware else "REAL robot"
    depth_note = (
        " +ZedXMiniRGBD" if use_zedx_mini_only else
        " +ZedMini" if use_zed_mini else
        " +LiDAR" if use_lidar else
        ""
    )
    panel_note = " (RViz with control panel)" if use_panel else ""
    print(
        f"Launching {len(ordered) + 1} panes "
        f"(robot={robot_profile}, gripper={gripper_profile}, "
        f"gripper_enabled={gripper_enabled}, {hw_mode}{depth_note}): "
        f"ur_bringup + {', '.join(ordered)}{panel_note}")

    # Update config with dynamic layout
    update_config(
        ordered,
        fake_hardware,
        use_panel,
        use_lidar,
        use_zed_mini,
        use_zedx_mini_only,
        use_vision,
        hand_eye_target,
        hand_eye_resolution,
        robot_profile,
        gripper_profile,
        gripper_enabled,
    )

    # Launch terminator with the layout
    subprocess.Popen(["terminator", "-l", "dynamic"])

if __name__ == "__main__":
    main()
