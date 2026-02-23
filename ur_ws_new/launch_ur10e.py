#!/usr/bin/env python3
"""Launch UR10e nodes in a single terminator window with panes."""
import sys
import subprocess
import os
import re
from pathlib import Path

WS = str(Path(__file__).resolve().parent)
REPO_ROOT = str(Path(WS).parent)
SOURCE = f"source {WS}/install/setup.bash"
CONFIG_PATH = os.path.expanduser("~/.config/terminator/config")
UNET_SCRIPT = str(Path(REPO_ROOT) / "bin" / "unet.sh")

RVIZ_CONFIG = str(Path(WS) / "install" / "rviz_ur10e_panel" / "share" / "rviz_ur10e_panel" / "rviz" / "view_robot.rviz")

def get_commands(fake_hardware=False, use_panel=False):
    hw = "true" if fake_hardware else "false"
    # Skip unet.sh for fake hardware since no real robot
    ur_prefix = "" if fake_hardware else f"{UNET_SCRIPT} && "
    # When panel is used, disable default rviz and launch rviz with our panel config
    launch_rviz = "false" if use_panel else "true"
    cmds = {
        "ur": f'{ur_prefix}{SOURCE} && ros2 launch ur_bringup ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.1.190 use_fake_hardware:={hw} launch_rviz:={launch_rviz}',
        "main": f'sleep 5 && {SOURCE} && ros2 run ur10e_curobo main',
        "vision": f'sleep 2 && {SOURCE} && ros2 run ur10e_curobo vision',
        "teleop": f'sleep 6 && {SOURCE} && ros2 run ur10e_curobo teleop',
        "gui": f'sleep 8 && {SOURCE} && ros2 run ur10e_curobo gui',
        "calibrate": f'sleep 6 && {SOURCE} && ros2 run ur10e_curobo calibrate',
    }
    if use_panel:
        cmds["rviz_panel"] = f'sleep 5 && {SOURCE} && rviz2 -d {RVIZ_CONFIG}'
    return cmds

TITLES = {
    "ur": "UR Bringup",
    "main": "Main",
    "vision": "Vision",
    "teleop": "Teleop",
    "gui": "GUI",
    "rviz_panel": "RViz+Panel",
    "calibrate": "Calibrate",
}

def make_command(cmd):
    return f'bash -c "{cmd}; exec bash"'

def build_layout(nodes, fake_hardware=False, use_panel=False):
    COMMANDS = get_commands(fake_hardware, use_panel)
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

def update_config(nodes, fake_hardware=False, use_panel=False):
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
    new_layout = build_layout(nodes, fake_hardware, use_panel)

    # Find [plugins] and insert before it
    final_lines = []
    for line in new_lines:
        if line.strip() == '[plugins]':
            final_lines.append(new_layout + '\n')
        final_lines.append(line)

    with open(CONFIG_PATH, 'w') as f:
        f.writelines(final_lines)

def main():
    valid = ["main", "vision", "teleop", "gui", "calibrate"]
    args = sys.argv[1:]

    # Check for fake hardware flag
    fake_hardware = "fake" in args
    args = [a for a in args if a != "fake"]

    nodes = [arg for arg in args if arg in valid]

    if not nodes:
        print("Usage: launch_ur10e [fake] [main] [vision] [teleop] [gui] [calibrate]")
        print()
        print("Options:")
        print("  fake      - Use fake/simulated hardware (no real robot)")
        print("  main      - Main control node (RViz includes control panel)")
        print("  vision    - Vision node")
        print("  teleop    - Teleop node")
        print("  gui       - GUI node (standalone PyQt)")
        print("  calibrate - Grasp force calibration tool")
        print()
        print("Examples:")
        print("  launch_ur10e main              # Real robot + main + RViz with panel")
        print("  launch_ur10e main vision       # Real robot + main + vision + RViz with panel")
        print("  launch_ur10e fake main         # Fake hardware + main")
        print("  launch_ur10e main teleop       # Real robot + main + teleop")
        print("  launch_ur10e calibrate         # Grasp force calibration only")
        print("  launch_ur10e calibrate teleop  # Calibrate + teleop (jog robot)")
        sys.exit(1)

    # When main is specified, use panel-integrated RViz
    # UR bringup launches with launch_rviz:=false, and we add a separate rviz_panel pane
    use_panel = "main" in nodes

    # Reorder to: main, vision, teleop, gui, calibrate
    ordered = []
    for n in ["main", "vision", "teleop", "gui", "calibrate"]:
        if n in nodes:
            ordered.append(n)

    # Add rviz_panel pane (separate RViz with integrated control panel)
    if use_panel:
        ordered.append("rviz_panel")

    hw_mode = "FAKE hardware" if fake_hardware else "REAL robot"
    panel_note = " (RViz with control panel)" if use_panel else ""
    print(f"Launching {len(ordered) + 1} panes ({hw_mode}): ur_bringup + {', '.join(ordered)}{panel_note}")

    # Update config with dynamic layout
    update_config(ordered, fake_hardware, use_panel)

    # Launch terminator with the layout
    subprocess.Popen(["terminator", "-l", "dynamic"])

if __name__ == "__main__":
    main()
