#!/usr/bin/env python3
"""Launch UR10e nodes in a single terminator window with panes."""
import sys
import subprocess
import os
import re

WS = "/home/datepalm2/manipulatorsdatepalm/ur_ws_new"
SOURCE = f"source {WS}/install/setup.bash"
CONFIG_PATH = os.path.expanduser("~/.config/terminator/config")
UNET_SCRIPT = "/home/datepalm2/manipulatorsdatepalm/bin/unet.sh"

def get_commands(fake_hardware=False):
    hw = "true" if fake_hardware else "false"
    # Skip unet.sh for fake hardware since no real robot
    ur_prefix = "" if fake_hardware else f"{UNET_SCRIPT} && "
    return {
        "ur": f'{ur_prefix}{SOURCE} && ros2 launch ur_bringup ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.1.190 use_fake_hardware:={hw} launch_rviz:=true',
        "main": f'sleep 5 && {SOURCE} && ros2 run ur10e_curobo main',
        "vision": f'sleep 7 && {SOURCE} && ros2 run ur10e_curobo vision',
        "teleop": f'sleep 6 && {SOURCE} && ros2 run ur10e_curobo teleop',
        "gui": f'sleep 8 && {SOURCE} && ros2 run ur10e_curobo gui',
    }

TITLES = {
    "ur": "UR Bringup",
    "main": "Main",
    "vision": "Vision",
    "teleop": "Teleop",
    "gui": "GUI",
}

def make_command(cmd):
    return f'bash -c "{cmd}; exec bash"'

def build_layout(nodes, fake_hardware=False):
    COMMANDS = get_commands(fake_hardware)
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

def update_config(nodes, fake_hardware=False):
    """Update terminator config with dynamic layout."""
    with open(CONFIG_PATH, 'r') as f:
        lines = f.readlines()

    # Find and remove existing [[dynamic]] section
    new_lines = []
    in_dynamic = False
    for line in lines:
        # Check if we're entering [[dynamic]] section
        if re.match(r'\s*\[\[dynamic\]\]', line):
            in_dynamic = True
            continue
        # Check if we're exiting to a new [[section]] or [plugins]
        if in_dynamic:
            if re.match(r'\s*\[\[(?!dynamic)', line) or line.strip() == '[plugins]':
                in_dynamic = False
            else:
                continue  # Skip lines inside [[dynamic]]
        new_lines.append(line)

    # Build new layout
    new_layout = build_layout(nodes, fake_hardware)

    # Find [plugins] and insert before it
    final_lines = []
    for line in new_lines:
        if line.strip() == '[plugins]':
            final_lines.append(new_layout + '\n')
        final_lines.append(line)

    with open(CONFIG_PATH, 'w') as f:
        f.writelines(final_lines)

def main():
    valid = ["main", "vision", "teleop", "gui"]
    args = sys.argv[1:]

    # Check for fake hardware flag
    fake_hardware = "fake" in args
    args = [a for a in args if a != "fake"]

    nodes = [arg for arg in args if arg in valid]

    if not nodes:
        print("Usage: launch_ur10e [fake] [main] [vision] [teleop] [gui]")
        print()
        print("Options:")
        print("  fake   - Use fake/simulated hardware (no real robot)")
        print("  main   - Main control node")
        print("  vision - Vision node")
        print("  teleop - Teleop node")
        print("  gui    - GUI node")
        print()
        print("Examples:")
        print("  launch_ur10e main              # Real robot + main")
        print("  launch_ur10e fake main         # Fake hardware + main")
        print("  launch_ur10e main teleop       # Real robot + main + teleop")
        print("  launch_ur10e fake main vision  # Fake hardware + main + vision")
        sys.exit(1)

    # Reorder to: main, vision, teleop, gui
    ordered = []
    for n in ["main", "vision", "teleop", "gui"]:
        if n in nodes:
            ordered.append(n)

    hw_mode = "FAKE hardware" if fake_hardware else "REAL robot"
    print(f"Launching {len(ordered) + 1} panes ({hw_mode}): ur_bringup + {', '.join(ordered)}")

    # Update config with dynamic layout
    update_config(ordered, fake_hardware)

    # Launch terminator with the layout
    subprocess.Popen(["terminator", "-l", "dynamic"])

if __name__ == "__main__":
    main()
