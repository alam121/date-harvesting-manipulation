"""Hardware-like colours for the UR10e + Delto asset (its meshes are all plain white).

White washes out against the bright NuRec scene (the small gripper practically vanished).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

# (diffuse, roughness, metallic): light-grey aluminium UR arm, dark anodised Delto hand.
ROBOT_LOOKS = {"arm": ((0.60, 0.62, 0.64), 0.35, 0.5), "gripper": ((0.05, 0.05, 0.06), 0.45, 0.3)}


def colour_robot(stage, robot_path: str = "/World/envs/env_0/Robot"):
    for name, (colour, roughness, metallic) in ROBOT_LOOKS.items():
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=colour, roughness=roughness, metallic=metallic)
        material.func(f"/World/Looks/robot_{name}", material)
    for link in stage.GetPrimAtPath(robot_path).GetChildren():
        visuals = link.GetChild("visuals")
        if visuals:
            # Instanced prototypes keep their own (white) binding; make each link's copy editable.
            visuals.SetInstanceable(False)
            look = "gripper" if link.GetName().startswith("gripper") else "arm"
            sim_utils.bind_visual_material(str(visuals.GetPath()), f"/World/Looks/robot_{look}", stage=stage)
