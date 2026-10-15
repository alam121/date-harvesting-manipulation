"""Measure the pull that detaches each date (checks --date_break_force / nurec_dates.py joints).

Ramps a horizontal force on every date at once (world +x, so it does not add to the weight the
stem already carries) and records the force at which each date comes loose. No tree or robot is
loaded, so it runs in about a minute.

  unset CONDA_SHLVL
  ~/miniconda3/envs/env_isaaclab/bin/python isaac/nurec/date_pull_test.py \
      ~/nurec_data/scenes/palm3/split/dates_physics.json --nurec_pose ... --break_force 5 --headless
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("manifest", type=Path)
parser.add_argument("--nurec_pose", type=float, nargs=7, required=True)
parser.add_argument("--break_force", type=float, default=5.0)
parser.add_argument("--break_torque", type=float, default=1.0e6)
parser.add_argument("--mass", type=float, default=0.01)
parser.add_argument("--ramp_seconds", type=float, default=4.0, help="Time to ramp to 2x the break force.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaacsim.core.prims import RigidPrim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nurec_dates import add_dates  # noqa: E402

DT = 1.0 / 240.0


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cpu"))
    add_dates(sim.stage, args.manifest, args.nurec_pose, mass=args.mass,
              break_force=args.break_force, break_torque=args.break_torque, visual_mode="none")
    sim.reset()
    dates = RigidPrim(prim_paths_expr="/World/Dates/date_[0-9][0-9]")
    dates.initialize()
    count = len(dates.prim_paths)
    start, _ = dates.get_world_poses()
    detached_at = np.full(count, np.nan)

    steps = int(args.ramp_seconds / DT)
    for step in range(steps):
        force = 2.0 * args.break_force * step / steps
        forces = torch.zeros((count, 3))
        forces[:, 0] = force
        forces[~torch.from_numpy(np.isnan(detached_at)), 0] = 0.0  # stop pushing once loose
        # Apply through each date's centre (positions are in the body frame, so zero).
        dates.apply_forces_and_torques_at_pos(forces=forces, positions=torch.zeros((count, 3)), is_global=True)
        sim.step()
        now, _ = dates.get_world_poses()
        moved = torch.linalg.norm(now - start, dim=1).numpy() > 0.005
        detached_at[np.isnan(detached_at) & moved] = force

    names = [Path(p).name for p in dates.prim_paths]
    weight = args.mass * 9.81
    for name, force in zip(names, detached_at):
        total = np.hypot(force, weight) if not np.isnan(force) else np.nan
        print(f"[PULL] {name}: detached at pull {force:5.2f} N (joint load {total:5.2f} N)")
    ok = np.abs(np.hypot(detached_at, weight) - args.break_force) < 0.1 * args.break_force
    print(f"[PULL] {int(np.sum(~np.isnan(detached_at)))}/{count} detached; "
          f"{int(ok.sum())} within 10% of the {args.break_force} N setting")


if __name__ == "__main__":
    main()
    os._exit(0)
