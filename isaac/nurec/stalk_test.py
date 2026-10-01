"""How a date behaves on its stalk: at rest, under a finger push, and pulled until it detaches.

Static dates (pinned to the tree) or the dynamic bunch, at the harvest scene's 120 Hz step:
  rest  3 s       drift of every date from its captured pose, residual speed (stability)
  push  1 N, 0.5 s on the date nearest the robot, away from it (a finger meeting it on insertion):
                  peak displacement, where it settles, whether the stem broke
  pull  ramp      outward pull on the same date until it comes off: detachment force

  unset CONDA_SHLVL
  ~/miniconda3/envs/env_isaaclab/bin/python isaac/nurec/stalk_test.py ~/nurec_data/scenes/palm3/split \\
      --nurec_pose ... [--bunch] [--stalk_stiffness 0.1] [--sag 0.01] --headless
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("split", type=Path)
parser.add_argument("--nurec_pose", type=float, nargs=7, required=True)
parser.add_argument("--bunch", action="store_true", help="Dates on the dynamic bunch (else pinned to the tree).")
parser.add_argument("--stalk_stiffness", type=float, default=0.1, help="N m/rad; 0 = rigid pin at the date centre.")
parser.add_argument("--stalk_damping_ratio", type=float, default=1.0)
parser.add_argument("--break_force", type=float, default=8.0)
parser.add_argument("--sag", type=float, default=0.0005, help="Bunch: design sag per joint [rad] (smaller = stiffer).")
parser.add_argument("--min_stiffness", type=float, default=4.0, help="Bunch: minimum joint stiffness [N m/rad].")
parser.add_argument("--drive_frequency", type=float, default=0.0,
                    help="Bunch: acceleration drives ringing at this frequency [Hz] (0 = force springs).")
parser.add_argument("--solver_iterations", type=int, default=255, help="Bunch: position iterations per section.")
parser.add_argument("--push", type=float, default=1.0, help="Finger push [N].")
parser.add_argument("--push_seconds", type=float, default=0.5)
parser.add_argument("--dt", type=float, default=1.0 / 120.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaacsim.core.prims import RigidPrim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nurec_bunch import add_bunch  # noqa: E402
from nurec_dates import add_dates  # noqa: E402

ROBOT_BASE = np.array([0.0, 0.0, 0.3])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    if args.bunch:
        add_bunch(sim.stage, args.split, args.nurec_pose, break_force=args.break_force, sag_rad=args.sag,
                  min_stiffness=args.min_stiffness, visual=False, stalk_stiffness=args.stalk_stiffness,
                  drive_frequency=args.drive_frequency, solver_iterations=args.solver_iterations)
    else:
        add_dates(sim.stage, args.split / "dates_physics.json", args.nurec_pose, break_force=args.break_force,
                  visual_mode="none", stalk_stiffness=args.stalk_stiffness,
                  stalk_damping_ratio=args.stalk_damping_ratio)
    sim.reset()
    dates = RigidPrim(prim_paths_expr="/World/Dates/date_[0-9][0-9]")
    dates.initialize()
    n = len(dates.prim_paths)
    names = [Path(p).name for p in dates.prim_paths]

    def pos():
        return dates.get_world_poses()[0].cpu().numpy()

    captured = pos()
    for _ in range(int(3.0 / args.dt)):
        sim.step()
    rest = pos()
    drift = np.linalg.norm(rest - captured, axis=1)
    speed = np.linalg.norm(dates.get_velocities().cpu().numpy()[:, :3], axis=1)
    print(f"[STALK] rest 3 s: date drift median {np.median(drift) * 100:.1f} cm, max {drift.max() * 100:.1f} cm; "
          f"residual speed max {speed.max() * 100:.2f} cm/s", flush=True)

    k = int(np.argmin(np.linalg.norm(rest - ROBOT_BASE, axis=1)))
    away = rest[k] - ROBOT_BASE
    away[2] = 0.0
    away /= np.linalg.norm(away)
    force = torch.zeros((n, 3), device=args.device)
    zeros = torch.zeros((n, 3), device=args.device)
    peak, held = 0.0, 0.0
    for step in range(int((args.push_seconds + 3.0) / args.dt)):
        if step == int(args.push_seconds / args.dt) - 1:
            held = float(np.linalg.norm(pos()[k] - rest[k]))  # displacement at the end of the push
        if step < int(args.push_seconds / args.dt):
            force[k] = torch.tensor(away * args.push, device=args.device, dtype=torch.float32)
        else:
            force[k] = 0.0
        dates.apply_forces_and_torques_at_pos(forces=force, positions=zeros, is_global=True)
        sim.step()
        peak = max(peak, float(np.linalg.norm(pos()[k] - rest[k])))
    settled = float(np.linalg.norm(pos()[k] - rest[k]))
    fell = settled > 0.05
    print(f"[STALK] push {args.push:.1f} N x {args.push_seconds:.1f} s on {names[k]}: moved up to {peak * 100:.1f} cm "
          f"({held * 100:.1f} cm at the end of the push), "
          f"{settled * 100:.1f} cm from rest after 3 s{' -- DETACHED' if fell else ''}", flush=True)

    if not fell:
        start = pos()[k]
        detached_at = None
        for step in range(int(4.0 / args.dt)):
            pull = 2.0 * args.break_force * step * args.dt / 4.0
            force[k] = torch.tensor(-away * pull if detached_at is None else np.zeros(3), device=args.device,
                                    dtype=torch.float32)
            dates.apply_forces_and_torques_at_pos(forces=force, positions=zeros, is_global=True)
            sim.step()
            if detached_at is None and np.linalg.norm(pos()[k] - start) > 0.04:
                detached_at = pull
        print(f"[STALK] pull towards the robot on {names[k]}: "
              f"{'came off at %.1f N' % detached_at if detached_at else 'did not come off'} "
              f"(break force {args.break_force:.1f} N)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
