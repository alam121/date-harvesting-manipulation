"""Physics check of the dynamic bunch (nurec_bunch.py): rest sag, poke response, date pull.

  unset CONDA_SHLVL
  ~/miniconda3/envs/env_isaaclab/bin/python isaac/nurec/bunch_test.py ~/nurec_data/scenes/palm3/split \
      --nurec_pose ... --headless
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("split", type=Path)
parser.add_argument("--nurec_pose", type=float, nargs=7, required=True)
parser.add_argument("--break_force", type=float, default=5.0)
parser.add_argument("--sag", type=float, default=0.01, help="Design sag per joint [rad].")
parser.add_argument("--damping_ratio", type=float, default=1.0)
parser.add_argument("--section_length", type=float, default=0.06, help="Physics section length [m].")
parser.add_argument("--date_mass", type=float, default=0.01)
parser.add_argument("--min_section_mass", type=float, default=0.02)
parser.add_argument("--min_stiffness", type=float, default=0.3)
parser.add_argument("--body_damping", type=float, default=6.0, help="Section linear/angular damping [1/s].")
parser.add_argument("--rest_seconds", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaacsim.core.prims import RigidPrim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nurec_bunch import add_bunch  # noqa: E402

DT = 1.0 / 240.0


def main():
    import time
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    built = add_bunch(sim.stage, args.split, args.nurec_pose, break_force=args.break_force,
                      sag_rad=args.sag, damping_ratio=args.damping_ratio, visual=False,
                      section_length=args.section_length, date_mass=args.date_mass,
                      min_section_mass=args.min_section_mass, min_stiffness=args.min_stiffness,
                      body_damping=args.body_damping)
    sim.reset()
    segments = RigidPrim(prim_paths_expr="/World/Bunch/section_[0-9][0-9][0-9]")
    dates = RigidPrim(prim_paths_expr="/World/Dates/date_[0-9][0-9]")
    segments.initialize()
    dates.initialize()
    count = len(segments.prim_paths)

    def seg_pos():
        return segments.get_world_poses()[0].cpu().numpy()

    rest0 = seg_pos()
    started = time.time()
    for _ in range(int(args.rest_seconds / DT)):
        sim.step()
    print(f"[BUNCH] {args.rest_seconds:.0f} s simulated in {time.time() - started:.1f} s wall ({args.device})", flush=True)
    rest = seg_pos()
    sag = np.linalg.norm(rest - rest0, axis=1)
    print(f"[BUNCH] after 5 s at rest: segment drift median {np.median(sag)*100:.1f} cm, "
          f"max {sag.max()*100:.1f} cm (tip of the longest strand)")
    speed = np.linalg.norm(segments.get_velocities().cpu().numpy()[:, :3], axis=1)
    print(f"[BUNCH] residual segment speed max {speed.max()*100:.2f} cm/s (settled if ~0)")
    carriers = set(built["date_segment"].values())
    for k in np.argsort(-speed)[:6]:
        name = segments.prim_paths[k]
        print(f"[BUNCH]   moving: {Path(name).name} {speed[k]*100:5.1f} cm/s, drift {sag[k]*100:5.1f} cm, "
              f"{'carries a date' if name in carriers else 'no date'}, depth {np.linalg.norm(rest0[k] - rest0[0]):.2f} m")

    # Poke: 0.2 s sideways push (world +y) on a segment half way down the longest chain.
    depth = np.linalg.norm(rest - rest[:1], axis=1)
    target = int(np.argsort(depth)[len(depth) // 2])
    forces = torch.zeros((count, 3), device=args.device)
    forces[target, 1] = 0.5
    start = seg_pos()[target]
    peak = 0.0
    for step in range(int(3.0 / DT)):
        if step < int(0.2 / DT):
            segments.apply_forces_and_torques_at_pos(forces=forces, positions=torch.zeros((count, 3), device=args.device), is_global=True)
        sim.step()
        peak = max(peak, float(np.linalg.norm(seg_pos()[target] - start)))
    back = float(np.linalg.norm(seg_pos()[target] - start))
    print(f"[BUNCH] poke 0.5 N x 0.2 s on {Path(segments.prim_paths[target]).name}: swung {peak*100:.1f} cm, "
          f"{back*100:.1f} cm from rest after 3 s")

    # Pull one date outward until it comes off; report its strand's motion.
    names = [Path(p).name for p in dates.prim_paths]
    date_start = dates.get_world_poses()[0].cpu().numpy()
    pick = int(np.argmin(np.linalg.norm(date_start - np.array([0.0, 0.0, 0.3]), axis=1)))
    strand = built["date_segment"][int(names[pick].split("_")[1])]
    strand_index = segments.prim_paths.index(strand)
    strand_start = seg_pos()[strand_index]
    offset0 = np.linalg.norm(date_start[pick] - strand_start)
    pull = torch.zeros((len(names), 3), device=args.device)
    detached_at, strand_peak = None, 0.0
    direction = torch.tensor([-1.0, 0.0, 0.0], device=args.device)
    for step in range(int(4.0 / DT)):
        force = 2.0 * args.break_force * step * DT / 4.0
        pull[pick] = direction * force if detached_at is None else 0.0
        dates.apply_forces_and_torques_at_pos(forces=pull, positions=torch.zeros((len(names), 3), device=args.device), is_global=True)
        sim.step()
        now = dates.get_world_poses()[0].cpu().numpy()[pick]
        strand_peak = max(strand_peak, float(np.linalg.norm(seg_pos()[strand_index] - strand_start)))
        # Detached = separated from its strand (the strand itself bends while pulled).
        if detached_at is None and np.linalg.norm(now - seg_pos()[strand_index]) - offset0 > 0.02:
            detached_at = force
    print(f"[BUNCH] pull on {names[pick]}: came off at {detached_at if detached_at is None else round(detached_at, 2)} N; "
          f"its strand ({Path(strand).name}) was dragged up to {strand_peak*100:.1f} cm")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
