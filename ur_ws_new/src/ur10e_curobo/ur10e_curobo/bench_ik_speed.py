#!/usr/bin/env python3
"""Benchmark the v2 IK cold/steady-state cost and world-update overhead.

Run with the cuRobo v2 environment: python -m ur10e_curobo.bench_ik_speed.
This benchmark computes plans only and never sends robot commands.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

import torch
from curobo.types import Pose, JointState
from .planner_v2 import PlannerV2, load_robot_config

try:
    from ur10e_curobo.config import WORLD_CONFIG as PROD_WORLD
except Exception:  # noqa: BLE001 - benchmark still runs with an empty world
    PROD_WORLD = None


def timed(fn: Callable[[], None], n: int) -> List[float]:
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def summarise(label: str, ms: List[float]) -> float:
    first = ms[0]
    rest = ms[1:] or ms
    med = statistics.median(rest)
    print(f"  {label:<34} first {first:7.1f}ms   median {med:7.1f}ms   "
          f"min {min(rest):6.1f}ms  max {max(rest):6.1f}ms")
    return med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-config", default="ur10e.yml")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--return-seeds", type=int, default=8)
    ap.add_argument("--empty-world", action="store_true",
                    help="use an empty collision world instead of production's")
    ap.add_argument("--update-world", action="store_true",
                    help="call update_world() after warmup, as production does "
                         "when it applies the safe-zone walls")
    ap.add_argument("--hard-pose", action="store_true",
                    help="target a far/low VERY_LOW-style pose instead of a "
                         "2cm nudge off the retract config")
    args = ap.parse_args()

    print("\ncuRobo IK benchmark — building the v2 planner (this takes a while)…")
    robot_cfg = load_robot_config(args.robot_config)
    # Match production's collision world. The first version of this benchmark
    # used an empty world and measured 17ms while the field sees 250ms, so the
    # world is one of the two remaining differences.
    if args.empty_world or PROD_WORLD is None:
        world = {}
        print("world: EMPTY")
    else:
        world = PROD_WORLD
        n_obs = sum(len(v) for v in PROD_WORLD.values() if hasattr(v, "__len__"))
        print(f"world: production WORLD_CONFIG ({n_obs} obstacle(s))")
    mg = PlannerV2(robot_cfg, world, interpolation_dt=0.004, num_trajopt_seeds=2)
    mg.warmup()
    print("v2 warmup done")
    dev = torch.device('cuda')
    start_js = mg.planner.default_joint_state.position.view(1, -1).clone()
    seed_t = start_js.clone().unsqueeze(0)
    retract_t = start_js.clone()
    state = mg.planner.compute_kinematics(
        JointState.from_position(start_js, joint_names=mg.planner.joint_names))
    tool = state.tool_poses.get_link_pose(mg.tool_frame)
    pos = tool.position.clone().view(1, 3)
    quat = tool.quaternion.clone().view(1, 4)
    if args.hard_pose:
        # A real VERY_LOW target from the field logs: far out, low, near the
        # edge of the workspace. IK difficulty is the other difference between
        # this benchmark and production.
        pos[0, 0], pos[0, 1], pos[0, 2] = 0.398, -0.974, 0.402
        print("pose: VERY_LOW field target [0.398, -0.974, 0.402]")
    else:
        pos[0, 0] += 0.02
        print("pose: retract config + 2cm")
    goal = Pose(position=pos.to(dev), quaternion=quat.to(dev))

    def call_a() -> None:
        mg.solve_ik(goal, seed_config=seed_t, retract_config=retract_t)

    def call_b() -> None:
        mg.solve_ik(goal, seed_config=seed_t, retract_config=retract_t,
                        return_seeds=args.return_seeds)

    if args.update_world:
        # Production wires _safe_zone_on_change AFTER warmup and then calls
        # motion_gen.update_world() to add the safe-zone wall panels. A world
        # update invalidates cuRobo's captured CUDA graphs, so if that is what
        # costs the field its 150-400ms per solve, it will show up here.
        from curobo.scene import Cuboid
        wm = mg.world_model
        wm.cuboid = list(wm.cuboid or []) + [
            Cuboid(name=f"safezone_{i}", pose=[0.0, 0.0, -2.0, 1, 0, 0, 0],
                   dims=[0.01, 2.0, 2.0])
            for i in range(4)
        ]
        mg.update_world(wm)
        print("update_world(): 4 safe-zone-style panels added AFTER warmup\n")

    n = args.iters
    print("=" * 78)
    print("  Shape A = corridor preflight       (no return_seeds)")
    print(f"  Shape B = VERY_LOW branch search   (return_seeds={args.return_seeds})")
    print("=" * 78)

    a_cold = summarise("A, straight after warmup()", timed(call_a, n))
    a_warm = summarise("A, repeated (steady state)", timed(call_a, n))
    b_cold = summarise("B, first use", timed(call_b, n))
    b_warm = summarise("B, repeated (steady state)", timed(call_b, n))

    alt: List[float] = []
    for i in range(n * 2):
        alt += timed(call_a if i % 2 == 0 else call_b, 1)
    a_alt = statistics.median(alt[0::2])
    b_alt = statistics.median(alt[1::2])
    print(f"  {'A and B alternating':<34} A median {a_alt:7.1f}ms   "
          f"B median {b_alt:7.1f}ms")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    field_ms = 250.0
    if a_warm < 50.0 and b_warm < 50.0:
        print(f"  Both shapes are fast in isolation (A {a_warm:.0f}ms, "
              f"B {b_warm:.0f}ms).")
        if a_alt > a_warm * 2 or b_alt > b_warm * 2:
            print(f"  But alternating degrades them (A {a_alt:.0f}ms, "
                  f"B {b_alt:.0f}ms): the two shapes EVICT each other's graph.")
            print("  FIX: stop mixing shapes. Use one call signature everywhere,")
            print("       or give each its own IKSolver instance.")
        else:
            print(f"  Alternating is fine too. The ~{field_ms:.0f}ms seen in the")
            print("  field is therefore NOT inherent to the call — look for what")
            print("  differs in production (world/obstacle updates between solves,")
            print("  or another component resetting the graph).")
    else:
        print(f"  Shapes are slow even in isolation (A {a_warm:.0f}ms, "
              f"B {b_warm:.0f}ms) and match the field's ~{field_ms:.0f}ms.")
        print("  The IK CUDA graph is not being used for this entry point.")
        print("  FIX: warm PlannerV2.solve_ik() with BOTH exact shapes right")
        print("       after motion_gen.warmup(), and keep the signatures stable.")
    if a_cold > a_warm * 2:
        print(f"\n  (A's first call is {a_cold / max(a_warm, 1e-6):.1f}x the steady "
              "state — that part IS one-off capture and a warmup would remove it.)")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
