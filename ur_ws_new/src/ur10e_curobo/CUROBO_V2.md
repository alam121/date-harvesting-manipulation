# cuRobo v2 runtime

This branch uses NVIDIA cuRobo v2 for motion planning, inverse kinematics,
forward kinematics, and collision checks. The running robot must be restarted
to load a changed planner; switching Git branches does not replace a running
Python process.

The upstream revision is pinned in `requirements-curobo-v2.txt`:
`78fd485fa82d9b9a063fb4985e371814587e666a`.

## Install and launch

From the repository root:

```bash
./bin/setup_curobo_v2
source /opt/ros/humble/setup.bash
cd ur_ws_new
colcon build --symlink-install --packages-select ur10e_curobo
source install/setup.bash
../bin/launch_harvest
```

The `main` ROS executable selects `~/.venvs/datepalm-curobo-v2/bin/python`.
Override with `UR10E_CUROBO_V2_PYTHON` if the environment lives elsewhere.
Vision, NuRec, RViz, and the robot driver keep their existing interpreters and
launch paths. The v2 environment reuses the Jetson's NVIDIA PyTorch installation
and ROS Python packages; it does not replace the system cuRobo installation.
The vendored v1 source tree is historical and is not imported by the v2 runtime.

Robot geometry is packaged under `ur10e_curobo/robot_assets`. The model preserves
`gripper_tip`, the six joint names, collision spheres, buffers, ignored pairs,
and custom URDF. Startup still applies the selected camera calibration profile.
The vendor's generic UR10e model is not substituted for the harvesting gripper.

## API changes

| Application operation | v2 implementation |
|---|---|
| Cartesian planning | `MotionPlanner.plan_pose`, `GoalToolPose` |
| Joint planning | `MotionPlanner.plan_cspace` |
| Seeded IK | `InverseKinematics.solve_pose` |
| FK and collision spheres | isolated `Kinematics.compute_kinematics` |
| Feasibility | v2 auxiliary rollout on `RobotState` |
| Obstacle updates | `Scene`, `MotionPlanner.update_world` |

`planner_v2.py` normalizes the single-robot result shape and rejects failed,
nonfinite, or multiple-plan results. Dense trajectory timing stays at the
configured interpolation period (4 ms by default). Interpolation capacity covers
20 seconds, with a minimum of 1000 samples. CUDA graphs stay enabled at runtime.

The v1 no-finetune option has been removed. v2 manages B-spline refinement;
a failed joint plan retries with graph seeding. Existing ROS trajectory speed,
settling, stop, safe-zone, and gripper checks remain in the application layer.
No v1 solver fallback is used.

## Verification

```bash
source /opt/ros/humble/setup.bash
source ur_ws_new/install/setup.bash
v2_python="$HOME/.venvs/datepalm-curobo-v2/bin/python"
"$v2_python" -m unittest discover -s ur_ws_new/src/ur10e_curobo/tests -v
"$v2_python" ur_ws_new/src/ur10e_curobo/tools/check_curobo_v2.py
```

The GPU check does not create a ROS node or command hardware. It checks FK,
28 collision spheres, batched feasibility, out-of-limit rejection, seeded IK,
joint and Cartesian planning, endpoint accuracy, interpolation shape, and
obstacle insertion/removal with CUDA graphs enabled.

Hardware grasp success and field timing must be measured separately after
launching this branch. Passing offline planning checks is not a measured field
performance improvement.

Upstream references:
[public API and v2 notice](https://github.com/NVlabs/curobo),
[installation](https://nvlabs.github.io/curobo/latest/getting-started/installation.html).

## Local verification on 2026-09-29

On the Jetson Orin (Python 3.10, NVIDIA PyTorch 2.3 / CUDA 12.2):

- Pinned installer and ROS package build completed.
- 71 regression/contract tests passed in the v2 environment.
- The GPU integration check passed with CUDA graphs enabled and 4 ms interpolation.
- Across 12 joint configurations, v1/v2 maximum tool-position difference was
  4.18e-7 m and maximum collision-sphere difference was 4.77e-7 m.
- The control node completed warmup and initialization in a separate localhost
  ROS domain with fake hardware selected and the gripper disabled.

These are local compatibility checks, not a change to NVIDIA's documented
platform support or validation of real robot execution. The existing live
control process was not restarted during migration.

### Graph attempt policy

`PlanOptions.graph_attempt` is one-based: `1` permits graph seeding on the
first attempt, and `None` permits it on the final attempt. The adapter converts
to cuRobo v2's zero-based index; for eight attempts, `None` becomes `7`.
Out-of-range indices are clamped to the available attempts. This policy does
not use `None` to disable graph seeding. The general default is five attempts;
explicit production policies in `config.py` retain their own attempt limits.
