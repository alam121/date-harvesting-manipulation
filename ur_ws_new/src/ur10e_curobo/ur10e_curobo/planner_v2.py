"""cuRobo v2 integration for the harvesting application's planning contracts.

No v1 implementation is loaded. This boundary normalizes tensor shapes and
keeps ROS trajectory construction independent of upstream solver internals.
"""
import copy
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from curobo.config_io import load_yaml
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, RobotState

from .planner_options import PlanOptions


ASSET_ROOT = Path(__file__).resolve().parent / 'robot_assets'


def load_robot_config(name='ur10e.yml'):
    config = load_yaml(str(ASSET_ROOT / name))
    kin = config['robot_cfg']['kinematics']
    kin['urdf_path'] = str(ASSET_ROOT / kin['urdf_path'])
    kin['asset_root_path'] = str(ASSET_ROOT / 'ur_description')
    if isinstance(kin['collision_spheres'], str):
        kin['collision_spheres'] = load_yaml(str(ASSET_ROOT / kin['collision_spheres']))['collision_spheres']
    return config


class RobotKinematics:
    """Isolated v2 FK buffers with the application's flat batch output contract."""
    def __init__(self, config):
        self.model = Kinematics(copy.deepcopy(config))
        self.kinematics_config = self.model.kinematics_config
        self.tool_frame = self.model.tool_frames[0]

    def forward(self, q):
        state = self.model.compute_kinematics(
            JointState.from_position(q, joint_names=self.model.joint_names))
        pose = state.tool_poses.get_link_pose(self.tool_frame)
        return (pose.position.reshape(-1, 3), pose.quaternion.reshape(-1, 4),
                None, None, None, None,
                state.robot_spheres.reshape(q.shape[0], -1, 4))


def _as_float(value):
    """Return a plain float for a scalar/tensor diagnostic, else None."""
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value).reshape(-1)
        if tensor.numel() == 0:
            return None
        return float(tensor[0].item())
    except Exception:
        return None


class PlanResult:
    """Application view of one v2 planning result.

    v2 returns ``Optional[TrajOptSolverResult]`` (``None`` when planning fails)
    and reports *feasibility* separately from *pose convergence*. The v1
    diagnostics the rest of the application logs -- ``status`` and
    ``valid_query`` -- are reconstructed here so callers keep that vocabulary
    without reaching into upstream internals.
    """

    def __init__(self, result, dt, *, wall_time=0.0, graph_attempt=None,
                 graph_attempt_index=None, graph_seeding_available=None):
        self.result = result
        self.success = bool(
            result is not None and torch.as_tensor(result.success).all().item())
        self.interpolation_dt = float(dt)
        self.solve_time = float(getattr(result, 'solve_time', 0.0) or 0.0)
        self.total_time = float(getattr(result, 'total_time', 0.0) or 0.0)
        # Wall clock measured in this process around the planner call. Compare
        # against solve_time/total_time (GPU-reported, summed over attempts) to
        # see how much of a plan is contention and launch overhead rather than
        # solver work.
        self.wall_time = float(wall_time)
        # v2's `feasible` is "constraints (collision, joint limits) satisfied,
        # independent of pose convergence" -- the same question v1's valid_query
        # answered. A single robot needs every seed feasible.
        feasible = None if result is None else getattr(result, 'feasible', None)
        self.valid_query = (
            bool(torch.as_tensor(feasible).all().item())
            if feasible is not None else None)
        self.position_error = _as_float(getattr(result, 'position_error', None))
        self.rotation_error = _as_float(getattr(result, 'rotation_error', None))
        self.num_seeds = int(getattr(result, 'num_seeds', 0) or 0)
        self.batch_size = int(getattr(result, 'batch_size', 0) or 0)
        # Application policy value and the index v2 actually received, so a
        # misconfigured graph policy is visible in the log rather than silent.
        self.graph_attempt = graph_attempt
        self.graph_attempt_index = graph_attempt_index
        self.graph_seeding_available = graph_seeding_available
        self.status = self._status()
        self._plan = self._freeze_plan() if self.success else None

    def _status(self):
        if self.success:
            return 'SUCCESS'
        if self.result is None:
            return 'NO_RESULT'
        if self.valid_query is False:
            # Endpoints are invalid, so retrying cannot help -- the same
            # distinction v1's status made and the callers' diagnostic text
            # assumes.
            return 'COLLISION_OR_JOINT_LIMIT'
        return 'PLANNING_FAILED'

    def diagnostics(self):
        """Structured v2 fields for the durable motion log."""
        return {
            'status': self.status,
            'valid_query': self.valid_query,
            'v2_solve_ms': round(self.solve_time * 1000.0, 1),
            'v2_total_ms': round(self.total_time * 1000.0, 1),
            'v2_wall_ms': round(self.wall_time * 1000.0, 1),
            'v2_position_error_m': (None if self.position_error is None
                                    else round(self.position_error, 5)),
            'v2_rotation_error_rad': (None if self.rotation_error is None
                                      else round(self.rotation_error, 5)),
            'v2_seeds': self.num_seeds,
            'v2_batch': self.batch_size,
            'graph_attempt': self.graph_attempt,
            'graph_attempt_index': self.graph_attempt_index,
            'graph_seeding_available': self.graph_seeding_available,
        }

    def _freeze_plan(self):
        if not self.success:
            raise RuntimeError('Cannot execute an unsuccessful v2 plan')
        trajectory = self.result.get_interpolated_plan()
        q = trajectory.position
        # v2 retains a leading problem batch. A single robot accepts one plan.
        while q.ndim > 2 and q.shape[0] == 1:
            q = q[0]
        if q.ndim != 2 or q.shape[0] < 2 or not torch.isfinite(q).all():
            raise ValueError(f'Invalid v2 trajectory shape/data: {tuple(q.shape)}')
        # Upstream owns reusable interpolation buffers. Snapshot before another
        # planning call (including background drop-off planning) can overwrite them.
        return JointState.from_position(q.detach().clone(), joint_names=trajectory.joint_names)

    def get_interpolated_plan(self):
        if self._plan is None:
            raise RuntimeError('Cannot execute an unsuccessful v2 plan')
        return self._plan


def plan_diagnostics(result) -> dict:
    """Structured v2 diagnostics for the durable motion log.

    Tolerates a missing result so call sites can splat this unconditionally:
    a failed graph-seeding retry clears the result before logging.
    """
    if result is None:
        return {'status': 'NO_RESULT', 'valid_query': None}
    diagnostics = getattr(result, 'diagnostics', None)
    if not callable(diagnostics):
        return {'status': str(getattr(result, 'status', 'UNKNOWN')),
                'valid_query': getattr(result, 'valid_query', None)}
    return diagnostics()


def scene_replay_snapshot(world):
    """Detached primitive geometry; explicitly flag unsupported scene types."""
    result = {}
    for kind, dimensions in (("cuboid", "dims"), ("sphere", "radius")):
        result[kind] = {}
        for obstacle in getattr(world, kind, None) or []:
            value = getattr(obstacle, dimensions)
            result[kind][obstacle.name] = {
                "pose": [float(x) for x in obstacle.pose],
                dimensions: ([float(x) for x in value] if dimensions == "dims" else float(value)),
            }
    unsupported = [kind for kind in ("mesh", "capsule", "cylinder", "voxel")
                   if getattr(world, kind, None)]
    return {"world_config": result, "complete": not unsupported,
            "unsupported_types": unsupported}


def gpu_activity_snapshot():
    """Jetson load is system-wide, in tenths of a percent; no CUDA sync."""
    root = Path('/sys/devices/platform/bus@0/17000000.gpu')
    result = {"monotonic_s": time.monotonic()}
    for name, path, divisor in (
            ('load_percent', root / 'load', 10.),
            ('frequency_hz', root / 'devfreq/17000000.gpu/cur_freq', 1.)):
        try:
            result[name] = float(path.read_text().strip()) / divisor
        except (OSError, ValueError):
            result[name] = None
    return result


class PlannerV2:
    def __init__(self, robot_config, world_config, interpolation_dt=.02,
                 num_trajopt_seeds=4, num_ik_seeds=32, use_cuda_graph=True):
        self.interpolation_dt = interpolation_dt
        self.world_model = Scene.create(copy.deepcopy(world_config))
        self._replay_world = scene_replay_snapshot(self.world_model)
        self._world_revision = 0
        self._world_updated_at = time.monotonic()
        self.config = MotionPlannerCfg.create(
            robot=copy.deepcopy(robot_config), scene_model=self.world_model,
            interpolation_dt=interpolation_dt,
            interpolation_buffer_size=max(1000, math.ceil(20.0 / interpolation_dt)),
            collision_cache={'cuboid': 64, 'mesh': 10},
            num_trajopt_seeds=num_trajopt_seeds, num_ik_seeds=num_ik_seeds,
            use_cuda_graph=use_cuda_graph)
        self.planner = MotionPlanner(self.config)
        self.fk_config = KinematicsCfg.from_data_dict(copy.deepcopy(robot_config))
        self.tool_frame = self.planner.tool_frames[0]
        self.num_ik_seeds = self.planner.ik_solver.config.num_seeds

    def warmup(self):
        self.planner.warmup()

    def _goal(self, pose):
        return GoalToolPose.from_poses({self.tool_frame: pose})

    def _graph_planner_available(self):
        return getattr(self.planner, 'graph_planner', None) is not None

    def _run_plan(self, call, start, goal_arg, options):
        """Run one v2 planner call and attach the application's diagnostics.

        `label`-free by design: the callers already name the phase, and the
        diagnostics travel back on the PlanResult into the durable motion log.
        """
        options = options or PlanOptions()
        graph_index = options.enable_graph_attempt()
        world = copy.deepcopy(self._replay_world)
        revision = self._world_revision
        world_age = time.monotonic() - self._world_updated_at
        gpu_before = gpu_activity_snapshot()
        cpu_t0 = time.process_time()
        t0 = time.perf_counter()
        result = call(goal_arg, start, max_attempts=options.max_attempts,
                      enable_graph_attempt=graph_index)
        wall_time = time.perf_counter() - t0
        cpu_time = time.process_time() - cpu_t0
        wrapped = PlanResult(
            result, self.interpolation_dt,
            wall_time=wall_time,
            graph_attempt=options.graph_attempt,
            graph_attempt_index=graph_index,
            graph_seeding_available=self._graph_planner_available())
        wrapped.runtime_diagnostics = {
            "scene": world, "world_revision": revision,
            "world_changed_during_plan": revision != self._world_revision,
            "world_age_s": world_age, "planner_call_s": wall_time,
            "process_cpu_s": cpu_time,
            "gpu_before": gpu_before, "gpu_after": gpu_activity_snapshot(),
            "gpu_scope": "system-wide endpoint samples, not per-process utilization",
        }
        return wrapped

    def plan_pose(self, start, goal, options=None):
        return self._run_plan(
            self.planner.plan_pose, start, self._goal(goal), options)

    def plan_joint(self, start, goal, options=None):
        return self._run_plan(self.planner.plan_cspace, start, goal, options)


    def solve_ik(self, pose, seed_config=None, retract_config=None, return_seeds=1):
        current = (JointState.from_position(retract_config.reshape(1, -1),
                                           joint_names=self.planner.joint_names)
                   if retract_config is not None else None)
        return self.planner.ik_solver.solve_pose(
            self._goal(pose), current_state=current, seed_config=seed_config,
            return_seeds=min(return_seeds, self.num_ik_seeds))

    def check_constraints(self, state):
        # A horizon of one checks geometry/joint bounds without inventing motion
        # derivatives between samples. Keep this separate from trajectory rollout.
        q = state.position.reshape(-1, 1, len(self.planner.joint_names))
        js = JointState.from_position(q, joint_names=self.planner.joint_names)
        robot_state = RobotState(joint_state=js,
                                 cuda_robot_model_state=self.planner.compute_kinematics(js))
        metrics = self.planner.ik_solver.auxiliary_rollout.compute_metrics_from_state(robot_state)
        feasible = metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False, sum_horizon=True)
        return SimpleNamespace(feasible=feasible)

    def check_start_state(self, state):
        valid = bool(torch.as_tensor(self.check_constraints(state).feasible).all().item())
        return valid, 'VALID' if valid else 'COLLISION_OR_JOINT_LIMIT'

    def update_world(self, world):
        self.planner.update_world(world)
        self.world_model = world
        self._replay_world = scene_replay_snapshot(world)
        self._world_revision += 1
        self._world_updated_at = time.monotonic()

    def reset(self, reset_seed=True):
        if reset_seed:
            self.planner.reset_seed()
