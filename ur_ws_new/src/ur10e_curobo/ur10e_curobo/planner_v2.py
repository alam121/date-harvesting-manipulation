"""cuRobo v2 integration for the harvesting application's planning contracts.

No v1 implementation is loaded. This boundary normalizes tensor shapes and
keeps ROS trajectory construction independent of upstream solver internals.
"""
import copy
import math
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


class PlanResult:
    def __init__(self, result, dt):
        self.result = result
        self.success = bool(result is not None and torch.as_tensor(result.success).all().item())
        self.status = 'SUCCESS' if self.success else 'PLANNING_FAILED'
        self.interpolation_dt = float(dt)
        self.solve_time = float(getattr(result, 'solve_time', 0.0))
        self.total_time = float(getattr(result, 'total_time', 0.0))
        self._plan = self._freeze_plan() if self.success else None

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


class PlannerV2:
    def __init__(self, robot_config, world_config, interpolation_dt=.02,
                 num_trajopt_seeds=4, num_ik_seeds=32, use_cuda_graph=True):
        self.interpolation_dt = interpolation_dt
        self.world_model = Scene.create(copy.deepcopy(world_config))
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

    def plan_pose(self, start, goal, options=None):
        options = options or PlanOptions()
        result = self.planner.plan_pose(
            self._goal(goal), start, max_attempts=options.max_attempts,
            enable_graph_attempt=(options.max_attempts if options.graph_attempt is None
                                  else options.graph_attempt))
        return PlanResult(result, self.interpolation_dt)

    def plan_joint(self, start, goal, options=None):
        options = options or PlanOptions()
        result = self.planner.plan_cspace(
            goal, start, max_attempts=options.max_attempts,
            enable_graph_attempt=(options.max_attempts if options.graph_attempt is None
                                  else options.graph_attempt))
        return PlanResult(result, self.interpolation_dt)

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

    def reset(self, reset_seed=True):
        if reset_seed:
            self.planner.reset_seed()
