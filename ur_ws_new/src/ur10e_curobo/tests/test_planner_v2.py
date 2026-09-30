"""Application contracts for v2; run in the v2 environment with ROS sourced."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from curobo.types import JointState
from ur10e_curobo.planner_options import PlanOptions
from ur10e_curobo.planner_v2 import PlanResult, PlannerV2, load_robot_config


class V2ResultTests(unittest.TestCase):
    def result(self, q, success=True):
        native = SimpleNamespace(success=torch.tensor([success]),
                                 get_interpolated_plan=lambda: JointState.from_position(q))
        return PlanResult(native, .004)

    def test_native_batch_seed_dimensions_are_removed(self):
        q = torch.zeros((1, 1, 12, 6))
        r = self.result(q)
        self.assertEqual(r.get_interpolated_plan().position.shape, (12, 6))
        self.assertEqual(r.interpolation_dt, .004)

    def test_result_owns_trajectory_before_next_solver_call(self):
        q = torch.zeros((1, 1, 12, 6))
        r = self.result(q)
        q.fill_(99.)
        self.assertEqual(r.get_interpolated_plan().position.abs().max().item(), 0.)

    def test_empty_trajectory_is_rejected(self):
        with self.assertRaises(ValueError):
            self.result(torch.zeros((0, 6)))

    def test_multiple_plans_are_not_flattened_into_one_motion(self):
        with self.assertRaises(ValueError):
            self.result(torch.zeros((1, 2, 12, 6))).get_interpolated_plan()

    def test_failure_never_produces_executable_trajectory(self):
        for r in [PlanResult(None, .004), self.result(torch.zeros((12, 6)), False)]:
            self.assertFalse(r.success)
            with self.assertRaises(RuntimeError):
                r.get_interpolated_plan()

    def test_nonfinite_trajectory_is_rejected(self):
        with self.assertRaises(ValueError):
            self.result(torch.full((12, 6), float('nan'))).get_interpolated_plan()


class V2PolicyTests(unittest.TestCase):
    def test_disabled_graph_does_not_enable_it_on_a_retry(self):
        p = object.__new__(PlannerV2)
        p.planner = Mock()
        p.planner.plan_cspace.return_value = None
        p.interpolation_dt = .004
        start, goal = object(), object()
        p.plan_joint(start, goal, PlanOptions(max_attempts=8, graph_attempt=None))
        p.planner.plan_cspace.assert_called_once_with(
            goal, start, max_attempts=8, enable_graph_attempt=8)

    def test_custom_gripper_and_collision_model_are_preserved(self):
        k = load_robot_config()['robot_cfg']['kinematics']
        self.assertEqual(k['tool_frames'], ['gripper_tip'])
        self.assertTrue(Path(k['urdf_path']).is_file())
        self.assertEqual(sum(len(v) for v in k['collision_spheres'].values()), 28)
        self.assertEqual(k['collision_sphere_buffer'], .01)
        self.assertEqual(k['self_collision_buffer']['forearm_link'], .035)
        self.assertEqual(len(k['cspace']['default_joint_position']), 6)
        self.assertNotIn('retract_config', k['cspace'])
