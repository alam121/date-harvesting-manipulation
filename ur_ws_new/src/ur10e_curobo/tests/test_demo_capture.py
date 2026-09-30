"""Hardware-free tests for the FINAL image/correction capture lifecycle."""
import ast
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    'demo_capture', Path(__file__).resolve().parents[1] / 'ur10e_curobo/demo_capture.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class CaptureTests(unittest.TestCase):
    def run_capture(self, verdict=True, fresh=True, stopped=False, write_ok=True, full_fresh=True, clicked=True):
        with TemporaryDirectory() as root:
            clock = SimpleNamespace(now=10.0)
            node = SimpleNamespace(
                cfg=SimpleNamespace(planner=SimpleNamespace(
                    demo_capture_enabled=True, demo_capture_dir=root,
                    demo_capture_timeout_s=0.3, demo_capture_sample_hz=10)),
                _latest_grasp_crop=np.zeros((3, 3, 3), dtype=np.uint8),
                _demo_full_frame=None, _demo_publish_preview=Mock(),
                _latest_grasp_crop_t=0, _latest_grasp_crop_meta=[1],
                current_joint_positions=[0.0] * 6, stop_requested=False,
                _demo_planned_final_pose=[.010, 0., 0., 1., 0., 0., 0.],
                pose=[0., 0., 0., 1., 0., 0., 0.],
                get_logger=lambda: Mock())
            node.get_end_effector_pose = lambda: node.pose
            calls = []

            def sleep(seconds):
                clock.now += seconds
                if fresh and not calls:
                    node._latest_grasp_crop_t = clock.now
                    node._latest_grasp_crop[:] = 30
                    if full_fresh:
                        node._demo_full_frame = (np.full((4, 6, 3), 42, dtype=np.uint8), clock.now, clock.now)
                else:
                    node.pose = [.012, 0., 0., 1., 0., 0., 0.]
                    node._latest_grasp_crop[:] = 200
                    if clicked:
                        node._demo_target_selection = {"pixel_xy": [2, 2], "source": "operator_click"}
                    node._demo_confirm = verdict
                    node.stop_requested = stopped
                calls.append(seconds)

            images = {}
            def write(image, path, name):
                images[name] = image.copy()
                return write_ok

            with patch.object(demo.time, 'time', lambda: clock.now), \
                 patch.object(demo.time, 'monotonic', lambda: clock.now), \
                 patch.object(demo.time, 'sleep', sleep), \
                 patch.object(demo, '_write_crop', write):
                result = demo.capture_correction(node)
            self.assertFalse(node._demo_capture_active)
            data = json.loads((Path(result) / 'sample.json').read_text()) if result else None
            if not result:
                self.assertEqual(list(Path(root).rglob('sample.json')), [])
            return data, images

    def test_target_labels_scale_and_reject_ambiguous_matches(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        meta = {'source_size': [400, 200], 'candidates': [
            {'fruit_id': 4, 'bbox_xyxy': [100, 50, 200, 100], 'base_xyz': [0, 0, 0]}]}
        _, label = demo.target_preview(image, meta, [0, 0, 0])
        self.assertEqual(label['suggested_fruit_id'], 4)
        self.assertEqual(label['candidates'][0]['saved_bbox_xyxy'], [50, 25, 100, 50])
        self.assertFalse(label['identity_verified'])
        meta['candidates'].append({'fruit_id': 5, 'bbox_xyxy': [0, 0, 20, 20], 'base_xyz': [.01, 0, 0]})
        _, label = demo.target_preview(image, meta, [0, 0, 0])
        self.assertIsNone(label['suggested_fruit_id'])

    def test_approach_motion_and_final_residual_have_distinct_labels(self):
        data, _ = self.run_capture()
        self.assertEqual(data['observation_stage'], 'APPROACH')
        self.assertEqual(data['desired_motion_base_m'], [.012, 0., 0.])
        self.assertAlmostEqual(data['residual_from_planned_final_base_m'][0], .002)
        self.assertEqual(data['before']['planned_final_pose'][:3], [.010, 0., 0.])

    def test_close_without_click_does_not_save(self):
        data, images = self.run_capture(clicked=False)
        self.assertIsNone(data)
        self.assertEqual(images, {})

    def test_operator_point_saved_without_detector_match(self):
        data, _ = self.run_capture()
        label = data['before']['target_label']
        self.assertEqual(label['operator_selection']['pixel_xy'], [2, 2])
        self.assertTrue(label['operator_point_confirmed'])
        self.assertIsNone(label['selected_fruit_id'])
        self.assertFalse(label['identity_verified'])

    def test_missing_full_frame_stops_capture(self):
        data, images = self.run_capture(full_fresh=False)
        self.assertIsNone(data)
        self.assertEqual(images, {})

    def test_before_image_is_frozen_and_delta_is_metres(self):
        data, images = self.run_capture()
        self.assertEqual(data['correction_base_m'], [.012, 0., 0.])
        self.assertEqual(data['correction_tool_m'], [.012, 0., 0.])
        self.assertEqual(data['correction_mm'], 12.)
        self.assertTrue(np.all(images['crop_before.jpg'] == 30))
        self.assertTrue(np.all(images['crop_after.jpg'] == 200))
        self.assertGreater(data['jog_samples'], 0)
        self.assertTrue(np.all(images["full_before.jpg"] == 42))

    def test_discard(self):
        self.assertIsNone(self.run_capture(verdict=False)[0])

    def test_timeout(self):
        self.assertIsNone(self.run_capture(verdict=None)[0])

    def test_stop_takes_precedence_over_confirm(self):
        self.assertIsNone(self.run_capture(stopped=True)[0])

    def test_stale_image_rejected(self):
        self.assertIsNone(self.run_capture(fresh=False)[0])

    def test_image_write_failure_invalidates_sample(self):
        self.assertIsNone(self.run_capture(write_ok=False)[0])


class DemoCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'ur10e_curobo/node.py'
        tree = ast.parse(source.read_text())
        callback = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == '_ui_command_cb')
        callback.args.args[1].annotation = None
        cls.gripper = SimpleNamespace(control_gripper=Mock())
        env = {'gripper_mod': cls.gripper}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), 'exec'), env)
        cls.callback = staticmethod(env['_ui_command_cb'])

    def setUp(self):
        self.gripper.control_gripper.reset_mock()
        self.node = SimpleNamespace(
            cfg=SimpleNamespace(planner=SimpleNamespace(concise_console_logs=True)),
            _demo_capture_active=True, _demo_capture_ready=True, _demo_confirm=None,
            _demo_target_selection={"pixel_xy": [2, 2]},
            robot_running=True, get_logger=lambda: Mock(),
            _session_record_gripper_action=Mock())

    def command(self, text):
        self.callback(self.node, SimpleNamespace(data=text))

    def test_close_accepts_without_direct_actuation(self):
        self.command('close')
        self.assertTrue(self.node._demo_confirm)
        self.gripper.control_gripper.assert_not_called()

    def test_close_requires_robot_program(self):
        self.node.robot_running = False
        self.command('close')
        self.assertIsNone(self.node._demo_confirm)
        self.gripper.control_gripper.assert_not_called()

    def test_close_waits_for_image(self):
        self.node._demo_capture_ready = False
        self.command('close')
        self.assertIsNone(self.node._demo_confirm)
        self.gripper.control_gripper.assert_not_called()

    def test_normal_close_unchanged(self):
        self.node._demo_capture_active = False
        self.command('close')
        self.gripper.control_gripper.assert_called_once_with(self.node, 'CLOSE')

    def test_uncheck_discards_hold(self):
        self.command('set_demo_capture false')
        self.assertFalse(self.node.cfg.planner.demo_capture_enabled)
        self.assertIs(self.node._demo_confirm, False)


if __name__ == '__main__':
    unittest.main()
