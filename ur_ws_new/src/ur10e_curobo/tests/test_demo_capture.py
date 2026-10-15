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
    def test_tool_xy_rotation_keeps_tool_depth_zero(self):
        half = 2 ** -0.5
        base = demo._quat_rotate([half, 0.0, 0.0, half], [.005, -.010, 0.0])
        tool = demo._quat_rotate_inverse([half, 0.0, 0.0, half], base)
        self.assertTrue(np.allclose(tool, [.005, -.010, 0.0], atol=1e-12))

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
                current_joint_positions=[0.0] * 6, joint_order=[str(i) for i in range(6)], stop_requested=False,
                _demo_planned_final_pose=[.010, 0., 0., 1., 0., 0., 0.],
                _demo_depth_trace={"base_forward_axis": "+X"},
                pose=[0., 0., 0., 1., 0., 0., 0.],
                get_logger=lambda: Mock())
            node.get_end_effector_pose = lambda: node.pose
            calls = []
            moves = []

            def sleep(seconds):
                clock.now += seconds
                if node._demo_full_frame is None:
                    if fresh and not calls and full_fresh:
                        node._latest_grasp_crop_t = clock.now
                        node._latest_grasp_crop[:] = 30
                        node._demo_full_frame = (np.full((4, 6, 3), 42, dtype=np.uint8), clock.now, clock.now)
                elif getattr(node, '_demo_target_selection', None) is None:
                    if clicked:
                        node._demo_target_selection = {"pixel_xy": [2, 2], "source": "operator_click"}
                elif not moves and getattr(
                        node, '_demo_xy_pending_step_tool_m', None) is None:
                    node._demo_xy_pending_step_tool_m = [.005, 0.0]
                    node.stop_requested = stopped
                else:
                    node._latest_grasp_crop[:] = 200
                    node._demo_full_frame = (
                        np.full((4, 6, 3), 200, dtype=np.uint8),
                        clock.now, clock.now)
                    node._demo_confirm = verdict
                calls.append(seconds)

            def move(target):
                moves.append(list(target))
                node.pose = list(target)
                return True

            images = {}
            def write(image, path, name):
                images[name] = image.copy()
                return write_ok

            with patch.object(demo.time, 'time', lambda: clock.now), \
                 patch.object(demo.time, 'monotonic', lambda: clock.now), \
                 patch.object(demo.time, 'sleep', sleep), \
                 patch.object(demo, '_write_crop', write):
                result = demo.capture_correction(node, move)
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

    def test_operator_click_confirms_one_detected_date(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        meta = {'source_size': [200, 100], 'candidates': [
            {'fruit_id': 4, 'bbox_xyxy': [50, 25, 100, 50],
             'base_xyz': [0, 0, 0]}]}
        marked, label = demo.target_preview(image, meta, [0, 0, 0])
        demo._apply_target_selection(
            marked, label, {'pixel_xy': [75, 35], 'source': 'operator_click'})
        self.assertEqual(label['selected_fruit_id'], 4)
        self.assertEqual(label['clicked_candidate_ids'], [4])
        self.assertTrue(label['identity_verified'])
        self.assertTrue(label['operator_point_confirmed'])

    def test_operator_crop_stays_fixed_size_at_image_edge(self):
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        crop, metadata = demo._operator_target_crop(image, [5, 295])
        self.assertEqual(crop.shape, (256, 256, 3))
        self.assertEqual(metadata['crop_xyxy'], [0, 44, 256, 300])
        self.assertEqual(metadata['operator_pixel_crop'], [5.0, 251.0])

    def test_xy_alignment_and_corrected_final_have_distinct_labels(self):
        data, _ = self.run_capture()
        self.assertEqual(data['observation_stage'], 'APPROACH')
        self.assertEqual(data['capture_mode'], 'approach_xy_alignment')
        self.assertEqual(data['desired_motion_base_m'], [.005, 0., 0.])
        self.assertEqual(data['corrected_final_pose'][:3], [.015, 0., 0.])
        self.assertEqual(data['before']['planned_final_pose'][:3], [.010, 0., 0.])

    def test_depth_trace_records_planar_correction_without_depth_action(self):
        data, _ = self.run_capture()
        trace = data['before']['depth_trace']
        self.assertEqual(trace['approach_xy_correction_tool_m'], [.005, 0., 0.])
        self.assertEqual(trace['corrected_final_tcp_base_xyz_m'], [.015, 0., 0.])
        self.assertTrue(data['depth_axis_locked'])

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
        self.assertEqual(data['schema_version'], 6)
        self.assertEqual(data['correction_base_m'], [.005, 0., 0.])
        self.assertEqual(data['correction_tool_m'], [.005, 0., 0.])
        self.assertEqual(data['correction_mm'], 5.)
        self.assertTrue(np.all(images['crop_before.jpg'] == 42))
        self.assertEqual(data['before']['crop_source'], 'operator_click')
        self.assertEqual(data['before']['crop_meta']['operator_pixel_full'], [2.0, 2.0])
        self.assertEqual(data['alignment_steps'], 1)
        self.assertEqual(data['alignment_path'][1]['command_tool_m'], [.005, 0., 0.])
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


class ClickConfirmedOutcomeTests(unittest.TestCase):
    def run_capture(self, clicked=True):
        with TemporaryDirectory() as root:
            clock = SimpleNamespace(now=10.0)
            node = SimpleNamespace(
                cfg=SimpleNamespace(planner=SimpleNamespace(
                    demo_capture_enabled=True, demo_capture_dir=root,
                    demo_capture_timeout_s=0.12)),
                _latest_grasp_crop=None,
                _latest_grasp_crop_t=0.0, _latest_grasp_crop_meta=[1],
                _demo_full_frame=None, _demo_publish_preview=Mock(),
                _demo_attempt_goal_xyz=[.1, 0., 0.],
                _demo_planned_final_pose=[.11, 0., 0., 1., 0., 0., 0.],
                _demo_depth_trace={"base_forward_axis": "+X"},
                latest_goal_pose=[.1, 0., 0.], latest_fruit_radius=.01,
                current_joint_positions=[0.0] * 6, joint_order=[str(i) for i in range(6)], stop_requested=False,
                get_end_effector_pose=lambda: [0., 0., 0., 1., 0., 0., 0.],
                get_logger=lambda: Mock())
            calls = []

            def sleep(seconds):
                clock.now += seconds
                if node._demo_full_frame is None:
                    node._latest_grasp_crop_t = clock.now
                    node._demo_full_frame = (
                        np.full((6, 8, 3), 42, dtype=np.uint8),
                        clock.now, clock.now)
                elif clicked:
                    node._demo_target_selection = {
                        "pixel_xy": [3, 2], "source": "operator_click"}
                calls.append(seconds)

            images = {}

            def write(image, path, name):
                images[name] = image.copy()
                return True

            with patch.object(demo.time, 'time', lambda: clock.now), \
                 patch.object(demo.time, 'monotonic', lambda: clock.now), \
                 patch.object(demo.time, 'sleep', sleep), \
                 patch.object(demo, '_write_crop', write):
                result = demo.capture_observation(node)
                if result:
                    node._demo_pending_record = result
                    self.assertIsNotNone(node._demo_sequence)
                    self.assertFalse(node._demo_sequence.closed)
                    demo.complete_capture(node)
                    sequence = json.loads((Path(result) / 'sequence/sequence.json').read_text())
                    self.assertEqual(sequence['trajectory_source'], 'automatic_execution')
                    self.assertEqual(sequence['label_source'], 'operator')
                    self.assertIsNone(sequence['operator_label'])
                    self.assertNotIn('outcome', sequence)
                    self.assertTrue(sequence['complete'])
                    self.assertIsNone(node._demo_sequence)
            data = json.loads((Path(result) / 'sample.json').read_text()) if result else None
            return data, images, node

    def test_click_is_required_and_saved_before_automatic_final(self):
        data, images, node = self.run_capture(clicked=True)
        self.assertEqual(data['schema_version'], 5)
        self.assertEqual(data['capture_mode'], 'click_confirmed_outcome')
        self.assertEqual(data['label_source'], 'operator')
        self.assertNotIn('auto_outcome', data)
        self.assertTrue(data['target_confirmation_required'])
        label = data['before']['target_label']
        self.assertEqual(label['operator_selection']['pixel_xy'], [3, 2])
        self.assertTrue(label['operator_point_confirmed'])
        self.assertTrue(np.all(images['crop_before.jpg'] == 42))
        self.assertEqual(data['before']['crop_meta']['source'], 'operator_click')
        self.assertIn('full_before_marked.jpg', images)
        self.assertFalse(node._demo_capture_active)
        self.assertFalse(node._demo_capture_ready)

    def test_no_click_discards_instead_of_writing_unverified_sample(self):
        data, images, node = self.run_capture(clicked=False)
        self.assertIsNone(data)
        self.assertEqual(images, {})
        self.assertFalse(node._demo_capture_active)


class DemoRegionScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).resolve().parents[1]
                  / 'ur10e_curobo/vision/scoring.py')
        tree = ast.parse(source.read_text())
        function = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == 'compute_demo_region_score')
        env = {'Dict': dict, 'Any': object}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), env)
        cls.score = staticmethod(env['compute_demo_region_score'])

        node_source = source.with_name('node.py')
        node_tree = ast.parse(node_source.read_text())
        selector = next(n for n in ast.walk(node_tree)
                        if isinstance(n, ast.FunctionDef)
                        and n.name == '_select_best_fruit')
        selector.args.args[0].annotation = None
        selector.returns = None
        select_env = {
            'List': list, 'Dict': dict, 'Any': object, 'Optional': object,
            'np': np, 'math': __import__('math'),
            'BEST_REUSE_THRESH': 0.08, 'SWITCH_THRESHOLD': 0.02,
            'TARGET_LOCK_RADIUS': 0.08, 'DEMO_REGION_WEIGHT': 2.0,
            'compute_demo_region_score': cls.score,
            'compute_fruit_score': lambda target, _previous: {
                'total_score': target['base_score'], 'components': {}},
        }
        exec(compile(ast.Module(body=[selector], type_ignores=[]),
                     str(node_source), 'exec'), select_env)
        cls.select = staticmethod(select_env['_select_best_fruit'])

    def test_image_is_split_into_left_center_right_thirds(self):
        targets = [
            {'bb': [0, 0, 20, 20], 'img_width': 120},
            {'bb': [50, 0, 70, 20], 'img_width': 120},
            {'bb': [100, 0, 120, 20], 'img_width': 120},
        ]
        for target, expected in zip(targets, ('left', 'center', 'right')):
            with self.subTest(expected=expected):
                self.assertEqual(self.score(target, expected), 1.0)
                self.assertEqual(self.score(target, 'left' if expected != 'left' else 'right'), 0.0)

    def test_missing_bbox_gets_no_region_bonus(self):
        self.assertEqual(self.score({'img_width': 120}, 'center'), 0.0)

    def test_preferred_third_dominates_only_in_demo_mode(self):
        targets = [
            {'bb': [0, 0, 20, 20], 'img_width': 120, 'base_score': 0.1},
            {'bb': [100, 0, 120, 20], 'img_width': 120, 'base_score': 1.3},
        ]
        vision = SimpleNamespace(
            excluded_positions=[], _prev_excluded_count=0,
            best_target_prev=None, _demo_region_enabled=True,
            _demo_target_region='left', target_lock_active=False,
            target_lock_position=None)
        self.assertEqual(self.select(vision, targets), 0)

        vision.best_target_prev = None
        vision._demo_region_enabled = False
        self.assertEqual(self.select(vision, targets), 1)


class DemoCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'ur10e_curobo/node.py'
        tree = ast.parse(source.read_text())
        callback = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == '_ui_command_cb')
        callback.args.args[1].annotation = None
        cls.gripper = SimpleNamespace(control_gripper=Mock())
        env = {
            'gripper_mod': cls.gripper,
            'demo_capture': demo,
            'String': SimpleNamespace,
        }
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), 'exec'), env)
        cls.callback = staticmethod(env['_ui_command_cb'])

    def setUp(self):
        self.gripper.control_gripper.reset_mock()
        self.node = SimpleNamespace(
            cfg=SimpleNamespace(planner=SimpleNamespace(
                concise_console_logs=True, demo_capture_enabled=True,
                demo_capture_jog_enabled=True, demo_target_region='default',
                demo_xy_step_m=.005)),
            _demo_capture_active=True, _demo_capture_ready=True, _demo_confirm=None,
            _demo_target_selection={"pixel_xy": [2, 2]},
            _demo_xy_busy=False, _demo_xy_pending_step_tool_m=None,
            robot_running=True, get_logger=lambda: Mock(),
            _session_record_gripper_action=Mock(), _refresh_camera_pub=Mock(),
            _motion_lock=Mock())
        self.node._motion_lock.locked.return_value = False

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

    def test_xy_step_queues_exact_five_mm_tool_action(self):
        self.command('demo_xy_step x -1')
        self.assertEqual(self.node._demo_xy_pending_step_tool_m, [-.005, 0.0])

    def test_xy_step_rejects_depth_axis(self):
        self.command('demo_xy_step z 1')
        self.assertIsNone(self.node._demo_xy_pending_step_tool_m)

    def test_xy_confirm_waits_for_pending_step(self):
        self.node._demo_xy_pending_step_tool_m = [0.0, .005]
        self.command('demo_xy_confirm')
        self.assertIsNone(self.node._demo_confirm)

    def test_xy_confirm_continues_without_closing_gripper(self):
        self.command('demo_xy_confirm')
        self.assertTrue(self.node._demo_confirm)
        self.gripper.control_gripper.assert_not_called()

    def test_uncheck_discards_hold(self):
        self.command('set_demo_capture false')
        self.assertFalse(self.node.cfg.planner.demo_capture_enabled)
        self.assertFalse(self.node.cfg.planner.demo_capture_jog_enabled)
        self.assertIs(self.node._demo_confirm, False)
        self.assertEqual(
            self.node._refresh_camera_pub.publish.call_args.args[0].data,
            'demo_region default enabled=0')

    def test_xy_mode_also_enables_capture(self):
        self.node._demo_capture_active = False
        self.node.cfg.planner.demo_capture_enabled = False
        self.node.cfg.planner.demo_capture_jog_enabled = False
        with patch.object(demo, 'refresh_label_queue'):
            self.command('set_demo_jog true')
        self.assertTrue(self.node.cfg.planner.demo_capture_jog_enabled)
        self.assertTrue(self.node.cfg.planner.demo_capture_enabled)
        self.assertEqual(
            self.node._refresh_camera_pub.publish.call_args.args[0].data,
            'demo_region default enabled=1')

    def test_demo_region_is_forwarded_to_vision(self):
        self.node.cfg.planner.demo_capture_enabled = True
        self.command('set_demo_region left')
        self.assertEqual(self.node.cfg.planner.demo_target_region, 'left')
        self.assertEqual(
            self.node._refresh_camera_pub.publish.call_args.args[0].data,
            'demo_region left enabled=1')

    def test_invalid_demo_region_is_rejected(self):
        self.node.cfg.planner.demo_target_region = 'center'
        self.command('set_demo_region above')
        self.assertEqual(self.node.cfg.planner.demo_target_region, 'center')
        self.node._refresh_camera_pub.publish.assert_not_called()

    def test_rviz_label_command_writes_the_queued_record(self):
        with TemporaryDirectory() as root:
            out = Path(root) / 'record'
            out.mkdir()
            sample = out / 'sample.json'
            sample.write_text(json.dumps({'operator_label': None}))
            self.node.cfg.planner.demo_capture_dir = root
            self.node._demo_label_queue = []
            demo.refresh_label_queue(self.node)
            self.command('label_demo good')
            self.assertEqual(json.loads(sample.read_text())['operator_label'], 'good')
            self.assertEqual(self.node._demo_label_queue, [])


class CompleteCaptureTests(unittest.TestCase):
    """The post-grasp hook finalizes records without detector labels."""

    def setUp(self):
        self.logger = Mock()
        self.node = SimpleNamespace(get_logger=lambda: self.logger)

    def write(self, root, sample):
        out = Path(root) / '20261004_120000_000'
        out.mkdir()
        (out / 'sample.json').write_text(json.dumps(sample))
        return str(out)

    def test_record_is_queued_without_a_detector_label(self):
        with TemporaryDirectory() as root:
            out = self.write(root, {'operator_label': None})
            self.node._demo_pending_record = out
            demo.complete_capture(self.node)
            saved = json.loads((Path(out) / 'sample.json').read_text())
            self.assertNotIn('auto_outcome', saved)
            self.assertIsNone(saved['operator_label'])
            self.assertEqual(self.node._demo_label_queue, [out])

    def test_the_record_is_cleared_so_the_next_cycle_cannot_inherit_it(self):
        with TemporaryDirectory() as root:
            out = self.write(root, {})
            self.node._demo_pending_record = out
            demo.complete_capture(self.node)
            self.assertIsNone(self.node._demo_pending_record)
            demo.complete_capture(self.node)
            saved = json.loads((Path(out) / 'sample.json').read_text())
            self.assertNotIn('auto_outcome', saved)

    def test_a_normal_harvest_with_no_record_is_silent(self):
        self.node._demo_pending_record = None
        demo.complete_capture(self.node)
        self.logger.warn.assert_not_called()
        self.logger.info.assert_not_called()

    def test_an_unwritable_record_warns_instead_of_raising(self):
        self.node._demo_pending_record = '/nonexistent/demo/record'
        demo.complete_capture(self.node)
        self.logger.warn.assert_called_once()


class OperatorLabelQueueTests(unittest.TestCase):
    def make_node(self, root):
        return SimpleNamespace(
            cfg=SimpleNamespace(planner=SimpleNamespace(demo_capture_dir=root)),
            _demo_label_queue=[], get_logger=lambda: Mock())

    def write(self, root, name, label=None):
        out = Path(root) / name
        out.mkdir()
        (out / 'sample.json').write_text(json.dumps({'operator_label': label}))
        (out / 'full_before_marked.jpg').write_bytes(b'image')
        return str(out)

    def test_existing_unlabelled_records_are_recovered_oldest_first(self):
        with TemporaryDirectory() as root:
            first = self.write(root, '20261004_120000_000')
            self.write(root, '20261004_120100_000', 'good')
            second = self.write(root, '20261004_120200_000')
            node = self.make_node(root)
            self.assertEqual(demo.refresh_label_queue(node), [first, second])
            status = demo.label_status(node)
            self.assertEqual(status['pending'], 2)
            self.assertEqual(status['record'], Path(first).name)
            self.assertTrue(status['image'].endswith('full_before_marked.jpg'))

    def test_rviz_label_writes_exact_record_and_advances_queue(self):
        with TemporaryDirectory() as root:
            first = self.write(root, 'a')
            second = self.write(root, 'b')
            node = self.make_node(root)
            demo.refresh_label_queue(node)
            self.assertTrue(demo.apply_operator_label(node, 'edge'))
            saved = json.loads((Path(first) / 'sample.json').read_text())
            self.assertEqual(saved['operator_label'], 'edge')
            self.assertIsNone(json.loads(
                (Path(second) / 'sample.json').read_text())['operator_label'])
            self.assertEqual(demo.label_status(node)['record'], 'b')

    def test_stale_record_id_does_not_label_next_demo(self):
        with TemporaryDirectory() as root:
            self.write(root, 'a')
            second = self.write(root, 'b')
            node = self.make_node(root)
            demo.refresh_label_queue(node)
            self.assertTrue(demo.apply_operator_label(node, 'good', 'a'))
            self.assertFalse(demo.apply_operator_label(node, 'good', 'a'))
            self.assertEqual(node._demo_label_queue, [second])

    def test_invalid_label_does_not_modify_or_advance(self):
        with TemporaryDirectory() as root:
            first = self.write(root, 'a')
            node = self.make_node(root)
            demo.refresh_label_queue(node)
            self.assertFalse(demo.apply_operator_label(node, 'maybe'))
            self.assertEqual(node._demo_label_queue, [first])
            self.assertIsNone(json.loads(
                (Path(first) / 'sample.json').read_text())['operator_label'])


if __name__ == '__main__':
    unittest.main()
