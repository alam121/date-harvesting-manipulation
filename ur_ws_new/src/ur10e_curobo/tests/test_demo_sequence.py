import unittest
from ur10e_curobo.demo_sequence import pair_frames

class SequenceTimingTests(unittest.TestCase):
    def test_nearest_source_state_not_arrival_order(self):
        rows = pair_frames([{'stamp_ns': 100_000_000}], [
            {'stamp_ns': 130_000_000, 'joints': [2]},
            {'stamp_ns': 90_000_000, 'joints': [1]}])
        self.assertEqual(rows[0]['joints'], [1.25])
        self.assertEqual(rows[0]['skew_ns'], 0)

    def test_stale_and_zero_stamps_never_paired(self):
        rows = pair_frames([{'stamp_ns': 0}, {'stamp_ns': 100_000_000}],
                           [{'stamp_ns': 200_000_000, 'joints': [1]}])
        self.assertTrue(all('joints' not in r for r in rows))

    def test_missing_states_retains_images(self):
        self.assertEqual(pair_frames([{'stamp_ns': 1, 'image':'a.jpg'}], []),
                         [{'stamp_ns': 1, 'image':'a.jpg'}])

    def test_disk_sequence_and_cleanup(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace as NS
        from ur10e_curobo.demo_sequence import DemoSequence
        recorder = DemoSequence(['b', 'a'])
        header = NS(stamp=NS(sec=1, nanosec=0))
        recorder.state(NS(header=header, name=['a','b'], position=[1,2]), 10)
        recorder.frame(NS(header=header, data=b'jpeg'), 20)
        with TemporaryDirectory() as destination:
            quality = recorder.save(destination)
            recorder.close()
            data = json.loads((Path(destination)/'sequence/sequence.json').read_text())
            self.assertEqual(data['frames'][0]['joints'], [2,1])
            self.assertEqual((Path(destination)/'sequence/000000.jpg').read_bytes(), b'jpeg')
            self.assertFalse(quality['timing_ok'])
            self.assertFalse(quality['training_ready'])

    def test_complete_episode_exports_fixed_rate_actions_and_grade(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace as NS
        from ur10e_curobo.demo_sequence import DemoSequence, update_training_readiness
        recorder = DemoSequence(['arm'], ['finger'])
        recorder.set_target('../full_before.jpg',[10,10],[100,100],
            {'selected_fruit_id':1,'identity_verified':True,
             'candidates':[{'fruit_id':1,'base_xyz':[0,1,2]}]})
        with TemporaryDirectory() as destination:
            recorder.stage(destination)
            for i in range(31):
                ns=10**9+i*100_000_000
                msg=NS(header=NS(stamp=NS(sec=ns//10**9,nanosec=ns%10**9)),
                       name=['arm','finger'],position=[i*.01,.5],effort=[])
                recorder.state(msg,ns);recorder.gripper_state(msg,ns)
                recorder.frame(NS(header=msg.header,data=b'jpeg'),ns)
            recorder.command('gripper.close_request',1,ns,source_stamp_ns=ns)
            q=recorder.finalize(complete=True)
            self.assertTrue(q['data_ready'])
            self.assertTrue(update_training_readiness(destination,'good'))
            data=json.loads((Path(destination)/'sequence/sequence.json').read_text())
            self.assertEqual(len(data['training_frames']),16)
            self.assertEqual(data['training_frames'][-1]['action'][-1],1)
            self.assertEqual(data['training_frames'][0]['action_dt_s'],.2)
            self.assertEqual(data['label_source'], 'operator')
            self.assertEqual(data['operator_label'], 'good')
            self.assertNotIn('outcome', data)
            self.assertTrue(data['quality']['training_ready'])
            recorder.close()

    def test_click_without_detection_exports_consistent_target_features(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace as NS
        from ur10e_curobo.demo_sequence import DemoSequence
        recorder = DemoSequence(['arm'], ['finger'])
        recorder.set_target('../full_before.jpg', [10, 20], [100, 100],
                            {'operator_point_confirmed': True, 'candidates': []})
        with TemporaryDirectory() as destination:
            for i in range(31):
                ns = 10**9 + i * 100_000_000
                msg = NS(header=NS(stamp=NS(sec=ns//10**9, nanosec=ns%10**9)),
                         name=['arm', 'finger'], position=[0., .5], effort=[])
                recorder.state(msg, ns)
                recorder.gripper_state(msg, ns)
                recorder.frame(NS(header=msg.header, data=b'jpeg'), ns)
            recorder.command('gripper.close_request', 1, ns, source_stamp_ns=ns)
            q = recorder.finalize(destination, complete=True)
            data = json.loads((Path(destination)/'sequence/sequence.json').read_text())
            self.assertTrue(q['data_ready'])
            self.assertIsNone(data['target']['base_xyz'])
            self.assertFalse(data['target']['identity_verified'])
            for frame in data['training_frames']:
                self.assertEqual(len(frame['observation_state']), len(data['observation_state_names']))
                self.assertEqual(frame['observation_state'][-2:], [.1, .2])
        recorder.close()

    def test_operator_label_alone_controls_readiness(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from ur10e_curobo.demo_sequence import update_training_readiness
        with TemporaryDirectory() as destination:
            path = Path(destination) / 'sequence' / 'sequence.json'
            path.parent.mkdir()
            sample_path = Path(destination) / 'sample.json'
            for label, data_ready, expected in [
                    ('good', True, True),
                    ('fail', True, False),
                    ('edge', True, False),
                    ('good', False, False)]:
                with self.subTest(label=label, data_ready=data_ready):
                    path.write_text(json.dumps({'quality': {'data_ready': data_ready}}))
                    sample_path.write_text(json.dumps({'operator_label': None}))
                    self.assertEqual(update_training_readiness(destination, label), expected)
                    saved = json.loads(path.read_text())
                    sample = json.loads(sample_path.read_text())
                    self.assertEqual(saved['label_source'], 'operator')
                    self.assertEqual(saved['operator_label'], label)
                    self.assertEqual(saved['quality']['training_ready'], expected)
                    self.assertEqual(sample['label_source'], 'operator')
                    self.assertEqual(sample['operator_label'], label)
                    self.assertEqual(sample['sequence_quality'], saved['quality'])

    def test_motion_window_excludes_wait_but_preserves_raw_frames(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from ur10e_curobo.demo_sequence import DemoSequence
        recorder = DemoSequence(['arm'], ['finger'])
        recorder.frames = [dict(stamp_ns=t, image='frame.jpg')
                           for t in [1, 2_000_000_000, 2_100_000_000, 4_000_000_000]]
        recorder.command('recording.motion_start', 1, 0, source_stamp_ns=2_000_000_000)
        recorder.command('gripper.close_request', 1, 0, source_stamp_ns=2_100_000_000)
        with TemporaryDirectory() as destination:
            q = recorder.finalize(destination, complete=True)
            data = json.loads((Path(destination)/'sequence/sequence.json').read_text())
            self.assertEqual(len(data['frames']), 4)
            self.assertEqual(q['raw_frames'], 4)
            self.assertEqual(q['frames'], 2)
            self.assertAlmostEqual(q['max_frame_gap_s'], .1)
            self.assertFalse(q['data_ready'])
        recorder.close()

    def test_uniform_export_rejects_gap(self):
        from ur10e_curobo.demo_sequence import fixed_rate_rows
        rows=[dict(stamp_ns=t,joints=[0],gripper_joints=[0]) for t in [10**9,2*10**9]]
        self.assertEqual(fixed_rate_rows(rows,10),[])

    def test_uniform_export_reaches_close_with_a_recent_source_frame(self):
        from ur10e_curobo.demo_sequence import fixed_rate_rows
        rows = [dict(stamp_ns=1_000_000_000 + i * 100_000_000,
                     joints=[0], gripper_joints=[0]) for i in range(10)]
        rows.append(dict(stamp_ns=1_950_000_000,
                         joints=[0], gripper_joints=[0]))
        out = fixed_rate_rows(rows, 5, close_time_ns=2_050_000_000)
        self.assertEqual(out[-1]['sample_stamp_ns'], 2_000_000_000)

    def test_gap_splits_instead_of_discarding_whole_episode(self):
        from ur10e_curobo.demo_sequence import fixed_rate_rows
        stamps=[1_000_000_000+i*100_000_000 for i in range(6)]
        stamps += [2_000_000_000+i*100_000_000 for i in range(6)]
        rows=[dict(stamp_ns=t,joints=[0],gripper_joints=[0]) for t in stamps]
        out=fixed_rate_rows(rows,5)
        self.assertEqual({r['segment_id'] for r in out},{0,1})
        self.assertEqual(sum(r['segment_end'] for r in out),2)

    def test_long_arm_gap_is_not_interpolated(self):
        out=pair_frames([{'stamp_ns':200_000_000}],
                        [{'stamp_ns':1,'joints':[0]},{'stamp_ns':400_000_000,'joints':[1]}])
        self.assertNotIn('joints',out[0])
