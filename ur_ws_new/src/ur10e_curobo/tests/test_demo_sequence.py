import unittest
from ur10e_curobo.demo_sequence import pair_frames

class SequenceTimingTests(unittest.TestCase):
    def test_nearest_source_state_not_arrival_order(self):
        rows = pair_frames([{'stamp_ns': 100_000_000}], [
            {'stamp_ns': 130_000_000, 'joints': [2]},
            {'stamp_ns': 90_000_000, 'joints': [1]}])
        self.assertEqual(rows[0]['joints'], [1])
        self.assertEqual(rows[0]['skew_ns'], -10_000_000)

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
