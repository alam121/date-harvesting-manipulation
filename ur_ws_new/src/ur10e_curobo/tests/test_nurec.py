"""Stereo metadata, JPEG pairing, and bounded transport without opening a camera."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from builtin_interfaces.msg import Time

spec = importlib.util.spec_from_file_location(
    'nurec', Path(__file__).resolve().parents[1] / 'ur10e_curobo/vision/nurec.py')
nurec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nurec)


class NuRecTests(unittest.TestCase):
    def setUp(self):
        self.cal = SimpleNamespace(
            left_cam=SimpleNamespace(fx=750., fy=750., cx=960., cy=540.),
            right_cam=SimpleNamespace(fx=750., fy=750., cx=960., cy=540.),
            get_camera_baseline=lambda: .05)
        self.pubs = {}
        def publisher(typ, topic, qos):
            self.pubs[topic] = Mock()
            return self.pubs[topic]
        self.node = SimpleNamespace(create_publisher=publisher, get_logger=lambda: Mock())
        with patch.object(nurec.threading, 'Thread'):
            self.capture = nurec.NuRecCapture(
                self.node, self.cal, (1920, 1080), 'zed_mini_left_camera_frame')
        self.addCleanup(self.capture.close)

    def test_scaled_calibration_preserves_metric_baseline(self):
        left, right = self.capture.infos
        self.assertEqual((left.width, left.height), (1280, 720))
        self.assertAlmostEqual(left.k[0], 500.)
        self.assertEqual(left.p[3], 0.)
        self.assertAlmostEqual(-right.p[3] / right.p[0], .05)
        self.assertEqual(list(right.d), [0.] * 5)
        self.assertEqual(list(right.r), [1., 0., 0., 0., 1., 0., 0., 0., 1.])

    def test_off_does_not_capture_or_publish(self):
        self.assertFalse(self.capture.due())
        self.capture.submit(None, None, None)
        self.assertTrue(self.capture._queue.empty())
        for pub in self.pubs.values():
            pub.publish.assert_not_called()

    def test_throttling(self):
        self.capture.enabled = True
        with patch.object(nurec.time, 'monotonic', side_effect=[0., .1, .34]):
            self.assertTrue(self.capture.due())
            self.assertFalse(self.capture.due())
            self.assertTrue(self.capture.due())

    def test_pair_stamp_calibration_and_decodable_jpeg(self):
        self.capture.enabled = True
        stamp = Time(sec=12, nanosec=345)
        left = np.full((720, 1280, 4), [20, 80, 140, 255], np.uint8)
        right = np.full((720, 1280, 4), [90, 30, 10, 255], np.uint8)
        self.capture._publish((left, right, stamp))
        for side, expected in [('left', left), ('right', right)]:
            root = '/vision/nurec/' + side
            msg = self.pubs[root + '/image_rect/compressed'].publish.call_args.args[0]
            info = self.pubs[root + '/camera_info'].publish.call_args.args[0]
            self.assertEqual(msg.header, info.header)
            self.assertEqual(msg.header.stamp, stamp)
            decoded = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (info.height, info.width, 3))
            self.assertLess(np.abs(decoded.astype(float) - expected[:, :, :3]).mean(), 3.)

    def test_latest_pair_replaces_old_and_owns_buffers(self):
        self.capture.enabled = True
        image = np.zeros((2, 2, 4), np.uint8)
        self.capture.submit(image, image, Time(sec=1))
        image[:] = 5
        self.capture.submit(image, image, Time(sec=2))
        image[:] = 9
        left, right, stamp = self.capture._queue.get_nowait()
        self.assertEqual(stamp.sec, 2)
        self.assertTrue(np.all(left == 5))
        self.assertTrue(np.all(right == 5))
        self.assertTrue(self.capture._queue.empty())

    def test_invalid_baseline_rejected(self):
        for b in [0., float('nan'), -1.]:
            self.cal.get_camera_baseline = lambda: b
            with self.assertRaises(ValueError):
                nurec.NuRecCapture(self.node, self.cal, (1920, 1080), 'left')

    def test_bad_second_image_does_not_publish_partial_pair(self):
        self.capture.enabled = True
        with self.assertRaises(ValueError):
            self.capture._publish((np.zeros((720, 1280, 3), np.uint8),
                                   np.zeros((1, 1, 3), np.uint8), Time()))
        for pub in self.pubs.values():
            pub.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
