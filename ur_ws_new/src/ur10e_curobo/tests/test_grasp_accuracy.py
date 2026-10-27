import math
import types
import unittest

from ur10e_curobo.grasp_episode_recorder import snapshot_close_geometry


def _node(date_age_s, pose, observation_poses):
    now_ns = 1_000_000_000_000
    log = types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None)

    def no_tf(*_):
        raise RuntimeError("no tf")

    return types.SimpleNamespace(
        get_end_effector_pose=lambda: pose,
        cfg=types.SimpleNamespace(planner=types.SimpleNamespace(
            closure_center_offset_tcp_m=[0.0, 0.0, 0.01], gripper_tip_extra_m=0.0)),
        active_goal_grasp_mode="NORMAL",
        _fruit_observation=types.SimpleNamespace(
            stamp_ns=now_ns, poses=observation_poses),
        get_clock=lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(
            nanoseconds=now_ns + int(date_age_s * 1e9))),
        tf_buffer=types.SimpleNamespace(lookup_transform=no_tf),
        get_logger=lambda: log)


class CloseGeometryTests(unittest.TestCase):
    # Tool z pointing along base -y (the harvest approach): +90 deg about x.
    POSE = [0.4, -1.2, 0.4, math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]

    def test_splits_tracking_perception_and_target_relative_error(self):
        date = (0.402, -1.225, 0.4)          # closure centre is at y=-1.21
        node = _node(0.1, self.POSE, (date, (0.5, -1.0, 0.4)))
        snap = snapshot_close_geometry(
            node, [0.402, -1.215, 0.4], [0.4, -1.2005, 0.4])
        self.assertAlmostEqual(snap["tracking_mm"], 0.5, places=3)
        self.assertAlmostEqual(math.hypot(*snap["perception_mm"]), 10.0, places=3)
        x, y, z = snap["target_rel_tool_mm"]
        self.assertAlmostEqual(x, 2.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)
        self.assertAlmostEqual(z, 15.0, places=3)   # date 15mm beyond: arm short
        self.assertAlmostEqual(snap["approach_pitch_deg"], 0.0, places=3)

    def test_stale_detection_keeps_tracking_only(self):
        node = _node(3.0, self.POSE, ((0.402, -1.225, 0.4),))
        snap = snapshot_close_geometry(node, [0.402, -1.215, 0.4], [0.4, -1.2, 0.4])
        self.assertNotIn("date_at_close", snap)
        self.assertIn("tracking_mm", snap)
        self.assertIn("old", snap["no_date_reason"])

    def test_falls_back_to_last_inflight_sighting_when_frame_hides_date(self):
        import time
        node = _node(0.1, self.POSE, ((0.5, -1.0, 0.4),))   # target not in frame
        report = {"last_match": [0.402, -1.225, 0.4],
                  "last_match_wall": time.time() - 0.8}
        snap = snapshot_close_geometry(
            node, [0.402, -1.215, 0.4], [0.4, -1.2, 0.4], inflight_report=report)
        self.assertEqual(snap["date_source"], "inflight_last_match")
        self.assertAlmostEqual(snap["target_rel_tool_mm"][2], 15.0, places=3)


if __name__ == "__main__":
    unittest.main()
