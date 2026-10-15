import unittest
from ur10e_curobo.goals import _stable_visible_fruits


class _Log:
    def info(self, *_):
        pass


class _Node:
    """Each read of all_fruits_seq advances to the next scripted frame."""

    def __init__(self, frames):
        self._frames, self._i = frames, -1

    def get_logger(self):
        return _Log()

    @property
    def all_fruits_seq(self):
        self._i = min(self._i + 1, len(self._frames) - 1)
        return self._i

    @property
    def all_fruit_poses(self):
        return self._frames[max(self._i, 0)]


DATE = [0.30, -1.20, -0.04]
GHOST = [0.33, -1.32, 0.05]


class StableVisibleFruitsTests(unittest.TestCase):
    def test_drops_single_frame_ghost(self):
        node = _Node([[DATE], [DATE], [[0.301, -1.201, -0.04], GHOST]])
        self.assertEqual(_stable_visible_fruits(node, timeout=0.5),
                         [[0.301, -1.201, -0.04]])

    def test_keeps_fruit_seen_in_two_of_three(self):
        node = _Node([[DATE, GHOST], [DATE], [DATE, [0.331, -1.32, 0.05]]])
        self.assertEqual(len(_stable_visible_fruits(node, timeout=0.5)), 2)

    def test_no_new_frames_falls_back_to_latest(self):
        node = _Node([[DATE, GHOST]])
        self.assertEqual(_stable_visible_fruits(node, timeout=0.2), [DATE, GHOST])


if __name__ == "__main__":
    unittest.main()
