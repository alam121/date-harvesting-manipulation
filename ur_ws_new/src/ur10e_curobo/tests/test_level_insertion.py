import unittest
from ur10e_curobo.level_insertion import level_target

class LevelInsertionTests(unittest.TestCase):
    def test_changes_only_tcp_height_without_mutating_original(self):
        target = [.2, -1.1, .417, 1., 0., 0., 0.]
        result = level_target(target, .4)
        self.assertEqual(result, [.2, -1.1, .4, 1., 0., 0., 0.])
        self.assertEqual(target[2], .417)

    def test_disabled_preserves_target(self):
        target = [.2, -1.1, .417, 1., 0., 0., 0.]
        self.assertEqual(level_target(target, None), target)

    def test_rejects_invalid_measured_height(self):
        with self.assertRaises(ValueError):
            level_target([0, 0, 0], float('nan'))
