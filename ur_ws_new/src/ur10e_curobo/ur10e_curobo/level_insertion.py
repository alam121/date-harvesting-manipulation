"""Endpoint policy for the supervised level-insertion experiment."""
import math


def level_target(target, approach_z):
    result = list(target)
    if approach_z is not None:
        if not math.isfinite(approach_z):
            raise ValueError("Measured APPROACH height must be finite")
        result[2] = float(approach_z)
    return result
