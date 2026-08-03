"""Camera-based grasp verification using fingertip-relative image geometry.

This intentionally does not control the robot.  It produces CONFIRMED,
REJECTED, or UNCERTAIN so it can first be compared with operator labels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualGraspResult:
    label: str
    score: float
    red_pixels: int
    corridor: tuple[int, int, int, int] | None
    reason: str


@dataclass(frozen=True)
class TemporalGraspResult:
    label: str
    after_score: float
    appearance_correlation: float
    reason: str


def _fingertip_relative_crop(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return a normalized crop around the red fingertips and grasp boundary."""
    if image_bgr is None or image_bgr.ndim != 3:
        return None
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 110, 120), (12, 255, 255))
    red |= cv2.inRange(hsv, (170, 110, 120), (179, 255, 255))
    search = np.zeros_like(red)
    search[int(0.70 * height):, int(0.40 * width):int(0.70 * width)] = 255
    red &= search
    ys, xs = np.where(red > 0)
    if xs.size < 250:
        return None
    x1 = int(np.percentile(xs, 2))
    x2 = int(np.percentile(xs, 98)) + 1
    fingertip_top = int(np.percentile(ys, 5))
    pad_x = int(0.60 * max(1, x2 - x1))
    x1 = max(0, x1 - pad_x)
    x2 = min(width, x2 + pad_x)
    y1 = max(0, fingertip_top - 100)
    if x2 <= x1 or height <= y1:
        return None
    return cv2.resize(image_bgr[y1:height, x1:x2], (160, 160))


def classify_grasp_pair(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
) -> TemporalGraspResult:
    """Classify a grasp from correctly timed before/after-reverse images.

    Rules are deliberately conservative and logging-only.  They were initialized
    from operator-labelled deep, tip, and failure pairs; ambiguous combinations
    are never converted into a confirmed miss.
    """
    before_crop = _fingertip_relative_crop(before_bgr)
    after_crop = _fingertip_relative_crop(after_bgr)
    after_result = classify_grasp_image(after_bgr)
    if before_crop is None or after_crop is None:
        return TemporalGraspResult(
            "UNCERTAIN", after_result.score, 0.0,
            "red fingertips were not reliable in both frames")

    before_gray = cv2.cvtColor(before_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    after_gray = cv2.cvtColor(after_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    before_flat = before_gray.reshape(-1)
    after_flat = after_gray.reshape(-1)
    if float(before_flat.std()) < 1.0 or float(after_flat.std()) < 1.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(before_flat, after_flat)[0, 1])

    score = after_result.score
    if score >= 0.45:
        label = "DEEP_GRASP"
        reason = "high date occupancy remains after reverse"
    elif score >= 0.32 and correlation >= 0.20:
        label = "TIP_GRASP"
        reason = "date remains near the fingertip boundary after reverse"
    elif score < 0.32 and correlation >= 0.30:
        label = "TIP_GRASP"
        reason = "low occupancy but retained fingertip-relative appearance"
    elif score <= 0.32 and correlation < 0.20:
        label = "POSSIBLE_MISS"
        reason = "low occupancy and date appearance did not remain with fingertips"
    else:
        label = "UNCERTAIN"
        reason = "temporal and occupancy evidence disagree"
    return TemporalGraspResult(label, score, correlation, reason)


def classify_grasp_image(
    image_bgr: np.ndarray,
    success_threshold: float = 0.45,
    failure_threshold: float = 0.38,
) -> VisualGraspResult:
    """Classify date occupancy below and between the two red fingertips.

    ``score`` is the fraction of date-coloured pixels in a corridor derived
    from the visible red fingertips.  The gap between the thresholds is
    deliberately reported as UNCERTAIN.
    """
    if image_bgr is None or image_bgr.ndim != 3:
        return VisualGraspResult("UNCERTAIN", 0.0, 0, None, "invalid image")

    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    red = cv2.inRange(hsv, (0, 110, 120), (12, 255, 255))
    red |= cv2.inRange(hsv, (170, 110, 120), (179, 255, 255))
    search = np.zeros_like(red)
    search[int(0.70 * height):, int(0.40 * width):int(0.70 * width)] = 255
    red &= search
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    ys, xs = np.where(red > 0)
    red_pixels = int(xs.size)
    if red_pixels < 250:
        return VisualGraspResult(
            "UNCERTAIN", 0.0, red_pixels, None, "red fingertips not reliably visible")

    # Percentiles reject isolated red objects and specular pixels.
    x1 = int(np.percentile(xs, 5))
    x2 = int(np.percentile(xs, 95)) + 1
    y1 = int(np.percentile(ys, 5))
    y2 = height
    if x2 - x1 < int(0.05 * width):
        return VisualGraspResult(
            "UNCERTAIN", 0.0, red_pixels, None, "fingertip corridor too narrow")

    # Dates in this setup occupy orange/brown hues.  Measuring only below the
    # fingertip-top line avoids confusing the hanging bunch and trunk above it
    # with a date retained by the gripper.
    date_colour = cv2.inRange(hsv, (3, 60, 25), (25, 255, 235))
    corridor_mask = date_colour[y1:y2, x1:x2]
    score = float(np.count_nonzero(corridor_mask)) / float(corridor_mask.size)

    if score >= success_threshold:
        label = "GRASP_CONFIRMED"
        reason = "date-coloured object occupies fingertip corridor"
    elif score <= failure_threshold:
        label = "MISS_CONFIRMED"
        reason = "fingertip corridor is mostly empty"
    else:
        label = "UNCERTAIN"
        reason = "score lies inside conservative uncertainty band"
    return VisualGraspResult(label, score, red_pixels, (x1, y1, x2, y2), reason)


def vote_results(results: Iterable[VisualGraspResult]) -> VisualGraspResult:
    """Use the median valid frame so one blurred frame cannot decide a grasp."""
    valid = [r for r in results if r.corridor is not None]
    if not valid:
        return VisualGraspResult("UNCERTAIN", 0.0, 0, None, "no valid frames")
    median = sorted(valid, key=lambda r: r.score)[len(valid) // 2]
    return VisualGraspResult(
        median.label,
        median.score,
        median.red_pixels,
        median.corridor,
        f"median of {len(valid)} valid frame(s): {median.reason}",
    )


def main() -> None:
    """Offline check for operator-labelled PNG/JPEG images."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify labelled grasp images")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    for name in args.images:
        result = classify_grasp_image(cv2.imread(name))
        print(
            f"{Path(name).name}: {result.label} score={result.score:.3f} "
            f"red_pixels={result.red_pixels} reason={result.reason}")


if __name__ == "__main__":
    main()
