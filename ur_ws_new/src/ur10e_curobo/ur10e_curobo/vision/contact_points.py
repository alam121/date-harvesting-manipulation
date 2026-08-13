"""Three-finger contact-region estimation for the selected date fruit.

This module deliberately does not command the robot.  It predicts three image-
space fingertip regions, checks their local depth support, and returns a quality
score that can be visualised and validated before it influences wrist motion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _empty_result(reason: str) -> Dict[str, Any]:
    return {
        "valid": False,
        "score": 0.0,
        "reason": reason,
        "center_px": None,
        "points_px": np.empty((0, 2), dtype=np.float32),
        "points_3d": np.empty((0, 3), dtype=np.float32),
        "patch_radius_px": 0.0,
        "rotation_deg": 0.0,
        "point_scores": np.empty((0,), dtype=np.float32),
        "candidates": [],
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground component."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)
    return cleaned


def _fit_mask_ellipse(mask: np.ndarray) -> Optional[tuple[np.ndarray, float, float, float]]:
    """Return ellipse center, semiaxes and angle (radians) for the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) >= 5:
        (cx, cy), (diameter_a, diameter_b), angle_deg = cv2.fitEllipse(contour)
        return (
            np.array([cx, cy], dtype=np.float32),
            max(1.0, 0.5 * float(diameter_a)),
            max(1.0, 0.5 * float(diameter_b)),
            math.radians(float(angle_deg)),
        )
    x, y, w, h = cv2.boundingRect(contour)
    return (
        np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32),
        max(1.0, 0.5 * w),
        max(1.0, 0.5 * h),
        0.0,
    )


def _patch_metrics(
    mask: np.ndarray,
    distance_map: np.ndarray,
    point: np.ndarray,
    patch_radius: int,
    roi_xyz: Optional[np.ndarray],
    target_depth: Optional[float],
) -> tuple[float, float, float, np.ndarray]:
    """Return mask coverage, edge support, depth support and median XYZ."""
    h, w = mask.shape
    px = int(round(float(point[0])))
    py = int(round(float(point[1])))
    x0 = max(0, px - patch_radius)
    x1 = min(w, px + patch_radius + 1)
    y0 = max(0, py - patch_radius)
    y1 = min(h, py + patch_radius + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0, np.full(3, np.nan, dtype=np.float32)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - px) ** 2 + (yy - py) ** 2 <= patch_radius ** 2
    circle_count = int(np.count_nonzero(circle))
    if circle_count == 0:
        return 0.0, 0.0, 0.0, np.full(3, np.nan, dtype=np.float32)

    local_mask = mask[y0:y1, x0:x1] > 0
    supported = circle & local_mask
    coverage = float(np.count_nonzero(supported)) / float(circle_count)
    edge_support = min(1.0, float(distance_map[py, px]) / max(1.0, patch_radius))

    point_3d = np.full(3, np.nan, dtype=np.float32)
    depth_support = 1.0
    if roi_xyz is not None and roi_xyz.shape[:2] == mask.shape:
        xyz_patch = roi_xyz[y0:y1, x0:x1, :3]
        finite = supported & np.isfinite(xyz_patch).all(axis=2) & (xyz_patch[:, :, 2] > 0.0)
        supported_count = max(1, int(np.count_nonzero(supported)))
        valid_count = int(np.count_nonzero(finite))
        valid_fraction = float(valid_count) / float(supported_count)
        if valid_count:
            point_3d = np.median(xyz_patch[finite], axis=0).astype(np.float32)
            if target_depth is not None and np.isfinite(target_depth):
                depth_delta = abs(float(point_3d[2]) - float(target_depth))
                consistency = math.exp(-((depth_delta / 0.035) ** 2))
            else:
                consistency = 1.0
            depth_support = valid_fraction * consistency
        else:
            depth_support = 0.0

    return coverage, edge_support, depth_support, point_3d


def estimate_three_finger_contacts(
    mask: np.ndarray,
    *,
    roi_xyz: Optional[np.ndarray] = None,
    target_depth: Optional[float] = None,
    focal_length_px: Optional[float] = None,
    fingertip_radius_m: float = 0.006,
    radial_fraction: float = 0.68,
    rotation_samples: int = 24,
    minimum_score: float = 0.60,
) -> Dict[str, Any]:
    """Estimate three supported fingertip contact regions.

    The three fingers are modelled 120 degrees apart.  Candidate wrist rotations
    over one symmetry period are evaluated using mask coverage, distance from the
    segmentation edge, and local depth validity.  Pixel coordinates are local to
    the supplied mask/ROI.
    """
    if mask is None or np.asarray(mask).ndim != 2:
        return _empty_result("missing mask")

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    binary = _largest_component(binary)
    area = int(np.count_nonzero(binary))
    if area < 80:
        return _empty_result("mask too small")

    h, w = binary.shape
    distance_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, max_distance, _, max_location = cv2.minMaxLoc(distance_map)
    if max_distance < 2.0:
        return _empty_result("insufficient interior space")

    moments = cv2.moments(binary)
    centroid = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float32,
    )
    interior_center = np.array(max_location, dtype=np.float32)
    cx_i = int(np.clip(round(float(centroid[0])), 0, w - 1))
    cy_i = int(np.clip(round(float(centroid[1])), 0, h - 1))
    if binary[cy_i, cx_i] and distance_map[cy_i, cx_i] >= 0.55 * max_distance:
        center = 0.65 * centroid + 0.35 * interior_center
    else:
        center = interior_center

    small_dim = float(min(h, w))
    geometric_limit = max(3, int(round(0.12 * small_dim)))
    if (
        focal_length_px is not None
        and target_depth is not None
        and focal_length_px > 0.0
        and target_depth > 0.05
    ):
        physical_radius = int(round(float(focal_length_px) * fingertip_radius_m / target_depth))
        patch_radius = int(np.clip(physical_radius, 3, geometric_limit))
    else:
        patch_radius = geometric_limit

    ellipse = _fit_mask_ellipse(binary)
    if ellipse is None:
        return _empty_result("ellipse fit failed")

    # Precompute approximate patch support for O(1) candidate evaluation. Exact
    # circular support and median XYZ are calculated only for the winning triad.
    kernel_size = 2 * patch_radius + 1
    binary_f = binary.astype(np.float32)
    coverage_map = cv2.blur(binary_f, (kernel_size, kernel_size))
    edge_support_map = np.clip(distance_map / max(1.0, patch_radius), 0.0, 1.0)
    depth_support_map = np.ones_like(distance_map, dtype=np.float32)
    if roi_xyz is not None and roi_xyz.shape[:2] == binary.shape:
        finite = np.isfinite(roi_xyz[:, :, :3]).all(axis=2) & (roi_xyz[:, :, 2] > 0.0)
        if target_depth is not None and np.isfinite(target_depth):
            consistency = np.exp(-((roi_xyz[:, :, 2] - float(target_depth)) / 0.035) ** 2)
            consistency[~finite] = 0.0
        else:
            consistency = finite.astype(np.float32)
        numerator = cv2.blur((binary_f * consistency).astype(np.float32), (kernel_size, kernel_size))
        denominator = cv2.blur(binary_f, (kernel_size, kernel_size))
        depth_support_map = np.divide(
            numerator, denominator,
            out=np.zeros_like(numerator), where=denominator > 1e-6,
        )
    point_score_map = (
        0.45 * coverage_map
        + 0.30 * edge_support_map
        + 0.25 * depth_support_map
    )

    # Evaluate all wrist rotations and three finger directions as one vectorised
    # batch. This keeps selected-date processing below a camera-frame budget.
    rotation_samples = max(6, int(rotation_samples))
    base_angles = np.linspace(
        0.0, 2.0 * math.pi / 3.0, rotation_samples, endpoint=False,
        dtype=np.float32,
    )
    angles = base_angles[:, None] + (
        np.arange(3, dtype=np.float32)[None, :] * (2.0 * math.pi / 3.0)
    )
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=2).astype(np.float32)

    ellipse_center, axis_a, axis_b, ellipse_angle = ellipse
    cos_a = math.cos(ellipse_angle)
    sin_a = math.sin(ellipse_angle)
    rotation_t = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)
    origin_local = rotation_t @ (center - ellipse_center)
    direction_local = directions @ rotation_t.T
    aa = (direction_local[:, :, 0] / axis_a) ** 2 + (direction_local[:, :, 1] / axis_b) ** 2
    bb = 2.0 * (
        origin_local[0] * direction_local[:, :, 0] / (axis_a ** 2)
        + origin_local[1] * direction_local[:, :, 1] / (axis_b ** 2)
    )
    cc = (origin_local[0] / axis_a) ** 2 + (origin_local[1] / axis_b) ** 2 - 1.0
    discriminant = bb * bb - 4.0 * aa * cc
    root = np.sqrt(np.maximum(discriminant, 0.0))
    safe_denominator = np.where(aa > 1e-9, 2.0 * aa, 1.0)
    root_1 = (-bb - root) / safe_denominator
    root_2 = (-bb + root) / safe_denominator
    extents = np.maximum(
        np.where(root_1 > 0.0, root_1, 0.0),
        np.where(root_2 > 0.0, root_2, 0.0),
    )
    extents[(discriminant < 0.0) | (aa <= 1e-9)] = 0.0

    candidate_ok = np.all(extents >= max(4.0, 1.8 * patch_radius), axis=1)
    contact_distances = np.maximum(patch_radius + 1.0, radial_fraction * extents)
    contact_distances = np.minimum(contact_distances, np.maximum(1.0, extents - 1.0))
    points_float = center[None, None, :] + contact_distances[:, :, None] * directions
    px = np.clip(np.rint(points_float[:, :, 0]).astype(np.int32), 0, w - 1)
    py = np.clip(np.rint(points_float[:, :, 1]).astype(np.int32), 0, h - 1)

    # Fitted ellipses smooth mask irregularities. Move unsupported points inward
    # as one batch until they enter the actual segmentation mask.
    for _ in range(6):
        outside = binary[py, px] == 0
        if not np.any(outside):
            break
        contact_distances[outside] *= 0.88
        points_float = center[None, None, :] + contact_distances[:, :, None] * directions
        px = np.clip(np.rint(points_float[:, :, 0]).astype(np.int32), 0, w - 1)
        py = np.clip(np.rint(points_float[:, :, 1]).astype(np.int32), 0, h - 1)
    candidate_ok &= np.all(binary[py, px] > 0, axis=1)

    points_all = np.stack((px, py), axis=2).astype(np.float32)
    coverages_all = coverage_map[py, px]
    depth_scores_all = depth_support_map[py, px]
    point_scores_all = point_score_map[py, px]
    side_lengths = np.linalg.norm(points_all - np.roll(points_all, -1, axis=1), axis=2)
    separation_scores = np.minimum(
        1.0, np.min(side_lengths, axis=1) / max(1.0, 3.0 * patch_radius))
    mean_scores = np.mean(point_scores_all, axis=1)
    minimum_scores = np.min(point_scores_all, axis=1)
    symmetry_scores = np.clip(
        1.0 - np.std(point_scores_all, axis=1) / np.maximum(mean_scores, 1e-6),
        0.0, 1.0,
    )
    triangle_centers = np.mean(points_all, axis=1)
    centering_scores = np.clip(
        1.0 - np.linalg.norm(triangle_centers - center[None, :], axis=1)
        / max(1.0, float(max_distance)),
        0.0, 1.0,
    )
    total_scores = (
        0.50 * mean_scores
        + 0.35 * minimum_scores
        + 0.15 * separation_scores
    )
    total_scores[~candidate_ok] = -1.0
    best_idx = int(np.argmax(total_scores))
    if total_scores[best_idx] < 0.0:
        return _empty_result("date too narrow for three contacts")

    # Preserve every evaluated rotation for Phase-3 diagnostics. These use the
    # same vectorised score maps that select the winner, so retaining all 24 has
    # negligible additional frame-time cost. The winning triad is refined with
    # exact circular patches below, as before.
    ranked_indices = np.argsort(total_scores)[::-1]
    candidates: List[Dict[str, Any]] = []
    for rank, candidate_idx_raw in enumerate(ranked_indices, start=1):
        candidate_idx = int(candidate_idx_raw)
        scores = point_scores_all[candidate_idx].astype(np.float32)
        candidates.append({
            "rank": rank,
            "sample_index": candidate_idx,
            "psi_rad": float(base_angles[candidate_idx]),
            "psi_deg": float(math.degrees(float(base_angles[candidate_idx]))),
            "center_px": center.astype(np.float32).copy(),
            "points_px": points_all[candidate_idx].astype(np.float32).copy(),
            "point_scores": scores.copy(),
            "minimum_contact_score": float(minimum_scores[candidate_idx]),
            "mean_contact_score": float(mean_scores[candidate_idx]),
            "symmetry_score": float(symmetry_scores[candidate_idx]),
            "centering_score": float(centering_scores[candidate_idx]),
            "separation_score": float(separation_scores[candidate_idx]),
            "coverage_scores": coverages_all[candidate_idx].astype(np.float32).copy(),
            "depth_support_scores": depth_scores_all[candidate_idx].astype(np.float32).copy(),
            "total_score": float(total_scores[candidate_idx]),
            "geometry_valid": bool(candidate_ok[candidate_idx]),
            "selected": candidate_idx == best_idx,
        })

    best = {
        "valid": bool(
            total_scores[best_idx] >= minimum_score
            and float(np.min(coverages_all[best_idx])) >= 0.70
            and float(np.min(depth_scores_all[best_idx])) >= 0.45
        ),
        "score": float(total_scores[best_idx]),
        "reason": "supported" if total_scores[best_idx] >= minimum_score else "weak contact support",
        "center_px": center.astype(np.float32),
        "points_px": points_all[best_idx],
        "points_3d": np.empty((0, 3), dtype=np.float32),
        "patch_radius_px": float(patch_radius),
        "rotation_deg": float(math.degrees(float(base_angles[best_idx]))),
        "point_scores": point_scores_all[best_idx].astype(np.float32),
        "candidates": candidates,
    }

    # Refine the winning points using circular patches and robust XYZ medians.
    exact_coverages = []
    exact_depth_scores = []
    exact_point_scores = []
    points_3d = []
    for point in best["points_px"]:
        coverage, edge_support, depth_support, point_3d = _patch_metrics(
            binary, distance_map, point, patch_radius, roi_xyz, target_depth,
        )
        exact_coverages.append(coverage)
        exact_depth_scores.append(depth_support)
        exact_point_scores.append(
            0.45 * coverage + 0.30 * edge_support + 0.25 * depth_support)
        points_3d.append(point_3d)
    side_lengths = [
        float(np.linalg.norm(best["points_px"][i] - best["points_px"][(i + 1) % 3]))
        for i in range(3)
    ]
    separation_score = min(1.0, min(side_lengths) / max(1.0, 3.0 * patch_radius))
    best["point_scores"] = np.asarray(exact_point_scores, dtype=np.float32)
    best["points_3d"] = np.asarray(points_3d, dtype=np.float32)
    best["score"] = float(
        0.50 * np.mean(exact_point_scores)
        + 0.35 * np.min(exact_point_scores)
        + 0.15 * separation_score)
    best["valid"] = bool(
        best["score"] >= minimum_score
        and min(exact_coverages) >= 0.70
        and min(exact_depth_scores) >= 0.45
    )
    best["reason"] = "supported" if best["valid"] else "one or more contacts lack support"
    # Keep the selected candidate consistent with the exact winner refinement.
    # Ranking remains based on the common vectorised evaluation used for all 24.
    selected = next(candidate for candidate in candidates if candidate["selected"])
    selected["point_scores_exact"] = best["point_scores"].copy()
    selected["points_3d"] = best["points_3d"].copy()
    selected["total_score_exact"] = best["score"]
    selected["valid_exact"] = best["valid"]
    if not best["valid"] and best["reason"] == "supported":
        best["reason"] = "one or more contacts lack support"
    return best
