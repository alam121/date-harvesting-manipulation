"""Fruit scoring and collision-free direction computation."""

import math
from typing import Dict, List, Optional, Any

import numpy as np

from .config import (
    SCORE_WEIGHTS,
    DIST_MIN, DIST_MAX,
    Z_STD_IDEAL, Z_STD_WORST,
    STICKY_BONUS,
    BEST_REUSE_THRESH,
    FRUIT_RADIUS_DEFAULT, FRUIT_RADIUS_MIN, FRUIT_RADIUS_MAX,
    APPROACH_CHECK_DIST,
    NUM_CANDIDATE_DIRS,
)
from .math_utils import ray_sphere_intersection


def estimate_fruit_radius(target: Dict[str, Any], fx: Optional[float] = None) -> float:
    """
    Estimate fruit radius from bounding box and depth using camera geometry.

    Args:
        target: dict with "bb" (x1,y1,x2,y2), "Zc" (depth in meters)
        fx: camera focal length in pixels (optional, uses default if None)

    Returns:
        Estimated radius in meters, clamped to [FRUIT_RADIUS_MIN, FRUIT_RADIUS_MAX]
    """
    if fx is None:
        fx = 700.0  # typical ZED camera focal length

    bb = target.get("bb")
    Zc = target.get("Zc", 0.5)

    if bb is None or Zc <= 0.1:
        return FRUIT_RADIUS_DEFAULT

    x1, y1, x2, y2 = bb
    bbox_width_px = x2 - x1
    bbox_height_px = y2 - y1

    # Use smaller dimension (more reliable for partially visible fruits)
    bbox_size_px = min(bbox_width_px, bbox_height_px)

    # Convert pixel size to meters using similar triangles:
    # real_size / depth = pixel_size / focal_length
    estimated_diameter = (bbox_size_px * Zc) / fx
    estimated_radius = estimated_diameter / 2.0

    # Clamp to reasonable range
    return max(FRUIT_RADIUS_MIN, min(FRUIT_RADIUS_MAX, estimated_radius))


def compute_fruit_score(
    target: Dict[str, Any],
    prev_pt_base: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute a comprehensive score for a detected fruit.

    Returns dict with:
        - total_score: weighted sum of all factors (0-1, higher is better)
        - components: individual score components for debugging
    """
    components = {}

    # 1. Distance score (closer = better)
    dist = target.get("dist", float("inf"))
    if dist <= DIST_MIN:
        dist_score = 1.0
    elif dist >= DIST_MAX:
        dist_score = 0.0
    else:
        dist_score = 1.0 - (dist - DIST_MIN) / (DIST_MAX - DIST_MIN)
    components["distance"] = dist_score

    # 2. Visibility score (higher vis_ratio = better)
    vis_ratio = target.get("vis_ratio", 0.0)
    vis_score = max(0.0, min(1.0, vis_ratio))
    components["visibility"] = vis_score

    # 3. Depth quality score (lower z_std = better)
    z_std = target.get("z_std", Z_STD_WORST)
    if z_std <= Z_STD_IDEAL:
        depth_score = 1.0
    elif z_std >= Z_STD_WORST:
        depth_score = 0.0
    else:
        depth_score = 1.0 - (z_std - Z_STD_IDEAL) / (Z_STD_WORST - Z_STD_IDEAL)
    components["depth_quality"] = depth_score

    # 4. Detection confidence score
    confidence = target.get("confidence", 0.5)
    conf_score = max(0.0, min(1.0, confidence))
    components["confidence"] = conf_score

    # 5. Ellipse quality score (bonus for valid orientation)
    has_ellipse = target.get("short_axis_cam") is not None
    ellipse_score = 1.0 if has_ellipse else 0.3  # partial credit if no ellipse
    components["ellipse"] = ellipse_score

    # 6. Center bias score (fruits near frame center have better depth data)
    bb = target.get("bb")
    img_w = target.get("img_width", 1280)
    img_h = target.get("img_height", 720)
    if bb is not None:
        x1, y1, x2, y2 = bb
        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0
        dist_from_center = math.hypot(bbox_cx - img_cx, bbox_cy - img_cy)
        max_dist = math.hypot(img_cx, img_cy)
        center_score = 1.0 - (dist_from_center / max_dist) if max_dist > 0 else 0.5
    else:
        center_score = 0.5  # neutral if no bbox
    components["center_bias"] = center_score

    # Weighted sum
    total = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        total += weight * components.get(key, 0.0)

    # Sticky bonus: if this fruit matches the previous best, add bonus
    if prev_pt_base is not None and target.get("pt_base") is not None:
        cur_pt = np.array([
            target["pt_base"].point.x,
            target["pt_base"].point.y,
            target["pt_base"].point.z,
        ], dtype=float)
        dist_to_prev = np.linalg.norm(cur_pt - prev_pt_base)
        if dist_to_prev < BEST_REUSE_THRESH:
            total += STICKY_BONUS
            components["sticky_bonus"] = STICKY_BONUS

    # Clamp final score
    total = max(0.0, min(1.0 + STICKY_BONUS, total))

    return {"total_score": total, "components": components}


def compute_collision_free_direction(
    best_target: Dict[str, Any],
    all_targets: List[Dict[str, Any]],
    best_idx: int,
    heatmap_dir: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Find the approach direction with most clearance from other fruits.

    Strategy:
    1. Sample directions in a hemisphere (toward camera = -Z in camera frame)
    2. For each direction, check clearance to other fruits
    3. Blend best clearance direction with heatmap direction for grasp accuracy

    Args:
        best_target: The target fruit we want to approach
        all_targets: List of all detected fruits
        best_idx: Index of best_target in all_targets
        heatmap_dir: Original heatmap-based direction (optional, for blending)

    Returns:
        dict with:
            - direction: 3D unit vector for approach direction in camera frame
            - clearance: Distance to nearest obstacle
            - is_collision_free: Whether the chosen direction is clear
    """
    centroid = np.array([best_target["Xc"], best_target["Yc"], best_target["Zc"]])
    best_radius = estimate_fruit_radius(best_target)

    # Collect other fruits as spheres with adaptive radii
    other_spheres = []
    for i, t in enumerate(all_targets):
        if i == best_idx:
            continue
        other_c = np.array([t["Xc"], t["Yc"], t["Zc"]])
        other_r = estimate_fruit_radius(t)
        other_spheres.append((other_c, other_r))

    # If no other fruits, just use heatmap direction
    if len(other_spheres) == 0:
        if heatmap_dir is not None:
            return {
                "direction": heatmap_dir,
                "clearance": float('inf'),
                "is_collision_free": True,
                "num_blocked": 0,
            }
        else:
            return {
                "direction": np.array([0.0, 0.0, -1.0]),  # default: toward camera
                "clearance": float('inf'),
                "is_collision_free": True,
                "num_blocked": 0,
            }

    # Generate candidate directions on a hemisphere (facing camera = -Z)
    candidate_dirs = []

    # Sample azimuth angles around Z axis
    for i in range(NUM_CANDIDATE_DIRS):
        azimuth = 2.0 * math.pi * i / NUM_CANDIDATE_DIRS

        # Multiple elevation angles (0 = horizontal, positive = toward camera)
        for elev_deg in [0, 20, 40, 60]:
            elev = math.radians(elev_deg)
            x = math.cos(azimuth) * math.cos(elev)
            y = math.sin(azimuth) * math.cos(elev)
            z = -math.sin(elev)  # negative Z = toward camera
            candidate_dirs.append(np.array([x, y, z], dtype=float))

    # Add straight toward camera
    candidate_dirs.append(np.array([0.0, 0.0, -1.0]))

    # Add the heatmap direction as a candidate (if available)
    if heatmap_dir is not None:
        candidate_dirs.append(heatmap_dir.copy())

    # Score each direction by clearance
    best_dir = None
    best_clearance = -1.0
    best_blocked = 0

    for d in candidate_dirs:
        d = d / (np.linalg.norm(d) + 1e-9)

        # Cast ray from centroid in this direction
        # Check for intersections with other fruit spheres
        min_clearance = float('inf')
        num_blocked = 0

        for (sphere_c, sphere_r) in other_spheres:
            hit, t_hit = ray_sphere_intersection(centroid, d, sphere_c, sphere_r)

            if hit and t_hit < APPROACH_CHECK_DIST:
                num_blocked += 1
                min_clearance = min(min_clearance, t_hit)
            elif not hit:
                # Compute closest approach distance
                # Project sphere center onto ray
                to_sphere = sphere_c - centroid
                t_closest = np.dot(to_sphere, d)
                if t_closest > 0:  # sphere is in front
                    closest_pt = centroid + t_closest * d
                    dist_to_center = np.linalg.norm(closest_pt - sphere_c)
                    clearance_at_closest = dist_to_center - sphere_r
                    if clearance_at_closest < min_clearance:
                        min_clearance = max(0.0, clearance_at_closest)

        # Prefer directions with higher clearance
        if min_clearance > best_clearance:
            best_clearance = min_clearance
            best_dir = d.copy()
            best_blocked = num_blocked

    # If heatmap direction has decent clearance, blend with it for grasp accuracy
    if heatmap_dir is not None and best_clearance > best_radius:
        # Check heatmap direction clearance
        hm_clearance = float('inf')
        hm_d = heatmap_dir / (np.linalg.norm(heatmap_dir) + 1e-9)
        for (sphere_c, sphere_r) in other_spheres:
            hit, t_hit = ray_sphere_intersection(centroid, hm_d, sphere_c, sphere_r)
            if hit:
                hm_clearance = min(hm_clearance, t_hit)

        # If heatmap direction is also clear, blend toward it
        if hm_clearance > best_radius * 2:
            # Blend: 60% collision-free, 40% heatmap for grasp accuracy
            blended = 0.6 * best_dir + 0.4 * hm_d
            blended = blended / (np.linalg.norm(blended) + 1e-9)
            best_dir = blended

    return {
        "direction": best_dir,
        "clearance": best_clearance,
        "is_collision_free": best_clearance > best_radius,
        "num_blocked": best_blocked,
        "estimated_radius": best_radius,
    }
