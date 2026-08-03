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

    # Don't clamp — let high-weight components (e.g. distance) dominate
    total = max(0.0, total)

    return {"total_score": total, "components": components}


def compute_collision_free_direction(
    best_target: Dict[str, Any],
    all_targets: List[Dict[str, Any]],
    best_idx: int,
    heatmap_dir: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Find the approach direction with most clearance from other fruits.
    Fully vectorized: all candidate directions tested against all spheres in
    a single numpy batch — no Python loops over candidates or spheres.
    """
    centroid = np.array([best_target["Xc"], best_target["Yc"], best_target["Zc"]], dtype=np.float64)
    best_radius = estimate_fruit_radius(best_target)

    _default_dir = heatmap_dir if heatmap_dir is not None else np.array([0.0, 0.0, -1.0])

    # Collect other fruits
    sphere_c_list, sphere_r_list = [], []
    for i, t in enumerate(all_targets):
        if i == best_idx:
            continue
        sphere_c_list.append([t["Xc"], t["Yc"], t["Zc"]])
        sphere_r_list.append(estimate_fruit_radius(t))

    if not sphere_c_list:
        return {"direction": _default_dir, "clearance": float('inf'),
                "is_collision_free": True, "num_blocked": 0}

    sphere_c = np.array(sphere_c_list, dtype=np.float64)  # (Ns, 3)
    sphere_r = np.array(sphere_r_list, dtype=np.float64)  # (Ns,)

    # ── Build candidate direction matrix (Nc, 3) ──────────────────────────────
    azimuths = np.linspace(0.0, 2.0 * math.pi, NUM_CANDIDATE_DIRS, endpoint=False)
    elevs = np.radians(np.array([0.0, 20.0, 40.0, 60.0]))
    az_g, el_g = np.meshgrid(azimuths, elevs)  # (4, Nd)
    xs = np.cos(az_g) * np.cos(el_g)
    ys = np.sin(az_g) * np.cos(el_g)
    zs = -np.sin(el_g)
    grid_dirs = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)  # (Nd*4, 3)
    extra = [np.array([0.0, 0.0, -1.0])]
    if heatmap_dir is not None:
        extra.append(heatmap_dir.copy())
    dirs = np.vstack([grid_dirs] + [np.array(e) for e in extra])  # (Nc, 3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs /= np.where(norms > 1e-9, norms, 1.0)

    # ── Vectorized ray-sphere intersection ────────────────────────────────────
    # oc[j] = centroid - sphere_c[j], shape (Ns, 3)
    oc = centroid - sphere_c  # (Ns, 3)
    # b[i, j] = 2 * dirs[i] · oc[j]  →  dirs @ oc.T has shape (Nc, Ns)
    b = 2.0 * (dirs @ oc.T)                      # (Nc, Ns)
    c_coef = np.sum(oc ** 2, axis=1) - sphere_r ** 2  # (Ns,)
    disc = b ** 2 - 4.0 * c_coef                  # (Nc, Ns) — c_coef broadcasts
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5
    hit = disc >= 0.0
    INF = 1e9
    # t_enter: smallest positive root, or INF if no valid intersection
    t_enter = np.where(hit & (t1 > 0.001), t1,
              np.where(hit & (t2 > 0.001), t2, INF))  # (Nc, Ns)

    # For near-miss (no hit): closest-approach clearance
    to_sphere = -oc  # (Ns, 3)  sphere_c - centroid
    t_cl = dirs @ to_sphere.T                        # (Nc, Ns)
    # closest point on ray, then distance to sphere surface
    cl_pts = centroid + t_cl[:, :, None] * dirs[:, None, :]  # (Nc, Ns, 3)
    cl_dist = np.linalg.norm(cl_pts - sphere_c, axis=2) - sphere_r  # (Nc, Ns)
    cl_dist = np.where((~hit) & (t_cl > 0), np.maximum(cl_dist, 0.0), INF)

    # Per-direction, per-sphere clearance = min(t_enter, cl_dist)
    clearance_mat = np.minimum(t_enter, cl_dist)       # (Nc, Ns)
    min_clearance = clearance_mat.min(axis=1)           # (Nc,) — worst sphere per dir
    num_blocked = (t_enter < APPROACH_CHECK_DIST).sum(axis=1)  # (Nc,)

    best_i = int(np.argmax(min_clearance))
    best_dir = dirs[best_i].copy()
    best_clearance = float(min_clearance[best_i])
    best_blocked = int(num_blocked[best_i])

    # Blend toward heatmap direction if it also has good clearance
    if heatmap_dir is not None and best_clearance > best_radius:
        hm_d = heatmap_dir / (np.linalg.norm(heatmap_dir) + 1e-9)
        hm_d_row = hm_d[None, :]                        # (1, 3)
        b_hm = 2.0 * (hm_d_row @ oc.T)                 # (1, Ns)
        disc_hm = b_hm ** 2 - 4.0 * c_coef
        sqrt_hm = np.sqrt(np.maximum(disc_hm, 0.0))
        t1_hm = (-b_hm - sqrt_hm) * 0.5
        t2_hm = (-b_hm + sqrt_hm) * 0.5
        hit_hm = disc_hm >= 0.0
        t_hm = np.where(hit_hm & (t1_hm > 0.001), t1_hm,
               np.where(hit_hm & (t2_hm > 0.001), t2_hm, INF))
        hm_clearance = float(t_hm.min())
        if hm_clearance > best_radius * 2:
            blended = 0.6 * best_dir + 0.4 * hm_d
            n_bl = np.linalg.norm(blended)
            best_dir = blended / n_bl if n_bl > 1e-9 else best_dir

    return {
        "direction": best_dir,
        "clearance": best_clearance,
        "is_collision_free": best_clearance > best_radius,
        "num_blocked": best_blocked,
        "estimated_radius": best_radius,
    }
