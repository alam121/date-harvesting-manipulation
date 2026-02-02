"""Visualization and drawing utilities for vision module."""

import math
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

from .config import (
    DRAW_ONLY_BEST, SHOW_REJECTED, SKIP_DRAW,
    APPROACH_CHECK_DIST,
)
from .math_utils import project_point_to_image
from .scoring import estimate_fruit_radius


class VisionVisualizer:
    """Handles all visualization for the vision node."""

    def __init__(self, intrinsics: Dict[str, float], image_scale: List[float], display_scale: float = 0.6):
        self.intrinsics = intrinsics
        self.image_scale = image_scale
        self.display_scale = display_scale

    def draw_rejected_targets(
        self,
        image: np.ndarray,
        rejected_targets: List[Dict[str, Any]]
    ) -> None:
        """Draw gray overlay on rejected detections."""
        if not SHOW_REJECTED:
            return

        for rej in rejected_targets:
            x1, y1, x2, y2 = rej["bb"]
            if x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                if roi.size:
                    gray_patch = np.full_like(roi, 128)
                    image[y1:y2, x1:x2] = cv2.addWeighted(gray_patch, 0.45, roi, 0.55, 0.0)
            cv2.rectangle(image, (x1, y1), (x2, y2), (150, 150, 150, 255), 2)
            if rej.get("reason"):
                cv2.putText(
                    image,
                    f"REJECT: {rej['reason']}",
                    (x1 + 6, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180, 255),
                    1,
                    cv2.LINE_AA,
                )

    def draw_blocking_fruits(
        self,
        image: np.ndarray,
        targets: List[Dict[str, Any]],
        best_idx: int
    ) -> None:
        """Draw red overlay on fruits blocking the approach path."""
        if best_idx is None or len(targets) <= 1:
            return

        t_best = targets[best_idx]
        best_centroid = np.array([t_best["Xc"], t_best["Yc"], t_best["Zc"]])
        approach_dir = t_best.get("approach_dir_cam")

        for i, t in enumerate(targets):
            if i == best_idx:
                continue

            other_centroid = np.array([t["Xc"], t["Yc"], t["Zc"]])
            x1o, y1o, x2o, y2o = t["bb"]

            # Check if this fruit blocks the approach path
            is_blocking = False
            if approach_dir is not None:
                to_other = other_centroid - best_centroid
                dist_to_other = np.linalg.norm(to_other)

                if dist_to_other > 0.01:
                    proj_dist = np.dot(to_other, approach_dir)
                    if proj_dist > 0 and proj_dist < APPROACH_CHECK_DIST:
                        perp_vec = to_other - proj_dist * approach_dir
                        perp_dist = np.linalg.norm(perp_vec)
                        other_radius = estimate_fruit_radius(t)
                        if perp_dist < other_radius * 2:
                            is_blocking = True

            # Apply transparent overlay based on blocking status
            if x2o > x1o and y2o > y1o:
                roi = image[y1o:y2o, x1o:x2o]
                if roi.size:
                    h_roi, w_roi = roi.shape[:2]
                    if is_blocking:
                        red_overlay = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
                        red_overlay[:, :, 2] = 180  # Red channel
                        red_overlay[:, :, 3] = 100  # Alpha
                        blended = cv2.addWeighted(red_overlay, 0.4, roi, 0.6, 0.0)
                        image[y1o:y2o, x1o:x2o] = blended

                        # Draw line from best fruit to blocking fruit
                        best_2d = project_point_to_image(best_centroid, self.intrinsics, self.image_scale)
                        other_2d = project_point_to_image(other_centroid, self.intrinsics, self.image_scale)
                        if best_2d and other_2d:
                            cv2.line(image, best_2d, other_2d, (0, 0, 200, 255), 2, cv2.LINE_AA)

                        cv2.putText(
                            image,
                            "BLOCKED",
                            (x1o + 5, y1o + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

    def draw_target(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        is_best: bool
    ) -> None:
        """Draw a single target with mask, bbox, and annotations."""
        x1, y1, x2, y2 = target["bb"]
        mask_resized = target["mask_resized"]
        Xc = target["Xc"]
        Yc = target["Yc"]
        Zc = target["Zc"]

        # Overlay mask
        try:
            h_roi, w_roi = mask_resized.shape
            colored_mask = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
            colored_mask[:, :, 1] = mask_resized  # green
            colored_mask[:, :, 3] = mask_resized  # alpha-like

            roi = image[y1:y2, x1:x2]
            blended = cv2.addWeighted(colored_mask, 0.45, roi, 0.55, 0.0)
            image[y1:y2, x1:x2] = blended

            # Heatmap overlay for BEST target
            if is_best:
                heatmap = target.get("heatmap")
                vis_r = float(target.get("vis_ratio", 0.0))
                if heatmap is not None:
                    hm = heatmap
                    if hm.shape[:2] != (h_roi, w_roi):
                        hm = cv2.resize(hm, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                    if hm.shape[2] == 3:
                        hm_rgba = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
                        hm_rgba[:, :, :3] = hm
                        alpha = int(80 + 150 * max(0.0, min(vis_r, 1.0)))
                        hm_rgba[:, :, 3] = alpha
                    else:
                        hm_rgba = hm
                    roi_best = image[y1:y2, x1:x2]
                    image[y1:y2, x1:x2] = cv2.addWeighted(hm_rgba, 0.6, roi_best, 0.4, 0.0)
        except Exception as e:
            print(f"[WARN] Mask overlay failed: {e}")

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0, 255), 2)

        # 2D centroid
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Color: best fruit = BLUE, others = RED
        if is_best:
            color = (255, 0, 0, 255)  # blue
            radius = 7
        else:
            color = (0, 0, 255, 255)  # red
            radius = 5

        cv2.circle(image, (cx, cy), radius, color, -1)

        # Draw approach axis for best target
        if is_best:
            self._draw_approach_arrows(image, target, cx, cy, x1, y1, w_roi, h_roi)

        # Draw labels
        self._draw_target_labels(image, target, cx, cy, is_best)

    def _draw_approach_arrows(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        cx: int, cy: int,
        x1: int, y1: int,
        w_roi: int, h_roi: int
    ) -> None:
        """Draw approach axis and direction arrows for best target."""
        axis_dir = target.get("approach_axis")
        if axis_dir is not None:
            vx, vy, vz = axis_dir
            n = math.sqrt(vx * vx + vy * vy)
            if n < 1e-6:
                vx, vy = 1.0, 0.0
            else:
                vx /= n
                vy /= n

            L = 60
            ax2 = int(cx + vx * L)
            ay2 = int(cy + vy * L)
            ax1 = int(cx - vx * L)
            ay1 = int(cy - vy * L)

            axis_color = (0, 255, 255, 255)
            cv2.arrowedLine(image, (cx, cy), (ax2, ay2), axis_color, 2, tipLength=0.25)
            cv2.line(image, (cx, cy), (ax1, ay1), axis_color, 2)

        # Show best heatmap peak direction (orange)
        peak_pt = target.get("best_point2d_smooth")
        if peak_pt is None:
            peak_pt = target.get("best_point2d")
        if peak_pt is not None:
            dest_x = int(round(x1 + peak_pt[0]))
            dest_y = int(round(y1 + peak_pt[1]))
            dest_x = max(0, min(dest_x, image.shape[1] - 1))
            dest_y = max(0, min(dest_y, image.shape[0] - 1))

            dx = float(dest_x - cx)
            dy = float(dest_y - cy)
            n = math.hypot(dx, dy)
            if n < 1e-3:
                dx, dy, n = 1.0, 0.0, 1.0
            dx /= n
            dy /= n

            L_out = max(w_roi, h_roi) + 10.0
            start_x = int(round(dest_x - dx * L_out))
            start_y = int(round(dest_y - dy * L_out))
            start_x = max(0, min(start_x, image.shape[1] - 1))
            start_y = max(0, min(start_y, image.shape[0] - 1))
            axis_color = (0, 165, 255, 255)  # orange
            cv2.arrowedLine(image, (start_x, start_y), (dest_x, dest_y), axis_color, 2, tipLength=0.25)

        # Draw 3D approach direction arrow (magenta)
        approach_dir_cam = target.get("approach_dir_cam")
        if approach_dir_cam is not None:
            Xc, Yc, Zc = target["Xc"], target["Yc"], target["Zc"]
            centroid_3d = np.array([Xc, Yc, Zc])
            arrow_len_3d = 0.08

            origin_2d = project_point_to_image(centroid_3d, self.intrinsics, self.image_scale)

            if origin_2d is not None:
                dir_end_3d = centroid_3d + approach_dir_cam * arrow_len_3d
                dir_end_2d = project_point_to_image(dir_end_3d, self.intrinsics, self.image_scale)

                if dir_end_2d:
                    cv2.arrowedLine(image, origin_2d, dir_end_2d, (255, 0, 255, 255), 3, tipLength=0.25)

                # Show clearance info
                clearance = target.get("clearance", float('inf'))
                is_clear = target.get("is_collision_free", True)
                if clearance < float('inf'):
                    clr_color = (0, 255, 0, 255) if is_clear else (0, 0, 255, 255)
                    clr_text = f"CLR {clearance*100:.0f}cm" if is_clear else "BLOCK"
                    cv2.putText(
                        image,
                        clr_text,
                        (target["bb"][0], target["bb"][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        clr_color,
                        2 if not is_clear else 1,
                        cv2.LINE_AA,
                    )

    def _draw_target_labels(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        cx: int, cy: int,
        is_best: bool
    ) -> None:
        """Draw labels and stats for a target."""
        if is_best:
            cv2.putText(
                image,
                "BEST",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            dist_grip = target.get("dist", 0.0)
            total_score = target.get("score", 0.0)
            cv2.putText(
                image,
                f"D:{dist_grip:.2f}m S:{total_score:.2f}",
                (cx + 10, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            dist_grip = target.get("dist", 0.0)
            cv2.putText(
                image,
                f"D:{dist_grip:.2f}m",
                (cx + 10, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_hud(self, image: np.ndarray, net_fps: float, loop_fps: float) -> None:
        """Draw heads-up display with FPS info."""
        cv2.putText(
            image,
            f"YOLO FPS: {net_fps:.1f}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Loop FPS: {loop_fps:.1f}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def render_frame(
        self,
        image: np.ndarray,
        targets: List[Dict[str, Any]],
        rejected_targets: List[Dict[str, Any]],
        best_idx: Optional[int],
        net_fps: float,
        loop_fps: float
    ) -> np.ndarray:
        """Render a complete frame with all visualizations."""
        if SKIP_DRAW:
            return image

        self.draw_rejected_targets(image, rejected_targets)
        self.draw_blocking_fruits(image, targets, best_idx)

        for i, t in enumerate(targets):
            if DRAW_ONLY_BEST and (best_idx is not None) and i != best_idx:
                continue
            self.draw_target(image, t, is_best=(i == best_idx))

        self.draw_hud(image, net_fps, loop_fps)

        # Resize for display
        display_image = cv2.resize(
            image,
            (int(image.shape[1] * self.display_scale), int(image.shape[0] * self.display_scale)),
            interpolation=cv2.INTER_AREA,
        )
        return display_image
