"""Visualization and drawing utilities for vision module."""

import math
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from .config import (
    DRAW_ONLY_BEST, DRAW_TOP_N, SHOW_REJECTED, SKIP_DRAW,
    SHOW_CLASSIFICATION_ZONES, CLASS_ZONE_MID_LEFT_THRESH, CLASS_ZONE_MID_RIGHT_THRESH,
    CLASS_ZONE_LOW_LEFT_THRESH, CLASS_ZONE_LOW_RIGHT_THRESH, APPROACH_CHECK_DIST,
    SHOW_GAP_DEBUG, SHOW_FINGER_CONTACTS,
)
from .math_utils import project_point_to_image
from .scoring import estimate_fruit_radius


class VisionVisualizer:
    """Handles all visualization for the vision node."""

    def __init__(
        self,
        intrinsics: Dict[str, float],
        image_scale: List[float],
        display_scale: float = 0.6,
        clean_harvest_overlay: bool = False,
    ):
        self.intrinsics = intrinsics
        self.image_scale = image_scale
        self.display_scale = display_scale
        self.show_classification_zones = SHOW_CLASSIFICATION_ZONES
        self.show_gap_debug = SHOW_GAP_DEBUG
        # Full harvesting needs an unobstructed view of the fruit.  Raw-YOLO
        # playback keeps the original class/confidence rendering.
        self.clean_harvest_overlay = clean_harvest_overlay
        self._s = 1.0  # active draw scale, set per-frame in render_frame
        self.last_fingertip_verification = "NO_TARGET"

    # ------------------------------------------------------------------
    # Helpers that respect the active draw scale
    # ------------------------------------------------------------------

    def _thick(self, base: int) -> int:
        """Scale a line/border thickness."""
        return max(1, round(base * self._s))

    def _font(self, base: float) -> float:
        """Scale a putText fontScale."""
        return max(0.25, base * self._s)

    def _radius(self, base: int) -> int:
        """Scale a circle radius."""
        return max(1, round(base * self._s))

    def _len(self, base: float) -> float:
        """Scale a pixel length (arrow, offset, etc.)."""
        return base * self._s

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
            cv2.rectangle(image, (x1, y1), (x2, y2), (150, 150, 150, 255), self._thick(2))
            if rej.get("reason"):
                cv2.putText(
                    image,
                    f"REJECT: {rej['reason']}",
                    (x1 + 6, y1 + round(self._len(18))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self._font(0.5),
                    (180, 180, 180, 255),
                    self._thick(1),
                    cv2.LINE_AA,
                )

    def draw_classification_zones(self, image: np.ndarray, viz_only: Optional[List[Dict[str, Any]]] = None) -> None:
        """Visualize image and bunch zones used by motion-side LEFT/CENTER/RIGHT logic."""
        if not self.show_classification_zones:
            return

        h, w = image.shape[:2]
        overlay = image.copy()

        bunch = None
        if viz_only:
            for obj in viz_only:
                if obj.get("class") == "bunch":
                    x1, y1, x2, y2 = obj["bb"]
                    if x2 > x1 and y2 > y1:
                        bunch = (x1, y1, x2, y2, obj.get("polygon"))
                        break

        low_y = int(0.60 * h)
        very_low_y = int(0.88 * h)
        mid_left_color = (255, 90, 40, 255)    # blue
        low_left_color = (255, 0, 200, 255)     # purple
        center_color = (80, 220, 80, 255)       # green
        mid_right_color = (60, 60, 255, 255)    # red
        low_right_color = (0, 165, 255, 255)    # orange

        cv2.line(overlay, (0, low_y), (w, low_y), (0, 255, 255, 255), self._thick(1), cv2.LINE_AA)
        cv2.line(overlay, (0, very_low_y), (w, very_low_y), (0, 180, 255, 255), self._thick(2), cv2.LINE_AA)

        if bunch is None:
            low_left = int(CLASS_ZONE_LOW_LEFT_THRESH * w)
            low_right = int(CLASS_ZONE_LOW_RIGHT_THRESH * w)
            mid_left = int(CLASS_ZONE_MID_LEFT_THRESH * w)
            mid_right = int(CLASS_ZONE_MID_RIGHT_THRESH * w)
            cv2.rectangle(overlay, (0, 0), (mid_left, low_y), mid_left_color, -1)
            cv2.rectangle(overlay, (mid_left, 0), (mid_right, low_y), center_color, -1)
            cv2.rectangle(overlay, (mid_right, 0), (w, low_y), mid_right_color, -1)
            cv2.rectangle(overlay, (0, low_y), (low_left, h), low_left_color, -1)
            cv2.rectangle(overlay, (low_left, low_y), (low_right, h), center_color, -1)
            cv2.rectangle(overlay, (low_right, low_y), (w, h), low_right_color, -1)
            cv2.line(overlay, (mid_left, 0), (mid_left, low_y), mid_left_color, self._thick(2), cv2.LINE_AA)
            cv2.line(overlay, (mid_right, 0), (mid_right, low_y), mid_right_color, self._thick(2), cv2.LINE_AA)
            cv2.line(overlay, (low_left, low_y), (low_left, h), low_left_color, self._thick(2), cv2.LINE_AA)
            cv2.line(overlay, (low_right, low_y), (low_right, h), low_right_color, self._thick(2), cv2.LINE_AA)
            image[:] = cv2.addWeighted(overlay, 0.14, image, 0.86, 0.0)
            cv2.putText(image, "fallback image zones", (self._radius(10), self._radius(24)),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55), (255, 255, 255, 255), self._thick(1), cv2.LINE_AA)
            return

        x1, y1, x2, y2, polygon = bunch
        bw = x2 - x1
        bh = y2 - y1
        low_left = x1 + int(CLASS_ZONE_LOW_LEFT_THRESH * bw)
        low_right = x1 + int(CLASS_ZONE_LOW_RIGHT_THRESH * bw)
        mid_left = x1 + int(CLASS_ZONE_MID_LEFT_THRESH * bw)
        mid_right = x1 + int(CLASS_ZONE_MID_RIGHT_THRESH * bw)
        bottom_y = y1 + int(0.80 * bh)
        mid_top = y1
        mid_bottom = min(max(low_y, y1), y2)
        low_top = max(low_y, y1)

        zone_overlay = image.copy()
        if mid_bottom > mid_top:
            cv2.rectangle(zone_overlay, (x1, mid_top), (mid_left, mid_bottom), mid_left_color, -1)
            cv2.rectangle(zone_overlay, (mid_left, mid_top), (mid_right, mid_bottom), center_color, -1)
            cv2.rectangle(zone_overlay, (mid_right, mid_top), (x2, mid_bottom), mid_right_color, -1)
            cv2.line(zone_overlay, (mid_left, mid_top), (mid_left, mid_bottom), mid_left_color, self._thick(2), cv2.LINE_AA)
            cv2.line(zone_overlay, (mid_right, mid_top), (mid_right, mid_bottom), mid_right_color, self._thick(2), cv2.LINE_AA)
        if y2 > low_top:
            cv2.rectangle(zone_overlay, (x1, low_top), (low_left, y2), low_left_color, -1)
            cv2.rectangle(zone_overlay, (low_left, low_top), (low_right, y2), center_color, -1)
            cv2.rectangle(zone_overlay, (low_right, low_top), (x2, y2), low_right_color, -1)
            cv2.line(zone_overlay, (low_left, low_top), (low_left, y2), low_left_color, self._thick(2), cv2.LINE_AA)
            cv2.line(zone_overlay, (low_right, low_top), (low_right, y2), low_right_color, self._thick(2), cv2.LINE_AA)

        cv2.rectangle(zone_overlay, (x1, bottom_y), (x2, y2), center_color, -1)

        zone_mask = np.zeros((h, w), dtype=np.uint8)
        if polygon is not None and len(polygon) > 2:
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(zone_mask, [pts], 255)
        else:
            cv2.rectangle(zone_mask, (x1, y1), (x2, y2), 255, -1)
        blended = cv2.addWeighted(zone_overlay, 0.25, image, 0.75, 0.0)
        image[zone_mask > 0] = blended[zone_mask > 0]

        if polygon is not None and len(polygon) > 2:
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 255, 255), thickness=self._thick(1))
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255, 255), self._thick(1))
        cv2.line(image, (x1, bottom_y), (x2, bottom_y), (0, 255, 255, 255), self._thick(2), cv2.LINE_AA)

        cv2.putText(image, "bunch-relative zones", (x1 + self._radius(4), max(self._radius(22), y1 - self._radius(8))),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.5), (255, 255, 255, 255), self._thick(1), cv2.LINE_AA)
        cv2.putText(image, "MID/HIGH", (x1 + self._radius(4), mid_top + self._radius(18)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.45), (255, 255, 255, 255), self._thick(1), cv2.LINE_AA)
        if y2 > low_top:
            cv2.putText(image, "LOW", (x1 + self._radius(4), low_top + self._radius(18)),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(0.45), (255, 255, 255, 255), self._thick(1), cv2.LINE_AA)
        cv2.putText(image, "bottom => CENTER", (x1 + self._radius(4), bottom_y + self._radius(18)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.45), (255, 255, 255, 255), self._thick(1), cv2.LINE_AA)

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

            if x2o > x1o and y2o > y1o:
                roi = image[y1o:y2o, x1o:x2o]
                if roi.size:
                    h_roi, w_roi = roi.shape[:2]
                    if is_blocking:
                        red_overlay = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
                        red_overlay[:, :, 2] = 180
                        red_overlay[:, :, 3] = 100
                        blended = cv2.addWeighted(red_overlay, 0.4, roi, 0.6, 0.0)
                        image[y1o:y2o, x1o:x2o] = blended

                        best_2d = project_point_to_image(best_centroid, self.intrinsics, self.image_scale)
                        other_2d = project_point_to_image(other_centroid, self.intrinsics, self.image_scale)
                        if best_2d and other_2d:
                            cv2.line(image, best_2d, other_2d, (0, 0, 200, 255), self._thick(2), cv2.LINE_AA)

                        cv2.putText(
                            image,
                            "BLOCKED",
                            (x1o + round(self._len(5)), y1o + round(self._len(15))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self._font(0.4),
                            (0, 0, 255, 255),
                            self._thick(1),
                            cv2.LINE_AA,
                        )

    def draw_target(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        is_best: bool,
        idx: int = -1
    ) -> None:
        """Draw a single target with mask, bbox, and annotations."""
        x1, y1, x2, y2 = target["bb"]
        mask_resized = target["mask_resized"]
        Zc = target["Zc"]

        # In the harvesting view, show the segmentation as an outline instead
        # of tinting the fruit.  The old 45% mask + best-target heatmap made it
        # difficult for the operator to see the actual date surface.
        if self.clean_harvest_overlay:
            try:
                contours, _ = cv2.findContours(
                    (mask_resized > 0).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                shifted = [
                    contour + np.asarray([[[x1, y1]]], dtype=contour.dtype)
                    for contour in contours
                ]
                contour_color = (
                    (0, 255, 255, 255) if is_best else (40, 230, 40, 255)
                )
                if shifted:
                    cv2.drawContours(
                        image, shifted, -1, contour_color,
                        self._thick(2), cv2.LINE_AA)
            except Exception as e:
                print(f"[WARN] Mask contour failed: {e}")
        else:
            self._draw_target_mask_overlay(
                image, target, is_best, x1, y1, x2, y2, mask_resized)

        # Bounding box: yellow for the selected goal, green for other valid
        # dates.  It is deliberately outline-only.
        box_color = (0, 255, 255, 255) if is_best else (40, 230, 40, 255)
        cv2.rectangle(
            image, (x1, y1), (x2, y2), box_color,
            self._thick(3 if is_best else 2), cv2.LINE_AA)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if self.clean_harvest_overlay:
            marker_size = self._radius(12 if is_best else 8)
            cv2.drawMarker(
                image, (cx, cy), box_color, cv2.MARKER_CROSS,
                marker_size, self._thick(2), cv2.LINE_AA)
            self._draw_external_date_label(
                image, target, x1, y1, x2, y2, is_best, box_color)
            return

        # Detailed diagnostic overlay retained for non-harvesting consumers.
        self._draw_detailed_target_annotations(
            image, target, cx, cy, x1, y1, x2, y2, is_best, idx)

    def _draw_target_mask_overlay(
        self, image, target, is_best, x1, y1, x2, y2, mask_resized,
    ) -> None:
        """Draw the legacy diagnostic mask/heatmap overlay."""
        try:
            h_roi, w_roi = mask_resized.shape
            colored_mask = np.zeros((h_roi, w_roi, 4), dtype=np.uint8)
            colored_mask[:, :, 1] = mask_resized  # green
            colored_mask[:, :, 3] = mask_resized  # alpha-like

            roi = image[y1:y2, x1:x2]
            blended = cv2.addWeighted(colored_mask, 0.45, roi, 0.55, 0.0)
            image[y1:y2, x1:x2] = blended

            # LiDAR depth dot overlay
            lidar_depth_uv = target.get("lidar_depth_uv")
            lidar_depths = target.get("lidar_depths")
            if lidar_depth_uv is not None and lidar_depths is not None and len(lidar_depths) > 0:
                z_near = float(np.min(lidar_depths))
                z_far = float(np.max(lidar_depths))
                z_range = max(z_far - z_near, 0.05)
                for (ux, uy), z in zip(lidar_depth_uv, lidar_depths):
                    px = int(round(ux)) + x1
                    py = int(round(uy)) + y1
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        t = float(np.clip((z - z_near) / z_range, 0.0, 1.0))
                        r = int(255 * max(0.0, 1.0 - 2 * t))
                        g = int(255 * (1.0 - abs(2 * t - 1.0)))
                        b = int(255 * max(0.0, 2 * t - 1.0))
                        cv2.circle(image, (px, py), self._radius(4), (b, g, r, 255), -1)

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

    def _draw_detailed_target_annotations(
        self, image, target, cx, cy, x1, y1, x2, y2, is_best, idx,
    ) -> None:
        """Draw the original detailed target diagnostics."""
        candidate_rank = target.get("candidate_rank")
        if is_best:
            color = (255, 0, 0, 255)
            radius = self._radius(7)
        elif candidate_rank is not None:
            color = (0, 255, 255, 255)
            radius = self._radius(6)
        else:
            color = (0, 0, 255, 255)
            radius = self._radius(5)

        cv2.circle(image, (cx, cy), radius, color, -1)

        if candidate_rank is not None:
            rank_color = (255, 0, 0, 255) if is_best else (0, 255, 255, 255)
            cv2.putText(
                image,
                f"#{candidate_rank}",
                (cx - round(self._len(15)), cy - round(self._len(15))),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font(0.6),
                rank_color,
                self._thick(2),
                cv2.LINE_AA,
            )

        if is_best:
            self._draw_approach_arrows(image, target, cx, cy, x1, y1, w_roi, h_roi)
            self.draw_fingertip_contacts(image, target)

        self._draw_target_labels(image, target, cx, cy, is_best, idx)

    def _draw_external_date_label(
        self, image, target, x1, y1, x2, y2, is_best, color,
    ) -> None:
        """Place a compact label outside the date box, never over the fruit."""
        text = "SELECTED" if is_best else "DATE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._font(0.55)
        thickness = self._thick(2)
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, scale, thickness)
        pad = self._radius(4)
        image_h, image_w = image.shape[:2]
        label_x = max(0, min(x1, image_w - text_w - 2 * pad))
        if y1 >= text_h + baseline + 3 * pad:
            bg_y2 = y1 - self._radius(3)
            bg_y1 = bg_y2 - text_h - baseline - 2 * pad
        else:
            bg_y1 = min(image_h - text_h - baseline - 2 * pad, y2 + self._radius(3))
            bg_y2 = bg_y1 + text_h + baseline + 2 * pad
        bg_y1 = max(0, bg_y1)
        bg_y2 = min(image_h - 1, bg_y2)
        cv2.rectangle(
            image, (label_x, bg_y1),
            (label_x + text_w + 2 * pad, bg_y2),
            (20, 20, 20, 255), -1)
        cv2.rectangle(
            image, (label_x, bg_y1),
            (label_x + text_w + 2 * pad, bg_y2),
            color, self._thick(1), cv2.LINE_AA)
        text_y = min(bg_y2 - pad - baseline, image_h - baseline - 1)
        cv2.putText(
            image, text, (label_x + pad, text_y), font, scale,
            color, thickness, cv2.LINE_AA)

    def draw_fingertip_contacts(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
    ) -> None:
        """Draw the predicted three-finger contact patches for the best date."""
        if not SHOW_FINGER_CONTACTS:
            return
        contacts = target.get("finger_contacts")
        if not contacts:
            return
        points = np.asarray(contacts.get("points_px", []), dtype=np.float32)
        center = contacts.get("center_px")
        if points.shape != (3, 2) or center is None:
            return

        x1, y1, _, y2 = target["bb"]
        absolute = [
            (int(round(x1 + float(point[0]))), int(round(y1 + float(point[1]))))
            for point in points
        ]
        center_abs = (
            int(round(x1 + float(center[0]))),
            int(round(y1 + float(center[1]))),
        )
        patch_radius = max(
            self._radius(4),
            int(round(float(contacts.get("patch_radius_px", 4.0)))),
        )
        colors = [
            (0, 255, 255, 255),   # F1 yellow
            (255, 0, 255, 255),   # F2 magenta
            (255, 255, 0, 255),   # F3 cyan
        ]
        valid = bool(contacts.get("valid", False))
        outline = (0, 255, 0, 255) if valid else (0, 80, 255, 255)

        cv2.polylines(
            image, [np.asarray(absolute, dtype=np.int32).reshape((-1, 1, 2))],
            isClosed=True, color=outline, thickness=self._thick(2),
            lineType=cv2.LINE_AA,
        )
        for idx, (point, color) in enumerate(zip(absolute, colors), start=1):
            cv2.circle(image, point, patch_radius, color, self._thick(2), cv2.LINE_AA)
            cv2.circle(image, point, self._radius(3), color, -1, cv2.LINE_AA)
            cv2.putText(
                image, f"F{idx}",
                (point[0] + self._radius(5), point[1] - self._radius(5)),
                cv2.FONT_HERSHEY_SIMPLEX, self._font(0.45), color,
                self._thick(1), cv2.LINE_AA,
            )
        cv2.drawMarker(
            image, center_abs, outline, markerType=cv2.MARKER_CROSS,
            markerSize=self._radius(14), thickness=self._thick(2),
            line_type=cv2.LINE_AA,
        )

        score = float(contacts.get("score", 0.0))
        state = "CONTACT OK" if valid else "CONTACT LOW"
        label_y = min(image.shape[0] - self._radius(8), y2 + self._radius(20))
        cv2.putText(
            image, f"{state} {score:.0%}",
            (max(2, x1), max(self._radius(18), label_y)),
            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55), outline,
            self._thick(2), cv2.LINE_AA,
        )

    def draw_actual_fingertip_verification(
        self,
        source: np.ndarray,
        image: np.ndarray,
        target: Optional[Dict[str, Any]],
    ) -> None:
        """Detect the three physical red fingertips and verify date containment."""
        if target is None or target.get("mask_resized") is None:
            self.last_fingertip_verification = "NO_TARGET"
            return

        bgr = source[:, :, :3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 105, 65), (12, 255, 255))
        red |= cv2.inRange(hsv, (168, 105, 65), (179, 255, 255))
        red = cv2.morphologyEx(
            red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        red = cv2.morphologyEx(
            red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        x1, y1, x2, y2 = target["bb"]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad_x, pad_y = int(1.2 * bw), int(1.2 * bh)
        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        rx2 = min(image.shape[1], x2 + pad_x)
        ry2 = min(image.shape[0], y2 + pad_y)
        roi = red[ry1:ry2, rx1:rx2]
        count, _, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
        min_area = max(12, int(0.0015 * bw * bh))
        max_area = max(min_area + 1, int(0.45 * bw * bh))
        blobs = []
        fruit_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue
            cx = float(centroids[idx, 0] + rx1)
            cy = float(centroids[idx, 1] + ry1)
            distance = float(np.linalg.norm(np.array([cx, cy]) - fruit_center))
            blobs.append((distance, -area, (cx, cy), area))
        blobs.sort()
        points = [entry[2] for entry in blobs[:3]]

        if len(points) == 2:
            p1 = np.asarray(points[0], dtype=np.float32)
            p2 = np.asarray(points[1], dtype=np.float32)
            segment = p2 - p1
            span = float(np.linalg.norm(segment))
            if span > 1.0:
                along = float(np.dot(fruit_center - p1, segment) / (span * span))
                closest = p1 + np.clip(along, 0.0, 1.0) * segment
                cross_error = float(np.linalg.norm(fruit_center - closest))
            else:
                along = -1.0
                cross_error = float("inf")
            span_ok = 0.50 * min(bw, bh) <= span <= 3.5 * max(bw, bh)
            between = 0.15 <= along <= 0.85
            centered = cross_error <= 0.35 * max(span, 1.0)
            possible_fit = span_ok and between and centered
            state = "POSSIBLE_FIT" if possible_fit else "NOT_FIT"
            self.last_fingertip_verification = (
                f"{state} tips=2 between={between} "
                f"along={along:.2f} cross={cross_error:.1f}px span={span:.1f}px")
            color = (0, 255, 255, 255) if possible_fit else (0, 165, 255, 255)
            p1i = tuple(np.round(p1).astype(int))
            p2i = tuple(np.round(p2).astype(int))
            cv2.line(image, p1i, p2i, color, self._thick(3), cv2.LINE_AA)
            for index, point in enumerate((p1i, p2i), 1):
                cv2.circle(image, point, self._radius(9), (255, 255, 0, 255),
                           self._thick(3), cv2.LINE_AA)
                if not self.clean_harvest_overlay:
                    cv2.putText(image, f"R{index}", (point[0] + 5, point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, self._font(0.45),
                                (255, 255, 0, 255), self._thick(1), cv2.LINE_AA)
            if not self.clean_harvest_overlay:
                cv2.putText(image, self.last_fingertip_verification,
                            (max(5, x1), max(24, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            self._font(0.58), color, self._thick(2), cv2.LINE_AA)
            return

        if len(points) != 3:
            self.last_fingertip_verification = f"NEED_2_TIPS detected={len(points)}"
            if not self.clean_harvest_overlay:
                cv2.putText(
                    image, self.last_fingertip_verification,
                    (max(5, x1), max(24, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    self._font(0.62), (0, 165, 255, 255), self._thick(2), cv2.LINE_AA)
            return

        triangle = np.asarray(points, dtype=np.float32)
        hull = cv2.convexHull(triangle).reshape(-1, 2)
        triangle_area = abs(float(cv2.contourArea(hull)))
        center_inside = cv2.pointPolygonTest(
            hull, (float(fruit_center[0]), float(fruit_center[1])), False) >= 0

        mask = target["mask_resized"] > 0
        ys, xs = np.nonzero(mask)
        if xs.size:
            sample_stride = max(1, xs.size // 1500)
            samples = np.column_stack((xs[::sample_stride] + x1,
                                       ys[::sample_stride] + y1))
            inside = sum(
                cv2.pointPolygonTest(hull, (float(px), float(py)), False) >= 0
                for px, py in samples)
            containment = inside / max(1, len(samples))
        else:
            containment = 0.0
        area_ok = triangle_area >= 0.12 * bw * bh
        fit = center_inside and area_ok and containment >= 0.50
        tri_center = triangle.mean(axis=0)
        dx = float(fruit_center[0] - tri_center[0])
        dy = float(fruit_center[1] - tri_center[1])
        state = "FIT" if fit else "NOT_FIT"
        self.last_fingertip_verification = (
            f"{state} tips=3 containment={containment:.2f} "
            f"dx={dx:+.1f}px dy={dy:+.1f}px")

        color = (0, 255, 0, 255) if fit else (0, 165, 255, 255)
        cv2.polylines(image, [hull.astype(np.int32).reshape(-1, 1, 2)],
                      True, color, self._thick(3), cv2.LINE_AA)
        for index, point in enumerate(points, 1):
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(image, center, self._radius(9), (255, 255, 0, 255),
                       self._thick(3), cv2.LINE_AA)
            if not self.clean_harvest_overlay:
                cv2.putText(image, f"R{index}", (center[0] + 5, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.45),
                            (255, 255, 0, 255), self._thick(1), cv2.LINE_AA)
        if not self.clean_harvest_overlay:
            cv2.putText(image, self.last_fingertip_verification,
                        (max(5, x1), max(24, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        self._font(0.58), color, self._thick(2), cv2.LINE_AA)

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

            L = self._len(60)
            ax2 = int(cx + vx * L)
            ay2 = int(cy + vy * L)
            ax1 = int(cx - vx * L)
            ay1 = int(cy - vy * L)

            axis_color = (0, 255, 255, 255)
            cv2.arrowedLine(image, (cx, cy), (ax2, ay2), axis_color, self._thick(2), tipLength=0.25)
            cv2.line(image, (cx, cy), (ax1, ay1), axis_color, self._thick(2))

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

            L_out = max(w_roi, h_roi) + self._len(10)
            start_x = int(round(dest_x - dx * L_out))
            start_y = int(round(dest_y - dy * L_out))
            start_x = max(0, min(start_x, image.shape[1] - 1))
            start_y = max(0, min(start_y, image.shape[0] - 1))
            cv2.arrowedLine(image, (start_x, start_y), (dest_x, dest_y),
                            (0, 165, 255, 255), self._thick(2), tipLength=0.25)

        surface_normal = target.get("surface_normal")
        if surface_normal is not None:
            Zc = target["Zc"]
            if Zc > 0:
                fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
                sx, sy = self.image_scale
                du = int(fx * surface_normal[0] / Zc * 0.06 * sx * self._s)
                dv = int(fy * surface_normal[1] / Zc * 0.06 * sy * self._s)
                cv2.arrowedLine(image, (cx, cy), (cx + du, cy + dv),
                                (255, 255, 0, 255), self._thick(1), tipLength=0.3)

        approach_dir_cam = target.get("approach_dir_cam")
        if approach_dir_cam is not None:
            Zc = target["Zc"]
            if Zc > 0:
                fx, fy = self.intrinsics["fx"], self.intrinsics["fy"]
                sx, sy = self.image_scale
                arrow_len_3d = 0.10
                du = int(fx * approach_dir_cam[0] / Zc * arrow_len_3d * sx * self._s)
                dv = int(fy * approach_dir_cam[1] / Zc * arrow_len_3d * sy * self._s)
                cv2.arrowedLine(image, (cx, cy), (cx + du, cy + dv),
                                (255, 0, 255, 255), self._thick(3), tipLength=0.25)

                clearance = target.get("clearance", float('inf'))
                is_clear = target.get("is_collision_free", True)
                if clearance < float('inf'):
                    clr_color = (0, 255, 0, 255) if is_clear else (0, 0, 255, 255)
                    clr_text = f"CLR {clearance*100:.0f}cm" if is_clear else "BLOCK"
                    cv2.putText(
                        image,
                        clr_text,
                        (target["bb"][0], target["bb"][1] - round(self._len(10))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self._font(0.4),
                        clr_color,
                        self._thick(2) if not is_clear else self._thick(1),
                        cv2.LINE_AA,
                    )

    def draw_gap_debug(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
    ) -> None:
        """Draw branch-ring samples and the selected clear-gap direction."""
        if not self.show_gap_debug:
            return

        debug = target.get("_ring_debug")
        if not debug:
            return

        cx = int(round(debug["cx"]))
        cy = int(round(debug["cy"]))
        ring_r = float(debug["r"])
        blocked = debug["blocked"]
        n_samples = int(debug["n_samples"])

        cv2.circle(
            image, (cx, cy), int(round(ring_r)),
            (180, 180, 180, 255), self._thick(1), cv2.LINE_AA,
        )
        for i, is_blocked in enumerate(blocked):
            angle = i * (2.0 * math.pi / n_samples)
            px = int(round(cx + ring_r * math.cos(angle)))
            py = int(round(cy + ring_r * math.sin(angle)))
            color = (0, 0, 255, 255) if is_blocked else (0, 255, 0, 255)
            cv2.circle(image, (px, py), self._radius(4), color, -1, cv2.LINE_AA)

        detected = bool(debug.get("detected", False))
        if detected:
            gap_angle = float(debug["gap_angle"])
            arrow_len = ring_r + self._len(35)
            end = (
                int(round(cx + arrow_len * math.cos(gap_angle))),
                int(round(cy + arrow_len * math.sin(gap_angle))),
            )
            cv2.arrowedLine(
                image, (cx, cy), end, (255, 255, 0, 255),
                self._thick(3), cv2.LINE_AA, tipLength=0.2,
            )
            label = f"GAP {math.degrees(gap_angle):.0f} deg | 2-FINGER"
            label_color = (255, 255, 0, 255)
        else:
            label = "NO USABLE GAP | 3-FINGER"
            label_color = (220, 220, 220, 255)

        cv2.putText(
            image, label,
            (max(4, cx - int(ring_r)), max(self._radius(18), cy - int(ring_r) - self._radius(8))),
            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55), label_color,
            self._thick(2), cv2.LINE_AA,
        )

    def _draw_target_labels(
        self,
        image: np.ndarray,
        target: Dict[str, Any],
        cx: int, cy: int,
        is_best: bool,
        idx: int = -1
    ) -> None:
        """Draw labels and stats for a target."""
        if idx >= 0:
            x1, y1 = target["bb"][0], target["bb"][1]
            cv2.putText(image, str(idx), (x1 + round(self._len(4)), y1 + round(self._len(40))),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(1.6), (255, 255, 0, 255),
                        self._thick(3), cv2.LINE_AA)
        if is_best:
            cv2.putText(
                image,
                "BEST",
                (cx + round(self._len(10)), cy - round(self._len(10))),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font(1.8),
                (255, 0, 0, 255),
                self._thick(3),
                cv2.LINE_AA,
            )
        dist_grip = target.get("dist", 0.0)
        total_score = target.get("score", 0.0)
        color = (0, 255, 0, 255) if is_best else (200, 200, 200, 255)
        cv2.putText(
            image,
            f"D:{dist_grip:.2f}m S:{total_score:.2f}",
            (cx + round(self._len(10)), cy + round(self._len(40))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._font(1.3),
            color,
            self._thick(3),
            cv2.LINE_AA,
        )

    def draw_hud(self, image: np.ndarray, net_fps: float, loop_fps: float) -> None:
        """Draw heads-up display with FPS info."""
        cv2.putText(
            image,
            f"YOLO FPS: {net_fps:.1f}",
            (12, round(self._len(55))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._font(1.7),
            (0, 255, 0, 255),
            self._thick(3),
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Loop FPS: {loop_fps:.1f}",
            (12, round(self._len(110))),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._font(1.7),
            (0, 200, 255, 255),
            self._thick(3),
            cv2.LINE_AA,
        )

    def draw_viz_only(self, image: np.ndarray, viz_only: List[Dict[str, Any]]) -> None:
        """Draw visualization-only detections.
        trunk  → orange bounding box
        bunch  → purple semi-transparent mask fill + contour outline
        """
        TRUNK_COLOR = (0, 165, 255, 255)
        BUNCH_COLOR = (220, 0, 255, 255)

        for v in viz_only:
            x1, y1, x2, y2 = v["bb"]
            cls = v.get("class", "?")
            conf = v.get("conf", 0.0)
            cls_lower = str(cls).lower()

            if cls_lower == "bunch":
                polygon = v.get("polygon")
                if polygon is not None and len(polygon) > 2:
                    try:
                        pts = polygon.reshape((-1, 1, 2))
                        overlay = image.copy()
                        cv2.fillPoly(overlay, [pts], BUNCH_COLOR)
                        cv2.addWeighted(overlay, 0.01, image, 0.8, 0, image)
                        cv2.polylines(image, [pts], isClosed=True, color=BUNCH_COLOR,
                                      thickness=self._thick(2))
                    except Exception:
                        cv2.rectangle(image, (x1, y1), (x2, y2), BUNCH_COLOR, self._thick(2))
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), BUNCH_COLOR, self._thick(2))
                cv2.putText(image, f"bunch {conf:.0%}",
                            (x1 + round(self._len(4)), y1 + round(self._len(18))),
                            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55),
                            BUNCH_COLOR, self._thick(2), cv2.LINE_AA)
            else:
                color = (
                    (40, 230, 40, 255)
                    if "date" in cls_lower else TRUNK_COLOR
                )
                polygon = v.get("polygon")
                if polygon is not None and len(polygon) > 2:
                    cv2.polylines(
                        image, [polygon.reshape((-1, 1, 2))], True,
                        color, self._thick(2), cv2.LINE_AA)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, self._thick(2))
                cv2.putText(image, f"{cls} {conf:.0%}",
                            (x1 + round(self._len(4)), y1 + round(self._len(18))),
                            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55),
                            color, self._thick(2), cv2.LINE_AA)

    def draw_lidar_points(
        self,
        image: np.ndarray,
        uv: np.ndarray,
        pts_cam: np.ndarray,
        z_min: float = 0.1,
        z_max: float = 5.0,
    ) -> None:
        """Draw projected LiDAR points colored by depth (red=close, blue=far)."""
        if uv is None or uv.shape[0] == 0:
            return
        h, w = image.shape[:2]
        zs = pts_cam[:, 2]
        t = np.clip((zs - z_min) / max(z_max - z_min, 1e-3), 0.0, 1.0)
        r = np.clip(255 * (1.0 - 2 * t),        0, 255).astype(np.uint8)
        g = np.clip(255 * (1.0 - abs(2*t-1.0)), 0, 255).astype(np.uint8)
        b = np.clip(255 * (2 * t - 1.0),         0, 255).astype(np.uint8)
        xs = uv[:, 0].astype(np.int32)
        ys = uv[:, 1].astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        dot_r = self._radius(2)
        for i in np.where(valid)[0]:
            cv2.circle(image, (xs[i], ys[i]), dot_r, (int(b[i]), int(g[i]), int(r[i]), 255), -1)

    def render_frame(
        self,
        image: np.ndarray,
        targets: List[Dict[str, Any]],
        rejected_targets: List[Dict[str, Any]],
        best_idx: Optional[int],
        net_fps: float,
        loop_fps: float,
        viz_only: Optional[List[Dict[str, Any]]] = None,
        lidar_uv: Optional[np.ndarray] = None,
        lidar_pts_cam: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render a complete frame with all visualizations."""
        if SKIP_DRAW:
            return image

        # Set active draw scale so all helper methods scale proportionally.
        self._s = self.display_scale

        # Resize first so all drawing happens on the smaller display image.
        if self.display_scale != 1.0:
            image = cv2.resize(
                image,
                (int(image.shape[1] * self.display_scale), int(image.shape[0] * self.display_scale)),
                interpolation=cv2.INTER_AREA,
            )
            s = self.display_scale
            def _scale_target(t):
                t = dict(t)
                x1, y1, x2, y2 = t["bb"]
                nx1, ny1, nx2, ny2 = int(x1*s), int(y1*s), int(x2*s), int(y2*s)
                t["bb"] = (nx1, ny1, nx2, ny2)
                if t.get("mask_resized") is not None:
                    new_w = max(1, nx2 - nx1)
                    new_h = max(1, ny2 - ny1)
                    t["mask_resized"] = cv2.resize(
                        t["mask_resized"], (new_w, new_h), interpolation=cv2.INTER_NEAREST
                    )
                if t.get("polygon") is not None:
                    t["polygon"] = (t["polygon"] * s).astype(np.int32)
                if t.get("_ring_debug") is not None:
                    debug = dict(t["_ring_debug"])
                    debug["cx"] *= s
                    debug["cy"] *= s
                    debug["r"] *= s
                    t["_ring_debug"] = debug
                if t.get("finger_contacts") is not None:
                    contacts = dict(t["finger_contacts"])
                    if contacts.get("center_px") is not None:
                        contacts["center_px"] = np.asarray(
                            contacts["center_px"], dtype=np.float32) * s
                    contacts["points_px"] = np.asarray(
                        contacts.get("points_px", []), dtype=np.float32) * s
                    contacts["patch_radius_px"] = float(
                        contacts.get("patch_radius_px", 0.0)) * s
                    t["finger_contacts"] = contacts
                return t
            targets = [_scale_target(t) for t in targets]
            rejected_targets = [_scale_target(t) for t in rejected_targets]
            if viz_only:
                viz_only = [_scale_target(t) for t in viz_only]

        verification_source = image.copy()

        if lidar_uv is not None and lidar_pts_cam is not None:
            self.draw_lidar_points(image, lidar_uv * self.display_scale, lidar_pts_cam)
        self.draw_classification_zones(image, viz_only)
        self.draw_rejected_targets(image, rejected_targets)
        if viz_only:
            self.draw_viz_only(image, viz_only)
        self.draw_blocking_fruits(image, targets, best_idx)

        draw_indices = set(range(len(targets)))
        if not DRAW_ONLY_BEST and DRAW_TOP_N is not None and DRAW_TOP_N > 0:
            scored = sorted(
                ((i, t.get("score", 0.0)) for i, t in enumerate(targets)),
                key=lambda item: item[1],
                reverse=True,
            )
            draw_indices = {i for i, _ in scored[:DRAW_TOP_N]}
            if best_idx is not None:
                draw_indices.add(best_idx)

        for i, t in enumerate(targets):
            if DRAW_ONLY_BEST and (best_idx is not None) and i != best_idx:
                continue
            if not DRAW_ONLY_BEST and i not in draw_indices:
                continue
            self.draw_target(image, t, is_best=(i == best_idx), idx=i)
            if i == best_idx:
                self.draw_gap_debug(image, t)

        verification_target = (
            targets[best_idx]
            if best_idx is not None and 0 <= best_idx < len(targets) else None)
        self.draw_actual_fingertip_verification(
            verification_source, image, verification_target)

        self.draw_hud(image, net_fps, loop_fps)

        return image
