"""Track of the one date being picked: position, velocity, uncertainty.

Constant-velocity Kalman filter in base_link. Each step is stamped with the
photo's acquisition time (observations are already transformed with the arm
pose at that instant), so camera motion is not mistaken for fruit motion and
velocity is the fruit's own -- e.g. sway in wind.

Association matches detections against the PREDICTED position, scaled by the
current uncertainty (Mahalanobis distance), plus image evidence published by
vision (physical size, colour, overlap with neighbours). When two detections
are about equally plausible, or the match overlaps a neighbour in the image,
the track lowers its confidence instead of switching to a neighbouring date.

No ROS dependency; numpy only.
"""

import math

import numpy as np

# 99.9% gate for a 3-DOF Mahalanobis distance.
GATE_D2 = 16.27


class SelectedFruitTrack:
    def __init__(self, position, stamp_s, pos_std=0.006, vel_std=0.05,
                 accel_std=0.3, max_gate_m=0.025, ambiguity_margin=4.0,
                 size_tol_mm=3.0, color_tol=12.0, overlap_iou=0.2,
                 appearance_gate=9.0):
        # pos_std: the planned target's own error (perception error ~4.5mm
        # median in the lab, 2026-10-27). accel_std: process noise for sway.
        # ponytail: fixed noise model; fit accel_std from the FINAL timelines
        # (outdoor sway) once there is data.
        self.x = np.array(list(position[:3]) + [0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([pos_std ** 2] * 3 + [vel_std ** 2] * 3)
        self.t = float(stamp_s)
        self.accel_std = accel_std
        self.max_gate_m = max_gate_m
        self.ambiguity_margin = ambiguity_margin
        self.size_tol_mm = size_tol_mm
        self.color_tol = color_tol
        self.overlap_iou = overlap_iou
        # A detection more than ~3 tolerances off in size or colour is not
        # this date, however close: the neighbour left alone in the gate when
        # the date itself is hidden.
        self.appearance_gate = appearance_gate
        self.features = None        # size/colour reference, from matches
        self.confidence = 1.0
        self.counts = {"updated": 0, "ambiguous": 0, "overlap": 0, "missed": 0}

    # -- Kalman ----------------------------------------------------------
    def _transition(self, dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.accel_std ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * q * dt ** 4 / 4.0
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * q * dt ** 3 / 2.0
        Q[3:, 3:] = np.eye(3) * q * dt ** 2
        return F, Q

    def predict(self, stamp_s):
        """(x, P) at stamp_s, without changing the track."""
        dt = max(0.0, float(stamp_s) - self.t)
        F, Q = self._transition(dt)
        return F @ self.x, F @ self.P @ F.T + Q

    def _advance(self, stamp_s):
        self.x, self.P = self.predict(stamp_s)
        self.t = max(self.t, float(stamp_s))

    @staticmethod
    def measurement_std(z_std):
        """Isotropic position noise from the detection's depth spread."""
        return min(0.012, 0.003 + 0.25 * max(0.0, float(z_std)))

    def _update(self, z, r_std):
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        S = H @ self.P @ H.T + np.eye(3) * r_std ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, dtype=float) - H @ self.x)
        self.P = (np.eye(6) - K @ H) @ self.P

    # -- association -----------------------------------------------------
    def _appearance_cost(self, feat):
        # NaN components (unknown to vision) contribute nothing.
        if self.features is None or feat is None:
            return 0.0
        ref, cost = self.features, 0.0
        if math.isfinite(feat[0]) and math.isfinite(ref[0]):
            cost += ((feat[0] - ref[0]) / self.size_tol_mm) ** 2
        if all(math.isfinite(v) for v in feat[1:4]) and np.all(np.isfinite(ref[1:4])):
            cost += (math.dist(feat[1:4], ref[1:4]) / self.color_tol) ** 2
        return cost

    def step(self, stamp_s, poses, z_stds, features=None, usable=None):
        """Advance to stamp_s and associate one frame of detections.

        poses: [(x, y, z)], z_stds: depth spread per detection (m),
        features: optional [diam_mm, L, a, b, overlap_iou] per detection,
        usable: optional bools (quality gate) per detection.
        Returns (index or None, status) where status is one of
        "updated", "overlap" (updated with reduced weight), "ambiguous",
        "missed".
        """
        self._advance(stamp_s)
        xp, P = self.x, self.P
        cands = []
        for i, p in enumerate(poses):
            if usable is not None and not usable[i]:
                continue
            r_std = self.measurement_std(z_stds[i])
            S = P[:3, :3] + np.eye(3) * r_std ** 2
            e = np.asarray(p[:3], dtype=float) - xp[:3]
            d2 = float(e @ np.linalg.solve(S, e))
            if d2 > GATE_D2 or float(np.linalg.norm(e)) > self.max_gate_m:
                continue
            feat = features[i] if features else None
            app = self._appearance_cost(feat)
            if app > self.appearance_gate:
                continue
            cands.append((d2 + app, i, r_std, feat))
        if not cands:
            self.counts["missed"] += 1
            self.confidence *= 0.8
            return None, "missed"
        cands.sort()
        if (len(cands) > 1
                and cands[1][0] - cands[0][0] < self.ambiguity_margin):
            # Two plausible dates: hold course on the prediction rather than
            # risk locking onto the neighbour.
            self.counts["ambiguous"] += 1
            self.confidence *= 0.7
            return None, "ambiguous"
        _, i, r_std, feat = cands[0]
        status = "updated"
        if feat is not None and feat[4] > self.overlap_iou:
            # Overlaps a neighbour in the image: its mask, depth and position
            # may be partly the neighbour's. Trust it less.
            r_std *= 2.0
            status = "overlap"
            self.counts["overlap"] += 1
            self.confidence *= 0.85
        else:
            self.counts["updated"] += 1
            self.confidence = min(1.0, self.confidence + 0.2)
        self._update(poses[i][:3], r_std)
        if feat is not None and status == "updated":
            f = np.asarray(feat[:4], dtype=float)
            if self.features is None:
                self.features = f
            else:
                old = self.features
                self.features = np.where(
                    np.isfinite(f),
                    np.where(np.isfinite(old), 0.7 * old + 0.3 * f, f), old)
        return i, status

    # -- outputs ---------------------------------------------------------
    def state(self, stamp_s=None):
        """Position, velocity and their 1-sigma, at stamp_s (default: last)."""
        x, P = (self.x, self.P) if stamp_s is None else self.predict(stamp_s)
        return {
            "position": x[:3].tolist(),
            "velocity": x[3:].tolist(),
            "pos_std_m": float(math.sqrt(max(0.0, np.trace(P[:3, :3]) / 3.0))),
            "vel_std_mps": float(math.sqrt(max(0.0, np.trace(P[3:, 3:]) / 3.0))),
            "confidence": float(self.confidence),
        }


if __name__ == "__main__":
    # A date swaying at 4 cm/s along x, a same-looking neighbour 30 mm away and
    # a differently coloured one touching it (22 mm), 10 Hz photos.
    rng = np.random.default_rng(0)
    trk = SelectedFruitTrack([0.0, 0.0, 0.0], 0.0)
    for k in range(1, 21):
        t = k * 0.1
        true = np.array([0.04 * t, 0.0, 0.0])
        poses = [true + rng.normal(0, 0.002, 3), true + [0.0, 0.030, 0.0],
                 true + [0.0, -0.022, 0.0]]
        feats = [[22.0, 50.0, 20.0, 40.0, 0.0], [22.0, 50.0, 20.0, 40.0, 0.0],
                 [22.0, 80.0, 0.0, 10.0, 0.0]]
        idx, status = trk.step(t, poses, [0.004] * 3, feats)
        assert idx == 0, (k, idx, status)
    s = trk.state()
    assert abs(s["velocity"][0] - 0.04) < 0.01, s
    assert s["pos_std_m"] < 0.004, s
    # Date hidden; only the differently coloured neighbour is near: not it.
    p = np.array(trk.predict(2.05)[0][:3])
    other = [22.0, 80.0, 0.0, 10.0, 0.0]
    assert trk.step(2.05, [p + [0, -0.012, 0]], [0.004], [other]) == (None, "missed")
    # Date hidden; two identical-looking candidates equally far either side:
    # refuse to pick one, and coast on the prediction.
    p = np.array(trk.predict(2.1)[0][:3])
    same = [22.0, 50.0, 20.0, 40.0, 0.0]
    idx, status = trk.step(2.1, [p + [0, 0.008, 0], p - [0, 0.008, 0]],
                           [0.004] * 2, [same, same])
    assert (idx, status) == (None, "ambiguous"), (idx, status)
    # Nothing in the gate: missed, and position uncertainty grows.
    before = trk.state()["pos_std_m"]
    assert trk.step(2.4, [p + [0, 0.2, 0]], [0.004])[1] == "missed"
    assert trk.state()["pos_std_m"] > before
    print("ok", {k: [round(v, 4) for v in s[k]] if isinstance(s[k], list)
                 else round(s[k], 4) for k in s})
