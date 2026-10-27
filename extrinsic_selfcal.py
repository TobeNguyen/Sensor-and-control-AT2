# Camera -> robot extrinsic self-calibration from the SURVEY spin.
# Robot rotates in place (stop-go, still while observing); tags at UNKNOWN fixed positions.
# Unknowns: camera forward offset, camera yaw, and every tag position (lateral offset fixed, measured).
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class ExtrinsicSelfCal:
    def __init__(self, fwd_nominal: float, lat_fixed: float = 0.0, max_range: float = 2.5):
        self.fwd_nom, self.lat, self.max_range = float(fwd_nominal), float(lat_fixed), max_range
        self.obs: list = []            # (x_est (3,), MarkerMeasurement) measured with NOMINAL extrinsics

    def add(self, x_est, markers) -> None:
        for m in markers:
            if 0.0 < m.range <= self.max_range:
                self.obs.append((np.array(x_est, float), m))

    # Camera-frame planar point [Z, -X] of a nominal measurement (nominal: lat 0, yaw 0)
    def _q(self, m):
        return np.array([m.range * math.cos(m.bearing) - self.fwd_nom, m.range * math.sin(m.bearing)])

    def solve(self, min_obs: int = 8, min_spread: float = math.radians(25)):
        # keep only tags seen across a wide part of the image: parallax carries the information
        by_id: dict = {}
        for x, m in self.obs:
            by_id.setdefault(int(m.marker_id), []).append((x, m))
        keep = [i for i, lst in by_id.items()
                if len(lst) >= min_obs and np.ptp([math.atan2(*self._q(m)[::-1]) for _, m in lst]) >= min_spread]
        if not keep:
            return None
        rows = [(keep.index(i), x, self._q(m)) for i in keep for x, m in by_id[i]]
        k = np.array([r[0] for r in rows])
        P = np.array([r[1][:2] for r in rows]); TH = np.array([r[1][2] for r in rows])
        Q = np.array([r[2] for r in rows])
        c, s = np.cos(TH), np.sin(TH)
        r_meas, b_meas = np.hypot(Q[:, 0], Q[:, 1]), np.arctan2(Q[:, 1], Q[:, 0])
        sig_r = 0.01 + 0.01 * r_meas ** 2          # camera = angle sensor: range noise grows with Z^2
        sig_b = 0.01

        # initial tag positions from the nominal extrinsics
        rx, ry = Q[:, 0] + self.fwd_nom, Q[:, 1]
        W = np.column_stack([P[:, 0] + c * rx - s * ry, P[:, 1] + s * rx + c * ry])
        m0 = np.array([W[k == j].mean(0) for j in range(len(keep))])

        def resid(p):
            fwd, yaw, M = p[0], p[1], p[2:].reshape(-1, 2)[k]
            dx, dy = M[:, 0] - P[:, 0], M[:, 1] - P[:, 1]
            ax = c * dx + s * dy - fwd                  # tag in the robot frame, minus camera position
            ay = -s * dx + c * dy - self.lat
            cy, sy = math.cos(yaw), math.sin(yaw)
            qx, qy = cy * ax + sy * ay, -sy * ax + cy * ay   # rotate into the camera frame
            return np.concatenate([(np.hypot(qx, qy) - r_meas) / sig_r,
                                   _wrap(np.arctan2(qy, qx) - b_meas) / sig_b])

        sol = least_squares(resid, np.concatenate([[self.fwd_nom, 0.0], m0.ravel()]),
                            loss="huber", f_scale=2.0)
        cov = np.linalg.pinv(sol.jac.T @ sol.jac)
        fwd, yaw = float(sol.x[0]), float(sol.x[1])
        return {"fwd": fwd, "lat": self.lat, "yaw": yaw,
                "std_fwd": float(math.sqrt(cov[0, 0])), "std_yaw": float(math.sqrt(cov[1, 1])),
                "n_obs": len(rows), "tags": keep,
                "rms": float(np.sqrt(np.mean(sol.fun ** 2)))}   # ~1 if the noise model is right

    # Re-express every stored observation with the calibrated extrinsics (for rebuilding the TagMap)
    def corrected(self, fwd: float, yaw: float):
        cy, sy = math.cos(yaw), math.sin(yaw)
        for x, m in self.obs:
            q = self._q(m)
            r = np.array([cy * q[0] - sy * q[1] + fwd, sy * q[0] + cy * q[1] + self.lat])
            yield x, replace(m, range=float(np.hypot(*r)), bearing=float(math.atan2(r[1], r[0])),
                             view_angle=float(_wrap(m.view_angle + yaw)))

    def apply_extrinsics(cfg, fwd: float, lat: float, yaw: float):
        """Robot BELIEF uses the calibrated camera mounting (the world is unchanged)."""
        return replace(cfg, camera=replace(cfg.camera, forward_offset=fwd, lateral_offset=lat, yaw=yaw))