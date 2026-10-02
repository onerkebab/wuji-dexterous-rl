"""B4 (W1-B14, W1-B15): which frames are usable, and the playback arrays that hold through gaps and return smoothly.

Usable frame = A's keypoints valid AND the hand seen: its 5 knuckles (MediaPipe 1, 5, 9, 13, 17) project inside the
RGB image with the session intrinsics; the wrist may be past the edge (W1-B15: hands enter and leave at the bottom
of the image, wrist last). Image bounds only, not an occlusion test.

Contract fields stay honest (NaN on unusable frames). The ``*_cmd`` playback arrays (``playback``) hold the last
playback pose through every gap, the home pose before the first usable frame, and after a gap (and at the session
start) move toward the tracked pose along a straight line in joint space at a capped joint speed until they catch
up; from then on they equal the tracked pose.
"""

from __future__ import annotations

import numpy as np

KNUCKLE_KP = (1, 5, 9, 13, 17)  # thumb CMC + 4 finger MCPs (MediaPipe-21)
ARM_SPEED_CAP_RAD_S = 2.0  # W1-B15 (approved 2026-10-01)
WUJI_SPEED_CAP_RAD_S = 3.0  # W1-B15 (approved 2026-10-01)


def knuckles_in_image(kp21_cam: np.ndarray, cam: dict | None) -> np.ndarray:
    """[T] bool: the 5 knuckles in front of the camera and inside the image. ``cam`` = the session metadata's
    camera dict (fx, fy, cx, cy, width, height); None (no metadata, e.g. the stub) = all in view.
    Non-finite keypoints count as out of view."""
    x = np.asarray(kp21_cam, float)[:, list(KNUCKLE_KP)]
    if cam is None:
        return np.isfinite(x).all((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        u = cam["fx"] * x[..., 0] / x[..., 2] + cam["cx"]
        v = cam["fy"] * x[..., 1] / x[..., 2] + cam["cy"]
        return ((x[..., 2] > 0) & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])).all(1)


def playback(q: np.ndarray, usable: np.ndarray, home: np.ndarray, t_s: np.ndarray,
             speed_cap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Playback array from a contract array ``q`` [T, n] (finite on usable frames).

    Held frames (not usable) repeat the previous playback row (``home`` [n] before the first usable frame).
    The first usable frame after a gap, or of the session, starts a catch-up: each frame moves toward ``q[t]``
    by at most ``speed_cap * dt`` (rad, max over joints; straight line in joint space; dt from the timestamps
    ``t_s``) and ends on the first frame where ``q[t]`` is within that step. Otherwise the playback row is ``q[t]``.

    Returns (q_cmd [T, n], held [T] bool = not usable, catching_up [T] bool: usable frames where q_cmd != q).
    """
    usable = np.asarray(usable, bool)
    T = len(q)
    out = np.empty((T, q.shape[1]), float)
    catching = np.zeros(T, bool)
    prev, tracking = np.asarray(home, float), False
    for t in range(T):
        if not usable[t]:
            out[t], tracking = prev, False
            continue
        if not tracking:
            dt = float(t_s[t] - t_s[t - 1]) if t > 0 else 0.0
            d = q[t] - prev
            m = np.abs(d).max()
            step = speed_cap * dt
            if m > step:
                out[t] = prev + d * (step / m)
                catching[t] = True
            else:
                out[t], tracking = q[t], True
        else:
            out[t] = q[t]
        prev = out[t]
    return out, ~usable, catching


def gap_segments(held: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True as half-open row ranges [start, end)."""
    d = np.diff(np.r_[0, np.asarray(held, int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def playback_stats(q_cmd: np.ndarray, held: np.ndarray, catching: np.ndarray, t_s: np.ndarray) -> dict:
    """Joint speed of the playback array (rad/s, max over joints): during catch-ups, on other usable frames, and at
    the frame that ends a gap (first usable frame after a held one); plus catch-up counts and durations."""
    held, catching = np.asarray(held, bool), np.asarray(catching, bool)
    dt = np.diff(t_s)
    speed = np.r_[0.0, np.abs(np.diff(q_cmd, axis=0)).max(1) / np.where(dt > 0, dt, np.nan)]
    usable = ~held
    gap_end = usable & np.r_[False, held[:-1]]
    m = lambda mask: float(np.nanmax(speed[mask])) if mask.any() else None  # noqa: E731
    runs = gap_segments(catching)
    dur = [float(t_s[b - 1] - t_s[a] + (t_s[a] - t_s[a - 1] if a > 0 else 0.0)) for a, b in runs]
    return {"max_speed_catching_up_rad_s": m(catching), "max_speed_tracking_rad_s": m(usable & ~catching & ~gap_end),
            "max_speed_gap_end_rad_s": m(gap_end), "n_catching_up": int(catching.sum()), "n_catch_ups": len(runs),
            "catch_up_duration_s": {"max": max(dur) if dur else None, "median": float(np.median(dur)) if dur else None}}
