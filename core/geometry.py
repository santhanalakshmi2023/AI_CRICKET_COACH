"""
core/geometry.py — Geometric helpers used across the pipeline
"""

import numpy as np


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Returns the angle (degrees) at vertex B formed by rays B→A and B→C.
    Works for both 2-D and 3-D points.
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos_theta = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def euclidean_speed(p1, p2) -> float:
    """Pixel distance between two 2-D points — use as a proxy for speed."""
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def exponential_smooth(prev: np.ndarray, curr: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """EMA smoothing: higher alpha → more weight on previous frame."""
    return alpha * np.array(prev) + (1.0 - alpha) * np.array(curr)


def rolling_mean(buffer: list, new_val: float, max_len: int) -> tuple[list, float]:
    """Append to buffer, trim to max_len, return (buffer, mean)."""
    buffer.append(new_val)
    if len(buffer) > max_len:
        buffer.pop(0)
    return buffer, float(np.mean(buffer))


def bat_swing_angle(tip: np.ndarray, grip: np.ndarray) -> float:
    """
    Angle of the bat relative to vertical (0° = perfectly upright).
    tip   — estimated bat-tip position (top of bat bbox)
    grip  — estimated grip position   (bottom of bat bbox / wrist)
    """
    dx = float(tip[0] - grip[0])
    dy = float(tip[1] - grip[1])
    angle_rad = np.arctan2(abs(dx), abs(dy) + 1e-9)
    return float(np.degrees(angle_rad))


def point_in_bbox(point, bbox, padding: float = 0.0) -> bool:
    """Return True if *point* (x, y) falls inside *bbox* (x1,y1,x2,y2) ± padding."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - padding) < x < (x2 + padding) and (y1 - padding) < y < (y2 + padding)
