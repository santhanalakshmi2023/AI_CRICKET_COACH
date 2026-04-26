"""
analysis/renderer.py  — production-grade annotator
"""

from __future__ import annotations
from collections import deque
import math
import cv2
import numpy as np

from core.models import PoseDetector
from analysis.biomechanics import FrameBiomechanics

_GREEN  = (50,  220,  50)
_CYAN   = (255, 220,   0)
_PINK   = (220,  50, 220)
_RED    = (50,   50, 220)
_WHITE  = (255, 255, 255)
_YELLOW = (0,   220, 220)
_ORANGE = (30,  165, 255)
_BLACK  = (0,     0,   0)


def _safe_int(val, fallback=0) -> int:
    """Convert float/nan/None to int safely."""
    try:
        if val is None:
            return fallback
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return int(f)
    except Exception:
        return fallback


def _fmt(val, unit="°", fallback="N/A") -> str:
    """Format a measurement for display. Shows N/A if invalid."""
    try:
        if val is None:
            return fallback
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return f"{int(f)}{unit}"
    except Exception:
        return fallback


class Renderer:
    def __init__(self, trail_length: int = 30):
        self._trail: deque[tuple[int, int]] = deque(maxlen=trail_length)

    def reset(self):
        self._trail.clear()

    def draw_frame(
        self,
        frame:       np.ndarray,
        batsman_kp:  np.ndarray,
        bat_bboxes:  list[np.ndarray],
        ball_bboxes: list[np.ndarray],
        bm:          FrameBiomechanics,
    ) -> np.ndarray:
        out = frame.copy()
        self._draw_skeleton(out, batsman_kp)
        self._draw_keypoint_dots(out, batsman_kp)
        self._draw_wrist_trail(out, batsman_kp)
        self._draw_bat_overlay(out, bat_bboxes, batsman_kp)
        self._draw_ball_overlay(out, ball_bboxes)
        self._draw_angle_arcs(out, batsman_kp, bm)
        self._draw_hud(out, bm)
        if bm.impact_detected:
            self._draw_impact_flash(out)
        return out

    def _draw_skeleton(self, frame: np.ndarray, kp: np.ndarray):
        # Draw bones with thickness based on importance
        important = {(5,7),(7,9),(6,8),(8,10)}
        for a, b in PoseDetector.SKELETON_PAIRS:
            pa = tuple(map(int, kp[a]))
            pb = tuple(map(int, kp[b]))
            if pa == (0,0) or pb == (0,0):
                continue
            thick = 3 if (a,b) in important or (b,a) in important else 2
            cv2.line(frame, pa, pb, _GREEN, thick, cv2.LINE_AA)

    def _draw_keypoint_dots(self, frame: np.ndarray, kp: np.ndarray):
        for i, (x, y) in enumerate(kp):
            if x == 0 and y == 0:
                continue
            if i in (PoseDetector.LEFT_WRIST, PoseDetector.RIGHT_WRIST):
                colour, r = _CYAN, 7
            elif i in (PoseDetector.LEFT_ELBOW, PoseDetector.RIGHT_ELBOW):
                colour, r = (0, 200, 255), 6
            elif i in (PoseDetector.LEFT_HIP, PoseDetector.RIGHT_HIP):
                colour, r = (255, 100, 50), 6
            elif i in (PoseDetector.LEFT_KNEE, PoseDetector.RIGHT_KNEE):
                colour, r = (50, 255, 200), 6
            else:
                colour, r = _GREEN, 4
            cv2.circle(frame, (int(x), int(y)), r, colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), r+1, _BLACK, 1, cv2.LINE_AA)

    def _draw_wrist_trail(self, frame: np.ndarray, kp: np.ndarray):
        lw = kp[PoseDetector.LEFT_WRIST]
        rw = kp[PoseDetector.RIGHT_WRIST]
        wrist = lw if np.any(lw != 0) else rw
        if np.all(wrist == 0):
            return
        pt = (int(wrist[0]), int(wrist[1]))
        self._trail.append(pt)
        pts = list(self._trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            colour = (int(50*alpha), int(200*alpha), int(255*(1-alpha)))
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pts[i-1], pts[i], colour, thickness, cv2.LINE_AA)

    def _draw_bat_overlay(self, frame: np.ndarray, bat_bboxes: list, kp: np.ndarray):
        if bat_bboxes:
            for bbox in bat_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1,y1), (x2,y2), _ORANGE, 2)
                cx = (x1+x2)//2
                cv2.line(frame, (cx,y1), (cx,y2), _YELLOW, 1, cv2.LINE_AA)
                cv2.putText(frame, "BAT", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _ORANGE, 2, cv2.LINE_AA)
        else:
            # Draw estimated bat line from wrist to mid-palm when bat not detected
            lw = kp[PoseDetector.LEFT_WRIST]
            rw = kp[PoseDetector.RIGHT_WRIST]
            le = kp[PoseDetector.LEFT_ELBOW]
            re = kp[PoseDetector.RIGHT_ELBOW]
            if np.any(lw != 0) and np.any(le != 0):
                # Extend line from elbow through wrist to estimate bat
                dx = lw[0] - le[0]
                dy = lw[1] - le[1]
                tip = (int(lw[0] + dx*1.5), int(lw[1] + dy*1.5))
                cv2.line(frame, tuple(map(int, lw)), tip, _ORANGE, 3, cv2.LINE_AA)
                cv2.putText(frame, "BAT(est)", (int(lw[0])+5, int(lw[1])-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ORANGE, 1, cv2.LINE_AA)

    def _draw_ball_overlay(self, frame: np.ndarray, ball_bboxes: list):
        for bbox in ball_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            r = max(8, (x2-x1)//2)
            cv2.circle(frame, (cx, cy), r, _RED, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (0,0,255), -1, cv2.LINE_AA)

    def _draw_angle_arcs(self, frame: np.ndarray, kp: np.ndarray, bm: FrameBiomechanics):
        """Draw angle value next to each joint with a coloured indicator dot."""
        KP = PoseDetector
        pairs = [
            (KP.LEFT_KNEE,     bm.knee_angle,     "K"),
            (KP.LEFT_HIP,      bm.hip_angle,      "H"),
            (KP.LEFT_ELBOW,    bm.elbow_angle,     "E"),
            (KP.LEFT_SHOULDER, bm.shoulder_angle,  "S"),
        ]
        for idx, val, prefix in pairs:
            x, y = int(kp[idx][0]), int(kp[idx][1])
            if x == 0 and y == 0:
                continue
            label = f"{prefix}:{_fmt(val)}"
            # Colour-code: green=good, yellow=caution, red=issue
            colour = self._angle_colour(prefix, val)
            # Background pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x+6, y-16), (x+tw+12, y+4), (0,0,0), -1)
            cv2.putText(frame, label, (x+8, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)

    def _angle_colour(self, prefix: str, val):
        """Returns green/yellow/red based on ideal ranges per joint."""
        try:
            v = float(val)
            if math.isnan(v):
                return _WHITE
        except Exception:
            return _WHITE

        ranges = {
            "K": (110, 140),
            "H": (90,  160),
            "E": (120, 160),
            "S": (60,  120),
        }
        lo, hi = ranges.get(prefix, (0, 180))
        if lo <= v <= hi:
            return (50, 255, 50)     # green
        elif abs(v - lo) < 20 or abs(v - hi) < 20:
            return (0, 220, 255)     # yellow
        else:
            return (50, 50, 255)     # red

    def _draw_hud(self, frame: np.ndarray, bm: FrameBiomechanics):
        h, w = frame.shape[:2]

        # Bat angle: show estimated if 0 (no bat box)
        bat_str = _fmt(bm.bat_swing_angle) if bm.bat_swing_angle > 0 else "est."
        wsp     = _fmt(bm.wrist_speed, unit=" px/s")
        stance  = _fmt(bm.front_foot_dist, unit=" px")

        lines = [
            ("Frame",  str(bm.frame_no)),
            ("Knee",   _fmt(bm.knee_angle)),
            ("Hip",    _fmt(bm.hip_angle)),
            ("Elbow",  _fmt(bm.elbow_angle)),
            ("BatAng", bat_str),
            ("WSpeed", wsp),
            ("Stance", stance),
        ]

        pad_x, pad_y = 10, 10
        line_h = 28
        box_w  = 230
        box_h  = len(lines) * line_h + 16

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad_x, pad_y),
                      (pad_x + box_w, pad_y + box_h), (10,10,10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (pad_x, pad_y),
                      (pad_x + box_w, pad_y + box_h), _ORANGE, 1)

        for i, (label, value) in enumerate(lines):
            y = pad_y + 20 + i * line_h
            cv2.putText(frame, f"{label:<7}: {value}",
                        (pad_x+8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, _WHITE, 1, cv2.LINE_AA)

        # Overall quality bar at bottom of HUD
        bar_y = pad_y + box_h + 6
        cv2.putText(frame, "CRICKET AI COACH", (pad_x, bar_y+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _ORANGE, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_impact_flash(frame: np.ndarray):
        h, w = frame.shape[:2]
        cv2.putText(frame, "IMPACT!", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,0,255), 5, cv2.LINE_AA)
        cv2.putText(frame, "IMPACT!", (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (0,100,255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,200), 8)