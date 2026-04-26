"""
analysis/biomechanics.py  — production version
Handles nan/None safely throughout. Estimates bat angle from wrist-elbow
vector when no bat bounding box is detected.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from core.geometry import (
    angle_between, euclidean_speed, rolling_mean,
    exponential_smooth,
)
from core.models import PoseDetector


def _safe(val: float, fallback: float = 0.0) -> float:
    """Return fallback if val is nan/inf/None."""
    try:
        f = float(val)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return fallback


def _valid_kp(pt: np.ndarray) -> bool:
    """True if keypoint was actually detected (non-zero)."""
    return np.any(pt != 0)


@dataclass
class FrameBiomechanics:
    frame_no:            int   = 0
    knee_angle:          float = 0.0
    hip_angle:           float = 0.0
    elbow_angle:         float = 0.0
    shoulder_angle:      float = 0.0
    wrist_speed:         float = 0.0
    bat_swing_angle:     float = 0.0
    bat_angle_estimated: bool  = False   # True when estimated from wrist-elbow
    front_foot_dist:     float = 0.0
    hip_shoulder_offset: float = 0.0
    impact_detected:     bool  = False


@dataclass
class PeakBiomechanics:
    frame_no:            int   = 0
    knee_angle:          float = 0.0
    hip_angle:           float = 0.0
    elbow_angle:         float = 0.0
    shoulder_angle:      float = 0.0
    wrist_speed:         float = 0.0
    bat_swing_angle:     float = 0.0
    bat_angle_estimated: bool  = False
    front_foot_dist:     float = 0.0
    hip_shoulder_offset: float = 0.0
    impact_detected:     bool  = False


class BiomechanicsExtractor:

    def __init__(self, fps: int = 30):
        self._fps = fps
        self._prev_kp           = None
        self._prev_wrist        = None
        self._prev_ball_centre  = None
        self._knee_buf:  list   = []
        self._hip_buf:   list   = []
        self._elbow_buf: list   = []
        self._max_wrist_speed   = 0.0
        self.peak                = None

    def reset(self):
        self.__init__(fps=self._fps)

    def process_frame(
        self,
        frame_no:    int,
        batsman_kp:  np.ndarray,
        bat_bboxes:  list,
        ball_bboxes: list,
    ) -> FrameBiomechanics:

        # Smooth keypoints
        if self._prev_kp is None:
            kp = batsman_kp.copy()
        else:
            kp = exponential_smooth(self._prev_kp, batsman_kp, alpha=0.55)
        self._prev_kp = kp

        KP = PoseDetector

        # Choose dominant side (left or right) — pick the side with more visible kps
        left_visible  = sum(_valid_kp(kp[i]) for i in [KP.LEFT_SHOULDER,  KP.LEFT_ELBOW,  KP.LEFT_WRIST,  KP.LEFT_HIP,  KP.LEFT_KNEE,  KP.LEFT_ANKLE])
        right_visible = sum(_valid_kp(kp[i]) for i in [KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST, KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE])
        side = "left" if left_visible >= right_visible else "right"

        if side == "left":
            shoulder, elbow, wrist = kp[KP.LEFT_SHOULDER],  kp[KP.LEFT_ELBOW],  kp[KP.LEFT_WRIST]
            hip, knee, ankle       = kp[KP.LEFT_HIP],       kp[KP.LEFT_KNEE],   kp[KP.LEFT_ANKLE]
            opp_shoulder           = kp[KP.RIGHT_SHOULDER]
            opp_hip, opp_ankle     = kp[KP.RIGHT_HIP],      kp[KP.RIGHT_ANKLE]
        else:
            shoulder, elbow, wrist = kp[KP.RIGHT_SHOULDER], kp[KP.RIGHT_ELBOW], kp[KP.RIGHT_WRIST]
            hip, knee, ankle       = kp[KP.RIGHT_HIP],      kp[KP.RIGHT_KNEE],  kp[KP.RIGHT_ANKLE]
            opp_shoulder           = kp[KP.LEFT_SHOULDER]
            opp_hip, opp_ankle     = kp[KP.LEFT_HIP],       kp[KP.LEFT_ANKLE]

        # Angles — only compute if all three keypoints are valid
        def safe_angle(a, b, c):
            if _valid_kp(a) and _valid_kp(b) and _valid_kp(c):
                return _safe(angle_between(a, b, c))
            return None

        knee_raw     = safe_angle(hip, knee, ankle)
        hip_raw      = safe_angle(shoulder, hip, knee)
        elbow_raw    = safe_angle(shoulder, elbow, wrist)
        neck         = (shoulder + opp_shoulder) / 2.0 if (_valid_kp(shoulder) and _valid_kp(opp_shoulder)) else None
        shoulder_raw = safe_angle(neck, shoulder, elbow) if neck is not None else None

        # Rolling mean — skip None values
        if knee_raw is not None:
            self._knee_buf, knee_angle = rolling_mean(self._knee_buf, knee_raw, 5)
        else:
            knee_angle = float(np.mean(self._knee_buf)) if self._knee_buf else 0.0

        if hip_raw is not None:
            self._hip_buf, hip_angle = rolling_mean(self._hip_buf, hip_raw, 5)
        else:
            hip_angle = float(np.mean(self._hip_buf)) if self._hip_buf else 0.0

        if elbow_raw is not None:
            self._elbow_buf, elbow_angle = rolling_mean(self._elbow_buf, elbow_raw, 5)
        else:
            elbow_angle = float(np.mean(self._elbow_buf)) if self._elbow_buf else 0.0

        shoulder_angle = _safe(shoulder_raw, 0.0)

        # Wrist speed
        wrist_speed = 0.0
        active_wrist = wrist if _valid_kp(wrist) else (kp[KP.RIGHT_WRIST] if _valid_kp(kp[KP.RIGHT_WRIST]) else None)
        if active_wrist is not None and self._prev_wrist is not None:
            wrist_speed = _safe(euclidean_speed(self._prev_wrist, active_wrist) * self._fps)
        if active_wrist is not None:
            self._prev_wrist = active_wrist.copy()

        # Bat swing angle
        bat_estimated = False
        bat_angle = 0.0
        if bat_bboxes:
            bbox  = bat_bboxes[0]
            tip   = np.array([(bbox[0]+bbox[2])/2, bbox[1]])
            grip  = np.array([(bbox[0]+bbox[2])/2, bbox[3]])
            dx    = float(tip[0] - grip[0])
            dy    = float(tip[1] - grip[1])
            bat_angle = _safe(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9)))
        elif _valid_kp(wrist) and _valid_kp(elbow):
            # Estimate bat angle from forearm direction
            dx = float(wrist[0] - elbow[0])
            dy = float(wrist[1] - elbow[1])
            bat_angle = _safe(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9)))
            bat_estimated = True

        # Stance width (ankle separation)
        front_foot_dist = 0.0
        if _valid_kp(ankle) and _valid_kp(opp_ankle):
            front_foot_dist = _safe(abs(float(ankle[0]) - float(opp_ankle[0])))

        # Hip-shoulder offset
        hip_shoulder_offset = 0.0
        if _valid_kp(hip) and _valid_kp(opp_hip) and _valid_kp(shoulder) and _valid_kp(opp_shoulder):
            hip_mid      = (hip + opp_hip) / 2.0
            shoulder_mid = (shoulder + opp_shoulder) / 2.0
            hip_shoulder_offset = _safe(abs(float(hip_mid[0] - shoulder_mid[0])))

        # Impact
        impact = self._check_impact(ball_bboxes, bat_bboxes)

        fb = FrameBiomechanics(
            frame_no            = frame_no,
            knee_angle          = knee_angle,
            hip_angle           = hip_angle,
            elbow_angle         = elbow_angle,
            shoulder_angle      = shoulder_angle,
            wrist_speed         = wrist_speed,
            bat_swing_angle     = bat_angle,
            bat_angle_estimated = bat_estimated,
            front_foot_dist     = front_foot_dist,
            hip_shoulder_offset = hip_shoulder_offset,
            impact_detected     = impact,
        )

        if wrist_speed > self._max_wrist_speed:
            self._max_wrist_speed = wrist_speed
            self.peak = PeakBiomechanics(**fb.__dict__)

        return fb

    def _check_impact(self, ball_bboxes, bat_bboxes) -> bool:
        from config import BALL_IMPACT_SPEED_THRESHOLD, BAT_BBOX_PADDING
        from core.geometry import point_in_bbox

        for ball in ball_bboxes:
            bx = (ball[0]+ball[2])/2
            by = (ball[1]+ball[3])/2
            curr = np.array([bx, by])
            if self._prev_ball_centre is not None:
                spd = euclidean_speed(self._prev_ball_centre, curr)
                if spd > BALL_IMPACT_SPEED_THRESHOLD:
                    for bat in bat_bboxes:
                        if point_in_bbox((bx,by), bat, padding=BAT_BBOX_PADDING):
                            self._prev_ball_centre = curr
                            return True
            self._prev_ball_centre = curr
        return False