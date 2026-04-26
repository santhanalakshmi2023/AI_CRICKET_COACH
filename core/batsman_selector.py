"""
core/batsman_selector.py
─────────────────────────
Production-grade batsman selector.

Problems solved vs v1:
  - Bowler/feeder in foreground was being picked (larger, closer to camera)
  - Now uses MULTIPLE signals: bat proximity, body size, position in frame,
    and a background-person heuristic (batsman is usually smaller = further away)
"""

from __future__ import annotations
import numpy as np
from core.models import PoseDetector


class BatsmanSelector:

    @staticmethod
    def _wrist(kp: np.ndarray) -> np.ndarray:
        lw, rw = kp[PoseDetector.LEFT_WRIST], kp[PoseDetector.RIGHT_WRIST]
        lv, rv = np.any(lw != 0), np.any(rw != 0)
        if lv and rv: return (lw + rw) / 2.0
        if lv: return lw
        if rv: return rw
        return np.zeros(2)

    @staticmethod
    def _body_height(kp: np.ndarray) -> float:
        """Pixel height of detected person — smaller = further from camera."""
        nose   = kp[PoseDetector.NOSE]
        lankle = kp[PoseDetector.LEFT_ANKLE]
        rankle = kp[PoseDetector.RIGHT_ANKLE]
        top    = nose if np.any(nose != 0) else kp[PoseDetector.LEFT_SHOULDER]
        bottom = lankle if np.any(lankle != 0) else rankle
        if np.all(top == 0) or np.all(bottom == 0):
            return float("inf")
        return float(abs(bottom[1] - top[1]))

    @staticmethod
    def _hip_centre(kp: np.ndarray) -> np.ndarray:
        lh = kp[PoseDetector.LEFT_HIP]
        rh = kp[PoseDetector.RIGHT_HIP]
        if np.any(lh != 0) and np.any(rh != 0):
            return (lh + rh) / 2.0
        return lh if np.any(lh != 0) else rh

    def select(
        self,
        all_keypoints: list[np.ndarray],
        bat_bboxes:    list[np.ndarray],
        frame_width:   int,
        frame_height:  int = 0,
    ) -> np.ndarray | None:
        if not all_keypoints:
            return None

        # Single person — trivial
        if len(all_keypoints) == 1:
            return all_keypoints[0]

        # ── Strategy 1: bat detected → pick closest wrist ─────────────
        if bat_bboxes:
            bat_centre = np.mean(
                [np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]) for b in bat_bboxes],
                axis=0
            )
            best, min_d = None, float("inf")
            for kp in all_keypoints:
                w = self._wrist(kp)
                if np.all(w == 0): continue
                d = np.linalg.norm(w - bat_centre)
                if d < min_d:
                    min_d, best = d, kp
            if best is not None:
                return best

        # ── Strategy 2: no bat → use composite score ──────────────────
        # Batsman heuristics vs bowler/feeder in foreground:
        #   • Batsman is usually in the BACKGROUND (smaller body height)
        #   • Batsman is near the horizontal centre of frame
        #   • Bowler is often in the FOREGROUND (larger body height)
        #
        # Score: lower is better batsman candidate
        scores = []
        for kp in all_keypoints:
            height = self._body_height(kp)
            hip    = self._hip_centre(kp)
            if np.all(hip == 0):
                hip = self._wrist(kp)

            # Normalised distance from frame centre (0=centre, 1=edge)
            cx_norm = abs(hip[0] - frame_width / 2.0) / (frame_width / 2.0 + 1e-9)

            # Body height score: smaller person = background = likely batsman
            # (normalise to 0-1 range using median)
            height_score = height  # raw pixels; we compare relatively

            # Combined: prefer smaller + more central
            score = height_score * 0.7 + cx_norm * frame_width * 0.3
            scores.append((score, kp))

        scores.sort(key=lambda x: x[0])
        return scores[0][1]