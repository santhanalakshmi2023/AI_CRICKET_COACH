"""
analysis/scorer.py
───────────────────
Converts peak biomechanics into normalised 0-10 scores.

Scoring criteria (cricket batting, right-handed convention)
───────────────────────────────────────────────────────────
  balance_score  — knee bend (110-140° ideal front-foot drive)
  power_score    — hip rotation (90-160°)
  timing_score   — elbow extension at contact (120-160°)
  bat_angle_score— bat swing near vertical (0-25° ideal)
  overall_score  — weighted composite
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from analysis.biomechanics import PeakBiomechanics
import config


@dataclass
class PerformanceScore:
    balance_score:   float
    power_score:     float
    timing_score:    float
    bat_angle_score: float
    overall_score:   float
    bat_speed:       float   # wrist_speed × 1.2 proxy


def _norm(value: float, ideal_lo: float, ideal_hi: float) -> float:
    """
    Returns 10.0 if value is in [ideal_lo, ideal_hi],
    decays linearly outside that range.
    """
    if ideal_lo <= value <= ideal_hi:
        return 10.0
    if value < ideal_lo:
        decay = (ideal_lo - value) / ideal_lo
    else:
        decay = (value - ideal_hi) / (180.0 - ideal_hi + 1e-9)
    return float(np.clip(10.0 * (1.0 - decay), 0.0, 10.0))


class Scorer:
    """Stateless — call score() with a PeakBiomechanics snapshot."""

    def score(self, peak: PeakBiomechanics) -> PerformanceScore:
        balance   = _norm(peak.knee_angle,       110, 140)
        power     = _norm(peak.hip_angle,          90, 160)
        timing    = _norm(peak.elbow_angle,       120, 160)
        bat_angle = _norm(peak.bat_swing_angle,     0,  25)

        w_b = config.BALANCE_WEIGHT
        w_p = config.POWER_WEIGHT
        w_t = config.TIMING_WEIGHT

        # Remaining weight goes to bat-angle
        w_a = max(0.0, 1.0 - w_b - w_p - w_t)

        overall = (
            w_b * balance
            + w_p * power
            + w_t * timing
            + w_a * bat_angle
        )

        return PerformanceScore(
            balance_score   = round(balance,   2),
            power_score     = round(power,     2),
            timing_score    = round(timing,    2),
            bat_angle_score = round(bat_angle, 2),
            overall_score   = round(overall,   2),
            bat_speed       = round(peak.wrist_speed * 1.2, 2),
        )
