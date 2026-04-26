"""
feedback/gemma_coach.py
────────────────────────
Generates natural-language coaching feedback using Gemma 3n 270M via
llama-cpp-python.

Install:
    pip install llama-cpp-python
    # Download model:
    # huggingface-cli download google/gemma-3n-E2B-it-GGUF \
    #   gemma-3n-270m-instruct.gguf --local-dir ./models/

Falls back to rule-based feedback if the model file is missing.
"""

from __future__ import annotations
import os
from dataclasses import dataclass

from analysis.biomechanics import PeakBiomechanics
from analysis.scorer import PerformanceScore
import config


@dataclass
class CoachFeedback:
    summary:       str
    strengths:     list[str]
    improvements:  list[str]
    drills:        list[str]
    raw_text:      str   # full LLM output


class GemmaCoach:
    """
    Wraps Gemma 3n 270M (GGUF via llama-cpp-python).
    Falls back to rule-based output when model is unavailable.
    """

    def __init__(self, model_path: str = config.GEMMA_MODEL_PATH):
        self._llm = None
        if os.path.exists(model_path):
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=1024,
                    n_threads=4,
                    verbose=False,
                )
                print(f"[GemmaCoach] Loaded model: {model_path}")
            except Exception as e:
                print(f"[GemmaCoach] Could not load Gemma — using rule-based fallback. ({e})")
        else:
            print(f"[GemmaCoach] Model not found at '{model_path}' — using rule-based fallback.")

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        peak:   PeakBiomechanics,
        scores: PerformanceScore,
        player: str = "the batsman",
    ) -> CoachFeedback:
        if self._llm is not None:
            return self._llm_feedback(peak, scores, player)
        return self._rule_feedback(peak, scores, player)

    # ─────────────────────────────────────────────────────────────────
    # LLM path
    # ─────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        peak:   PeakBiomechanics,
        scores: PerformanceScore,
        player: str,
    ) -> str:
        return (
            f"<start_of_turn>user\n"
            f"You are an expert cricket batting coach. Analyse the following "
            f"biomechanical data for {player} and give concise, actionable coaching "
            f"feedback in this exact JSON format:\n"
            f"{{\n"
            f'  "summary": "<one sentence overall assessment>",\n'
            f'  "strengths": ["<point 1>", "<point 2>"],\n'
            f'  "improvements": ["<point 1>", "<point 2>", "<point 3>"],\n'
            f'  "drills": ["<drill 1>", "<drill 2>"]\n'
            f"}}\n\n"
            f"Biomechanics:\n"
            f"  Knee angle        : {peak.knee_angle:.1f}°  (ideal 110-140°)\n"
            f"  Hip angle         : {peak.hip_angle:.1f}°   (ideal 90-160°)\n"
            f"  Elbow angle       : {peak.elbow_angle:.1f}° (ideal 120-160°)\n"
            f"  Shoulder angle    : {peak.shoulder_angle:.1f}°\n"
            f"  Bat swing angle   : {peak.bat_swing_angle:.1f}° from vertical (ideal <25°)\n"
            f"  Wrist speed       : {peak.wrist_speed:.0f} px/s\n"
            f"  Stance width      : {peak.front_foot_dist:.0f} px\n"
            f"  Hip-shoulder offset: {peak.hip_shoulder_offset:.0f} px\n"
            f"  Impact detected   : {'Yes' if peak.impact_detected else 'No'}\n\n"
            f"Scores (out of 10):\n"
            f"  Balance   : {scores.balance_score}\n"
            f"  Power     : {scores.power_score}\n"
            f"  Timing    : {scores.timing_score}\n"
            f"  Bat angle : {scores.bat_angle_score}\n"
            f"  Overall   : {scores.overall_score}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    def _llm_feedback(
        self,
        peak:   PeakBiomechanics,
        scores: PerformanceScore,
        player: str,
    ) -> CoachFeedback:
        import json

        prompt = self._build_prompt(peak, scores, player)
        response = self._llm(
            prompt,
            max_tokens=config.GEMMA_MAX_TOKENS,
            temperature=config.GEMMA_TEMPERATURE,
            stop=["<end_of_turn>"],
        )
        raw = response["choices"][0]["text"].strip()

        try:
            # Strip markdown fences if present
            clean = raw.replace("```json", "").replace("```", "").strip()
            data  = json.loads(clean)
            return CoachFeedback(
                summary      = data.get("summary", ""),
                strengths    = data.get("strengths", []),
                improvements = data.get("improvements", []),
                drills       = data.get("drills", []),
                raw_text     = raw,
            )
        except json.JSONDecodeError:
            # Return raw text wrapped in structure
            return CoachFeedback(
                summary      = raw[:200],
                strengths    = [],
                improvements = [],
                drills       = [],
                raw_text     = raw,
            )

    # ─────────────────────────────────────────────────────────────────
    # Rule-based fallback
    # ─────────────────────────────────────────────────────────────────

    def _rule_feedback(
        self,
        peak:   PeakBiomechanics,
        scores: PerformanceScore,
        player: str,
    ) -> CoachFeedback:
        strengths, improvements, drills = [], [], []

        # ── Balance / Knee ────────────────────────────────────────────
        if 110 <= peak.knee_angle <= 140:
            strengths.append("Good knee bend — solid base and weight transfer.")
        elif peak.knee_angle > 140:
            improvements.append(
                f"Knee angle too straight ({peak.knee_angle:.0f}°). "
                "Flex the front knee more to absorb pace and maintain balance."
            )
            drills.append("Wall-sit holds (30 s) to build front-leg strength.")
        else:
            improvements.append(
                f"Knee bent too deeply ({peak.knee_angle:.0f}°). "
                "Raise your body position slightly for better weight shift."
            )

        # ── Hip rotation ─────────────────────────────────────────────
        if 90 <= peak.hip_angle <= 160:
            strengths.append("Hip rotation is within optimal range — good power generation.")
        elif peak.hip_angle > 160:
            improvements.append(
                "Over-rotation of hips — you may be losing bat control through the line."
            )
            drills.append("Shadow batting with a band around knees to limit over-rotation.")
        else:
            improvements.append(
                "Insufficient hip rotation — you are leaving power on the table. "
                "Drive the front hip towards mid-on at the moment of contact."
            )
            drills.append("Hip-turn medicine-ball throws against a wall (3×10).")

        # ── Elbow / timing ───────────────────────────────────────────
        if 120 <= peak.elbow_angle <= 160:
            strengths.append("Good elbow extension at contact — timing looks solid.")
        elif peak.elbow_angle < 120:
            improvements.append(
                "Elbow too bent at contact — you may be hitting early. "
                "Let the ball come closer before releasing the shot."
            )
            drills.append("Tee-drills focusing on contact point: hit the ball when it is level with your front foot.")
        else:
            improvements.append(
                "Arm over-extended — ensure the leading elbow guides the bat."
            )

        # ── Bat swing angle ──────────────────────────────────────────
        if peak.bat_swing_angle <= 25:
            strengths.append(f"Bat swing is nearly vertical ({peak.bat_swing_angle:.0f}°) — excellent straight-bat technique.")
        else:
            improvements.append(
                f"Bat swing angle is {peak.bat_swing_angle:.0f}° from vertical — "
                "work on a straighter bat path for drives."
            )
            drills.append("Hanging-string drill: swing the bat through a hanging ball on a string to groove a straight path.")

        # ── Stance width ─────────────────────────────────────────────
        if peak.front_foot_dist < 80:
            improvements.append("Narrow stance — widen your base for better stability against pace bowling.")
        elif peak.front_foot_dist > 300:
            improvements.append("Very wide stance — may restrict footwork. Consider a slightly narrower base.")

        # ── Speed / bat speed ────────────────────────────────────────
        if scores.bat_speed > 800:
            strengths.append(f"High bat speed ({scores.bat_speed:.0f} units) — excellent power potential.")
        elif scores.bat_speed < 300:
            improvements.append("Bat speed is low — focus on wrist snap and forearm strength.")
            drills.append("Wrist-roller exercises and light-bat speed drills.")

        # ── Overall summary ──────────────────────────────────────────
        ov = scores.overall_score
        if ov >= 8:
            summary = f"{player} shows excellent technique — focus on consistency and match simulation."
        elif ov >= 6:
            summary = f"{player} has a solid foundation with a few areas to polish for the next level."
        elif ov >= 4:
            summary = f"{player} needs focused technical work; the fundamentals need reinforcing before match practice."
        else:
            summary = f"{player} requires basic biomechanical correction — prioritise coached net sessions."

        raw = (
            f"Summary: {summary}\n"
            f"Strengths: {'; '.join(strengths)}\n"
            f"Improvements: {'; '.join(improvements)}\n"
            f"Drills: {'; '.join(drills)}"
        )

        return CoachFeedback(
            summary      = summary,
            strengths    = strengths,
            improvements = improvements,
            drills       = drills,
            raw_text     = raw,
        )
