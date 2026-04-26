"""
storage/database.py
────────────────────
Handles all SQLite persistence for performance sessions.

Schema
──────
  sessions   — one row per video analysed
  frames     — optional per-frame log (for detailed timeline replay)
"""

from __future__ import annotations
import sqlite3
import json
from datetime import datetime
from dataclasses import asdict

import config
from analysis.biomechanics import PeakBiomechanics
from analysis.scorer        import PerformanceScore
from feedback.gemma_coach   import CoachFeedback


class Database:
    """
    Context-manager safe. Use as:
        with Database() as db:
            db.save_session(...)
    Or instantiate and call db.close() manually.
    """

    DDL_SESSIONS = """
    CREATE TABLE IF NOT EXISTS sessions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        player_name   TEXT    NOT NULL,
        video_path    TEXT    NOT NULL,
        output_video  TEXT,
        -- peak biomechanics
        knee_angle          REAL,
        hip_angle           REAL,
        elbow_angle         REAL,
        shoulder_angle      REAL,
        wrist_speed         REAL,
        bat_swing_angle     REAL,
        front_foot_dist     REAL,
        hip_shoulder_offset REAL,
        impact_frame        INTEGER,
        impact_detected     INTEGER,
        -- scores
        balance_score   REAL,
        power_score     REAL,
        timing_score    REAL,
        bat_angle_score REAL,
        overall_score   REAL,
        bat_speed       REAL,
        -- feedback (stored as JSON)
        feedback_json   TEXT,
        created_at      TEXT NOT NULL
    )
    """

    def __init__(self, db_path: str = config.DB_PATH):
        self._conn   = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._cursor.execute(self.DDL_SESSIONS)
        self._conn.commit()

    # ── Context manager ───────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._conn.close()

    # ── Write ─────────────────────────────────────────────────────────
    def save_session(
        self,
        player:       str,
        video_path:   str,
        output_video: str,
        peak:         PeakBiomechanics,
        scores:       PerformanceScore,
        feedback:     CoachFeedback,
    ) -> int:
        """Insert a session row. Returns the new row id."""
        feedback_json = json.dumps({
            "summary":      feedback.summary,
            "strengths":    feedback.strengths,
            "improvements": feedback.improvements,
            "drills":       feedback.drills,
        })

        self._cursor.execute(
            """
            INSERT INTO sessions VALUES (
                NULL,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,
                ?,?
            )
            """,
            (
                player, video_path, output_video,
                # biomechanics
                peak.knee_angle, peak.hip_angle, peak.elbow_angle,
                peak.shoulder_angle, peak.wrist_speed, peak.bat_swing_angle,
                peak.front_foot_dist, peak.hip_shoulder_offset,
                peak.frame_no, int(peak.impact_detected),
                # scores
                scores.balance_score, scores.power_score,
                scores.timing_score, scores.bat_angle_score,
                scores.overall_score, scores.bat_speed,
                # feedback + timestamp
                feedback_json,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        self._conn.commit()
        return self._cursor.lastrowid

    # ── Read ──────────────────────────────────────────────────────────
    def get_history(self, player: str) -> list[dict]:
        """Return all sessions for *player*, newest first."""
        rows = self._cursor.execute(
            "SELECT * FROM sessions WHERE player_name=? ORDER BY id DESC",
            (player,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session(self, session_id: int) -> dict | None:
        row = self._cursor.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_players(self) -> list[str]:
        rows = self._cursor.execute(
            "SELECT DISTINCT player_name FROM sessions ORDER BY player_name"
        ).fetchall()
        return [r[0] for r in rows]

    def delete_session(self, session_id: int) -> bool:
        self._cursor.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self._conn.commit()
        return self._cursor.rowcount > 0

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        # Deserialise feedback JSON back to dict
        if "feedback_json" in d and d["feedback_json"]:
            d["feedback"] = json.loads(d["feedback_json"])
        return d
