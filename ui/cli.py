"""
ui/cli.py
──────────
Rich terminal interface for the Cricket AI Coach.

Menu
────
  1  Analyse video   — full pipeline (YOLO → biomechanics → Gemma → DB)
  2  View history    — list all sessions for a player
  3  Session detail  — drill into a specific session
  4  List players    — all players in the database
  5  Delete session  — remove a record
  0  Exit
"""

from __future__ import annotations
import os
import sys


# ── Optional rich terminal colours (graceful fallback) ────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich         import print as rprint
    _RICH = True
except ImportError:
    _RICH = False


class CLI:
    """
    Drives the full application from the terminal.
    All heavy imports are deferred to action methods so startup is fast.
    """

    def __init__(self):
        if _RICH:
            self._console = Console()

    def _print(self, text: str, style: str = ""):
        if _RICH:
            self._console.print(text, style=style)
        else:
            print(text)

    # ─────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────

    def run(self):
        self._header()
        while True:
            self._menu()
            choice = input("\nChoice: ").strip()
            print()
            if   choice == "1": self._analyse_video()
            elif choice == "2": self._view_history()
            elif choice == "3": self._session_detail()
            elif choice == "4": self._list_players()
            elif choice == "5": self._delete_session()
            elif choice == "0":
                self._print("Goodbye! 🏏", style="bold green")
                sys.exit(0)
            else:
                self._print("Invalid choice — try again.", style="yellow")

    # ─────────────────────────────────────────────────────────────────
    # Menu options
    # ─────────────────────────────────────────────────────────────────

    def _analyse_video(self):
        player = input("Player name : ").strip()
        if not player:
            self._print("Player name cannot be empty.", style="red")
            return

        path = input("Video path  : ").strip()
        if not os.path.isfile(path):
            self._print(f"File not found: {path}", style="red")
            return

        default_out = f"output_{player.replace(' ','_')}.mp4"
        out = input(f"Output path [{default_out}]: ").strip() or default_out

        self._print("\n⚙  Running analysis pipeline …", style="cyan")

        # ── Lazy imports (heavy — only load when needed) ──────────────
        from analysis.pipeline  import VideoPipeline
        from analysis.scorer    import Scorer
        from feedback.gemma_coach import GemmaCoach
        from storage.database   import Database

        pipeline = VideoPipeline()
        scorer   = Scorer()
        coach    = GemmaCoach()

        def progress(frame_no, total):
            if frame_no % 30 == 0 or frame_no == total:
                pct = int(frame_no / max(total, 1) * 100)
                print(f"\r  Processing … {pct:3d}%  (frame {frame_no}/{total})", end="", flush=True)

        peak = pipeline.run(path, output_path=out, progress_cb=progress)
        print()   # newline after progress

        if peak is None:
            self._print("No batsman detected in the video. Check camera angle / lighting.", style="red")
            return

        scores   = scorer.score(peak)
        feedback = coach.generate(peak, scores, player=player)

        # ── Display results ───────────────────────────────────────────
        self._print_biomechanics(peak)
        self._print_scores(scores)
        self._print_feedback(feedback)

        # ── Persist ──────────────────────────────────────────────────
        with Database() as db:
            row_id = db.save_session(player, path, out, peak, scores, feedback)
        self._print(f"\n✅  Saved as session #{row_id}. Annotated video → {out}", style="green")

    def _view_history(self):
        player = input("Player name: ").strip()
        if not player:
            return

        from storage.database import Database
        with Database() as db:
            sessions = db.get_history(player)

        if not sessions:
            self._print(f"No history found for '{player}'.", style="yellow")
            return

        self._print(f"\n📋  History for {player}  ({len(sessions)} session/s)\n", style="bold")

        if _RICH:
            t = Table(show_header=True, header_style="bold magenta")
            for col in ["ID", "Date", "Overall", "Balance", "Power", "Timing", "Bat Speed", "Video"]:
                t.add_column(col)
            for s in sessions:
                t.add_row(
                    str(s["id"]),
                    s["created_at"],
                    f"{s['overall_score']:.1f}",
                    f"{s['balance_score']:.1f}",
                    f"{s['power_score']:.1f}",
                    f"{s['timing_score']:.1f}",
                    f"{s['bat_speed']:.0f}",
                    os.path.basename(s["output_video"] or ""),
                )
            self._console.print(t)
        else:
            for s in sessions:
                print(
                    f"  [{s['id']}] {s['created_at']}  "
                    f"Overall={s['overall_score']:.1f}  "
                    f"BatSpeed={s['bat_speed']:.0f}"
                )

    def _session_detail(self):
        sid = input("Session ID: ").strip()
        if not sid.isdigit():
            self._print("Invalid ID.", style="red")
            return

        from storage.database import Database
        with Database() as db:
            s = db.get_session(int(sid))

        if s is None:
            self._print("Session not found.", style="red")
            return

        print(f"\n── Session #{s['id']}  •  {s['player_name']}  •  {s['created_at']} ──\n")
        print(f"  Video        : {s['video_path']}")
        print(f"  Output       : {s['output_video']}")
        print(f"  Impact frame : {s['impact_frame']}  (detected={bool(s['impact_detected'])})\n")

        print(f"  Biomechanics:")
        for k in ["knee_angle","hip_angle","elbow_angle","shoulder_angle",
                  "bat_swing_angle","wrist_speed","front_foot_dist"]:
            print(f"    {k:22s}: {s[k]:.1f}")

        print(f"\n  Scores:")
        for k in ["balance_score","power_score","timing_score","bat_angle_score","overall_score","bat_speed"]:
            print(f"    {k:22s}: {s[k]:.2f}")

        fb = s.get("feedback", {})
        if fb:
            print(f"\n  Summary: {fb.get('summary','')}")
            print("  Strengths:")
            for x in fb.get("strengths", []):
                print(f"    ✔ {x}")
            print("  Improvements:")
            for x in fb.get("improvements", []):
                print(f"    ✖ {x}")
            print("  Drills:")
            for x in fb.get("drills", []):
                print(f"    ➤ {x}")

    def _list_players(self):
        from storage.database import Database
        with Database() as db:
            players = db.list_players()
        if not players:
            self._print("No players in the database yet.", style="yellow")
        else:
            self._print("\n🏏  Players in database:\n", style="bold")
            for p in players:
                print(f"   • {p}")

    def _delete_session(self):
        sid = input("Session ID to delete: ").strip()
        if not sid.isdigit():
            self._print("Invalid ID.", style="red")
            return
        confirm = input(f"Delete session #{sid}? [y/N]: ").strip().lower()
        if confirm != "y":
            self._print("Cancelled.", style="yellow")
            return
        from storage.database import Database
        with Database() as db:
            ok = db.delete_session(int(sid))
        if ok:
            self._print(f"Session #{sid} deleted.", style="green")
        else:
            self._print("Session not found.", style="red")

    # ─────────────────────────────────────────────────────────────────
    # Display helpers
    # ─────────────────────────────────────────────────────────────────

    def _header(self):
        banner = (
            "\n"
            "  ╔══════════════════════════════════════╗\n"
            "  ║    🏏  Cricket AI Coach  v2.0  🏏    ║\n"
            "  ║   YOLO Pose + Gemma 3n 270M Feedback ║\n"
            "  ╚══════════════════════════════════════╝\n"
        )
        print(banner)

    def _menu(self):
        print(
            "  1  Analyse video\n"
            "  2  View player history\n"
            "  3  Session detail\n"
            "  4  List players\n"
            "  5  Delete session\n"
            "  0  Exit"
        )

    def _print_biomechanics(self, peak):
        print("\n── Peak Biomechanics (highest bat-speed frame) ──")
        rows = [
            ("Knee angle",          f"{peak.knee_angle:.1f}°",      "110-140° ideal"),
            ("Hip angle",           f"{peak.hip_angle:.1f}°",       "90-160° ideal"),
            ("Elbow angle",         f"{peak.elbow_angle:.1f}°",     "120-160° ideal"),
            ("Shoulder angle",      f"{peak.shoulder_angle:.1f}°",  ""),
            ("Bat swing angle",     f"{peak.bat_swing_angle:.1f}°", "< 25° ideal"),
            ("Wrist speed",         f"{peak.wrist_speed:.0f} px/s", ""),
            ("Stance width",        f"{peak.front_foot_dist:.0f} px",""),
            ("Hip-shoulder offset", f"{peak.hip_shoulder_offset:.0f} px",""),
            ("Impact detected",     str(peak.impact_detected),      ""),
        ]
        for name, val, note in rows:
            note_str = f"  ({note})" if note else ""
            print(f"  {name:24s}: {val:>12s}{note_str}")

    def _print_scores(self, scores):
        print("\n── Performance Scores (out of 10) ──")
        items = [
            ("Balance",   scores.balance_score),
            ("Power",     scores.power_score),
            ("Timing",    scores.timing_score),
            ("Bat angle", scores.bat_angle_score),
            ("Overall",   scores.overall_score),
            ("Bat speed", scores.bat_speed),
        ]
        for label, val in items:
            bar = "█" * int(val) + "░" * (10 - int(min(val, 10)))
            if label == "Bat speed":
                print(f"  {label:12s}: {val:>7.0f} px/s")
            else:
                print(f"  {label:12s}: {bar}  {val:.1f}")

    def _print_feedback(self, feedback):
        print("\n── 🎙  Coach Feedback ──")
        print(f"\n  {feedback.summary}\n")
        if feedback.strengths:
            print("  Strengths:")
            for s in feedback.strengths:
                print(f"    ✔ {s}")
        if feedback.improvements:
            print("  Improvements:")
            for s in feedback.improvements:
                print(f"    ✖ {s}")
        if feedback.drills:
            print("  Drills to practice:")
            for s in feedback.drills:
                print(f"    ➤ {s}")
        print()