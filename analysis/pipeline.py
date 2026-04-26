"""
analysis/pipeline.py
─────────────────────
Orchestrates the full video → biomechanics → annotated output pipeline.
Returns a structured result dict ready for scoring + feedback.
"""

from __future__ import annotations
import cv2
import numpy as np

from core.models        import PoseDetector, ObjectDetector
from core.batsman_selector import BatsmanSelector
from analysis.biomechanics import BiomechanicsExtractor, PeakBiomechanics
from analysis.renderer     import Renderer
import config


class VideoPipeline:
    """
    Usage:
        pipeline = VideoPipeline()
        peak = pipeline.run("my_video.mp4", "output.mp4")
    """

    def __init__(self):
        self._pose_detector  = PoseDetector(config.POSE_MODEL_PATH)
        self._obj_detector   = ObjectDetector(config.OBJ_MODEL_PATH)
        self._selector       = BatsmanSelector()
        self._extractor      = BiomechanicsExtractor()
        self._renderer       = Renderer()

    def run(
        self,
        video_path:  str,
        output_path: str = config.DEFAULT_OUTPUT_PATH,
        progress_cb=None,   # optional callable(frame_no, total_frames)
    ) -> PeakBiomechanics | None:
        """
        Process *video_path* frame-by-frame.
        Writes annotated video to *output_path*.
        Returns the PeakBiomechanics snapshot (highest wrist-speed frame),
        or None if no batsman was detected.
        """
        self._extractor.reset()
        self._renderer.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Re-initialise extractor with correct fps
        self._extractor = BiomechanicsExtractor(fps=fps)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            if progress_cb:
                progress_cb(frame_no, total)

            # ── Detection ────────────────────────────────────────────
            all_kps     = self._pose_detector.detect(frame)
            obj_results = self._obj_detector.detect(frame)
            bat_bboxes  = obj_results["bats"]
            ball_bboxes = obj_results["balls"]

            # ── Batsman selection ─────────────────────────────────────
            batsman_kp = self._selector.select(all_kps, bat_bboxes, width)

            if batsman_kp is None:
                writer.write(frame)
                continue

            # ── Biomechanics ─────────────────────────────────────────
            bm = self._extractor.process_frame(
                frame_no, batsman_kp, bat_bboxes, ball_bboxes
            )

            # ── Render ───────────────────────────────────────────────
            annotated = self._renderer.draw_frame(
                frame, batsman_kp, bat_bboxes, ball_bboxes, bm
            )
            writer.write(annotated)

        cap.release()
        writer.release()

        return self._extractor.peak
