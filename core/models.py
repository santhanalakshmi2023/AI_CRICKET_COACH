"""
core/models.py — Thin wrappers around YOLOv8 pose + object models
"""

from ultralytics import YOLO
import numpy as np
import config


class PoseDetector:
    """
    Wraps YOLOv8-pose.
    Returns a list of (N, 17, 2) numpy arrays — one per detected person.
    """

    # YOLOv8 keypoint indices (COCO 17-point skeleton)
    NOSE        = 0
    LEFT_EYE    = 1; RIGHT_EYE   = 2
    LEFT_EAR    = 3; RIGHT_EAR   = 4
    LEFT_SHOULDER  = 5; RIGHT_SHOULDER  = 6
    LEFT_ELBOW     = 7; RIGHT_ELBOW     = 8
    LEFT_WRIST     = 9; RIGHT_WRIST     = 10
    LEFT_HIP       = 11; RIGHT_HIP      = 12
    LEFT_KNEE      = 13; RIGHT_KNEE     = 14
    LEFT_ANKLE     = 15; RIGHT_ANKLE    = 16

    SKELETON_PAIRS = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW),   (LEFT_ELBOW,  LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER,  LEFT_HIP),    (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP,  RIGHT_HIP),
        (LEFT_HIP,  LEFT_KNEE),   (LEFT_KNEE,  LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE),  (RIGHT_KNEE, RIGHT_ANKLE),
    ]

    def __init__(self, model_path: str = config.POSE_MODEL_PATH):
        self._model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        Returns list of keypoint arrays (shape: [17, 2]) for each person.
        Filters out detections where fewer than 10 keypoints are visible.
        """
        results = self._model(frame, verbose=False)
        kp_obj = results[0].keypoints
        if kp_obj is None:
            return []

        all_kps = kp_obj.xy.cpu().numpy()   # (num_persons, 17, 2)
        valid = []
        for person_kp in all_kps:
            visible = np.sum(np.any(person_kp != 0, axis=1))
            if visible >= 10:
                valid.append(person_kp)
        return valid


class ObjectDetector:
    """
    Wraps YOLOv8-obj. Returns ball and bat bounding boxes found in a frame.
    """

    def __init__(self, model_path: str = config.OBJ_MODEL_PATH):
        self._model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> dict[str, list[np.ndarray]]:
        """
        Returns:
            {
                "balls": [np.array([x1, y1, x2, y2]), ...],
                "bats":  [np.array([x1, y1, x2, y2]), ...],
            }
        """
        results = self._model(frame, verbose=False)
        balls, bats = [], []

        for box in results[0].boxes:
            cls    = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy()

            if cls == config.COCO_BALL_CLASS:
                balls.append(coords)
            elif cls == config.COCO_BAT_CLASS:
                bats.append(coords)

        return {"balls": balls, "bats": bats}
