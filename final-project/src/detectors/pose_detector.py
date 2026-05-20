from __future__ import annotations

import math
from collections import deque
from typing import Any

import cv2
import numpy as np

from .base import BaseDetector
from ..types import PoseResult


class PoseDetector(BaseDetector):
    """MediaPipe-based pose detector with optional depth-aware safety cues."""

    def __init__(self, config: dict):
        self.static_image_mode = config.get("static_image_mode", False)
        self.model_complexity = config.get("model_complexity", 1)
        self.min_detection_confidence = config.get("min_detection_confidence", 0.5)
        self.min_tracking_confidence = config.get("min_tracking_confidence", 0.5)

        self.open_posture_min_ratio = config.get("open_posture_min_ratio", 0.2)
        self.open_posture_max_ratio = config.get("open_posture_max_ratio", 1.2)
        self.face_cover_ratio = config.get("face_cover_ratio", 0.6)

        self.depth_min_meters = config.get("depth_min_meters", 0.1)
        self.depth_max_meters = config.get("depth_max_meters", 2.0)
        self.history_size = config.get("history_size", 10)
        self.depth_outlier_threshold = config.get("depth_outlier_threshold", 0.4)
        self.withdrawal_threshold_meters = config.get("withdrawal_threshold_meters", 0.12)
        self.posture_drop_threshold = config.get("posture_drop_threshold", 0.15)

        self.pose = None
        self._mp_pose = None
        self._interaction_z_history: deque[float] = deque(maxlen=self.history_size)
        self._posture_history: deque[float] = deque(maxlen=self.history_size)

    def reset_state(self) -> None:
        """Clear temporal state so each recording is processed independently."""
        self._interaction_z_history.clear()
        self._posture_history.clear()

    def load_model(self, device: str = "cpu") -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is required for PoseDetector. Install it with `pip install mediapipe`."
            ) from exc

        self._mp_pose = mp.solutions.pose
        self.pose = self._mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        print("[PoseDetector] Loaded MediaPipe Pose")

    def predict(self, image: np.ndarray, **kwargs) -> PoseResult | None:
        if self.pose is None or self._mp_pose is None:
            raise RuntimeError("PoseDetector model is not loaded. Call load_model() before predict().")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None

        depth_frame = kwargs.get("depth_frame")
        landmarks = results.pose_landmarks.landmark

        l_shoulder = landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_wrist = landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST]
        nose = landmarks[self._mp_pose.PoseLandmark.NOSE]

        shoulder_width = math.dist([l_shoulder.x, l_shoulder.y], [r_shoulder.x, r_shoulder.y])
        wrist_dist = math.dist([l_wrist.x, l_wrist.y], [r_wrist.x, r_wrist.y])
        ratio = wrist_dist / (shoulder_width + 1e-6)
        open_posture_score = self._clamp(
            (ratio - self.open_posture_min_ratio)
            / (self.open_posture_max_ratio - self.open_posture_min_ratio + 1e-6)
        )
        self._posture_history.append(open_posture_score)

        dist_l_face = math.dist([l_wrist.x, l_wrist.y], [nose.x, nose.y])
        dist_r_face = math.dist([r_wrist.x, r_wrist.y], [nose.x, nose.y])
        min_face_dist = min(dist_l_face, dist_r_face)
        hand_to_mouth_ratio = min_face_dist / (shoulder_width + 1e-6)
        is_covering_mouth = hand_to_mouth_ratio < self.face_cover_ratio

        interaction_z_meters = None
        closest_wrist_px = None
        is_withdrawing = False
        z_displacement_m: float | None = None
        posture_drop: float | None = None

        if depth_frame is not None:
            (
                interaction_z_meters,
                closest_wrist_px,
                is_withdrawing,
                z_displacement_m,
                posture_drop,
            ) = self._detect_withdrawal(image.shape, depth_frame, [l_wrist, r_wrist])

        return PoseResult(
            open_posture_score=open_posture_score,
            is_covering_mouth=is_covering_mouth,
            is_withdrawing=is_withdrawing,
            interaction_z_meters=interaction_z_meters,
            closest_wrist_px=closest_wrist_px,
            has_pose=True,
            hand_to_mouth_ratio=hand_to_mouth_ratio,
            z_displacement_m=z_displacement_m,
            posture_drop=posture_drop,
        )

    def _detect_withdrawal(
        self,
        image_shape: tuple[int, ...],
        depth_frame: Any,
        wrists: list[Any],
    ) -> tuple[float | None, tuple[int, int] | None, bool, float | None, float | None]:
        h, w = image_shape[:2]
        valid_depths: list[tuple[float, int, int]] = []

        for wrist in wrists:
            px_x, px_y = int(wrist.x * w), int(wrist.y * h)
            if 0 <= px_x < w and 0 <= px_y < h:
                z = float(depth_frame.get_distance(px_x, px_y))
                if self.depth_min_meters < z < self.depth_max_meters:
                    valid_depths.append((z, px_x, px_y))

        if not valid_depths:
            return None, None, False, None, None

        valid_depths.sort(key=lambda item: item[0])
        closest_z, px_x, px_y = valid_depths[0]

        if self._interaction_z_history:
            last_z = self._interaction_z_history[-1]
            if abs(closest_z - last_z) > self.depth_outlier_threshold:
                closest_z = last_z

        self._interaction_z_history.append(closest_z)
        is_withdrawing = False
        z_displacement: float | None = None
        posture_drop: float | None = None

        if (
            len(self._interaction_z_history) == self._interaction_z_history.maxlen
            and len(self._posture_history) >= 6
        ):
            history_list = list(self._interaction_z_history)
            start_z = sum(history_list[:3]) / 3.0
            end_z = sum(history_list[-3:]) / 3.0
            z_displacement = end_z - start_z

            posture_list = list(self._posture_history)
            start_posture = sum(posture_list[:3]) / 3.0
            end_posture = sum(posture_list[-3:]) / 3.0
            posture_drop = start_posture - end_posture
            is_crossing_arms = posture_drop > self.posture_drop_threshold

            if z_displacement > self.withdrawal_threshold_meters and not is_crossing_arms:
                is_withdrawing = True

        return closest_z, (px_x, px_y), is_withdrawing, z_displacement, posture_drop

    @staticmethod
    def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(max_value, value))
