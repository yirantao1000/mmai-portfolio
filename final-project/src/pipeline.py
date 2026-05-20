import numpy as np

from .detectors.face_detector import FaceDetector
from .detectors.emotion_detector import EmotionDetector
from .detectors.gaze_detector import GazeDetector
from .detectors.pose_detector import PoseDetector
from .comfort import IntegratedComfortScorer
from .phases import Phase
from .types import FrameResult


class IntegratedPipeline:
    """Orchestrates emotion detection (face/emotion/gaze) and posture detection,
    producing a unified FrameResult with combined comfort scoring."""

    def __init__(self, config: dict):
        # Emotion branch (GPU)
        self.face_detector = FaceDetector(config.get("face_detector", {}))
        self.emotion_detector = EmotionDetector(config.get("emotion_detector", {}))
        self.gaze_detector = GazeDetector(config.get("gaze_detector", {}))

        # Posture branch (CPU)
        self.pose_detector = PoseDetector(config.get("pose_detector", {}))

        # Combined scorer
        self.comfort_scorer = IntegratedComfortScorer(config.get("comfort", {}))
        self.device = config.get("device", "cuda:0")

    def load_models(self) -> None:
        print(f"Loading models (GPU: {self.device}, CPU: MediaPipe)...")
        self.face_detector.load_model(self.device)
        self.emotion_detector.load_model(self.device)
        self.gaze_detector.load_model(self.device)
        self.pose_detector.load_model("cpu")
        print("All models loaded.")

    def reset_state(self) -> None:
        """Reset temporal state between recordings."""
        self.pose_detector.reset_state()
        self.comfort_scorer.reset()

    def set_phase(self, phase: Phase) -> None:
        """Drive phase-aware comfort scoring.

        Calibration drives this from sidecar keypoints; the robot controller drives
        it from its own state machine (intent signaling vs service execution).
        """
        self.comfort_scorer.set_phase(phase)

    @property
    def phase(self) -> Phase:
        return self.comfort_scorer.phase

    def process_frame(self, color_frame: np.ndarray, timestamp_ms: float = 0.0,
                      depth_frame=None) -> FrameResult:
        result = FrameResult(timestamp_ms=timestamp_ms)

        # --- Emotion branch (color only) ---
        bbox = self.face_detector.predict(color_frame)
        if bbox is not None:
            result.face_detected = True
            result.face_bbox = bbox

            face_crop = self.face_detector.crop_face(color_frame, bbox)
            if face_crop.size > 0:
                result.emotion = self.emotion_detector.predict(face_crop)

            result.gaze = self.gaze_detector.predict(color_frame)

        # --- Posture branch (color + depth) ---
        # Skip MediaPipe entirely when posture is disabled for the active phase.
        if self.comfort_scorer.current_posture_weight > 0.0:
            result.pose = self.pose_detector.predict(color_frame, depth_frame=depth_frame)

        # --- Combined scoring ---
        timestamp_s = timestamp_ms / 1000.0
        emotion_score, posture_score, integrated_score = self.comfort_scorer.update(result, timestamp_s)
        result.emotion_comfort_score = emotion_score
        result.posture_comfort_score = posture_score
        result.integrated_comfort_score = integrated_score

        return result
