from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class EmotionResult:
    valence: float  # -1.0 to 1.0
    arousal: float  # -1.0 to 1.0
    dominant_emotion: str  # e.g. "happy", "sad", "neutral"
    emotion_probs: dict = field(default_factory=dict)  # {emotion: probability}


@dataclass
class GazeResult:
    yaw: float  # degrees, horizontal angle
    pitch: float  # degrees, vertical angle
    is_looking_at_camera: bool


@dataclass
class PoseResult:
    open_posture_score: float  # [0.0, 1.0]
    is_covering_mouth: bool
    is_withdrawing: bool
    interaction_z_meters: Optional[float] = None
    closest_wrist_px: Optional[tuple[int, int]] = None
    has_pose: bool = True
    # Raw signals — surfaced so detection thresholds can be tuned offline without
    # re-running MediaPipe. See scripts/extract_features.py (calibration Stage A).
    hand_to_mouth_ratio: Optional[float] = None  # min_wrist_to_nose_dist / shoulder_width
    z_displacement_m: Optional[float] = None     # end-of-history minus start-of-history depth
    posture_drop: Optional[float] = None         # start-of-history minus end-of-history open-posture


@dataclass
class FrameResult:
    # Emotion detection
    face_bbox: Optional[BBox] = None
    face_detected: bool = False
    emotion: Optional[EmotionResult] = None
    gaze: Optional[GazeResult] = None

    # Posture detection
    pose: Optional[PoseResult] = None

    # Comfort scores
    emotion_comfort_score: float = 50.0  # 0-100
    posture_comfort_score: float = 50.0  # 0-100
    integrated_comfort_score: float = 50.0  # 0-100

    timestamp_ms: float = 0.0
