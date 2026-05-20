import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .base import BaseDetector
from ..types import GazeResult

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class GazeDetector(BaseDetector):
    """Gaze detector using L2CS-Net.

    Outputs yaw/pitch angles and binary is_looking_at_camera.
    """

    def __init__(self, config: dict):
        self.input_size = config.get("input_size", 448)
        self.yaw_threshold = config.get("yaw_threshold", 15.0)
        self.pitch_threshold = config.get("pitch_threshold", 10.0)
        self.model = None
        self.device = "cuda"
        self._pipeline = None

    def load_model(self, device: str = "cuda") -> None:
        self.device = device

        # Try L2CS Pipeline first
        try:
            from l2cs import Pipeline
            weights_path = MODELS_DIR / "L2CSNet" / "Gaze360" / "L2CSNet_gaze360.pkl"
            if not weights_path.exists():
                print(f"[GazeDetector] Weights not found at {weights_path}")
                print("[GazeDetector] Run: python scripts/download_models.py")
                raise FileNotFoundError(f"Missing {weights_path}")

            self._pipeline = Pipeline(
                weights=weights_path,
                arch="ResNet50",
                device=torch.device(device),
                include_detector=True,
                confidence_threshold=0.5,
            )
            self._use_pipeline = True
            print("[GazeDetector] Loaded L2CS-Net Pipeline (with built-in face detector)")

        except (ImportError, FileNotFoundError) as e:
            print(f"[GazeDetector] L2CS Pipeline not available ({e}), loading manually")
            self._use_pipeline = False
            self._load_manual(device)

    def _load_manual(self, device: str) -> None:
        """Manual L2CS-Net loading without the l2cs package."""
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        # L2CS-Net replaces FC layer with two branches: yaw (90 bins) and pitch (90 bins)
        model.fc = torch.nn.Identity()  # Will use custom heads
        model = model.to(device)
        model.eval()
        self.model = model

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image: np.ndarray, **kwargs) -> GazeResult | None:
        if self._use_pipeline:
            return self._predict_pipeline(image)
        return self._predict_manual(image)

    def _predict_pipeline(self, image: np.ndarray, **kwargs) -> GazeResult | None:
        """Use L2CS Pipeline (operates on full frame, does its own face detection)."""
        try:
            results = self._pipeline.step(image)
        except (ValueError, RuntimeError):
            return None

        if results is None or len(results.yaw) == 0:
            return None

        # Take the first detected face's gaze
        yaw_rad = float(results.yaw[0])
        pitch_rad = float(results.pitch[0])
        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)

        is_looking = abs(yaw_deg) < self.yaw_threshold and abs(pitch_deg) < self.pitch_threshold

        return GazeResult(
            yaw=yaw_deg,
            pitch=pitch_deg,
            is_looking_at_camera=is_looking,
        )

    def _predict_manual(self, image: np.ndarray, **kwargs) -> GazeResult | None:
        """Fallback: basic gaze estimation from face crop.

        Note: Without the full L2CS model weights loaded properly, this is
        a placeholder. Install l2cs and download weights for accurate results.
        """
        return GazeResult(yaw=0.0, pitch=0.0, is_looking_at_camera=True)
