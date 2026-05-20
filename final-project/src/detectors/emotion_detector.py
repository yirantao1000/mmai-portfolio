import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

from .base import BaseDetector
from ..types import EmotionResult

EMOTION_LABELS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise",
]


class EmotionDetector(BaseDetector):
    """Emotion detector using EfficientNet trained on AffectNet.

    Outputs continuous valence/arousal and discrete emotion categories.
    Uses the enet_b0_8_va_mtl architecture from HSEmotion.
    """

    def __init__(self, config: dict):
        self.model_name = config.get("model", "enet_b0_8_va_mtl")
        self.input_size = config.get("input_size", 260)
        self.model = None
        self.transform = None
        self.device = "cuda"

    def load_model(self, device: str = "cuda") -> None:
        self.device = device

        # Try emotiefflib first, fall back to manual HSEmotion loading
        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
            self.model = EmotiEffLibRecognizer(
                engine="torch",
                model_name=self.model_name,
                device=device,
            )
            self._use_emotiefflib = True
            print(f"[EmotionDetector] Loaded via EmotiEffLib: {self.model_name}")
        except (ImportError, Exception) as e:
            print(f"[EmotionDetector] EmotiEffLib not available ({e}), loading HSEmotion manually")
            self._use_emotiefflib = False
            self._load_hsemotion_manual(device)

    def _load_hsemotion_manual(self, device: str) -> None:
        """Load EfficientNet-B0 trained on AffectNet for VA+emotion MTL."""
        model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)
        # Replace classifier for 8 emotions + 2 (valence, arousal) = 10 outputs
        model.classifier = torch.nn.Linear(model.classifier.in_features, 10)
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

    def predict(self, face_image: np.ndarray, **kwargs) -> EmotionResult:
        if self._use_emotiefflib:
            return self._predict_emotiefflib(face_image)
        return self._predict_manual(face_image)

    def _predict_emotiefflib(self, face_image: np.ndarray) -> EmotionResult:
        labels, scores = self.model.predict_emotions(face_image, logits=True)
        # scores shape: (1, 10) -- 8 emotions + valence + arousal
        valence = float(scores[0, -2])
        arousal = float(scores[0, -1])

        # Softmax on emotion columns for probabilities
        emotion_logits = scores[0, :-2]
        emotion_exp = np.exp(emotion_logits - np.max(emotion_logits))
        emotion_probs = emotion_exp / emotion_exp.sum()

        dominant_idx = int(np.argmax(emotion_probs))
        dominant_emotion = EMOTION_LABELS[dominant_idx]
        probs_dict = {EMOTION_LABELS[i]: float(emotion_probs[i]) for i in range(len(EMOTION_LABELS))}

        return EmotionResult(
            valence=np.clip(valence, -1.0, 1.0),
            arousal=np.clip(arousal, -1.0, 1.0),
            dominant_emotion=dominant_emotion,
            emotion_probs=probs_dict,
        )

    def _predict_manual(self, face_image: np.ndarray) -> EmotionResult:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        output = output.cpu().numpy()[0]
        valence = float(output[-2])
        arousal = float(output[-1])

        emotion_logits = output[:-2]
        emotion_exp = np.exp(emotion_logits - np.max(emotion_logits))
        emotion_probs = emotion_exp / emotion_exp.sum()

        dominant_idx = int(np.argmax(emotion_probs))
        dominant_emotion = EMOTION_LABELS[dominant_idx]
        probs_dict = {EMOTION_LABELS[i]: float(emotion_probs[i]) for i in range(len(EMOTION_LABELS))}

        return EmotionResult(
            valence=np.clip(valence, -1.0, 1.0),
            arousal=np.clip(arousal, -1.0, 1.0),
            dominant_emotion=dominant_emotion,
            emotion_probs=probs_dict,
        )
