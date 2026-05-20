import numpy as np

from .base import BaseDetector
from ..types import BBox


class FaceDetector(BaseDetector):
    """Face detector using RetinaFace (from face_detection package)."""

    def __init__(self, config: dict):
        self.conf_threshold = config.get("confidence_threshold", 0.5)
        self.select_strategy = config.get("select_strategy", "largest")
        self.bbox_expand = config.get("bbox_expand", 0.2)
        self.detector = None

    def load_model(self, device: str = "cuda") -> None:
        from face_detection import RetinaFace

        gpu_id = 0 if "cuda" in device else -1
        if ":" in device:
            try:
                gpu_id = int(device.split(":")[1])
            except ValueError:
                gpu_id = 0

        self.detector = RetinaFace(gpu_id=gpu_id)
        print(f"[FaceDetector] Loaded RetinaFace (GPU: {gpu_id})")

    def predict(self, image: np.ndarray, **kwargs) -> BBox | None:
        faces = self.detector(image)
        if faces is None or len(faces) == 0:
            return None

        # RetinaFace returns list of (bbox_array, landmarks_array, confidence_scalar)
        valid = []
        for face in faces:
            if isinstance(face, tuple) and len(face) == 3:
                bbox_coords, _landmarks, conf = face
                conf = float(conf)
            elif isinstance(face, (list, np.ndarray)):
                bbox_data = np.array(face).flatten()
                if len(bbox_data) >= 5:
                    bbox_coords = bbox_data[:4]
                    conf = float(bbox_data[4])
                else:
                    continue
            else:
                continue

            if conf >= self.conf_threshold:
                valid.append((bbox_coords, conf))

        if not valid:
            return None

        if self.select_strategy == "largest":
            idx = max(
                range(len(valid)),
                key=lambda i: (valid[i][0][2] - valid[i][0][0]) * (valid[i][0][3] - valid[i][0][1]),
            )
        else:
            idx = max(range(len(valid)), key=lambda i: valid[i][1])

        bbox_coords, conf = valid[idx]
        x1, y1, x2, y2 = bbox_coords
        return BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=conf)

    def crop_face(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = image.shape[:2]
        expand = self.bbox_expand
        bw, bh = bbox.width, bbox.height
        x1 = max(0, int(bbox.x1 - bw * expand / 2))
        y1 = max(0, int(bbox.y1 - bh * expand / 2))
        x2 = min(w, int(bbox.x2 + bw * expand / 2))
        y2 = min(h, int(bbox.y2 + bh * expand / 2))
        return image[y1:y2, x1:x2]
