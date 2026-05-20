#!/usr/bin/env python3
"""Download model weights for the emotion/comfort detection pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_l2cs_weights():
    """Download L2CS-Net Gaze360 weights from Google Drive."""
    weights_path = MODELS_DIR / "L2CSNet_gaze360.pkl"
    if weights_path.exists():
        print(f"[L2CS-Net] Weights already exist: {weights_path}")
        return

    try:
        import gdown
    except ImportError:
        print("Install gdown first: pip install gdown")
        sys.exit(1)

    print("[L2CS-Net] Downloading Gaze360 weights...")
    # Google Drive file ID for L2CSNet_gaze360.pkl
    file_id = "1FyIreSDMnWBTv-JBq0YDPfKqDhRBRoGo"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(weights_path), quiet=False)
    print(f"[L2CS-Net] Saved to {weights_path}")


def download_yolov8_face():
    """YOLOv8-face weights are auto-downloaded by ultralytics on first use."""
    print("[YOLOv8-Face] Weights auto-downloaded on first use by ultralytics.")


def download_hsemotion():
    """HSEmotion/EmotiEffLib weights are auto-downloaded on first use."""
    print("[HSEmotion] Weights auto-downloaded on first use by EmotiEffLib.")
    print("[HSEmotion] If using manual loading, pretrained EfficientNet-B0 is pulled from timm.")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}\n")

    download_yolov8_face()
    download_hsemotion()
    download_l2cs_weights()

    print("\nDone. All model weights ready.")


if __name__ == "__main__":
    main()
