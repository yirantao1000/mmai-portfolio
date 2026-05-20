#!/usr/bin/env python3
"""Stage A — Raw-signal feature cache.

Runs the vision models (RetinaFace / EfficientNet-B0 / L2CS-Net / MediaPipe Pose)
over every .bag in data/ and writes per-frame raw signals to cache/<scenario>/<name>.parquet.

The cache is *raw*: no thresholds applied to gaze angles, no booleans for
mouth-covering or withdrawal events. This is deliberate — Stage B (parameter
search) tunes those thresholds as optimization variables without having to
re-run the expensive vision models.

Output schema:
  timestamp_s, frame_idx,
  face_detected, valence, arousal,
  gaze_yaw_deg, gaze_pitch_deg, gaze_available,
  pose_detected, open_posture_score, hand_to_mouth_ratio,
  interaction_z_m, z_displacement_m, posture_drop
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_source import BagSource
from src.pipeline import IntegratedPipeline


CACHE_ROOT = PROJECT_ROOT / "cache"


def cache_path_for(bag_path: Path) -> Path:
    """cache/<scenario>/<bag-stem>.parquet"""
    rel = bag_path.relative_to(PROJECT_ROOT / "data")
    scenario = rel.parts[0]
    return CACHE_ROOT / scenario / (bag_path.stem + ".parquet")


def extract_one(bag_path: Path, pipeline: IntegratedPipeline) -> pd.DataFrame | None:
    source = BagSource(str(bag_path), real_time=False)
    if not source.open():
        return None

    pipeline.reset_state()
    rows: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame, depth_frame, timestamp_ms = source.read()
            if not ret or frame is None:
                break

            # We call process_frame for side effects (it updates scorer), but we
            # re-run the underlying detectors' outputs through the FrameResult.
            # Easier: call the detectors directly here to keep the cache decoupled
            # from any scoring logic. However the FrameResult already carries the
            # raw fields we need, so reuse it.
            result = pipeline.process_frame(frame, timestamp_ms, depth_frame=depth_frame)

            rows.append({
                "timestamp_s": timestamp_ms / 1000.0,
                "frame_idx": frame_idx,
                "face_detected": bool(result.face_detected),
                "valence": float(result.emotion.valence) if result.emotion else np.nan,
                "arousal": float(result.emotion.arousal) if result.emotion else np.nan,
                "dominant_emotion": result.emotion.dominant_emotion if result.emotion else "",
                "gaze_yaw_deg": float(result.gaze.yaw) if result.gaze else np.nan,
                "gaze_pitch_deg": float(result.gaze.pitch) if result.gaze else np.nan,
                "gaze_available": bool(result.gaze is not None),
                "pose_detected": bool(result.pose is not None and result.pose.has_pose),
                "open_posture_score": (
                    float(result.pose.open_posture_score)
                    if result.pose and result.pose.has_pose else np.nan
                ),
                "hand_to_mouth_ratio": (
                    float(result.pose.hand_to_mouth_ratio)
                    if result.pose and result.pose.hand_to_mouth_ratio is not None else np.nan
                ),
                "interaction_z_m": (
                    float(result.pose.interaction_z_meters)
                    if result.pose and result.pose.interaction_z_meters is not None else np.nan
                ),
                "z_displacement_m": (
                    float(result.pose.z_displacement_m)
                    if result.pose and result.pose.z_displacement_m is not None else np.nan
                ),
                "posture_drop": (
                    float(result.pose.posture_drop)
                    if result.pose and result.pose.posture_drop is not None else np.nan
                ),
            })
            frame_idx += 1
    finally:
        source.release()

    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage A — extract raw features from all .bag files.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--force", action="store_true", help="Re-extract even if cache exists.")
    parser.add_argument("--only", nargs="*", help="Optional: only extract these scenarios (e.g. sc02_comfortable).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bags = sorted((PROJECT_ROOT / "data").rglob("*.bag"))
    if args.only:
        allowed = set(args.only)
        bags = [b for b in bags if b.relative_to(PROJECT_ROOT / "data").parts[0] in allowed]
    if not bags:
        print("No .bag files to extract.")
        return 1

    print(f"Found {len(bags)} .bag files. Loading models...")
    pipeline = IntegratedPipeline(config)
    pipeline.load_models()

    t_start = time.time()
    written = 0
    skipped = 0

    for i, bag in enumerate(bags, 1):
        out = cache_path_for(bag)
        if out.exists() and not args.force:
            print(f"  [{i}/{len(bags)}] {bag.name} — cached, skipping")
            skipped += 1
            continue

        print(f"  [{i}/{len(bags)}] extracting {bag.name}...", flush=True)
        df = extract_one(bag, pipeline)
        if df is None:
            print("    SKIPPED (could not open)")
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"    wrote {len(df)} rows to {out.relative_to(PROJECT_ROOT)}")
        written += 1

    elapsed = time.time() - t_start
    print(f"\nDone. wrote={written}, cached_hit={skipped}, elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
