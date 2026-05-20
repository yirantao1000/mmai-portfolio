#!/usr/bin/env python3
"""One-time frame cache preparation.

Reads every .bag at a fixed --fps, writes JPEGs to:
    frame_cache/<scenario>/<bag_stem>/<frame_idx:06d>.jpg
and emits a manifest CSV at frame_cache/manifest.csv with columns
    scenario, bag_stem, frame_idx, timestamp_s, jpeg_path

The student dataset joins this manifest against per-source annotation
parquets so training does not have to re-decode .bag files every epoch.

Usage:
    python scripts/prepare_frames.py
    python scripts/prepare_frames.py --fps 15 --resize-max 320
    python scripts/prepare_frames.py --only sc02_comfortable
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import find_bags, iter_bag_frames, scenario_of


def prepare_one(
    bag_path: Path,
    data_root: Path,
    cache_root: Path,
    fps: float,
    resize_max: int,
    quality: int,
    overwrite: bool,
) -> list[dict]:
    scen = scenario_of(bag_path, data_root)
    out_dir = cache_root / scen / bag_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_written = 0
    n_skipped = 0
    for sample in iter_bag_frames(bag_path, target_fps=fps, resize_max_side=resize_max):
        jpeg_path = out_dir / f"{sample.frame_idx:06d}.jpg"
        rel_path = jpeg_path.relative_to(cache_root).as_posix()
        if jpeg_path.exists() and not overwrite:
            n_skipped += 1
        else:
            ok = cv2.imwrite(
                str(jpeg_path),
                sample.bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
            )
            if not ok:
                continue
            n_written += 1
        rows.append({
            "scenario": scen,
            "bag_stem": bag_path.stem,
            "frame_idx": int(sample.frame_idx),
            "timestamp_s": float(sample.timestamp_s),
            "jpeg_path": rel_path,
        })
    print(f"      wrote {n_written}, reused {n_skipped} jpegs in {out_dir}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a JPEG frame cache from .bag files.")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--cache-root", type=str, default=str(PROJECT_ROOT / "frame_cache"))
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--resize-max", type=int, default=320)
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    bags = find_bags(data_root, args.only)
    if not bags:
        print(f"No bags under {data_root}.")
        return 1

    print(f"Preparing {len(bags)} bags @ {args.fps} fps -> {cache_root}")
    t0 = time.time()
    all_rows: list[dict] = []
    for i, bag in enumerate(bags, 1):
        scen = scenario_of(bag, data_root)
        print(f"  [{i}/{len(bags)}] {scen}/{bag.name}", flush=True)
        rows = prepare_one(
            bag, data_root, cache_root, args.fps, args.resize_max, args.quality, args.overwrite
        )
        all_rows.extend(rows)

    manifest_path = cache_root / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        prev = pd.read_csv(manifest_path)
        df = pd.concat([prev, pd.DataFrame(all_rows)], ignore_index=True)
        df = df.drop_duplicates(subset=["scenario", "bag_stem", "frame_idx"], keep="last")
    else:
        df = pd.DataFrame(all_rows)
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest with {len(df)} rows -> {manifest_path}")
    print(f"Done in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
