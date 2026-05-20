#!/usr/bin/env python3
"""Heuristic annotator — runs the existing IntegratedPipeline over each .bag and
dumps per-frame integrated_comfort_score as an annotation file.

Output: annotations/<scenario>/<bag_stem>__heuristic.parquet  (one row per video frame).

Usage:
    python scripts/annotate_heuristic.py
    python scripts/annotate_heuristic.py --only sc02_comfortable sc04_sudden_withdrawal
    python scripts/annotate_heuristic.py --config config/deploy.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import (
    annotation_path,
    find_bags,
    score100_to_5,
    scenario_of,
    to_typed,
    write_annotations,
)
from src.bag_source import BagSource
from src.phases import find_sidecar, phase_at, windows_from_sidecar
from src.pipeline import IntegratedPipeline


def annotate_one(bag_path: Path, pipeline: IntegratedPipeline, data_root: Path) -> pd.DataFrame:
    sidecar = find_sidecar(bag_path)
    windows = windows_from_sidecar(sidecar) if sidecar is not None else None

    src = BagSource(str(bag_path), real_time=False)
    if not src.open():
        print(f"  ! could not open {bag_path}")
        return None

    pipeline.reset_state()
    rows: list[dict] = []
    scen = scenario_of(bag_path, data_root)
    stem = bag_path.stem
    frame_idx = 0
    try:
        while True:
            ok, color, depth, ts_ms = src.read()
            if not ok or color is None:
                break
            t_s = ts_ms / 1000.0
            phase = phase_at(windows, t_s) if windows is not None else "intent"
            pipeline.set_phase(phase)
            result = pipeline.process_frame(color, ts_ms, depth_frame=depth)

            rows.append({
                "timestamp_s": float(t_s),
                "frame_idx": int(frame_idx),
                "comfort_score": float(result.integrated_comfort_score),
                "comfort_score_5": float(score100_to_5(result.integrated_comfort_score)),
                "confidence": float(np.nan),
                "rationale": "",
                "source": "heuristic",
                "scenario": scen,
                "bag_stem": stem,
                "phase": phase,
            })
            frame_idx += 1
    finally:
        src.release()

    if not rows:
        return None
    return to_typed(pd.DataFrame(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Heuristic annotator (current pipeline -> per-frame score).")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "config" / "deploy.yaml"))
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--annotations-root", type=str, default=str(PROJECT_ROOT / "annotations"))
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional list of scenarios to process (e.g. sc02_comfortable).")
    parser.add_argument("--force", action="store_true", help="Re-annotate even if output exists.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.annotations_root)

    bags = find_bags(data_root, args.only)
    if not bags:
        print(f"No .bag files found under {data_root}")
        return 1

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Found {len(bags)} bags. Loading pipeline (config={args.config})...")
    pipeline = IntegratedPipeline(config)
    pipeline.load_models()

    t0 = time.time()
    written = skipped = failed = 0
    for i, bag in enumerate(bags, 1):
        scen = scenario_of(bag, data_root)
        out_path = annotation_path(out_root, scen, bag.stem, "heuristic")
        if out_path.exists() and not args.force:
            print(f"  [{i}/{len(bags)}] {scen}/{bag.name} — already annotated, skipping")
            skipped += 1
            continue
        print(f"  [{i}/{len(bags)}] annotating {scen}/{bag.name}...", flush=True)
        df = annotate_one(bag, pipeline, data_root)
        if df is None:
            failed += 1
            continue
        write_annotations(df, out_path)
        print(f"      wrote {len(df)} rows -> {out_path.relative_to(PROJECT_ROOT)}")
        written += 1

    print(f"\nDone in {time.time() - t0:.1f}s. wrote={written}, skipped={skipped}, failed={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
