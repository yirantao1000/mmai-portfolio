#!/usr/bin/env python3
"""Render annotated videos: draw per-source comfort scores on top of each .bag.

Each source's score appears as a colored top bar with the numeric value above it.
Multiple sources stack vertically so VLM and heuristic (and human) can be compared
side-by-side over the same frames.

Outputs:
    rendering_output/<scenario>__<bag_stem>__<sources>.mp4
By default uses a low resolution (480x360) and modest bitrate to save disk.

Usage:
    python scripts/render_annotations.py                                # all bags, all sources found
    python scripts/render_annotations.py --sources vlm heuristic        # explicit
    python scripts/render_annotations.py --only sc02_comfortable
    python scripts/render_annotations.py --width 480 --fps 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import (
    annotation_path,
    find_bags,
    iter_bag_frames,
    scenario_of,
)
from src.phases import find_sidecar, phase_at, windows_from_sidecar
from src.video_writer import H264Writer
from src.visualization import _lerp_comfort_color


SOURCE_LABEL = {
    "heuristic": "heuristic",
    "vlm":       "VLM",
    "human":     "human",
    "student":   "student",
}


def load_source_table(annotations_root: Path, scenario: str, bag_stem: str, source: str) -> pd.DataFrame | None:
    p = annotation_path(annotations_root, scenario, bag_stem, source)
    if not p.exists():
        return None
    return pd.read_parquet(p).sort_values("timestamp_s").reset_index(drop=True)


def interp_score(table: pd.DataFrame | None, t_s: float) -> float:
    if table is None or len(table) == 0:
        return float("nan")
    ts = table["timestamp_s"].to_numpy(dtype=np.float32)
    sc = table["comfort_score"].to_numpy(dtype=np.float32)
    if t_s <= ts[0]:
        return float(sc[0])
    if t_s >= ts[-1]:
        return float(sc[-1])
    return float(np.interp(t_s, ts, sc))


def draw_score_bar(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                   score: float, label: str, tau: float = 80.0) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 50, 50), -1)
    if not np.isnan(score):
        s = max(0.0, min(100.0, float(score)))
        fill_w = int(round(w * s / 100.0))
        color = _lerp_comfort_color(s)
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)
    tau_x = x + int(w * tau / 100.0)
    cv2.line(canvas, (tau_x, y - 2), (tau_x, y + h + 2), (240, 240, 240), 1)
    text_score = "--" if np.isnan(score) else f"{score:5.1f}"
    cv2.putText(canvas, f"{label}: {text_score}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)


def render_one(
    bag_path: Path,
    data_root: Path,
    annotations_root: Path,
    sources: list[str],
    out_path: Path,
    fps_out: float,
    width_out: int,
) -> None:
    scen = scenario_of(bag_path, data_root)
    sidecar = find_sidecar(bag_path)
    windows = windows_from_sidecar(sidecar) if sidecar is not None else None

    tables = {s: load_source_table(annotations_root, scen, bag_path.stem, s) for s in sources}
    available = [s for s in sources if tables[s] is not None]
    if not available:
        print(f"  ! no annotations found for {bag_path.name}; skipping")
        return

    frames_iter = iter_bag_frames(bag_path, target_fps=fps_out, resize_max_side=width_out)

    bar_h = 18
    pad = 6
    header_h = 30 + (bar_h + 16) * len(available) + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with H264Writer(out_path, fps=fps_out) as writer:
        for sample in frames_iter:
            img = sample.bgr
            h, w = img.shape[:2]
            canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
            canvas[header_h:header_h + h, 0:w] = img

            phase = phase_at(windows, sample.timestamp_s) if windows is not None else ""
            top_text = (f"{scen}  {bag_path.stem}  t={sample.timestamp_s:5.2f}s"
                        f"   phase={phase}")
            cv2.putText(canvas, top_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (235, 235, 235), 1, cv2.LINE_AA)

            y = 30 + 14
            bar_w = w - 16
            for s in available:
                score = interp_score(tables[s], sample.timestamp_s)
                draw_score_bar(canvas, x=8, y=y, w=bar_w, h=bar_h,
                               score=score, label=SOURCE_LABEL.get(s, s))
                y += bar_h + 16

            writer.write(canvas)
        n_written = writer.n_written

    try:
        rel = out_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_path
    print(f"  wrote {n_written} frames -> {rel}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render annotated videos with score overlays.")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--annotations-root", type=str, default=str(PROJECT_ROOT / "annotations"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "rendering_output"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--sources", nargs="+",
                        default=["heuristic", "vlm", "human", "student"],
                        help="Score sources to render. Missing ones are skipped silently.")
    parser.add_argument("--fps", type=float, default=15.0,
                        help="Output video FPS (default 15 to save space).")
    parser.add_argument("--width", type=int, default=480,
                        help="Resize so max(h,w)==this. Lower = smaller files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only render the first N bags (debug).")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Optional split JSON; combined with --split-set to filter bags.")
    parser.add_argument("--split-set", type=str, default="test", choices=["train", "test"],
                        help="Which side of --split-file to render.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir)
    bags = find_bags(data_root, args.only)

    if args.split_file:
        import json as _json
        with open(args.split_file) as f:
            split_data = _json.load(f)
        key = "all_test_stems" if args.split_set == "test" else "all_train_stems"
        keep_keys = {(s, t) for s, t in split_data.get(key, [])}
        bags = [b for b in bags if (scenario_of(b, data_root), b.stem) in keep_keys]
        print(f"  filtered to {len(bags)} bags via --split-file (set={args.split_set})")

    if args.limit:
        bags = bags[:args.limit]
    if not bags:
        print(f"No bags under {data_root}.")
        return 1

    for i, bag in enumerate(bags, 1):
        scen = scenario_of(bag, data_root)
        suffix = "_".join(args.sources)
        out_path = out_dir / f"{scen}__{bag.stem}__{suffix}.mp4"
        print(f"[{i}/{len(bags)}] {scen}/{bag.name}")
        render_one(
            bag, data_root, Path(args.annotations_root),
            sources=args.sources, out_path=out_path,
            fps_out=args.fps, width_out=args.width,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
