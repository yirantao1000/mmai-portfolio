#!/usr/bin/env python3
"""Render one or more `.bag` files to H.264 MP4s with a lightweight overlay
that visualizes the event-time labels in the sidecar JSON.

This is meant for quick spot-checking of new recordings — see what the
camera saw and where each labelled event falls on a single-stripe timeline.

For each frame the overlay shows:
  - top-left: scenario  bag_stem  t=XX.XXs  phase=approach|intent|execution|--
  - timeline strip: a horizontal bar from t=0 to t=duration with colored
    vertical markers at start_time / signal_time / handover_time / abort_time /
    end_time; the current playback position is shown as a moving white tick.

Usage examples:

    # Render the 5 sample bags from the new data
    python scripts/preview_bag.py \\
        data/sc01_walkby/RawData_unlabelled_bagfiles/2026-05-08_16-24-50.bag \\
        data/sc02_comfortable/RawData_unlabelled_bagfiles/2026-05-08_16-27-50.bag \\
        --out-dir renders/v2_new_data_preview

    # Render every bag in a directory tree
    python scripts/preview_bag.py data/sc02_comfortable --recursive
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import iter_bag_frames
from src.phases import find_sidecar, phase_at, windows_from_sidecar
from src.video_writer import H264Writer


# Colors (BGR)
COLOR_BG       = (35, 35, 35)
COLOR_TXT      = (235, 235, 235)
COLOR_DIM      = (140, 140, 140)
COLOR_TIMELINE = (60, 60, 60)
COLOR_CURSOR   = (255, 255, 255)
EVENT_COLORS = {
    "start_time":    (180, 180,  60),   # cyan-ish
    "signal_time":   ( 60, 200, 240),   # yellow
    "handover_time": ( 60, 220,  60),   # green
    "abort_time":    ( 80,  60, 230),   # red
    "end_time":      (200, 120, 200),   # magenta
}


def scenario_of_bag(bag_path: Path, project_root: Path) -> str:
    """Return e.g. 'sc02_comfortable' given a bag path under data/."""
    try:
        rel = bag_path.relative_to(project_root / "data")
        return rel.parts[0]
    except ValueError:
        return bag_path.parent.parent.name


def draw_timeline(
    canvas: np.ndarray,
    x: int, y: int, w: int, h: int,
    duration_s: float,
    labels: dict,
    cursor_t: float,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_TIMELINE, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)

    def t2x(t: float) -> int:
        if duration_s <= 0:
            return x
        return x + int(round(w * max(0.0, min(1.0, t / duration_s))))

    # Event markers + small text labels
    short_names = {
        "start_time": "S",
        "signal_time": "sig",
        "handover_time": "H",
        "abort_time": "A",
        "end_time": "E",
    }
    for k, c in EVENT_COLORS.items():
        v = labels.get(k)
        if v is None:
            continue
        mx = t2x(float(v))
        cv2.line(canvas, (mx, y - 2), (mx, y + h + 2), c, 2)
        cv2.putText(canvas, short_names.get(k, k), (mx - 8, y + h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA)

    # Cursor
    cx = t2x(float(cursor_t))
    cv2.line(canvas, (cx, y - 4), (cx, y + h + 4), COLOR_CURSOR, 2)


def render_one(bag_path: Path, out_path: Path, fps_out: float, width_out: int) -> None:
    scen = scenario_of_bag(bag_path, PROJECT_ROOT)
    sidecar = find_sidecar(bag_path)
    labels: dict = {}
    duration_s: float = 0.0
    scenario_label = scen
    windows = None
    if sidecar is not None:
        with open(sidecar) as f:
            meta = json.load(f)
        labels = meta.get("labels", {}) or {}
        duration_s = float(meta.get("duration_seconds", 0.0) or 0.0)
        scenario_label = f"{scen}  ({meta.get('scenario_code', '?')})"
        windows = windows_from_sidecar(sidecar)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_h = 56     # space for top text + timeline + event tick labels
    margin = 8

    with H264Writer(out_path, fps=fps_out) as writer:
        for sample in iter_bag_frames(bag_path, target_fps=fps_out, resize_max_side=width_out):
            img = sample.bgr
            h, w = img.shape[:2]
            canvas = np.full((h + header_h, w, 3), COLOR_BG, dtype=np.uint8)
            canvas[header_h:header_h + h, 0:w] = img

            phase = phase_at(windows, sample.timestamp_s) if windows is not None else "--"
            top = (f"{scenario_label}   {bag_path.stem}   "
                   f"t={sample.timestamp_s:5.2f}s   phase={phase}")
            cv2.putText(canvas, top, (margin, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TXT, 1, cv2.LINE_AA)

            # Effective duration: prefer JSON value, otherwise use cursor for now
            eff_dur = duration_s if duration_s > 0 else max(sample.timestamp_s, 0.1)
            draw_timeline(
                canvas,
                x=margin, y=24, w=w - 2 * margin, h=10,
                duration_s=eff_dur, labels=labels,
                cursor_t=sample.timestamp_s,
            )

            writer.write(canvas)
        n = writer.n_written

    try:
        rel = out_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_path
    print(f"  wrote {n} frames -> {rel}  ({out_path.stat().st_size/1024:.0f} KB)")


def resolve_bags(paths: list[str], recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".bag":
            out.append(path)
        elif path.is_dir():
            pattern = "**/*.bag" if recursive else "*.bag"
            out.extend(sorted(path.glob(pattern)))
        else:
            print(f"  ! not a .bag or dir: {p}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+",
                    help="One or more .bag files or directories.")
    ap.add_argument("--out-dir", default="renders/v2_new_data_preview")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into directories when looking for .bag files.")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=640,
                    help="Resize so max(h,w)==this. Default 640 keeps faces readable.")
    args = ap.parse_args()

    bags = resolve_bags(args.paths, args.recursive)
    if not bags:
        print("no .bag files found.")
        return 1

    out_dir = Path(args.out_dir)
    print(f"rendering {len(bags)} bag(s) -> {out_dir}\n")
    for i, bag in enumerate(bags, 1):
        scen = scenario_of_bag(bag, PROJECT_ROOT)
        out_path = out_dir / f"{scen}__{bag.stem}__preview.mp4"
        print(f"[{i}/{len(bags)}] {bag.name}")
        try:
            render_one(bag, out_path, fps_out=args.fps, width_out=args.width)
        except Exception as e:
            print(f"  ! failed: {e}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
