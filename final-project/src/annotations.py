"""Shared annotation IO and bag-frame extraction utilities.

Annotation file layout:
    annotations/<scenario>/<bag_stem>__<source>.parquet

Schema (every annotator writes the same columns; missing values are NaN):
    timestamp_s        float32   # seconds from bag start
    frame_idx          int32     # frame index in the original bag
    comfort_score      float32   # 0-100, normalized comfort
    comfort_score_5    float32   # 1-5 ordinal (raw VLM/human scale; NaN for heuristic)
    confidence         float32   # 0-1 (NaN if not produced)
    rationale          string    # short text (VLM rationale, human note); empty otherwise
    source             string    # "heuristic" | "vlm" | "human"
    scenario           string    # e.g. "sc02_comfortable"
    bag_stem           string    # original .bag file stem
    phase              string    # "approach" | "intent" | "execution" | "" (no sidecar)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


SCHEMA = {
    "timestamp_s": "float32",
    "frame_idx": "int32",
    "comfort_score": "float32",
    "comfort_score_5": "float32",
    "confidence": "float32",
    "rationale": "string",
    "source": "string",
    "scenario": "string",
    "bag_stem": "string",
    "phase": "string",
}


def empty_table() -> pd.DataFrame:
    df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in SCHEMA.items()})
    return df


def to_typed(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DataFrame to the canonical schema (fills missing columns with NaN)."""
    out = empty_table()
    for col, dt in SCHEMA.items():
        if col in df.columns:
            try:
                out[col] = df[col].astype(dt)
            except (TypeError, ValueError):
                out[col] = df[col]
        else:
            out[col] = pd.Series([np.nan] * len(df), dtype=dt)
    return out


def annotation_path(annotations_root: Path, scenario: str, bag_stem: str, source: str) -> Path:
    return Path(annotations_root) / scenario / f"{bag_stem}__{source}.parquet"


def write_annotations(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_typed(df).to_parquet(out_path, index=False)


def read_annotations(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_all_annotations(
    annotations_root: Path,
    source: str,
    scenarios: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Concatenate all annotations for a given source into one DataFrame."""
    root = Path(annotations_root)
    if not root.exists():
        return empty_table()
    parts = []
    for scen_dir in sorted(root.iterdir()):
        if not scen_dir.is_dir():
            continue
        if scenarios is not None and scen_dir.name not in set(scenarios):
            continue
        for f in sorted(scen_dir.glob(f"*__{source}.parquet")):
            parts.append(read_annotations(f))
    if not parts:
        return empty_table()
    return pd.concat(parts, ignore_index=True)


# -------- 0–100 ↔ 1–5 conversions ----------------------------------------------

def score5_to_100(x: float | np.ndarray) -> float | np.ndarray:
    """Linear map 1..5 -> 0..100. Values outside [1,5] are clipped."""
    arr = np.asarray(x, dtype=np.float32)
    arr = np.clip(arr, 1.0, 5.0)
    return ((arr - 1.0) / 4.0) * 100.0


def score100_to_5(x: float | np.ndarray) -> float | np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    arr = np.clip(arr, 0.0, 100.0)
    return 1.0 + (arr / 100.0) * 4.0


# -------- Bag frame extraction (shared by VLM + human + render) ----------------

@dataclass
class FrameSample:
    frame_idx: int
    timestamp_s: float
    bgr: np.ndarray   # H x W x 3, BGR uint8


def iter_bag_frames(
    bag_path: str | Path,
    start_s: float | None = None,
    end_s: float | None = None,
    target_fps: float | None = None,
    resize_max_side: int | None = None,
) -> Iterator[FrameSample]:
    """Yield FrameSample objects from a .bag file.

    Parameters
    ----------
    target_fps : if given, decimate to roughly this fps (defaults to native ~30 fps)
    resize_max_side : if given, resize so max(h, w) == this (preserves aspect ratio)
    """
    import cv2  # local import to avoid pulling cv2 when unused
    from .bag_source import BagSource

    src = BagSource(str(bag_path), real_time=False)
    if not src.open():
        return

    last_emit_s = -1e9
    period = (1.0 / target_fps) if target_fps else 0.0
    frame_idx = 0
    try:
        while True:
            ok, color, _depth, ts_ms = src.read()
            if not ok or color is None:
                break
            t_s = ts_ms / 1000.0
            frame_idx += 1
            if start_s is not None and t_s < start_s:
                continue
            if end_s is not None and t_s > end_s:
                break
            if period > 0 and (t_s - last_emit_s) < period - 1e-6:
                continue
            last_emit_s = t_s

            img = color
            if resize_max_side is not None:
                h, w = img.shape[:2]
                m = max(h, w)
                if m > resize_max_side:
                    scale = resize_max_side / m
                    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))

            yield FrameSample(frame_idx=frame_idx - 1, timestamp_s=float(t_s), bgr=img)
    finally:
        src.release()


def find_bags(data_root: Path, scenario_filter: list[str] | None = None) -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        return []
    bags = sorted(root.rglob("*.bag"))
    if scenario_filter is None:
        return bags
    keep = set(scenario_filter)
    return [b for b in bags if b.relative_to(root).parts[0] in keep]


def scenario_of(bag_path: Path, data_root: Path) -> str:
    """e.g. data/sc02_comfortable/Bright/foo.bag -> 'sc02_comfortable'."""
    return Path(bag_path).resolve().relative_to(Path(data_root).resolve()).parts[0]
