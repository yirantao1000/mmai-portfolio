"""Dataset wiring: annotations (per-source parquet) + a JPEG frame cache.

Frame cache layout (produced by scripts/prepare_frames.py):
    frame_cache/<scenario>/<bag_stem>/<frame_idx:06d>.jpg
plus a manifest CSV at frame_cache/manifest.csv with columns
    scenario, bag_stem, frame_idx, timestamp_s, jpeg_path

The dataset matches each row of an annotation parquet to the nearest cached
frame (by frame_idx) and serves (image_tensor, score, meta).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..annotations import load_all_annotations


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_train_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size + 16, input_size + 16)),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_eval_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_manifest(frame_cache_root: Path) -> pd.DataFrame:
    manifest = Path(frame_cache_root) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"manifest.csv not found at {manifest}. "
            f"Run `python scripts/prepare_frames.py` first."
        )
    df = pd.read_csv(manifest)
    return df


@dataclass
class SplitSpec:
    train_stems: list[tuple[str, str]]   # [(scenario, bag_stem), ...]
    test_stems: list[tuple[str, str]]


def build_split(
    annotations: pd.DataFrame,
    n_test: int = 2,
    seed: int = 0,
    require_test_scenarios: Sequence[str] | None = ("sc02_comfortable", "sc04_sudden_withdrawal"),
) -> SplitSpec:
    """Random hold-out by recording, but optionally guarantee that the test
    set contains at least one recording from each `require_test_scenarios`
    so we can sanity-check abort + continue at eval time."""
    rng = np.random.default_rng(seed)
    keys = list({(s, b) for s, b in zip(annotations["scenario"], annotations["bag_stem"])})
    if not keys:
        return SplitSpec(train_stems=[], test_stems=[])

    keys_sorted = sorted(keys)
    test_set: list[tuple[str, str]] = []

    if require_test_scenarios:
        by_scen: dict[str, list[tuple[str, str]]] = {}
        for sc, st in keys_sorted:
            by_scen.setdefault(sc, []).append((sc, st))
        for scen in require_test_scenarios:
            options = by_scen.get(scen, [])
            if not options:
                continue
            pick = options[int(rng.integers(0, len(options)))]
            test_set.append(pick)

    remaining = [k for k in keys_sorted if k not in set(test_set)]
    while len(test_set) < n_test and remaining:
        idx = int(rng.integers(0, len(remaining)))
        test_set.append(remaining.pop(idx))

    test_set = test_set[:max(n_test, len(test_set))]
    train = [k for k in keys_sorted if k not in set(test_set)]
    return SplitSpec(train_stems=train, test_stems=test_set)


class ComfortFrameDataset(Dataset):
    """Joins annotations + frame cache and serves (image, score, meta) tuples.

    Parameters
    ----------
    annotations : pd.DataFrame
        Output of `src.annotations.load_all_annotations(...)` (already filtered
        by source).
    frame_cache_root : Path to frame_cache (with manifest.csv).
    keep_stems : iterable of (scenario, bag_stem) to keep; others are dropped.
    transform : torchvision transform applied to the BGR-then-RGB-converted image.
    """

    def __init__(
        self,
        annotations: pd.DataFrame,
        frame_cache_root: Path,
        keep_stems: Sequence[tuple[str, str]] | None = None,
        transform=None,
    ):
        manifest = load_manifest(frame_cache_root)
        self._frame_cache_root = Path(frame_cache_root)

        ann = annotations.copy()
        ann = ann.dropna(subset=["comfort_score"])
        if keep_stems is not None:
            keep = set(keep_stems)
            ann = ann[ann.apply(
                lambda r: (r["scenario"], r["bag_stem"]) in keep, axis=1)]

        # First try exact join on (scenario, bag_stem, frame_idx); for the
        # rows that don't match (different sample rates between annotation
        # and cache), fall back to nearest-frame join with tolerance.
        cache_cols = manifest[["scenario", "bag_stem", "frame_idx", "jpeg_path"]]
        exact = ann.merge(cache_cols, on=["scenario", "bag_stem", "frame_idx"], how="inner")

        merged_keys = set(zip(exact["scenario"], exact["bag_stem"], exact["frame_idx"]))
        unmatched_mask = ~ann.apply(
            lambda r: (r["scenario"], r["bag_stem"], int(r["frame_idx"])) in merged_keys,
            axis=1,
        ) if len(ann) > 0 else None
        unmatched = ann[unmatched_mask] if unmatched_mask is not None else ann.iloc[0:0]
        nearest_extra = self._nearest_frame_join(unmatched, manifest) if len(unmatched) > 0 else pd.DataFrame()

        if len(nearest_extra) > 0:
            merged = pd.concat([exact, nearest_extra], ignore_index=True)
        else:
            merged = exact

        self.df = merged.reset_index(drop=True)
        if transform is None:
            transform = default_eval_transform()
        self.transform = transform

    @staticmethod
    def _nearest_frame_join(ann: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
        """Fall back to nearest-frame_idx matching when annotations sample at a
        different rate than the cache (e.g. VLM @ 2 Hz, cache @ 15 Hz)."""
        out_rows = []
        groups = manifest.groupby(["scenario", "bag_stem"])
        for (scen, stem), grp in ann.groupby(["scenario", "bag_stem"]):
            try:
                cache_grp = groups.get_group((scen, stem))
            except KeyError:
                continue
            cache_indices = cache_grp["frame_idx"].to_numpy()
            cache_paths = cache_grp["jpeg_path"].to_numpy()
            for _, row in grp.iterrows():
                fi = int(row["frame_idx"])
                pos = int(np.argmin(np.abs(cache_indices - fi)))
                if abs(int(cache_indices[pos]) - fi) > 30:  # >30 frames apart -> skip
                    continue
                row_out = row.to_dict()
                row_out["jpeg_path"] = cache_paths[pos]
                out_rows.append(row_out)
        return pd.DataFrame(out_rows) if out_rows else pd.DataFrame()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["jpeg_path"]
        full_path = path if Path(path).is_absolute() else (self._frame_cache_root / path)
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read {full_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb)
        score = float(row["comfort_score"])
        meta = {
            "scenario": row["scenario"],
            "bag_stem": row["bag_stem"],
            "frame_idx": int(row["frame_idx"]),
            "timestamp_s": float(row["timestamp_s"]),
            "phase": str(row.get("phase", "") or ""),
        }
        return x, torch.tensor(score, dtype=torch.float32), meta


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)


def load_source_dataset(
    annotations_root: Path,
    frame_cache_root: Path,
    source: str,
    keep_stems: Sequence[tuple[str, str]] | None = None,
    transform=None,
) -> ComfortFrameDataset:
    ann = load_all_annotations(annotations_root, source=source)
    return ComfortFrameDataset(
        annotations=ann,
        frame_cache_root=frame_cache_root,
        keep_stems=keep_stems,
        transform=transform,
    )
