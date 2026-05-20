#!/usr/bin/env python3
"""Evaluate a trained student model and render side-by-side videos.

Reads a checkpoint produced by `train_student.py`. For each evaluation bag
(by default: the held-out test set; optionally one extra random training
recording for sanity-check), runs the model on every cached frame, writes a
`<scenario>__<bag_stem>__student.parquet` annotation, prints metrics against
the original training source, and renders an overlay video that compares
ground-truth (training source) and student predictions side by side.

Usage:
    python scripts/eval_student.py --checkpoint checkpoints/vlm/best.pt
    python scripts/eval_student.py --checkpoint checkpoints/heuristic/best.pt --include-training-sample 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import (
    annotation_path,
    load_all_annotations,
    score100_to_5,
    to_typed,
    write_annotations,
)
from src.student.dataset import default_eval_transform, load_manifest
from src.student.model import build_default_model
from src.video_writer import H264Writer


def _load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})
    backbone = saved_args.get("backbone", "mobilenetv3_small_100")
    model = build_default_model(backbone_name=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    return model, ckpt


def _predict_for_bag(
    model,
    device,
    cache_root: Path,
    manifest: pd.DataFrame,
    scenario: str,
    bag_stem: str,
    transform,
    batch: int = 32,
) -> pd.DataFrame:
    rows = manifest[(manifest["scenario"] == scenario) & (manifest["bag_stem"] == bag_stem)]
    rows = rows.sort_values("frame_idx").reset_index(drop=True)
    if len(rows) == 0:
        return pd.DataFrame()

    preds: list[float] = []
    timestamps: list[float] = []
    frame_indices: list[int] = []

    buf_x: list[torch.Tensor] = []
    buf_meta: list[tuple[int, float]] = []

    def flush():
        if not buf_x:
            return
        x = torch.stack(buf_x, 0).to(device, non_blocking=True)
        with torch.no_grad():
            p = model(x)["score"].cpu().numpy()
        for i, (fi, ts) in enumerate(buf_meta):
            preds.append(float(p[i]))
            frame_indices.append(int(fi))
            timestamps.append(float(ts))
        buf_x.clear()
        buf_meta.clear()

    for _, r in rows.iterrows():
        path = r["jpeg_path"]
        full = path if Path(path).is_absolute() else (cache_root / path)
        img = cv2.imread(str(full), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = transform(img_rgb)
        buf_x.append(x)
        buf_meta.append((int(r["frame_idx"]), float(r["timestamp_s"])))
        if len(buf_x) >= batch:
            flush()
    flush()

    return pd.DataFrame({
        "timestamp_s": timestamps,
        "frame_idx": frame_indices,
        "comfort_score": preds,
    })


def _metrics(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    if len(pred) == 0 or len(gt) == 0:
        return {"n": 0}
    pred = pred.sort_values("timestamp_s")
    gt = gt.sort_values("timestamp_s")
    aligned = np.interp(pred["timestamp_s"].to_numpy(),
                        gt["timestamp_s"].to_numpy(),
                        gt["comfort_score"].to_numpy())
    p = pred["comfort_score"].to_numpy()
    return {
        "n": int(len(p)),
        "mae": float(np.mean(np.abs(p - aligned))),
        "rmse": float(np.sqrt(np.mean((p - aligned) ** 2))),
        "pearson": float(np.corrcoef(p, aligned)[0, 1]) if len(p) > 1 else float("nan"),
    }


def _render_compare(
    scenario: str,
    bag_stem: str,
    cache_root: Path,
    manifest: pd.DataFrame,
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    out_path: Path,
    fps_out: float,
    width_out: int,
    source_label: str,
) -> None:
    from src.visualization import _lerp_comfort_color

    rows = manifest[(manifest["scenario"] == scenario) & (manifest["bag_stem"] == bag_stem)]
    rows = rows.sort_values("timestamp_s").reset_index(drop=True)
    if len(rows) == 0:
        return

    gt_sorted = gt.sort_values("timestamp_s")
    pred_sorted = pred.sort_values("timestamp_s")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    bar_h = 18
    header_h = 30 + (bar_h + 16) * 2 + 6

    with H264Writer(out_path, fps=fps_out) as writer:
        for _, r in rows.iterrows():
            path = r["jpeg_path"]
            full = path if Path(path).is_absolute() else (cache_root / path)
            img = cv2.imread(str(full), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if width_out is not None:
                h0, w0 = img.shape[:2]
                m = max(h0, w0)
                if m > width_out:
                    scale = width_out / m
                    img = cv2.resize(img, (int(round(w0 * scale)), int(round(h0 * scale))))
            h, w = img.shape[:2]
            canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
            canvas[header_h:header_h + h, 0:w] = img

            t = float(r["timestamp_s"])
            gt_score = float(np.interp(t,
                                        gt_sorted["timestamp_s"].to_numpy(),
                                        gt_sorted["comfort_score"].to_numpy())) if len(gt_sorted) > 0 else float("nan")
            pred_score = float(np.interp(t,
                                          pred_sorted["timestamp_s"].to_numpy(),
                                          pred_sorted["comfort_score"].to_numpy())) if len(pred_sorted) > 0 else float("nan")

            cv2.putText(canvas,
                        f"{scenario}  {bag_stem}  t={t:5.2f}s   eval source={source_label}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

            for j, (label, score) in enumerate([
                (f"GT ({source_label})", gt_score),
                ("student", pred_score),
            ]):
                y = 30 + 14 + j * (bar_h + 16)
                x = 8
                bar_w = w - 16
                cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
                if not np.isnan(score):
                    s = max(0.0, min(100.0, float(score)))
                    fill_w = int(round(bar_w * s / 100.0))
                    color = _lerp_comfort_color(s)
                    cv2.rectangle(canvas, (x, y), (x + fill_w, y + bar_h), color, -1)
                cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (200, 200, 200), 1)
                tau_x = x + int(bar_w * 80.0 / 100.0)
                cv2.line(canvas, (tau_x, y - 2), (tau_x, y + bar_h + 2), (240, 240, 240), 1)
                text = "--" if np.isnan(score) else f"{score:5.1f}"
                cv2.putText(canvas, f"{label}: {text}", (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)

            writer.write(canvas)
    try:
        rel = out_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_path
    print(f"  wrote eval video -> {rel}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained comfort regressor.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache-root", type=str, default=str(PROJECT_ROOT / "frame_cache"))
    parser.add_argument("--annotations-root", type=str, default=str(PROJECT_ROOT / "annotations"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "rendering_output" / "student"))
    parser.add_argument("--include-training-sample", type=int, default=1,
                        help="Also evaluate this many randomly-picked training recordings.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--render-fps", type=float, default=15.0)
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--no-render", action="store_true",
                        help="Skip the per-bag GT-vs-student comparison video.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"error: checkpoint not found: {ckpt_path}")
        return 1

    print(f"loading {ckpt_path}")
    model, ckpt = _load_checkpoint(ckpt_path, device)
    saved_args = ckpt.get("args", {})
    source = saved_args.get("source", "vlm")
    transform = default_eval_transform(saved_args.get("input_size", 224))

    print(f"  source={source}")
    test_stems = [tuple(x) for x in ckpt["split"]["test_stems"]]
    train_stems = [tuple(x) for x in ckpt["split"]["train_stems"]]
    print(f"  test bags: {test_stems}")

    eval_targets = list(test_stems)
    if args.include_training_sample > 0 and train_stems:
        rng = np.random.default_rng(args.seed)
        n = min(args.include_training_sample, len(train_stems))
        idxs = rng.choice(len(train_stems), size=n, replace=False)
        for i in idxs:
            eval_targets.append(train_stems[int(i)])

    cache_root = Path(args.cache_root).resolve()
    manifest = load_manifest(cache_root)
    gt_table = load_all_annotations(Path(args.annotations_root), source=source)

    out_dir = Path(args.out_dir) / source
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"source": source, "checkpoint": str(ckpt_path), "results": []}

    for scen, stem in eval_targets:
        is_test = (scen, stem) in set(test_stems)
        tag = "test" if is_test else "train_sample"
        print(f"\n[{tag}] {scen}/{stem}")
        t0 = time.time()
        pred = _predict_for_bag(
            model, device, cache_root, manifest, scen, stem,
            transform=transform, batch=args.batch,
        )
        if len(pred) == 0:
            print("  ! no cached frames; skipping")
            continue
        gt = gt_table[(gt_table["scenario"] == scen) & (gt_table["bag_stem"] == stem)]

        out_pred = pred.copy()
        out_pred["comfort_score_5"] = score100_to_5(out_pred["comfort_score"].to_numpy())
        out_pred["confidence"] = np.nan
        out_pred["rationale"] = ""
        out_pred["source"] = "student"
        out_pred["scenario"] = scen
        out_pred["bag_stem"] = stem
        out_pred["phase"] = ""
        ann_out_path = annotation_path(Path(args.annotations_root), scen, stem, "student")
        write_annotations(to_typed(out_pred), ann_out_path)

        metrics = _metrics(pred, gt)
        print(f"  metrics vs {source}: mae={metrics.get('mae', float('nan')):.2f}  "
              f"rmse={metrics.get('rmse', float('nan')):.2f}  "
              f"pearson={metrics.get('pearson', float('nan')):.3f}  "
              f"({time.time() - t0:.1f}s)")

        if args.no_render:
            video_rel = ""
        else:
            video_path = out_dir / f"{scen}__{stem}__{tag}.mp4"
            _render_compare(
                scen, stem, cache_root, manifest, gt, pred, video_path,
                fps_out=args.render_fps, width_out=args.render_width,
                source_label=source,
            )
            try:
                video_rel = str(video_path.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                video_rel = str(video_path)
        summary["results"].append({
            "scenario": scen, "bag_stem": stem, "tag": tag,
            "metrics": metrics, "video": video_rel,
        })

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
