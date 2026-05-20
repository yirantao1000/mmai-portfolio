#!/usr/bin/env python3
"""Train the student comfort regressor.

Reads annotations from `--source` (vlm | heuristic | human), splits by
recording (random hold-out of N bags, with sc02 + sc04 guaranteed in the
test set so we sanity-check both abort and continue), trains, and saves a
checkpoint + manifest of train/test bags.

Usage:
    python scripts/train_student.py --source vlm
    python scripts/train_student.py --source heuristic --epochs 12 --batch 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import load_all_annotations
from src.student.dataset import (
    ComfortFrameDataset,
    build_split,
    collate_with_meta,
    default_eval_transform,
    default_train_transform,
)
from src.student.model import build_default_model


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for x, y, _meta in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)["score"]
            all_pred.append(pred.cpu().numpy())
            all_gt.append(y.cpu().numpy())
    if not all_pred:
        return {"n": 0}
    pred = np.concatenate(all_pred)
    gt = np.concatenate(all_gt)
    return {
        "n": int(len(gt)),
        "mae": float(np.mean(np.abs(pred - gt))),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "pearson": float(np.corrcoef(pred, gt)[0, 1]) if len(gt) > 1 else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train comfort regressor on per-source annotations.")
    parser.add_argument("--source", type=str, required=True, choices=["vlm", "heuristic", "human"],
                        help="Annotation source to train on.")
    parser.add_argument("--annotations-root", type=str, default=str(PROJECT_ROOT / "annotations"))
    parser.add_argument("--cache-root", type=str, default=str(PROJECT_ROOT / "frame_cache"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--n-test", type=int, default=2,
                        help="Number of held-out test recordings (ignored when --split-file is set).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-file", type=str, default=None,
                        help="Optional path to a JSON split (see scripts/make_split.py). "
                             "When set, --n-test / --seed are ignored.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-fps", type=float, default=None,
                        help="Optional decimation: keep only annotations whose timestamps "
                             "are at least 1/x apart (useful for the dense heuristic source).")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir) / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[source={args.source}] loading annotations ...")
    ann = load_all_annotations(Path(args.annotations_root), source=args.source)
    if len(ann) == 0:
        print(f"error: no annotations found at {args.annotations_root}/<scenario>/*__{args.source}.parquet")
        return 1

    if args.max_train_fps is not None and args.max_train_fps > 0:
        period = 1.0 / args.max_train_fps
        keep_idx = []
        for (scen, stem), grp in ann.groupby(["scenario", "bag_stem"]):
            grp = grp.sort_values("timestamp_s")
            last_t = -1e9
            for idx, row in grp.iterrows():
                if row["timestamp_s"] - last_t >= period - 1e-6:
                    keep_idx.append(idx)
                    last_t = float(row["timestamp_s"])
        ann = ann.loc[keep_idx]
        print(f"  decimated to {len(ann)} rows @ ~{args.max_train_fps} Hz")

    if args.split_file:
        with open(args.split_file) as f:
            split_data = json.load(f)
        from src.student.dataset import SplitSpec
        split = SplitSpec(
            train_stems=[tuple(x) for x in split_data["all_train_stems"]],
            test_stems=[tuple(x) for x in split_data["all_test_stems"]],
        )
        print(f"  using split file: {args.split_file} (name={split_data.get('name','?')})")
    else:
        split = build_split(ann, n_test=args.n_test, seed=args.seed)
    print(f"  test bags: {split.test_stems}")
    print(f"  train bags: {len(split.train_stems)} recordings")

    train_tx = default_train_transform(args.input_size)
    eval_tx = default_eval_transform(args.input_size)

    train_ds = ComfortFrameDataset(
        annotations=ann, frame_cache_root=Path(args.cache_root),
        keep_stems=split.train_stems, transform=train_tx,
    )
    test_ds = ComfortFrameDataset(
        annotations=ann, frame_cache_root=Path(args.cache_root),
        keep_stems=split.test_stems, transform=eval_tx,
    )
    print(f"  train rows: {len(train_ds)}   test rows: {len(test_ds)}")
    if len(train_ds) == 0 or len(test_ds) == 0:
        print("error: empty train or test set; did you run prepare_frames.py and the annotators?")
        return 1

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=collate_with_meta, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        collate_fn=collate_with_meta, pin_memory=(device.type == "cuda"),
    )

    model = build_default_model(
        backbone_name=args.backbone, pretrained=not args.no_pretrained,
    ).to(device)
    print(f"  model: {args.backbone}, params={model.num_parameters() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss(beta=5.0)   # Huber, robust to label noise

    history = []
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for x, y, _meta in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)["score"]
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        scheduler.step()
        train_loss = running / max(1, n)

        metrics = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        elapsed = time.time() - t0
        print(f"  epoch {epoch:>2}/{args.epochs}  loss={train_loss:.3f}  "
              f"test mae={metrics.get('mae', float('nan')):.2f}  "
              f"rmse={metrics.get('rmse', float('nan')):.2f}  "
              f"pearson={metrics.get('pearson', float('nan')):.3f}  "
              f"({elapsed:.1f}s)",
              flush=True)

        is_best = metrics.get("mae", float("inf")) < best_mae
        if is_best:
            best_mae = metrics["mae"]
            ckpt_path = out_dir / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "metrics": metrics,
                "split": {
                    "train_stems": split.train_stems,
                    "test_stems": split.test_stems,
                },
                "history": history,
            }, ckpt_path)

    last_path = out_dir / "last.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "epoch": args.epochs,
        "metrics": history[-1] if history else {},
        "split": {"train_stems": split.train_stems, "test_stems": split.test_stems},
        "history": history,
    }, last_path)

    best_path = out_dir / "best.pt"
    try:
        ckpt_rel = str(best_path.relative_to(PROJECT_ROOT))
    except ValueError:
        ckpt_rel = str(best_path)
    summary = {
        "source": args.source,
        "best_mae": best_mae,
        "final_metrics": history[-1] if history else {},
        "split": {"train_stems": split.train_stems, "test_stems": split.test_stems},
        "checkpoint": ckpt_rel,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nbest MAE: {best_mae:.2f}")
    print(f"checkpoints saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
