#!/usr/bin/env python3
"""Summarize v2 test-set performance: pairwise agreement among heuristic,
vlm, and student on the 10 held-out bags. Joins on nearest cached frame.

Outputs a markdown table to stdout and writes `reports/v2_test_summary.json`.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import read_annotations, annotation_path  # noqa: E402


def pair_metrics(a: pd.DataFrame, b: pd.DataFrame, tol_s: float = 0.20) -> dict:
    """Join a and b on nearest timestamp_s (forward+backward, tolerance tol_s)."""
    if len(a) == 0 or len(b) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "n": 0}
    a_sorted = a[["timestamp_s", "comfort_score"]].sort_values("timestamp_s").reset_index(drop=True)
    b_sorted = b[["timestamp_s", "comfort_score"]].sort_values("timestamp_s").reset_index(drop=True)
    a_sorted["_t"] = a_sorted["timestamp_s"]
    b_sorted["_t"] = b_sorted["timestamp_s"]
    j = pd.merge_asof(
        a_sorted.rename(columns={"comfort_score": "a"}),
        b_sorted.rename(columns={"comfort_score": "b"}),
        on="_t", direction="nearest", tolerance=tol_s,
    ).dropna(subset=["a", "b"])
    if len(j) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "n": 0}
    err = j["a"].to_numpy() - j["b"].to_numpy()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if len(j) > 1 and j["a"].std() > 1e-6 and j["b"].std() > 1e-6:
        pearson = float(np.corrcoef(j["a"], j["b"])[0, 1])
    else:
        pearson = float("nan")
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "n": int(len(j))}


def main() -> int:
    split_path = PROJECT_ROOT / "splits" / "v2.json"
    ann_root = PROJECT_ROOT / "annotations"
    with open(split_path) as f:
        split = json.load(f)
    test_pairs = [tuple(x) for x in split["all_test_stems"]]

    rows = []
    for scen, stem in test_pairs:
        h_path = annotation_path(ann_root, scen, stem, "heuristic")
        v_path = annotation_path(ann_root, scen, stem, "vlm")
        s_path = annotation_path(ann_root, scen, stem, "student")
        if not (h_path.exists() and v_path.exists() and s_path.exists()):
            print(f"  skip {scen}/{stem}: missing one of heuristic/vlm/student")
            continue
        h = read_annotations(h_path)
        v = read_annotations(v_path)
        s = read_annotations(s_path)

        m_hv = pair_metrics(h, v)
        m_sh = pair_metrics(s, h)
        m_sv = pair_metrics(s, v)
        rows.append({
            "scenario": scen, "bag_stem": stem,
            "session": "2026-04" if stem.startswith("2026-04") else "2026-05",
            "heur_mean": float(h["comfort_score"].mean()),
            "vlm_mean": float(v["comfort_score"].mean()),
            "stu_mean": float(s["comfort_score"].mean()),
            "h_vs_v_mae": m_hv["mae"], "h_vs_v_r": m_hv["pearson"],
            "s_vs_h_mae": m_sh["mae"], "s_vs_h_r": m_sh["pearson"],
            "s_vs_v_mae": m_sv["mae"], "s_vs_v_r": m_sv["pearson"],
        })

    df = pd.DataFrame(rows)

    print("\n[per-bag pairwise agreement, MAE / Pearson r on 0-100 scale]\n")
    print(f"{'scenario':<27} {'stem':<22} {'session':<8} "
          f"{'meanH':>5} {'meanV':>5} {'meanS':>5} "
          f"{'H<>V':>10} {'S<>H':>10} {'S<>V':>10}")
    for _, r in df.iterrows():
        print(f"{r['scenario']:<27} {r['bag_stem']:<22} {r['session']:<8} "
              f"{r['heur_mean']:>5.1f} {r['vlm_mean']:>5.1f} {r['stu_mean']:>5.1f} "
              f"{r['h_vs_v_mae']:>4.1f}/{r['h_vs_v_r']:>4.2f} "
              f"{r['s_vs_h_mae']:>4.1f}/{r['s_vs_h_r']:>4.2f} "
              f"{r['s_vs_v_mae']:>4.1f}/{r['s_vs_v_r']:>4.2f}")

    def agg(sub: pd.DataFrame, label: str) -> None:
        n = len(sub)
        print(f"\n[{label}  n={n} bags]")
        for key in ["h_vs_v_mae", "h_vs_v_r", "s_vs_h_mae", "s_vs_h_r", "s_vs_v_mae", "s_vs_v_r"]:
            vals = sub[key].dropna().to_numpy()
            if len(vals) == 0:
                continue
            print(f"  {key:<14}  mean={vals.mean():.2f}  median={np.median(vals):.2f}")

    agg(df, "all test bags")
    agg(df[df.session == "2026-04"], "session 2026-04 (in-distribution)")
    agg(df[df.session == "2026-05"], "session 2026-05 (cross-session)")

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    import os as _os
    tag = _os.environ.get("SUMMARY_TAG", "v2")
    out_path = out_dir / f"{tag}_test_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "split": str(split_path.relative_to(PROJECT_ROOT)),
            "rows": df.to_dict("records"),
        }, f, indent=2)
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
