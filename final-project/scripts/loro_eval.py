#!/usr/bin/env python3
"""Leave-one-recording-out trajectory evaluation.

For each of the 31 recordings (train + test combined), replays under the given
config and reports per-scenario slope-sign agreement with bootstrap CI.
Writes a JSON report; does NOT touch config/deploy.yaml.

Usage:
  loro_eval.py --config config/deploy.yaml --report reports/loro_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_test import DESIRED_SLOPE_SIGN, config_to_params
from scripts.optimize_params import (
    load_recordings, reduce_mean_full, replay_series, slope_fit,
)


def load_all(split_path: Path):
    return load_recordings(split_path, "train") + load_recordings(split_path, "test")


def bootstrap_rate(agreements: list[int], n_boot: int = 1000, seed: int = 42):
    if not agreements:
        return float("nan"), float("nan")
    arr = np.array(agreements, dtype=np.float32)
    rng = np.random.default_rng(seed)
    rates = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser(description="LORO trajectory evaluation.")
    parser.add_argument("--split", default=str(PROJECT_ROOT / "data" / "split.json"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    params = config_to_params(cfg)

    all_recs = load_all(Path(args.split))
    print(f"Loaded {len(all_recs)} recordings (train+test).")

    # Per-recording trajectory metrics (LORO is inherent: each recording is scored
    # independently; the "leave one out" framing matters only if params were re-fit
    # per fold. We're evaluating a fixed config, so per-rec metrics ARE LORO.)
    rows: list[dict] = []
    by_scen: dict[str, dict] = {}
    for r in all_recs:
        ts, integ = replay_series(r, params)
        if integ.size < 2:
            continue
        slope = slope_fit(ts, integ)
        mean_comfort = float(np.mean(integ))
        k = max(1, int(round(integ.size * 0.10)))
        start_mean = float(np.mean(integ[:k]))
        end_mean = float(np.mean(integ[-k:]))
        want = DESIRED_SLOPE_SIGN.get(r.scenario)
        agree = (
            int((want > 0 and slope > 0) or (want < 0 and slope < 0))
            if want is not None else None
        )
        rows.append({
            "scenario": r.scenario, "name": r.name,
            "slope": float(slope), "mean_comfort": mean_comfort,
            "start_mean": start_mean, "end_mean": end_mean,
            "slope_sign_match": agree,
        })
        d = by_scen.setdefault(r.scenario, {"slopes": [], "means": [], "agreements": []})
        d["slopes"].append(float(slope))
        d["means"].append(mean_comfort)
        if agree is not None:
            d["agreements"].append(agree)

    # Summary
    print(f"\n{'scenario':<28} {'n':>3} {'mean_comfort':>12} {'mean_slope':>11} "
          f"{'agree':>10} {'CI':>18}")
    print("-" * 90)
    summary: dict[str, dict] = {}
    for scen in sorted(by_scen):
        d = by_scen[scen]
        agreements = d["agreements"]
        agree_str = f"{sum(agreements)}/{len(agreements)}" if agreements else "-"
        lo, hi = bootstrap_rate(agreements) if agreements else (float("nan"), float("nan"))
        print(f"{scen:<28} {len(d['slopes']):>3} {np.mean(d['means']):>12.1f} "
              f"{np.mean(d['slopes']):>+11.4f} {agree_str:>10} "
              f"[{lo:>.2f},{hi:>.2f}]")
        summary[scen] = {
            "n": len(d["slopes"]),
            "mean_comfort": float(np.mean(d["means"])),
            "mean_slope": float(np.mean(d["slopes"])),
            "agree": int(sum(agreements)) if agreements else None,
            "agree_n": len(agreements) if agreements else 0,
            "agree_rate": float(np.mean(agreements)) if agreements else None,
            "agree_ci": [lo, hi] if agreements else None,
        }

    report = {"config": args.config, "per_recording": rows, "per_scenario": summary}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
