#!/usr/bin/env python3
"""Stage C — evaluate deploy.yaml on the held-out test split.

Reports per-scenario comfort (mean + bootstrap 95% CI), abort-decision counts at
the calibrated τ*, and a ROC-style sweep over τ ∈ [30, 80]. The ROC is what the
robot controller in §7.3 consults to pick its deployment threshold.
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

from scripts.optimize_params import (
    ABORT, CONTINUE, PARAM_NAMES, Recording,
    load_recordings, score_recording, youden_j,
    replay_series, reduce_mean_full, slope_fit,
)


DESIRED_SLOPE_SIGN = {
    "sc02_comfortable": +1,
    "sc01_walkby": -1,
    "sc03_gradual_discomfort": -1,
    "sc04_sudden_withdrawal": -1,
    "sc05_distracted": -1,
}


def bootstrap_ci(xs: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(xs, dtype=np.float32)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def config_to_params(cfg: dict) -> dict:
    c = cfg.get("comfort", {})
    pw = c.get("phase_weights", {})
    gd = cfg.get("gaze_detector", {})
    pd_ = cfg.get("pose_detector", {})
    return {
        "we_intent": pw.get("intent", {}).get("emotion_weight", 0.6),
        "wp_intent": pw.get("intent", {}).get("posture_weight", 0.4),
        "wg_intent": pw.get("intent", {}).get("gaze_weight", 0.6),
        "we_exec": pw.get("execution", {}).get("emotion_weight", 0.35),
        "wp_exec": pw.get("execution", {}).get("posture_weight", 0.65),
        "wg_exec": pw.get("execution", {}).get("gaze_weight", 0.1),
        "yaw_threshold": gd.get("yaw_threshold", 25.0),
        "pitch_threshold": gd.get("pitch_threshold", 20.0),
        "face_cover_ratio": pd_.get("face_cover_ratio", 0.6),
        "mouth_cover_penalty": c.get("mouth_cover_penalty", 30.0),
        "withdrawal_threshold_m": pd_.get("withdrawal_threshold_meters", 0.12),
        "withdrawal_penalty": c.get("withdrawal_penalty", 25.0),
        "gamma": c.get("gamma", 0.5),
        "delta": c.get("delta", 0.3),
        "ema_time_constant_s": c.get("ema_time_constant_s", 0.5),
        "posture_drop_gate": pd_.get("posture_drop_gate", 0.15),
        "no_face_target": c.get("no_face_target", 35.0),
        "no_face_rate": c.get("no_face_rate", 0.15),
        "no_pose_target": c.get("no_pose_target", 50.0),
        "no_pose_rate": c.get("no_pose_rate", 0.30),
    }


def roc_sweep(scored: list[tuple[float, str]]) -> list[dict]:
    n_abort = sum(1 for _, lab in scored if lab == "abort")
    n_cont = sum(1 for _, lab in scored if lab == "continue")
    points = []
    for tau in np.arange(30.0, 80.5, 1.0):
        tp = sum(1 for c, lab in scored if lab == "abort" and c <= tau)
        fp = sum(1 for c, lab in scored if lab == "continue" and c <= tau)
        tpr = tp / n_abort if n_abort > 0 else 0.0
        fpr = fp / n_cont if n_cont > 0 else 0.0
        points.append({"tau": float(tau), "abort_tpr": tpr, "abort_fpr": fpr})
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage C — evaluate on held-out test split.")
    parser.add_argument("--split", default=str(PROJECT_ROOT / "data" / "split.json"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "deploy.yaml"))
    parser.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "calibration_report.json"))
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}. Run scripts/optimize_params.py first.")
        return 1
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    params = config_to_params(cfg)

    test = load_recordings(Path(args.split), "test")
    if not test:
        print("no test recordings found (did extract_features run?)")
        return 1
    print(f"Loaded {len(test)} held-out test recordings.\n")

    # Per-recording score AND per-recording series (for trajectory metrics)
    series: list[tuple[Recording, np.ndarray, np.ndarray]] = [
        (r, *replay_series(r, params)) for r in test
    ]
    scored: list[tuple[Recording, float]] = [
        (r, reduce_mean_full(ts, integ)) for r, ts, integ in series
    ]

    # Per-scenario stats
    by_scen: dict[str, list[float]] = {}
    for r, c in scored:
        by_scen.setdefault(r.scenario, []).append(c)

    print(f"{'scenario':<28} {'n':>3} {'mean':>6} {'95% CI':>20} {'min':>6} {'max':>6}")
    print("-" * 75)
    scenario_stats = {}
    for scen in sorted(by_scen):
        vals = by_scen[scen]
        mean = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals)
        scenario_stats[scen] = {
            "n": len(vals), "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "min": float(np.min(vals)), "max": float(np.max(vals)),
        }
        print(f"{scen:<28} {len(vals):>3} {mean:>6.1f}  [{lo:>5.1f}, {hi:>5.1f}]  {np.min(vals):>6.1f} {np.max(vals):>6.1f}")

    # Abort decision at calibrated τ*
    tau_star = cfg.get("comfort", {}).get("abort_threshold")
    scored_label = [(c, r.label) for r, c in scored]
    decisions = None
    if tau_star is not None:
        print(f"\nAbort decisions at τ* = {tau_star}:")
        tp = fp = tn = fn = 0
        for c, lab in scored_label:
            if lab == "abort":
                if c <= tau_star: tp += 1
                else: fn += 1
            elif lab == "continue":
                if c <= tau_star: fp += 1
                else: tn += 1
        print(f"  abort-correct (TP): {tp}   abort-missed (FN): {fn}")
        print(f"  false-abort (FP):   {fp}   continue-correct (TN): {tn}")
        decisions = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "tau_star": float(tau_star)}

    # ROC
    roc = roc_sweep(scored_label)
    best_j, best_tau = youden_j(scored_label)
    print(f"\nTest-set best Youden J = {best_j:.3f} at τ = {best_tau:.1f}")
    print("ROC (selected points):")
    for pt in roc[::5]:
        print(f"  τ={pt['tau']:>5.1f}   TPR={pt['abort_tpr']:.2f}   FPR={pt['abort_fpr']:.2f}")

    # Trajectory metrics: slope, start_mean (first 10%), end_mean (last 10%),
    # slope_sign_match vs the desired per-scenario direction.
    print("\nTrajectory metrics (per recording):")
    print(f"{'scenario':<28} {'name':<28} {'slope':>8} {'start':>6} {'end':>6} {'sign?':>5}")
    print("-" * 90)
    traj_rows: list[dict] = []
    agree_by_scen: dict[str, list[int]] = {}
    slopes_by_scen: dict[str, list[float]] = {}
    for r, ts, integ in series:
        if integ.size < 2:
            continue
        slope = slope_fit(ts, integ)
        k = max(1, int(round(integ.size * 0.10)))
        start_mean = float(np.mean(integ[:k]))
        end_mean = float(np.mean(integ[-k:]))
        want = DESIRED_SLOPE_SIGN.get(r.scenario)
        if want is None:
            sign_match = None
            tag = "?"
        else:
            sign_match = int((want > 0 and slope > 0) or (want < 0 and slope < 0))
            tag = "Y" if sign_match else "N"
            agree_by_scen.setdefault(r.scenario, []).append(sign_match)
        slopes_by_scen.setdefault(r.scenario, []).append(slope)
        print(f"{r.scenario:<28} {r.name:<28} {slope:>+8.4f} {start_mean:>6.1f} {end_mean:>6.1f} {tag:>5}")
        traj_rows.append({
            "scenario": r.scenario, "name": r.name,
            "slope": float(slope), "start_mean": start_mean, "end_mean": end_mean,
            "slope_sign_match": sign_match,
        })

    print("\nPer-scenario slope-sign agreement:")
    for scen in sorted(agree_by_scen):
        agrees = agree_by_scen[scen]
        ss = slopes_by_scen.get(scen, [])
        rate = sum(agrees) / len(agrees) if agrees else float("nan")
        mean_slope = float(np.mean(ss)) if ss else float("nan")
        print(f"  {scen:<28} n={len(agrees)}  agree={sum(agrees)}/{len(agrees)}  "
              f"({rate:.0%})  mean_slope={mean_slope:+.4f}")

    # Detection-rate sanity
    print("\nDetection-rate sanity (test split):")
    for r, _ in scored:
        face_rate = float(r.df["face_detected"].mean())
        pose_rate = float(r.df["pose_detected"].mean())
        print(f"  {r.scenario:<28} {r.name}  face={face_rate:.0%}  pose={pose_rate:.0%}")

    report = {
        "config": str(cfg_path.relative_to(PROJECT_ROOT)),
        "n_test": len(test),
        "scenario_stats": scenario_stats,
        "decisions_at_tau_star": decisions,
        "roc": roc,
        "best_j_on_test": {"J": best_j, "tau": best_tau},
        "trajectory": traj_rows,
        "slope_sign_agreement": {
            scen: {
                "n": len(agree_by_scen[scen]),
                "agree": int(sum(agree_by_scen[scen])),
                "rate": float(sum(agree_by_scen[scen]) / len(agree_by_scen[scen])),
                "mean_slope": float(np.mean(slopes_by_scen[scen])),
            }
            for scen in agree_by_scen
        },
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    try:
        rel_rep = Path(args.report).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        rel_rep = Path(args.report).resolve()
    print(f"\nWrote {rel_rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
