#!/usr/bin/env python3
"""Trajectory-Aware Objective Race.

Phase 1: cheap race across 4 variants (A, B, C, F) @ maxiter=15 popsize=8.
         Score each on TRAIN with
             combined = 0.5·J_train + 0.3·slope_sign_agreement + 0.2·margin/100.
Phase 2: full DE (maxiter=50 popsize=15) on the top-2 variants from Phase 1.
         Writes config/deploy_<variant>.yaml for each finalist.
Phase 3: evaluate each finalist on the held-out test via evaluate_test.py.
         Aggregates into calibration_race_report.json; prints leaderboard.

Promotion to config/deploy.yaml is gated on user confirmation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_params import (
    BOUNDS, PARAM_NAMES, VARIANTS, Recording,
    build_objective, load_recordings, params_to_dict,
    reduce_mean_full, reduce_mean_late, replay_series,
    slope_fit, stratified_folds, youden_j,
)


DESIRED_SLOPE_SIGN = {
    "sc02_comfortable": +1,
    "sc01_walkby": -1,
    "sc03_gradual_discomfort": -1,
    "sc04_sudden_withdrawal": -1,
    "sc05_distracted": -1,
}


# ---------- Phase-1 scoring (all level-space for cross-variant comparability) --

def score_candidate(
    x: np.ndarray, recordings: list[Recording]
) -> dict:
    """Compute the composite (J, slope_agreement, margin) on full-train in
    level space, so candidates driven by different objectives are comparable."""
    p = params_to_dict(x)
    series = [(r, *replay_series(r, p)) for r in recordings]

    # Level J on full-train using mean_full.
    scored_level: list[tuple[float, str]] = []
    for r, ts, integ in series:
        c = reduce_mean_full(ts, integ)
        if not np.isnan(c):
            scored_level.append((float(c), r.label))
    j_train, tau_star = youden_j(scored_level, tau_range=(30.0, 80.5))

    # Slope-sign agreement across all recordings with a desired direction.
    agreements: list[int] = []
    slopes_by_scen: dict[str, list[float]] = {}
    for r, ts, integ in series:
        if integ.size < 2:
            continue
        slope = slope_fit(ts, integ)
        slopes_by_scen.setdefault(r.scenario, []).append(slope)
        want = DESIRED_SLOPE_SIGN.get(r.scenario)
        if want is None:
            continue
        agreements.append(int((want > 0 and slope > 0) or (want < 0 and slope < 0)))
    agree_rate = float(np.mean(agreements)) if agreements else 0.0

    # Margin: mean(sc02 last-10% comfort) − mean(sc01 last-10% comfort).
    sc02_ends: list[float] = []
    sc01_ends: list[float] = []
    for r, ts, integ in series:
        if integ.size < 2:
            continue
        k = max(1, int(round(integ.size * 0.10)))
        end_mean = float(np.mean(integ[-k:]))
        if r.scenario == "sc02_comfortable":
            sc02_ends.append(end_mean)
        elif r.scenario == "sc01_walkby":
            sc01_ends.append(end_mean)
    margin = (
        float(np.mean(sc02_ends) - np.mean(sc01_ends))
        if sc02_ends and sc01_ends else 0.0
    )

    combined = 0.5 * j_train + 0.3 * agree_rate + 0.2 * (margin / 100.0)

    per_scenario_slope = {
        scen: float(np.mean(ss)) for scen, ss in slopes_by_scen.items()
    }

    return {
        "combined": float(combined),
        "j_train": float(j_train),
        "tau_star": float(tau_star),
        "slope_agreement": float(agree_rate),
        "margin": float(margin),
        "per_scenario_slope": per_scenario_slope,
    }


# ---------- Config writer ------------------------------------------------------

def write_deploy(
    config_in: Path, config_out: Path, best: dict, variant: str,
    tau_star: float,
) -> None:
    with open(config_in) as f:
        cfg = yaml.safe_load(f)
    cfg["gaze_detector"]["yaw_threshold"] = round(best["yaw_threshold"], 2)
    cfg["gaze_detector"]["pitch_threshold"] = round(best["pitch_threshold"], 2)
    cfg["pose_detector"]["face_cover_ratio"] = round(best["face_cover_ratio"], 3)
    cfg["pose_detector"]["withdrawal_threshold_meters"] = round(best["withdrawal_threshold_m"], 3)
    cfg["pose_detector"]["posture_drop_gate"] = round(best["posture_drop_gate"], 3)
    cfg["comfort"]["gamma"] = round(best["gamma"], 3)
    cfg["comfort"]["delta"] = round(best["delta"], 3)
    cfg["comfort"]["mouth_cover_penalty"] = round(best["mouth_cover_penalty"], 2)
    cfg["comfort"]["withdrawal_penalty"] = round(best["withdrawal_penalty"], 2)
    cfg["comfort"]["ema_time_constant_s"] = round(best["ema_time_constant_s"], 3)
    cfg["comfort"]["no_face_target"] = round(best.get("no_face_target", 35.0), 2)
    cfg["comfort"]["no_face_rate"] = round(best.get("no_face_rate", 0.15), 3)
    cfg["comfort"]["no_pose_target"] = round(best.get("no_pose_target", 50.0), 2)
    cfg["comfort"]["no_pose_rate"] = round(best.get("no_pose_rate", 0.30), 3)
    cfg["comfort"]["phase_weights"]["intent"] = {
        "emotion_weight": round(best["we_intent"], 3),
        "posture_weight": round(best["wp_intent"], 3),
        "gaze_weight":    round(best["wg_intent"], 3),
    }
    cfg["comfort"]["phase_weights"]["execution"] = {
        "emotion_weight": round(best["we_exec"], 3),
        "posture_weight": round(best["wp_exec"], 3),
        "gaze_weight":    round(best["wg_exec"], 3),
    }
    cfg["comfort"]["abort_threshold"] = round(tau_star, 2)
    cfg["comfort"]["calibrated_variant"] = variant
    with open(config_out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------- One DE run ---------------------------------------------------------

def run_de(
    variant: str, train: list[Recording], folds: list[list[int]],
    maxiter: int, popsize: int, eps: float, seed: int,
) -> tuple[dict, float, np.ndarray]:
    """Run DE for one variant. Returns (metrics, wall_clock_s, x_best)."""
    objective = build_objective(variant)
    t0 = time.time()
    result = differential_evolution(
        objective,
        BOUNDS,
        args=(train, folds, eps),
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        seed=seed,
        polish=True,
        updating="deferred",
        workers=1,
        disp=False,
    )
    elapsed = time.time() - t0
    metrics = score_candidate(result.x, train)
    return metrics, elapsed, result.x


# ---------- Race driver --------------------------------------------------------

def phase1(
    train: list[Recording], folds: list[list[int]],
    maxiter: int, popsize: int, eps: float, seed: int,
    cache_dir: Path,
) -> list[dict]:
    print(f"\n{'='*70}\n  PHASE 1 — cheap race @ maxiter={maxiter} popsize={popsize}\n{'='*70}\n")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for variant in ("A", "B", "C", "F"):
        print(f"--- variant {variant} ---")
        metrics, elapsed, x = run_de(
            variant, train, folds, maxiter, popsize, eps, seed
        )
        best = params_to_dict(x)
        print(f"  J={metrics['j_train']:.3f}  slope_agree={metrics['slope_agreement']:.2f}  "
              f"margin={metrics['margin']:+.1f}  combined={metrics['combined']:.3f}  "
              f"t={elapsed:.1f}s  τ*={metrics['tau_star']:.1f}")
        for scen in sorted(metrics["per_scenario_slope"]):
            print(f"    slope {scen:<28} {metrics['per_scenario_slope'][scen]:+.4f}")
        # Write per-variant config for traceability.
        write_deploy(
            PROJECT_ROOT / "config" / "default.yaml",
            cache_dir / f"{variant}.yaml",
            best, variant, metrics["tau_star"],
        )
        rows.append({
            "variant": variant,
            "metrics": metrics,
            "params": best,
            "wall_clock_s": elapsed,
        })
    return rows


def phase2(
    top_variants: list[str], train: list[Recording], folds: list[list[int]],
    maxiter: int, popsize: int, eps: float, seed: int,
) -> dict[str, dict]:
    print(f"\n{'='*70}\n  PHASE 2 — full DE @ maxiter={maxiter} popsize={popsize} on top-{len(top_variants)}\n{'='*70}\n")
    finalists: dict[str, dict] = {}
    for variant in top_variants:
        print(f"--- variant {variant} (full DE) ---")
        metrics, elapsed, x = run_de(
            variant, train, folds, maxiter, popsize, eps, seed
        )
        best = params_to_dict(x)
        print(f"  J={metrics['j_train']:.3f}  slope_agree={metrics['slope_agreement']:.2f}  "
              f"margin={metrics['margin']:+.1f}  combined={metrics['combined']:.3f}  "
              f"t={elapsed:.1f}s  τ*={metrics['tau_star']:.1f}")
        for scen in sorted(metrics["per_scenario_slope"]):
            print(f"    slope {scen:<28} {metrics['per_scenario_slope'][scen]:+.4f}")
        out_cfg = PROJECT_ROOT / "config" / f"deploy_{variant}.yaml"
        write_deploy(
            PROJECT_ROOT / "config" / "default.yaml",
            out_cfg, best, variant, metrics["tau_star"],
        )
        print(f"  wrote {out_cfg.relative_to(PROJECT_ROOT)}")
        finalists[variant] = {
            "metrics": metrics,
            "params": best,
            "wall_clock_s": elapsed,
            "config_path": str(out_cfg.relative_to(PROJECT_ROOT)),
        }
    return finalists


def phase3(finalists: dict[str, dict]) -> dict[str, dict]:
    print(f"\n{'='*70}\n  PHASE 3 — held-out evaluation\n{'='*70}\n")
    results: dict[str, dict] = {}
    for variant, info in finalists.items():
        cfg_path = PROJECT_ROOT / info["config_path"]
        report_path = PROJECT_ROOT / "reports" / f"calibration_report_{variant}.json"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_test.py"),
            "--config", str(cfg_path),
            "--report", str(report_path),
        ]
        print(f"--- variant {variant}: {' '.join(cmd)} ---")
        rc = subprocess.call(cmd)
        if rc != 0 or not report_path.exists():
            print(f"  [warn] evaluate rc={rc}; skipping variant {variant} in aggregation.")
            continue
        with open(report_path) as f:
            results[variant] = json.load(f)
    return results


def leaderboard(rows_phase1: list[dict], test_results: dict[str, dict]) -> None:
    print(f"\n{'='*70}\n  LEADERBOARD\n{'='*70}\n")
    print(f"{'var':>3} {'P1_combined':>12} {'P1_J':>6} {'P1_agree':>8} {'P1_margin':>9}  "
          f"{'TEST_J':>7} {'sc02_slope_ok':>13}  {'sc02_mean':>10}")
    print("-" * 95)
    by_var = {row["variant"]: row for row in rows_phase1}
    for variant in ("A", "B", "C", "F"):
        row = by_var.get(variant)
        if row is None:
            continue
        m = row["metrics"]
        test = test_results.get(variant)
        if test is not None:
            test_j = test.get("best_j_on_test", {}).get("J", float("nan"))
            agree = test.get("slope_sign_agreement", {}).get("sc02_comfortable", {})
            sc02_agree = f"{agree.get('agree', 0)}/{agree.get('n', 0)}"
            sc02_stats = test.get("scenario_stats", {}).get("sc02_comfortable", {})
            sc02_mean = sc02_stats.get("mean", float("nan"))
        else:
            test_j = float("nan"); sc02_agree = "-"; sc02_mean = float("nan")
        print(f"{variant:>3} {m['combined']:>12.3f} {m['j_train']:>6.3f} "
              f"{m['slope_agreement']:>8.2f} {m['margin']:>+9.1f}  "
              f"{test_j:>7.3f} {sc02_agree:>13}  {sc02_mean:>10.1f}")


# ---------- CLI ----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Trajectory-aware objective race.")
    parser.add_argument("--split", default=str(PROJECT_ROOT / "data" / "split.json"))
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p1-maxiter", type=int, default=15)
    parser.add_argument("--p1-popsize", type=int, default=8)
    parser.add_argument("--p2-maxiter", type=int, default=50)
    parser.add_argument("--p2-popsize", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=2,
                        help="How many Phase-1 variants advance to Phase 2.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny budgets (maxiter=3 popsize=4 both phases).")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Only run Phase 1 (useful for exploration).")
    parser.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "calibration_race_report.json"))
    args = parser.parse_args()

    if args.smoke:
        args.p1_maxiter, args.p1_popsize = 3, 4
        args.p2_maxiter, args.p2_popsize = 3, 4

    split = Path(args.split)
    if not split.exists():
        print(f"split manifest not found: {split}. Run make_split.py first.")
        return 1

    # Idempotent: archive current deploy.yaml as deploy.baseline.yaml once.
    deploy = PROJECT_ROOT / "config" / "deploy.yaml"
    baseline = PROJECT_ROOT / "config" / "deploy.baseline.yaml"
    if deploy.exists() and not baseline.exists():
        shutil.copy2(deploy, baseline)
        print(f"archived {deploy.name} → {baseline.name}")

    print(f"Loading train recordings from {split.relative_to(PROJECT_ROOT)}...")
    train = load_recordings(split, "train")
    print(f"  {len(train)} train recordings")
    if not train:
        return 1
    folds = stratified_folds(train, 5, seed=args.seed)

    cache_dir = PROJECT_ROOT / "cache" / "race"
    rows_p1 = phase1(
        train, folds, args.p1_maxiter, args.p1_popsize,
        args.eps, args.seed, cache_dir,
    )
    with open(cache_dir / "phase1.json", "w") as f:
        json.dump(rows_p1, f, indent=2)

    if args.skip_phase2:
        print("\nSkipping Phase 2/3 (--skip-phase2).")
        return 0

    # Select top-k by combined score.
    ranked = sorted(rows_p1, key=lambda r: r["metrics"]["combined"], reverse=True)
    top_variants = [r["variant"] for r in ranked[: args.top_k]]
    print(f"\nTop-{args.top_k} variants by Phase-1 combined score: {top_variants}")

    finalists = phase2(
        top_variants, train, folds, args.p2_maxiter, args.p2_popsize,
        args.eps, args.seed,
    )
    test_results = phase3(finalists)

    leaderboard(rows_p1, test_results)

    # Persist aggregated report.
    report = {
        "phase1": rows_p1,
        "top_variants": top_variants,
        "finalists": finalists,
        "test_results": test_results,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {Path(args.report).relative_to(PROJECT_ROOT)}")

    print("\nPromote a winner to config/deploy.yaml by copying the chosen")
    print("config/deploy_<variant>.yaml over deploy.yaml (baseline is preserved).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
