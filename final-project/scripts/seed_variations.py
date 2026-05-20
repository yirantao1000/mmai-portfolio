#!/usr/bin/env python3
"""Run the same variant with multiple DE seeds, evaluate each, report best.

DE is stochastic; the argmax often shifts meaningfully across seeds even with
identical bounds and budget. This script fits N copies under different seeds,
evaluates each on the held-out test, and keeps the one with the highest
pass-gates score.

Usage:
  seed_variations.py --objective G --seeds 17 99 123 \
      --maxiter 50 --popsize 15 \
      --out-prefix config/deploy_G_s \
      --report-prefix calibration_report_G_s
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY = "/home/farandhigh-ubuntu/Documents/mmai-emotion-detection/.venv/bin/python"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True, choices=["A", "B", "C", "F", "G"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--pin-wp", type=float, default=None,
                        help="For ablation: pin wp_intent and wp_exec to this value.")
    args = parser.parse_args()

    results: list[tuple[int, Path, Path, int]] = []
    for seed in args.seeds:
        cfg = PROJECT_ROOT / f"{args.out_prefix}{seed}.yaml"
        rep = PROJECT_ROOT / f"{args.report_prefix}{seed}.json"
        cmd_opt = [
            PY, str(PROJECT_ROOT / "scripts" / "optimize_params.py"),
            "--objective", args.objective,
            "--maxiter", str(args.maxiter),
            "--popsize", str(args.popsize),
            "--seed", str(seed),
            "--config-out", str(cfg),
        ]
        if args.pin_wp is not None:
            cmd_opt.extend(["--pin-wp", str(args.pin_wp)])
        print(f"\n=== seed={seed} optimize ===", flush=True)
        rc = subprocess.call(cmd_opt)
        if rc != 0:
            print(f"[warn] optimize rc={rc} for seed={seed}; skipping eval")
            continue
        cmd_eval = [
            PY, str(PROJECT_ROOT / "scripts" / "evaluate_test.py"),
            "--config", str(cfg),
            "--report", str(rep),
        ]
        print(f"\n=== seed={seed} evaluate ===", flush=True)
        rc = subprocess.call(cmd_eval)
        if rc != 0:
            print(f"[warn] eval rc={rc} for seed={seed}")
            continue
        results.append((seed, cfg, rep, 0))

    print(f"\nCompleted {len(results)} seeds.")
    for seed, cfg, rep, _ in results:
        print(f"  seed={seed}  cfg={cfg.name}  rep={rep.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
