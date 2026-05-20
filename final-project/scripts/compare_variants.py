#!/usr/bin/env python3
"""Read calibration_report_*.json files and score each variant against the
plan's pass criteria. Prints a leaderboard and writes reports/variant_gates.json.

Gates (per Stage-C plan, verification step 4):
  sc02 slope >= +0.02
  sc01/sc04 slope <= -0.02
  sc02 mean > 80
  sc01/sc04 mean < 65
  J >= 0.80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GATES = {
    "sc02_slope_min": 0.02,
    "sc01_slope_max": -0.02,
    "sc04_slope_max": -0.02,
    "sc02_mean_min": 80.0,
    "sc01_mean_max": 65.0,
    "sc04_mean_max": 65.0,
    "j_min": 0.80,
}


def eval_report(path: Path) -> dict:
    with open(path) as f:
        rep = json.load(f)
    stats = rep.get("scenario_stats", {})
    agree = rep.get("slope_sign_agreement", {})
    j = rep.get("best_j_on_test", {}).get("J", 0.0)

    sc02_slope = agree.get("sc02_comfortable", {}).get("mean_slope")
    sc01_slope = agree.get("sc01_walkby", {}).get("mean_slope")
    sc04_slope = agree.get("sc04_sudden_withdrawal", {}).get("mean_slope")
    sc02_mean = stats.get("sc02_comfortable", {}).get("mean")
    sc01_mean = stats.get("sc01_walkby", {}).get("mean")
    sc04_mean = stats.get("sc04_sudden_withdrawal", {}).get("mean")

    def _ok(x, op, gate):
        if x is None:
            return False
        return (x >= gate) if op == "ge" else (x <= gate)

    checks = {
        "sc02_slope_positive": _ok(sc02_slope, "ge", GATES["sc02_slope_min"]),
        "sc01_slope_negative": _ok(sc01_slope, "le", GATES["sc01_slope_max"]),
        "sc04_slope_negative": _ok(sc04_slope, "le", GATES["sc04_slope_max"]),
        "sc02_mean_high":      _ok(sc02_mean, "ge", GATES["sc02_mean_min"]),
        "sc01_mean_low":       _ok(sc01_mean, "le", GATES["sc01_mean_max"]),
        "sc04_mean_low":       _ok(sc04_mean, "le", GATES["sc04_mean_max"]),
        "j_min":               _ok(j, "ge", GATES["j_min"]),
    }
    return {
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_absolute() else path),
        "j": j,
        "sc02_slope": sc02_slope, "sc01_slope": sc01_slope, "sc04_slope": sc04_slope,
        "sc02_mean": sc02_mean,   "sc01_mean": sc01_mean,   "sc04_mean": sc04_mean,
        "checks": checks,
        "passes_all": all(checks.values()),
        "n_pass": sum(1 for v in checks.values() if v),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="calibration_report_*.json paths")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "variant_gates.json"))
    args = parser.parse_args()

    results: list[tuple[str, dict]] = []
    for path in args.reports:
        p = Path(path)
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        label = p.stem.replace("calibration_report_", "")
        results.append((label, eval_report(p)))

    # sort by n_pass desc, then J desc
    results.sort(key=lambda x: (x[1]["n_pass"], x[1]["j"]), reverse=True)

    print(f"{'tag':<20} {'pass':>5} {'J':>5} "
          f"{'sc02_slope':>10} {'sc01_slope':>10} {'sc04_slope':>10} "
          f"{'sc02_mean':>9} {'sc01_mean':>9} {'sc04_mean':>9}")
    print("-" * 100)
    for label, r in results:
        mark = "ALL" if r["passes_all"] else f"{r['n_pass']}/7"

        def fmt(v, pad):
            return f"{v:>{pad}.3f}" if v is not None else f"{'-':>{pad}}"

        print(f"{label:<20} {mark:>5} {fmt(r['j'], 5)} "
              f"{fmt(r['sc02_slope'], 10)} {fmt(r['sc01_slope'], 10)} {fmt(r['sc04_slope'], 10)} "
              f"{fmt(r['sc02_mean'], 9)} {fmt(r['sc01_mean'], 9)} {fmt(r['sc04_mean'], 9)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"gates": GATES, "results": dict(results)}, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
