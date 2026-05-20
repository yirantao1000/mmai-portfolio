#!/usr/bin/env python3
"""Archive a calibration run's config + report to reports/ and config/history/
with a timestamped tag, and append a one-line summary to reports/RUNS.md.

Usage:
  archive_run.py --tag race1 --config config/deploy_A.yaml --report calibration_report_A.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _summarize(report_path: Path) -> dict:
    if not report_path.exists():
        return {}
    with open(report_path) as f:
        rep = json.load(f)
    out: dict = {}
    bj = rep.get("best_j_on_test") or {}
    if bj:
        out["test_J"] = bj.get("J")
        out["test_tau"] = bj.get("tau")
    agree = rep.get("slope_sign_agreement") or {}
    for scen, d in agree.items():
        out[f"{scen}_agree"] = f"{d.get('agree')}/{d.get('n')}"
        out[f"{scen}_slope"] = round(d.get("mean_slope", float("nan")), 4)
    stats = rep.get("scenario_stats") or {}
    for scen, d in stats.items():
        out[f"{scen}_mean"] = round(d.get("mean", float("nan")), 1)
    dec = rep.get("decisions_at_tau_star") or {}
    if dec:
        out["decisions"] = f"TP{dec.get('tp')}/FN{dec.get('fn')}/FP{dec.get('fp')}/TN{dec.get('tn')}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Short label, e.g. race1_A.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", default=None,
                        help="Optional evaluate_test report to summarize.")
    parser.add_argument("--note", default="", help="Free-text note for RUNS.md row.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_dir = PROJECT_ROOT / "config" / "history"
    rep_dir = PROJECT_ROOT / "reports"
    hist_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    src_cfg = Path(args.config)
    dst_cfg = hist_dir / f"{stamp}_{args.tag}.yaml"
    shutil.copy2(src_cfg, dst_cfg)
    print(f"archived config → {dst_cfg.relative_to(PROJECT_ROOT)}")

    dst_rep = None
    summary: dict = {}
    if args.report:
        src_rep = Path(args.report)
        dst_rep = rep_dir / f"{stamp}_{args.tag}.json"
        shutil.copy2(src_rep, dst_rep)
        print(f"archived report → {dst_rep.relative_to(PROJECT_ROOT)}")
        summary = _summarize(src_rep)

    # Append a row to RUNS.md.
    runs_md = rep_dir / "RUNS.md"
    if not runs_md.exists():
        runs_md.write_text(
            "# Calibration Runs\n\n"
            "Each row is one config + its held-out test metrics, kept for the write-up.\n\n"
            "| timestamp | tag | config | J | sc02_mean | sc01_mean | sc04_mean | "
            "sc02_slope | sc01_slope | sc04_slope | decisions | note |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
    row = (
        f"| {stamp} | {args.tag} | {dst_cfg.relative_to(PROJECT_ROOT)} | "
        f"{summary.get('test_J', '-')} | "
        f"{summary.get('sc02_comfortable_mean', '-')} | "
        f"{summary.get('sc01_walkby_mean', '-')} | "
        f"{summary.get('sc04_sudden_withdrawal_mean', '-')} | "
        f"{summary.get('sc02_comfortable_slope', '-')} | "
        f"{summary.get('sc01_walkby_slope', '-')} | "
        f"{summary.get('sc04_sudden_withdrawal_slope', '-')} | "
        f"{summary.get('decisions', '-')} | "
        f"{args.note} |\n"
    )
    with open(runs_md, "a") as f:
        f.write(row)
    print(f"appended row to {runs_md.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
