#!/usr/bin/env python3
"""Calibration orchestrator.

Four stages, each independently re-runnable:

  calibrate.py split       → write data/split.json (stratified 75/25)
  calibrate.py extract     → cache raw per-frame features (runs vision models)
  calibrate.py optimize    → differential-evolution search → config/deploy.yaml
  calibrate.py evaluate    → held-out evaluation → reports/calibration_report.json
  calibrate.py all         → runs the four in sequence

Caching raw signals means iterating on the objective function does not re-run
the slow models. Only `extract` touches the GPU.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STAGES = {
    "split":    SCRIPTS / "make_split.py",
    "extract":  SCRIPTS / "extract_features.py",
    "optimize": SCRIPTS / "optimize_params.py",
    "evaluate": SCRIPTS / "evaluate_test.py",
    "race":     SCRIPTS / "race_objectives.py",
}


def run(stage: str, forward_args: list[str]) -> int:
    script = STAGES[stage]
    cmd = [sys.executable, str(script), *forward_args]
    print(f"\n{'='*70}\n> {' '.join(cmd)}\n{'='*70}\n", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration pipeline orchestrator.")
    parser.add_argument("stage", choices=[*STAGES.keys(), "all"])
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Args forwarded to the stage.")
    args = parser.parse_args()

    if args.stage != "all":
        return run(args.stage, args.extra)

    for stage in ("split", "extract", "optimize", "evaluate"):
        rc = run(stage, [])
        if rc != 0:
            print(f"\n[{stage}] failed (rc={rc}); aborting.")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
