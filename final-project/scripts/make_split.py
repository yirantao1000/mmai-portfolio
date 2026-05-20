#!/usr/bin/env python3
"""Write a stratified 75/25 train/test split manifest to data/split.json.

Stratified per scenario so each class is represented in both sets. Seed is
fixed so the split is deterministic across reruns; change `--seed` only if
the default split is clearly unlucky (one scenario's face-detection rate
diverges wildly between train and test, per §Verification step 6).
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_PATH = DATA_DIR / "split.json"


def scenario_dirs() -> list[Path]:
    return sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("sc"))


def bags_for_scenario(scenario_dir: Path) -> list[Path]:
    raw = scenario_dir / "RawData_unlabelled_bagfiles"
    if not raw.is_dir():
        return []
    return sorted(raw.glob("*.bag"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stratified train/test split manifest.")
    parser.add_argument("--test-frac", type=float, default=0.25, help="Held-out fraction (default 0.25).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    parser.add_argument("--out", type=str, default=str(SPLIT_PATH), help="Output path for split.json.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest: dict = {"seed": args.seed, "test_frac": args.test_frac, "scenarios": {}}

    total_train, total_test = 0, 0
    for scen_dir in scenario_dirs():
        bags = bags_for_scenario(scen_dir)
        if not bags:
            print(f"[skip] {scen_dir.name} — no .bag files")
            continue

        shuffled = bags[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * args.test_frac))
        test_bags = shuffled[:n_test]
        train_bags = shuffled[n_test:]

        manifest["scenarios"][scen_dir.name] = {
            "train": [str(p.relative_to(PROJECT_ROOT)) for p in sorted(train_bags)],
            "test":  [str(p.relative_to(PROJECT_ROOT)) for p in sorted(test_bags)],
        }
        total_train += len(train_bags)
        total_test += len(test_bags)
        print(f"  {scen_dir.name:<28} train={len(train_bags)}  test={len(test_bags)}")

    print(f"\n  total                         train={total_train}  test={total_test}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
