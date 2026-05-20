#!/usr/bin/env python3
"""Create a deterministic train/test split JSON over `data/<scen>/<stem>.bag`.

Strategy: pick `n_test_per_scenario` bag stems from each scenario for test, rest
go to train. Unlabelled bags (no sidecar JSON) always go to train (we can still
distill on them but can't evaluate cleanly without phase windows).

Writes `splits/<name>.json`:

    {
      "name": "v2",
      "seed": 42,
      "n_test_per_scenario": 2,
      "scenarios": {
        "sc01_walkby":    {"train": ["..."], "test": ["..."]},
        ...
      },
      "all_train_stems": [["sc01_walkby","2026-04-14_22-35-00"], ...],
      "all_test_stems":  [["sc01_walkby","2026-04-14_22-36-18"], ...]
    }
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_bags(data_root: Path) -> dict[str, list[str]]:
    """Return {scenario: [bag_stem, ...]} sorted by stem."""
    scens: dict[str, list[str]] = {}
    for scen_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("sc")):
        bags = sorted(scen_dir.rglob("*.bag"))
        scens[scen_dir.name] = [b.stem for b in bags]
    return scens


def has_sidecar(data_root: Path, scen: str, stem: str) -> bool:
    """True if a sidecar JSON exists for this bag (in scenario root or next to bag)."""
    scen_dir = data_root / scen
    if (scen_dir / f"{stem}.json").exists():
        return True
    for p in scen_dir.rglob(f"{stem}.json"):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "splits" / "v2.json"))
    ap.add_argument("--name", default="v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test-per-scenario", type=int, default=2)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scens = find_bags(data_root)
    rng = random.Random(args.seed)

    all_train: list[tuple[str, str]] = []
    all_test: list[tuple[str, str]] = []
    scen_blocks: dict[str, dict[str, list[str]]] = {}

    for scen, stems in scens.items():
        labelled = [s for s in stems if has_sidecar(data_root, scen, s)]
        unlabelled = [s for s in stems if s not in set(labelled)]

        # Pick n_test from labelled only
        n_test = min(args.n_test_per_scenario, len(labelled))
        shuffled = labelled[:]
        rng.shuffle(shuffled)
        test_stems = sorted(shuffled[:n_test])
        train_stems = sorted([s for s in labelled if s not in set(test_stems)] + unlabelled)

        scen_blocks[scen] = {"train": train_stems, "test": test_stems}
        all_train.extend([(scen, s) for s in train_stems])
        all_test.extend([(scen, s) for s in test_stems])

    out = {
        "name": args.name,
        "seed": args.seed,
        "n_test_per_scenario": args.n_test_per_scenario,
        "scenarios": scen_blocks,
        "all_train_stems": [list(p) for p in all_train],
        "all_test_stems": [list(p) for p in all_test],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote split -> {out_path}")
    print(f"  scenarios: {len(scens)}")
    print(f"  train bags: {len(all_train)}")
    print(f"  test bags:  {len(all_test)}")
    print()
    for scen, b in scen_blocks.items():
        print(f"  {scen:30s}  train={len(b['train'])}  test={len(b['test'])}")
    print()
    print("  test bags:")
    for scen, stem in all_test:
        print(f"    {scen}/{stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
