"""Move the new Dropbox dataset (in data_incoming/) into the canonical
`data/` layout that the existing tooling expects.

New (Dropbox) layout                          Canonical layout
-------------------------------------         --------------------------------------------
data_incoming/sc*/<stem>.bag             ->   data/sc*/RawData_unlabelled_bagfiles/<stem>.bag
data_incoming/sc*/Labelled/<stem>.json   ->   data/sc*/<stem>.json

Run with `--dry-run` first to see what would happen; without it the moves
are committed and the `data_incoming/` scenario dirs are cleaned up.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "data_incoming"
TARGET_ROOT = PROJECT_ROOT / "data"
SCENARIOS = ["sc01_walkby", "sc02_comfortable", "sc03_gradual_discomfort",
             "sc04_sudden_withdrawal", "sc05_distracted"]


def plan_moves() -> list[tuple[Path, Path]]:
    """Return list of (src, dst) pairs to move."""
    moves: list[tuple[Path, Path]] = []
    for scen in SCENARIOS:
        src_scen = SOURCE_ROOT / scen
        dst_scen = TARGET_ROOT / scen
        if not src_scen.exists():
            continue

        # bags: data_incoming/sc*/<stem>.bag  ->  data/sc*/RawData_unlabelled_bagfiles/<stem>.bag
        for bag in sorted(src_scen.glob("*.bag")):
            dst = dst_scen / "RawData_unlabelled_bagfiles" / bag.name
            moves.append((bag, dst))

        # sidecars: data_incoming/sc*/Labelled/<stem>.json  ->  data/sc*/<stem>.json
        labelled = src_scen / "Labelled"
        if labelled.exists():
            for js in sorted(labelled.glob("*.json")):
                dst = dst_scen / js.name
                moves.append((js, dst))
    return moves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions but don't move anything.")
    ap.add_argument("--keep-incoming", action="store_true",
                    help="Don't remove data_incoming/ scenario dirs after success.")
    args = ap.parse_args()

    moves = plan_moves()
    print(f"planned {len(moves)} moves\n")

    # Validate no destination collisions
    collisions = [(s, d) for s, d in moves if d.exists()]
    if collisions:
        print("ERROR: destination collisions (would overwrite):")
        for s, d in collisions:
            print(f"  {s}  ->  {d}  (already exists!)")
        return 1

    n_bag = sum(1 for s, d in moves if s.suffix == ".bag")
    n_json = sum(1 for s, d in moves if s.suffix == ".json")
    print(f"  bags:    {n_bag}")
    print(f"  sidecars:{n_json}")
    print()

    if args.dry_run:
        print("--- dry-run, showing first 10 of each kind ---")
        bags = [(s, d) for s, d in moves if s.suffix == ".bag"][:10]
        jsons = [(s, d) for s, d in moves if s.suffix == ".json"][:10]
        print("bags:")
        for s, d in bags:
            print(f"  {s.relative_to(PROJECT_ROOT)}  ->  {d.relative_to(PROJECT_ROOT)}")
        print("sidecars:")
        for s, d in jsons:
            print(f"  {s.relative_to(PROJECT_ROOT)}  ->  {d.relative_to(PROJECT_ROOT)}")
        print("\nrun again WITHOUT --dry-run to commit.")
        return 0

    # Execute
    for i, (src, dst) in enumerate(moves, 1):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        if i % 10 == 0 or i == len(moves):
            print(f"  moved {i}/{len(moves)}")

    if not args.keep_incoming:
        # Drop the now-empty data_incoming/sc* dirs; keep the zip file though
        for scen in SCENARIOS:
            scen_dir = SOURCE_ROOT / scen
            if scen_dir.exists():
                shutil.rmtree(scen_dir)
                print(f"  removed {scen_dir.relative_to(PROJECT_ROOT)}")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
