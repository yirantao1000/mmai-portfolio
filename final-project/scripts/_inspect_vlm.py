"""Quick sanity check on VLM output across scenarios."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
samples = [
    ("sc01_walkby", "2026-04-14_22-35-00__vlm.parquet"),
    ("sc02_comfortable", "2026-04-14_22-44-18__vlm.parquet"),
    ("sc03_gradual_discomfort", "2026-04-14_22-54-17__vlm.parquet"),
    ("sc04_sudden_withdrawal", "2026-04-14_22-57-29__vlm.parquet"),
    ("sc05_distracted", "2026-04-14_23-00-51__vlm.parquet"),
]
for scen, name in samples:
    f = ROOT / "annotations" / scen / name
    if not f.exists():
        print(f"NOT YET: {scen}/{name}")
        continue
    df = pd.read_parquet(f)
    print(f"=== {scen} / {name} ===")
    print(
        f"  rows={len(df):3d}  score: min={df.comfort_score.min():3.0f}  "
        f"mean={df.comfort_score.mean():3.0f}  max={df.comfort_score.max():3.0f}  "
        f"conf_mean={df.confidence.mean():.2f}"
    )
    g = df.groupby("phase")["comfort_score"].agg(["mean", "count"])
    print("  by phase:")
    for p, row in g.iterrows():
        print(f"    {p:10s}: mean={row['mean']:5.1f}  n={int(row['count']):3d}")
    print(f"  rationale (mid frame):  {df.iloc[len(df)//2].rationale[:170]!r}")
    print(f"  rationale (last frame): {df.iloc[-1].rationale[:170]!r}")
    print()
