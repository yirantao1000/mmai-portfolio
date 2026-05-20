"""Summarize VLM annotations across the full corpus, per scenario / phase."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "annotations"

frames = []
for parquet in ANN.rglob("*__vlm.parquet"):
    df = pd.read_parquet(parquet)
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)

print(f"Total frames: {len(all_df)}")
print(f"Bags: {all_df['bag_stem'].nunique()}")
print(f"Scenarios: {sorted(all_df['scenario'].unique())}\n")

# Per scenario
print("=== Per-scenario summary ===")
scen_summary = (
    all_df.groupby("scenario")
    .agg(
        bags=("bag_stem", "nunique"),
        frames=("comfort_score", "size"),
        score_mean=("comfort_score", "mean"),
        score_std=("comfort_score", "std"),
        score_min=("comfort_score", "min"),
        score_max=("comfort_score", "max"),
        conf_mean=("confidence", "mean"),
    )
    .round(2)
)
print(scen_summary.to_string())
print()

# Per scenario x phase
print("=== Per-scenario x phase mean score ===")
piv = (
    all_df.groupby(["scenario", "phase"])["comfort_score"]
    .mean()
    .unstack(fill_value=np.nan)
    .round(1)
)
print(piv.to_string())
print()

# Sanity expectations
print("=== Sanity checks against scenario semantics ===")
expected = {
    "sc01_walkby": "low (person not engaging)",
    "sc02_comfortable": "high (smooth handover, positive)",
    "sc03_gradual_discomfort": "high->mid (gradual decline)",
    "sc04_sudden_withdrawal": "mid->low (sudden withdrawal at execution)",
    "sc05_distracted": "mid (engaged but distracted)",
}
for s, expect in expected.items():
    if s in scen_summary.index:
        m = scen_summary.loc[s, "score_mean"]
        print(f"  {s:28s} mean={m:5.1f}  (expect {expect})")
