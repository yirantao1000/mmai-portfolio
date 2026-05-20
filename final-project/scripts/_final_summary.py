"""Generate the final cross-source / cross-student summary report."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "annotations"


def load_source(source: str) -> pd.DataFrame:
    rows = []
    for p in ANN.rglob(f"*__{source}.parquet"):
        rows.append(pd.read_parquet(p))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def teacher_agreement(heur: pd.DataFrame, vlm: pd.DataFrame) -> dict:
    """Per-bag teacher-vs-teacher agreement on shared timestamps (nearest join)."""
    out_rows = []
    for (scen, stem), g_v in vlm.groupby(["scenario", "bag_stem"]):
        g_h = heur[(heur["scenario"] == scen) & (heur["bag_stem"] == stem)]
        if g_h.empty:
            continue
        g_v = g_v.sort_values("timestamp_s").reset_index(drop=True)
        g_h = g_h.sort_values("timestamp_s").reset_index(drop=True)
        merged = pd.merge_asof(
            g_v[["timestamp_s", "comfort_score"]].rename(columns={"comfort_score": "vlm"}),
            g_h[["timestamp_s", "comfort_score"]].rename(columns={"comfort_score": "heur"}),
            on="timestamp_s", direction="nearest", tolerance=0.5,
        ).dropna()
        if len(merged) < 5:
            continue
        if merged["vlm"].std() == 0 or merged["heur"].std() == 0:
            r = float("nan")
        else:
            r, _ = pearsonr(merged["vlm"], merged["heur"])
        out_rows.append({
            "scenario": scen,
            "bag_stem": stem,
            "n": len(merged),
            "mean_vlm": float(merged["vlm"].mean()),
            "mean_heur": float(merged["heur"].mean()),
            "mae": float(np.mean(np.abs(merged["vlm"] - merged["heur"]))),
            "pearson": float(r),
        })
    return pd.DataFrame(out_rows)


def main() -> None:
    print("=" * 78)
    print("FINAL SUMMARY: heuristic vs VLM(GPT-5.5) annotation + student distillation")
    print("=" * 78)

    heur = load_source("heuristic")
    vlm = load_source("vlm")
    student_h = load_source("student")  # written by eval (latest run = vlm overwrites heur)

    # 1. dataset coverage
    print("\n[1] Dataset coverage")
    for name, df in [("heuristic", heur), ("vlm", vlm)]:
        if df.empty:
            print(f"  {name:9s}: empty")
            continue
        print(f"  {name:9s}: {len(df):5d} rows  /  "
              f"{df['bag_stem'].nunique():2d} bags  /  "
              f"{df['scenario'].nunique()} scenarios  "
              f"(score mean={df.comfort_score.mean():.1f}, std={df.comfort_score.std():.1f})")

    # 2. per-scenario teacher comparison
    print("\n[2] Per-scenario mean comfort score (teacher = annotator output)")
    scen_table = pd.DataFrame({
        "heuristic": heur.groupby("scenario")["comfort_score"].mean(),
        "vlm (gpt-5.5)": vlm.groupby("scenario")["comfort_score"].mean(),
    }).round(1)
    scen_table["delta(vlm-heur)"] = (scen_table["vlm (gpt-5.5)"] - scen_table["heuristic"]).round(1)
    print(scen_table.to_string())

    # 3. teacher agreement
    print("\n[3] Teacher-vs-teacher agreement per bag (matched on timestamp)")
    agr = teacher_agreement(heur, vlm)
    if agr.empty:
        print("  (no matched bags)")
    else:
        scen_agg = agr.groupby("scenario").agg(
            bags=("bag_stem", "count"),
            mean_vlm=("mean_vlm", "mean"),
            mean_heur=("mean_heur", "mean"),
            mean_mae=("mae", "mean"),
            mean_pearson=("pearson", "mean"),
        ).round(2)
        print(scen_agg.to_string())
        print(f"\n  overall: bags={len(agr)}  "
              f"mean MAE={agr['mae'].mean():.1f}  "
              f"mean Pearson={agr['pearson'].mean():.3f}")

    # 4. student model metrics (from eval summary jsons)
    print("\n[4] Student model performance (1.58M-param MobileNetV3-Small)")
    for src, ckpt in [("heuristic", "checkpoints/heuristic/summary.json"),
                      ("vlm",       "checkpoints/vlm/summary.json")]:
        path = ROOT / ckpt
        if not path.exists():
            print(f"  {src}: no summary")
            continue
        d = json.loads(path.read_text())
        final = d.get("final_metrics", {})
        n_train = len(d.get("split", {}).get("train_stems", []))
        n_test = len(d.get("split", {}).get("test_stems", []))
        print(f"  source={src:9s}: best test MAE={d.get('best_mae', float('nan')):.2f}  "
              f"final RMSE={final.get('rmse', float('nan')):.2f}  "
              f"final Pearson={final.get('pearson', float('nan')):.3f}  "
              f"(n_train_bags={n_train}, n_test_bags={n_test}, n_test_rows={final.get('n', '?')})")

    print("\n[5] Per-bag eval (test + sample held-in)")
    for src, dirpath in [("heuristic", "renders/v1_eval_heuristic_student/heuristic"),
                          ("vlm",       "renders/v1_eval_vlm_student/vlm")]:
        sp = ROOT / dirpath / "eval_summary.json"
        if not sp.exists():
            continue
        d = json.loads(sp.read_text())
        print(f"  -- student trained on {src} --")
        for r in d.get("results", []):
            m = r["metrics"]
            print(f"    [{r['tag']:12s}] {r['scenario']:25s}/{r['bag_stem']}  "
                  f"MAE={m.get('mae', float('nan')):.2f}  "
                  f"R={m.get('pearson', float('nan')):.3f}")

    print("\n[6] Artifact paths")
    print(f"  annotations:   {ANN}")
    print(f"  comparison videos (heur vs vlm): renders/v1_heuristic_vs_vlm/")
    print(f"  student eval (heur):             renders/v1_eval_heuristic_student/heuristic/")
    print(f"  student eval (vlm):              renders/v1_eval_vlm_student/vlm/")
    print(f"  checkpoints:                     checkpoints/{{heuristic,vlm}}/best.pt")


if __name__ == "__main__":
    main()
