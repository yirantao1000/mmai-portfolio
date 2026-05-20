#!/usr/bin/env python3
"""Stage B — joint parameter search via differential evolution.

Reads cached raw features (Stage A) + JSON sidecar keypoints, replays scoring
entirely in numpy for a candidate config, and maximizes a direct classification
objective aligned with the §7.3 abort decision.

Objective (maximize):
  J(τ*) + ε · margin(sc02, sc01)
where J(τ) = TPR_abort(sc01+sc04 @ τ) + TNR_continue(sc02 @ τ) - 1
(Youden's J), τ* = argmax over τ ∈ [30, 80].

5-fold CV is stratified by class label {continue, abort, ambiguous} so each fold
has a mix of sc02 and sc01+sc04; ambiguous (sc03+sc05) files are scored but don't
contribute to J (they're reported separately at test time).

Writes best config to config/deploy.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.phases import PhaseWindows, find_sidecar, phase_at, windows_from_sidecar


# -------- Labels ---------------------------------------------------------------

CONTINUE = {"sc02_comfortable"}
ABORT = {"sc01_walkby", "sc04_sudden_withdrawal"}
# sc03 (gradual discomfort) and sc05 (distracted) are ambiguous — scored but
# excluded from the J objective. They're reported at test time for completeness.


def class_of(scenario: str) -> str:
    if scenario in CONTINUE:
        return "continue"
    if scenario in ABORT:
        return "abort"
    return "ambiguous"


# -------- Cached-frame dataset -------------------------------------------------

@dataclass
class Recording:
    scenario: str
    name: str
    df: pd.DataFrame
    windows: PhaseWindows
    label: str  # "continue" | "abort" | "ambiguous"


def load_recordings(split_path: Path, which: str) -> list[Recording]:
    """Load cached parquets listed under split[which]."""
    with open(split_path) as f:
        manifest = json.load(f)

    recordings: list[Recording] = []
    for scenario, groups in manifest["scenarios"].items():
        for rel in groups.get(which, []):
            bag_path = PROJECT_ROOT / rel
            parquet_path = PROJECT_ROOT / "cache" / scenario / (bag_path.stem + ".parquet")
            sidecar_path = find_sidecar(bag_path)
            if not parquet_path.exists():
                print(f"  [warn] missing cache: {parquet_path.relative_to(PROJECT_ROOT)} — run extract_features first")
                continue
            if sidecar_path is None:
                print(f"  [warn] missing sidecar for {bag_path.relative_to(PROJECT_ROOT)}")
                continue
            df = pd.read_parquet(parquet_path)
            windows = windows_from_sidecar(sidecar_path)
            recordings.append(Recording(
                scenario=scenario, name=bag_path.stem, df=df, windows=windows,
                label=class_of(scenario),
            ))
    return recordings


# -------- Scoring replay (pure numpy, fast) -----------------------------------

# Parameter vector layout — kept flat so differential_evolution can mutate it.
PARAM_NAMES = [
    "we_intent", "wp_intent", "wg_intent",
    "we_exec",   "wp_exec",   "wg_exec",
    "yaw_threshold",       # degrees, gaze engaged if |yaw| < this
    "pitch_threshold",
    "face_cover_ratio",    # hand_to_mouth_ratio < this → covering mouth
    "mouth_cover_penalty",
    "withdrawal_threshold_m",
    "withdrawal_penalty",
    "gamma",               # valence coeff
    "delta",               # arousal coeff
    "ema_time_constant_s",
    "posture_drop_gate",   # withdrawing only fires if posture_drop <= this
    "no_face_target",      # smoothed emotion decays toward this when face missing
    "no_face_rate",        # decay rate toward no_face_target (1/s)
    "no_pose_target",      # smoothed posture decays toward this when pose missing
    "no_pose_rate",        # decay rate toward no_pose_target (1/s)
]

BOUNDS = [
    (0.1, 0.9), (0.1, 0.9), (0.0, 1.0),   # intent weights
    (0.1, 0.9), (0.1, 0.9), (0.0, 1.0),   # execution weights
    (5.0, 40.0),                           # yaw threshold
    (5.0, 40.0),                           # pitch threshold
    (0.3, 0.9),                            # face_cover_ratio
    (0.0, 50.0),                           # mouth_cover_penalty
    (0.05, 0.30),                          # withdrawal_threshold_m
    (0.0, 50.0),                           # withdrawal_penalty
    (0.0, 1.0),                            # gamma
    (0.0, 1.0),                            # delta
    (0.1, 2.0),                            # ema_time_constant_s
    (0.10, 0.25),                          # posture_drop_gate
    (30.0, 80.0),                          # no_face_target
    (0.05, 0.50),                          # no_face_rate
    (30.0, 80.0),                          # no_pose_target
    (0.05, 0.50),                          # no_pose_rate
]


def params_to_dict(x: np.ndarray) -> dict:
    return dict(zip(PARAM_NAMES, x.tolist()))


def replay_series(rec: Recording, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Replay one recording under params `p`. Returns (ts, integrated) restricted
    to the intent+execution window. Reducers consume these arrays."""
    df = rec.df
    ts = df["timestamp_s"].to_numpy(dtype=np.float32)
    n = len(ts)
    if n == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    dt = np.diff(ts, prepend=ts[0] - 1.0 / 30.0)
    dt = np.clip(dt, 1e-3, None)

    tau = p["ema_time_constant_s"]
    alpha = np.where(tau > 0, 1.0 - np.exp(-dt / tau), 1.0).astype(np.float32)

    gaze_yaw = df["gaze_yaw_deg"].to_numpy(dtype=np.float32)
    gaze_pitch = df["gaze_pitch_deg"].to_numpy(dtype=np.float32)
    gaze_avail = df["gaze_available"].to_numpy(dtype=bool)
    looking = gaze_avail & (np.abs(gaze_yaw) < p["yaw_threshold"]) & (np.abs(gaze_pitch) < p["pitch_threshold"])

    valence = df["valence"].to_numpy(dtype=np.float32)
    arousal = df["arousal"].to_numpy(dtype=np.float32)
    face_det = df["face_detected"].to_numpy(dtype=bool)

    phases = np.array([phase_at(rec.windows, t) for t in ts])

    we_by_phase = {"approach": 0.5, "intent": p["we_intent"], "execution": p["we_exec"]}
    wp_by_phase = {"approach": 0.5, "intent": p["wp_intent"], "execution": p["wp_exec"]}
    wg_by_phase = {"approach": 0.3, "intent": p["wg_intent"], "execution": p["wg_exec"]}

    wg_f = np.array([wg_by_phase[ph] for ph in phases], dtype=np.float32)
    we_f = np.array([we_by_phase[ph] for ph in phases], dtype=np.float32)
    wp_f = np.array([wp_by_phase[ph] for ph in phases], dtype=np.float32)
    total = we_f + wp_f
    total[total <= 0] = 1.0
    we_f, wp_f = we_f / total, wp_f / total

    gamma, delta = p["gamma"], p["delta"]
    raw_e = gamma * valence - delta * arousal + wg_f * looking.astype(np.float32)
    max_mag = gamma + delta + wg_f
    max_mag = np.where(max_mag <= 0, 1.0, max_mag)
    instant_e = ((raw_e + max_mag) / (2.0 * max_mag)) * 100.0
    instant_e = np.clip(instant_e, 0.0, 100.0)
    instant_e = np.where(face_det, instant_e, np.nan)

    open_posture = df["open_posture_score"].to_numpy(dtype=np.float32)
    pose_det = df["pose_detected"].to_numpy(dtype=bool)
    hand_to_mouth = df["hand_to_mouth_ratio"].to_numpy(dtype=np.float32)
    z_disp = df["z_displacement_m"].to_numpy(dtype=np.float32)
    posture_drop = df["posture_drop"].to_numpy(dtype=np.float32)

    covering = np.where(np.isnan(hand_to_mouth), False, hand_to_mouth < p["face_cover_ratio"])
    withdrawing = np.where(
        np.isnan(z_disp) | np.isnan(posture_drop),
        False,
        (z_disp > p["withdrawal_threshold_m"]) & (posture_drop <= p["posture_drop_gate"]),
    )

    instant_p = open_posture * 100.0
    instant_p = np.where(covering, instant_p - p["mouth_cover_penalty"], instant_p)
    instant_p = np.where(withdrawing, instant_p - p["withdrawal_penalty"], instant_p)
    instant_p = np.clip(instant_p, 0.0, 100.0)
    instant_p = np.where(pose_det, instant_p, np.nan)

    no_face_target = p.get("no_face_target", 35.0)
    no_face_rate = p.get("no_face_rate", 0.15)
    no_pose_target = p.get("no_pose_target", 50.0)
    no_pose_rate = p.get("no_pose_rate", 0.30)

    sm_e = np.empty(n, dtype=np.float32)
    sm_p = np.empty(n, dtype=np.float32)
    e_prev, p_prev = 50.0, 50.0
    for i in range(n):
        a = alpha[i]
        if np.isnan(instant_e[i]):
            decay_a = 1.0 - np.exp(-dt[i] * no_face_rate)
            e_prev = e_prev + (no_face_target - e_prev) * decay_a
        else:
            e_prev = a * instant_e[i] + (1 - a) * e_prev
        if np.isnan(instant_p[i]):
            decay_a = 1.0 - np.exp(-dt[i] * no_pose_rate)
            p_prev = p_prev + (no_pose_target - p_prev) * decay_a
        else:
            p_prev = a * instant_p[i] + (1 - a) * p_prev
        sm_e[i] = e_prev
        sm_p[i] = p_prev

    integrated = we_f * sm_e + wp_f * sm_p

    intent = rec.windows.intent
    execution = rec.windows.execution
    lo = intent[0] if intent is not None else ts[0]
    hi = execution[1] if execution is not None else (intent[1] if intent is not None else ts[-1])
    mask = (ts >= lo) & (ts <= hi)
    if not mask.any():
        return ts, integrated
    return ts[mask], integrated[mask]


# -------- Reducers -------------------------------------------------------------
# Each reducer takes (ts, integrated) from replay_series and returns a scalar.

def reduce_mean_full(ts: np.ndarray, integrated: np.ndarray) -> float:
    if integrated.size == 0:
        return float("nan")
    return float(np.mean(integrated))


def reduce_mean_late(ts: np.ndarray, integrated: np.ndarray, frac: float = 0.30) -> float:
    if integrated.size == 0:
        return float("nan")
    k = max(1, int(round(integrated.size * frac)))
    return float(np.mean(integrated[-k:]))


def reduce_delta_late_minus_early(
    ts: np.ndarray, integrated: np.ndarray, frac: float = 0.30
) -> float:
    if integrated.size < 2:
        return 0.0
    k = max(1, int(round(integrated.size * frac)))
    early = float(np.mean(integrated[:k]))
    late = float(np.mean(integrated[-k:]))
    return late - early


def slope_fit(ts: np.ndarray, integrated: np.ndarray) -> float:
    """Least-squares slope of integrated vs ts. Returns 0 for <2 samples."""
    if integrated.size < 2:
        return 0.0
    slope, _ = np.polyfit(ts, integrated, 1)
    return float(slope)


def tanh_shape(slope: float, window_s: float, k: float = 3.0) -> float:
    """Smooth, bounded mapping of slope to [-1, 1].

    For a full 0→100 sweep across the mask window, `slope * window_s / 100`
    is 1.0, so tanh(k * 1.0) ≈ 0.995 for k=3. Prevents DE from parking in
    flat regions where a sign() would flip on ε perturbations.
    """
    return float(np.tanh(k * slope * window_s / 100.0))


# -------- Backwards-compatible wrapper -----------------------------------------

def score_recording(rec: Recording, p: dict) -> float:
    """Back-compat shim: replays series and returns mean(integrated)."""
    ts, integrated = replay_series(rec, p)
    return reduce_mean_full(ts, integrated)


def score_all(recordings: list[Recording], p: dict) -> dict[str, float]:
    return {r.name: score_recording(r, p) for r in recordings}


# -------- Objective ------------------------------------------------------------

def youden_j(
    scores: list[tuple[float, str]],
    tau_range: tuple[float, float] = (30.0, 80.5),
    step: float = 1.0,
) -> tuple[float, float]:
    """Given [(comfort, label)], sweep τ over `tau_range`, return (best_J, τ*)."""
    comforts = np.array([c for c, _ in scores], dtype=np.float32)
    labels = [lab for _, lab in scores]
    n_abort = sum(1 for lab in labels if lab == "abort")
    n_cont = sum(1 for lab in labels if lab == "continue")
    if n_abort == 0 or n_cont == 0:
        return 0.0, (tau_range[0] + tau_range[1]) / 2

    best_j, best_tau = -1.0, (tau_range[0] + tau_range[1]) / 2
    for tau in np.arange(tau_range[0], tau_range[1], step):
        tp = sum(1 for c, lab in zip(comforts, labels) if lab == "abort" and c <= tau)
        tn = sum(1 for c, lab in zip(comforts, labels) if lab == "continue" and c > tau)
        j = tp / n_abort + tn / n_cont - 1.0
        if j > best_j:
            best_j, best_tau = j, float(tau)
    return best_j, best_tau


def stratified_folds(recordings: list[Recording], n_folds: int, seed: int = 42) -> list[list[int]]:
    """Return a list of fold index lists; each fold is stratified by label."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for i, r in enumerate(recordings):
        by_label.setdefault(r.label, []).append(i)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for label, idxs in by_label.items():
        rng.shuffle(idxs)
        for k, idx in enumerate(idxs):
            folds[k % n_folds].append(idx)
    return folds


# -------- Variants -------------------------------------------------------------
# Each variant is (reducer, tau_range) for the level-J variants, plus a marker
# for the split-objective variant F which adds a sc02-shape term.

VARIANTS = {
    "A": {"reducer": reduce_mean_full,             "tau_range": (30.0, 80.5)},
    "B": {"reducer": reduce_mean_late,             "tau_range": (30.0, 80.5)},
    "C": {"reducer": reduce_delta_late_minus_early, "tau_range": (-30.0, 30.5)},
    "F": {"reducer": reduce_mean_full,             "tau_range": (30.0, 80.5)},
    "G": {"reducer": reduce_mean_full,             "tau_range": (30.0, 80.5)},
    "I": {"reducer": reduce_mean_full,             "tau_range": (30.0, 80.5)},
}


def _level_j_objective(
    x: np.ndarray,
    recordings: list[Recording],
    folds: list[list[int]],
    eps: float,
    reducer,
    tau_range: tuple[float, float],
) -> float:
    p = params_to_dict(x)
    series = [replay_series(r, p) for r in recordings]
    comforts = np.array([reducer(ts, integ) for ts, integ in series], dtype=np.float32)
    labels = [r.label for r in recordings]

    js: list[float] = []
    for fold in folds:
        if not fold:
            continue
        held = [(float(comforts[i]), labels[i]) for i in fold]
        j, _ = youden_j(held, tau_range=tau_range)
        js.append(j)
    mean_j = float(np.mean(js)) if js else 0.0

    sc02_mask = np.array([r.scenario == "sc02_comfortable" for r in recordings])
    sc01_mask = np.array([r.scenario == "sc01_walkby" for r in recordings])
    if sc02_mask.any() and sc01_mask.any():
        margin = float(np.mean(comforts[sc02_mask]) - np.mean(comforts[sc01_mask]))
    else:
        margin = 0.0

    return -(mean_j + eps * (margin / 100.0))


def _all_scen_shape_objective_G(
    x: np.ndarray,
    recordings: list[Recording],
    folds: list[list[int]],
    eps: float,
) -> float:
    """Variant G: 0.5·J(level) + 0.5·mean(tanh_shape · desired_sign) over ALL
    recordings with a desired direction. Unlike F this penalizes sc03/sc04/sc05
    failing to decrease, not just sc02 failing to rise."""
    p = params_to_dict(x)
    series = [replay_series(r, p) for r in recordings]
    comforts = np.array(
        [reduce_mean_full(ts, integ) for ts, integ in series], dtype=np.float32
    )
    labels = [r.label for r in recordings]

    js: list[float] = []
    for fold in folds:
        if not fold:
            continue
        held = [(float(comforts[i]), labels[i]) for i in fold]
        j, _ = youden_j(held, tau_range=(30.0, 80.5))
        js.append(j)
    mean_j = float(np.mean(js)) if js else 0.0

    desired = {
        "sc02_comfortable": +1, "sc01_walkby": -1,
        "sc03_gradual_discomfort": -1, "sc04_sudden_withdrawal": -1,
        "sc05_distracted": -1,
    }
    shape_vals: list[float] = []
    for rec, (ts, integ) in zip(recordings, series):
        if integ.size < 2:
            continue
        want = desired.get(rec.scenario)
        if want is None:
            continue
        window_s = float(ts[-1] - ts[0]) if ts.size >= 2 else 1.0
        signed = tanh_shape(slope_fit(ts, integ), window_s) * want
        shape_vals.append(signed)
    shape = float(np.mean(shape_vals)) if shape_vals else 0.0

    return -(0.5 * mean_j + 0.5 * shape)


def _split_objective_F(
    x: np.ndarray,
    recordings: list[Recording],
    folds: list[list[int]],
    eps: float,
) -> float:
    """Variant F: 0.7·J(level) + 0.3·mean(tanh_shape) over sc02 recordings only."""
    p = params_to_dict(x)
    series = [replay_series(r, p) for r in recordings]
    comforts = np.array([reduce_mean_full(ts, integ) for ts, integ in series], dtype=np.float32)
    labels = [r.label for r in recordings]

    js: list[float] = []
    for fold in folds:
        if not fold:
            continue
        held = [(float(comforts[i]), labels[i]) for i in fold]
        j, _ = youden_j(held, tau_range=(30.0, 80.5))
        js.append(j)
    mean_j = float(np.mean(js)) if js else 0.0

    shape_vals: list[float] = []
    for rec, (ts, integ) in zip(recordings, series):
        if rec.scenario != "sc02_comfortable" or integ.size < 2:
            continue
        window_s = float(ts[-1] - ts[0]) if ts.size >= 2 else 1.0
        shape_vals.append(tanh_shape(slope_fit(ts, integ), window_s))
    shape = float(np.mean(shape_vals)) if shape_vals else 0.0

    return -(0.7 * mean_j + 0.3 * shape)


def _sc02_guarded_shape_objective_I(
    x: np.ndarray,
    recordings: list[Recording],
    folds: list[list[int]],
    eps: float,
) -> float:
    """Variant I: 0.5·J(level) + 0.5·weighted_shape − penalty, where
    weighted_shape averages per-scenario signed tanh-shape with sc02 weighted 3×
    and penalty = 1.0·max(0, -sc02_signed_shape) so DE cannot sacrifice sc02
    trajectory for aggregate shape (the failure mode seen in G).
    """
    p = params_to_dict(x)
    series = [replay_series(r, p) for r in recordings]
    comforts = np.array(
        [reduce_mean_full(ts, integ) for ts, integ in series], dtype=np.float32
    )
    labels = [r.label for r in recordings]

    js: list[float] = []
    for fold in folds:
        if not fold:
            continue
        held = [(float(comforts[i]), labels[i]) for i in fold]
        j, _ = youden_j(held, tau_range=(30.0, 80.5))
        js.append(j)
    mean_j = float(np.mean(js)) if js else 0.0

    desired = {
        "sc02_comfortable": +1, "sc01_walkby": -1,
        "sc03_gradual_discomfort": -1, "sc04_sudden_withdrawal": -1,
        "sc05_distracted": -1,
    }
    per_scen_signed: dict[str, list[float]] = {}
    for rec, (ts, integ) in zip(recordings, series):
        if integ.size < 2:
            continue
        want = desired.get(rec.scenario)
        if want is None:
            continue
        window_s = float(ts[-1] - ts[0]) if ts.size >= 2 else 1.0
        signed = tanh_shape(slope_fit(ts, integ), window_s) * want
        per_scen_signed.setdefault(rec.scenario, []).append(signed)

    means = {scen: float(np.mean(vals)) for scen, vals in per_scen_signed.items() if vals}
    weights = {"sc02_comfortable": 3.0}  # others default to 1.0
    num = 0.0
    denom = 0.0
    for scen, m in means.items():
        w = weights.get(scen, 1.0)
        num += w * m
        denom += w
    weighted_shape = num / denom if denom > 0 else 0.0

    sc02_shape = means.get("sc02_comfortable", 0.0)
    penalty = max(0.0, -sc02_shape)

    return -(0.5 * mean_j + 0.5 * weighted_shape - 1.0 * penalty)


def build_objective(variant: str):
    if variant == "F":
        return _split_objective_F
    if variant == "G":
        return _all_scen_shape_objective_G
    if variant == "I":
        return _sc02_guarded_shape_objective_I
    spec = VARIANTS[variant]
    reducer = spec["reducer"]
    tau_range = spec["tau_range"]

    def _obj(x, recordings, folds, eps):
        return _level_j_objective(x, recordings, folds, eps, reducer, tau_range)

    return _obj


# Kept for any external callers (back-compat).
def cv_objective(x: np.ndarray, recordings: list[Recording], folds: list[list[int]], eps: float) -> float:
    return _level_j_objective(
        x, recordings, folds, eps, reduce_mean_full, (30.0, 80.5)
    )


# -------- Driver ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Stage B — optimize comfort-scoring parameters.")
    parser.add_argument("--split", default=str(PROJECT_ROOT / "data" / "split.json"))
    parser.add_argument("--config-in", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--config-out", default=str(PROJECT_ROOT / "config" / "deploy.yaml"))
    parser.add_argument("--objective", choices=list(VARIANTS.keys()), default="A",
                        help="Objective variant: A=mean_full, B=mean_late, C=delta, F=split(J+sc02-shape).")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.1, help="Margin reward weight.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-wp", type=float, default=None,
                        help="Pin wp_intent and wp_exec to this value (e.g. 0.0 for no-pose ablation).")
    args = parser.parse_args()

    # Bounds can be overridden at runtime for ablations. Copy so global stays clean.
    bounds = list(BOUNDS)
    if args.pin_wp is not None:
        wp_i_idx = PARAM_NAMES.index("wp_intent")
        wp_e_idx = PARAM_NAMES.index("wp_exec")
        # Tiny non-zero width keeps DE happy; functionally pinned.
        eps_w = 1e-6
        bounds[wp_i_idx] = (args.pin_wp, args.pin_wp + eps_w)
        bounds[wp_e_idx] = (args.pin_wp, args.pin_wp + eps_w)
        print(f"[ablation] wp_intent and wp_exec pinned to {args.pin_wp}")

    split_path = Path(args.split)
    if not split_path.exists():
        print(f"split manifest not found: {split_path}. Run scripts/make_split.py first.")
        return 1

    print(f"Loading train recordings from {split_path.relative_to(PROJECT_ROOT)}...")
    train = load_recordings(split_path, "train")
    print(f"  {len(train)} train recordings")
    if not train:
        return 1

    folds = stratified_folds(train, args.folds, seed=args.seed)

    objective = build_objective(args.objective)
    print(f"\nObjective variant: {args.objective}")
    print(f"Optimizing {len(PARAM_NAMES)} parameters over {args.maxiter} iterations × popsize={args.popsize}...")
    result = differential_evolution(
        objective,
        bounds,
        args=(train, folds, args.eps),
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=1e-3,
        seed=args.seed,
        polish=True,
        updating="deferred",
        workers=1,
        disp=True,
    )

    best = params_to_dict(result.x)
    print("\nBest parameters:")
    for k, v in best.items():
        print(f"  {k:>28s} = {v:.4f}")

    # Production abort_threshold is ALWAYS derived from level-space mean_full,
    # regardless of which reducer drove the optimization. This keeps deploy.yaml
    # semantically consistent: the runtime pipeline produces a comfort level,
    # not a delta.
    series = [replay_series(r, best) for r in train]
    level_scores = [(reduce_mean_full(ts, integ), r.label) for r, (ts, integ) in zip(train, series)]
    j_level, tau_star = youden_j(level_scores, tau_range=(30.0, 80.5))
    print(f"\nFull-train J (level, mean_full) = {j_level:.3f} at τ* = {tau_star:.1f}")

    # Per-scenario level summary
    print("Per-scenario mean comfort (train, level):")
    by_scen: dict[str, list[float]] = {}
    for r, (c, _) in zip(train, level_scores):
        by_scen.setdefault(r.scenario, []).append(c)
    for scen in sorted(by_scen):
        vals = by_scen[scen]
        print(f"  {scen:<28s} n={len(vals)}  mean={np.mean(vals):.1f}  min={np.min(vals):.1f}  max={np.max(vals):.1f}")

    # Per-scenario slope summary (diagnostic)
    print("Per-scenario slope (train, polyfit on smoothed series):")
    slopes_by_scen: dict[str, list[float]] = {}
    for r, (ts, integ) in zip(train, series):
        if integ.size >= 2:
            slopes_by_scen.setdefault(r.scenario, []).append(slope_fit(ts, integ))
    for scen in sorted(slopes_by_scen):
        ss = slopes_by_scen[scen]
        print(f"  {scen:<28s} n={len(ss)}  mean_slope={np.mean(ss):+.4f}")

    # Write deploy yaml
    with open(args.config_in) as f:
        cfg = yaml.safe_load(f)

    cfg["gaze_detector"]["yaw_threshold"] = round(best["yaw_threshold"], 2)
    cfg["gaze_detector"]["pitch_threshold"] = round(best["pitch_threshold"], 2)
    cfg["pose_detector"]["face_cover_ratio"] = round(best["face_cover_ratio"], 3)
    cfg["pose_detector"]["withdrawal_threshold_meters"] = round(best["withdrawal_threshold_m"], 3)
    cfg["pose_detector"]["posture_drop_gate"] = round(best["posture_drop_gate"], 3)
    cfg["comfort"]["gamma"] = round(best["gamma"], 3)
    cfg["comfort"]["delta"] = round(best["delta"], 3)
    cfg["comfort"]["mouth_cover_penalty"] = round(best["mouth_cover_penalty"], 2)
    cfg["comfort"]["withdrawal_penalty"] = round(best["withdrawal_penalty"], 2)
    cfg["comfort"]["ema_time_constant_s"] = round(best["ema_time_constant_s"], 3)
    cfg["comfort"]["no_face_target"] = round(best.get("no_face_target", 35.0), 2)
    cfg["comfort"]["no_face_rate"] = round(best.get("no_face_rate", 0.15), 3)
    cfg["comfort"]["no_pose_target"] = round(best.get("no_pose_target", 50.0), 2)
    cfg["comfort"]["no_pose_rate"] = round(best.get("no_pose_rate", 0.30), 3)
    cfg["comfort"]["phase_weights"]["intent"] = {
        "emotion_weight": round(best["we_intent"], 3),
        "posture_weight": round(best["wp_intent"], 3),
        "gaze_weight":    round(best["wg_intent"], 3),
    }
    cfg["comfort"]["phase_weights"]["execution"] = {
        "emotion_weight": round(best["we_exec"], 3),
        "posture_weight": round(best["wp_exec"], 3),
        "gaze_weight":    round(best["wg_exec"], 3),
    }
    cfg["comfort"]["abort_threshold"] = round(tau_star, 2)
    cfg["comfort"]["calibrated_variant"] = args.objective

    out = Path(args.config_out).resolve()
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    try:
        rel = out.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out
    print(f"\nWrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
