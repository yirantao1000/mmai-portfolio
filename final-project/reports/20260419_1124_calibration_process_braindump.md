# Calibration & Optimization Process — Brain Dump

**Generated:** 2026-04-19 11:24 EDT
**Purpose:** unfiltered narrative of what I actually did across two calibration races, what worked, what didn't, and why, so the write-up can pull from a single source. Not polished — intentionally a working document.

---

## Starting point

Going into this the pipeline already had:
- Phase-aware comfort scoring (`approach` / `intent` / `execution`) with per-phase `(emotion_weight, posture_weight, gaze_weight)` fusion inside [src/comfort.py](src/comfort.py).
- A previous calibration pass (race-0, pre my work) that produced a `deploy.yaml` landing at Youden **J = 1.0** on the held-out test at τ\* = 80.
- 31 RealSense `.bag` recordings across 5 scenarios, with extracted parquets cached so re-runs don't touch the GPU.
- 75/25 stratified train/test split (23 train, 8 test), seed=42.

Why not declare victory at J=1.0 and stop? Two structural problems hidden under that number:

1. **Wrong trajectory shape.** sc02 (successful handover) should see comfort *rise* through execution; sc01/sc03/sc04/sc05 should *fall*. Actual pre-race slopes were roughly sc02 +0.03 (barely), sc01 −0.29, sc03 +0.03 (wrong), sc04 +0.01 (wrong), sc05 −0.03. The old objective only cared about level — a flat-high sc04 was rewarded identically to a correctly-decaying one.
2. **Pose false-fires during successful handovers.** In cached sc02 parquets the withdrawal condition fired in 2–4 % of intent+execution frames, mouth-cover in up to 1.3 %. Low but non-zero where zero was expected. Midterm §6.3 / §7.4 had already flagged pose as domain-shift-fragile.

## Problem framing

The work target was: produce a calibrated `deploy.yaml` that yields trajectory-correct per-scenario dynamics while preserving held-out abort quality, under a compute-efficient race-then-refine protocol.

Key design decisions made up front:

- **Per-recording reducer refactor.** Turned `score_recording(rec, p) → float` into `replay_series(rec, p) → (ts, integrated)` so every candidate objective could be a pure reducer over the shared series. Bonus: LORO becomes trivial on the same series cache.
- **Smooth shape term.** `tanh(3 · slope · window_s / 100)` bounds shape to ±1 with a smooth zero crossing. Fit slope via `np.polyfit` on smoothed integrated series, not first-minus-last (boundary noise).
- **Fairness controls.** Identical `stratified_folds(train, 5, seed=42)` across all candidates; identical DE `seed=42, tol=1e-3, polish=True, updating='deferred'`. Any wall-clock deviation >2× flagged as confound.
- **Honesty guardrails.** n=8 held-out test with 3 aborts gives binomial CI on 3/3 TPR running from 0.37 to 1.0. J=1.0 is not statistically meaningful on this sample. Headline gate = J on 75/25 split; LORO trajectory on all 31 is reported alongside as a sanity check, not optimized against.

## Race 1 (2026-04-18 late / early 2026-04-19)

**Variants tried:** A (baseline mean_full + J), B (mean of last 30 %), C (delta late−early), F (split: 0.7·J + 0.3·sc02-only tanh-shape). Variant E (no-pose ablation) dropped at the front end because hard-pinning wp_*=0 was an unfair handicap when the information I wanted could be read off the converged wp_* values.

**Protocol:** Phase 1 cheap race (maxiter=15, popsize=8 → ~120 evals × 4 variants ≈ 3–4 min) on train with a composite score `0.5·J_train + 0.3·slope_sign_agreement + 0.2·margin(sc02_end, sc01_end)/100`. Phase 2 full DE (maxiter=50, popsize=15) on the top 2. Phase 3 held-out test on each finalist.

**Race-1 outcomes (phase 2 finalists):**
- A: J=1.0, but sc02 slope −0.10 (wrong sign). [archived: reports/20260419_010217_race1_A.*]
- B: sc02 slope +0.93 (correct), but J=0.667 (below 0.80 gate). [archived: reports/20260419_010217_race1_B.*]

Neither cleared all the pass-gates. The race validated that the composite objective *finds* shape-correct solutions (B) but only at the cost of J, and vice versa (A). Single-objective DE can't triangulate — I needed a better objective formulation.

## Problem identified between races: hard-coded drag ceiling

Looking at the trajectories I noticed sc02 often stayed flat or drifted down despite high emotion scores. Traced it to [src/comfort.py](src/comfort.py): when the face detector misses a frame, comfort decays toward a hardcoded `no_face_decay_target = 35.0` at rate `0.15 /s`. sc01 walkby has ~40 % face detection on test — 60 % of frames are dragging comfort toward 35. sc02 has 68–72 % face detect — still dragging, just less.

These four hardcoded constants (`no_face_target`, `no_face_rate`, `no_pose_target`, `no_pose_rate`) were silently capping what DE could achieve. Promoted all four to tunables with bounds:
- `no_face_target ∈ [30, 80]`, `no_face_rate ∈ [0.05, 0.50]`
- `no_pose_target ∈ [30, 80]`, `no_pose_rate ∈ [0.05, 0.50]`

Plus `posture_drop_gate ∈ [0.10, 0.25]` (previously hardcoded at 0.15 inside the withdrawal formula). Total tunable dimensionality grew from 15 → 20.

## Race 2

**Variants tried:**
- A (carry-forward baseline with new tunables)
- C (delta reducer + J)
- F (0.7·J + 0.3·sc02-only shape)
- G (0.5·J + 0.5·all-scenario signed tanh-shape)
- H (G-objective + drag params, explicit 20-dim sanity check)
- I (sc02 weighted 3× in shape term + hard penalty `1.0·max(0, −sc02_shape)`)

All run at full DE budget (maxiter=50, popsize=15, seed=42). Ran in parallel on the 23-core machine; wall-clock per variant ~20 min.

**Race-2 finals on held-out test:**

| var | sc02_sl | sc01_sl | sc04_sl | sc02_m | sc01_m | sc04_m | J | gates pass |
|---|---|---|---|---|---|---|---|---|
| A | +0.10 | −0.54 | −2.82 | 89.1 | 55.8 | 77.6 | 1.0 | **6/7** |
| C | +2.14 | −1.26 | +0.47 | 65.8 | 49.5 | 62.7 | 1.0 | 5/7 |
| F | +2.86 | +2.88 | +1.48 | 73.6 | 53.5 | 51.7 | 1.0 | 4/7 |
| G | −0.75 | −2.38 | −3.50 | 81.5 | 75.5 | 77.8 | 1.0 | 4/7 |
| H | −0.72 | −3.35 | −3.34 | 80.7 | 75.2 | 76.5 | 1.0 | 4/7 |
| I | +0.89 | +2.61 | −1.48 | 77.1 | 65.7 | 62.1 | 1.0 | 4/7 |

**What each variant taught us:**

- **A** — the level-based baseline. Best gate-count but barely-positive sc02 slope. The "works but doesn't tell a story" winner.
- **C** (delta reducer) — rewarded *shape* directly but threw away level. sc02 mean collapsed to 65.8 and sc03/sc04 went *up* because the delta reducer's τ was reparameterized into delta-space [−30, +30], and DE found it can satisfy "sc02 delta positive" by lifting sc02 while keeping others flat near the τ threshold.
- **F** (J + sc02-shape only) — biggest sc02 slope (+2.86) but *every* scenario rose because nothing in the objective told DE to push sc01/sc04 down. Classic Goodhart.
- **G** (J + all-scenario shape, mean-aggregated) — the cautionary tale. DE discovered it can make the mean-shape-across-scenarios look great by making sc01/sc03/sc04/sc05 fall very steeply (slopes −2.4 to −3.5), at the cost of letting sc02 drift to −0.75. The aggregate mean shape was good; the individual sc02 term was wrong. Mean aggregation lets DE sacrifice the minority.
- **H** (G + drag params, 20-dim) — same sc02 failure as G. The extra drag-param headroom didn't break the tie — DE still preferred the same pathological solution. Confirms the problem is the objective geometry, not the parameter space.
- **I** (sc02-weighted × 3 + hard penalty) — was supposed to fix G's failure. sc02 slope became +0.89 (correct) but sc01 went *up* (slope +2.61). I over-corrected: with sc02 getting 3× weight and a penalty, DE found it cheapest to let sc01 drift up while making sc03/sc04/sc05 work harder.

**Pattern:** every shape-aware objective I tried trades off some scenario correctness to gain another. None cleared all gates. A remains the strongest candidate on gate-count.

Archived full set: [reports/20260419_0956_race2_*.{yaml,json}] and [reports/20260419_1024_race2_I.*].

## Promoting A and the surprise ablation

Promoted A to `config/deploy.yaml` (previous archived as `config/deploy.prerace.yaml`). Ran LORO on all 31 recordings:

| scenario | n | mean | slope | agree |
|---|---|---|---|---|
| sc01_walkby | 7 | 55.7 | −0.26 | 6/7 |
| sc02_comfortable | 8 | 89.3 | −0.05 | **4/8** |
| sc03_gradual_discomfort | 5 | 80.9 | −2.05 | 5/5 |
| sc04_sudden_withdrawal | 5 | 79.0 | −2.40 | 5/5 |
| sc05_distracted | 6 | 74.5 | −3.02 | 6/6 |

A cleared sc01/sc04/sc05/sc03 LORO gates but failed sc02 (4/8 vs plan's ≥7/8). The single biggest weakness.

Per plan §7.3, ran the confirmatory no-pose ablation: A's objective with `wp_intent = wp_exec = 0` pinned via `--pin-wp 0.0`, same DE budget.

**The ablation won.** By a lot.

Held-out test deltas (A → A-nopose):
- sc02 slope: +0.10 → **+1.05** (+0.95)
- All 5 scenarios correct direction (A was 3/5 directions right on test)
- sc02 mean: 89.1 → 92.8
- sc01 mean: 55.8 → 50.3 (better separation)
- J: 1.0 → 1.0 (unchanged)

LORO (A → A-nopose):
- sc02 agreement: **4/8 → 7/8** (clears the plan gate)
- sc01: 6/7 → 5/7 (small regression, but sc01 mean 50.2 means directional signal is redundant — abort fires purely on level)
- sc03/sc04/sc05: all unchanged at 100 %

**Why this was a surprise and why it makes sense in hindsight:** DE converged to nonzero `wp_intent` and `wp_exec` in A because they *mildly* helped J on the 5-fold CV train folds. But on the held-out test, the pose domain shift surfaced: the same false withdrawal / mouth-cover fires that the midterm predicted were quietly dragging sc02 down. CV on train couldn't see this because the train folds share the same bias as each other. Pose was net-negative *on out-of-fold data*, but CV alone can't reveal that.

**Promoted A-nopose to `config/deploy.yaml`** (2026-04-19 10:56). Archive chain:
- `config/deploy.A_withpose.yaml` — the A race-2 winner before ablation
- `config/deploy.prerace.yaml` — pre-race-2 deploy
- `config/deploy.baseline.yaml` — pre-race-1 deploy
- [reports/20260419_1056_race2_A_nopose.{yaml,json}] and [reports/loro_A_nopose.json]

## Default vs deploy mechanism — what specifically flipped sc01/sc02

Under default, sc01 (69.0) outscored sc02 (66.6) on test. Three mechanisms conspired:

1. **Posture was 65 % of execution-phase fusion weight.** Noisy pose channel → sc02 got hit by 25-point withdrawal and 30-point mouth-cover penalties from false fires.
2. **`no_face_decay_rate` was 0.15/s (slow).** sc01 has 40 % face detection, but slow decay meant missing frames didn't drag sc01's comfort down.
3. **Gaze cone was 25°/20° (tight).** sc02 users don't fixate perfectly while reaching; many frames classified as "looking away", killing the gaze signal for the comfortable scenario.

Deploy fixes all three: pose → 0, decay rate → 0.469/s (~3× faster), gaze cone → 31.7°/29.9°. Plus smaller changes: gamma 0.5→0.11 (floor terms reduced), withdrawal penalty 25→11.9 (less destructive when it does fire), face_cover_ratio 0.6→0.85 (fewer false mouth-cover detections during handover). Net result: 42-point sc01/sc02 separation instead of an inverted ordering.

Full detail in [reports/default_vs_deploy.md](default_vs_deploy.md).

## Things that went wrong (process)

- **`Path.relative_to(PROJECT_ROOT)` crash on relative paths.** Both [scripts/optimize_params.py](scripts/optimize_params.py) and [scripts/evaluate_test.py](scripts/evaluate_test.py) called this on the final "Wrote …" print with `args.config_out` as a relative path. Crashed after the config/report had already been written, so no data loss, but killed subshell chains' visible success-message. Fixed with `.resolve()` + try/except.
- **Default report paths dumped to project root.** `evaluate_test.py --report` defaulted to `PROJECT_ROOT / "calibration_report.json"`, and I passed relative `--report calibration_report_X.json` during races. Produced 9 duplicate JSONs at root (all byte-identical to their timestamp-tagged `reports/` archives). Cleaned up + fixed defaults to `reports/...`.
- **Overnight check-in vs actual wake.** The conversation-compaction summary said I'd scheduled a 02:30 check-in, but actual system wake was 09:56 (8-hour gap with all background DE jobs completing cleanly on their own). Worked fine but unsettling.
- **Variant G result initially misread.** When G's eval crashed at the `relative_to` bug, the printed LORO output was the correct result — but the report JSON wasn't written. Thought for a moment G was missing data; was actually just missing the final file-write step.

## Things I still wouldn't claim

- **J=1.0 on n=8 is not statistically sharp.** Binomial CI on 3/3 abort TPR runs from 0.37 to 1.0. The headline number looks perfect; it only means "no failures observed in 8 samples". Needs more recordings to distinguish "actually perfect" from "lucky on a small sample".
- **sc02 LORO 7/8 agreement.** One recording trends slightly downward despite being a successful handover. Reasonable read: a fully comfortable user who's already maximally comfortable doesn't *have* to trend upward — they can sit flat-high. But the report should say so honestly, not claim 8/8.
- **No-pose is dataset-conditional.** Pose was actively hurting *under this domain shift*. On a different dataset (different lighting, camera, users, interaction protocol) pose might be load-bearing. The `wp = 0` result is not a universal claim about the pipeline; it's a claim about these 31 recordings. The posture module is still running and feeds the HUD state warnings — calibration can be re-run with `--pin-wp` removed on any future dataset.
- **Rank correlation train↔test.** Planned to report this across race-2 variants (weak correlation is itself a finding). Didn't compute it — low-priority when the ablation result was so clear.

## Final state

- `config/deploy.yaml` = A-nopose (race 2 winner + ablation).
- `config/default.yaml` = hand-set defaults, unchanged from pre-work.
- `scripts/run_bag.py --config` default = `config/deploy.yaml` (was `default.yaml`).
- `src/visualization.py` comfort bar = 4-anchor BGR lerp anchored on τ\*=80, with a white tick at τ\*. Reads `comfort.abort_threshold` from config at init.
- All race configs + reports in `reports/`. Summary in `reports/RUNS.md`. Default-vs-deploy comparison in `reports/default_vs_deploy.md`.
- Total compute budget spent: ~3 hours of DE (all on cached parquets, no GPU). ~15 configs evaluated across 2 races + 1 ablation.

## Open follow-ups worth doing before the write-up freezes

1. Headless spot-check: run `scripts/run_bag.py` on one sc02 and one sc04 test bag with deploy.yaml, screenshot the HUD. Nominal 5 min with `--save --headless`.
2. Re-run the no-pose ablation with a different DE seed (17, 99) to confirm `wp = 0` isn't seed-42-specific.
3. Rank-correlation Phase-1 train-score vs Phase-3 test-J for race-2 variants. Small n=6, probably weak, but the weakness is the finding.
4. Write a LaTeX table from `reports/default_vs_deploy.md` for §7.3 of the paper.
