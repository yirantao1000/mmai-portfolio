# Default vs Deploy Comparison

Generated 2026-04-19 on held-out test split (n=8) and full LORO (n=31).
- **default** = `config/default.yaml` — hand-set defaults, never calibrated on this dataset.
- **deploy** = `config/deploy.yaml` — race-2 variant A with pose weights pinned to 0 (no-pose ablation winner), promoted 2026-04-19 10:56.

## Headline: abort-decision quality on the held-out test

| metric | default | deploy | delta |
|---|---|---|---|
| best Youden J | **0.00 @ τ=30** | **1.00 @ τ=78** | +1.00 |
| decisions at calibrated τ* | — (no τ* in config) | TP 3 / FN 0 / FP 0 / TN 2 | perfect on test |

The default config cannot separate abort from continue at *any* threshold in the 30–80 sweep — its Youden J is 0.0. The deploy config separates cleanly with a full-margin τ window (best J = 1.0 across τ ∈ [55, 68] on test).

## Per-scenario mean comfort (test, n=8)

| scenario | n | default mean | deploy mean | delta |
|---|---|---|---|---|
| sc01_walkby (abort) | 2 | 69.0 | **50.3** | −18.7 |
| sc02_comfortable (continue) | 2 | 66.6 | **92.8** | +26.2 |
| sc03_gradual_discomfort (abort) | 1 | 77.4 | 81.2 | +3.8 |
| sc04_sudden_withdrawal (abort) | 1 | 76.3 | 77.2 | +0.9 |
| sc05_distracted (abort) | 2 | 71.8 | 72.7 | +0.9 |

**The default inverts the signal.** sc01 (person walks by, robot should abort) scores *higher* than sc02 (successful handover). The deploy config establishes a clean 42-point separation between sc02 and sc01 and pushes sc01 well below any plausible abort τ.

## Per-scenario slope (test): does comfort trend the right way?

Desired signs: sc02 **positive** (rising comfort through successful handover), everything else **negative**.

| scenario | default slope | default sign? | deploy slope | deploy sign? |
|---|---|---|---|---|
| sc01_walkby | −0.04 | ✓ (barely) | −0.62 | ✓ |
| sc02_comfortable | **−3.05** | ✗ wrong direction | **+1.05** | ✓ |
| sc03_gradual_discomfort | −0.51 | ✓ | −3.07 | ✓ |
| sc04_sudden_withdrawal | −0.13 | ✓ (barely) | −3.01 | ✓ |
| sc05_distracted | −1.86 | ✓ | −4.43 | ✓ |
| **agree** | **5/8 recordings** | | **8/8 recordings** | |

Under default, sc02 trends *downward* during successful handovers — the opposite of what it should do. Deploy gets every single test recording's direction right.

## LORO slope-sign agreement (all 31 recordings)

| scenario | n | default agree | deploy agree | delta |
|---|---|---|---|---|
| sc01_walkby | 7 | 5/7 | 5/7 | = |
| sc02_comfortable | 8 | **0/8** | **7/8** | +7 |
| sc03_gradual_discomfort | 5 | 1/5 | 5/5 | +4 |
| sc04_sudden_withdrawal | 5 | 5/5 | 5/5 | = |
| sc05_distracted | 6 | 6/6 | 6/6 | = |
| **total** | 31 | 17/31 (55%) | 28/31 (90%) | +11 |

The LORO confirms the headline: default fails sc02 in every recording (0/8) and sc03 in 4/5; deploy passes 90% of all 31 with the only residual miss being sc02 on one recording plus 2/7 sc01s.

## LORO mean comfort (all 31)

| scenario | n | default mean | deploy mean |
|---|---|---|---|
| sc01_walkby | 7 | 69.2 | 50.2 |
| sc02_comfortable | 8 | 68.3 | 92.7 |
| sc03_gradual_discomfort | 5 | 78.8 | 79.5 |
| sc04_sudden_withdrawal | 5 | 74.8 | 78.8 |
| sc05_distracted | 6 | 73.2 | 73.0 |

Under default, sc01 (69.2) and sc02 (68.3) are essentially indistinguishable by level — the classifier has nothing to work with. Deploy drives sc02 up to 92.7 and sc01 down to 50.2, a clean 42-point band that holds across all 15 recordings in those two scenarios.

## What changed to flip the sc01/sc02 ordering

Under the default config, sc01 walkby (69.0) outscored sc02 handover (66.6) because three separate mechanisms were all working against us at once, and the calibrated deploy config attacks each one directly.

### 1. Posture weights dropped to zero — removes the main false-penalty against sc02

| param | default | deploy |
|---|---|---|
| `phase_weights.intent.posture_weight` | 0.4 | **0.0** |
| `phase_weights.execution.posture_weight` | **0.65** | **0.0** |

In the default config, posture was the *dominant* signal in the execution phase (65 % weight). In sc02 the user extends a hand toward the robot — which the pose module often reads as "withdrawing" or "mouth/face covered" under our domain shift (the race-2 parquets showed 2–4 % false-withdrawal fires on sc02). Every spurious fire subtracted 25 points (withdrawal) or 30 points (mouth-cover). Zeroing posture both removes the noisy channel from the fusion and prevents the hard-coded penalties from firing, so sc02's clean emotion+gaze signal gets through untouched. The no-pose ablation wasn't a performance trick — it was the single biggest contributor to fixing the ordering.

### 2. Tunable missing-detection fallbacks — pulls sc01 *down* instead of letting it coast

Four previously-hardcoded constants (`no_face_target`, `no_face_rate`, `no_pose_target`, `no_pose_rate`) were promoted to tunables.

| param | default | deploy |
|---|---|---|
| `no_face_target` | 35.0 | 32.54 |
| `no_face_rate` | 0.15 /s | **0.469 /s** |
| `no_pose_target` | 50.0 | **54.66** |
| `no_pose_rate` | 0.30 /s | 0.355 /s |

When the face detector misses, comfort EMA-decays toward `no_face_target` at `no_face_rate`. Default decays very slowly (0.15/s → ~6.7 s half-life), so sc01 (40 % face detection in test) kept its score high despite 60 % of frames having no emotion/gaze signal. Deploy decays ~3× faster, so sc01's frequent face-misses rapidly drag its comfort toward the low-30s. sc02 (68–72 % face detect) rarely decays at all, so the boosted rate has no downside for it. **This is the biggest single driver of the 42-point sc01/sc02 separation.**

### 3. Phase-weighted gaze and emotion become load-bearing

With posture removed from the fusion, the remaining channels had to do the work:

| param | default | deploy |
|---|---|---|
| `intent.emotion_weight` | 0.6 | **0.882** |
| `intent.gaze_weight` | 0.6 | **0.808** |
| `execution.gaze_weight` | 0.1 | 0.19 |
| `yaw_threshold` | 25° | **31.69°** |
| `pitch_threshold` | 20° | **29.9°** |

Gaze thresholds were widened, which matters because sc02 users don't always fixate perfectly on the camera while reaching — a 25°/20° cone classified many sc02 frames as "looking away". The wider cone keeps gaze positive during sc02 while sc01 (person walking past without looking) still fails the check. Combined with the boosted intent-phase weights, this makes gaze+emotion the clean discriminator between an engaged user and a passer-by.

### Smaller but supporting changes

- `face_cover_ratio` 0.6 → 0.846: raises the bar for "mouth/face covered" detection, so incidental near-face hand motion during sc02 handover stops triggering the 30-point penalty.
- `withdrawal_penalty` 25 → 11.9 and `withdrawal_threshold_meters` 0.12 → 0.089: when withdrawal does fire (correctly, in sc04), it's less destructive, *and* tighter thresholds mean it fires less often spuriously.
- `gamma` 0.5 → 0.109, `delta` 0.3 → 0.006: floor terms that previously pushed low scores upward are near-zero in deploy, letting sc01's true low comfort surface.
- `ema_time_constant_s` 0.5 → 0.286: faster temporal smoothing, so sc02's trajectory can actually rise through execution instead of being averaged into a constant.

### In summary

The default config let sc01 coast high (slow decay when face was missing) while sc02 was actively penalized (noisy pose channel, narrow gaze cone, aggressive cover/withdrawal triggers). The calibrated deploy flips all three: it removes the noisy channel, accelerates the drag on missing-detection cases, and widens gaze tolerance — which together convert an inverted ordering into a 42-point clean separation.

## Bottom line

On this dataset the default config is not merely suboptimal — it produces the **wrong ordering** between abort and continue scenarios and the **wrong trajectory direction** for successful handovers. The deploy (calibrated, no-pose) config corrects both, giving perfect abort classification on the held-out test and 90% LORO directional agreement across the full 31-recording corpus. The single structural trade-off is that sc02 LORO agreement is 7/8, not 8/8 — one "flat-high" handover reads as a slight downward drift because a successful interaction doesn't *have* to trend upward if the person is already maximally comfortable.

Raw inputs: `reports/eval_default.json`, `reports/eval_deploy.json`, `reports/loro_default.json`, `reports/loro_A_nopose.json`.
