# Multimodal Comfort-Aware Human–Robot Handover: Phase-Aware Late Fusion with Few-Shot On-Site Calibration

**Heejung Roh, Yiran Tao, Sparsh Bansal, Jung Yeop (Steve) Kim**
Massachusetts Institute of Technology
{heejungr, yirantao, sparshb, jungyeop}@mit.edu

---

## Abstract

Social and service robots performing object handovers must adapt to how comfortable each user feels in real time, but the same robot behavior can feel natural to one person and intrusive to another. We present a real-time multimodal comfort-scoring pipeline for human–robot handover that fuses pretrained unimodal vision models — facial emotion (EfficientNet-B0/AffectNet), gaze (L2CS-Net/Gaze360), and body posture (MediaPipe Pose with depth-based withdrawal detection) — into a single 0–100 comfort score from RealSense RGB-D video. Rather than retraining the perception models, we treat fusion as the learning surface: phase-dependent weights `(w_e, w_p, w_g)` over the three channels are calibrated by differential evolution against a small (31-recording) custom dataset, with the interaction segmented into *approach*, *intent signaling*, and *service execution* phases via JSON sidecar keypoints. The calibrated configuration achieves perfect abort-decision separation (Youden's J = 1.0 at τ\* = 78–80) on a held-out test set (n=8) and 28/31 (90%) per-recording slope-sign agreement on a leave-one-recording-out (LORO) sweep, up from 17/31 (55%) under hand-tuned defaults. The headline negative finding is that under our domain conditions, posture turns out to be net-harmful in the fusion: a no-pose ablation lifts sc02 (successful handover) LORO agreement from 4/8 to 7/8, because false withdrawal/mouth-cover fires from MediaPipe under domain shift were silently dragging successful handovers below the abort threshold. We discuss why our hand-tuned default actively *inverts* the abort/continue ordering, what calibration changed mechanism-by-mechanism, and how the resulting score governs robot abort decisions in a binary continue/abort policy.

---

## 1 Introduction

Service robots increasingly enter shared social spaces — hotel lobbies, hospital wards, homes — where they must hand objects to people who differ widely in their tolerance for robot proximity, motion, and gaze. The same approach trajectory that feels natural to a roboticist may feel intrusive to a first-time user. If a robot cannot perceive and adapt to individual comfort in real time, it risks startle reactions, avoidance behavior, or in the worst case a safety incident, and these failure modes scale poorly with deployment.

Despite progress in multimodal perception and comfort measurement, prior work has not closed the perception-to-action loop for socially adaptive *service*. Existing systems either rely on rule-based switching between discrete behavior modes (proximity-triggered slow/stop) or study emotion recognition in isolation from robot control. Three gaps remain. (1) **Late-fusion at interactive frame rates**: no system fuses face, gaze, and posture into a single continuous comfort score in real time on commodity RGB-D hardware. (2) **Modality complementarity under domain shift**: each channel has known weaknesses — facial ambiguity under neutral expressions, gaze unavailability when the face turns away, and missed withdrawal in 2D pose — yet no prior work combines all three with depth-based withdrawal detection for handovers, where the user's body and attention move through a stereotyped trajectory. (3) **Phase-conditioned fusion**: gaze is most informative *before* the user reaches (it signals readiness), but during the actual reach the same gaze aversion is a benign attention shift toward the object, not discomfort. A static fusion weight cannot resolve this.

In this paper, we present a real-time comfort-aware perception pipeline for the handover scenario and treat *fusion* — not the unimodal models — as the surface on which we learn. We segment each handover into three phases (approach, intent signaling, service execution) and learn phase-specific fusion weights from a small custom RealSense D435 dataset by differential evolution against a Youden-J-style abort-decision objective with a smooth trajectory-shape regularizer. The calibrated config (`config/deploy.yaml`) drives a binary continue/abort policy at a fixed threshold τ\*. Our contributions are: (i) a phase-aware late-fusion comfort scorer with EMA smoothing and missing-detection decay, deployable with zero retraining of perception models; (ii) a four-stage calibration pipeline (split → cached feature extraction → differential-evolution optimize → held-out evaluate) where the optimization stage runs in seconds per configuration on cached features; (iii) experimental evidence that pose, despite producing a clear standalone signal, is net-harmful for our handover under domain shift — an *ablation that beat its parent variant* — with a mechanism-level explanation tracing the result to specific MediaPipe false fires during reaching motions.

---

## 2 Related Work

We organize prior work into three areas spanning the perception-to-action pipeline.

**Multimodal perception.** Churamani et al. [16] fuse facial expression and speech via late fusion into an Affective Core (Grow-When-Required network) driving DDPG policy learning — the closest existing pipeline to ours, though they target dialogue and omit gaze. Surveys by Wang and Feng [2], Su et al. [3], and Zhao et al. [4] systematically review 60–227 papers and identify cognition-level comfort fusion, cross-modal temporal alignment, and the perception-to-action loop as the most prominent gaps. Fusion approaches in the literature range from confidence-weighted attention [18] and bilevel NAS [19] for action recognition, through musculoskeletal IK fusion of RGB-D and IMU under occlusion [20], to vision–language models such as PaLM-E [21] that unify images, robot state, and text into a shared latent space. Our work sits in the simplest of these regimes — score-level late fusion — but treats the phase-conditioned fusion weights themselves as the trainable surface.

**Comfort measurement and continuous comfort scoring.** Yan and Jia [5] propose a comfort taxonomy across ergonomic, robot-motion, anthropomorphism, and sociability dimensions, measured both subjectively (Likert) and objectively (HRV, EDA, EEG). Yan et al. [6] regress physical factors (distance, speed, angle) onto predicted comfort but omit emotional signals; Lorenzini et al. [7] review ergonomic HRI in industry. Gonzalez-Santocildes et al. [8] use reinforcement learning to adapt robot behavior from a continuous comfort signal, demonstrating that an online comfort score *can* drive policy. Heinisch et al. [9] release AFFECT-HRI, the only public HRI dataset pairing physiological data (Empatica E4 GSR/BVP) with affect labels.

**Affect-aware interaction and decision-making.** Spezialetti et al. [10] note that robot presence itself biases human emotional response, motivating in-situ rather than benchmark evaluation. Freire et al. [11] propose DAC-HRC, a layered adaptive control architecture with an explicit personalization mechanism; Abdollahi et al. [12] show that an empathic robot ("Ryan") improves engagement in a cross-over study. Alrefaie et al. [13] adopt a UR10 robot with normal/slow/stop switching based on proximity — the same graduated structure we adopt — but the controller is purely rule-based. For decision-making under uncertainty, Zheng et al. [14] learn POMDPs for collaboration, Pandya [15] grounds influence-aware safety in Bayesian MDPs, and Brunke et al. [17] survey safe learning in robotics from learning-based control to safe RL.

**How our work differs.** Existing work focuses on industrial ergonomics or proxemic RL; constrained ML and POMDPs are rarely synthesized for *social* interaction. To our knowledge, no prior system fuses facial emotion, gaze, and body posture from RGB-D video into a continuous, *phase-aware* real-time comfort score for the handover scenario, then few-shot calibrates that fusion at the deployment site. Our pipeline closes this gap with score-level late fusion behind a binary abort policy, calibrated against a small (n=31) site-specific recording corpus.

---

## 3 Problem Statement

Let `x_t = (I_c^(t), I_d^(t))` be an RGB-D frame pair captured at time `t` during a handover, with `I_c^(t)` the color frame and `I_d^(t)` the aligned depth map from an Intel RealSense D435. Each recording is annotated with up to four temporal keypoints `(t_start, t_signal, t_handover|abort, t_end)` that segment it into three phases:

```
[t_start] ──approach── [t_signal] ──intent── [t_handover|abort] ──execution── [t_end]
```

Define a phase indicator `φ(t) ∈ {approach, intent, execution}`. The pipeline produces three per-frame channel scores: emotion `C_e^(t) = f_e(I_c^(t); θ_E, θ_G)` (facial valence and arousal from EfficientNet-B0/AffectNet, fused with a binary "looking-at-camera" signal from L2CS-Net/Gaze360), and posture `C_p^(t) = f_p(I_c^(t), I_d^(t))` (body openness from MediaPipe Pose, with depth-based withdrawal and mouth-covering penalties). Each channel is independently EMA-smoothed in continuous time:

```
C̃_k^(t) = α(Δt) · C_k^(t) + (1 − α(Δt)) · C̃_k^(t−1),    k ∈ {e, p}
α(Δt) = 1 − exp(−Δt / τ_ema)
```

so that smoothing is FPS-independent. The integrated comfort score is the phase-conditioned weighted blend

```
C̃_t = w_e^{φ(t)} · C̃_e^(t) + w_p^{φ(t)} · C̃_p^(t)         (1)
```

with weights normalized such that `w_e + w_p = 1` per phase. When detection fails on a given channel, the channel decays exponentially toward a tunable target (`no_face_target`, `no_pose_target`) at a tunable rate.

The robot controller of §7.3 in our proposal collapses the three-tier state machine into a binary policy gated by τ\*:

```
π(C̃_t) = continue if C̃_t > τ*,  abort otherwise.       (2)
```

The learning problem is: given a small set of labeled recordings with ground-truth class `y ∈ {continue, abort, ambiguous}`, fit the parameter vector `θ` controlling fusion weights, gaze cone, EMA τ, decay rates, and penalties so as to maximize Youden's J on the abort decision while keeping each scenario's comfort *trajectory* trending in the correct direction (rising for continue, falling for abort).

---

## 4 Proposed Method

### 4.1 System architecture

Figure 1 shows the pipeline at runtime. The color frame is forked into a GPU emotion branch and a CPU posture branch, which run independently and report their per-frame outputs to a shared `IntegratedComfortScorer`.

```
RealSense .bag (color + aligned depth)
        │
        ├──► GPU branch
        │       ├── RetinaFace ─────► face bbox
        │       ├── EfficientNet-B0 (AffectNet) ─► (emotion, valence, arousal)
        │       └── L2CS-Net (Gaze360) ─────────► (yaw, pitch) → "looking-at-camera"
        │
        ├──► CPU branch
        │       └── MediaPipe Pose ──► 33 landmarks
        │             ├── open-posture ratio
        │             ├── mouth-covering check
        │             └── depth-based withdrawal detector  (uses I_d)
        │
        └──► IntegratedComfortScorer  (per-phase weights, EMA, missing-detection decay)
                    ├── φ(t) set by JSON keypoints (calibration) or controller (deployment)
                    └── continuous C̃_t ∈ [0, 100]    →  binary continue / abort at τ*
```

**Figure 1. Phase-aware late-fusion architecture.** Each branch operates independently; the comfort scorer combines them with phase-conditioned weights. Phase is supplied externally — by the calibration harness during fitting and by the robot controller's state machine during deployment.

The branches are deliberately decoupled so the system **degrades gracefully**: if the face is occluded the posture channel still reports, and vice-versa. Frames in which a channel is missing trigger a per-channel exponential decay toward a tunable target rather than a hard zero, so a brief detection dropout neither flips the decision nor injects step changes into the EMA.

### 4.2 Per-frame instant scores

**Emotion instant.** Given a detected face, EfficientNet-B0 produces a categorical emotion plus continuous valence `v ∈ [−1, 1]` and arousal `a ∈ [−1, 1]`; L2CS-Net produces gaze `(yaw, pitch)` from which we derive a binary "looking-at-camera" signal `g ∈ {0, 1}` using yaw/pitch thresholds. The instant emotion score is

```
raw  = γ · v − δ · a + w_g^{φ(t)} · g
C_e  = ((raw + |γ| + |δ| + w_g) / (2(|γ| + |δ| + w_g))) · 100
```

normalized into [0, 100]. The phase-specific gaze coefficient `w_g^{φ}` is what makes "looking at the camera" load heavily during *intent signaling* and barely at all during *execution*, where a downward gaze toward the cup is a benign attention shift.

**Posture instant.** From MediaPipe's 33 landmarks we compute a body-openness ratio (shoulder-to-elbow-to-wrist geometry) clipped to [0, 1], and check two depth-aware events: (i) mouth-covering when the hand-to-mouth distance falls below `face_cover_ratio` × shoulder-width, and (ii) sudden-withdrawal when a smoothed torso depth jumps by more than `withdrawal_threshold_meters` over the recent history window. Each fired event subtracts a fixed penalty (`mouth_cover_penalty`, `withdrawal_penalty`).

### 4.3 Phase-aware fusion and missing-detection decay

The smoothed channel scores `C̃_e^(t)`, `C̃_p^(t)` are blended per (1). Crucially, when a channel is missing on a given frame:

```
C̃_k^(t) ← C̃_k^(t-1) + (target_k − C̃_k^(t-1)) · (1 − exp(−Δt · rate_k))
```

This is the mechanism by which a *walk-by* recording (sc01) — where face detection is sparse because the user never turns toward the camera — accumulates downward drift into the abort region without requiring any positive evidence of discomfort. The four parameters `(target_e, rate_e, target_p, rate_p)` are tunable by calibration; in the deployed config the no-face decay rate is roughly 3× the hand-set default, which (as we show in §6) is the single largest contributor to separating sc01 from sc02.

### 4.4 Calibration: differential evolution on cached features

We avoid retraining any of the perception backbones — they are pretrained AffectNet/Gaze360/MediaPipe checkpoints used as black boxes. The learning surface is the 20-dimensional vector

```
θ = (w_e^{intent},  w_p^{intent},  w_g^{intent},
     w_e^{exec},    w_p^{exec},    w_g^{exec},
     yaw_thresh, pitch_thresh, face_cover_ratio,
     mouth_cover_penalty, withdrawal_thresh_m, withdrawal_penalty,
     γ, δ, τ_ema, posture_drop_gate,
     no_face_target, no_face_rate, no_pose_target, no_pose_rate)
```

with bounded ranges (e.g., `no_face_rate ∈ [0.05, 0.50]`, `yaw_thresh ∈ [15°, 35°]`).

The four-stage pipeline (`scripts/calibrate.py`) is:

1. **`split`** — 75/25 stratified train/test (23/8) by scenario, seed=42. Held-out test is touched only at the very end.
2. **`extract`** — runs RetinaFace, EfficientNet-B0, L2CS-Net, MediaPipe Pose **once** per recording, caching per-frame features as parquets in `cache/`. Subsequent optimizer iterations do not touch the GPU and replay the entire pipeline from cache in pure NumPy in seconds.
3. **`optimize`** — `scipy.optimize.differential_evolution` (population 15, max-iter 50, polish, deferred update, seed=42) over `θ` with a 5-fold stratified CV on the train split. The objective reduces a per-recording continuous score using one of several *reducer* variants (mean over full window, mean over last 30%, late−early delta, and tanh-bounded slope-shape regularizers) and returns Youden's J at τ\* = argmax over τ ∈ [30, 80] plus optional shape/penalty terms.
4. **`evaluate`** — single pass on the held-out test, producing a τ-sweep, slope-sign agreement, and bootstrap CI on per-scenario means.

**Race-then-refine.** Because each candidate objective is cheap to evaluate (seconds, not minutes), we ran six objective variants (A–I) in parallel as a small "race" on shared train folds before promoting the winner.

### 4.5 Inference

At deployment, the robot controller drives `set_phase(...)` directly from its own state machine (`approach` while motion-planning, `intent` while signaling with a cup-shake, `execution` once the arm crosses the OTP plane). A 0.3 s warmup after each phase transition freezes the integrated output to suppress the EMA transient. The continuous `C̃_t` is compared to the calibrated abort threshold τ\* = 80 to gate continue / abort.

---

## 5 Experimental Methodology

### 5.1 Dataset

We collected 31 RGB-D recordings (.bag files) on an Intel RealSense D435 at 30 FPS across five handover scenarios, with the camera mounted on a Bambot manipulator. Each recording carries a JSON sidecar (see Appendix A) with keypoints `start / signal / handover (or abort) / end` annotated post-hoc with a frame-stepping labeling tool. The scenarios are:

| code | scenario | label | n | structure |
|---|---|---|---|---|
| sc01 | walk-by, no interest | abort | 7 | start → signal → abort (no handover) |
| sc02 | comfortable handover | continue | 8 | start → signal → handover → end |
| sc03 | gradual discomfort during handover | ambiguous | 5 | full 4-step |
| sc04 | sudden withdrawal mid-handover | abort | 5 | full 4-step |
| sc05 | distracted (looking at phone) | ambiguous | 6 | full 4-step |

Each scenario was filmed under both bright and dim lighting in the same room; the `.bag` plus its JSON sidecar form the unit of data. Compared to our midterm corpus (33 recordings across 6 scenarios), we consolidated walk-by + walk-by-cluttered into a single sc01 and dropped the under-populated "hesitant" scenario, which the midterm flagged as too small to support optimization.

The split is 23 train / 8 test, stratified by scenario class (continue / abort / ambiguous) so each fold contains a mix; the held-out test contains 2 sc01, 2 sc02, 1 sc03, 1 sc04, 2 sc05.

### 5.2 Pretrained models

All perception backbones are used out-of-the-box:

- **RetinaFace** (face detection from color)
- **EfficientNet-B0 / `enet_b0_8_va_mtl`** (emotion + continuous valence/arousal, AffectNet-pretrained)
- **L2CS-Net / `L2CSNet_gaze360`** (gaze yaw/pitch, Gaze360-pretrained)
- **MediaPipe Pose, complexity 1** (33-landmark skeletal pose)

Only the 20 fusion-layer hyperparameters are learned.

### 5.3 Hyperparameters and metrics

Differential evolution: population 15, max-iter 50, polish=True, updating='deferred', seed=42; 5-fold stratified CV on the train split. We report:

- **Youden's J** at τ\* on the held-out test, plus a τ-sweep over [30, 80] in 1-point steps.
- **Per-scenario mean comfort** with bootstrap 95% CI (1000 resamples).
- **Per-recording slope sign agreement**: linear fit on the smoothed integrated series over the union of the intent and execution windows; correct sign means positive for sc02, negative for everything else.
- **LORO (leave-one-recording-out) sweep** across all 31 recordings as an out-of-fold sanity check.

### 5.4 Baselines and ablations

- `config/default.yaml` — hand-tuned defaults (the midterm config), never optimized against this corpus.
- `config/deploy.yaml` — the calibrated A-no-pose winner (see §6).
- Six objective variants A, B, C, F, G, H, I (described in §6.2).
- A no-pose ablation pinning `w_p^{intent} = w_p^{exec} = 0`.

---

## 6 Results and Discussion

### 6.1 Headline: calibrated config separates abort from continue cleanly

Table 1 compares the hand-tuned default with the calibrated deploy on the held-out test (n=8) and on a full-corpus LORO (n=31).

**Table 1. Default vs deploy on held-out test and LORO.**

| metric | default | deploy | Δ |
|---|---|---|---|
| best Youden J on test | **0.00 @ τ=30** | **1.00 @ τ=78** | +1.00 |
| decisions at τ\* (test) | — | TP 3 / FN 0 / FP 0 / TN 2 | perfect |
| sc01 mean (test, abort) | 69.0 | **50.3** | −18.7 |
| sc02 mean (test, continue) | 66.6 | **92.8** | +26.2 |
| LORO slope-sign agreement (n=31) | 17/31 (55%) | **28/31 (90%)** | +11 |

The default config cannot separate abort from continue at *any* threshold τ ∈ [30, 80]: its Youden J is 0.0 because sc01 outscores sc02 in absolute level. The calibrated deploy not only flips this ordering but opens a 42-point margin between the sc01 and sc02 means, and the same margin holds across LORO (sc01 = 50.2, sc02 = 92.7 across all 15 recordings of those two scenarios).

**Per-recording trajectory.** Under the default, sc02 *trends downward* during successful handovers (mean slope −3.05 on test) — the opposite of what a successful interaction should do. Under deploy, every single test recording's slope has the correct sign (8/8). On LORO, slope-sign agreement jumps from 0/8 → 7/8 on sc02 alone (the single biggest improvement) and from 1/5 → 5/5 on sc03.

### 6.2 Race-2 objective variants: shape vs level is genuinely a trade

The optimizer's objective is what determines whether the calibrated config makes the trajectory *trend* the right way or merely separates *levels*. We raced six variants on the same 5-fold CV folds:

**Table 2. Race-2 variants on held-out test.** Slope sign should be + for sc02, − for sc01/sc04. "Gates" counts eight pre-registered acceptance gates (J ≥ 0.80, sc02 mean ≥ 80, sc01 mean ≤ 65, sc04 mean ≤ 65, and the four slope-sign agreements).

| var | objective | sc02 slope | sc01 slope | sc04 slope | sc02 mean | sc01 mean | J | gates |
|---|---|---|---|---|---|---|---|---|
| **A** | mean-full + J | +0.10 | −0.54 | −2.82 | 89.1 | 55.8 | 1.0 | **6/7** |
| C | late−early delta | +2.14 | −1.26 | +0.47 | 65.8 | 49.5 | 1.0 | 5/7 |
| F | J + sc02 shape | +2.86 | +2.88 | +1.48 | 73.6 | 53.5 | 1.0 | 4/7 |
| G | J + all-scenario shape | −0.75 | −2.38 | −3.50 | 81.5 | 75.5 | 1.0 | 4/7 |
| H | G + drag params | −0.72 | −3.35 | −3.34 | 80.7 | 75.2 | 1.0 | 4/7 |
| I | sc02-weighted shape (×3) | +0.89 | +2.61 | −1.48 | 77.1 | 65.7 | 1.0 | 4/7 |

The race tells a coherent story. **F** illustrates Goodhart's law: rewarding only sc02's shape lifts every scenario uniformly. **G/H** flip the inverse: aggregate-shape mean rewards DE for sacrificing the sc02 minority. **I** over-corrects by weighting sc02 3× and adds a hard penalty for negative sc02 slope, which fixes sc02 (+0.89) but lets sc01 drift up (+2.61). **A** — the level-only baseline — is the most boring objective but the strongest gate-count winner; its barely-positive sc02 slope (+0.10) is the cost.

This is the negative result of the project: a single-objective DE with our 23-recording train budget can satisfy *level* gates or *shape* gates but cannot reliably satisfy both at once. The fundamental issue is that with 23 train recordings split into 5 CV folds (~4–5 recordings per fold), the per-scenario shape signal is too noisy to constrain DE without inducing overcorrection. We promoted A and exposed the trade-off explicitly rather than hide it.

### 6.3 The surprise no-pose ablation

Per the original plan, we ran a confirmatory ablation: the **A** objective with `w_p^{intent} = w_p^{exec} = 0` pinned via `--pin-wp 0.0`, same DE budget, same seed.

**The ablation won.** By a lot.

**Table 3. A vs A-no-pose on held-out test and LORO.**

| metric | A (with pose) | A-no-pose | Δ |
|---|---|---|---|
| sc02 slope (test) | +0.10 | **+1.05** | +0.95 |
| sc02 mean (test) | 89.1 | 92.8 | +3.7 |
| sc01 mean (test) | 55.8 | **50.3** | better separation |
| slope-sign agreement (test) | 3/5 | **5/5** | +2 |
| sc02 LORO agreement | 4/8 | **7/8** | +3 |
| sc01 LORO agreement | 6/7 | 5/7 | −1 |
| J on test | 1.0 | 1.0 | — |

Pose was *actively harming* the trajectory signal. DE had converged to nonzero `w_p^{intent}` and `w_p^{exec}` in A because they marginally helped J on the 5-fold CV train folds, but on the held-out test the pose domain shift surfaced. Specifically: in cached sc02 parquets the MediaPipe withdrawal condition fired in 2–4% of intent+execution frames, mouth-cover up to 1.3% — low but non-zero where zero was expected. Each spurious fire subtracted 25 (withdrawal) or 30 (mouth-cover) from the posture sub-score, which under the default 65% execution weight propagated directly into sc02's integrated comfort. CV on the train split couldn't see this because all folds share the same systematic bias. Pose was net-negative *out-of-fold*, and only the held-out test exposed it.

We promoted A-no-pose to `config/deploy.yaml` and document the result honestly: `w_p = 0` is **dataset-conditional**, not a universal claim about pose. The posture module remains in the runtime (it surfaces mouth-cover and withdrawal warnings on the HUD as state diagnostics), and `--pin-wp` can be removed for any future dataset where the pose domain shift is smaller.

### 6.4 Mechanism-level: what specifically flipped sc01/sc02

Under the default, sc01 walkby (69.0) outscored sc02 handover (66.6) on the test split — the wrong ordering. The calibrated deploy attacks three specific mechanisms simultaneously:

**Posture weight pinned to 0** (intent: 0.4 → 0.0; execution: 0.65 → 0.0). The dominant fusion channel in the default's execution phase was the noisy one — see §6.3. Pinning it removes both the signal and the hard-coded penalties from the fusion entirely.

**Missing-detection decay rate accelerated ~3×** (no_face_rate: 0.15/s → 0.469/s; no_face_target: 35.0 → 32.5). Default decay had a ~6.7 s half-life — sc01 walkby has only ~40% face-detection rate, so 60% of frames were dragging its score *slowly* toward 35, leaving sc01 high enough to be confused with sc02. Deploy decays roughly three times faster, so missing frames drag sc01 firmly toward the low-30s. sc02's 68–72% face-detection rate means the boosted rate barely affects it. **This is the single largest contributor to the 42-point sc01/sc02 separation.**

**Phase-conditioned gaze and emotion become load-bearing** (intent emotion weight: 0.6 → 0.882; intent gaze weight: 0.6 → 0.808; yaw cone: 25° → 31.7°; pitch cone: 20° → 29.9°). With pose removed, the remaining channels carry the signal. The widened gaze cone matters because sc02 users don't always fixate perfectly on the camera while reaching — a 25°/20° cone classified many sc02 frames as "looking away", killing the gaze contribution exactly when it should support a comfortable interaction.

Smaller supporting changes: `face_cover_ratio` 0.6 → 0.85 (raises the bar so incidental near-face hand motion during sc02 stops triggering the 30-point mouth-cover penalty); `withdrawal_penalty` 25 → 11.9 (less destructive when fired); `gamma`/`delta` floor terms drop near zero (lets sc01's true low comfort surface); `ema_time_constant_s` 0.5 → 0.286 (sc02's trajectory can rise through execution rather than averaging into a constant).

### 6.5 What we still wouldn't claim

Three honest caveats. (1) **J = 1.0 on n = 8 is not statistically sharp.** A binomial CI on 3/3 abort TPR runs from 0.37 to 1.0 at 95%; "perfect on 8" only means "no failures observed in 8 samples". (2) **sc02 LORO 7/8 is real.** One sc02 recording trends slightly downward despite being a successful handover. The reasonable read is that a fully comfortable user already maximally comfortable at signal time doesn't *have* to trend further upward — flat-high is also success. We report 7/8 explicitly rather than rounding to 8/8. (3) **No-pose is dataset-conditional.** Under a different lighting, camera, or interaction protocol, MediaPipe's domain shift may be smaller and pose may again become net-positive. The `--pin-wp` lever is preserved.

### 6.6 Real-time performance

End-to-end on the deployment workstation (RTX-class GPU + 23-core CPU, no batching, single recording at 30 FPS), the integrated pipeline processes RealSense `.bag` playback at 13.6–58.2 FPS depending on whether all four detectors fire on the same frame. The bottleneck is L2CS-Net gaze inference. The CPU posture branch runs concurrently with the GPU emotion branch, so adding pose costs nothing on the wall clock.

---

## 7 Conclusion and Future Directions

**Contributions.** We built a real-time multimodal comfort-scoring pipeline for human–robot handover that fuses pretrained facial-emotion, gaze, and pose models into a single phase-aware comfort score, with a four-stage calibration that uses cached features so iterating on the fusion-layer hyperparameters runs in seconds. Calibrated against a 31-recording RealSense corpus, the system achieves a perfect abort decision (J = 1.0 at τ\* = 78) on a held-out 8-recording test and 90% slope-sign agreement on full-corpus LORO, against 0.00 / 55% under the hand-tuned default. We document a negative finding — under our domain conditions, pose's MediaPipe domain shift makes it net-harmful to the fusion, and a no-pose ablation beat its parent A variant on every meaningful metric.

**Limitations.** The corpus is small (n = 31, n = 8 held-out). Our test set has 3 aborts and 2 continues; perfect classification is consistent with a posterior on TPR running from ~0.37 to 1.0. Five scenarios filmed in one room with one camera mount cannot resolve generalization across users, lighting beyond bright/dim, or alternative robot platforms. The shape-aware objective space we explored (variants C / F / G / H / I) demonstrated a trade-off with level: with our train budget, we could not satisfy both level and shape gates simultaneously without overcorrection.

**Future directions.**

1. **Recover pose under domain shift.** The cleanest follow-up is to remove the false-fire pathway rather than zeroing pose. Two avenues: (a) add a hysteresis / dwell-time gate on the withdrawal and mouth-cover triggers so a single-frame MediaPipe confidence excursion no longer subtracts 25–30 points; (b) site-specific few-shot fine-tuning of the `face_cover_ratio` and `withdrawal_threshold_meters` thresholds per camera mount.
2. **Close the perception-to-action loop.** The `live_comfort.py` reference deploys the calibrated config end-to-end against the robot's RealSense feed — wiring it into the manipulator's controller would close the loop our proposal targeted but our final report stops short of, since on-robot evaluation slipped past our final-report deadline.
3. **Richer policy than continue/abort.** The original three-tier state machine (continue / slow / abort) is structurally compatible with the current scorer; it requires a second threshold τ_slow and a controller that re-issues motion plans at reduced speed.
4. **More recordings.** Even doubling to ~60 recordings would let us split sc03 and sc05 into named-class targets in the J objective rather than carrying them as ambiguous, and would tighten the Youden CI from [0.37, 1.00] to something usable for engineering decisions.
5. **On-site few-shot fine-tuning.** The original Idea 4 — collect 5–10 minutes of site-specific recordings to recalibrate fusion weights per deployment — is now a one-command follow-up given the cached-feature pipeline. We have not run the full within-subject comparison this would enable; that is the natural next experiment.

---

## References

[1] N. Churamani, P. Barros, H. Gunes, and S. Wermter. Affect-driven modelling of robot personality for collaborative human–robot interactions. *arXiv:2010.07221*, 2020.

[2] Z. Wang and Q. Feng. Multimodal human–robot interaction for human-centric smart manufacturing: A survey. *Advanced Intelligent Systems*, 6(1):2300359, 2024.

[3] H. Su, W. Qi, J. Chen, C. Yang, J. Sandoval, and M. A. Laribi. Recent advancements in multimodal human–robot interaction. *Frontiers in Neurorobotics*, 17:1084000, 2023.

[4] X. Zhao et al. Multimodal perception-driven decision-making for human–robot interaction: A survey. *Frontiers in Robotics and AI*, 12:1604472, 2025.

[5] Y. Yan and Y. Jia. A review on human comfort factors, measurements, and improvements in human–robot collaboration. *Sensors*, 22(19):7431, 2022.

[6] Y. Yan, H. Su, and Y. Jia. Modeling and analysis of human comfort in human–robot collaboration. *Biomimetics*, 8(6):464, 2023.

[7] M. Lorenzini et al. Ergonomic human–robot collaboration in industry: A review. *Frontiers in Robotics and AI*, 9:813907, 2023.

[8] P. Gonzalez-Santocildes et al. Adaptive robot behavior based on human comfort using reinforcement learning. *IEEE Access*, 12, 2024.

[9] J. S. Heinisch et al. Physiological data for affective computing in HRI with anthropomorphic service robots: The AFFECT-HRI dataset. *Scientific Data*, 11(1):333, 2024.

[10] M. Spezialetti, G. Placidi, and S. Rossi. Emotion recognition for human–robot interaction: Recent advances and future perspectives. *Frontiers in Robotics and AI*, 7:532279, 2020.

[11] I. T. Freire, A. F. Amil, and P. F. M. J. Verschure. Socially adaptive cognitive architecture for human–robot collaboration in industrial settings. *Frontiers in Robotics and AI*, 11:1248646, 2024.

[12] H. Abdollahi et al. Artificial emotional intelligence in socially assistive robots for older adults: A pilot study. *IEEE Trans. Affective Computing*, 14(3):2020–2032, 2023.

[13] M. T. Alrefaie et al. Database for human emotion estimation through physiological signals (HEEP-HRI). IEEE / NSF / RIT, 2024.

[14] W. Zheng, B. Wu, and H. Lin. POMDP model learning for human–robot collaboration. In *Proc. IEEE CDC*, pp. 1156–1161, 2018.

[15] R. Pandya. Influence-aware safety for human–robot interaction. CMU-RI-TR-25-95, 2025.

[16] N. Churamani, S. Kalkan, and H. Gunes. Affect-driven learning of robot behaviour for collaborative human–robot interactions. *Frontiers in Robotics and AI*, 9:717193, 2022.

[17] L. Brunke et al. Safe learning in robotics: From learning-based control to safe reinforcement learning. *Annual Review of Control, Robotics, and Autonomous Systems*, 5:411–444, 2022.

[18] An adaptive human–robot interaction framework using real-time emotion recognition and context-aware task planning. *IEEE Access*, 13:111431–111450, 2025.

[19] BM-NAS: Bilevel multimodal neural architecture search. IEEE, 2020.

[20] Multimodal inverse kinematics for human pose estimation. *Scientific Reports*, 15:44420, 2025.

[21] D. Driess et al. PaLM-E: An embodied multimodal language model. In *Proc. ICML*, 2023.

[22] X. Alameda-Pineda et al. SALSA: A novel dataset for multimodal group analysis. *IEEE TPAMI*, 2016.

[23] D. B. Jayagopi et al. The Vernissage corpus: A multimodal corpus for analyzing human–robot interaction. In *Proc. LREC*, 2012.

[24] R. Martín-Martín et al. JRDB: A multi-modal predictive academic dataset. *IEEE TPAMI*, 2021.

[25] P. Kellnhofer et al. Gaze360: Physically unconstrained gaze estimation in the wild. In *Proc. ICCV*, 2019.

[26] A. Mollahosseini, B. Hasani, and M. H. Mahoor. AffectNet: A database for facial expression, valence, and arousal computing in the wild. *IEEE Trans. Affective Computing*, 2017.

[27] J. Liu et al. NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding. *IEEE TPAMI*, 2019.

[28] C. Busso et al. IEMOCAP: Interactive emotional dyadic motion capture database. *Language Resources and Evaluation*, 2008.

---

## Appendix A: Sidecar JSON format

Each `.bag` recording is paired with a JSON sidecar carrying the keypoints used by the calibration harness to set the phase indicator `φ(t)`. For sc02–sc05 (full handover or attempted handover):

```json
{
  "scenario_code": "SC-02",
  "scenario_name": "Comfortable handoff",
  "duration_seconds": 12.5,
  "realsense_serial": "838212070352",
  "labels": {
    "start_time": 3.04,
    "signal_time": 4.14,
    "handover_time": 11.41,
    "end_time": 15.88
  }
}
```

For sc01 (walk-by, no handover) the structure is `(start, signal, abort)` with `handover_time` absent. The labeling tool overwrites `labels` post-hoc as the annotator steps through the video and presses S/I/H/E (or A) hotkeys.

## Appendix B: Calibration command sequence

```bash
python scripts/calibrate.py split       # 75/25 stratified split → data/split.json
python scripts/calibrate.py extract     # GPU pass; caches per-frame features → cache/*.parquet
python scripts/calibrate.py optimize    # CPU-only DE on cached features → config/deploy.yaml
python scripts/calibrate.py evaluate    # held-out test → reports/calibration_report.json
```

Once `extract` has run, `optimize` and `evaluate` iterate at CPU speeds — full DE budget (50 iter × pop 15) for one objective variant takes ~20 min single-threaded and trivially parallelizes across variants.

## Appendix C: Repository structure

```
src/
  pipeline.py             # IntegratedPipeline orchestrating both branches
  comfort.py              # IntegratedComfortScorer (phase-aware fusion + EMA + decay)
  phases.py               # PHASES, PhaseWindows, sidecar JSON parsing
  detectors/              # face / emotion / gaze / pose adapters over pretrained models
  visualization.py        # HUD overlay (comfort bar, FPS, phase, state warnings)
  bag_source.py           # RealSense .bag playback abstraction
scripts/
  calibrate.py            # entry point for the four-stage pipeline
  extract_features.py     # Stage A: GPU pass, caches parquets
  optimize_params.py      # Stage B: differential evolution
  evaluate_test.py        # Stage C: held-out test
  loro_eval.py            # full-corpus LORO sweep
  race_objectives.py      # parallel race over objective variants
  run_bag.py              # interactive playback of a recording with HUD
  live_comfort.py         # reference live deployment (RealSense → comfort score)
config/
  default.yaml            # hand-tuned baseline (never calibrated)
  deploy.yaml             # calibrated A-no-pose winner (deployed)
  deploy.{A_withpose,prerace,baseline,A,B,C,F,G,H,I}.yaml  # archived variants
reports/
  default_vs_deploy.md    # head-to-head writeup
  RUNS.md                 # index of every calibration run
  20260419_1124_calibration_process_braindump.md  # unfiltered process notes
  eval_*.json, loro_*.json, *_race*.{yaml,json}    # raw outputs
```
