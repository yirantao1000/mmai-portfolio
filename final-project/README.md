# Integrated Emotion + Posture Detection

Real-time comfort-aware perception system for human-robot handover scenarios. Combines GPU-based emotion/gaze detection with CPU-based posture analysis on Intel RealSense D435 `.bag` recordings, producing a unified comfort score.

## Architecture

```
RealSense .bag (color + depth)
        │
        ├──► Face Detection (RetinaFace)
        │       ├──► Emotion Detection (EfficientNet-B0, AffectNet)
        │       └──► Gaze Estimation (L2CS-Net, Gaze360)
        │
        ├──► Pose Detection (MediaPipe Pose)
        │       ├──► Open Posture Scoring
        │       ├──► Mouth Covering Detection
        │       └──► Depth-based Withdrawal Detection
        │
        └──► Integrated Comfort Scorer
                ├──► Phase-aware fusion (approach / intent / execution)
                ├──► Per-phase weights over emotion, gaze, posture channels
                ├──► Missing-detection decay toward tunable targets
                └──► Unified Score (0-100, EMA-smoothed) + abort threshold τ*
```

Fusion weights are phase-dependent and calibrated by differential evolution against held-out recordings (see [Calibration](#calibration)). The deployed config (`config/deploy.yaml`) sets `posture_weight = 0` in both intent and execution phases — see [reports/default_vs_deploy.md](reports/default_vs_deploy.md) for why.

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for emotion/gaze models)
- Intel RealSense SDK 2.0

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Models

Download pre-trained models using the provided script:

```bash
python scripts/download_models.py
```

Or symlink from an existing emotion detection repo:

```bash
ln -s /path/to/mmai-emotion-detection/models models
```

### Data

Place `.bag` files in `data/` organized by scenario and lighting:

```
data/
├── sc02_comfortable/
│   ├── Bright/
│   │   └── recording.bag
│   └── Dark/
│       └── recording.bag
├── sc04_sudden_withdrawal/
│   └── ...
```

Or symlink from an existing data directory:

```bash
ln -s /path/to/mmai-emotion-detection/data data
```

## Usage

### Interactive Playback

```bash
python scripts/run_bag.py
```

Uses `config/deploy.yaml` (the calibrated config) by default. Pass `--config config/default.yaml` to run against the uncalibrated baseline.

Select files interactively. Controls:
- `q` — quit
- `n` — skip to next file

### Process Specific File

```bash
python scripts/run_bag.py data/sc02_comfortable/Dark/recording.bag
```

### Save Annotated Videos

```bash
python scripts/run_bag.py --save
python scripts/run_bag.py --save --save-dir rendering_output/
```

### Headless Mode

```bash
python scripts/run_bag.py --headless
python scripts/run_bag.py --headless --save --save-dir rendering_output/
```

### Multiple File Selection

At the interactive prompt, enter:
- `3` — single file
- `1,4,7` — comma-separated
- `2-5` — range
- `0` — all files

## Visualization Layout

```
+--------------------------------------------------+
| [Comfort: 72/100]                      FPS: 28   |
|                                                   |
| happy (V:+0.45 A:-0.10)       Posture: 0.75      |
| Emotion Score: 68              Depth Z: 0.85m     |
| Gaze: Looking at camera       Posture Score: 82   |
|                                                   |
|        [face bbox]         [depth marker]         |
|                                                   |
|         STATE: SCARED (mouth/face covered)        |
| sc02/Bright  t=5.2s  frame=156                    |
+--------------------------------------------------+
```

## Configuration

Two configs ship with the repo:
- `config/default.yaml` — hand-set baseline, never calibrated.
- `config/deploy.yaml` — the calibrated config used at runtime. Produced by the calibration pipeline below.

Key sections (identical schema in both):

| Section | Description |
|---------|-------------|
| `face_detector` | RetinaFace confidence threshold, bbox expansion |
| `emotion_detector` | EfficientNet model, input size |
| `gaze_detector` | L2CS-Net yaw/pitch thresholds |
| `pose_detector` | MediaPipe settings, face-cover ratio, withdrawal threshold, posture-drop gate |
| `comfort` | Per-phase fusion weights, gamma/delta, EMA time constant, mouth-cover and withdrawal penalties, missing-detection decay targets, `abort_threshold` (τ*) |
| `visualization` | Toggle individual overlay elements |

## Comfort Scoring

**Emotion comfort** (0-100): Based on valence, arousal, and gaze engagement.

**Posture comfort** (0-100): Based on open posture score, with penalties for mouth covering and sudden withdrawal. Still computed; surfaces on the HUD as state warnings and the side-panel posture score.

**Integrated comfort**: Phase-weighted blend of emotion, gaze, and posture channels, with independent EMA smoothing per component. Weights are set per phase (`approach` / `intent` / `execution`) and calibrated against labeled recordings. Under the deployed config, posture weight is 0 in both intent and execution (see [reports/default_vs_deploy.md](reports/default_vs_deploy.md) for the rationale) — the abort decision runs on emotion + gaze, with missing-detection decay providing a baseline pull when the face or pose drops out.

## Calibration

Calibration is a four-stage pipeline orchestrated by `scripts/calibrate.py`:

```bash
python scripts/calibrate.py split      # 75/25 stratified train/test (data/split.json)
python scripts/calibrate.py extract    # runs vision models once, caches features
python scripts/calibrate.py optimize   # differential-evolution search → config/deploy.yaml
python scripts/calibrate.py evaluate   # held-out evaluation → reports/calibration_report.json
```

Once `extract` has produced cached parquets, `optimize` and `evaluate` run without the GPU — iterating on the objective function takes seconds per configuration, not minutes.

**Objective variants** (select via `scripts/optimize_params.py --objective`): `A` mean+J baseline, `B` mean-late, `C` delta, `F` J+sc02-shape, `G` J+all-scenario-shape, `I` sc02-guarded shape. `scripts/race_objectives.py` races a chosen set and picks a winner.

**Ablations:** `scripts/optimize_params.py --pin-wp 0.0` pins posture weights to zero (used to confirm pose was net-negative under the current domain shift).

**Full-dataset sanity check:** `scripts/loro_eval.py --config CONFIG --report reports/loro_X.json` replays all 31 recordings against a given config and reports per-scenario slope-sign agreement.

Every run archives to `reports/` with a timestamped filename; `reports/RUNS.md` indexes them.

## Reports and writeup material

- `reports/RUNS.md` — index of every calibration run with headline metrics and notes.
- `reports/default_vs_deploy.md` — side-by-side comparison of the uncalibrated default config against the deployed calibrated config on the held-out test and LORO.
- `reports/20260419_1124_calibration_process_braindump.md` — unfiltered narrative of the calibration process across race 1 and race 2, including what was tried, what failed, and why. Intended as source material for the write-up.

## Dependencies

- **GPU**: PyTorch, timm, EmotiEffLib, L2CS-Net
- **CPU**: MediaPipe Pose
- **Shared**: OpenCV, NumPy, pyrealsense2
