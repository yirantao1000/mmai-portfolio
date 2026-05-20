# V3 Distillation Pipeline

End-to-end pipeline that turns the existing multimodal comfort-scoring pipeline
into (a) a labelled dataset, (b) an independent VLM second opinion, and
(c) a 1.58M-param student model that can run on the robot's onboard compute.

For empirical results on the 10 held-out test bags, see
[`V3_TEST_REPORT.md`](V3_TEST_REPORT.md).
Rendered comparison videos + full VLM prompt are mirrored at
<https://huggingface.co/datasets/yirantao1000/mmai-comfort-handover-v3>.

---

## Architecture

```
.bag (RealSense color + depth)
    │
    ├──► annotate_heuristic.py ──► annotations/<scen>/<stem>__heuristic.parquet
    │      (existing multimodal pipeline; dense ~30 Hz per-frame 0-100 score)
    │
    ├──► annotate_vlm.py       ──► annotations/<scen>/<stem>__vlm.parquet
    │      (gpt-5.5; sees only raw frames + timestamps; 2 Hz output)
    │
    └──► annotate_human.py     ──► annotations/<scen>/<stem>__human-<rater>.parquet
           (optional, OpenCV-based scrubber + 1-5 keys; ~2 Hz output)

annotations/ + frame_cache/ ──► train_student.py ──► checkpoints_v2/<source>/best.pt
                                       │
                                       └──► eval_student.py ──► annotations/<scen>/<stem>__student.parquet

annotations/<scen>/*.parquet ──► render_annotations.py ──► renders/<...>.mp4
       (any subset of {heuristic, vlm, student, human}, H.264, 480px @ 15 fps)
```

All annotators write the **same parquet schema** (`src/annotations.py`) so any
downstream consumer can swap sources.

---

## Reproduce from scratch

```powershell
# 0) install
pip install -r requirements.txt

# 1) Deterministic per-scenario test split -> splits/v2.json
#    (2 bags per scenario; mixes 2026-04 and 2026-05 sessions for cross-session eval)
python scripts/make_v2_split.py --seed 42 --n-test-per-scenario 2

# 2) Heuristic teacher on every bag (skips bags already annotated)
python scripts/annotate_heuristic.py

# 3) JPEG frame cache for fast student training (skips frames already cached)
python scripts/prepare_frames.py --fps 15 --resize-max 320

# 4) Train the student on heuristic labels with the v2 split
python scripts/train_student.py \
    --source heuristic --split-file splits/v2.json \
    --max-train-fps 5 --epochs 10 --out-dir checkpoints_v2

# 5) VLM teacher on the 10 test bags only (cheap, ~$5)
$env:OPENAI_API_KEY = "sk-..."     # or put one line in .openai_key at project root
python scripts/annotate_vlm.py --split-file splits/v2.json --split-set test --sleep-between 0.3

# 6) Run the student on the test bags and write its parquets
python scripts/eval_student.py --checkpoint checkpoints_v2/heuristic/best.pt \
    --include-training-sample 0 --no-render --out-dir rendering_output_v2/student

# 7) Render the 3-way comparison videos (heuristic / VLM / student) on the test set
python scripts/render_annotations.py \
    --split-file splits/v2.json --split-set test \
    --sources heuristic vlm student --out-dir renders/v3_3way_test \
    --fps 15 --width 480

# 8) Compute pairwise agreement metrics
$env:SUMMARY_TAG = "v3"; python scripts/_v2_test_summary.py
```

---

## Honesty / no-leakage policy for the VLM

The VLM (`scripts/annotate_vlm.py`) sees **only raw frames and timestamps**.
It is never told:

- the scenario name (e.g. `sc02_comfortable`) or its description
- the expected outcome (abort vs continue)
- the interaction phase at any frame
- any sidecar event time (start / signal / handover / abort / end)
- anything else a human labelled offline

The `scenario` and `phase` columns in the output parquet are filled in **after**
inference, as analysis metadata only. The full prompt is bundled in the HF
dataset as `VLM_PROMPT.md`.

---

## Design tricks in the VLM annotator (4 of them)

| # | trick | implementation | purpose |
|---|---|---|---|
| 1 | sliding window with overlap | window=6, stride=3 | each frame gets averaged across 2 windows |
| 2 | ordinal soft distribution | model returns `p = [p1..p5]` summing to 1 | preserves uncertainty for KL distillation |
| 3 | per-window rationale | one sentence per call | spot-check + future rationale distillation |
| 4 | causal EMA smoothing | `tau = 0.6 s` after overlap-averaging | real-time-style output curve |

---

## Train / test split (`splits/v2.json`)

- Selected by `scripts/make_v2_split.py` with seed=42
- 47 train bags / 10 test bags / 0 throwaway (3 unlabelled new bags go to train)
- 2 test bags per scenario, mixing 2026-04 (5) and 2026-05 (5) sessions
- Test list is enumerated in the head of `splits/v2.json` under `all_test_stems`
- Both `train_student.py`, `annotate_vlm.py`, `render_annotations.py` accept
  `--split-file splits/v2.json --split-set {train,test}` to filter bags

---

## Parquet schema (`src/annotations.py`)

| column | type | notes |
|---|---|---|
| `timestamp_s` | float32 | seconds from bag start |
| `frame_idx` | int32 | bag-native frame index, consistent across sources |
| `comfort_score` | float32 | 0-100, EMA-smoothed (canonical column used by training + rendering) |
| `comfort_score_5` | float32 | 1-5 ordinal expected value (raw, no EMA); NaN for heuristic |
| `confidence` | float32 | `max(p)` for VLM, NaN elsewhere |
| `rationale` | string | per-window VLM sentence or per-frame human note |
| `source` | string | `heuristic` / `vlm` / `human` / `student` |
| `scenario`, `bag_stem` | string | identifiers |
| `phase` | string | `approach` / `intent` / `execution` / `""` — **metadata only, never fed to VLM** |

---

## Adding a new teacher

1. Write `scripts/annotate_<name>.py` that emits the schema above with
   `source = "<name>"`.
2. (optional) Add `<name>` to `train_student.py --source` choices and
   `render_annotations.py --sources` default list.

That's it — the rest of the pipeline (training, eval, rendering, metrics) is
source-agnostic.

## Adding a new student backbone

`src/student/model.py` factory is `build_default_model(backbone=...)`. Any
timm model works as a feature extractor; the head is a single linear that
regresses to 0-100. Pass `--backbone <name>` to `train_student.py`.

---

## What's not in git

| not committed | where it actually lives |
|---|---|
| `data/` (raw `.bag` recordings, ~44 GB) | shared lab drive / Dropbox |
| `frame_cache/` (JPEG cache, ~216 MB) | regenerate via `prepare_frames.py` |
| `annotations/` (per-bag parquets) | regenerate via the three annotators |
| `checkpoints_v2/` (student weights, ~12 MB) | regenerate via `train_student.py` |
| `renders/` (output videos) | mirrored to HF dataset above |
| `models/**/*.{pt,pth,onnx,pkl}` (open-source weights) | `scripts/download_models.py` |
| `.openai_key` | secret |

---

## Pointers

- Empirical results: `V3_TEST_REPORT.md`
- Earlier v2 (leaky VLM) results, kept for comparison: `V2_TEST_REPORT.md`
- Per-bag metrics JSON: `reports/v3_test_summary.json` (and `v2_*` for v2)
- Old v2 distillation guide (slightly out of date — kept for history): `docs/v2_distillation.md`
- Original pipeline / calibration docs: `README.md`
