# V3 Test-set Report — heuristic vs VLM (gpt-5.5, no leakage) vs student

Re-ran the VLM annotator after removing all human-labelled metadata from the
VLM input (scenario name, scenario descriptions, per-frame phase, event-
anchored priors) and switching the default model to `gpt-5.5`. Student and
heuristic are unchanged from V2; only the VLM column moved.

## Setup

| item | value |
|---|---|
| Split | `splits/v2.json` (same 10 test bags) |
| Heuristic teacher | unchanged (V2 parquets) |
| VLM teacher | `gpt-5.5` via Responses API, **no scenario / phase / event leakage** |
| Student | `checkpoints_v2/heuristic/best.pt` (unchanged) |
| Backup of leaky v2 VLM | `annotations/_vlm_v2_leaky_backup/` |

## Per-bag pairwise agreement (MAE on 0-100 / Pearson r)

| scenario | stem | session | meanH | meanV | meanS | H↔V | S↔H | S↔V |
|---|---|---|---|---|---|---|---|---|
| sc01_walkby            | 2026-04-14_22-39-13 | 04 | 51.0 | 24.2 | 51.5 | 34.2 / **0.56** | 2.3 / 0.82 | 34.2 / 0.61 |
| sc01_walkby            | 2026-05-08_16-24-50 | 05 | 53.9 | 31.7 | 47.4 | 29.7 / **0.51** | 5.8 / 0.28 | 30.8 / 0.76 |
| sc02_comfortable       | 2026-04-14_22-45-49 | 04 | 74.5 | 58.5 | 73.7 | 19.7 / **0.83** | 2.0 / 0.99 | 19.1 / 0.82 |
| sc02_comfortable       | 2026-04-14_22-48-05 | 04 | 76.1 | 57.6 | 75.9 | 20.1 / **0.91** | 2.2 / 0.99 | 19.8 / 0.92 |
| sc03_gradual_discomfort| 2026-05-08_16-45-35 | 05 | 79.3 | 60.2 | 77.5 | 23.8 / **0.73** | 3.4 / 0.96 | 22.5 / 0.71 |
| sc03_gradual_discomfort| 2026-05-08_16-53-45 | 05 | 62.3 | 43.3 | 56.2 | 24.4 / **0.67** | 7.1 / 0.62 | 26.1 / 0.55 |
| sc04_sudden_withdrawal | 2026-05-08_16-54-18 | 05 | 57.0 | 33.4 | 55.9 | 30.5 / **0.63** | 5.6 / 0.55 | 31.5 / 0.54 |
| sc04_sudden_withdrawal | 2026-05-08_17-04-51 | 05 | 62.0 | 28.8 | 63.9 | 33.2 / **0.93** | 5.7 / 0.96 | 36.4 / 0.95 |
| sc05_distracted        | 2026-04-14_23-02-10 | 04 | 66.2 | 42.3 | 67.6 | 25.5 / **0.80** | 4.5 / 0.92 | 26.6 / 0.86 |
| sc05_distracted        | 2026-04-14_23-03-34 | 04 | 61.2 | 26.1 | 62.0 | 35.9 / **0.86** | 3.6 / 0.91 | 36.9 / 0.81 |

## Aggregate

| split | n | H↔V MAE / r | S↔H MAE / r | S↔V MAE / r |
|---|---|---|---|---|
| all test bags          | 10 | **27.7 / 0.74** | 4.2 / 0.80 | 28.4 / 0.75 |
| 2026-04 (in-session)   | 5  | 27.1 / 0.79  | 2.9 / 0.93 | 27.3 / 0.80 |
| 2026-05 (cross-session)| 5  | 28.3 / 0.69  | 5.5 / 0.68 | 29.5 / 0.70 |

## V2 vs V3 comparison (same 10 test bags)

|  | V2 (gpt-4.1-mini, leaky prompt) | V3 (gpt-5.5, no leakage) |
|---|---|---|
| H↔V MAE | 25.3 | 27.7 |
| H↔V Pearson r | **-0.10** | **+0.74** |
| H↔V r on 2026-04 | +0.08 | +0.79 |
| H↔V r on 2026-05 | -0.28 | +0.69 |
| sc04_sudden_withdrawal VLM mean | 60.0 / 59.8 | 33.4 / 28.8 |
| sc04 H↔V Pearson r | -0.47 / -0.72 | +0.63 / +0.93 |

## Takeaways

1. **The "two teachers don't agree" finding from V2 was largely an artifact.**
   With a stronger model and no leakage of scenario / phase / event timing,
   the heuristic-vs-VLM correlation jumped from r = -0.10 → +0.74. They DO
   agree on trajectory direction; V2's negative correlation came from a
   combination of the weaker `gpt-4.1-mini` model and the event-anchored
   priors hard-pulling VLM scores in directions that didn't match the
   heuristic's smoother dynamics.

2. **The two teachers still disagree on absolute level (MAE ≈ 28).** VLM
   systematically scores **lower** than the heuristic — vlm_mean is 24-60
   while heuristic_mean is 51-79 on the same recordings. This is a real
   calibration gap, not noise: trajectories track each other but the VLM is
   harsher by ~20 points on 0-100.

3. **Withdrawal scenarios now look correct.** sc04 was the most striking case
   in V2: VLM gave 60/100 on a withdrawal recording. With the leakage removed
   and gpt-5.5, VLM correctly scores those low (28-33) and ALSO tracks the
   heuristic's downward trajectory (r = 0.63 and 0.93). This is the cleanest
   piece of evidence that the V2 "disagreement" was largely artificial.

4. **Student is still a faithful copy of the heuristic.** S↔H numbers are
   unchanged from V2 because we did not retrain. S↔V correlation rose
   (0.75 vs -0.03 in V2) purely because VLM became more aligned with
   heuristic — not because the student learned anything new.

5. **Cross-session generalisation gap persists.** Both H↔V agreement and
   S↔H agreement are weaker on 2026-05 (new recording session) than on
   2026-04 — confirming the cross-session distribution shift is real and not
   an artifact of the VLM teacher.

## Files

| artefact | path |
|---|---|
| VLM annotations (gpt-5.5, no leakage) | `annotations/<scen>/<stem>__vlm.parquet` |
| VLM annotations (leaky V2, archived) | `annotations/_vlm_v2_leaky_backup/` |
| VLM annotations (old prompt, archived) | `annotations/_vlm_old_prompt_backup/` |
| 3-way comparison videos | `renders/v3_3way_test/*.mp4` |
| Per-bag JSON | `reports/v3_test_summary.json` |
| Previous V2 report | `V2_TEST_REPORT.md` |
| Run log | `annotations/_vlm_v3_test_run.log` |

## Reproduce

```powershell
$venv = ".venv\Scripts\python.exe"

& $venv scripts\annotate_vlm.py --split-file splits\v2.json --split-set test --sleep-between 0.3
& $venv scripts\render_annotations.py --split-file splits\v2.json --split-set test --sources heuristic vlm student --out-dir renders\v3_3way_test
$env:SUMMARY_TAG = "v3"; & $venv scripts\_v2_test_summary.py
```
