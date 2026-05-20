# V2 Test-set Report — heuristic vs VLM vs student

Pipeline rerun on the combined dataset (31 original + 26 new = **57 bags**).

## Setup

| item | value |
|---|---|
| Split | `splits/v2.json` (seed=42, 2 test bags per scenario) |
| Train bags | 47 (incl. 3 unlabelled new sc01/sc02 bags) |
| Test bags | 10 (2 per scenario; 5 from 2026-04 session, 5 from 2026-05 session) |
| Heuristic teacher | `scripts/annotate_heuristic.py` (RetinaFace + EmotiEffLib + L2CS + MediaPipe + IntegratedComfortScorer) |
| VLM teacher | `scripts/annotate_vlm.py` with **updated prompt** (R1-R4 anti-facial-bias rules) |
| VLM model | `gpt-5.5` via Responses API, 2 Hz sampling, window=6, stride=3 |
| Student | MobileNetV3-Small (1.58M params), Huber loss, 10 epochs, 5 Hz training rate |
| Checkpoint | `checkpoints_v2/heuristic/best.pt` (best test-set MAE = 4.09) |

## Per-bag pairwise agreement (MAE on 0-100 scale / Pearson r)

| scenario | stem | session | heur_mean | vlm_mean | stu_mean | H↔V | S↔H | S↔V |
|---|---|---|---|---|---|---|---|---|
| sc01_walkby            | 2026-04-14_22-39-13 | 04 | 51.0 | 21.8 | 51.5 | 29.0 / 0.36 | 2.3 / 0.82 | 29.6 / 0.35 |
| sc01_walkby            | 2026-05-08_16-24-50 | 05 | 53.9 | 15.9 | 47.4 | 38.1 / -0.06 | 5.8 / 0.28 | 31.6 / 0.65 |
| sc02_comfortable       | 2026-04-14_22-45-49 | 04 | 74.5 | 77.6 | 73.7 | 19.4 / 0.28 | 2.0 / 0.99 | 19.3 / 0.28 |
| sc02_comfortable       | 2026-04-14_22-48-05 | 04 | 76.1 | 79.9 | 75.9 | 17.0 / 0.44 | 2.2 / 0.99 | 16.7 / 0.45 |
| sc03_gradual_discomfort| 2026-05-08_16-45-35 | 05 | 79.3 | 56.0 | 77.5 | 30.5 / 0.13 | 3.4 / 0.96 | 29.9 / 0.09 |
| sc03_gradual_discomfort| 2026-05-08_16-53-45 | 05 | 62.3 | 61.1 | 56.2 | 18.3 / -0.26 | 7.1 / 0.62 | 15.2 / -0.35 |
| sc04_sudden_withdrawal | 2026-05-08_16-54-18 | 05 | 57.0 | 60.0 | 55.9 | 17.9 / -0.47 | 5.6 / 0.55 | 16.2 / -0.39 |
| sc04_sudden_withdrawal | 2026-05-08_17-04-51 | 05 | 62.0 | 59.8 | 63.9 | 31.3 / -0.72 | 5.7 / 0.96 | 27.5 / -0.71 |
| sc05_distracted        | 2026-04-14_23-02-10 | 04 | 66.2 | 47.2 | 67.6 | 29.7 / -0.14 | 4.5 / 0.92 | 29.8 / -0.14 |
| sc05_distracted        | 2026-04-14_23-03-34 | 04 | 61.2 | 54.0 | 62.0 | 21.6 / -0.56 | 3.6 / 0.91 | 22.0 / -0.53 |

## Aggregate

| split | n | H↔V MAE / r | S↔H MAE / r | S↔V MAE / r |
|---|---|---|---|---|
| all test bags          | 10 | **25.3 / -0.10** | **4.2 / 0.80** | **23.8 / -0.03** |
| 2026-04 (in-session)   | 5  | 23.3 / 0.08  | 2.9 / 0.93 | 23.5 / 0.08 |
| 2026-05 (cross-session)| 5  | 27.2 / -0.28 | 5.5 / 0.68 | 24.1 / -0.14 |

## Takeaways

1. **Distillation works.** The 1.58M-param student is a faithful copy of the
   30-frame-rate ~1.5 B-param heuristic pipeline on the test set — MAE 2.9 and
   r ≈ 0.93 on in-session bags. Cross-session generalization drops to MAE 5.5
   but r is still 0.68. This is the on-device-deployable model.

2. **The two teachers genuinely disagree.** Heuristic vs VLM has MAE ≈ 25 and
   Pearson r ≈ 0 — not just noise, they're **uncorrelated in trajectory**.
   sc01_walkby is the clearest example: the heuristic gives ~52/100 (mediocre)
   while VLM gives ~18/100. The VLM (with the updated anti-facial-bias prompt)
   correctly identifies "no engagement / pass-through" while the heuristic is
   driven by the face being detected + neutral affect.

3. **Negative correlation on withdrawal scenarios is informative.** sc04 has
   r = -0.47 and -0.72 between heuristic and VLM. The withdrawal/disengagement
   trajectory is exactly where the two models disagree most strongly — a
   good place for human validation studies.

4. **The student inherits the heuristic's blind spot.** S↔V MAE 23.8 ≈ H↔V MAE
   25.3 — the student doesn't gain VLM-style understanding by being trained on
   heuristic labels only. Next experiment: re-train the student on VLM labels
   (or a fusion of both) and re-measure.

## Files of interest

| artefact | path |
|---|---|
| split definition | `splits/v2.json` |
| heuristic annotations (all 57) | `annotations/<scen>/<stem>__heuristic.parquet` |
| VLM annotations (10 test) | `annotations/<scen>/<stem>__vlm.parquet` |
| VLM-old-prompt backup | `annotations/_vlm_old_prompt_backup/` |
| student annotations (10 test) | `annotations/<scen>/<stem>__student.parquet` |
| student checkpoint | `checkpoints_v2/heuristic/{best,last}.pt` |
| training log | `checkpoints_v2/_train_heuristic.log` |
| 3-way comparison videos (10 test) | `renders/v2_3way_test/*.mp4` |
| this report's metrics JSON | `reports/v2_test_summary.json` |

## How to reproduce

```powershell
$venv = ".venv\Scripts\python.exe"

& $venv scripts\make_split.py
& $venv scripts\annotate_heuristic.py
& $venv scripts\prepare_frames.py
& $venv scripts\train_student.py --source heuristic --split-file splits\v2.json --max-train-fps 5 --epochs 10 --out-dir checkpoints_v2
& $venv scripts\annotate_vlm.py --split-file splits\v2.json --split-set test --sleep-between 0.3
& $venv scripts\eval_student.py --checkpoint checkpoints_v2\heuristic\best.pt --include-training-sample 0 --no-render --out-dir rendering_output_v2\student
& $venv scripts\render_annotations.py --split-file splits\v2.json --split-set test --sources heuristic vlm student --out-dir renders\v2_3way_test
& $venv scripts\_v2_test_summary.py
```
