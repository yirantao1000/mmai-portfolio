# V2 — VLM/heuristic distillation pipeline

Goal: use the existing multi-model heuristic pipeline **and** a frozen VLM as
two independent annotators of `comfort_score` over each `.bag`, then train a
single small per-frame regressor that can run real-time on the robot.

```
.bag (RealSense)
   │
   ├──► annotate_heuristic.py ──► annotations/<scen>/<stem>__heuristic.parquet  (30 Hz)
   │
   ├──► annotate_vlm.py       ──► annotations/<scen>/<stem>__vlm.parquet       (~2 Hz)
   │
   └──► annotate_human.py     ──► annotations/<scen>/<stem>__human-<rater>.parquet  (optional, 2 Hz)

annotations/  +  frame_cache/  ──► train_student.py  ──► checkpoints/<source>/best.pt
                                              │
                                              └─►  eval_student.py  ──► rendering_output/student/...
                                                                       + annotations/<scen>/<stem>__student.parquet
```

All annotation files share a single schema (see `src/annotations.py`) so any
downstream consumer can swap sources transparently.

## End-to-end run

```bash
# 0) install (in the project venv)
pip install -r requirements.txt

# 1) heuristic annotator (uses the calibrated config + your existing models)
python scripts/annotate_heuristic.py
# -> annotations/<scen>/<stem>__heuristic.parquet (one row per .bag frame)

# 2) VLM annotator
export OPENAI_API_KEY=sk-...
python scripts/annotate_vlm.py --model gpt-4.1-mini --fps 2 --window 6 --stride 3
# Tip: try `--dry-run` first to make sure the pipeline runs end-to-end without
# spending tokens; it produces synthetic soft labels with the same schema.

# 3) (optional) human dense annotation, e.g. for a 5-bag held-out validation
python scripts/annotate_human.py --interactive --rater alice
# 1-5 keys to score, arrows to step, s to save.

# 4) render side-by-side overlays of the two automatic annotators
python scripts/render_annotations.py --sources heuristic vlm --width 480 --fps 15
# -> rendering_output/<scen>__<stem>__heuristic_vlm.mp4

# 5) prepare a JPEG frame cache (one-time; lets training skip .bag decoding)
python scripts/prepare_frames.py --fps 15 --resize-max 320

# 6) train two student models, one per teacher source
python scripts/train_student.py --source heuristic --epochs 10
python scripts/train_student.py --source vlm       --epochs 10
# Each saves checkpoints/<source>/{best,last}.pt and a summary.json with
# the train/test split. The split holds out 2 recordings (sc02 + sc04 by
# default) so each test set covers both abort and continue.

# 7) evaluate + render comparison videos for both students
python scripts/eval_student.py --checkpoint checkpoints/heuristic/best.pt
python scripts/eval_student.py --checkpoint checkpoints/vlm/best.pt
# Renders eval videos with GT-vs-student bars stacked on top of the original
# frames, plus a per-bag metrics JSON. Adds one randomly-picked training
# recording for each model so you can sanity-check fit on seen data.
```

## Output paths at a glance

| What                         | Where                                                          |
|------------------------------|----------------------------------------------------------------|
| heuristic per-frame labels   | `annotations/<scen>/<stem>__heuristic.parquet`                 |
| VLM per-timestamp labels     | `annotations/<scen>/<stem>__vlm.parquet`                       |
| human labels                 | `annotations/<scen>/<stem>__human-<rater>.parquet`             |
| student predictions          | `annotations/<scen>/<stem>__student.parquet`                   |
| comparison overlay videos    | `rendering_output/<scen>__<stem>__<sources>.mp4`               |
| student eval overlay videos  | `rendering_output/student/<source>/<scen>__<stem>__<tag>.mp4`  |
| trained checkpoints          | `checkpoints/<source>/{best,last}.pt`                          |
| frame cache + manifest       | `frame_cache/<scen>/<stem>/*.jpg`, `frame_cache/manifest.csv`  |

## Annotation schema (parquet columns)

| col              | type    | notes                                                  |
|------------------|---------|--------------------------------------------------------|
| timestamp_s      | float32 | seconds from bag start                                 |
| frame_idx        | int32   | bag-native frame index (consistent across sources)     |
| comfort_score    | float32 | 0–100 (canonical, what the student is trained against) |
| comfort_score_5  | float32 | 1–5 ordinal (raw scale; NaN for heuristic)             |
| confidence       | float32 | 0–1, NaN if not produced                               |
| rationale        | string  | VLM/human note, empty otherwise                        |
| source           | string  | `heuristic` \| `vlm` \| `human` \| `student`           |
| scenario         | string  | `sc02_comfortable`, etc.                               |
| bag_stem         | string  | original `.bag` filename stem                          |
| phase            | string  | `approach` \| `intent` \| `execution` \| ""            |

## VLM annotator design tricks (mapped to methods-section bullets)

1. **Sliding-window prompts with overlap** (`--window 6 --stride 3`) — adjacent
   calls share frames, predictions are averaged across overlaps.
2. **Event-conditioned prompting** — sidecar phase string is fed in as a per-frame
   tag, scenario-specific descriptions seed scenario context.
3. **Soft 1-5 distribution output** (not a single integer) — supports
   ordinal-regression / KL distillation downstream.
4. **Free-text rationale capture** — one sentence per window, logged into the
   parquet for inspection and future rationale distillation.
5. **Event-anchored prior blending** — after `abort_time` the soft is blended
   with `[0.4, 0.6, 0, 0, 0]`; after a successful `handover_time + 1s` with
   `[0, 0, 0, 0.4, 0.6]`. Blend weight = 0.6.
6. **Causal EMA smoothing** (`--smooth-tau-s 0.6`) — final per-timestamp
   `comfort_score` mimics what a deployed real-time system would emit.

## Notes / gotchas

- `annotate_vlm.py --dry-run` is the recommended way to verify the data path
  before spending tokens; it skips the API and writes synthetic Dirichlet-soft
  labels in the same schema.
- The student trains on continuous `comfort_score`. The 5-bin distribution is
  not currently stored in the parquet; if you want soft-distribution distillation
  later, persist `soft_5` from `annotate_vlm.py` and add a KL term to
  `train_student.py`.
- For the heuristic source, you may want to decimate the 30 Hz labels with
  `train_student.py --max-train-fps 5` so VLM-trained and heuristic-trained
  students see comparable amounts of supervision.
