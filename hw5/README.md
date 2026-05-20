# HW5 — Allergy-Aware Recipe Planning Agent

Submission for **MAS.S60 / 6.S985 — Modeling: Multimodal AI, Spring 2026 (MIT)**.

## What this agent does

A goal-directed agent that helps users plan recipes given their available ingredients,
allergies, and dietary restrictions. The agent is built on
[`smolagents`](https://github.com/huggingface/smolagents) `ToolCallingAgent`, runs on
`gpt-5.4-mini` (with `gpt-5.4-nano` as a comparison config in the online evaluation),
and is exposed end-to-end through a Discord bot in the shared `MMAI Agents World`
server.

The agent has access to four tools:

| Tool | Source | Role |
|---|---|---|
| `web_search` | `smolagents` built-in (DuckDuckGo) | Find candidate recipes from the open web |
| `visit_webpage` | `smolagents` built-in | Fetch a specific URL and pull text |
| `allergen_checker` | **custom** (Part 3 Problem 4) | Validate that a draft ingredient list does not contain any user-stated allergen / restricted item |
| `ingredient_substitution` | **custom** (Part 3 Problem 4) | Propose dietetically-appropriate substitutes when the checker flags a violation |

In Part 4 (multimodal) a fifth tool `read_food_image` is added for the vision A/B
comparison; it is *not* part of the production Discord agent.

The system prompt enforces a strict safety policy: the agent must call
`allergen_checker` before any recipe finalisation when the user states an allergy
or diet, must refuse out-of-scope or dangerous requests (medical advice, raw fugu,
home canning of low-acid foods, alcohol distillation, sub-1200 kcal/day plans), and
must *not* over-refuse legitimate accommodation requests like "peanut-free pad
thai".

## Notebook layout

The deliverable is a single Jupyter notebook organised into the assignment's six
parts:

| Part | What's in it |
|---|---|
| **Part 1** | Reading reflection (4 surveys / domain papers) — written content only |
| **Part 2** | Observability + evaluation design: 12-task `eval_set.json`, 3 metrics in `metric_spec.json`, the unified `trace_schema.json`, `EXPERIMENT_REGISTRY` |
| **Part 3** | Baseline `ToolCallingAgent` (Problem 3) → custom-tool agent with `allergen_checker` + `ingredient_substitution` (Problem 4); side-by-side eval on 5 representative tasks |
| **Part 4** | (P1) Vision A/B comparison: `text_only_mm_agent` vs. `vision_mm_agent` over a custom `mm_eval_set.json`. (P2) Safety + policy evaluation: 8 prompts × `BEFORE` (Part 3 baseline prompt) vs. `AFTER` (hardened prompt with explicit refusal policy) |
| **Part 5** | (P1) Langfuse setup. (P2) Trace recording + diagnosis of one suboptimal run. (P3) Online evaluation on a 2×2 grid (model = `gpt-5.4-mini` / `gpt-5.4-nano`, tool set = full / minimal) over 8 prompts → 32 runs, scored on success rate / latency / cost / tokens |
| **Part 6** | Same agent wrapped in a Discord bot, deployed to `MMAI Agents World`. Mention-only trigger strategy |

## Repository contents

```
hw5/
├── hw5_yirantao.ipynb                     # main notebook (Parts 1–6)
├── hw5_yirantao.pdf                       # written report (figures + discussion + tables)
├── README.md                              # this file
├── .gitignore
│
├── artifacts/                             # all evaluation outputs (reproducibility)
│   ├── eval_set.json                      # 12 tasks (normal / edge / ambiguous / adversarial)
│   ├── mm_eval_set.json                   # 5 multimodal tasks (Part 4 P1)
│   ├── metric_spec.json                   # 3 metrics + thresholds
│   ├── trace_schema.json                  # 17-field per-run trace schema
│   ├── experiment_registry.json           # which agent / tool set each run used
│   └── runs/
│       ├── baseline/                      # Part 3 P3: built-in-tools baseline
│       ├── custom/                        # Part 3 P4: + allergen_checker + ingredient_substitution
│       ├── baseline_vs_custom.json        # side-by-side
│       ├── mm_textonly/                   # Part 4 P1: Version A traces
│       ├── mm_vision/                     # Part 4 P1: Version B traces
│       ├── mm_textonly_vs_vision.json     # A/B summary
│       ├── safety/                        # Part 4 P2: BEFORE + AFTER traces × 8 prompts + adversarial_v1.jpg (typographic prompt-injection image)
│       └── online_eval/                   # Part 5 P3: 4 configs × 8 prompts
│
├── diagram.png                            # Part 4 P1 architecture diagram
├── Screenshot.png                         # Part 6 Discord interaction
├── langfuse_p3_traces_list.png            # Part 5 P3 — 32 runs in Langfuse dashboard
├── langfuse_p3_full_trace.png             # Part 5 P3 — primary_full trace detail
├── langfuse_p3_minimal_trace.png          # Part 5 P3 — alt_minimal trace detail
└── langfuse_p3_minimal_trace_v2.png       # Part 5 P3 — alt_minimal alternate
```

The written report `hw5_yirantao.pdf` contains the figures, comparison tables,
and per-part discussion that the notebook itself does not include — TAs should
read the PDF as the primary writeup and refer to the notebook for the
underlying code and raw artifacts.

## Reproducing the runs

1. Open `hw5_yirantao.ipynb` in JupyterLab / VS Code / Colab (an A100 is recommended for
   speed but not strictly required — the agent itself is API-bound).
2. Set the following environment variables before running (or supply them via
   the in-notebook `getpass` prompts):
   - `OPENAI_API_KEY`
   - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
     (only needed for Part 5)
   - `DISCORD_TOKEN` (only needed for Part 6)
3. Run cells top-to-bottom. Each part is self-contained and re-creates its own
   `artifacts/runs/<…>/` subfolder, so individual parts can be re-run
   independently.
4. **Cell 40 (Part 6) is long-running** — it starts the Discord bot's event
   loop and will not return until you stop it. Skip it if you only want to
   reproduce evaluations.

## Notes on what's in `artifacts/runs/`

- Every per-run trace JSON conforms to `artifacts/trace_schema.json` (17 fields:
  ids, query, expected behaviour, model, tool set, full step list, final answer,
  latency, token counts, judge score, pass flags, error, timestamp).
- Per-run files are named `<task_id>_<short_uuid>.json`. Multiple files for the
  same task id under `baseline/` reflect re-runs across the session — the file
  used by the writeup is the one referenced from the corresponding `*_summary.json`.
- All judge scores in this submission come from a `gpt-5.4-mini` LLM judge with a
  task-specific rubric defined inline in the notebook (see Part 3 P3, Part 4 P1,
  Part 4 P2, Part 5 P3).

## Trigger strategy for the Discord bot (Part 6)

The bot uses **mention-only** triggering: it only acts when a user explicitly
`@`s it. This keeps cost predictable in a shared multi-bot world, avoids false
triggers from incidental keyword mentions, and composes cleanly with other
students' bots that also activate on `@`. Full reflection is in `hw5_yirantao.pdf`;
the demo screenshot is `Screenshot.png`.

## How HW5 connects to the rest of the portfolio

HW5 is the *agents* assignment, and it is where the portfolio pivots from
"fine-tune a model on a static dataset" (HW3 SFT/LoRA, HW4 GRPO) to
"orchestrate an LLM that has to call external tools, follow safety policy,
and operate inside a real chat environment". The eval/observability scaffolding
built here (the unified `trace_schema.json`, the `EXPERIMENT_REGISTRY`, the
LLM-judge rubric pattern, and the Langfuse trace dashboard) is also the
methodological precursor to the calibration/evaluation thinking used in the
final project, where a 4-channel comfort scorer is calibrated against a small
RGB-D corpus and cross-checked against an independent VLM judge.

> A standalone copy of this assignment also lives at
> [github.com/yirantao1000/mmai-hw5](https://github.com/yirantao1000/mmai-hw5);
> the version here is the canonical one and is identical in content.
