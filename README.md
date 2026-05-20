# Yiran Tao — Multimodal AI Portfolio

Personal portfolio for **MAS.S60 / 6.S985 — Modeling: Multimodal AI** (MIT,
Spring 2026), maintained per the course's GitHub-repository requirement.

> *"This repository should bring everything together in one place: past
> assignments, ongoing explorations, and final projects. Each repository must
> be clearly organized, publicly readable within the class, and include a
> top-level README.md, a README.md for each assignment, and all completed
> work so far."* — course final instructions

This repo is that one place: every weekly assignment, its writeup, and the
final group project, each with its own self-contained README.

## Contents

| folder | what's in it | primary deliverable |
|---|---|---|
| [`hw1/`](hw1) | **Dataset choice, modality extraction, visualization & evaluation design.** Picks CLAMP as the substrate for the semester's "classical multimodal" arc, extracts haptic time-series + visual-prior modalities, and commits to the evaluation protocol (Accuracy / Macro-F1 / Balanced Accuracy / MCC). | [`hw1/yirantao_hw1.pdf`](hw1/yirantao_hw1.pdf) |
| [`hw2/`](hw2) | **Multimodal fusion & alignment on CLAMP and AV-MNIST.** Implements and compares Early / Late / Tensor / LMF fusion (Tensor Fusion wins at 85.43% val), plus an InfoNCE contrastive-alignment experiment between haptic and visual-prior modalities. | [`hw2/yirantao_hw2.pdf`](hw2/yirantao_hw2.pdf) |
| [`hw3/`](hw3) | **VLM prompting + LoRA fine-tuning** on the beans leaf-disease dataset. Baseline zero-shot, three prompt-engineering variants, and a LoRA fine-tune that takes held-out exact-match accuracy from **50% → 100%**. | [`hw3/yirantao_hw3.pdf`](hw3/yirantao_hw3.pdf) |
| [`hw4/`](hw4) | **GRPO reinforcement learning for VLMs** on the same beans dataset. Implements GRPO end-to-end and compares it against the HW3 SFT/LoRA baseline — **75% accuracy + 100% format compliance**, with full reasoning traces. | [`hw4/yirantao_hw4.pdf`](hw4/yirantao_hw4.pdf) |
| [`hw5/`](hw5) | **Allergy-aware recipe-planning agent.** Goal-directed `smolagents` `ToolCallingAgent` with custom `allergen_checker` + `ingredient_substitution` tools, full eval harness (12-task `eval_set`, 17-field unified trace schema, LLM-judge rubric, 2×2 model×toolset online eval grid in Langfuse), safety-policy A/B, and a deployed Discord bot in the shared `MMAI Agents World` server. | [`hw5/hw5_yirantao.pdf`](hw5/hw5_yirantao.pdf) |
| [`final-project/`](final-project) | **Group final project — A Real-Time Perception Pipeline for Adaptive Human-Robot Interaction.** My personal contribution within the team is the **V3 distillation pipeline + VLM cross-check** ([`final-project/MY_CONTRIBUTION.md`](final-project/MY_CONTRIBUTION.md)). | [`final-project/FINAL_REPORT.pdf`](final-project/FINAL_REPORT.pdf) |

Each subfolder has its own `README.md` with the writeup index, key results,
files list, and a "how this connects to the rest of the portfolio" note.

## How the portfolio fits together

The five homeworks plus the final project trace a single coherent story
about *how I learned to do multimodal AI this semester*, organised into
three arcs:

1. **Classical multimodal (HW1 → HW2).**
   Pick a dataset (CLAMP), extract heterogeneous modalities (haptic
   time-series + visual priors) into a reproducible pipeline, and use it
   to compare *raw* fusion and alignment methods — Early / Late / Tensor /
   LMF fusion plus InfoNCE contrastive alignment. The takeaways (when does
   higher-order fusion help, when does alignment quality matter more than
   loss reduction) are the ones that the later foundation-model work
   takes for granted.

2. **Foundation-model adaptation (HW3 → HW4).**
   Same fine-grained vision task (beans leaf disease) used twice as the
   substrate for two very different adaptation regimes:
   *supervised fine-tuning* with LoRA (HW3) vs. *reinforcement learning*
   with GRPO (HW4). HW3 wins on raw classification accuracy
   (50% → 100%), HW4 wins on behavioural alignment
   (100% format-compliant `Answer:` outputs with explicit step-by-step
   reasoning) — a clean apples-to-apples comparison of "imitate the
   correct answer" vs. "optimise for the desired behaviour".

3. **Agents + real-system perception (HW5 → final project).**
   HW5 builds a goal-directed agent that has to call external tools,
   follow a safety policy, and operate inside a real Discord environment,
   with a serious eval harness (unified trace schema, LLM judge,
   Langfuse-backed online eval grid) attached. The final project then
   takes that same "calibrate, evaluate against a small site-specific
   corpus, cross-check against an independent judge" methodology and
   applies it end-to-end to a real-time, multimodal human-robot
   interaction system.

## Final Project (TL;DR)

> **A Real-Time Perception Pipeline for Adaptive Human-Robot Interaction**
> Heejung Roh, **Yiran Tao**, Sparsh Bansal, Jung Yeop (Steve) Kim — MIT

Late-fuses four frozen pretrained vision models (face, expression, gaze,
depth-aware pose) into a continuous 0–100 *comfort score* that gates a
robot's handover policy in real time. The fusion layer is calibrated by
differential evolution from a small (n = 31) site-specific RGB-D corpus
(`config/deploy.yaml` lifts Youden's J from 0.00 → 1.00 on held-out test).
The ~70 M-param teacher is then **distilled into a 1.58 M-param
single-frame MobileNetV3-Small student** (44× smaller) that tracks the
teacher at MAE 4.1 / r = 0.80, cross-checked against an independent
GPT-5.5 VLM judge (r = 0.74 on trajectory shape) to verify the signal
reflects observable behavior rather than calibration artifacts.

- Final report (PDF): [`final-project/FINAL_REPORT.pdf`](final-project/FINAL_REPORT.pdf)
- My contribution within the project: [`final-project/MY_CONTRIBUTION.md`](final-project/MY_CONTRIBUTION.md)
- Reproducible pipeline guide: [`final-project/V3_PIPELINE.md`](final-project/V3_PIPELINE.md)
- Headline numbers: [`final-project/V3_TEST_REPORT.md`](final-project/V3_TEST_REPORT.md)
- Side-by-side comparison videos + verbatim VLM prompt: HuggingFace dataset [`yirantao1000/mmai-comfort-handover-v3`](https://huggingface.co/datasets/yirantao1000/mmai-comfort-handover-v3)
- Team source repo: [`mmai-social-robots/mmai-integrated-detection`](https://github.com/mmai-social-robots/mmai-integrated-detection) (branch [`yiran/v3-distillation`](https://github.com/mmai-social-robots/mmai-integrated-detection/tree/yiran/v3-distillation))

Per the course rule that *"each team member should include their own copy of
the project in their repository"*, the final project lives here as a complete
self-contained copy under [`final-project/`](final-project), not just a
pointer to the team repo.

## Reproducing things

Each subfolder is self-contained:

- **HW1 / HW2**: open the notebook in JupyterLab or Colab. HW1 needs
  `CLAMP_dataset_filtered.npz` (`.npz` file from the
  [CLAMP dataset](https://emprise.cs.cornell.edu/clamp/)); HW2 reuses
  the same extraction and additionally pulls AV-MNIST for the unimodal
  baselines in Problem 3.
- **HW3 / HW4**: notebooks pull the
  [`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans)
  dataset directly from HuggingFace; GPU strongly recommended for the
  LoRA fine-tune (HW3) and the GRPO RL loop (HW4).
- **HW5**: see [`hw5/README.md`](hw5/README.md). Requires `OPENAI_API_KEY`
  to run the agent, `LANGFUSE_*` keys to reproduce the online eval, and a
  `DISCORD_TOKEN` to reproduce the deployed Part-6 bot.
- **Final project**: see [`final-project/README.md`](final-project/README.md)
  for the full pipeline (calibrate → extract → optimise → evaluate) and the
  V3 distillation guide in [`final-project/V3_PIPELINE.md`](final-project/V3_PIPELINE.md).
  Needs Intel RealSense `.bag` recordings as input.

## Standalone-repo note

HW5 also exists as a standalone repository at
[`yirantao1000/mmai-hw5`](https://github.com/yirantao1000/mmai-hw5)
(originally created for the HW5 deadline submission). The copy under
[`hw5/`](hw5) here is identical in content and is the canonical one going
forward.
