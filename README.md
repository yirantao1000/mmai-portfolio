# Yiran Tao — Multimodal AI Portfolio

Personal portfolio for **MAS.S60 / 6.S985 Multimodal AI** (MIT, Spring 2026), maintained per the course's GitHub-repository requirement.

> *"This repository should bring everything together in one place: past assignments, ongoing explorations, and final projects."* — course final instructions

## Contents

| folder | what's in it |
|---|---|
| [`final-project/`](final-project) | Group final project: real-time perception pipeline for adaptive human-robot interaction. My personal contribution within the team is the **V3 distillation pipeline + VLM cross-check** (see `final-project/MY_CONTRIBUTION.md`). |

## Final Project (TL;DR)

> A Real-Time Perception Pipeline for Adaptive Human-Robot Interaction
> Heejung Roh, **Yiran Tao**, Sparsh Bansal, Jung Yeop (Steve) Kim — MIT

Late-fuses four frozen pretrained vision models (face, expression, gaze, depth-aware pose) into a continuous 0–100 comfort score that gates a robot's handover policy in real time. The fusion layer is calibrated by differential evolution from a small (n = 31) site-specific RGB-D corpus (`config/deploy.yaml` lifts Youden's J from 0.00 → 1.00 on held-out test). The ~70 M-param teacher is then **distilled into a 1.58 M-param single-frame MobileNetV3-Small student** (44× smaller) that tracks the teacher at MAE 4.1 / r = 0.80, cross-checked against an independent GPT-5.5 VLM judge (r = 0.74 on trajectory shape) to verify the signal reflects observable behavior rather than calibration artifacts.

- Final report (PDF): [`final-project/FINAL_REPORT.pdf`](final-project/FINAL_REPORT.pdf)
- My contribution within the project: [`final-project/MY_CONTRIBUTION.md`](final-project/MY_CONTRIBUTION.md)
- Reproducible pipeline guide: [`final-project/V3_PIPELINE.md`](final-project/V3_PIPELINE.md)
- Headline numbers: [`final-project/V3_TEST_REPORT.md`](final-project/V3_TEST_REPORT.md)
- Side-by-side comparison videos + verbatim VLM prompt: HuggingFace dataset [`yirantao1000/mmai-comfort-handover-v3`](https://huggingface.co/datasets/yirantao1000/mmai-comfort-handover-v3)
- Team source repo: [`mmai-social-robots/mmai-integrated-detection`](https://github.com/mmai-social-robots/mmai-integrated-detection) (branch [`yiran/v3-distillation`](https://github.com/mmai-social-robots/mmai-integrated-detection/tree/yiran/v3-distillation))

## A note on earlier assignments

The semester's mini-assignments were submitted as team deliverables in the shared `mmai-social-robots` org rather than as separate individual artifacts; for that reason this portfolio currently focuses on the final project. The final project itself is included as a complete self-contained copy under `final-project/` per the course rule that *"each team member should include their own copy of the project in their repository."*
