# My Contribution to the Group Final Project

The final project — *A Real-Time Perception Pipeline for Adaptive Human-Robot Interaction* — is a four-person team effort with Heejung Roh, Sparsh Bansal, and Jung Yeop (Steve) Kim. This file documents which pieces of the project I (Yiran Tao) personally led, so a grader can quickly tell who did what.

## Headline contribution

I built and shipped the **V3 distillation pipeline** described in §7 (*Edge Deployment*) of the final report, including the independent GPT-5.5 VLM cross-check that probes whether the teacher's calibrated comfort signal reflects observable behavior rather than calibration artifacts. Concretely:

- Designed and implemented the **annotation pipeline** that turns the calibrated teacher into per-frame supervision: `scripts/annotate_heuristic.py`, `scripts/annotate_vlm.py`, `scripts/annotate_human.py`, `scripts/prepare_frames.py`, `scripts/render_annotations.py`, `scripts/eval_student.py`, `scripts/train_student.py`, `scripts/make_v2_split.py`, all the helper scripts under `scripts/_*.py`, and the shared schema / dataloader / model / writer under `src/annotations.py`, `src/student/`, `src/video_writer.py`.
- Curated the **57-bag distillation corpus** (merged the original 31 calibration recordings with 26 new bags collected in the 2026-05 session) and the deterministic train/test split (`splits/v2.json`, 47/10, 2 per scenario).
- Trained the **MobileNetV3-Small student** (1.58 M params, 44× smaller than the teacher) by Huber regression on heuristic labels — final S↔T MAE 4.1 / r 0.80 on the held-out test set.
- Engineered the **GPT-5.5 VLM judge** with explicit no-leakage prompt design (scenario / phase / event timing are all withheld). Sliding-window K=6 stride=3, ordinal soft distribution over {1..5}, per-window rationale, causal EMA. Full verbatim system prompt and four design tricks documented at [`VLM_PROMPT.md`](https://huggingface.co/datasets/yirantao1000/mmai-comfort-handover-v3/blob/main/VLM_PROMPT.md). T↔V per-frame trajectory Pearson r = 0.74 reported in the paper.
- Wrote the **three-way comparison rendering** (heuristic / VLM / student tracks overlaid on the original video) and the per-bag agreement metrics (`reports/v3_test_summary.json`).
- Mirrored the 10 rendered test videos + the verbatim VLM prompt + per-bag metrics to the HuggingFace dataset [`yirantao1000/mmai-comfort-handover-v3`](https://huggingface.co/datasets/yirantao1000/mmai-comfort-handover-v3) so the artefacts are inspectable without cloning the repo.
- Authored the V3-pipeline-related docs in this folder: [`V3_PIPELINE.md`](V3_PIPELINE.md), [`V3_TEST_REPORT.md`](V3_TEST_REPORT.md), and the predecessor [`V2_TEST_REPORT.md`](V2_TEST_REPORT.md) (the leaky-prompt baseline that motivated removing scenario / phase / event leakage from the VLM input).
- For the team report I wrote §7 (Edge Deployment) end to end and contributed to the abstract, contributions, introduction, and limitations sections that reference the distillation result.

## Robot deployment

Beyond the V3 distillation pipeline above, I also led the **end-to-end deployment of the comfort scorer onto the physical Rainbow Robotics RB-Y1 manipulator** — bringing the calibrated pipeline out of recorded `.bag` evaluation and onto the actual robot's on-board compute with the live RealSense D435 feed driving the controller in closed loop. This is what makes the system a real-time HRI demo rather than just an offline analysis on a static dataset.

## Git provenance

Every line of code I added is reachable from commits on the upstream team repository's [`yiran/v3-distillation`](https://github.com/mmai-social-robots/mmai-integrated-detection/tree/yiran/v3-distillation) branch (authored as `Yiran Tao <yirantao1000@gmail.com>`). The folder you are reading was created by cloning that branch on 2026-05-19.
