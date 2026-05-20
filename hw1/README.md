# HW1 — Dataset Choice, Modalities, Visualization & Evaluation Design

Submission for **MAS.S60 / 6.S985 — Modeling: Multimodal AI, Spring 2026 (MIT)**.

This assignment is the *foundational* one: pick the dataset I want to build on
for the rest of the semester, decide which modalities I will actually use,
extract them into a clean pipeline, look at them with my own eyes (i.e.
visualizations), and commit to an evaluation protocol before doing any
modelling. The written report `yirantao_hw1.pdf` walks through all six parts in
detail; this README is the index.

## What the project is going to be (Part 1)

A **visuo-haptic material recognition** model that classifies an object's
material from multimodal sensor signals, with explicit focus on situations
where vision alone is unreliable (occlusion, clutter, visually-similar
materials). The end goal motivates everything downstream: which dataset,
which modalities, and which metrics.

**Dataset chosen:** [CLAMP](https://emprise.cs.cornell.edu/clamp/) — directly
designed for material-related perception, multimodal, large enough to be
practical, and ships a preprocessed `CLAMP_dataset_filtered.npz` that gives a
reproducible entry point. Known drawbacks logged in the PDF: class imbalance
and in-the-wild label noise.

**Modalities used:**
- **Haptic time-series** — force, thermal, vibration, proprioception-derived
  signals. Encodes physical properties (i.e. the actual material).
- **Visual priors** — image-derived material probability vectors over the
  CLAMP class set. Provides an appearance-based complementary cue.

Other modalities (raw images/video, language, audio) were *deliberately* left
out at this stage to keep the pipeline tractable; the PDF gives the reasoning.

## Modality extraction pipeline (Part 2)

The notebook (`yirantao_hw1.ipynb`) implements the actual extraction from the
`.npz`:

- Converts the object-dtype field into dense `[N, C, T]` tensors.
- Uses `feature_key` to map semantic channel names → indices so haptic
  sub-modalities are split correctly.
- Converts the per-sample visual-prior dicts into fixed-length 14-D
  class-aligned probability vectors.

These steps are the ones the PDF lists as the main practical pain points.

## Visualizations (Part 3)

Five visualizations on a fixed-seed 2,500-sample subset, each chosen to
*inform a modelling decision*, not to look pretty:

| Plot | What it answered |
|---|---|
| t-SNE on multimodal summary features | Are classes naturally separable? → only weak local grouping; classes overlap a lot, so we'll need real fusion. |
| Multi-channel time-series sample plots | Do haptic channels carry signal? → force rises and stabilises, active-thermal decays, contact-mic shows transients. Signal is real. |
| Material label histogram | Are classes balanced? → no (`hard_plastic` dominates `foam` / `brass` / `porcelain`), so use imbalance-aware metrics. |
| Channel-level boxplots | Are channel scales comparable? → no (thermal channels dominate, derivative/mic channels ≈ 0), so we need per-channel normalisation. |
| Channel correlation heatmap | Are channels redundant or complementary? → some pairs are highly correlated (`Force` ↔ `Force diff`), others nearly independent. Motivates multimodal fusion design. |

All five plots and the discussion are embedded in the PDF.

## Evaluation protocol (Part 4)

Committed to **Accuracy + Macro-F1 + Balanced Accuracy + MCC**. Rationale (in
the PDF): accuracy for intuition, the other three to be robust under the
observed class imbalance. Other candidates considered but deferred: Weighted-F1,
per-class P/R, top-k accuracy, calibration metrics (ECE / Brier), full
confusion matrix.

## Prompting exercise (Part 5)

Three zero-shot LLM prompts written and tested for very different output
shapes — single-word classification (restaurant review sentiment), single-word
face-emotion classification with explicit allowed-label set, and strict-JSON
information extraction from a paragraph. Full prompts are in §Part 5 of the
PDF.

## Reflection (Part 6)

Most interesting topic: the visualization section, because each plot was
explicitly tied to a downstream modelling decision rather than treated as
decoration. Biggest unexpected difficulty: **practical data accessibility** —
many "open" datasets are partially released, fragmented, or too large to use
in a homework timeline. Resolution: treat dataset selection as an engineering
decision (reproducibility + feasibility first), not just a research wish-list.

## Files

| File | What it is |
|---|---|
| `yirantao_hw1.ipynb` | Notebook with the CLAMP extraction pipeline, visualizations, metric stubs, and prompt examples. |
| `yirantao_hw1.pdf` | Written report covering all six parts. **The PDF is the primary writeup**; the notebook is the underlying code. |

## How HW1 connects to the rest of the portfolio

HW2 (fusion + alignment) builds *directly* on the CLAMP pipeline created here
— the same `CLAMP_dataset_filtered.npz` extraction is reused to compare
early / late / tensor / LMF fusion, plus an InfoNCE contrastive alignment
experiment between haptic and visual-prior modalities. So HW1 is not a
standalone exercise; it is the substrate everything CLAMP-related in HW2
runs on.
