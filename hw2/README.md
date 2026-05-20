# HW2 — Multimodal Fusion & Alignment

Submission for **MAS.S60 / 6.S985 — Modeling: Multimodal AI, Spring 2026 (MIT)**.

This assignment is where the CLAMP pipeline from HW1 actually gets used to
*compare fusion and alignment methods head-to-head*. The written report
`yirantao_hw2.pdf` covers all seven parts; this README is the index and the
TL;DR of results.

## Conceptual framing (Part 1, reading reflection)

Six short essays grounded in the *Align Before Fuse* / ALBEF reading and the
course fusion/alignment taxonomy. The key positions I committed to:

- **Fusion stage**: *Intermediate (mid-level) fusion* on top of modality-specific
  encoders — a 1-D CNN for haptic time-series and an MLP for visual priors.
- **Fusion mechanism**: compare *additive* (concat + MLP) against *gated*
  fusion, because my target use-case explicitly cares about scenarios where
  vision is unreliable, and gated fusion can learn to down-weight a noisy
  modality.
- **Alignment**: *explicit* alignment via *InfoNCE-style contrastive learning*
  at the haptic↔visual-prior pair level. DTW-style continuous alignment is
  not needed because the visual modality is a static vector, not a sequence.
- **Platonic Representation Hypothesis (PRH)**: I expect implicit alignment
  to *not* emerge at CLAMP scale because (i) the visual modality is already a
  compressed model-derived prior (not raw pixels), (ii) haptics carries strong
  modality-unique cues (thermal, micro-vibration) that are not recoverable
  from vision, and (iii) we are well below the regimes where PRH evidence is
  strongest. Explicit alignment is still warranted.
- **Risks of over-aligning**: loss of haptic-unique information, bias
  propagation from noisy visual priors, brittleness under missing/corrupted
  vision. The PDF spells out controlled experiments for each.

Cited works in the discussion: the original PRH paper, *Revisiting PRH: An
Aristotelian View* (Gröger, Wen, Brbić 2026), *Getting Aligned on
Representational Alignment* (Sucholutsky et al. TMLR 2025), the model-stitching
literature, and *How Not to Stitch Representations to Measure Similarity*
(Balogh & Jelasity 2024/25).

## Part 2 — PyTorch / einsum warmups (Problems 1–2)

Tensor ops, einsum primitives (`einsum('ij,jk->ik')`, transpose, diagonals,
element-wise vs. matrix product). Setup for everything that follows.

## Unimodal AV-MNIST baselines (Problem 3)

Tuned hyperparameters per single modality on AV-MNIST.

| Modality | Best hyperparams | Best val acc | Best test acc |
|---|---|---|---|
| Audio | `lr=1e-3`, `wd=1e-3`, `epochs=10`, `bsz=128`, `dropout=0.0` | **41.68%** | **41.17%** |
| Image | `lr=1e-3`, `wd=1e-3`, `epochs=6`, `bsz=128`, `dropout=0.3` | **68.92%** | **64.57%** |

**Takeaway:** clear modality gap (image ≫ audio). The PDF discusses six concrete
ways to close it (capacity, SpecAugment, scheduling, weighted loss, KD from a
multimodal teacher, auxiliary unimodal losses during multimodal training).

## Fusion theory (Problem 4)

Why training plateaus → optimization vs. alignment diagnosis (ALBEF-style:
modalities may live in different representation spaces, fix it by aligning
*before* fusing). Catalogue of fusion alternatives: late ensembling,
multiplicative / bilinear, tensor fusion, LMF, gated, cross-attention. Early
vs. late fusion trade-offs discussed in the PDF.

## Multimodal fusion on CLAMP (Problem 5)

Four fusion methods implemented and compared on the same CLAMP setup from
HW1. Shared training controls: AdamW, `lr=1e-3`, `wd=1e-4`, 15 epochs,
batch 128, CrossEntropyLoss, fixed seed, 80/10/10 split. Embedding dim 64
(48 for Tensor Fusion to keep params manageable), LMF rank 8.

| Model | Validation accuracy |
|---|---|
| Visual prior (unimodal baseline) | 84.43% |
| Haptic (unimodal baseline) | 33.69% |
| Early Fusion | 84.19% |
| Late Fusion | 84.27% |
| **Tensor Fusion** | **85.43%** |
| LMF Fusion | 85.04% |

**Best technique:** **Tensor Fusion** — highest val acc (85.43%) and highest
test acc (83.96%) among multimodal models. The likely reason (discussed in
the PDF): material recognition benefits from higher-order cross-modal
interactions, which Tensor Fusion explicitly captures via outer products.
**LMF is the better practical trade-off**: nearly the same accuracy at
much lower parameter cost.

The PDF also includes per-method visualizations of *number of parameters*,
*memory usage*, and *time-to-convergence*, plus a unimodal-vs-multimodal
pros/cons discussion.

## Contrastive alignment on CLAMP (Problem 6)

InfoNCE-style contrastive learning between haptic embeddings and visual-prior
embeddings, using `einsum` to compute the similarity matrix and symmetric
cross-entropy on the diagonal.

Headline retrieval numbers: **Recall@1 = 3.50%**, **Recall@5 = 9.50%** — i.e.
exact instance matching remains hard. But the structure is interesting:

- The post-alignment similarity heatmap (Figure 1) does *not* show a clean
  diagonal, but it is not random either — there are coherent regions.
- Top-k similarity scores (Figure 2) cluster very tightly: many candidates
  look nearly equally compatible, so ranking is unstable.
- Two successful retrieval examples (Figures 3–4) and two failures
  (Figures 5–6) are visualized as three-panel (haptic query / retrieved
  visual prior / GT visual prior) plots. *Even the failures share the
  dominant class in the visual-prior distribution* — the alignment is
  semantically reasonable, just not instance-precise.
- 100-epoch training significantly drops train loss (to 2.6487) but does
  **not** proportionally improve retrieval, which is itself an interesting
  observation about loss-vs-retrieval mismatch.

The PDF also answers why cross-entropy is the standard contrastive loss
(InfoNCE / simultaneous attract+repel / efficient in-batch negatives /
stable softmax gradients).

**Net conclusion:** the model has learned *coarse semantic alignment* between
heterogeneous modalities (time-series ↔ probability vectors), but precise
instance-level matching is hard — primarily because of material-property
overlap and modality compression in the visual prior.

## Reflection (Problem 7)

Most interesting concept: the *interaction between fusion and alignment*.
Concretely — Tensor Fusion / LMF beat the baselines under supervised
classification, but contrastive learning showed that lower training loss
does not automatically translate to strong instance-level alignment. That
makes the "align before fuse" intuition concrete rather than abstract.

Most useful concepts for the project ahead: heterogeneity-aware fusion
design, einsum-based interaction modelling, contrastive alignment as a
complementary objective to supervised classification, and the accuracy-vs-
parameter trade-off (Tensor Fusion vs. LMF) for scalable deployment.

## Files

| File | What it is |
|---|---|
| `yirantao_hw2.ipynb` | Full notebook: tensor warmups, AV-MNIST unimodal training, all four CLAMP fusion implementations, and the contrastive-alignment experiment with the heatmap + per-sample visualizations. |
| `yirantao_hw2.pdf` | Written report with the 7-part discussion, all six figures, hyperparameters, and the comparison tables. **Read the PDF first**; use the notebook for the underlying code. |

## How HW2 connects to the rest of the portfolio

HW2 closes out the "classical multimodal" arc that started in HW1. From
HW3 onwards the portfolio pivots to *foundation-model-driven* multimodal
work (VLMs, fine-tuning, RL, agents), so HW2's main role is to ground the
"raw fusion + alignment" trade-offs that the later FM-based work
takes for granted.
