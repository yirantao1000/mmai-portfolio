# HW3 — Vision-Language Models: Prompting & LoRA Fine-Tuning

Submission for **MAS.S60 / 6.S985 — Modeling: Multimodal AI, Spring 2026 (MIT)**.

First "foundation-model" assignment of the semester: take a pretrained
vision-language model (VLM), measure how well it works zero-shot on a
fine-grained visual task, then try to improve it with (a) prompt engineering
and (b) **LoRA fine-tuning**. The written report `yirantao_hw3.pdf` covers
the full Parts 1–7; this README is the index and the TL;DR of results.

## Concept primer (Part 1, reading reflection)

Five essay questions on the multimodal-LLM literature:

1. **Multimodal data noise** — typology (mismatched pairs, partial mismatch,
   missing/corrupted modalities, low-information pairs, label noise, social
   bias), how it hurts (false negatives/positives, shortcut learning, biased
   fusion), and how to build estimators (CLIP-style alignment scoring +
   informativeness + source quality + ensemble agreement). My "unlimited
   budget" pipeline: dedup → multi-model semantic scoring → active human
   verification on ambiguous samples → rebalance → bias audit → provenance
   metadata.
2. **Frozen LLM backbones for multimodal tasks** — the Frozen paper's
   intuition (preserve LLM priors, learn a small adapter that maps non-text
   features into the LLM's embedding space), the encoder families that fit
   into it (CNN/ViT for images, spectrogram encoders for audio,
   spatiotemporal for video), and how the LLM processes the result via
   prefix/adapter/prompt tuning rather than literally "seeing pixels".
3. **Instruction-tuning data selection for multimodal models** — extends the
   gradient-similarity / low-rank influence approach from the reading with
   three multimodal-specific layers: enforce multimodal grounding (no
   text-only shortcuts), filter by cross-modal alignment quality, balance
   high-influence samples against diversity.
4. **Detecting AI-generated content beyond watermarking** — three families
   (forensic detection, provenance / C2PA Content Credentials, contextual
   verification), modality-specific tells (lighting/freq for images,
   spectral / prosody / vocoder artifacts for audio, stylometric weakness
   for text), and why provenance is currently the most durable practical
   tool despite the cat-and-mouse problem in pure detection.
5. **Does Sora "understand" the world?** — basically agree with LeCun.
   Generation ≠ causal prediction; Sora's plausibility is not evidence of a
   robust world model. JEPA/V-JEPA framed as a complementary path that
   predicts in representation space. Final stance: generation is a useful
   *component* of AGI but unlikely to be sufficient on its own.

## Dataset (Part 2)

[`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans)
— fine-grained 3-class bean-leaf disease classification (`angular leaf spot`,
`bean rust`, `healthy`).

- Used the **official train split** for training and the **official val + test
  splits** as the held-out evaluation set — no leakage, reproducible, aligned
  with the standard benchmark protocol.
- Discussed (in the PDF) the failure modes of converting non-image data into
  images (artificial spatial structure, info loss, encoding inconsistency,
  low-level texture artifacts), and conversely the failure modes of converting
  images into text/audio (loss of fine-grained visual evidence — spot shape,
  edge sharpness, color distribution, vein patterns).

## Baseline VLM zero-shot (Part 3)

Strict label-only prompt + deterministic decoding (fixed allowed-label set,
no sampling, constrained gen length). Result:

- Coarse-grained classification works (the `healthy` sample is classified
  correctly).
- **Fine-grained discrimination fails**: every diseased sample was collapsed
  into `angular leaf spot`. The model behaves like a binary healthy-vs-sick
  detector and uses a shortcut (`unhealthy leaf → angular leaf spot`) rather
  than reasoning about lesion shape / distribution / texture.

Diagnosis: classic failure mode when a generic VLM is dropped onto a
specialised domain task with no adaptation.

## Prompt engineering (Part 4)

Three prompting strategies compared on the same held-out images:

| Prompt | Description |
|---|---|
| `format_constrained` | The Part 3 baseline — single label from the fixed allowed set, no extra text. |
| `reason_then_answer` | Symptom-level guidance (lesion shape, vein-limited patterns, rust-like pustules) + structured `Reason: ... / Final: <label>` output. |
| `few_shot_image_qa` | Two image-label support examples (`train_00263.jpg → angular leaf spot`, `train_00590.jpg → bean rust`) prepended in-context. |

**Result: none of the prompt changes improved exact-match accuracy on this
task.** `reason_then_answer` and `few_shot_image_qa` actually scored *lower*
than the format-constrained baseline.

Discussion in the PDF: the model often produced *fluent reasoning that didn't
match its final answer*, i.e. prompt-based reasoning improved interpretability
but not decision quality. Likely causes — added instructions increase
cognitive load on the visual classification task; two support examples on a
fine-grained problem invite superficial pattern matching rather than learning
the actual class boundary.

## LoRA fine-tuning (Part 5)

Best configuration found:

| Hyperparameter | Value |
|---|---|
| `NUM_EPOCHS` | 6 |
| `LR` | 1e-4 |
| `BSZ_PER_DEV` | 16 |
| `GRAD_ACCUM` | 2 |
| `EVAL_SPLIT` | 0.1 |
| `SEED` | 42 |
| `MAX_SEQ_LEN` | 256 |
| `SHORTEST_EDGE` | 224 |
| `LORA_R` | 4 |
| `LORA_ALPHA` | 8 |
| `LORA_DROPOUT` | 0.05 |
| `LORA_TARGET` | `["q_proj", "k_proj", "v_proj", "o_proj"]` |

**Most impactful knobs**, in order of observed effect:
1. **Image resolution** (`SHORTEST_EDGE`) — determines whether subtle lesion
   patterns survive preprocessing.
2. **Learning rate** — too high overwrites pretrained knowledge, too low
   underfits the small dataset.
3. **Training intensity** (`NUM_EPOCHS` + effective batch size from
   `BSZ_PER_DEV × GRAD_ACCUM`).
4. LoRA capacity (`LORA_R`, `LORA_ALPHA`) — secondary but still meaningful.

## Post-training evaluation (Part 6)

Re-ran the held-out images:

| Model | Exact-match accuracy |
|---|---|
| Pretrained VLM (zero-shot, baseline prompt) | 50% |
| **LoRA fine-tuned VLM** | **100%** |

The headline correction: the `bean rust → angular leaf spot` shortcut from the
baseline was eliminated. No new errors introduced on this evaluation set.
LoRA gave the model just enough capacity to learn the fine-grained class
boundary that prompting alone could not move.

## Reflection (Part 7)

- **Most interesting concept**: parameter-efficient fine-tuning (LoRA) — that
  a small number of trainable parameters can flip a 50% model into a 100%
  model on a domain-specific task while keeping the base model frozen.
- **Most useful concept going forward**: LoRA-based fine-tuning, full stop.
  Prompt engineering turned out to be a *diagnostic tool* in this setting,
  not a performance lever.
- **Wish-list for future homework**: more experiments with the "frozen LLM +
  modality encoder + adapter" architectures discussed in lecture (Flamingo /
  MiniGPT-4 / LLaMA-Adapter style), and direct comparisons of how
  instruction-tuning data quality and selection affect downstream
  performance.

## Files

| File | What it is |
|---|---|
| `yirantao_hw3.ipynb` | Full notebook: GPU/library setup, dataset loading, baseline zero-shot run, all three prompt-engineering variants, the LoRA fine-tuning loop, and the post-training evaluation. |
| `yirantao_hw3.pdf` | Written report with the 7-part discussion, prompt examples, hyperparameter tables, and per-image evaluation results. **The PDF is the primary writeup**; the notebook is the underlying code. |

## How HW3 connects to the rest of the portfolio

HW3 is the *SFT half* of the "VLM adaptation" arc. HW4 will redo
the same beans dataset with **GRPO RL** instead of SFT and explicitly
compare the two — so the LoRA numbers from this HW3 are the supervised
baseline that the HW4 GRPO model is measured against.
