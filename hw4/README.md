# HW4 — GRPO Reinforcement Learning for Vision-Language Models

Submission for **MAS.S60 / 6.S985 — Modeling: Multimodal AI, Spring 2026 (MIT)**.

This assignment is the **RL counterpart to HW3**: take the same beans
disease-classification VLM setup, but instead of supervised fine-tuning with
LoRA, train it with **Group Relative Policy Optimization (GRPO)** — the
DeepSeekMath-style RL algorithm that drops the critic and computes the
baseline from group statistics over sampled completions. The written report
`yirantao_hw4.pdf` covers Parts 1–8; this README is the index and the TL;DR
of results.

## Concept primer (Part 1, reading reflection)

Three deep-dive essays on the RL fine-tuning literature for LLMs/VLMs:

1. **GRPO vs PPO** — How GRPO removes the learned value function: for each
   prompt sample a group of `G` completions from the old policy, score them
   with a reward function, then use the **group mean reward as the baseline**
   and normalize by the **group standard deviation**. Each completion's
   advantage is therefore `A_i = (r_i - r̄) / (σ_r + ε)` — purely *relative*
   ranking within the group. **Trade-offs** discussed: more memory-efficient
   (no critic the size of the policy), but training quality now depends
   entirely on (a) reward diversity within the group, (b) sampling cost
   from `G` completions per prompt, and (c) reward function design — there
   is no learned token-aware baseline to stabilise things.

2. **Reward design: learned RM vs rule-based reward** — Learned RM is good
   for subjective/nuanced behaviours (helpfulness, style, reasoning quality)
   but vulnerable to *model exploitation* (the policy learns to game the RM
   instead of solving the task; this is why KL regularisation matters).
   Rule-based reward is good for verifiable tasks (math regex match,
   sandboxed code tests) but vulnerable to *specification gaming* (gaming
   answer extraction formats, overfitting to visible tests, satisfying
   format reward without improving correctness). Neither is "safe by
   default".

3. **SFT vs GRPO** — SFT (what I did in HW3) gives a dense **token-level
   cross-entropy** signal against a gold target sequence; GRPO gives a
   **scalar reward** per sampled output and learns from *relative* group
   ranking. Prefer SFT when you have clean labels, a well-defined target,
   and want stable cheap training. Prefer GRPO when you care about
   reasoning/behaviour that is hard to teach by imitation, you have a
   reward signal but not one gold sequence, or you want to optimise for
   *final* task performance. **In practice they compose**: DeepSeekMath
   does SFT *then* GRPO, and this homework follows the same pattern.

## Part 2 — GRPO implementation

Hands-on implementation walking through the GRPO objective. Key conceptual
checkpoints surfaced in the PDF:

- **Why group-based advantage normalization works as a baseline** — policy
  optimization mainly depends on *relative* ranking, not absolute reward
  values, so as long as the model can tell better-than-average from
  worse-than-average completions for the same prompt, it can improve
  without a critic. **Failure mode**: if all `G` completions for a prompt
  get the same reward, every advantage is 0, no gradient signal, no
  update. GRPO depends on reward diversity within the group; weak reward
  design or low sampling diversity stalls training.
- **The clipping term `clip(ρ_{i,t}, 1-ε, 1+ε)`** — inherited from PPO,
  enforces a trust region so updates stay bounded. Without it: unstable
  training / divergence, overfitting to a handful of high-reward samples,
  policy collapse to narrow repetitive behaviours, and degradation of the
  base model's pretrained capabilities. Particularly important under
  GRPO precisely *because* there is no critic to stabilise training.

## GRPO training on beans (Problem 7)

Best hyperparameters:

| Hyperparameter | Value |
|---|---|
| `num_generations` | 8 |
| `max_completion_length` | 256 |
| `learning_rate` | 1e-5 |
| `max_steps` | 100 |
| `epsilon` | 0.2 |
| `temperature` | 0.6 |
| `beta` | 0.0 |

Held-out evaluation (8 bean-leaf samples):

| Metric | Result |
|---|---|
| **Accuracy** | **75.0%** |
| **Format compliance** (`Answer:` structure used) | **100.0%** |

**Reward dynamics**: the **format reward stabilised quickly** — the model
learnt the required `Answer: <label>` structure early. The **accuracy reward
improved more gradually and stayed noisier**, because the hard part of the
task is not the output format, it is distinguishing `bean rust` from
`angular leaf spot` visually.

**Most impactful hyperparameters in my experiments**:

1. `num_generations` — bigger groups give a more informative relative-reward
   signal and reduce the tendency to collapse onto a single unhealthy label.
2. `max_completion_length` — too small and the output gets truncated before
   `Answer:`, which kills both the format reward and the accuracy reward.
3. `temperature` — lower temperature reduces completion variability and
   makes the final label more reliable on visually-similar classes.

Default values for `epsilon`, `beta`, and the LoRA params worked fine; the
PDF documents what role each *should* play in general, but for this
specific 3-class visual task they had a smaller effect than the three knobs
above.

## GRPO vs SFT/LoRA (HW3 vs HW4) comparison

| Aspect | HW3 SFT/LoRA | HW4 GRPO |
|---|---|---|
| Training signal | Dense token-level CE on gold answer | Sparse scalar reward over sampled completions |
| Convergence speed | Fast and stable | Slow and noisy |
| Best held-out accuracy on beans | **100%** (4 samples) | 75.0% (8 samples) |
| Output-format compliance | Not enforced | **100%** with explicit reasoning |
| Best for | Pure supervised classification | Shaping *how* the model responds (structured reasoning + final label) |

**Bottom line for this dataset**: SFT/LoRA gave *better classification
accuracy*; GRPO gave *better behavioural alignment* (explicit step-by-step
reasoning followed by a clean `Answer:` line, every time). This is the
expected result — beans is a small fine-grained classification problem
where SFT is a natural fit, while GRPO is built for the harder objective of
shaping behaviour beyond what gold-label imitation can teach.

## Reflection (Problem 8.3)

- Compared with the **zero-shot base model** and the **HW3 SFT model**, the
  GRPO-trained model used the `Answer:` format much more consistently
  (100% vs sometimes / rarely).
- It also produced **explicit reasoning before the answer** much more
  reliably — every held-out example had reasoning then a final `Answer:`
  line.
- The base model was the weakest overall; the HW3 SFT model still wins on
  raw classification accuracy on this dataset; the HW4 GRPO model wins on
  *structured-reasoning behaviour* — which is exactly the axis GRPO was
  configured to reward.

## Files

| File | What it is |
|---|---|
| `yirantao_hw4.ipynb` | Full notebook: GPU/library setup, dataset preparation, the GRPO implementation walkthrough, the training loop, and the held-out evaluation comparison with the HW3 SFT baseline. |
| `yirantao_hw4.pdf` | Written report with Parts 1–8 (reading reflection, GRPO understanding, hyperparameter analysis, GRPO-vs-SFT comparison, and the reflection). **The PDF is the primary writeup**; the notebook is the underlying code. |

## How HW4 connects to the rest of the portfolio

HW4 closes the "VLM adaptation" arc (HW3 SFT/LoRA → HW4 GRPO RL) on the
exact same dataset, giving a clean apples-to-apples comparison of dense
supervised fine-tuning vs. reward-based RL on a fine-grained
classification task. From HW5 the portfolio pivots again — this time to
**agents**, where the same underlying LLM has to call tools and act in a
multi-step environment, rather than being fine-tuned on a static dataset.
