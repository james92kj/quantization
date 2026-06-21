# Practical LLM Quantization — Learning by Doing

Take a real language model, **shrink it** (quantize it), and **prove the smaller version
still answers well**. This repo is built **one phase at a time** — each phase is a real
experiment on a real model with real numbers, not toy snippets.

The headline we're chasing: **3× smaller, basically the same quality** — shown with a
hard metric and side-by-side answers.

---

## What is quantization (in one paragraph)

A model stores its weights as 16-bit numbers (fp16). Quantization replaces them with
low-bit integers — **int8** (8-bit) or **int4** (4-bit) — plus a tiny bit of float
metadata (a *scale*, sometimes a *zero-point*) shared across a small block of weights.
Fewer bits per weight → **smaller file, less memory, faster** — at the cost of some
rounding error. The whole craft is keeping that error small enough that the model's
*answers don't change*. This repo measures exactly that trade-off.

---

## What is perplexity (the metric we use) — the intuition, made clear

Perplexity answers one question: **when the model reads real text, how surprised is it by
each next word?** Lower = less surprised = better.

The clean mental image: **perplexity = the average number of words the model is
effectively torn between at each step.**

**Why "number of choices"?** Think of equally-likely guessing games:
- A fair **coin** → 2 outcomes, each has probability `1/2`. Notice `1 ÷ (1/2) = 2`.
- A fair **die** → 6 outcomes, each `1/6`, and `1 ÷ (1/6) = 6`.

So for *equally-likely* options, `1 ÷ probability` recovers the **number of options**.
A model's situation isn't equally likely, but we can still ask: *"the model gave the
correct next word probability `p` — a fair die with how many sides would feel this
uncertain?"* Answer: **`1/p`**.

| model gave the true word | effective choices (`1/p`) | meaning |
|---|---|---|
| `p = 0.9` | ~1.1 | confident & correct — **great** |
| `p = 0.25` | 4 | torn between ~4 words — meh |
| `p = 0.02` | 50 | flailing among ~50 words — **bad** |

A model's probability is a **fixed budget of 1.0**. Every bit it spends on wrong words is
stolen from the right one. A good model concentrates its budget on the actual next word
(high `p` → small `1/p` → low perplexity). A bad model spreads it thin (low `p` on the
truth → high perplexity).

**Combining across all the words** uses the *geometric* mean, because choices compound by
**multiplying** (like a lock: 2 options on the first dial × 8 on the second = 16 combos,
and the typical per-step factor is `√16 = 4`, not the average `5`). That product-then-root
is exactly what `exp(average of the logs)` computes — which is the formula:

```
perplexity = exp( mean over tokens of ( -ln p_true_next_token ) )
```

Read right-to-left: take the probability of each true next word → `-ln` turns it into
"surprise" (= log of the number of choices) → average → `exp` turns it back into a plain
"number of choices."

**For quantization:** rounding the weights makes the model a little more surprised, so
perplexity rises a little. The size of that rise is our first, cheapest measure of quality
lost. 8-bit ≈ no change; good 4-bit ≈ +1–5%; a broken quant explodes (or goes NaN). We
**never** judge perplexity in absolute terms — only *same text, two models, lower wins*.

> Full walkthrough with worked examples: [`notes/01_perplexity.md`](notes/01_perplexity.md).
> All four quality metrics (perplexity, KL-divergence, generation, task accuracy):
> [`06_evaluation/HOW_TO_MEASURE_QUALITY.md`](06_evaluation/HOW_TO_MEASURE_QUALITY.md).

---

## Results so far

**Qwen2.5-0.5B-Instruct** (work horse), WikiText-2 perplexity:

| precision | size on disk | perplexity | vs fp16 |
|---|---|---|---|
| fp16 (baseline) | 953 MB | **12.67** | — |
| MLX 4-bit | 276 MB (**3.5× smaller**) | _Phase 2 (next)_ | |

Live scoreboard: [`notes/scoreboard.md`](notes/scoreboard.md).

---

## Roadmap (one phase at a time)

- [x] **Phase 1 — Baseline.** Measure the *original* fp16 model's quality. You can't judge
  a shrunk model without the reference number. → PPL 12.67.
- [ ] **Phase 2 — Local quantization (MLX / GGUF).** Make it ~3× smaller on Apple Silicon,
  measure again, compare side by side. ← next
- [ ] **Phase 3 — Calibration.** Why good 4-bit methods need a small "calibration" dataset.
- [ ] **Phase 4 — Production CUDA methods.** bitsandbytes (NF4), GPTQ, AWQ on a GPU.
- [ ] **Phase 5 — Rigorous evaluation.** KL-divergence, lm-eval task accuracy.
- [ ] **Phase 6 — From scratch.** Implement int4/int8 quant by hand to own it.

Full plan: [`00_roadmap/ROADMAP.md`](00_roadmap/ROADMAP.md).

---

## Run it yourself

```bash
pip install -U torch transformers datasets accelerate sentencepiece

# Phase 1 — measure the fp16 baseline (Apple Silicon: --device mps; CUDA: --device cuda)
cd 06_evaluation
python perplexity.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps
```

## Repo layout

```
00_roadmap/        the full learning plan
01_local_mlx/      Phase 2 — MLX (Apple-native) quantization
02_local_gguf/     Phase 2 — llama.cpp / GGUF
03_cuda_bnb/       Phase 4 — bitsandbytes (NF4 / int8)
04_cuda_gptq_awq/  Phase 4 — GPTQ & AWQ
05_calibration/    Phase 3 — calibration datasets
06_evaluation/     quality measurement harness (perplexity, KL, generation)
07_from_scratch/   Phase 6 — implement quantization by hand
notes/             plain-language concept notes + the scoreboard
```

*Hardware used: Apple M4 (16 GB) for local phases; a single cloud GPU for the CUDA phases.*
