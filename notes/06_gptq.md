# Phase 4b — GPTQ (error-compensating 4-bit) on CUDA

Model **Qwen2.5-1.5B-Instruct**, HF ruler (same as 4a — comparable to the bnb rows in
[`05_bnb_quantization.md`](05_bnb_quantization.md), NOT to the llama.cpp rows). fp16 anchor = 8.6453.
Script: [`04_cuda_gptq_awq/gptq_quantize.py`](../04_cuda_gptq_awq/gptq_quantize.py).

## The idea — GPTQ vs NF4

NF4 (Phase 4a) rounded **every weight independently**, blind to the rest. **GPTQ** quantizes a layer
one weight at a time and, after rounding each weight, **adjusts the remaining not-yet-quantized
weights to cancel the error it just introduced**. To know *how* to adjust, it estimates which weights
matter from a small **calibration set** of real text (second-order / Hessian info). Same 4 bits, but
the roundings are chosen to minimize the layer's **output** error — not just per-weight rounding error.

This is the Phase-3 imatrix idea taken further: imatrix *measured* importance; **GPTQ measures
importance AND compensates.** (AWQ, 4c, will protect salient channels by rescaling — a third take on
the same outlier/importance theme.)

## Library note — why llm-compressor, not GPTQModel

`GPTQModel` (the usual lib) is unbuildable on our DLVM: its `setup.py` imports the broken `pcre`
package at build time, and there's no `nvcc` to compile its CUDA kernels anyway. We use
**llm-compressor** (pure-PyTorch GPTQ, same algorithm). See [`SETUP_CUDA.md`](../SETUP_CUDA.md).
Scheme **W4A16** = 4-bit weights / 16-bit activations, group size 128, `lm_head` left in fp16.

## ⚠️ The big lesson: calibration DATA quality decides everything

**Methodology rule:** calibrate on WikiText-2 **train**, evaluate on **test**. Never calibrate on the
eval set — it leaks and fakes a good score.

**First attempt FAILED — and the failure was the lesson.** I built the calibration set by tokenizing
WikiText *rows* with `truncation=2048`. But WikiText rows are short lines:

```
calib sample token-lengths: min 49, median 137, mean 158, max 559
reaching the 2048 target:  0 / 256      under 256 tokens: 226 / 256
```

So GPTQ estimated its Hessian from 256 **stubs** (~158 tokens), compensated in the wrong direction,
and produced **PPL 11.9491 (+38% vs fp16) — WORSE than calibration-free NF4 (9.3078).** A calibrated
method losing to an uncalibrated one is a red flag: it meant our *setup* was wrong, not that GPTQ is
bad. **Takeaway: a surprising number is almost always a setup bug, not a law of nature — dig, don't
record it.**

**The fix** (standard GPTQ recipe): concatenate the whole corpus, then slice it into real 2048-token
chunks — exactly how the perplexity harness chunks the test set. Changed *only this one variable*
(left `actorder` off) so any improvement cleanly attributes to fixing the calibration.

## Results — Qwen2.5-1.5B-Instruct (HF ruler)

| 4-bit method | calibration | actorder | PPL | Δ vs fp16 (8.6453) |
|---|---|---|---|---|
| NF4 (bnb) | none | — | 9.3078 | +7.66% |
| GPTQ — bad calib (158-tok stubs) | broken | off | 11.9491 | +38.2% 🚩 |
| GPTQ — proper calib (2048-tok) | 256 × 2048 | off | 9.4401 | +9.19% |
| **GPTQ — proper calib + act-order** | 256 × 2048 | **on** | **9.0716** | **+4.93%** ✅ |

**The full arc — each lever isolated (one variable at a time):**
1. **Broken calibration → 11.95** (worse than no-calibration NF4). Stubby 158-tok sequences ruin the
   Hessian. *Calibration data quality is everything.*
2. **Fix calibration (2048-tok chunks) → 9.44** (ties NF4). The fix recovered 2.5 PPL.
3. **Add act-order → 9.07 — GPTQ now BEATS NF4** (+4.93% vs +7.66%). Quantizing important columns
   first, with the rest free to compensate their error, is what pulls GPTQ ahead.

**Honest surprise (cuts the other way):** act-order bought **0.37 PPL** here — far more than the ~0.1
the GPTQ authors saw on their 13B. On *this* small model act-order mattered MORE, not less, overshooting
the informed prediction. (Real data > priors — exactly why we ran it.) Cost: eval 122 s vs 91 s — the
column permutation slows inference, the documented act-order tradeoff, confirmed.

**Conclusion:** calibration-free NF4 is a strong, zero-effort 4-bit baseline; GPTQ done *properly*
(real calibration + act-order) beats it at the same bit-width — the payoff of spending compute to
choose roundings that minimize output error.

_(Runtime note: eval VRAM was 9.3 GB because, without fast kernels, compressed-tensors expands the
4-bit weights to fp16 for the forward pass. The storage win is the 1.625 GB on-disk, not eval VRAM.)_

## What is activation ordering (`actorder` / `desc_act`)?

GPTQ quantizes a weight matrix **one column at a time**, compensating the remaining columns after
each. So **order matters**: a column quantized *early* has the whole rest of the matrix to absorb its
error; a column quantized *last* has nothing left to compensate it.

**Act-order** = process columns in order of **decreasing importance** (importance = activation
magnitude through that column). Important columns go **first** (max compensation room left); unimportant
ones go last (little room, but they barely matter). Without it, an important column might be quantized
last and its error gets stuck → worse quality. Cost: reordering breaks the group layout → slightly
slower inference (the only reason to skip it).

*Analogy:* packing a suitcase where unplaced items cushion placed ones — pack the wine glass first so
everything cushions it; pack socks last.

## Literature check (verifying "GPTQ ≈ NF4 at 4-bit; wins at 3-bit / big models / act-order")

Investigated against the GPTQ paper + repo + benchmarks. **Mostly confirmed, one correction:**
- ✅ **GPTQ ≈ NF4 at 4-bit, NF4 often slightly ahead** — 4-bit formats land within ~6% of fp16; NF4's
  nonuniform grid gives it a ~0.5 PPL edge over uniform int4. Our NF4 9.31 vs GPTQ 9.44 fits this band.
- ✅ **GPTQ wins grow at 3-bit / larger models** — 4-bit ≈ 2–8% degradation vs 3-bit ≈ 8–15%; a 175B
  loses only 0.03 PPL at 4-bit.
- ⚠️ **act-order is MORE model-dependent than first stated.** GPTQ authors' numbers: OPT-66B 4-bit
  9.55→9.34 and 3-bit 14.16→9.95 (huge); LLaMA-7B 7.15→6.09 (big); 13B only ~0.1 PPL (modest). So on
  our **1.5B at 4-bit, act-order may buy only ~0.1** — possibly NOT enough to overtake NF4's 0.13 lead.
  The actorder experiment is genuinely uncertain → worth running to find out.

Sources: GPTQ paper arXiv 2210.17323 · github.com/IST-DASLab/gptq README · oobabooga quant blog ·
theaiengineer Qwen2.5 quant benchmark.

## Knobs to explore later
- `actorder` (a.k.a. `desc_act`) — quantize columns in order of activation importance; usually helps.
- `group_size` (128 here) — finer groups = better quality, more scale overhead.
- `dampening_frac` (0.01) — numerical stability of the Hessian solve.
- More / different calibration data (C4 vs WikiText; in-domain vs out-of-domain — the Phase-3 lesson).
