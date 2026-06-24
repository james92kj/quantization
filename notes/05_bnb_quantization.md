# Phase 4a — bitsandbytes (LLM.int8 + NF4) on CUDA

First Track-B phase: real GPU (NVIDIA **L4 24 GB**, GCP `quant-l4`), the production 4-bit/8-bit
methods you actually ship. Model **Qwen2.5-1.5B-Instruct** (same as Phase 3, for continuity).

> **Ruler:** HF sliding-window perplexity (`06_evaluation/perplexity.py`, WikiText-2-raw, win 2048 /
> stride 1024). This is a **different ruler** from Phase 3's llama.cpp number (~10.4) — do NOT
> cross-compare. Everything below is internally comparable: same model, same ruler, only the
> quantization changes. Script: [`03_cuda_bnb/bnb_eval.py`](../03_cuda_bnb/bnb_eval.py).

## Plain-English primer (read this first)

**The one fact everything rests on.** A model is just a big pile of numbers ("parameters" /
"weights"). This model has **1.544 billion** of them. Quantization is one simple idea: **store each
number using fewer bits, to save space.** Fewer bits = smaller = less GPU memory, but also less
precise → the model gets slightly "dumber." Every row in the results table is the *same* 1.544B
numbers, stored at different precision.

**How much space each number takes.** Memory is in bytes; 1 byte = 8 bits.

| precision | bits/number | bytes/number | 1.544B numbers take… |
|---|---|---|---|
| fp16 | 16 | 2 | 1.544B × 2 = **3.088 GB** ✅ |
| int8 | 8 | 1 | ~1.5 GB (measured 1.845; rest is overhead) |
| nf4  | 4 | 0.5 | ~0.77 GB (measured 1.153; rest is overhead) |

Halve the bits → roughly halve the memory. `fp16 → int8 → nf4` is just `16 → 8 → 4` bits/number.
"Overhead" = small bookkeeping data (the scales) the library stores alongside the numbers.

**The "0.889 B params artifact" = a display bug, not a model problem.** The script asked PyTorch
"how many numbers?" For 4-bit, the library crams **two 4-bit numbers into one 1-byte slot** (8 bits =
two 4-bit values, a perfect fit). PyTorch counted the *slots* (≈0.889B) instead of the actual
*numbers* (1.544B), so the `params` and `bytes/param` columns came out wrong. The **memory** number
(1.153 GB) was always correct. Script now doubles the packed count → honest 0.747 bytes/param.

**What PPL / "+7.66%" means.** PPL = perplexity = a quality score, **lower is better** ("how confused
the model is predicting text"). fp16 = 8.6453 is the gold standard. int8 = 8.6869 → only 0.48% worse
(≈identical). nf4 = 9.3078 → 7.66% worse (noticeably dumber, still works). **The whole lesson:** more
compression saves more space but costs more quality. 8-bit = safe & gentle; 4-bit = aggressive, real
savings, real cost.

## The idea (vs Phase 3)

bitsandbytes is **calibration-free**: it derives quant ranges straight from the weights at load time
and dequantizes on the fly inside each matmul. No importance matrix, no calibration set. The "just
works" entry to GPU quantization — and the basis of **QLoRA**. All controlled by one object,
`BitsAndBytesConfig`.

## Results — Qwen2.5-1.5B-Instruct (HF ruler)

| mode | weights VRAM | vs fp16 | bytes/param* | peak VRAM (eval) | WikiText-2 PPL | Δ PPL | eval time |
|------|-------------|---------|-------------|------------------|----------------|-------|-----------|
| **fp16** (baseline) | 3.088 GB | 1.0× | 2.00 | 6.96 GB | **8.6453** | — | 62 s |
| **int8** (`LLM.int8()`) | 1.845 GB | 1.67× | 1.20 | 5.72 GB | 8.6869 | **+0.48%** | 96 s |
| **nf4** (NF4 + double-quant) | 1.153 GB | 2.68× | 0.747 | 5.03 GB | 9.3078 | **+7.66%** | 71 s |

\* bytes/param = weights VRAM ÷ **true** 1.544 B params. NOTE: bnb packs two 4-bit weights per
`uint8`, so `tensor.numel()` reports *half* for nf4 layers — the raw script printed a misleading
"0.889 B params / 1.30 bytes/param" for nf4. The honest 0.747 (~6 effective bits: 4-bit linears +
fp16 embedding/lm_head + scale overhead) uses the real param count. Script was fixed to count packed
params ×2.

## What to read

1. **int8 quality: +0.48% — near-lossless.** `LLM.int8()` keeps a fp16 side-path for the rare
   outlier columns, so the dangerous dimensions never get quantized. For 8-bit this tiny a hit is
   exactly the expected result. (Full method → open question **Q3**.)
2. **int8 memory: 1.67× smaller, not the naive 2×.** "8-bit" isn't a clean 1 byte/param: actual
   1.20 includes per-row fp16 scales **plus** the fp16 outlier columns. Overhead is real — budget
   for it.
3. **⚠️ int8 is SLOWER, not faster (96 s vs 62 s).** The counterintuitive lesson: **bitsandbytes
   int8 buys memory, not speed.** The decompose-outliers → two-path matmul → recombine has overhead
   that, on a small model, outweighs the int8 compute win. "Fewer bits" ≠ "faster." GPTQ/AWQ (4b/4c)
   use fused kernels and *can* speed up — we'll measure and contrast.
4. **nf4 memory: 2.68× smaller — the headline win, and why QLoRA exists** (a 4-bit base frees VRAM
   to fine-tune LoRA adapters on top).
5. **nf4 quality: +7.66% — the real cost of *calibration-free* 4-bit.** This is the motivation for
   4b/4c: GPTQ/AWQ spend a calibration pass to claw most of this back **at the same 4 bits** — the
   Phase-3 lesson, now on GPU.
6. **Bigger models quantize more gracefully** (recurring theme): MLX naive 4-bit cost **+19.8%** on a
   0.5B (Phase 2); NF4 here costs **+7.66%** on a 1.5B. Expect the gap to shrink further at 3B/7B.
7. **The 8-bit-vs-4-bit decision, in one line:** int8 = +0.48% / 1.85 GB vs nf4 = +7.66% / 1.15 GB.
   ~0.7 GB more saved costs ~7% quality → pick int8 when quality-critical, nf4 when memory-bound
   (and especially for QLoRA, where adapter fine-tuning recovers quality).

## Open questions spawned here
- **Q2** — "Sliding Window Attention enabled but not implemented for `sdpa`" warning (benign for our
  2048-tok eval, but understand the backends).
- **Q3** — what *is* `LLM.int8()`, completely (outliers, mixed-precision decomposition, why lossless,
  why slow). Paper: arXiv 2208.07339.

_(See [`open_questions.md`](open_questions.md).)_

## Reproduce
```bash
# on the GPU VM (L4), after: pip install bitsandbytes 'transformers==4.49.0' accelerate datasets
python3 bnb_eval.py --mode fp16
python3 bnb_eval.py --mode int8
python3 bnb_eval.py --mode nf4
```
