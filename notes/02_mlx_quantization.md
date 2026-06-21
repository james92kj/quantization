# Phase 2 — Local quantization with MLX (the measure-don't-trust loop)

**Goal:** take a real model, shrink it to 4-bit on Apple Silicon, and *measure* both sides
of the trade — the size/speed win and the quality cost. This is the naive baseline of
quantization; Phases 3–4 exist to beat it.

Work horse: **Qwen2.5-0.5B-Instruct**, on an Apple M4 (16 GB), via `mlx-lm`.

---

## What we did
```bash
# 1. Quantize fp16 -> 4-bit, group-wise affine, group_size 64
python -m mlx_lm convert --model Qwen/Qwen2.5-0.5B-Instruct -q --q-bits 4 --q-group-size 64 \
    --mlx-path ./qwen2.5-0.5b-4bit-g64
# -> "Quantized model with 4.502 bits per weight."

# 2. Perplexity, SAME harness for both (01_local_mlx/mlx_perplexity.py)
python mlx_perplexity.py --model Qwen/Qwen2.5-0.5B-Instruct     # fp16
python mlx_perplexity.py --model ./qwen2.5-0.5b-4bit-g64        # 4-bit

# 3. Side-by-side generation
python -m mlx_lm generate --model <fp16 | 4-bit> --prompt "..." --max-tokens 80
```

## Results (real, measured on M4)

| | fp16 | 4-bit (g64) | change |
|---|---|---|---|
| bits / weight | 16 | **4.502** | — |
| size on disk | 953 MB | **290 MB** | **3.3× smaller** |
| WikiText-2 perplexity (MLX ruler) | 14.26 | **17.09** | **+2.83 (+19.8%)** |
| peak memory (generation) | 1.057 GB | **0.323 GB** | **3.3× less** |
| generation speed | 90 tok/s | **228 tok/s** | **2.5× faster** |

---

## The five lessons

### 1. Bits-per-weight isn't the bit count you asked for (4 → 4.502)
A group of 64 weights stores `64 × 4 = 256` bits of values **plus** a shared scale + zero-point
(2 × fp16 = 32 bits). `(256 + 32) / 64 = 4.5` bits/weight. The extra 0.5 is the metadata
overhead — exactly the scale/zero-point cost, made concrete.

### 2. Group-wise quantization = "one scale per small block, so outliers can't crush everyone"
Per-tensor (one scale for the whole matrix) lets a single outlier weight stretch the scale and
round all the small weights to zero. Splitting each row into groups of 64, each with its own
scale, gives every region local resolution. `group_size` is the dial: smaller = more accurate,
more overhead. 64 is the standard sweet spot. (See [[01_perplexity]] for the metric.)

### 3. Naive 4-bit on a *small* model costs real quality (+20% perplexity)
`mlx_lm convert` is plain **round-to-nearest** — it rounds each weight to the closest level and
looks at nothing else (no calibration, no error compensation). Two reasons the loss is large
here: (a) 0.5B models have little redundancy, so every weight matters — **model size is the #1
factor in quantization robustness**; a 7B+ model typically loses only 1–3% at 4-bit. (b) it's the
naive method. That ~20% gap is the prize Phase 3 (calibration) and Phase 4 (GPTQ/AWQ) win back.

### 4. Perplexity ≠ generation quality (they can disagree)
Perplexity rose ~20%, yet the 4-bit model's answer to "explain quantization" was coherent and
arguably more precise. Perplexity is a *sensitive average over 300k tokens*; the lost "sharpness"
shows up on hard tasks (reasoning, math, rare facts), not easy prompts. → judge with task
benchmarks (Phase 5), and never over-read a single greedy sample.

### 5. The real wins are speed and memory
3.3× less RAM and 2.5× faster generation — because Apple Silicon inference is memory-bandwidth
bound, and 4-bit weights are ~3.5× fewer bytes to move per token. This is why a 7B model that
won't fit in 16 GB at fp16 runs comfortably at 4-bit.

---

## The loop you just learned (reused for every later method)
`baseline → quantize → measure (size + perplexity + generation) → judge the trade.`
Only step 2 changes in Phases 3–6. Verdict for a 0.5B run locally: **worth it** — 3.3× smaller,
2.5× faster, still coherent — and we now know the exact quality cost the smarter methods must beat.
