# Phase 3 — Calibration, demonstrated (GGUF imatrix)

**Goal:** prove that *calibration* — showing the model real data so it quantizes intelligently —
recovers quality at **zero size cost**. We did the cleanest possible controlled experiment:
the same model, same quant scheme, same size, toggling only the importance matrix.

Work horse: **Qwen2.5-1.5B-Instruct** (hidden 1536 = 6×256, so K-quants don't fall back —
see [[03_gguf_output_and_imatrix_gotcha]] for why the 0.5B was unusable here).

---

## What calibration is
Naive round-to-nearest rounds every weight to its closest level, looking at **nothing**, treating
all weights as equally important. But some weights — the ones multiplied by large activations on
real data — matter far more. **Calibration runs a small sample of real text through the model,
measures which weights matter, and spends precision on those.** It needs *data* because "which
weights matter" is defined by the data distribution, not by the weights alone.

In GGUF this is the **importance matrix (imatrix)**: `llama-imatrix` runs the fp16 model over
calibration text and records each weight's importance; `llama-quantize --imatrix` then allocates
the K-quant bits accordingly. (Legacy quants `q5_0/q8_0` ignore the imatrix — only K/I-quants use it.)

## The controlled experiment
```bash
# fp16 GGUF -> naive Q4_K_M (no calibration)
llama-quantize qwen2.5-1.5b-instruct-fp16.gguf qwen-q4km-naive.gguf Q4_K_M
llama-perplexity -m qwen-q4km-naive.gguf -f wiki.test.raw -ngl 99   # 10.5501

# calibration: build the importance matrix from real text
llama-imatrix -m qwen2.5-1.5b-instruct-fp16.gguf -f calib.txt -o qwen.imatrix -ngl 99

# fp16 GGUF -> calibrated Q4_K_M (ONLY change: --imatrix)
llama-quantize --imatrix qwen.imatrix qwen2.5-1.5b-instruct-fp16.gguf qwen-q4km-imat.gguf Q4_K_M
llama-perplexity -m qwen-q4km-imat.gguf -f wiki.test.raw -ngl 99    # 10.4191
```

## Result (Qwen2.5-1.5B, WikiText-2, llama.cpp ruler)

| Q4_K_M | perplexity | size | bits/wt |
|---|---|---|---|
| naive (no calibration) | 10.5501 ± 0.074 | 1060 MiB | 5.00 |
| **+ imatrix (calibrated)** | **10.4191 ± 0.073** | **1060 MiB** | **5.00** |
| **gain** | **−0.131 (−1.24%)** | **0 (identical)** | same |

**Lower perplexity at identical size** — calibration improved the model *for free*, purely by
spending the same bits more wisely.

## The lessons
1. **Calibration is free quality.** Same bits, same size, lower perplexity — only the *choice* of
   rounding changed, guided by data.
2. **It's a paired comparison** (same eval text, only quantization differs), so the ~1.2% gain is
   real and consistent despite the individual ± bars nearly overlapping.
3. **The gain grows as bits drop.** At 5 bpw there's little damage to undo (~1.2%); at 2–3 bit the
   imatrix is the difference between usable and garbage — which is why I-quants *require* it.
4. **Prerequisite:** K-quants/imatrix need dims divisible by 256, else tensors fall back to legacy
   (which ignore the imatrix). Pick models with hidden/ffn divisible by 256.
5. **Same idea as GPTQ/AWQ** (Phase 4): use real data to quantize smartly. The imatrix is the
   lightweight GGUF version; GPTQ (Hessian error-compensation) and AWQ (activation-aware scaling)
   push it much further.

> Reproduce: `bash 02_local_gguf/reproduce_phase3.sh`
