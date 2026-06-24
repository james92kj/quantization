# Scoreboard

Running record of every model/precision we measure. **The whole game: shrink the model
(smaller, faster) while keeping perplexity close to the fp16 baseline.**

> Perplexity is only comparable within the **same ruler** (engine + windowing). We keep two
> rulers below and never cross-compare them. Harnesses: `06_evaluation/perplexity.py` (HF,
> overlapping sliding window) and `01_local_mlx/mlx_perplexity.py` (MLX, non-overlapping).

## Qwen2.5-0.5B-Instruct (work horse)

### MLX ruler — the apples-to-apples Phase 2 comparison
| precision | bits/wt | size on disk | WikiText-2 PPL | Δ vs fp16 | peak mem (gen) | speed |
|---|---|---|---|---|---|---|
| **fp16 (baseline)** | 16 | 953 MB | **14.26** | — | 1.057 GB | 90 tok/s |
| **MLX 4-bit (g64)** | 4.502 | 290 MB (**3.3×↓**) | **17.09** | **+19.8%** | 0.323 GB (**3.3×↓**) | 228 tok/s (**2.5×↑**) |

**Read:** naive round-to-nearest 4-bit on a tiny 0.5B model costs ~20% perplexity — real loss,
the bar Phase 3 (calibration) / Phase 4 (GPTQ/AWQ) aim to beat. Generations still coherent;
huge memory/speed win. Details: [`02_mlx_quantization.md`](02_mlx_quantization.md).

### HF ruler — Phase 1 baseline only (different windowing, do NOT compare to MLX rows)
| precision | size | WikiText-2 PPL |
|---|---|---|
| fp16 | 953 MB | 12.67 (full test, sliding window) |

## Qwen2.5-1.5B-Instruct — Phase 3 calibration (GGUF, llama.cpp ruler)
Controlled A/B: same `Q4_K_M`, same 5.00 bpw / 1060 MiB, toggling only the importance matrix.

| Q4_K_M | WikiText-2 PPL | size | calibration |
|---|---|---|---|
| naive | 10.5501 ± 0.074 | 1060 MiB | none |
| **+ imatrix** | **10.4191 ± 0.073** | 1060 MiB (same) | yes |
| gain | **−0.131 (−1.24%)** | 0 | — |

**Read:** calibration lowered perplexity at *identical* size — free quality from spending the same
bits more wisely. Gain is modest at 5 bpw but grows sharply at lower bits. Details:
[`04_calibration_imatrix.md`](04_calibration_imatrix.md). (0.5B was unusable here — hidden 896 not
÷256 → legacy fallback that ignores the imatrix; see `03_gguf_output_and_imatrix_gotcha.md`.)

## Qwen2.5-1.5B-Instruct — Phase 4a bitsandbytes (CUDA L4, HF ruler)
Calibration-free GPU quant. **HF ruler** (`06_evaluation/perplexity.py`) — NOT comparable to the
llama.cpp rows above. fp16 baseline re-measured on this ruler = 8.6453.

| mode | weights VRAM | vs fp16 | PPL | Δ vs fp16 | eval time |
|---|---|---|---|---|---|
| **fp16** | 3.088 GB | 1.0× | **8.6453** | — | 62 s |
| **int8** (LLM.int8) | 1.845 GB | 1.67× | 8.6869 | **+0.48%** | 96 s (slower!) |
| **nf4** (NF4 + dq) | 1.153 GB | **2.68×** | 9.3078 | **+7.66%** | 71 s |

**Read:** 8-bit is near-free quality but buys *memory, not speed* (int8 ran slower). 4-bit NF4 is
2.68× smaller for +7.66% PPL — the *calibration-free* 4-bit cost that GPTQ/AWQ (4b/4c) aim to beat at
the same bit-width. Details: [`05_bnb_quantization.md`](05_bnb_quantization.md).

## Qwen2.5-1.5B-Instruct — Phase 4b GPTQ (CUDA L4, HF ruler)
GPTQ W4A16 (group 128, no actorder) via llm-compressor. Calibrate on WikiText train, eval on test.

| 4-bit method | calibration | actorder | PPL | Δ vs fp16 (8.6453) |
|---|---|---|---|---|
| NF4 (bnb, 4a) | none | — | 9.3078 | +7.66% |
| GPTQ — broken calib (158-tok stubs) | bad | off | 11.9491 | +38.2% 🚩 |
| GPTQ — proper calib (256 × 2048) | good | off | 9.4401 | +9.19% |
| **GPTQ — proper calib + act-order** | good | **on** | **9.0716** | **+4.93%** ✅ |

**Read:** the arc, one lever at a time — (1) calibration data quality is everything: stubby sequences
made GPTQ *worse* than no-calibration NF4 (11.95); (2) fixing to full 2048-tok chunks recovered to
9.44 (≈NF4); (3) adding act-order → **9.07, GPTQ beats NF4** (+4.93% vs +7.66%). Calibration-free NF4
is a strong zero-effort baseline; GPTQ done properly beats it at the same 4 bits. (Act-order bought
0.37 PPL here — more than the literature's ~0.1 on a 13B; cost: slower inference.)
Details: [`06_gptq.md`](06_gptq.md).

## Coming next
- Phase 4b/4c: **GPTQ / AWQ** on a CUDA GPU — calibration taken further (Hessian / activation-aware);
  beat NF4's +7.66% at the same 4 bits.
- Bigger work horses (Qwen2.5-3B / 7B) — expect a *much* smaller quality drop (large models are
  far more quantization-robust).
