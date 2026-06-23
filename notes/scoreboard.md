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

## Coming next
- Phase 4: **GPTQ / AWQ** on a CUDA GPU — calibration taken further (Hessian / activation-aware).
- Bigger work horses (Qwen2.5-3B / 7B) — expect a *much* smaller quality drop (large models are
  far more quantization-robust).
