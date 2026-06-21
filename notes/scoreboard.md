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

## Coming next
- Phase 3: re-quantize this model with **calibration** → watch the +19.8% gap shrink.
- Bigger work horses (Qwen2.5-3B / 7B) — same pipeline; expect a *much* smaller quality drop
  (large models are far more quantization-robust).
