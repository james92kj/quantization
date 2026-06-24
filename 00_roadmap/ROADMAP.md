# Practical LLM Quantization — Roadmap

> Your setup: **Apple M4 / 16 GB unified memory** (local) + **RunPod / GCP** (CUDA).
> Style: **libraries-first → then peel back to implementation.**
> You already know the theory: symmetric/asymmetric, weight packing/unpacking, scales & zero-points.
> This roadmap turns that into *muscle memory* on real models.

---

## The one-paragraph mental model

Quantization replaces high-precision weights/activations (fp16/bf16, 16 bits) with low-bit integers
(int8, int4) plus a tiny amount of float metadata (a **scale**, sometimes a **zero-point**), grouped
so that one scale covers a small block of values. Three things decide quality:
**(1) granularity** (per-tensor → per-channel → per-group, finer = better), **(2) what you quantize**
(weights only, or weights *and* activations — activations are harder because of outliers), and
**(3) how you choose the scales** (naive min/max, vs. calibration-driven, vs. error-compensating like GPTQ,
vs. outlier-protecting like AWQ). Everything else is engineering around those three knobs.

---

## Two tracks, run in parallel

| Track | Hardware | Why | Tools |
|-------|----------|-----|-------|
| **A — Local, instant feedback** | M4 16GB | Quantize & run *today*, no setup tax, Metal-accelerated | MLX, llama.cpp/GGUF, PyTorch+MPS |
| **B — Production CUDA stack** | RunPod / GCP | The methods the industry actually ships; need CUDA kernels | bitsandbytes, GPTQModel, AutoAWQ, vLLM, lm-eval |

Do Track A first each week for intuition, then reproduce the "real" version on Track B.

---

## Phase 0 — Environment (Day 0, ~30 min)

- [ ] Local Python env for MLX + PyTorch (`conda`/`venv`), install `mlx-lm`, `torch`, `transformers`, `datasets`, `huggingface_hub`.
- [ ] Build/install `llama.cpp` with Metal.
- [ ] `huggingface-cli login` (gated models like Llama need it).
- [ ] On RunPod: pick a template (PyTorch + CUDA 12.x), confirm `nvidia-smi`. Keep a `setup.sh` so every pod is reproducible.
- [ ] Pick the **work horse models** (small enough to iterate fast):
  - `Qwen2.5-0.5B-Instruct` and `Llama-3.2-1B-Instruct` → fit easily at fp16 on 16GB, fast.
  - `Llama-3.2-3B` / `Mistral-7B` → the "feels like a real model" tier (7B fp16 ≈ 14GB, tight on 16GB → this is *why* we quantize).

> **Definition of done for the whole roadmap:** you can take any HF model, quantize it three different ways,
> measure the quality drop with perplexity + a task benchmark + a side-by-side generation, and explain the
> tradeoff you chose.

---

## Phase 1 — Baseline: load fp16 and measure (Week 1)

The discipline that separates practitioners from tutorial-followers: **always establish the fp16 baseline first.**
You cannot judge a quantized model without the reference number.

1. Load `Llama-3.2-1B` in fp16/bf16. Record: disk size, RAM footprint, tokens/sec, and **WikiText-2 perplexity**.
2. Generate from a fixed set of 5 prompts. Save the outputs verbatim — this is your "golden" qualitative set.
3. Same for a 3B model.

**Artifacts:** `labs/01_baseline.py`, a `notes/baselines.md` table (model · precision · size · PPL · tok/s).

➡ Folder: `06_evaluation/` (perplexity harness lives here, reused everywhere).

---

## Phase 2 — Local quantization, the easy win (Week 1–2)

### 2a. MLX (Apple-native) — `01_local_mlx/`
- Quantize a model to 4-bit and 8-bit with `mlx_lm.convert` (group-wise affine quant).
- Run generation. Compare size, speed, PPL, and the golden prompts against the fp16 baseline.
- Understand `group_size` and `bits` — this is per-group symmetric/affine quant, exactly the theory you learned, in production form.

### 2b. llama.cpp / GGUF — `02_local_gguf/`
- Convert HF → GGUF fp16 → quantize to `Q8_0`, `Q5_K_M`, `Q4_K_M`, `Q4_0`, and an I-quant (`IQ4_XS`).
- Learn the **K-quant** family (mixed precision per tensor-type) vs legacy vs I-quants.
- Measure with `llama-perplexity`. Build the **PPL-vs-bits-per-weight curve** — the single most instructive plot in quantization.
- Bonus: **importance matrix (imatrix)** — your first taste of calibration, locally.

**Deliverable for Phase 2:** one chart, x = bits/weight, y = perplexity, one curve per method. You will *see* the cliff below ~4 bits.

---

## Phase 3 — Calibration, properly (Week 2) — `05_calibration/`

This is the conceptual heart of "good" PTQ.

- **Why** calibration exists: error-compensating and activation-aware methods need to see real data to choose scales that minimize *output* error, not just weight rounding error.
- Build a reusable calibration set: load `C4` / `WikiText-2`, tokenize, sample N=128–512 sequences of length 512–2048.
- The golden rule: **calibration data should resemble deployment data.** You'll prove this by deliberately mis-calibrating (e.g. calibrate a code model on prose) and watching quality drop.
- Static vs dynamic activation quantization; where calibration collects activation ranges (min/max vs percentile vs entropy).

**Deliverable:** `05_calibration/build_calib.py` — one function every later method imports.

---

## Phase 4 — The production PTQ methods, on CUDA (Week 3) — RunPod/GCP

Now Track B. Same model, four methods, one comparison table.

### 4a. bitsandbytes — `03_cuda_bnb/`
- `LLM.int8()` (8-bit with outlier handling) and **NF4 + double quantization** (4-bit), loaded via `BitsAndBytesConfig`.
- Zero calibration needed (on-the-fly) — the easy entry to 4-bit on GPU, and the basis of **QLoRA**.

### 4b. GPTQ — `04_cuda_gptq_awq/`
- Error-compensating, second-order (Hessian-based) layer-wise quant. **Needs calibration.**
- Use the maintained library (GPTQModel; AutoGPTQ is legacy). Quantize 3B/7B to 4-bit, save, reload.
- Key knobs: `bits`, `group_size`, `desc_act`, `damp`.

### 4c. AWQ — `04_cuda_gptq_awq/`
- Activation-aware: find salient weight channels via activation magnitude, scale to protect them. **Needs calibration.**
- Contrast with GPTQ conceptually and empirically.

### 4d. Serve & compare
- Load the GPTQ/AWQ models in **vLLM**, measure throughput.
- One master table: method · bits · PPL Δ · MMLU/HellaSwag Δ · VRAM · tok/s.

---

## Phase 5 — Rigorous evaluation (Week 3–4) — `06_evaluation/`

- Perplexity harness (sliding window) — already built in Phase 1, now run across *all* artifacts.
- **lm-evaluation-harness**: hellaswag, arc_easy, mmlu — task accuracy, not just PPL.
- **KL divergence** between fp16 and quantized output distributions — the most sensitive detector of a bad quant.
- Build the acceptance rubric: what Δ is OK for 8-bit vs 4-bit vs sub-4-bit, and the failure signatures (repetition loops, broken reasoning, degenerate logits).

---

## Phase 6 — Peel back to implementation (Week 4+) — `07_from_scratch/`

Now that you've *driven* the libraries, reimplement the core to own it:
1. From scratch: int8 & int4 symmetric/asymmetric quant of a `nn.Linear`, per-tensor → per-channel → per-group. Pack/unpack int4 into uint8. (You know this theory — now make a layer that actually runs.)
2. A `QuantLinear` module: store packed weights + scales, dequantize-on-the-fly in `forward`, swap it into a real model, measure PPL.
3. Mini-GPTQ: implement the layer-wise Hessian error-compensation loop on a single linear layer and watch it beat naive round-to-nearest at the same bit-width.
4. (Stretch) A toy AWQ scaling search.

**Deliverable:** your own `quant.py` that quantizes a small model end-to-end and lands within a known PPL gap of the library version.

---

## The frontier (read, don't necessarily run)

SmoothQuant (migrate activation outliers into weights) · HQQ (fast, calibration-free) ·
rotation methods **QuaRot / SpinQuant** (rotate away outliers) · **FP8 / MXFP4** (hardware-native low precision) ·
AQLM / extreme 2-bit. We'll point at papers as each becomes relevant.

---

## How we'll work together

- Each phase = a short concept note (`notes/`) + a runnable lab (`labs/` or the numbered folders) you execute and we read the numbers together.
- I keep a running scoreboard in `notes/scoreboard.md`.
- You run, paste me the output, we interpret. When a number surprises us, we dig.

## Progress tracker

- [x] Phase 0 — Environment
- [x] Phase 1 — fp16 baseline + PPL harness
- [x] Phase 2 — MLX + GGUF local quant, PPL-vs-bits curve
- [x] Phase 3 — Calibration set (GGUF imatrix A/B)
- [x] Phase 4 — bnb / GPTQ / AWQ on CUDA (L4) — master table in `notes/scoreboard.md`
- [ ] Phase 5 — lm-eval + KL divergence
- [ ] Phase 6 — From-scratch implementation
