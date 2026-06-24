# Phase 4 — the journey & failure log (lab notebook)

Every obstacle hit getting bitsandbytes / GPTQ / AWQ running on a fresh GCP L4, with symptom → root
cause → fix → lesson. The failures matter more than the final numbers: this is the debugging trail.
Pairs with [`SETUP_CUDA.md`](../SETUP_CUDA.md) (the clean recipe distilled from these scars) and
[`research-engineer-loop.md`](research-engineer-loop.md) (the method).

## Chronological log

### 0. Provision + environment discovery (steps, no failure)
- Created GCP L4 VM `quant-l4` (g2-standard-8, image `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`).
- Discovered: **no conda** (python = system `/usr/bin/python3` 3.10; torch in `/usr/local/.../dist-packages`).
- Discovered: **no `nvcc`** — runtime CUDA + driver only, no compiler toolkit. (Bites GPTQModel later.)
- Habit adopted: **verify every library imports before using it** (bnb, llmcompressor), and **introspect
  unfamiliar APIs** (oneshot/GPTQModifier/AWQModifier signatures) before writing scripts.

### 1. ❌ torchaudio ABI crash (Phase 4a, first model load)
- **Symptom:** `OSError: undefined symbol: torch_library_impl` from `torchaudio/_torchaudio.abi3.so`.
- **Root cause:** `pip install -U transformers` pulled bleeding-edge transformers that *eagerly* imports
  torchaudio (an audio loss class); the DLVM's preinstalled torchaudio is ABI-mismatched vs torch 2.9.1.
- **Fix:** pin **`transformers==4.49.0`** (predates that import). We don't use audio.
- **Lesson:** the production CUDA stack's tax is version/ABI agreement across torch/transformers/etc. Pin.

### 2. ⚠️ nf4 param-count display bug (Phase 4a, reporting)
- **Symptom:** nf4 reported "0.889 B params / 1.30 bytes/param" — implausible.
- **Root cause:** bnb packs two 4-bit weights per `uint8`; `tensor.numel()` counts the *packed* slots
  (half). The memory number (1.153 GB) was fine; only the derived per-param column was wrong.
- **Fix:** count `Params4bit` tensors ×2 → honest 0.747 bytes/param.
- **Lesson:** a surprising *derived* metric can be a display bug, not a real effect. Sanity-check the math.

### 3. ❌ GPTQModel install — broken `pcre` build dep
- **Symptom:** `pip install gptqmodel` → `No matching distribution found for pypcre>=0.2.14`.
- **Root cause:** every `pypcre` sdist on PyPI ships corrupt metadata (project name "unknown"); pip
  refuses it. It's a build-system requirement of GPTQModel.
- **First fix attempt:** `--no-build-isolation` (skips build deps) → new error `ModuleNotFoundError:
  No module named 'pcre'` because `setup.py` does `import pcre` at build time. Still blocked.

### 4. ❌ GPTQModel — no wheels + no nvcc (compounding #3)
- **Symptom:** no prebuilt wheel for torch 2.9/cu129 (sdist-only) → must build from source → needs `nvcc`
  → absent. Doubly blocked.
- **Fix:** **pivot to `llm-compressor`** (Neural Magic) — same GPTQ algorithm, pure PyTorch, no compiler,
  no pcre. Installs clean; transformers held at 4.49.0.
- **Lesson:** when a library is unbuildable in your env, switch toolchain rather than fight the compiler.
  Prefer pure-Python / prebuilt-wheel paths on a driver-only VM.

### 5. ❌ GPTQ calibration bug — calibrated GPTQ WORSE than NF4
- **Symptom:** GPTQ PPL **11.95 (+38%)**, worse than calibration-free NF4 (9.31). A calibrated method
  losing to an uncalibrated one = red flag.
- **Dig:** 6-line diagnostic on calibration samples → median **158 tokens**, none near the intended 2048
  (WikiText rows are short lines, tokenized per-row).
- **Root cause:** stubby sequences → garbage Hessian estimate → GPTQ compensates wrongly.
- **Fix:** concatenate corpus, slice into real 2048-token chunks → **9.44**. Recovered 2.5 PPL.
- **Lesson:** **calibration DATA quality decides everything.** A shocking result is almost always a setup
  bug, not a law of nature — dig with a cheap diagnostic before re-running.

### 6. 🔎 GPTQ ≈ NF4 (surprise, not a failure) → act-order experiment
- Proper-calib GPTQ 9.44 still didn't beat NF4 9.31. Verified vs literature (GPTQ ≈ NF4 at 4-bit on small
  models; NF4 is a strong nonuniform/group-64 baseline). Ran one more lever: **`actorder=True` → 9.07**,
  GPTQ beats NF4 (+4.93% vs +7.66%). Act-order helped 0.37 PPL — *more* than the literature's ~0.1 on a
  13B. Recorded the overshoot honestly.

### 7. ❌ AWQ full-run crash — offload + tied-embedding shape bug
- **Symptom:** full 256×2048 AWQ run → `RuntimeError: size of tensor a (8960) must match b (151936)` in
  `update_offload_parameter` at `on_end`. (151936 = vocab/lm_head; 8960 = MLP intermediate.)
- **Why smoke passed but full failed:** only calibration size changed (16×512 → 256×2048). The big
  calibration filled GPU memory → llm-compressor **offloaded** the tied embedding/`lm_head` to CPU → the
  offloaded-param update path has a shape bug with Qwen2.5's **tied embeddings**. Small smoke stayed
  resident, never hit that branch.
- **First fix attempt (WRONG):** assumed memory pressure → CPU offload; retried at 128×512. **Failed
  identically.** The error is deterministic, not memory-driven. (Good: we *checked* the hypothesis
  instead of believing it.)
- **Real root cause:** Qwen2.5 has **tied embeddings** (`lm_head` shares its weight tensor with
  `embed_tokens`). llm-compressor's AWQ scale-update follows the tie and writes a down-proj-shaped
  scale (8960) into the shared vocab tensor (151936) → crash. The smoke run skipped the offending write.
- **Real fix:** **untie** before quantizing — `lm_head.weight = Parameter(lm_head.weight.clone())` +
  `config.tie_word_embeddings = False`. Numerically identical model; `ignore=["lm_head"]` now excludes it
  cleanly. AWQ then ran to completion.
- **Lesson:** a deterministic shape error is NOT a memory bug — read the shapes (151936 = vocab → it's
  the embedding). **Tied embeddings are a recurring sharp edge** in quantization tooling; untying is the
  standard workaround. Also: verify hypotheses cheaply before committing to them.

## Failure tally
| # | area | class | one-line fix |
|---|------|-------|--------------|
| 1 | transformers/torchaudio | ABI mismatch | pin transformers==4.49.0 |
| 2 | bnb reporting | display bug | count packed 4-bit params ×2 |
| 3 | GPTQModel install | broken build dep | (led to #4) |
| 4 | GPTQModel build | no wheels/nvcc | pivot to llm-compressor |
| 5 | GPTQ calibration | bad input data | concat+chunk to 2048-tok |
| 7 | AWQ full run | offload + tied-emb | calibrate 128×512 (stay on-GPU) |
