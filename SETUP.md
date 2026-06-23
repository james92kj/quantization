# Setup — everything needed to reproduce this repo from scratch

Reproduced on **Apple M4 (16 GB), macOS**. CUDA phases (4+) will use a cloud GPU.
Do these once, then run the per-phase `reproduce_phaseN.sh` scripts.

---

## 1. Python environment
```bash
# conda (used here) or any venv with Python 3.10–3.12
conda create -n quant python=3.12 -y && conda activate quant
# or:  python3 -m venv .venv && source .venv/bin/activate
```

## 2. Python packages
```bash
pip install -U \
  torch transformers datasets accelerate sentencepiece \  # Phase 1 (HF perplexity)
  mlx-lm \                                                 # Phase 2 (Apple-native quant)
  "huggingface_hub[cli]"                                   # model downloads (the `hf` CLI)
```

## 3. llama.cpp (Phases 2b/3 — GGUF + imatrix, Metal-accelerated)
```bash
brew install llama.cpp     # provides llama-quantize, llama-imatrix, llama-perplexity, llama-cli
# verify:
llama-quantize --help | head -1
```
> Not on macOS/Homebrew? Build from source: `git clone https://github.com/ggml-org/llama.cpp &&
> cd llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j`
> (binaries land in `build/bin/`).

## 4. Hugging Face (only if you use gated models)
All work-horse models here (`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct(-GGUF)`) are
**open** — no login needed. For gated models (e.g. Llama): `hf auth login`.

---

## 5. Reproduce, in order
Each script is self-contained (downloads its model, builds its data, runs + prints the numbers):

```bash
# Phase 1 — fp16 baseline perplexity
bash 06_evaluation/reproduce_phase1.sh

# Phase 2 — MLX 4-bit quantization (size / perplexity / generation)
bash 01_local_mlx/reproduce_phase2.sh

# Phase 3 — calibration via GGUF imatrix (naive vs calibrated A/B)
bash 02_local_gguf/reproduce_phase3.sh
```

Expected numbers are printed inline and recorded in [`notes/scoreboard.md`](notes/scoreboard.md).

## Hardware note
Perplexity/quantization times below are for an Apple M4 (16 GB). Bigger models or different
hardware will differ, but the *relative* results (size cuts, naive-vs-calibrated gaps) hold.
