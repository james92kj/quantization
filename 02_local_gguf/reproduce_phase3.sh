#!/usr/bin/env bash
# Phase 3 — reproduce the calibration (GGUF imatrix) experiment end-to-end.
# Requires llama.cpp (brew install llama.cpp) and Python with `datasets`.
# Hardware used: Apple M4 (16 GB), Metal.
#
#   bash reproduce_phase3.sh
#
# Shows that an importance matrix (calibration) lowers perplexity at IDENTICAL size.
set -euo pipefail
cd "$(dirname "$0")"

# --- prerequisites (see ../SETUP.md) ---
command -v llama-quantize >/dev/null  || { echo "ERROR: llama.cpp not found. See ../SETUP.md  (brew install llama.cpp)"; exit 1; }
command -v hf >/dev/null              || { echo "ERROR: hf CLI not found. See ../SETUP.md  (pip install 'huggingface_hub[cli]')"; exit 1; }
python -c "import datasets" 2>/dev/null || { echo "ERROR: python 'datasets' missing. See ../SETUP.md"; exit 1; }

FP16=qwen2.5-1.5b-instruct-fp16.gguf

echo "### 0. fp16 GGUF (hidden 1536 = 6x256, so K-quants don't fall back)"
[ -f "$FP16" ] || hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF "$FP16" --local-dir .

echo "### 1. text files: eval = wikitext-2 TEST, calibration = wikitext-2 TRAIN (kept separate)"
python - <<'PY'
from datasets import load_dataset
te = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
open("wiki.test.raw","w").write("\n\n".join(t for t in te["text"] if t.strip()))
tr = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
buf=[]; n=0
for t in tr["text"]:
    if t.strip(): buf.append(t); n+=len(t)
    if n>200000: break
open("calib.txt","w").write("\n\n".join(buf))
PY

echo "### 2. NAIVE Q4_K_M (no calibration) + perplexity"
llama-quantize "$FP16" qwen-q4km-naive.gguf Q4_K_M
llama-perplexity -m qwen-q4km-naive.gguf -f wiki.test.raw -ngl 99 2>&1 | grep "Final estimate"  # ~10.55

echo "### 3. CALIBRATION: build the importance matrix from real text"
llama-imatrix -m "$FP16" -f calib.txt -o qwen.imatrix -ngl 99

echo "### 4. CALIBRATED Q4_K_M (only change: --imatrix) + perplexity"
llama-quantize --imatrix qwen.imatrix "$FP16" qwen-q4km-imat.gguf Q4_K_M
llama-perplexity -m qwen-q4km-imat.gguf -f wiki.test.raw -ngl 99 2>&1 | grep "Final estimate"  # ~10.42

echo
echo "DONE. Expected: naive PPL ~10.55 vs imatrix ~10.42 at the SAME 5.00 bpw / 1060 MiB."
echo "Calibration = lower perplexity at zero size cost. See ../notes/04_calibration_imatrix.md"
