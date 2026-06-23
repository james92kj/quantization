#!/usr/bin/env bash
# Phase 2 — reproduce the MLX 4-bit quantization experiment end-to-end.
# Hardware used: Apple M4 (16 GB). Any Apple Silicon Mac works.
#
#   bash reproduce_phase2.sh
#
# Produces: a 4-bit model, perplexity for fp16 vs 4-bit (same ruler), and
# side-by-side generations. Expected numbers (M4) are noted inline.
set -euo pipefail
cd "$(dirname "$0")"

# --- prerequisites (see ../SETUP.md) — needs Apple Silicon + mlx-lm ---
python -c "import mlx.core" 2>/dev/null || { echo "ERROR: mlx not found (Apple Silicon only). See ../SETUP.md  (pip install mlx-lm)"; exit 1; }

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
OUT="./qwen2.5-0.5b-4bit-g64"

echo "### 0. install deps"
pip install -q -U mlx-lm datasets

echo "### 1. quantize fp16 -> 4-bit (group-wise affine, group_size 64)"
rm -rf "$OUT"
python -m mlx_lm convert --model "$MODEL" -q --q-bits 4 --q-group-size 64 --mlx-path "$OUT"
# -> "Quantized model with 4.502 bits per weight."

echo "### 2. size comparison"
echo -n "fp16 (HF cache): "; du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct 2>/dev/null | awk '{print $1}'
echo -n "4-bit (MLX)    : "; du -sh "$OUT" | awk '{print $1}'
# -> 953M  vs  290M   (3.3x smaller)

echo "### 3. perplexity, SAME harness for both (WikiText-2 test, MLX engine)"
echo "-- fp16 --";  python mlx_perplexity.py --model "$MODEL"   # -> ~14.26
echo "-- 4-bit --"; python mlx_perplexity.py --model "$OUT"     # -> ~17.09  (+19.8%)

echo "### 4. side-by-side generation"
P="Explain what model quantization is in two sentences."
echo "-- fp16 --";  python -m mlx_lm generate --model "$MODEL" --prompt "$P" --max-tokens 80
echo "-- 4-bit --"; python -m mlx_lm generate --model "$OUT"   --prompt "$P" --max-tokens 80

echo
echo "DONE. Expected: 3.3x smaller, 2.5x faster generation, PPL 14.26 -> 17.09 (+19.8%),"
echo "answers still coherent. See ../notes/02_mlx_quantization.md for the full write-up."
