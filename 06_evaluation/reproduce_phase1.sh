#!/usr/bin/env bash
# Phase 1 — reproduce the fp16 baseline measurement.
# See ../SETUP.md for prerequisites. Hardware used: Apple M4 (16 GB).
#
#   bash reproduce_phase1.sh            # default work horse (Qwen2.5-0.5B-Instruct)
#   bash reproduce_phase1.sh Qwen/Qwen2.5-1.5B-Instruct
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
DEVICE="${2:-mps}"     # mps (Apple) | cuda | cpu

echo "### Phase 1 — fp16 baseline perplexity for $MODEL on $DEVICE"
pip install -q -U torch transformers datasets accelerate sentencepiece
python perplexity.py --model "$MODEL" --device "$DEVICE"
# Qwen2.5-0.5B-Instruct, full WikiText-2 test (HF sliding window) -> ~12.67
echo "DONE. This fp16 number is the reference every quantized model is judged against."
