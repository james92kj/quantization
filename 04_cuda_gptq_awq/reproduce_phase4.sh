#!/usr/bin/env bash
# Reproduce Phase 4 (CUDA): bitsandbytes (4a) + GPTQ (4b) + AWQ (4c) on Qwen2.5-1.5B-Instruct.
#
# Phase 4 needs a CUDA GPU. Provision one first per ../SETUP_CUDA.md (GCP L4 used here), then copy
# the four scripts into the VM home dir so the imports line up, e.g. from the repo root:
#
#   gcloud compute scp 03_cuda_bnb/bnb_eval.py 04_cuda_gptq_awq/gptq_quantize.py \
#       04_cuda_gptq_awq/awq_quantize.py 06_evaluation/perplexity.py quant-l4:~/ --zone us-central1-a
#
# Then run THIS script ON the VM (bash reproduce_phase4.sh). All numbers are on the HF perplexity
# ruler (WikiText-2-raw, win 2048 / stride 1024); fp16 anchor = 8.6453. See notes/scoreboard.md.
#
# Cost: stop the VM when done -> gcloud compute instances stop quant-l4 --zone us-central1-a
set -euo pipefail

echo "### 0. Sanity: GPU + CUDA torch"
python3 -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))'

echo "### 1. Install the pinned stack"
# transformers PINNED to 4.49.0: newer eagerly imports torchaudio, whose preinstalled .so is
# ABI-mismatched vs torch 2.9.1 (OSError: undefined symbol). GPTQModel is unbuildable here
# (broken pcre build dep + no nvcc) -> llm-compressor runs the same GPTQ/AWQ algorithms, pure-PyTorch.
python3 -m pip install -q -U bitsandbytes 'transformers==4.49.0' accelerate datasets ninja llmcompressor

echo "### 2. Phase 4a -- bitsandbytes (fp16 / int8 / nf4)"
python3 bnb_eval.py --mode fp16    # expect weights 3.088 GB, PPL 8.6453
python3 bnb_eval.py --mode int8    # expect 1.845 GB, PPL 8.6869 (+0.48%, near-lossless, but SLOWER)
python3 bnb_eval.py --mode nf4     # expect 1.153 GB (2.68x), PPL 9.3078 (+7.66%)

echo "### 3. Phase 4b -- GPTQ (calibration matters; act-order is the lever)"
python3 gptq_quantize.py              # proper 2048-tok calib, no act-order -> PPL 9.4401 (~= NF4)
python3 gptq_quantize.py --actorder   # + activation ordering          -> PPL 9.0716 (beats NF4) BEST

echo "### 4. Phase 4c -- AWQ (calibration-insensitive; untie lm_head for Qwen2.5 tied embeddings)"
python3 awq_quantize.py --samples 128 --seqlen 512    # PPL 10.0046
python3 awq_quantize.py --samples 256 --seqlen 2048   # PPL 10.0046 (IDENTICAL -> calibration-insensitive)

echo "### Done. Master table in notes/scoreboard.md. Failure trail in notes/phase4-journey-and-failures.md."
