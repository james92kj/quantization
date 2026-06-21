# How to Measure the Quality of a Quantized Model

> **The one rule:** a quantized model is only ever judged **relative to its own fp16 baseline**,
> measured with the **exact same harness** (same code, dataset, stride, tokenizer).
> Absolute numbers mean nothing across different setups. Always run fp16 first.

There are **4 layers**, cheap→expensive, coarse→sensitive. Use them in order.

---

## Layer 1 — Perplexity (PPL): "does it still model language?"

**What it is.** `PPL = exp(mean negative log-likelihood per token)` on held-out text
(standard: WikiText-2 test). It's `exp(cross-entropy)`. Lower = better. It's the cheapest
sanity signal that quantization didn't break the model.

**How.** `06_evaluation/perplexity.py` (sliding window so every token is scored with full
left-context, no double-counting). Run it on fp16, then on the quantized model, compare.

```bash
python perplexity.py --model Qwen/Qwen2.5-0.5B-Instruct --device mps          # baseline
python perplexity.py --model ./my-quant-dir          --device mps             # quantized
```

**Reading the delta.** Report **%Δ = (ppl_quant − ppl_fp16)/ppl_fp16**, not the raw gap.
Rule of thumb:

| %Δ PPL vs fp16 | Verdict |
|---|---|
| < 1% | essentially lossless (typical for 8-bit) |
| 1–5% | good (typical for a *calibrated* 4-bit: GPTQ/AWQ/Q4_K_M) |
| 5–15% | degraded — usually naive RTN 4-bit, or a bad calib set |
| > 15% / NaN / hundreds | **broken** — zero scale, unhandled layer, saturated ranges |

**PPL's blind spot:** a model can keep PPL almost flat yet lose reasoning / instruction-following.
That's why Layers 2–4 exist.

---

## Layer 2 — KL divergence + top-1 agreement: "same distribution as fp16?"

**What it is.** The *most sensitive* metric. For each token position, compare the full
next-token probability distribution of fp16 (`P`) vs quantized (`Q`):
`KL(P‖Q) = Σ P·log(P/Q)`, averaged over a corpus. **0 = identical.** Also track
**top-1 agreement** = % of positions where both models' argmax token matches.

Two models can have near-identical PPL but diverge here — KL catches damage PPL hides.
Report **mean KL** *and* the **tail (p99/max)** — the tail finds tokens where the quant goes
badly wrong even when the average looks fine.

**How.** Built into `06_evaluation/compare_quality.py` (Layer 2 section). On llama.cpp/GGUF
use `llama-perplexity --kl-divergence-base` (record fp16 logits) then `--kl-divergence`.

Rough reading: mean KL < 0.01 and top-1 agreement > 98% = a healthy 4-bit quant.

---

## Layer 3 — Side-by-side generation: "does it still *sound* right?"

Metrics miss pathologies. Generate (greedy, `do_sample=False`) from a **fixed golden prompt
set** with fp16, save the outputs, then generate from the quant and eyeball them together.
`compare_quality.py` does this automatically. You're hunting for:

- **repetition loops** / degenerate text → activation outliers not handled (need SmoothQuant / better method)
- **format or language drift** (e.g. chat model stops following the template) → calibrated on the wrong distribution
- **broken reasoning / math** that PPL didn't flag → sub-4-bit knowledge loss

Greedy decoding is deterministic, so differences are real signal, not sampling noise.

---

## Layer 4 — Task accuracy (lm-evaluation-harness): "can it still *do things*?"

PPL/KL measure likelihood; tasks measure **capability**. The EleutherAI harness is standard.

```bash
pip install lm-eval
lm_eval --model hf \
  --model_args pretrained=./my-quant-dir,dtype=float16 \
  --tasks hellaswag,arc_challenge,mmlu \
  --num_fewshot 0 --batch_size auto --output_path ./results
```

Run the identical command on fp16 and the quant; compare **per-task accuracy**. Add a task in
your **deployment domain** (`gsm8k` math, `humaneval` code, `ifeval` instruction-following) —
sub-4-bit quants often hold PPL but drop several points on MMLU/GSM8K.
(Heavy; run on the GPU box, not the Mac.)

**Acceptable drop:** 8-bit ≈ within noise (<0.5pt). Good 4-bit ≈ <1–2pt on MMLU/ARC.
Bigger drops → reconsider method, bits, or calibration data.

---

## The standard quality run (do this for every quant you make)

1. **PPL** quant vs fp16, same harness → smoke test (`perplexity.py` / `compare_quality.py`)
2. **KL + top-1 agreement** vs fp16 → sensitive regression catch
3. **Golden-prompt generations** side by side → catch repetition/format pathologies
4. **lm-eval** on mmlu + arc + hellaswag + one domain task → capability check

A quant that passes all four within the thresholds above is safe to ship.

## Common failure signatures (and the usual cause)

| Symptom | Likely cause |
|---|---|
| PPL explodes / NaN | zero scale, a layer left unquantized, overflow in low-bit accumulation |
| PPL fine, task accuracy tanks | calibration/distribution mismatch, or sub-4-bit knowledge loss |
| High tail KL, low top-1 agreement, OK mean | specific channels/layers mis-quantized; outlier handling failed |
| Repetition loops / degenerate output | activation outliers not smoothed (needs SmoothQuant), or wrong calib format |
| Fine on calib-like text, collapses on code/other language | calibration set didn't cover deployment distribution |
