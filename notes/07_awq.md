# Phase 4c — AWQ (activation-aware weight quantization) on CUDA

Model **Qwen2.5-1.5B-Instruct**, HF ruler (comparable to the 4a/4b bnb+GPTQ rows). fp16 anchor =
8.6453. Via **llm-compressor** `AWQModifier` (compile-free, same toolchain as GPTQ). Script:
[`04_cuda_gptq_awq/awq_quantize.py`](../04_cuda_gptq_awq/awq_quantize.py).

## The idea — the third take on "importance"
- imatrix (Ph3) **measured** importance; GPTQ (4b) **compensates** for it; **AWQ protects** it.
- AWQ scales salient weight channels up (and the feeding activations down — math cancels) so important
  channels land on a finer part of the 4-bit grid. Needs calibration to find salient channels.
- "Mappings" tell AWQ which preceding module absorbs each scale — see [`awq-mappings-explained.md`](awq-mappings-explained.md)
  and `open_questions.md` Q4. For Qwen2 they auto-resolve into 4 patterns per block.

## ⚠️ Two failures getting here (see journey log)
1. Full 256×2048 run crashed; **Qwen2.5 tied embeddings** → llm-compressor wrote a scale into the
   shared `lm_head`/`embed_tokens` tensor (8960 vs 151936). **Fix: untie before quantizing.**
2. (First mis-diagnosed as memory/offload; retry at 128×512 failed identically → it was deterministic,
   the tie. Verified the hypothesis was wrong before trusting it.)

## Result — fair comparison (calibration matched at 256×2048)

| 4-bit method | calibration | PPL | Δ vs fp16 (8.6453) |
|---|---|---|---|
| GPTQ + act-order | 256×2048 | **9.0716** | +4.93% |
| NF4 (bnb) | none | 9.3078 | +7.66% |
| GPTQ (no act-order) | 256×2048 | 9.4401 | +9.19% |
| **AWQ** | 128×512 | 10.0046 | +15.7% |
| **AWQ** | **256×2048** | **10.0046** | **+15.7%** |

**AWQ is calibration-INSENSITIVE — confirmed decisively.** 128×512 and 256×2048 gave the *identical*
PPL (10.0046, to 4 decimals). 8× more calibration data moved nothing. (It needs activation *magnitude*
statistics, which converge almost instantly — unlike GPTQ's Hessian, which *required* the 2048-tok fix.)

**So AWQ trailing is NOT about data — it's the scheme.** With calibration matched, the only remaining
confound is our **symmetric** `W4A16` vs original AWQ's **asymmetric** (zero-point) quant. That is now
the clearly-isolated next experiment: re-run AWQ with an asymmetric 4-bit scheme and re-compare.

**To make it fair (next):** re-run AWQ at 256×2048 (matching GPTQ); optionally try an asymmetric 4-bit
scheme. Only then compare AWQ vs GPTQ vs NF4 as a verdict.

## Why the fair re-run (256×2048) — decision recorded before the result

**The question we're actually asking** is "AWQ *the method* vs GPTQ *the method*." But the preliminary
runs differed in TWO things at once: the method *and* the calibration budget (AWQ 128×512 = 65K tok vs
GPTQ 256×2048 = 524K tok, 8× more). Comparing them as-is conflates "AWQ vs GPTQ" with "less data vs
more data" — you can't attribute the 10.00-vs-9.07 gap to the method.

**The fix is single-variable isolation** (same discipline as the GPTQ act-order experiment): hold
calibration *constant* at 256×2048, change only the method. Then any remaining gap is the method's
doing, not the data budget's.

**Prediction:** AWQ is largely calibration-*insensitive* (it needs activation magnitudes, not a full
Hessian), so we expect only a small move (~0.1). If AWQ stays well above NF4/GPTQ even with matched
calibration, that points the finger at the *remaining* confound — our **symmetric** W4A16 scheme vs
AWQ's usual **asymmetric** quant — as the real cause, and sets up that as the next experiment.

**Still NOT controlled (acknowledged):** the symmetric-vs-asymmetric scheme. One confound at a time.

## Honest takeaway so far
On this 1.5B at 4-bit, with matched-ish effort: **GPTQ+act-order (9.07) leads, NF4 (9.31) is a strong
zero-effort baseline, AWQ (prelim 10.00) needs a fair re-run before judging.** The *method* lessons
(measure importance → compensate → protect; calibration data quality; tied-embedding pitfalls) matter
more than this single leaderboard.
