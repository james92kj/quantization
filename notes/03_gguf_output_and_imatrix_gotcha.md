# Reading llama.cpp output + the 256-divisibility gotcha (Phase 3 setup)

Notes from quantizing with `llama-quantize` / measuring with `llama-perplexity`, and a
real-world surprise that shaped how we run the calibration (imatrix) experiment.

---

## How to read `llama-quantize` output (4 sections)

1. **Backend loading** (`ggml_metal_*` / `load_backend` lines) — llama.cpp detecting hardware.
   Key line on Mac: `GPU name: Apple M4`, `has unified memory = true`,
   `recommendedMaxWorkingSetSize ~11453 MB` (≈ usable GPU memory). Startup noise.

2. **Model metadata dump** (`llama_model_loader: - kv ...`) — the model's spec card:
   `architecture`, `block_count` (layers), **`embedding_length`** (hidden size — matters below),
   `vocab`, tokenizer info. Informational.

3. **Per-tensor conversion log** (`[ N/291] name - [shape], type=f16, converting to <type> .. size X -> Y`)
   — one line per tensor showing what each weight matrix was quantized to and how much it shrank.

4. **Final summary**
   ```
   model size = 1202 MiB (16.00 BPW)   <- the fp16 input
   quant size =  462 MiB ( 6.16 BPW)   <- the result   (bits-per-weight!)
   WARNING: 145 of 291 tensor(s) required fallback quantization
   ```
   `BPW` = actual bits/weight. **If it's far above the nominal (Q4_K_M ≈ 4.85), read the warnings.**

## How to read `llama-perplexity` output
- Splits the text into **512-token chunks**; prints a **running** perplexity `[n]value` after each
  chunk (cumulative average — noisy early, converges later; the wobble is normal).
- Final line: `Final estimate: PPL = 16.2043 +/- 0.12399` → the value over the whole set, with the
  **standard error** (±). Differences smaller than the ± are not meaningful.

---

## 🚨 The gotcha: K-quants need dimensions divisible by 256

**K-quants and I-quants tile each row into super-blocks of 256 weights.** If a weight matrix's
input dimension isn't divisible by 256, that tensor **cannot** use `q4_K`/`q6_K` and **falls back
to a legacy quant** (`q5_0`, `q8_0`) — which is *more* bits.

**Real example — Qwen2.5-0.5B-Instruct** has hidden size **896**, and `896 / 256 = 3.5` ❌.
So nearly every attention + ffn_gate/up tensor (input dim 896) fell back to `q5_0`/`q8_0`.
Only `ffn_down` (input dim 4864 = 19×256 ✓) stayed K-quant. Result:
**145 of 291 tensors fell back**, and "Q4_K_M" came out at **6.16 bpw / 462 MB** instead of true 4-bit.

### Why it ruins the calibration demo on this model
**Legacy quants (`q5_0`, `q8_0`) completely ignore the imatrix.** The importance matrix only steers
**K-quants and I-quants.** With most of the 0.5B fallen back to legacy, an imatrix would only touch
the few `ffn_down` tensors → the calibration effect would be **muted and hard to see.**

### The fix → use a 256-divisible model
We pivoted to **Qwen2.5-1.5B-Instruct**: hidden size **1536 = 6×256 ✓**, ffn **8960 = 35×256 ✓**.
Every tensor satisfies the rule → no fallbacks → `Q4_K_M` stays true K-quant → the imatrix actually
bites, so the naive-vs-calibrated perplexity gap is clean and visible.

> **Rule of thumb:** for GGUF K-quant experiments, pick models whose hidden + ffn dims are multiples
> of 256 (most ≥1B models: hidden 1536, 2048, 2304, 4096…). Tiny models with "odd" dims (896) don't
> quantize cleanly and shouldn't be used to study K-quant/imatrix behavior.

---

## Reproducing (text files used for both eval and calibration)
```python
from datasets import load_dataset
te = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
open("wiki.test.raw","w").write("\n\n".join(t for t in te["text"] if t.strip()))  # eval
tr = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
buf=[]; n=0
for t in tr["text"]:
    if t.strip(): buf.append(t); n+=len(t)
    if n>200000: break
open("calib.txt","w").write("\n\n".join(buf))                                     # calibration (separate split)
```
Eval split (test) and calibration split (train) are kept **separate** so we don't calibrate on the
text we measure on. The A/B itself (naive `Q4_K_M` vs `Q4_K_M` + imatrix) is in
[[scoreboard]] once measured.
