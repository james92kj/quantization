# Open questions — to revisit

A running list of things to come back to and fully internalize.

---

## Q1. Why is the imatrix gain "statistically real" even though the ± error bars nearly overlap?

**Context:** Phase 3 result — naive Q4_K_M = 10.5501 ± 0.074 vs imatrix Q4_K_M = 10.4191 ± 0.073.
The gap is 0.131 (~1.2%), but the ±0.07 bars almost touch. So is the improvement real or noise?

**Short answer: real — because it's a PAIRED comparison.**

- The **±0.07 is not "uncertainty about the model"** — it's how much perplexity *bounces* from
  passage to passage (the running PPL swung 10→18→15 across the ~580 chunks). It's text-difficulty
  variation, and it is **shared by both models** (they ran on the exact same `wiki.test.raw`).
- **Runner analogy:** two runners whose times swing ±5 min day-to-day (weather). Timed on
  *different* days, a 1-min gap is meaningless. Raced *side by side, same hills, same weather*, and
  one wins by 1 min on every hill → that 1 min is rock solid. The ±5 min was shared and **cancels**.
- Same here: naive and imatrix saw the *same* hard/easy passages; on essentially every passage the
  imatrix model was a hair lower. The shared passage-difficulty variation cancels, so the **0.13 gap
  is far more certain than the individual ±0.07 implies.** Rule "overlapping bars = no difference"
  applies to *independent* measurements on *different* data — not to paired ones.

**To do later:** actually compute the per-chunk paired difference (or run `llama-perplexity` with KL
/ a paired stats output) to *show* the consistency rather than argue it. Also revisit:
- Why the gain is only ~1.2% (we're at 5 bpw — little damage to undo; calibration's payoff grows
  sharply at 2–3 bit, which is why I-quants *require* an imatrix). [Photo-compression analogy.]
- The big-picture link: imatrix *measures* importance; GPTQ also *compensates* (Hessian), AWQ
  *rescales* salient channels. Same idea, more powerful → Phase 4.

---

## Q2. "Sliding Window Attention is enabled but not implemented for `sdpa`" — what does it mean, and how would we implement SWA?

**Context:** Phase 4a, loading `Qwen2.5-1.5B-Instruct` for perplexity on the L4. transformers
printed this warning at load. Not an error — eval ran fine (PPL 8.6453). Parked to tackle later.

**The questions to answer:**
1. What *is* sliding-window attention (SWA)? (Each token attends only to the last `W` tokens
   instead of all previous tokens → attention cost goes from O(n²) to O(n·W); the trick behind
   long-context models like Mistral / Qwen's long-context configs.)
2. What does "enabled but not implemented for `sdpa`" mean? (The model *config* declares a sliding
   window, but the attention backend in use — PyTorch's `scaled_dot_product_attention`, "sdpa" —
   doesn't apply the window mask, so it silently runs *full* attention instead. Hence "unexpected
   results may be encountered." For Qwen2.5-1.5B the configured window is large / only kicks in past
   a length threshold, so for our 2048-token eval windows it's a no-op — which is why our number is
   still trustworthy.)
3. How do you actually get SWA applied? (Pick an attention backend that implements the window —
   e.g. `attn_implementation="flash_attention_2"`, or eager with an explicit sliding mask — and/or
   confirm the threshold at which the window engages. Worth a small experiment: does forcing FA2
   change PPL for long contexts?)

**Why it matters for us:** it's an *attention* detail, not a *quantization* one, but it touches our
perplexity ruler — if a window silently does/doesn't apply, long-context PPL could shift. For Phase 4
our 2048-token windows are below the threshold, so it doesn't affect the bnb fp16-vs-int8-vs-nf4
comparison. Revisit when we care about long-context eval.

---

## Q3. What *is* `LLM.int8()` — completely? (the 8-bit method behind bnb `load_in_8bit=True`)

**Context:** Phase 4a. 8-bit bnb gave PPL 8.6869 vs fp16 8.6453 (+0.48%, near-lossless) at 1.20
bytes/param, but eval got *slower* (96s vs 62s). Want a full mental model of *why* it's lossless
and *why* it's slow. Paper: Dettmers et al., 2022, "LLM.int8(): 8-bit Matrix Multiplication for
Transformers at Scale" (arXiv 2208.07339).

**The questions to answer (deep dive):**
1. **The core problem it solves:** why does naive int8 (per-tensor/per-row absmax) *break* on large
   transformers? → **emergent activation outliers**: a few feature dimensions develop huge magnitudes
   (>6σ) past ~6.7B params; one outlier blows up the quantization scale for its whole row and crushes
   all the normal values to near-zero. (Tie back to Phase 3 imatrix / AWQ: outliers are the recurring
   villain of quantization.)
2. **The mixed-precision decomposition (the actual trick):** split each matmul's columns into a tiny
   "outlier" set (kept in **fp16**) and the rest (quantized to **int8**, vector-wise / per-row+per-col
   absmax). Do two matmuls — int8 for the bulk, fp16 for the ~0.1% outlier columns — then sum. That's
   why it's near-lossless: the dangerous dimensions never get quantized.
3. **Why 1.20 bytes/param, not 1.0:** the int8 weights + per-vector fp16 scales + the fp16 outlier
   columns all cost storage → the overhead we measured.
4. **Why it's *slower* not faster:** the decompose → two-path matmul → recombine has overhead that, on
   small models, outweighs the int8 compute win. bnb int8 buys **memory, not speed**. (Contrast: GPTQ/
   AWQ fused 4-bit kernels *can* speed up. Confirm empirically in 4b/4c.)
5. **vs NF4 (`load_in_4bit`):** different method entirely — NF4 is a 4-bit *NormalFloat* datatype +
   double-quant, no outlier side-path. When do you pick 8-bit (int8) vs 4-bit (NF4)? (quality budget
   vs memory budget; NF4 is the QLoRA default.)
6. **Relation to QLoRA:** NF4 (not int8) is the QLoRA base; understand why 4-bit + LoRA adapters, not
   8-bit, became the fine-tuning standard.

**Deliverable for the deep dive:** be able to draw the decomposition diagram from memory and explain,
in one sentence each, *why lossless* and *why slow*. Possibly read the paper via the paper-reader.
