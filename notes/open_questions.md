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
