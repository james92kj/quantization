# The research-engineer loop (the method this project trains)

The point of this repo isn't the numbers — it's the *habit* that produces trustworthy numbers. Stated
in `00_roadmap/ROADMAP.md` ("we interpret; when a number surprises us, we dig"), it's the standard
empirical method applied to quantization:

> **predict → measure → get surprised → dig → explain**

1. **Predict** — commit to an expected number *before* running. A prediction makes a surprise visible;
   without one, a wrong result looks normal.
2. **Measure** — run on a fixed ruler (same model, data, window). Comparable measurement or it's noise.
3. **Get surprised** — notice when measurement ≠ prediction. The surprise is a *signal*, not an
   annoyance. Especially: a result that violates a known ordering (e.g. a *calibrated* method losing to
   an *uncalibrated* one) is almost always a setup bug, not a law of nature.
4. **Dig** — find the cause with a cheap diagnostic before spending a big run. (Check inputs, lengths,
   dtypes, configs.) Don't re-run blindly; isolate one variable.
5. **Explain** — write down the mechanism and, when it's a factual claim, verify it against sources.
   Record the honest result even when it contradicts your prediction.

## Worked examples — both lived in Phase 4b (one session)

**Surprise A — calibration bug.**
- *Predict:* GPTQ (calibrated) beats NF4 (not). *Measure:* GPTQ PPL 11.95 — **worse** (+38%).
- *Dig:* a 6-line diagnostic showed calibration samples had median **158 tokens**, not the intended
  2048 (WikiText rows are short lines). *Explain:* stubby sequences ruin GPTQ's Hessian estimate.
  Fixed to concatenate-and-chunk → **9.44**, recovering 2.5 PPL. The "law" was never broken; our setup was.

**Surprise B — act-order overshoot.**
- *Predict:* literature says act-order ≈ +0.1 PPL on small models — maybe not enough to beat NF4.
- *Measure:* act-order gave **9.07** (a 0.37 gain) — GPTQ **beats** NF4.
- *Dig/Explain:* verified the prediction against the GPTQ paper's own numbers (13B saw ~0.1), so our
  1.5B benefiting *more* is a genuine data point, recorded as-is. Real data beat the informed prior.

## Why it matters
Tutorial-followers report the first number they get. Practitioners predict, get surprised, and dig —
which is how you catch the calibration bug instead of publishing "GPTQ is worse than NF4." Every
surprise this project hits gets this treatment; that's the whole game. See also
[`open_questions.md`](open_questions.md) (the parked "dig later" list) and
[`handling-warnings.md`](handling-warnings.md) (triage as a sub-skill of step 4).
