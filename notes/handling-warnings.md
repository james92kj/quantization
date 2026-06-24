# Handling log noise — a triage checklist

Most of what scrolls past during ML work is benign. Triaging it fast is a real skill. The danger is
never the warning itself — it's ignoring one *without reading it*. A warning is **guilty until you
prove it innocent**, but most prove innocent in under a minute.

## The checklist (in order)

1. **Error or warning?** Look for `Traceback` / `Error`. A traceback **stopped** the program → must
   fix. A *warning* means execution continued → you get to *decide*.
2. **Did you still get a sane result?** A plausible output (e.g. a perplexity of 8.6, not `nan` or
   50000) means the warning didn't silently wreck the run.
3. **Read the literal claim, then check if its precondition is true for YOUR usage.** Many warnings
   describe a failure mode that doesn't apply to how you're calling the code.
4. **Locate who emitted it** — which library, which call. A *tokenizer* warning is not a *model*
   warning. Knowing the source usually settles whether it matters.
5. **Decide:** ignore / suppress / fix / investigate. Suppress only AFTER you understand it — silencing
   an un-understood warning is how real bugs hide.

## Worked examples (Phase 4)

### "Token indices sequence length is longer than the specified maximum (299078 > 131072)"
- **Source:** the *tokenizer*, fired reflexively because we tokenized the entire concatenated WikiText
  corpus in one call.
- **Literal claim:** "running *this sequence* through the model will cause indexing errors."
- **Precondition true for us?** No. We never forward the 299k-token sequence — the perplexity harness
  slices it into 2048-token windows first. The harmful condition never happens.
- **Verdict: ignore.**

### "Sliding Window Attention is enabled but not implemented for `sdpa`"
- **Source:** the *model* — config declares a sliding window, but the `sdpa` attention backend doesn't
  apply the window mask, so it silently runs full attention.
- **Precondition true for us?** SWA only engages past a length threshold; our eval windows are 2048
  tokens, below it → attention is correct → our number stands. (Deeper dive parked as Q2 in
  [`open_questions.md`](open_questions.md).)
- **Verdict: ignore for short-context eval; revisit for long-context.**

### Counter-example — a `Traceback`, NOT a warning
`OSError: undefined symbol: torch_library_impl` (torchaudio) **stopped** the run → had to fix (pin
`transformers==4.49.0`). Step 1 of the checklist catches this immediately: it's an error, not noise.
