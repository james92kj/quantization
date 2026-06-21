# Scoreboard

Running record of every model/precision we measure. **The whole game: shrink the model
(smaller size) while keeping perplexity close to the fp16 baseline.**

> All perplexity = WikiText-2 (`Salesforce/wikitext`, raw, test split), sliding window
> (max_len 2048, stride 1024), measured with `06_evaluation/perplexity.py`.
> Compare only rows measured the **same way**.

## Qwen2.5-0.5B-Instruct (work horse)

| precision | size on disk | WikiText-2 PPL | Δ vs fp16 | tok/s | verdict |
|---|---|---|---|---|---|
| **fp16 (baseline)** | 953 MB | **12.67** (full test) | — | — | reference |
| MLX 4-bit | 276 MB (**3.5× smaller**) | _Phase 2_ | _Phase 2_ | | |

## Notes
- A quick run with `--max-tokens 20000` on the same fp16 model gives ~11.6 — a *subset*
  number, not comparable to the full-test 12.67. Always match settings before comparing.
- Bigger work horses (Qwen2.5-3B / 7B) to be added as later rows once downloaded — same pipeline.
