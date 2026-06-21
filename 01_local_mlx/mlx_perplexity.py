"""
Perplexity harness for MLX models (Apple Silicon native).

mlx-lm has no built-in perplexity tool, so we compute it ourselves — the SAME
definition as the HF harness (exp of mean per-token negative log-likelihood on
WikiText-2), so fp16 and the 4-bit MLX model are judged identically.

We use simple non-overlapping windows of `ctx` tokens (fast, and perfectly fair
as long as BOTH models use the exact same setting — which they do here).

Usage:
    python mlx_perplexity.py --model Qwen/Qwen2.5-0.5B-Instruct          # fp16 (mlx loads & runs it)
    python mlx_perplexity.py --model ./qwen2.5-0.5b-4bit-g64             # our 4-bit export
    # add --max-tokens 60000 for a quick pass
"""
import argparse, math, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from datasets import load_dataset


def eval_ppl(model, tokenizer, ctx=2048, max_tokens=None):
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer.encode("\n\n".join(data["text"]))
    if max_tokens:
        ids = ids[:max_tokens]

    total_nll, total_tok = 0.0, 0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1]          # +1 so the last input token has a target
        if len(chunk) < 2:
            break
        x = mx.array(chunk[:-1])[None]      # [1, L]  inputs
        y = mx.array(chunk[1:])[None]       # [1, L]  true next tokens
        logits = model(x).astype(mx.float32)        # [1, L, V]
        # sum of per-token negative log-likelihood over this window
        nll = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(nll)
        total_nll += nll.item()
        total_tok += y.size
    return math.exp(total_nll / total_tok), total_tok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id (fp16) or local MLX dir (quantized)")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()

    print(f"Loading {args.model} via mlx-lm ...")
    model, tokenizer = load(args.model)
    t0 = time.time()
    ppl, n = eval_ppl(model, tokenizer, ctx=args.ctx, max_tokens=args.max_tokens)
    print(f"WikiText-2 perplexity (MLX): {ppl:.4f}   over {n} tokens   ({time.time()-t0:.0f}s)")
