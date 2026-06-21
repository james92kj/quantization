"""
Reusable perplexity harness (sliding window) for HF causal LMs.

Perplexity = exp(mean negative log-likelihood per token).
Lower is better. This is THE reference metric for quantization quality:
you always compare quantized PPL against the fp16 baseline PPL on the SAME
text, same window, same stride.

Why a sliding window: a long document is longer than the context window, so we
slide a window of `max_len` with overlap `stride`, and only score the new
(non-overlapping) tokens each step so nothing is double-counted and every token
is predicted with full left-context.

Works on Apple Silicon (mps), CUDA, or CPU. Used by every phase.

Usage:
    from perplexity import eval_perplexity
    ppl = eval_perplexity(model, tokenizer, device="mps")
"""
import torch
from datasets import load_dataset


@torch.no_grad()
def eval_perplexity(model, tokenizer, device=None,
                    dataset_name="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1",
                    split="test", max_len=2048, stride=1024, max_tokens=None):
    """Compute sliding-window perplexity on WikiText-2 (default).

    Returns a float (perplexity). Set max_tokens to cap runtime while iterating.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    data = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(data["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    n_tokens = input_ids.size(1)
    if max_tokens:
        n_tokens = min(n_tokens, max_tokens)

    nlls, total_scored = [], 0
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + max_len, n_tokens)
        trg_len = end - prev_end          # number of NEW tokens to actually score
        ids = input_ids[:, begin:end]
        targets = ids.clone()
        targets[:, :-trg_len] = -100      # ignore the overlapping context tokens

        out = model(ids, labels=targets)
        # HF returns mean loss over the (trg_len) scored tokens; rescale to a sum.
        neg_log_likelihood = out.loss * trg_len
        nlls.append(neg_log_likelihood)
        total_scored += trg_len
        prev_end = end
        if end == n_tokens:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_scored)
    return ppl.item()


if __name__ == "__main__":
    import argparse, time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="mps")        # mps | cuda | cpu
    p.add_argument("--dtype", default="float16")     # float16 | bfloat16 | float32
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap tokens for a fast smoke run, e.g. 20000")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading {args.model} as {args.dtype} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)

    t0 = time.time()
    ppl = eval_perplexity(model, tok, device=args.device, max_tokens=args.max_tokens)
    print(f"WikiText-2 perplexity: {ppl:.4f}   ({time.time()-t0:.1f}s)")
