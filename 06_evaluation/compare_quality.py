"""
Measure the QUALITY of a quantized model vs its fp16 baseline — for real.

This is the core "did my quantization hurt the model?" harness. It runs the
4 layers of quality measurement, from cheapest/least-sensitive to most:

  1. PERPLEXITY  — does it still model language? (cheap, coarse)
  2. KL DIVERGENCE + TOP-1 AGREEMENT — does it produce the SAME distribution
     as fp16? (sensitive: catches damage perplexity misses)
  3. SIDE-BY-SIDE GENERATION — does it still *sound* right on real prompts?
     (catches repetition loops / format drift that metrics miss)
  4. (separate) lm-eval-harness for task accuracy — see HOW_TO_MEASURE_QUALITY.md

The golden rule: a quantized model is only judged RELATIVE to its own fp16
baseline, measured with the EXACT same harness. Absolute numbers are meaningless
across different code/strides/datasets.

Works on mps (Mac) / cuda / cpu.

Example (Mac, comparing fp16 baseline vs an 8-bit-ish quantize via bitsandbytes
is CUDA-only — on Mac compare fp16 vs a smaller dtype or an MLX/GGUF export
through their own tools; this script is the generic HF-vs-HF comparer):

    python compare_quality.py \
        --baseline Qwen/Qwen2.5-0.5B-Instruct \
        --quantized ./my-quantized-model-dir \
        --device mps --max-tokens 20000
"""
import argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from perplexity import eval_perplexity   # reuse the harness next to this file

PROMPTS = [
    "Explain what model quantization is in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
]


def load(model_id, device, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    # Quantized models loaded via device_map already sit on GPU; guard the .to()
    if not getattr(model, "hf_device_map", None):
        model = model.to(device)
    return model.eval(), tok


@torch.no_grad()
def kl_and_agreement(ref_model, q_model, tok, device,
                     n_chunks=16, seq_len=512):
    """Mean KL(P_ref || P_quant), tail (p99) KL, and top-1 agreement %, over
    WikiText-2 text. Both models must share a tokenizer/vocab (same base model)."""
    from datasets import load_dataset
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids[0]

    kls, agree, total = [], 0, 0
    for c in range(n_chunks):
        chunk = ids[c * seq_len:(c + 1) * seq_len]
        if chunk.numel() < 8:
            break
        x = chunk.unsqueeze(0).to(device)
        logp_ref = F.log_softmax(ref_model(x).logits.float(), dim=-1)
        logp_q = F.log_softmax(q_model(x).logits.float(), dim=-1)
        kl = (logp_ref.exp() * (logp_ref - logp_q)).sum(-1)   # [1, T]
        kls.append(kl.flatten().cpu())
        agree += (logp_ref.argmax(-1) == logp_q.argmax(-1)).sum().item()
        total += x.numel()
    allkl = torch.cat(kls)
    return {
        "mean_kl": allkl.mean().item(),
        "p99_kl": allkl.quantile(0.99).item(),
        "max_kl": allkl.max().item(),
        "top1_agreement_pct": 100.0 * agree / total,
    }


@torch.no_grad()
def generate(model, tok, device, prompt, max_new=80):
    if tok.chat_template:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inp = tok(text, return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="fp16 reference model id/path")
    p.add_argument("--quantized", required=True, help="quantized model id/path")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-tokens", type=int, default=20000,
                   help="cap PPL tokens for speed while iterating")
    p.add_argument("--skip-kl", action="store_true",
                   help="skip KL (needs both models resident at once — RAM heavy)")
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"\n{'='*70}\n1. PERPLEXITY (lower=better; compare the delta)\n{'='*70}")
    base, tok = load(args.baseline, args.device, dtype)
    t0 = time.time()
    ppl_base = eval_perplexity(base, tok, device=args.device, max_tokens=args.max_tokens)
    print(f"  baseline   {args.baseline}: {ppl_base:.4f}  ({time.time()-t0:.0f}s)")

    print(f"\n  -- baseline generations (golden set) --")
    gens_base = {q: generate(base, tok, args.device, q) for q in PROMPTS}
    for q, a in gens_base.items():
        print(f"   Q: {q}\n   A: {a}\n")

    if not args.skip_kl:
        print(f"{'='*70}\n2. KL DIVERGENCE vs baseline (0=identical; sensitive)\n{'='*70}")
    # load quantized; keep baseline for KL if requested
    quant, tokq = load(args.quantized, args.device, dtype)
    t0 = time.time()
    ppl_q = eval_perplexity(quant, tokq, device=args.device, max_tokens=args.max_tokens)

    if not args.skip_kl:
        try:
            stats = kl_and_agreement(base, quant, tok, args.device)
            print(f"  mean KL: {stats['mean_kl']:.5f}   p99 KL: {stats['p99_kl']:.5f}"
                  f"   max KL: {stats['max_kl']:.4f}")
            print(f"  top-1 agreement: {stats['top1_agreement_pct']:.2f}%")
        except Exception as e:
            print(f"  (KL skipped: {e})")

    print(f"\n{'='*70}\n3. PERPLEXITY DELTA + GENERATIONS\n{'='*70}")
    pct = 100.0 * (ppl_q - ppl_base) / ppl_base
    print(f"  baseline  PPL: {ppl_base:.4f}")
    print(f"  quantized PPL: {ppl_q:.4f}   (Δ {ppl_q-ppl_base:+.4f}, {pct:+.2f}%)  ({time.time()-t0:.0f}s)")
    verdict = ("LOSSLESS-ish" if pct < 1 else "GOOD" if pct < 5 else
               "DEGRADED" if pct < 15 else "BROKEN")
    print(f"  verdict (PPL heuristic): {verdict}")

    print(f"\n  -- quantized generations (compare to golden set above) --")
    for q in PROMPTS:
        print(f"   Q: {q}\n   base : {gens_base[q]}\n   quant: {generate(quant, tokq, args.device, q)}\n")


if __name__ == "__main__":
    main()
