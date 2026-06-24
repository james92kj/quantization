"""
Phase 4b — GPTQ (via llm-compressor) on CUDA.

GPTQ vs NF4 (Phase 4a): NF4 rounded each weight independently, blind to the rest.
GPTQ quantizes a layer one weight at a time and, after rounding each weight, ADJUSTS the
remaining (not-yet-quantized) weights to cancel the error it just introduced. To know how,
it estimates which weights matter from a small CALIBRATION SET of real text (second-order /
Hessian info). Same 4 bits, but roundings chosen to minimize the layer's OUTPUT error.
=> should beat NF4's +7.66% at the same bit-width. (This is the Phase-3 imatrix idea taken
further: imatrix MEASURED importance; GPTQ measures importance AND compensates.)

Library note: GPTQModel (the usual lib) is unbuildable here (broken `pcre` build dep + no nvcc),
so we use llm-compressor, which runs the SAME GPTQ algorithm in pure PyTorch. Scheme W4A16 =
4-bit weights / 16-bit activations, group_size 128 (compressed-tensors preset).

METHODOLOGY: calibrate on WikiText-2 *train*, evaluate on *test*. Never calibrate on the eval
set -- that would leak and flatter the score.

Run (on the GPU VM):
    python3 gptq_quantize.py            # quantize + save + eval (~10-15 min on an L4)
    python3 gptq_quantize.py --eval-only  # just re-eval an already-saved model
"""
import argparse, os, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from perplexity import eval_perplexity   # sits next to this file on the VM


def build_calibration(tokenizer, n_samples, seqlen):
    """N FULL-LENGTH samples from WikiText-2 TRAIN (not test).

    WikiText rows are short lines (~150 tokens). Tokenizing per-row gives stubby
    sequences that ruin GPTQ's Hessian estimate. The right recipe: concatenate the
    whole corpus, then slice it into real `seqlen`-token chunks -- the same way the
    perplexity harness chunks the test set. Returns a HF Dataset of `input_ids`."""
    from datasets import Dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks = []
    for i in range(0, n_samples * seqlen, seqlen):
        chunk = ids[i:i + seqlen]
        if len(chunk) < seqlen:
            break
        chunks.append({"input_ids": chunk.tolist()})
    print(f"  built {len(chunks)} calibration chunks of {seqlen} tokens each")
    return Dataset.from_list(chunks)


def dir_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default=None)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--damp", type=float, default=0.01, help="GPTQ dampening_frac")
    p.add_argument("--actorder", action="store_true",
                   help="enable activation ordering (desc_act): quantize important columns first")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()

    # separate output dir per setting so runs don't overwrite each other
    if args.out is None:
        args.out = os.path.expanduser(
            "~/qwen2.5-1.5b-gptq-w4a16" + ("-actorder" if args.actorder else ""))

    tok = AutoTokenizer.from_pretrained(args.model)

    if not args.eval_only:
        print(f"Loading {args.model} (fp16) for GPTQ calibration ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cuda")

        print(f"Building calibration set: {args.samples} samples x {args.seqlen} tok (WikiText-2 train) ...")
        calib = build_calibration(tok, args.samples, args.seqlen)

        # The GPTQ recipe. W4A16 = 4-bit weights, group 128. lm_head left in fp16.
        recipe_kwargs = dict(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
            dampening_frac=args.damp,
        )
        if args.actorder:
            recipe_kwargs["actorder"] = "group"   # quantize columns by descending activation importance
        recipe = GPTQModifier(**recipe_kwargs)

        print("Running GPTQ (one layer at a time, error-compensating) ...")
        t0 = time.time()
        oneshot(
            model=model,
            dataset=calib,
            recipe=recipe,
            num_calibration_samples=args.samples,
            max_seq_length=args.seqlen,
            output_dir=args.out,
        )
        print(f"GPTQ done in {(time.time()-t0)/60:.1f} min  ->  saved to {args.out}")
        del model
        torch.cuda.empty_cache()

    # Reload the saved 4-bit model from disk and measure quality on the HF ruler.
    print(f"\nReloading saved model from {args.out} for eval ...")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.out, device_map="cuda")
    t1 = time.time()
    ppl = eval_perplexity(model, tok, device="cuda", max_tokens=args.max_tokens)
    eval_s = time.time() - t1
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("\n" + "=" * 56)
    print(f"  model        {args.model}")
    print(f"  method       GPTQ W4A16 (group 128, damp {args.damp}, actorder={args.actorder})")
    print(f"  calibration  {args.samples} x {args.seqlen} tok, WikiText-2 train")
    print(f"  on-disk size {dir_size_gb(args.out):.3f} GB")
    print(f"  peak VRAM    {peak_gb:.3f} GB   (during eval)")
    print(f"  WikiText-2 PPL  {ppl:.4f}")
    print(f"  eval {eval_s:.1f}s")
    print("=" * 56)


if __name__ == "__main__":
    main()
