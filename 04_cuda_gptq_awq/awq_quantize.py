"""
Phase 4c — AWQ (Activation-aware Weight Quantization) via llm-compressor, on CUDA.

The third take on the importance theme (imatrix MEASURED it, GPTQ COMPENSATES for it):
AWQ PROTECTS it. A small fraction of weight *channels* are "salient" -- they get multiplied by
large-magnitude activations, so quantization error there hurts the output most. AWQ finds those
channels from a calibration set and SCALES THEM UP before quantizing (compensating by scaling the
matching activations down). Salient channels land on a finer part of the 4-bit grid -> protected.
Unlike GPTQ there's no error-feedback loop; it's a per-channel rescale ("smoothing") + plain quant.

Same scheme as 4b (W4A16, group 128, lm_head fp16), same calibration (2048-tok WikiText-2 train
chunks), same HF-ruler eval -> directly comparable to the NF4 / GPTQ rows.

Run (on the GPU VM):
    # smoke test first (cheap, ~2-3 min) -- validate the pipeline before the real run:
    python3 awq_quantize.py --samples 16 --seqlen 512 --max-tokens 20000
    # full run (~10-12 min):
    python3 awq_quantize.py
"""
import argparse, os, time
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from perplexity import eval_perplexity   # sits next to this file on the VM


def build_calibration(tokenizer, n_samples, seqlen):
    """N FULL-LENGTH samples from WikiText-2 TRAIN: concatenate the corpus, then slice into
    real `seqlen`-token chunks (same recipe that fixed GPTQ in 4b). Returns a HF Dataset."""
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
    p.add_argument("--out", default=os.path.expanduser("~/qwen2.5-1.5b-awq-w4a16"))
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    if not args.eval_only:
        print(f"Loading {args.model} (fp16) for AWQ calibration ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cuda")

        # Qwen2.5 ties lm_head to embed_tokens. llm-compressor's AWQ scale-update follows the tie and
        # crashes writing a scale into the shared tensor (shape 8960 vs vocab 151936). Untie by giving
        # lm_head its own copy so ignore=["lm_head"] excludes it cleanly. Numerically identical model.
        if getattr(model.config, "tie_word_embeddings", False):
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
            model.config.tie_word_embeddings = False
            print("  untied lm_head from embed_tokens (AWQ tied-embedding workaround)")

        print(f"Building calibration set: {args.samples} samples x {args.seqlen} tok (WikiText-2 train) ...")
        calib = build_calibration(tok, args.samples, args.seqlen)

        # AWQ recipe. Same W4A16 target as GPTQ; AWQ protects salient channels via scaling.
        recipe = AWQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
        )

        print("Running AWQ (find salient channels -> scale -> quantize) ...")
        t0 = time.time()
        oneshot(
            model=model,
            dataset=calib,
            recipe=recipe,
            num_calibration_samples=args.samples,
            max_seq_length=args.seqlen,
            output_dir=args.out,
        )
        print(f"AWQ done in {(time.time()-t0)/60:.1f} min  ->  saved to {args.out}")
        del model
        torch.cuda.empty_cache()

    print(f"\nReloading saved model from {args.out} for eval ...")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.out, device_map="cuda")
    t1 = time.time()
    ppl = eval_perplexity(model, tok, device="cuda", max_tokens=args.max_tokens)
    eval_s = time.time() - t1
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("\n" + "=" * 56)
    print(f"  model        {args.model}")
    print(f"  method       AWQ W4A16 (group 128)")
    print(f"  calibration  {args.samples} x {args.seqlen} tok, WikiText-2 train")
    print(f"  on-disk size {dir_size_gb(args.out):.3f} GB")
    print(f"  peak VRAM    {peak_gb:.3f} GB   (during eval)")
    print(f"  WikiText-2 PPL  {ppl:.4f}")
    print(f"  eval {eval_s:.1f}s")
    print("=" * 56)


if __name__ == "__main__":
    main()
