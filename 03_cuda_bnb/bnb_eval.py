"""
Phase 4a — bitsandbytes: load one model three ways and measure the tradeoff.

Three modes, all on the SAME model and SAME perplexity ruler (HF sliding window,
WikiText-2-raw), so the numbers are directly comparable:

  fp16  -- baseline, 16-bit weights, no quantization.
  int8  -- LLM.int8(): 8-bit weights, with a fp16 side-path for outlier columns
           (the trick that makes 8-bit lossless-ish). Zero calibration.
  nf4   -- 4-bit NormalFloat + double quantization (the QLoRA recipe). Weights are
           stored in a 4-bit float-like grid tuned for normally-distributed weights;
           "double quant" also quantizes the per-block scales to save a bit more.
           Zero calibration -- ranges come from the weights themselves, on the fly.

bitsandbytes quantizes WEIGHTS ONLY and dequantizes on the fly inside each matmul;
there is no separate calibration pass (contrast GPTQ/AWQ in 4b/4c).

Run (on the GPU VM):
    python3 bnb_eval.py --mode fp16
    python3 bnb_eval.py --mode int8
    python3 bnb_eval.py --mode nf4
"""
import argparse, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from perplexity import eval_perplexity   # sits next to this file on the VM


def true_param_count(model):
    """Count ORIGINAL weights. bnb packs two 4-bit weights into one uint8, so a
    Params4bit tensor's .numel() reports half — double it to get the real count."""
    n = 0
    for p in model.parameters():
        if p.__class__.__name__ == "Params4bit":
            n += p.numel() * 2
        else:
            n += p.numel()
    return n


def load_model(model_id, mode):
    """Return (model) loaded per `mode`. device_map='cuda' keeps everything on GPU 0."""
    if mode == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda")

    if mode == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda")

    if mode == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",            # NormalFloat-4 (vs "fp4")
            bnb_4bit_use_double_quant=True,       # quantize the scales too
            bnb_4bit_compute_dtype=torch.bfloat16 # matmul math runs in bf16
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="cuda")

    raise ValueError(f"unknown mode {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--mode", required=True, choices=["fp16", "int8", "nf4"])
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap tokens for a fast smoke run, e.g. 20000")
    args = p.parse_args()

    torch.cuda.reset_peak_memory_stats()

    print(f"Loading {args.model}  mode={args.mode} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = load_model(args.model, args.mode)
    load_s = time.time() - t0

    # VRAM held by the resident weights (right after load, before any forward pass).
    weights_gb = torch.cuda.memory_allocated() / 1e9
    n_params = true_param_count(model)   # counts packed 4-bit weights correctly

    t1 = time.time()
    ppl = eval_perplexity(model, tok, device="cuda", max_tokens=args.max_tokens)
    eval_s = time.time() - t1

    # Peak VRAM during the eval forward passes (weights + activations + KV).
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("\n" + "=" * 56)
    print(f"  model        {args.model}")
    print(f"  mode         {args.mode}")
    print(f"  params       {n_params/1e9:.3f} B")
    print(f"  weights VRAM {weights_gb:.3f} GB   ({weights_gb*1e9/n_params:.2f} bytes/param)")
    print(f"  peak VRAM    {peak_gb:.3f} GB   (during eval)")
    print(f"  WikiText-2 PPL  {ppl:.4f}")
    print(f"  load {load_s:.1f}s   eval {eval_s:.1f}s")
    print("=" * 56)


if __name__ == "__main__":
    main()
