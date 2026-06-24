# Setup — CUDA cloud GPU (Phase 4+)

Companion to [`SETUP.md`](SETUP.md). Phases 1–3 ran locally on the M4; Phase 4 (bitsandbytes, GPTQ,
AWQ) needs CUDA kernels, so we run on a cloud GPU. This documents the exact VM, environment, and the
**dependency gotchas** that cost real time — so the next setup is 10 minutes, not an afternoon.

## The GPU VM (GCP)

Project `aibrix-demo-2026`, account james92kj@gmail.com. L4 quota confirmed
(`NVIDIA_L4_GPUS = 1` in us-central1). Provision:

```bash
gcloud compute instances create quant-l4 \
  --project=aibrix-demo-2026 --zone=us-central1-a \
  --machine-type=g2-standard-8 \                       # 8 vCPU, 32 GB RAM, 1x L4
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \   # CUDA 12.9 + PyTorch 2.9 + driver 580, preinstalled
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \                     # GPUs can't live-migrate
  --metadata=install-nvidia-driver=True --scopes=cloud-platform
```

- **L4 24 GB** fits a 7B in fp16 for calibration; plenty for our 1.5B work-horse. ~$0.85/hr on-demand.
- The **Deep Learning VM image** ships torch 2.9.1+cu129 working out of the box — verify with
  `python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'`.

### Environment quirks (important)
- **No conda.** Python is system `/usr/bin/python3` (3.10); torch lives in `/usr/local/lib/.../dist-packages`.
  `pip install` lands user packages in `~/.local` (takes precedence). Just use `python3` directly.
- **Runtime CUDA only — no `nvcc`.** The image has the CUDA *runtime* + driver but NOT the compiler
  toolkit. So anything that builds CUDA extensions from source (GPTQModel, AutoGPTQ) will fail. Prefer
  libraries with prebuilt wheels or pure-PyTorch paths (bitsandbytes ships wheels; llm-compressor is
  pure Python).

## Python packages

```bash
# Phase 4a (bitsandbytes)
python3 -m pip install -U bitsandbytes 'transformers==4.49.0' accelerate datasets

# Phase 4b (GPTQ via llm-compressor)
python3 -m pip install ninja llmcompressor 'transformers==4.49.0'
```

### Why `transformers==4.49.0` is pinned (the torchaudio trap)
Bleeding-edge transformers eagerly `import torchaudio` (for an audio loss class). The DLVM's
preinstalled torchaudio is **ABI-mismatched** against torch 2.9.1 → `OSError: undefined symbol:
torch_library_impl` on *any* model load. We don't use audio, so pin transformers to **4.49.0**, which
predates that import. (Uninstalling torchaudio does NOT fix it — the import is unguarded, so it would
just switch to a ModuleNotFoundError.)

### Why GPTQModel didn't work (→ llm-compressor)
`pip install gptqmodel` is **doubly blocked** here: (1) its `setup.py` does `import pcre` at build
time, and the `pypcre` package on PyPI ships corrupt metadata (`name "unknown"`) so pip can't install
it — even with `--no-build-isolation`; (2) no prebuilt wheels (sdist-only) + no `nvcc` → can't build
from source. We pivoted to **llm-compressor** (Neural Magic / vLLM ecosystem), which runs the *same*
GPTQ algorithm in pure PyTorch. Same lesson, more production-relevant, installs clean.

## Workflow (how we actually run things)

Scripts live in the repo (`03_cuda_bnb/`, `04_cuda_gptq_awq/`); the perplexity harness is reused from
`06_evaluation/perplexity.py`. Copy them next to each other on the VM, then run over SSH:

```bash
# copy script + the perplexity harness into the VM home dir (so `import perplexity` works)
gcloud compute scp 03_cuda_bnb/bnb_eval.py 06_evaluation/perplexity.py quant-l4:~/ --zone us-central1-a

# short job: run and stream output back
gcloud compute ssh quant-l4 --zone us-central1-a --command="python3 bnb_eval.py --mode nf4"

# long job (GPTQ ~7 min): run in background, read the output file when notified
```

## Cost discipline 💸

The L4 bills ~$0.85/hr **while RUNNING**. Stop it between sessions (stopped = no compute charge, only
~$1/mo for the disk):

```bash
gcloud compute instances stop  quant-l4 --zone us-central1-a   # pause
gcloud compute instances start quant-l4 --zone us-central1-a   # resume (same disk, same env)
gcloud compute instances delete quant-l4 --zone us-central1-a  # done with Phase 4 entirely
```
