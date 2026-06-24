# AWQ "mappings" — full explanation (saved verbatim, for follow-up questions)

> Saved as-is so I can re-read and ask more sub-questions later. Companion to the short version in
> [`open_questions.md`](open_questions.md) Q4. Paper: AWQ, Lin et al. 2023 (arXiv 2306.00978).

"mappings" is the one genuinely AWQ-specific concept, and it's worth getting right.

## What a "mapping" is

AWQ's trick is **scaling**: to protect a salient weight channel, it multiplies that channel's weights
by a scale `s` and divides the activations feeding it by the same `s`. The math cancels
(`(W·s)·(x/s) = W·x`) — output unchanged — but the weights now sit on a finer part of the 4-bit grid.

The catch: **where does the "divide the activations by `s`" actually happen?** You can't just scale
activations in mid-air — you push that scale **back into whatever module produced them**. So AWQ needs
to know, for each linear layer it's quantizing, **which preceding module feeds it**. That pairing —
"these linear layers get their input from that module" — **is a mapping.**

## Do all layers have mappings? — No, they share a few repeating patterns

Mappings aren't per-individual-layer. They're a **small set of structural rules that repeat in every
transformer block.** For a Llama/Qwen-style block there are ~4:

| smooth (absorbs the scale) | → balance (the linears being protected) |
|----------------------------|------------------------------------------|
| `input_layernorm`          | `q_proj`, `k_proj`, `v_proj`             |
| `v_proj`                   | `o_proj`                                 |
| `post_attention_layernorm` | `gate_proj`, `up_proj`                   |
| `up_proj`                  | `down_proj`                              |

So a 28-layer model isn't 28×7 independent decisions — it's these **4 patterns applied 28 times.**
Note `q/k/v` are "**smoothed together**": they all read the same input (the layernorm output), so they
share **one** scale vector computed from that shared input.

**"Auto-resolves for Qwen2"** means llm-compressor ships these standard patterns for Llama/Qwen-family
models, so you don't hand-write them. A novel architecture (unusual block structure) would need you to
define the mappings yourself — that's the one case where AWQ needs manual work.

**Analogy:** a mapping is a paired volume knob — turn the input feeding a group of layers down, and you
must turn the knob on whatever feeds it up by the same amount, or the music changes. The mapping table
says which knobs are wired together.

---

## My sub-questions (to ask later)

_(add questions here as they come up; we'll work through them)_

1.
2.
3.
