# Perplexity — in simple words

## What it is (intuition)
Perplexity measures **how surprised / confused a language model is by real text.**
**Lower = less surprised = better model.**

Think of it as the model's **average "branching factor"**: at each word, how many
options is it effectively torn between?
- Perplexity **3** → at each step the model is about as unsure as if it were picking
  between **3** equally-likely words.
- Perplexity **100** → it's flailing, as unsure as choosing among 100 words.

A model that perfectly predicts the text has perplexity **1** (no surprise at all).
Random guessing over a 50,000-token vocabulary has perplexity ~50,000.

## How you calculate it
You give the model **real text it didn't write**, and at every position you ask:
*"what probability did you assign to the word that actually came next?"*

1. For each true next-token, grab the probability the model gave it: `p₁, p₂, …, pₙ`.
2. Take the **negative log** of each (small probability → big surprise).
3. **Average** those → this is the *cross-entropy* (average surprise per token, in nats).
4. **Exponentiate** it → perplexity.

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\ln p_i\right)$$

### Tiny worked example
Text: **"the cat sat"**. Suppose the model assigned:
- `p(cat | "the")   = 0.2`  → surprise `-ln 0.2 = 1.61`
- `p(sat | "the cat") = 0.5` → surprise `-ln 0.5 = 0.69`

Average surprise = `(1.61 + 0.69) / 2 = 1.15`
**Perplexity = exp(1.15) ≈ 3.16** → "on average, as unsure as choosing among ~3 words."

## Input and output
- **INPUT:** a chunk of **real text** (we use WikiText-2, Wikipedia articles),
  turned into tokens by the model's tokenizer. *No labels needed* — the text itself
  is the answer key (each token's "correct prediction" is just the next token).
- **OUTPUT:** a **single number** — the perplexity. That's it.

## Why we care for quantization
Quantizing rounds the weights → the model's probabilities shift slightly → it gets a
bit more "surprised" → **perplexity goes up a little.** The size of that increase is
our first, cheapest measure of *how much quality we lost.*
- 8-bit: PPL barely moves (<1%) → basically lossless.
- Good 4-bit: PPL up ~1–5% → great trade for 3–4× smaller.
- Broken quant: PPL explodes (or goes to NaN).

## One catch — the "sliding window"
A Wikipedia document is longer than the model's context window, so we can't score it
in one shot. We slide a window across the text and only score the **new** tokens each
step (so every token is predicted with full left-context and nothing is counted twice).
That's the extra machinery inside `06_evaluation/perplexity.py` — the idea is still
just "average surprise on real text, then exp()."
