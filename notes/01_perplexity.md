# Perplexity — in simple words

## What it is (intuition)
Perplexity measures **how surprised / confused a language model is by real text.**
**Lower = less surprised = better model.**

The clean mental image: perplexity is the model's **average "branching factor"** — at each
word, how many options is it effectively torn between?
- Perplexity **3** → as unsure as picking between **3** equally-likely words.
- Perplexity **100** → flailing, as unsure as choosing among 100 words.
- Perplexity **1** → perfect prediction, no surprise at all.

A good model concentrates its probability **on the actual next word**; a bad model spreads
it thin. **More effective options = more confused = worse.**

---

## Step 1 — A language model outputs probabilities
At each position the model outputs a probability for **every possible next token** (its whole
vocabulary, ~150k tokens for Qwen). They sum to 1 — it's a fixed **budget of 1.0**.

## Step 2 — Look only at the probability of the *actual* next word
We use **real text**, so we know the true next word. We grab the probability the model gave
**that** word. High → predicted well. Low → caught off guard. Every bit of the budget the
model wastes on wrong words is stolen from the right one.

## Step 3 — Turn each probability into "surprise" via negative log
`surprise = −ln(p)`. Small probability → big surprise.
- `p = 0.9` → `−ln 0.9 = 0.11` (barely surprised)
- `p = 0.01` → `−ln 0.01 = 4.6` (very surprised)

## Step 4 — Average the surprise over all tokens
This average is the **cross-entropy** ("average surprise per token").

## Step 5 — Exponentiate
$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\ln p_i\right)$$

---

## Why "1/p = number of choices"
For *equally-likely* options, `1 ÷ probability` recovers the count:
- fair **coin** → 2 outcomes, each `1/2`, and `1 ÷ (1/2) = 2`.
- fair **die** → 6 outcomes, each `1/6`, and `1 ÷ (1/6) = 6`.

The model isn't equally likely, but we still ask: *"it gave the true word probability `p` — a
fair die with how many sides would feel this uncertain?"* Answer: **`1/p`**.

| model gave the true word | effective choices (`1/p`) | meaning |
|---|---|---|
| `p = 0.9` | ~1.1 | confident & correct — **great** |
| `p = 0.25` | 4 | torn between ~4 words — meh |
| `p = 0.02` | 50 | flailing among ~50 words — **bad** |

---

## Why we combine steps with a GEOMETRIC mean (the tricky part)

### Question 1: why multiply the per-step choices?
The model predicts a **sequence**, and to count how many whole sequences it's spread across,
you **multiply** (basic counting / rule of product). Example:
- step 1 torn between 2 words: `{cat, dog}`
- step 2 torn between 8 words: `{1..8}`

```
cat-1 cat-2 cat-3 cat-4 cat-5 cat-6 cat-7 cat-8     ← 8
dog-1 dog-2 dog-3 dog-4 dog-5 dog-6 dog-7 dog-8     ← 8
                                          = 16 total
```
**2 × 8 = 16** (not 2 + 8 = 10). Each first word opens a *fresh* set of 8 — possibilities branch.

### Question 2: why take the root, not the average?
We want one number `k` such that *if every step were equally uncertain (`k` choices each), we'd
get the same total of 16*. With 2 steps: `k × k = 16` → `k = √16 = 4`.

The plain average (2+8)/2 = 5 **fails the test**: `5 × 5 = 25 ≠ 16`. Only `4 × 4 = 16`. So the
honest "typical per-step branching" is the **geometric mean** `√(2×8) = 4`.

**Investment analogy:** money grows ×2 then ×8 → total ×16 over two years. Typical yearly
multiplier is ×4 (`4×4=16`), **not** ×5. Growth compounds (multiplies), so the right "typical"
is the geometric mean. Uncertainty across words compounds the same way.

### Why ln and exp appear — just a calculator trick
`ln` turns multiplication into addition, so a geometric mean is easy to compute:
$$\sqrt{2 \times 8} = (2\times8)^{1/2} = \exp\!\Big(\tfrac{\ln 2 + \ln 8}{2}\Big) = 4$$
So `exp(average of logs)` **is literally** "multiply them all and take the root." That's the
*only* reason `ln`/`exp` are in the formula — nothing deeper.

---

## A tiny end-to-end example
Text: **"the cat sat"**. Model assigned:
- `p(cat | the)   = 0.2` → surprise `−ln 0.2 = 1.61`
- `p(sat | the cat) = 0.5` → surprise `−ln 0.5 = 0.69`

avg surprise = `(1.61 + 0.69)/2 = 1.15` → **PPL = exp(1.15) ≈ 3.16** ("as unsure as ~3 words").

---

## What to INFER from a perplexity number

### 1. The literal read
PPL ~12 → for each next word, the model behaves as if choosing among ~12 equally-likely words.

### 2. Put it on a scale (this is the real inference)
```
   1  ──────────────── 12 ──────────────────────────── ~150,000
perfect              your                          random guessing
(impossible          model                         (vocab ≈ 150k tokens)
 on real text)
```
- **150,000** = learned nothing; every vocab token equally likely.
- **~12** = the model narrowed ~150,000 possibilities down to ~12. It eliminated **99.99%** of
  the vocabulary and hesitates only among a dozen sensible candidates. *That's* "it understands
  English" as a number. (Bigger models reach ~7–9 = even tighter shortlist.)

### 3. For quantization, the **change** is the point, not the number
We measure the fp16 baseline only to have a "before." After quantizing, read the **delta**:

| after quantizing, PPL becomes… | infer |
|---|---|
| ≈ same (e.g. 12.7) | **quality held** — ship it |
| +~5% (e.g. 13.3) | minor loss — usually fine for 3–4× smaller |
| 20+ | real damage — model got noticeably dumber |
| 100s / NaN | **broken** — a mistake in the quantization |

### The one warning — never over-read the absolute number
Only compare perplexities measured the **same way** (same text, harness, window). "12 is good"
in the abstract is meaningless, and you can't compare your number to a blog's number (different
dataset/tokenizer = different scale). We freeze our **own** fp16 baseline and judge every quant
against **that**. Same text, two models, lower wins.

---

## Input and output
- **INPUT:** a chunk of **real text** (WikiText-2, raw Wikipedia), tokenized. *No labels needed* —
  each token's "correct answer" is just the next token.
- **OUTPUT:** a **single number** — the perplexity.

## Why it matters for quantization
Quantizing rounds the weights → probabilities shift → the model gets a bit more surprised →
perplexity rises a little. That rise is our first, cheapest measure of quality lost.
8-bit ≈ <1% (lossless); good 4-bit ≈ +1–5%; a broken quant explodes or goes NaN.

## One catch — the "sliding window"
A Wikipedia document is longer than the model's context window, so we slide a window across the
text and only score the **new** tokens each step (every token predicted with full left-context,
nothing double-counted). That's the extra machinery in `06_evaluation/perplexity.py` — the idea
is still just "average surprise on real text, then exp()."
