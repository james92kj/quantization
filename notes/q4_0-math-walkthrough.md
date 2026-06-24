# Q4_0 Quantization — Line-by-Line Math (with dry run)

The simplest quantizer (Generation 1: Legacy quants). Symmetric, 4-bit, block of 32 weights.
We learn the **clean symmetric `/7` convention** (easier to reason about); production llama.cpp
uses a slightly different `/8` convention (see note at the bottom).

## The whole machine (6 lines)

```
1. amax = max|w|          ← find the biggest magnitude in the block
2. d    = amax / 7        ← the scale (step size)
3. q_i  = round(w_i / d)  ← turn each weight into a small integer
4. store q_i (4 bits) + d (fp16)
5. ŵ_i  = q_i * d         ← dequantize (reconstruct)
6. error = w_i - ŵ_i      ← what we lost
```

## Toy block (real blocks are 32 weights; using 8 for readability)

```
w = [ 0.12, -0.45, 0.30, -0.05, 0.88, -0.22, 0.10, -0.67 ]
```

---

## Line 1 — `amax = max|w|`

`|·|` = absolute value: strip the minus signs, compare only *size*, not direction.

```
magnitudes: 0.12  0.45  0.30  0.05  0.88  0.22  0.10  0.67
largest   = 0.88   →   amax = 0.88
```

**Why:** the single biggest weight defines how far the grid must stretch. A big *negative*
weight stretches it just as much as a big positive one (that's why we drop the sign).

---

## Line 2 — `d = amax / 7`

```
d = 0.88 / 7 = 0.12571...  ≈ 0.1257
```

`d` = the **scale** / **step size** = the real-world distance between two neighboring ticks.

### Why divide by 7?

4 bits = 16 codes. Symmetric grid of integers we're allowed to use:

```
−7 −6 −5 −4 −3 −2 −1  0  +1 +2 +3 +4 +5 +6 +7
```

The furthest tick is **7**. The biggest weight (0.88) must land on that furthest tick:

```
0.88 = 7 × d   →   d = 0.88 / 7
```

So `7` = "the largest integer code available." Dividing by it makes the biggest weight map to
the edge of the grid → we use the **full range**, waste nothing on range.

**Key distinction:** `d` converts *positions/ticks* (−7…+7) into *weights*. So you divide by the
furthest **position** (7), NEVER by the **count** of codes (16). 16 is "how many," not "how far."

---

## Why stop at 7 and not 8? (the wasted-code subtlety)

Count the symmetric grid `−7…+7`:

```
negatives: −7…−1  → 7 values
zero:        0     → 1 value
positives: +1…+7  → 7 values
                   ───────────
             total = 15 values
```

But 4 bits = **16 codes**. So **one code is wasted.**

Why not reach ±8? That grid is `−8…+8` = **17 values** > 16 codes. Doesn't fit.

**Deeper reason:** a perfectly symmetric grid needs an **odd** number of levels (one dead-center
for zero, equal counts each side). 16 is **even** → can't be perfectly symmetric → drop to nearest
odd (15) → one code unused. **The wasted code is the price of clean symmetry.**

### The twist — real llama.cpp Q4_0 doesn't waste it

| Convention | Grid | Levels | Scale | Trade-off |
|---|---|---|---|---|
| Clean symmetric (we learn this) | −7…+7 | 15 | `amax / 7` | centered, 1 code wasted |
| Real llama.cpp Q4_0 | −8…+7 | 16 | `amax / 8` | all codes used, slightly lopsided |

Production uses −8…+7 (one extra slot on the negative side) so all 16 codes are used, accepting a
tiny asymmetry. Its scale is `amax / 8`.

---

## Line 3 — `q_i = round(w_i / d)`

Turn each real weight into its nearest integer tick. Read right-to-left:
- `w_i / d` → "how many steps of size `d` does this weight reach?" (a fraction)
- `round(...)` → snap to the **nearest whole tick**

Whole block with `d = 0.1257`:

| `w_i`  | `w_i / d` | `q_i` |
|--------|-----------|-------|
| 0.12   | 0.955     | 1     |
| −0.45  | −3.579    | −4    |
| 0.30   | 2.386     | 2     |
| −0.05  | −0.398    | 0     |
| 0.88   | 7.000     | 7  ← max, lands exactly |
| −0.22  | −1.750    | −2    |
| 0.10   | 0.796     | 1     |
| −0.67  | −5.329    | −5    |

```
q = [ 1, -4, 2, 0, 7, -2, 1, -5 ]
```

- The max weight (0.88) lands exactly on +7 → reconstructs with **zero error** (we designed `d` that way).
- Tiny weights near zero (−0.05) collapse to the **0** tick.
- `round` snaps to *nearest*: −3.58 → −4 (crossed the −3.5 halfway line).

## Line 4 — storage: `q_i (4 bits) + d (fp16)`

Each `q` needs **4 bits** (16 codes = 2⁴). Plus **one fp16 scale (16 bits) per block**, shared.

```
real block of 32:
  32 integers × 4 bits = 128 bits
  1 scale × 16 bits    =  16 bits
                          ───────
                   total = 144 bits  →  144 / 32 = 4.5 bits/weight
```

**Where 4.5 comes from:** 4 bits (own integer) + 16/32 = 0.5 bits (shared scale slice) = **4.5**.

Bigger block → scale spread thinner → cheaper:

```
block 8:   16/8   = 2.0  → 6.0 bits/weight
block 32:  16/32  = 0.5  → 4.5 bits/weight   ← real Q4_0
block 256: 16/256 ≈ 0.06 → 4.06 bits/weight
```

**The tension (why not make blocks huge?):** bigger blocks = cheaper (scale spread thinner).
So why not go huge? Because **one scale must serve every weight in the block** — and if the
block is huge, a single far-flung **outlier** weight stretches `d` for everyone, making the grid
**coarse for the many small weights**. **32 is the sweet spot** between "cheap scale" and "scale
that actually fits the local weights." (This exact trade-off is what **K-quants' super-blocks**
attack later.)

## Line 5 — `ŵ_i = q_i × d` (dequantize)

`ŵ` ("w-hat") = the **reconstructed** weight (what we read back). We only kept `q` and `d`, so we
rebuild via `ŵ = q × d`. This is Line 3 run backwards (Line 3 divides by d; Line 5 multiplies).

```
w = 0.30  → q = 2  → ŵ = 2 × 0.1257 = 0.2514   (undershoot, rounded toward zero)
w = -0.45 → q = -4 → ŵ = -4 × 0.1257 = -0.5028 (OVERSHOOT, rounded away from zero)
```

Rounding can push **either direction** (toward or away from zero) — whichever tick is nearest.

## Line 6 — `error = w_i − ŵ_i` (what we lost)

| `w_i` | `q_i` | `ŵ_i` | `error` |
|-------|-------|-------|---------|
| 0.12  | 1  | 0.1257  | −0.0057 |
| −0.45 | −4 | −0.5028 | +0.0528 |
| 0.30  | 2  | 0.2514  | +0.0486 |
| −0.05 | 0  | 0.0000  | −0.0500 |
| 0.88  | 7  | 0.8799  | +0.0001 ← max, near-zero error |
| −0.22 | −2 | −0.2514 | +0.0314 |
| 0.10  | 1  | 0.1257  | −0.0257 |
| −0.67 | −5 | −0.6285 | −0.0415 |

Read-offs:
1. **Max weight (0.88) ≈ zero error** — we built `d` so it lands exactly on tick 7.
2. **Worst case `−0.05 → 0`** — a small weight erased to zero (proportionally a 100% error).
3. **Every error ≤ d/2 = 0.0628** — the rounding guarantee (below).

### The rounding guarantee — error ≤ d/2

Dequant values can only land on ticks at multiples of `d`. Any real weight falls in a gap between
two ticks; `round()` picks the nearer. The farthest it can ever be from the nearest tick is the
**midpoint = d/2**, because past the midpoint the *other* tick becomes nearer and pulls it back.

```
worst-case error per weight = d/2 = 0.1257/2 = 0.0628
```

Smaller `d` (finer grid) → smaller d/2 → less error. **That's why we fight to keep `d` small.**

### Why we can't just pick a tiny d — the outlier problem

`d = amax/7` is **forced** by the biggest weight in the block; we don't choose it freely.
One outlier inflates `amax`, inflating `d` for the **whole shared block**:

```
normal:  amax=0.88 → d=0.1257 → worst error 0.0628
+outlier 5.0: amax=5.0 → d=0.7143 → worst error 0.357  (~6× worse for EVERYONE)
```

With the inflated d, healthy weights collapse to the 0 tick (their `w/d` falls below 0.5):

```
0.30 / 0.7143 = 0.420 → round → 0   (erased — 0 is the nearest integer tick, <0.5)
0.12 / 0.7143 = 0.168 → round → 0
-0.22/ 0.7143 = -0.308→ round → 0
```

**Punchline:** `d` is shared and set by the single largest weight, so one outlier widens the ticks
for everyone and crushes the small weights. This is the **central weakness of one-scale-per-block**
— exactly what K-quants (smaller scale regions) and I-quants (reshaped tick placement) fix later.

## Progress — COMPLETE ✅

- [x] Line 1 — amax = max|w|
- [x] Line 2 — d = amax / 7 (why 7, even/odd symmetry, wasted code)
- [x] Line 3 — q_i = round(w_i / d)
- [x] Line 4 — storage & bits/weight (4.5 = 4 + 16/32)
- [x] Line 5 — dequantize (ŵ_i = q_i × d), over/undershoot
- [x] Line 6 — error (w_i − ŵ_i), d/2 guarantee, outlier problem

### Next up
K-quants: super-blocks (256 = 8 sub-blocks of 32) + quantized sub-scales + mixed precision —
i.e. how they shrink the region each scale covers to fix the outlier problem above.
