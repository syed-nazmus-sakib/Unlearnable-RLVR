# E5 — scale inside the Qwen family instead of changing vendor

**Scoping only. Nothing has run. No GPU time has been spent on this.**

The Llama arm died on extraction, not on design. So the replacement keeps the
verifier, the dataset, the prompt template and the extractor exactly as they are
and changes **one thing: model scale**. That answers the question a reviewer is
more likely to ask — *is this only true at 0.5B?* — and it does it inside a
family where extraction is measured, not assumed.

**Same 1023 prompts** (`simplelr_qwen_level1to4_sub1k`), same instruction, same
guard, same tau. Only the model changes. That is a cleaner comparison than E4
would ever have been, because E4 changed vendor, dataset and prompt at once.

---

## 1. Does a 3B backward pass fit in 24 GB? Yes, with room to spare.

The 38 GB that ruled out 3B training was **almost entirely optimizer state**.
E5 needs gradients, not training, so that term is zero.

Configs fetched, not assumed:

| | hidden | vocab | layers | tied | lm_head params |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 896 | 151936 | 24 | yes | 136.1 M |
| Qwen2.5-1.5B | 1536 | 151936 | 28 | yes | 233.4 M |
| Qwen2.5-3B | 2048 | 151936 | 36 | yes | 311.2 M |

**Qwen2.5-3B, batch 1, seq 1024, bf16, backward for the lm_head gradient:**

| term | size | note |
|---|---|---|
| parameters | 6.18 GB | 3.09 B x 2 bytes |
| gradient of the tied embedding | 1.24 GB | 311.2 M x 4 (fp32 accumulation) |
| logits + cross-entropy workspace | ~0.9 GB | 1024 x 151936 |
| activations for the backward pass | ~1.5–2.5 GB | 36 layers, no checkpointing |
| CUDA context and fragmentation | ~1.0 GB | |
| **total** | **≈ 11–12 GB** | against **24 GB** |
| **optimizer state (m, v)** | **0 GB** | **the term that killed training** |

Even the lazy implementation — `requires_grad` left on for every parameter, so
full gradients are allocated — adds 6.18 GB for a total near 18 GB and still
fits. Training needed 25.6 GB of Adam moments on top of that, which is what did
not fit.

**Verdict: 3B fits comfortably. 1.5B fits trivially (≈ 8 GB).**

**One caveat that applies at every scale, including the published 0.5B work:**
`tie_word_embeddings = true` throughout the family, so the gradient we call the
"lm_head gradient" is the gradient of a **tied** parameter and carries an input-
embedding contribution as well as the output-projection one. That is already
true of every 0.5B number in the paper, so it is a consistency note rather than
a new problem — but it should be stated once, in the method.

---

## 2. Disk: the projection is mandatory, as it was for Llama

| | per-prompt vector (fp16) | 150 prompts x 2 conditions |
|---|---|---|
| 0.5B (actual, on disk) | 272 MB | 9.0 GB observed |
| 1.5B | 467 MB | **140 GB** |
| 3B | 622 MB | **187 GB** |

**Free space: 73 GB.** Storing full vectors is infeasible at either scale, so the
CountSketch projection (65536 buckets, 1039x saving, already validated at 0.5B)
is a **dependency, not an optimisation** — and protocol item 7 requires it be
**re-validated at the new scale** against exact values on a ~20-prompt subset
rather than inherited. Validation transient: 20 x 622 MB = 12.4 GB at 3B, which
fits.

---

## 3. The cheap gate, in minutes

One screening pass over the same 1023 prompts at N=32 = **32,736 responses**.

Throughput is **measured on this box**, not assumed: 0.5B ran at 15.7–18.1
resp/s during E3, and Llama-3.2-3B (3.2 B, i.e. the same class as Qwen2.5-3B)
ran at 7.9 resp/s and 12.5 resp/s across the two E4 gates.

| model | assumed rate | gate cost |
|---|---|---|
| Qwen2.5-1.5B | 13–16 resp/s | **35–42 min** |
| Qwen2.5-3B | 8–12 resp/s | **45–68 min** |

Plus a one-off download: 3.1 GB (1.5B) or 6.2 GB (3B).

---

## 4. Expected `[invalid]` share — and the gate line must NOT be A39.3's

**Direct evidence we have:** Qwen2.5-0.5B, on this exact dataset, with this
exact extractor: **9.77 %** of all rollouts fail extraction, correlated with
pass rate at **-0.513**.

**Direct evidence we have for 1.5B or 3B: none.** We have never run a Qwen model
at any other scale through this harness. Any number quoted for them before the
gate would be an assumption, and this note does not offer one.

**What can be said from the structure rather than assumed:** extraction failure
at 0.5B is concentrated in the hard tercile (21.65 %) and near-absent in the
easy one (0.21 %). A more capable model moves prompts toward the easy end, so
the rate should fall. That is a directional expectation, not a measurement, and
the gate exists to test it.

**The A39.3 line cannot be reused here, and this matters.** That line was
`share < 5 % and |corr| < 0.15`. **Our own Qwen2.5-0.5B baseline — the model the
entire paper rests on — fails both**, at 9.77 % and -0.513. Applying A39.3 to
E5 would condemn the scale-up for an artefact we already tolerate in the primary
model, which would be incoherent.

> **Proposed E5 gate, relative rather than absolute:** the scale-up's extraction
> failure share must be **no worse than the 0.5B baseline of 9.77 %**, and its
> correlation with pass rate no stronger than **-0.513**. If it is worse than
> the model we already publish, the arm stops; if it is comparable or better,
> it proceeds. To be registered before the gate runs.

---

## 5. Tier construction, and the K test that killed the Llama primary

**Construction: terciles of the same 1023 prompts by the new model's own guarded
pass rate at N=32.** Not extreme selection — that is precisely what produced
zero usable prompts at K=11 for Llama (A39.1).

The matched-K design needs **K correct AND K incorrect** out of 128. The band of
true pass rates that delivers that with probability >= 0.90:

| K | usable band of p | width |
|---|---|---|
| 4 | 0.052 – 0.948 | 0.896 |
| 6 | 0.072 – 0.928 | 0.856 |
| 8 | 0.091 – 0.909 | 0.818 |
| 11 | **0.118 – 0.882** | 0.764 |

At 0.5B's own measured rates (mean p = 0.4694, near the optimum of 0.5) the
whole 1023-prompt set gives:

| K | expected usable prompts |
|---|---|
| 4 | 655 |
| 6 | 599 |
| 8 | 553 |
| **11** | **494** |

And because the band is wide, this is **robust to the model getting better**:

| pass-rate shift | mean p | K=4 | K=6 | K=8 | K=11 |
|---|---|---|---|---|---|
| +0.00 | 0.469 | 655 | 599 | 553 | 494 |
| +0.10 | 0.549 | 735 | 720 | 702 | **654** |
| +0.20 | 0.619 | 671 | 664 | 658 | **647** |
| +0.30 | 0.684 | 623 | 618 | 614 | **608** |

**Against Llama's 0.0 usable prompts at K=11, this is the decisive difference.**
Llama failed because an all-Level-5 pool pushed the hard tier to zero correct.
Here the pool is levels 1–4 and the model sits mid-band, so K=11 stays feasible
across any plausible improvement. Only a model above p ≈ 0.88 would break it,
which no 3B model reaches on this set.

---

## 6. Full arm cost, and what dominates

Gradient-pass time scaled from the measured 0.5B lever-1 passes (20–24 min per
(K, subsample) over 130 prompts): ~2.5x for 1.5B, ~4.5x for 3B.

| stage | 1.5B | 3B |
|---|---|---|
| gate screening | 0.7 h | 0.9 h |
| main generation, 150 x 128 | 0.4 h | 0.5 h |
| natural-n gradient pass | 1.0 h | 1.7 h |
| **matched-K, 4 levels x 6 subsamples** | **22.0 h** | **40.0 h** |
| split-half repeatability | 1.0 h | 1.7 h |
| projection re-validation at scale | 1.2 h | 1.5 h |
| **total, 6 subsamples** | **≈ 26 h** | **≈ 46 h** |
| **total, 3 subsamples** | **≈ 15 h** | **≈ 26 h** |

**The matched-K sweep dominates at roughly 85 % of the total, at every scale and
every subsample count.** Everything else together is under 6 hours.

### A reduced arm that still answers the reviewer's question

The generality question is *does the K confound replicate at another scale*, not
*what is the full attenuation curve at another scale*. Running only the two
endpoint levels, K = 4 and K = 11, at 3 subsamples, is 6 passes instead of 24:

| | 1.5B | 3B |
|---|---|---|
| **reduced arm (K = 4 and 11 only, 3 subsamples)** | **≈ 8 h** | **≈ 13 h** |

It gives the ratio at the top matched level and the slope between two points. It
does not give the fitted attenuation curve, and it cannot support a `lambda`
refit at the new scale. That is the honest trade.

---

## 7. Honest odds this repeats the Llama outcome

**Low: roughly 15 % at 1.5B, 20 % at 3B.** The reasoning, with the parts that
are evidence separated from the parts that are judgement:

**Evidence that it will not repeat:**
* The Llama failure was extraction, and extraction is **measured** to work in
  this family at 0.5B (9.77 %, and 74.9 % of Llama rollouts contained no
  `\boxed` at all against a far lower rate here).
* Same tokenizer, same `\boxed` convention the extractor expects, same dataset,
  same template — four of the five things that differed for Llama are held fixed.
* The K-feasibility test that actually killed the Llama primary is predicted to
  pass with 494–654 usable prompts at K=11 across any plausible pass rate.
* The memory question is settled by arithmetic, not hope: ~12 GB against 24 GB.

**Residual risks, in order of size:**
1. **Projection re-validation fails at the new scale** (~10 %). Protocol item 7
   requires agreement under 1 % on group means; if the sketch degrades at
   311 M-dimensional gradients the arm stops after ~3 h. This is the single
   largest risk and it is a hard gate.
2. **Extraction is worse than 0.5B** (~5 %). Possible if the larger base model
   is chattier, but it is a base model on the same prompt, not an instruct model
   being asked to change habits.
3. **Disk** (~3 %). 73 GB free is adequate only because the projection is used;
   a bug that writes full vectors fills the drive in 12 prompts.

**What would still be unanswered even if it succeeds:** this tests scale within
one family. A reviewer asking about a *different* family gets the E4 answer —
we tried, and the verifier could not measure it — which is now a reported
finding rather than a gap.

---

## 8. The recommendation, given five weeks and a 31-to-10-page cut

**Qwen2.5-1.5B, reduced arm, ≈ 8 GPU-hours.** It is the cheapest thing that
converts "we only tested 0.5B" into a measured second point, it runs overnight
twice, and it leaves the page-cut untouched. The 3B full arm at 46 h is a week
of the machine and buys a fitted curve nobody asked for.

**If the gate result is good and time allows, the natural upgrade is 1.5B at 6
subsamples (26 h), not 3B.** Subsample count buys interval width, which is what
the E1 intervals actually needed; parameter count buys a headline.

**Decision points, in order:** (1) authorise the gate at ~40 min; (2) read the
extraction share and the realised tier spans against the relative line in §4;
(3) decide reduced versus full arm with those numbers in hand. Nothing after
step 1 without an explicit answer.
