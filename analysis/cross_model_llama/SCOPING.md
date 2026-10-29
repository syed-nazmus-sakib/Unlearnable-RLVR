# E4 — second model (Llama-3.2-3B-Instruct): design answer and costing

**No GPU work has been done. Nothing starts until the design and the costing are
agreed.**

Confirmations first, as asked:

* `huggingface-cli whoami` → **chandak-chakma**, token at
  `/home/rmedu/.cache/huggingface/token` (37 bytes, 2026-08-12 16:17)
* `meta-llama/Llama-3.2-3B-Instruct` **config fetched successfully** — gated
  access is working

| config field | value |
|---|---|
| hidden_size | 3072 |
| num_hidden_layers | 28 |
| vocab_size | **128256** |
| tie_word_embeddings | **True** |
| torch_dtype | bfloat16 |

---

## 1. The design question: yes, difficulty tiers are the right call

**Agreed, and it is not a compromise — it is the only available option that does
not change the method being studied.**

### Why the training route is not merely expensive but unavailable

Your instinct that "3B trained five times will not fit on a 24 GB card" is
right, and understated. **Full-parameter GRPO at 3B does not fit even once:**

| component | bf16/fp32 | size |
|---|---|---|
| parameters | bf16 | 6.4 GB |
| gradients | bf16 | 6.4 GB |
| Adam moments (m, v) | fp32 | 25.6 GB |
| **total before activations or KV cache** | | **≈ 38 GB** |

Against 24 GB. So the choice is not five runs versus one — it is zero runs
versus a **different training method** (LoRA), and LoRA-GRPO labels would not be
comparable to Chen's full-parameter definition or to our own 0.5B labels. A
cheaper route to training-defined groups at 3B does not exist on this hardware,
and I could not find one worth proposing.

### Why tiers test the same thing

The section 7 confound is **mechanical in K**: measured similarity behaves as
`c/(1 + λ/K)`, so it rises with the number of correct rollouts averaged,
whatever the prompts are called. Reproducing it needs *variation in K*, not the
unlearnable label.

Our own data says the label is close to a difficulty tier anyway:

* untrained pass rate predicts the unlearnable label at **AUC 0.927**
* trained pass rate at **AUC 0.996**
* a difficulty-matched control was **infeasible** — 134 prompts needed at
  pass@1 ≤ 0.05 outside the label, only 17 exist

"Unlearnable" is very nearly a synonym for "lowest pass rate", which is exactly
why tiers are a fair substitute here — and that equivalence is itself already a
result of ours, so the substitution is justified from measurement rather than
convenience.

### What must be said in the text

Reported as **difficulty tiers by initial-policy pass rate**, never as an
unlearnable set. The generalisation claimed is **the confound, not the
definition**: that similarity measured at unequal correct-rollout counts is
inflated in the same way in a second model of a different family. E4 says
nothing about whether Llama has unlearnable prompts.

---

## 2. THE GATE THAT COMES FIRST — MATH Hard is entirely Level 5

`lighteval/MATH-Hard`, **ungated**, 2304 train / 1324 test, and **every one of
the 3628 problems is Level 5**.

That is a problem for this design and it needs saying before any commitment.
Our 0.5B tiers had natural correct-rollout counts of **11 / 17 / 65** out of
~116 rollouts, i.e. pass rates of roughly 0.09 / 0.15 / 0.56. **A top tier at
p ≈ 0.56 requires prompts Llama-3.2-3B-Instruct solves more than half the time,
and an all-Level-5 set may simply not contain them.** If the achievable pass
rates compress into a narrow low band, K barely varies, and the confound test
loses the leverage it depends on.

**Registered gate, before the expensive stages:**

> Screen 800 MATH-Hard train prompts at N=32. If the measured pass rates cannot
> produce three tiers whose median correct-rollout counts span **at least 4x**,
> the run does not proceed as designed. Cost of learning this: ~4 GPU-hours out
> of ~30.

Options if the gate fails, decided in advance rather than after: (a) widen to
MATH levels 1–4 for the top tier and state the departure from Chen's pairing;
(b) drop to two tiers and report the reduced leverage; (c) abandon E4. **My
recommendation is (a)** — Chen's pairing is worth following where it can be, but
the confound is the object of study, and a spread we cannot produce is worth
more than a pairing we can.

---

## 3. Costing

### 3a. Download and disk

| item | size |
|---|---|
| safetensors + tokenizer + config | **6.5 GB** |
| full repo incl. `original/` .pth duplicate | 12.86 GB |

Fetch **safetensors only** — the `original/` folder is a duplicate in
Meta's checkpoint format and doubles the download for nothing.

### 3b. Generation, from measured 0.5B throughput

Measured on this box during E3 (not assumed): **15.7 resp/s** on MATH500 and
**18.1 resp/s** on the 2000-prompt pool, eval-only sglang at 0.5B.

Llama-3.2-3B is **6.4x the parameters**; decode is memory-bandwidth bound, and
Level 5 responses run longer than levels 1–4. Estimate **~2.0 resp/s**,
deliberately conservative (a 1/8 derating rather than 1/6.4).

| stage | responses | hours @ 2.0/s |
|---|---|---|
| screening: 800 prompts x 32 | 25,600 | **3.6 h** |
| main pass: 150 prompts x 128 | 19,200 | **2.7 h** |

The main pass is generated **once**. Matched-K costs no further generation —
levels are subsampled from the same pool, which is what made the 0.5B sweep
affordable.

### 3c. The gradient pass, and why the projection is now mandatory

The lm_head gradient at 3B is **128256 x 3072 = 394 M floats = 788 MB per prompt
in fp16**, against 272 MB at 0.5B — **2.9x**.

| | 0.5B (actual) | 3B (projected need) |
|---|---|---|
| per-prompt vector | 272 MB | **788 MB** |
| 150 prompts, one pass | 35 GB | **118 GB** |
| free disk | — | **79 GB** |

**Storing full vectors is infeasible.** The CountSketch projection already
validated on real gradients (`E1_rescue/analysis/projection_validation.md`,
seed 20260808, 65536 buckets) reduces this to **0.26 MB per prompt, a 1039x
saving**, with group means within 0.52 % and the easy/D_u ratio 2.666x → 2.654x.

**This makes the projection a dependency, not an optimisation**, and it requires
one code change: project **on write**, inside the gradient script, so the full
788 MB vector is never persisted. Protocol item 7 then applies — the projection
must be re-validated at 3B against exact values on a subset (20 prompts x
788 MB = 15.8 GB, affordable) rather than assumed to carry over from 0.5B.

Gradient-pass time, scaled from the measured 0.5B lever-1 passes (20–24 min per
(K, subsample) over 130 prompts) by ~5x for compute and gradient size:

| stage | passes | hours |
|---|---|---|
| natural-n pass | 1 | ~2 h |
| matched-K, 4 levels x 3 subsamples | 12 | ~20 h |
| split-half repeatability (protocol item 6) | 1 | ~1 h |

**Matched-K levels achievable: K = 4, 6, 8, 11**, the same ladder as 0.5B — set
by the tiers' natural counts, not by compute. Six subsamples instead of three
would add ~20 h; **three is the honest starting point**, with the A22 precedent
that more subsamples can be bought later if the intervals demand it.

### 3d. Totals

| | GPU hours |
|---|---|
| screening (the gate) | 3.6 |
| main generation | 2.7 |
| natural-n gradients | 2.0 |
| matched-K, 3 subsamples | 20.0 |
| split-half | 1.0 |
| projection validation at 3B | 1.5 |
| **total** | **≈ 31 h** |

**Peak disk ≈ 25 GB** (6.5 model + 15.8 transient full vectors during projection
validation + rollouts and projected gradients under 1 GB), against **79 GB
free**. Disk stops being the binding constraint the moment the projection is
used — but only then.

---

## 4. The section 8 protocol, applied from the start

First run where we follow our own protocol from the beginning instead of
arriving at it by correction.

| # | item | how it is met in E4 |
|---|---|---|
| 1 | matched counts | K = 4, 6, 8, 11 with exactly K correct and K incorrect per prompt, fixed cohort across all levels |
| 2 | reported budget and cap | N = 128 rollouts per prompt stated with the tier definition; no cap applied, and its absence stated |
| 3 | reported attrition per tier | the guard/zero-variance attrition (A30) recomputed per tier and reported even if zero |
| 4 | measured not fitted noise | λ measured directly per tier, as in A14, before any curve is fitted |
| 5 | joint bootstrap | over prompts AND subsamples, prompts drawn once per draw and shared across levels (A15 primary) |
| 6 | split-half floor | disjoint halves of one generation pass, plus the unrelated-prompt floor, before any group claim |
| 7 | projection validated | re-validated at 3B against exact values on a 20-prompt subset, not inherited from 0.5B |
| 8 | no extrapolation | nothing reported past K = 11; the ratio is read at the top level, never at K → ∞ |

**Registered before the data**, in the same form as A31: the tier gate above,
the primary quantity (easy/hard ratio at the top matched level), the NO POWER
width threshold, and the branch that fires if the confound does **not** replicate
— which would be a result against our own section 7 and gets equal prominence.

---

## 5. What I would ask you to decide

1. **The gate**: is a ~4 h screening run authorised, with the run stopping there
   if three tiers cannot span 4x?
2. **The fallback**: if the gate fails, (a) widen to levels 1–4 for the top tier,
   or (c) abandon? I recommend (a).
3. **Subsamples**: three to start (~31 h total), or six up front (~51 h)?
4. **The projection code change** — project on write — is a real modification to
   the gradient script and the single largest technical risk in this plan.
