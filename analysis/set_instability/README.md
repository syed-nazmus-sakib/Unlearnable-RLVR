# E0 — Seed stability of the unlearnable set

**Question:** how much of "unlearnable" is a stable property of a prompt, and how much
is seed noise that a 3-seed intersection happens not to filter out?

Chen et al. ([arXiv:2605.16787](https://arxiv.org/abs/2605.16787), ICML 2026) define
unlearnable as the intersection over **three** GRPO seeds of prompts still at
pass@1 ≤ τ after training. The intersection is a noise filter, but the paper never
reports how much it filters, and its own Limitations concede that "a fraction of
examples will sit near the boundary." That fraction is unmeasured — and every
downstream claim in the paper (gradient outlierness, low reasoning quality, resistance
to every intervention they tried) is a claim about whichever set the intersection
produced.

E0 measures it. It runs first because:

1. **Everything downstream is defined against D_u.** E1 (quality-conditioned gradient
   analysis) and E2 (conversion rate) both need a D_u you trust. Running them on a
   set that is 40% noise wastes the GPU time and produces an unreproducible number.
2. **It is a result on its own.** If a large share of the published set is seed noise,
   that is a publishable correction, and it lands without needing any new method.
3. **It is nearly free.** The reproduction already has seed 1; this is four more runs
   of an existing pipeline.

## Why this is worth doing — evidence already in hand

Running the analyzer on the existing single-seed partition:

| | |
|---|---|
| unlearnable (seed 1, τ=0.1) | 156 |
| of those, mean pass@1 within ±0.06 of τ | **61 (39%)** |

pass@1 is a count over 32 samples, so τ=0.1 means **flagged iff ≤ 3 correct**
(τ·n = 3.2). Three correct is flagged; four is not. A 39% boundary mass means a large
part of the set is one lucky rollout from changing class — exactly the regime where
adding seeds should shrink the set substantially. Whether it does is the experiment.

## Files

| file | what it does |
|---|---|
| `run_seed.sh <seed>` | train → eval → prune → record. Every stage skip-if-done |
| `run_all.sh [seeds…]` | `run_seed.sh` over a seed list, sequentially (default `2 3 4 5`) |
| `classify.sh [seeds…]` | the repo's `finalize_unlearnable_example_id.py` over N seeds |
| `analyze_stability.py` | decay curve, per-prompt flag counts, boundary mass, report |
| `manifest.tsv` | seed → run dir / results glob / reward-stats CSV. Written by `run_seed.sh` |
| `analysis/` | `stability_report.md`, `per_example.csv`, `stability.png` |
| `partitions/` | one id-partition directory per seed subset |

## Running it

```bash
cd /media/rmedu/3039-8E90/LLM/EXPERIMENTS/E0_seed_stability

nohup bash run_all.sh 2 3 4 5 > logs/run_all.log 2>&1 &   # ~12h, one seed at a time

bash classify.sh                    # all seeds in the manifest
bash classify.sh 1 2 3              # the paper's 3-seed protocol, for comparison
python analyze_stability.py         # -> analysis/
```

Seed 1 is already in `manifest.tsv`, pointing at the existing
`qwen_0.5b_run1` artifacts — it is not re-run.

## Cost

| | per seed | 4 seeds |
|---|---|---|
| train to `global_step_120` | ~2.5 h | ~10 h |
| pass@1 eval (32 × 1023) | ~0.5 h | ~2 h |
| disk, before pruning | 26 GB | 104 GB |
| disk, after pruning | ~1 GB | ~4 GB |

242 GB free at setup, so the unpruned footprint would have fit but left no room for
E1/E2. `run_seed.sh` prunes to `global_step_120/actor/huggingface` plus
`prompt_reward_stats.csv` — the only artifacts classification and the gradient
analysis read — **after** the eval JSON exists, so a failed eval never destroys the
checkpoint it needs. Set `PRUNE=0` to keep the FSDP shards and optimizer state.

## Three things this wrapper does that `repro/train_1gpu.sh` does not

1. **Stops at step 120 automatically.** `train_1gpu.sh` runs 372 generation batches,
   which overshoots to ~210 optimizer steps because the gen-batch:opt-step ratio decays
   from 2.8 to 1.76 as the policy sharpens. Seed 1 was stopped by hand. The watchdog
   polls for `global_step_120/actor/huggingface/model.safetensors` and waits for its
   size to be unchanged across two consecutive 30 s polls before terminating, so it
   never stops mid-write.
2. **Kills the process group.** `train_1gpu.sh` has an EXIT/INT/TERM trap that calls
   `ray stop`, but bash defers trap handlers while a foreground child (`bash train.sh`)
   is running, so signalling that script alone does not stop training promptly.
   `set -m` puts the job in its own process group so the whole tree is signalled at
   once; Ray cleanup is then repeated by the wrapper because the deferred trap may
   never have fired.
3. **Prunes and records**, so `classify.sh` and `analyze_stability.py` need no
   hand-edited paths.

## What the analysis reports

- **Decay curve** — mean |unlearnable| over *all* C(K,j) subsets of size j, so the
  curve is not an artifact of seed ordering. The marginal change from the (K−1)th to
  the Kth seed says whether it has converged.
- **Per-prompt flag count** — in how many of K seeds each prompt is flagged, with a
  Wilson score interval on the flag probability (closed form, no scipy, well-behaved
  at k=0 and k=K). Prompts at K/K are `stable_unlearnable`; those in between are
  `boundary`.
- **Boundary mass** — how many candidates sit within ±0.06 of τ, i.e. are threshold
  artifacts as much as findings.
- **Pairwise Jaccard** between seeds' flagged sets. Low values mean the flagged sets
  are largely seed-specific and the intersection is doing most of the definitional work.

## Decision rule

| outcome | what it means | what to do |
|---|---|---|
| Stable set large, curve flattened | unlearnability is a real per-prompt property | use `class == stable_unlearnable` as D_u for E1/E2; state that it is the K-seed set |
| Stable set small, curve still falling | the published 3-seed number is substantially seed noise | that is the paper — a correction to an ICML result, no new method needed. Re-run any downstream analysis on the stable subset |

Either way, carry `boundary` as a **separate group** rather than pooling it into D_u.
Pooling is what makes the 3-seed number fragile in the first place.

## Gotchas inherited from the reproduction

- **The groups are not disjoint.** `easy` is defined on *initial* pass@1 while
  `unlearnable` is defined on *final* pass@1, so a prompt that starts above τ and ends
  below it lands in both (20 such prompts at seed 1). `analyze_stability.py` reports
  per-prompt classes independently of that partition, but `classify.sh` output inherits
  the overlap.
- **Evaluate `@train`, not `@test`.** `extra_info["index"]` runs 0–8138 on train and
  0–499 on test over *different* problems, so feeding `@test` results to the aggregator
  silently matches unrelated prompts against training reward stats.
  `repro/eval.sh` already defaults to `@train`.
- **`pkill -f 'ray::'`** in the cleanup path matches Ray worker processes by their
  standard `ray::` prefix. If you run other Ray workloads on this box, stop them first.
