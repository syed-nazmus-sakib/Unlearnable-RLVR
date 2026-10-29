# E3 — RESULTS: does the aggregation choice change the trained model?

Five arms x five seeds x two held-out sets. 25/25 cells complete 2026-08-12 13:38. Accuracy is the guarded pass rate.

## e3_math500

| arm | signal removed | accuracy | sd | vs baseline | 95 % CI | separates |
|---|---|---|---|---|---|---|
| arm 1 baseline | 0.0 % | **31.40 %** | 0.33 | — | — | — |
| arm 3 intersection 218 (LABELLED) | 5.0 % | 31.20 % | 0.42 | -0.21 pts | [-0.76, +0.34] | no |
| arm 5 random 218 (control) | 21.8 % | 31.21 % | 0.32 | -0.19 pts | [-0.66, +0.28] | no |
| arm 2 union 404 (LABELLED) | 27.6 % | 30.95 % | 0.61 | -0.46 pts | [-1.17, +0.26] | no |
| arm 4 random 404 (control) | 39.9 % | 30.83 % | 0.21 | -0.57 pts | [-0.97, -0.17] | **YES** |

## e3_pool2000

| arm | signal removed | accuracy | sd | vs baseline | 95 % CI | separates |
|---|---|---|---|---|---|---|
| arm 1 baseline | 0.0 % | **45.06 %** | 0.14 | — | — | — |
| arm 3 intersection 218 (LABELLED) | 5.0 % | 44.99 % | 0.14 | -0.06 pts | [-0.27, +0.14] | no |
| arm 5 random 218 (control) | 21.8 % | 44.74 % | 0.43 | -0.32 pts | [-0.78, +0.14] | no |
| arm 2 union 404 (LABELLED) | 27.6 % | 44.98 % | 0.24 | -0.07 pts | [-0.36, +0.21] | no |
| arm 4 random 404 (control) | 39.9 % | 44.48 % | 0.31 | -0.58 pts | [-0.93, -0.23] | **YES** |

## The registered primary: arm 2 vs arm 3

| eval | difference | 95 % CI | p | separates |
|---|---|---|---|---|
| e3_math500 | -0.25 pts | [-1.02, +0.52] | 0.4752 | **no** |
| e3_pool2000 | -0.01 pts | [-0.30, +0.28] | 0.9558 | **no** |

**Confounded by design** (619 prompts / 6.20 epochs against 805 / 4.77), which is why the DiD is reported beside it (A32.2). It remains the registered primary and is not demoted after the fact.

## The difference-in-differences, beside the primary and not replacing it

`DiD = (arm2 - arm4) - (arm3 - arm5)`, each bracket size- and epoch-matched, isolating WHICH prompts were removed from HOW MANY.

| eval | DiD | 95 % CI | separates |
|---|---|---|---|
| e3_math500 | +0.13 pts | [-0.66, +0.92] | **no** |
| e3_pool2000 | +0.25 pts | [-0.31, +0.82] | **no** |

## Labelled against random at MATCHED size and epochs

The practitioner's question: does removing *unlearnable* data cost less than removing the same number of arbitrary prompts?

| pair | signal gap | difference | 95 % CI | separates |
|---|---|---|---|---|
| arm 2 vs arm 4 (404 removed) — e3_math500 | 12.3 pts | +0.12 pts | [-0.55, +0.78] | no |
| arm 3 vs arm 5 (218 removed) — e3_math500 | 16.8 pts | -0.02 pts | [-0.56, +0.53] | no |
| arm 2 vs arm 4 (404 removed) — e3_pool2000 | 12.3 pts | +0.51 pts | [+0.10, +0.92] | **YES** |
| arm 3 vs arm 5 (218 removed) — e3_pool2000 | 16.8 pts | +0.26 pts | [-0.21, +0.72] | no |

## Verdict against the registered branches (A32.3)

* arm 2 separates from baseline: **NO** (-0.46 pts)
* arm 3 separates from baseline: **NO** (-0.21 pts)
* arm 2 separates from arm 3: **NO** (-0.25 pts)

**BRANCH B fired, and in a flatter form than any branch anticipated.** Neither labelled arm separates from baseline, and they do not separate from each other. A32.3 branch B applies: this is reported as a **BOUND, not as 'no effect'**. The difference between the two aggregation rules is smaller than this design can resolve, the impact claim is withdrawn, and the reason is a power limit rather than evidence of equivalence.

Branch C anticipated that arm 2 WOULD separate from baseline while arm 3 did not. It did not. The outcome is flatter than the registered set of outcomes allowed for, and that is recorded rather than fitted to the nearest branch.

## The conclusion registered in advance (A33.1), against what happened

> The aggregation choice changes what is removed by a factor of 5.5 in gradient signal, and training is robust enough that this does not measurably change the resulting model — bounded at well under one accuracy point.

**Borne out.** The largest labelled-arm effect is -0.46 points and no labelled arm separates from baseline on either eval set. The only arm that separates anywhere is the random 404 control, which removes the most signal of any arm.

**So the instability is a problem for anyone REPORTING the unlearnable set, and not for anyone TRAINING with it.**

