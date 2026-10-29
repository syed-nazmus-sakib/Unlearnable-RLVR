# B2_REPORT — refined oracle + real-generation certified routing

Protocol frozen in REGISTERED.md amendment A82 before any generation.
All numbers computed by `emit_b2_report.py` from
`analysis/refinement_stats.json`, `analysis/b1_refined_table.json`,
`analysis/b2_results.json`. Original B1 gates stand recorded as
FAIL/FAIL/PARTIAL; nothing here re-labels them.

## A. Oracle refinement (133 topped prompts, +256 real draws each)

* Reference actions changed after top-up: **13 of 133**.
* Still ambiguous at refined n (<0.99 posterior mass on own action):
  **92**.
* Unambiguous population: **931 of 1023**.
* Refined EB prior: α₀=0.297, β₀=0.333.
* Changed qids are listed in `analysis/refinement_stats.json`; the original
  N=256 analysis is preserved untouched alongside this one.

## B. B1 headline metrics regenerated against the REFINED oracle

Same schedules and seed as the original run — only truth labels moved;
δ values were not retuned.

| arm | meanN | err all | err unamb | J rerun | agree | false | missed | PR-regret |
|---|---|---|---|---|---|---|---|---|
| plugin N=8 | 8.0 | 14.3% | 11.4% | 0.808 | 0.507 | 10.3% | 4.0% | 0.0096 |
| plugin N=16 | 16.0 | 9.8% | 6.6% | 0.863 | 0.588 | 6.5% | 3.3% | 0.0072 |
| certified_seq delta=0.9 | 18.1 | 5.3% | 2.2% | 0.924 | 0.705 | 2.1% | 3.3% | 0.0051 |
| certified_seq delta=0.95 | 21.9 | 5.0% | 1.9% | 0.930 | 0.726 | 1.9% | 3.1% | 0.0047 |
| certified_seq delta=0.99 | 31.2 | 4.5% | 1.5% | 0.936 | 0.753 | 2.0% | 2.5% | 0.0041 |
| plugin N=32 | 32.0 | 6.5% | 3.4% | 0.907 | 0.682 | 3.9% | 2.6% | 0.0053 |

## C. Real-generation certification (fresh sglang draws, two independent executions)

Band [0.25, 0.75]; T=0.7, top_p=0.95, max_new_tokens=1024; rounds of 8,
start 8, cap 64; every drawn rollout charged. Prior fitted on the refined
oracle (α₀=0.297, β₀=0.333);
unambiguous n=931; ṽ=0.0963.

### Primary metrics

| arm | meanN | err all | err unamb | agree | J | false | missed | PR-regret |
|---|---|---|---|---|---|---|---|---|
| fixed_plugin N=16 | 16.0 | 10.5% | 6.9% | 0.860 | 0.590 | 6.8% | 2.9% | 0.0074 |
| certified_seq delta=0.9 | 20.3 | 4.9% | 2.1% | 0.931 | 0.732 | 2.0% | 2.7% | 0.0047 |
| certified_seq delta=0.95 | 23.4 | 5.1% | 2.5% | 0.931 | 0.729 | 2.0% | 3.0% | 0.0049 |
| fixed_plugin N=32 | 32.0 | 6.7% | 3.4% | 0.919 | 0.716 | 3.8% | 2.4% | 0.0056 |
| certified_seq delta=0.99 | 35.5 | 4.6% | 2.0% | 0.940 | 0.766 | 2.0% | 2.6% | 0.0043 |

### Prospective success criterion (registered in A82)

Required of some certified δ: error < fixed N=32, cost ≤ 32, agreement >
fixed N=32, unambiguous improvement holds.

| arm | err<fix32 | cost≤32 | agree>fix32 | unamb holds | verdict |
|---|---|---|---|---|---|
| delta=0.9 | yes | yes | yes | yes | **PASS** |
| delta=0.95 | yes | yes | yes | yes | **PASS** |
| delta=0.99 | yes | no | yes | yes | **fail** |

**Criterion verdict: PASS**
Passing deltas: delta=0.9, delta=0.95

### Secondary

Replay-set Jaccard across executions is reported in the primary table's J
column. Cumulative stopping distribution:

* δ=0.9: cumulative stopped by N — 8:52% 16:67% 24:80% 32:85% 40:86% 48:89% 56:91% 64:100%
* δ=0.95: cumulative stopped by N — 8:52% 16:61% 24:71% 32:77% 40:80% 48:83% 56:85% 64:100%
* δ=0.99: cumulative stopped by N — 8:0% 16:39% 24:49% 32:58% 40:67% 48:71% 56:75% 64:100%

Error by distance to nearest band edge (fixed_plugin|N=16, fixed_plugin|N=32, certified_seq|delta=0.9, certified_seq|delta=0.95, certified_seq|delta=0.99):

| boundary distance | fixed_plugin|N=16 | fixed_plugin|N=32 | certified_seq|delta=0.9 | certified_seq|delta=0.95 | certified_seq|delta=0.99 |
|---|---|---|---|---|---|
| d<0.05 | 45.5% (n=99) | 38.4% (n=99) | 30.3% (n=99) | 30.3% (n=99) | 29.3% (n=99) |
| 0.05<=d<0.10 | 25.4% (n=138) | 15.2% (n=138) | 8.7% (n=138) | 9.4% (n=138) | 8.0% (n=138) |
| 0.10<=d<0.20 | 8.7% (n=309) | 3.2% (n=309) | 2.3% (n=309) | 2.6% (n=309) | 2.3% (n=309) |
| d>=0.20 | 0.0% (n=477) | 0.0% (n=477) | 0.2% (n=477) | 0.2% (n=477) | 0.0% (n=477) |


## D. Reading notes

* The certified arm's samples-per-prompt distribution is expected to be
  bimodal: safe prompts stop at 8–16, band-edge prompts run deep.
* Within an execution all arms consume common draws (paired comparison);
  independence lives ACROSS executions, which is what the agreement column
  measures.
* This remains one model/population/checkpoint; E3's null still forbids any
  claim that routing differences propagate to trained outcomes.
