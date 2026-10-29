"""Emit B2_REPORT.md from refinement_stats.json, b1_refined_table.json,
b2_results.json. Every number computed here (A29)."""
import json

import numpy as np

import route_core as rc

ST = json.load(open("analysis/refinement_stats.json"))
RT = json.load(open("analysis/b1_refined_table.json"))
B2 = json.load(open("analysis/b2_results.json"))


def tbl(rows):
    hdr = ("| arm | meanN | err all | err unamb | agree | J | false "
           "| missed | PR-regret |\n|---|---|---|---|---|---|---|---|---|")
    body = "\n".join(
        f"| {r['arm']} {r['config']} | {r['mean_N']:.1f} "
        f"| {r['err_all']*100:.1f}% | {r['err_unamb']*100:.1f}% "
        f"| {r['action_agreement'] if 'action_agreement' in r else r['agreement_rerun']:.3f} "
        f"| {r['jaccard_rerun']:.3f} | {r['false_route']*100:.1f}% "
        f"| {r['missed_route']*100:.1f}% | {r['regret_mean']:.4f} |"
        for r in sorted(rows, key=lambda x: x["mean_N"]))
    return hdr + "\n" + body


refined_rows = "\n".join(
    f"| {r['arm']} {r['config']} | {r['mean_N']:.1f} "
    f"| {r['err_all']*100:.1f}% | {r['err_unamb']*100:.1f}% "
    f"| {r['agreement_rerun']:.3f} | {r['jaccard_rerun']:.3f} "
    f"| {r['false_route']*100:.1f}% | {r['missed_route']*100:.1f}% "
    f"| {r['regret_mean']:.4f} |"
    for r in sorted(RT["rows"], key=lambda x: x["mean_N"]))

fixed32 = next(r for r in B2["rows"] if r["arm"] == "fixed_plugin"
               and r["config"] == "N=32")
crit = B2["prospective_criterion"]
crit_rows = "\n".join(
    f"| {c['delta']} | {'yes' if c['err_lower_than_fixed32'] else 'no'} "
    f"| {'yes' if c['cost_le_32'] else 'no'} "
    f"| {'yes' if c['agreement_higher_than_fixed32'] else 'no'} "
    f"| {'yes' if c['unamb_improvement_holds'] else 'no'} "
    f"| **{'PASS' if c['all_pass'] else 'fail'}** |"
    for c in crit["per_delta"])

stop_lines = []
for delta, fr in B2["stopping_fractions_certified"].items():
    parts = " ".join(f"{lvl}:{fr[lvl]*100:.0f}%" for lvl in sorted(
        fr.keys(), key=lambda s: int(s)))
    stop_lines.append(f"* δ={delta.split('=')[1]}: cumulative stopped by N — {parts}")
STOP_BLOCK = "\n".join(stop_lines)

bound = B2["boundary_bins"]
keys = list(bound.keys())
bin_labels = list(bound[keys[0]].keys())
bound_rows = ""
for label in bin_labels:
    cells = " | ".join(
        f"{bound[k][label]['err']*100:.1f}% (n={bound[k][label]['n_prompts']})"
        for k in keys)
    bound_rows += f"| {label} | {cells} |\n"
bound_hdr = "| boundary distance | " + " | ".join(keys) + " |\n"
bound_hdr += "|" + "---|" * (len(keys) + 1)

meta = B2["meta"]
report = f"""# B2_REPORT — refined oracle + real-generation certified routing

Protocol frozen in REGISTERED.md amendment A82 before any generation.
All numbers computed by `emit_b2_report.py` from
`analysis/refinement_stats.json`, `analysis/b1_refined_table.json`,
`analysis/b2_results.json`. Original B1 gates stand recorded as
FAIL/FAIL/PARTIAL; nothing here re-labels them.

## A. Oracle refinement (133 topped prompts, +256 real draws each)

* Reference actions changed after top-up: **{ST['changed_action']} of {ST['topped']}**.
* Still ambiguous at refined n (<{rc.ORACLE_UNCERTAIN_MAXP:.2f} posterior mass on own action):
  **{ST['still_ambiguous_at_refined_n']}**.
* Unambiguous population: **{ST['unambiguous_total']} of 1023**.
* Refined EB prior: α₀={ST['prior_refined']['alpha0']:.3f}, β₀={ST['prior_refined']['beta0']:.3f}.
* Changed qids are listed in `analysis/refinement_stats.json`; the original
  N=256 analysis is preserved untouched alongside this one.

## B. B1 headline metrics regenerated against the REFINED oracle

Same schedules and seed as the original run — only truth labels moved;
δ values were not retuned.

| arm | meanN | err all | err unamb | J rerun | agree | false | missed | PR-regret |
|---|---|---|---|---|---|---|---|---|
{refined_rows}

## C. Real-generation certification (fresh sglang draws, two independent executions)

Band [0.25, 0.75]; T=0.7, top_p=0.95, max_new_tokens=1024; rounds of 8,
start 8, cap 64; every drawn rollout charged. Prior fitted on the refined
oracle (α₀={meta['prior_refined']['alpha0']:.3f}, β₀={meta['prior_refined']['beta0']:.3f});
unambiguous n={meta['n_unambiguous']}; ṽ={meta['vbar_refined']:.4f}.

### Primary metrics

{tbl(B2['rows'])}

### Prospective success criterion (registered in A82)

Required of some certified δ: error < fixed N=32, cost ≤ 32, agreement >
fixed N=32, unambiguous improvement holds.

| arm | err<fix32 | cost≤32 | agree>fix32 | unamb holds | verdict |
|---|---|---|---|---|---|
{crit_rows}

**Criterion verdict: {crit['verdict']}**
{('Passing deltas: ' + ', '.join(crit['passing_deltas'])) if crit['passing_deltas'] else ''}

### Secondary

Replay-set Jaccard across executions is reported in the primary table's J
column. Cumulative stopping distribution:

{STOP_BLOCK}

Error by distance to nearest band edge ({', '.join(keys)}):

{bound_hdr}
{bound_rows}

## D. Reading notes

* The certified arm's samples-per-prompt distribution is expected to be
  bimodal: safe prompts stop at 8–16, band-edge prompts run deep.
* Within an execution all arms consume common draws (paired comparison);
  independence lives ACROSS executions, which is what the agreement column
  measures.
* This remains one model/population/checkpoint; E3's null still forbids any
  claim that routing differences propagate to trained outcomes.
"""

open("B2_REPORT.md", "w").write(report)
print("wrote B2_REPORT.md,", len(report), "chars")
