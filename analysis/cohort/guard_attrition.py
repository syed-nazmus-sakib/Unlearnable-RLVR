"""A30 — the guard/zero-variance attrition as a FINDING, and the cohort-B
numbers the paper needs.

Part 1. The attrition. Our guard marks a rollout incorrect when
`extracted_answer == "[invalid]"` even at score 1.0; Chen's get_group_rewards
drops any prompt whose scores have zero sd. The two together delete prompts
silently, and only ones whose incorrect pool is dominated by guard-rejected
score-1 rollouts can be hit.

The build sampler is REPLAYED here rather than re-derived: same seed rule
(1000*K + SUB), same iteration order, same two rng.sample calls, so the exact
rollouts each cell drew are reconstructed and the predicted drop is checked
against the drop actually observed in attenuation.json. A prediction that is not
confronted with the observation is just a story.

Part 2. Three numbers for the paper, all on ONE stated cohort (A28.1).

  $HOME/miniconda3/envs/vllm/bin/python guard_attrition.py
"""
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
SRC = os.path.join(REPO, "results", "e1_gradsim_initial_GUARDED.json")
OUT = os.path.join(EXP, "analysis", "guard_attrition.md")
GROUPS = ["du71", "learnable", "easy"]
LABEL = {"du71": "D_u", "learnable": "learnable", "easy": "easy"}
SEED = 20260808
NBOOT = 5000
SUBS = [1, 2, 3, 4, 5, 6]


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


coh = json.load(open(os.path.join(EXP, "adaprefix", "gradsim_cohorts.json")))
group_of = {int(q): g for g, qs in coh.items() for q in qs}
by_q = defaultdict(lambda: {"ok": [], "bad": []})
for r in json.load(open(SRC)):
    v = r.get("verification", {}) or {}
    by_q[int(r["question_id"])]["ok" if guarded(v) else "bad"].append(v)

att = json.load(open(os.path.join(EXP, "analysis", "attenuation.json")))
Ks = sorted({int(k) for g in att for k in att[g]})


def present(g, K, sub):
    for e in att[g][str(K)]:
        if e.get("sub") == sub and "per_prompt" in e:
            return {int(q) for q in e["per_prompt"]}
    return None


def replay(K, SUB):
    """Reproduce build_matchedK.py's draw exactly, and return, per prompt, the
    score sd of the 2K rollouts it selected."""
    rng = random.Random(1000 * K + SUB)
    sd = {}
    for q in sorted(by_q):
        d = by_q[q]
        if len(d["ok"]) < K or len(d["bad"]) < K:
            continue
        sel = rng.sample(d["ok"], K) + rng.sample(d["bad"], K)
        sd[q] = float(np.std([v["score"] for v in sel]))
    return sd


L = []
W = L.append
W("# A30 — guard/zero-variance attrition, and the cohort-B numbers\n")
W("## 1. The attrition, predicted and observed\n")
W("For every (K, subsample) the build draw is replayed and a prompt is "
  "PREDICTED to drop when the 2K selected rollouts have zero score sd. That "
  "prediction is checked against which prompts are actually absent from "
  "`attenuation.json`.\n")

pred_all, obs_all, ever = defaultdict(set), defaultdict(set), defaultdict(set)
mismatch = []
for K in Ks:
    for SUB in SUBS:
        sd = replay(K, SUB)
        for g in GROUPS:
            elig = {q for q in sd if group_of.get(q) == g}
            pres = present(g, K, SUB)
            if pres is None:
                continue
            pred = {q for q in elig if sd[q] == 0.0}
            obs = elig - pres
            pred_all[g] |= pred
            obs_all[g] |= obs
            ever[g] |= obs
            if pred != obs:
                mismatch.append((g, K, SUB, sorted(pred), sorted(obs)))

W("| group | eligible prompts | distinct prompts dropped in >=1 subsample | "
  "predicted by the zero-sd rule |")
W("|---|---|---|---|")
for g in GROUPS:
    elig_any = {q for q in by_q if group_of.get(q) == g
                and len(by_q[q]["ok"]) >= 4 and len(by_q[q]["bad"]) >= 4}
    W(f"| {LABEL[g]} | {len(elig_any)} | **{len(ever[g])}** "
      f"({sorted(ever[g]) if ever[g] else 'none'}) | {len(pred_all[g])} |")
W("")
if mismatch:
    W(f"**{len(mismatch)} cell(s) where prediction and observation DISAGREE** — "
      "listed, not hidden, because the mechanism is only established if it "
      "predicts the right prompts:\n")
    for g, K, SUB, p, o in mismatch[:10]:
        W(f"* {LABEL[g]} K={K} sub={SUB}: predicted {p}, observed {o}")
    W("")
else:
    W("**Prediction and observation agree in every one of the "
      f"{len(Ks) * len(SUBS) * len(GROUPS)} cells.** The mechanism is "
      "established, not merely plausible: it names the exact prompts lost.\n")

W("### Is it structurally impossible for D_u and learnable?\n")
W("Not impossible — *conditional on a draw*, any prompt whose sampled incorrect "
  "rollouts are all guard-rejected score-1 would be dropped. What differs is "
  "whether such a draw can occur at all:\n")
W("| group | prompts with an ENTIRELY guard-rejected incorrect pool | "
  "max share of any incorrect pool | max score-1 rollouts in one pool | "
  "largest single-prompt drop probability at K=4 |")
W("|---|---|---|---|---|")
worst, maxshare, maxns = {}, {}, {}
for g in GROUPS:
    sh, ns_max, pmax = [], 0, 0.0
    for q in by_q:
        if group_of.get(q) != g:
            continue
        nb = len(by_q[q]["bad"])
        if nb == 0:
            continue
        ns = sum(1 for v in by_q[q]["bad"] if int(v.get("score", 0)) == 1)
        sh.append(ns / nb)
        ns_max = max(ns_max, ns)
        p = 1.0
        for i in range(4):
            p *= max(ns - i, 0) / max(nb - i, 1)
        pmax = max(pmax, p)
    worst[g], maxshare[g], maxns[g] = pmax, max(sh), ns_max
    W(f"| {LABEL[g]} | {int((np.array(sh) == 1.0).sum())} | {max(sh):.3f} | "
      f"{ns_max} | {pmax:.2e} |")
W("")
W("**Not structurally impossible — just vanishingly unlikely.** D_u and "
  "learnable prompts do carry guard-rejected score-1 rollouts (up to "
  f"{max(maxns['du71'], maxns['learnable'])} in one pool), so a draw of K "
  "all-score-1 is possible in principle. What differs is the SHARE: no D_u or "
  "learnable prompt exceeds "
  f"{max(maxshare['du71'], maxshare['learnable']):.3f} of its incorrect pool "
  f"against {maxshare['easy']:.3f} for the worst easy one, so the largest "
  f"single-prompt drop probability at K=4 is {worst['du71']:.1e} for D_u and "
  f"{worst['learnable']:.1e} for learnable, against {worst['easy']:.1e} for "
  "easy. The zeros in the "
  "expected-count table are rounding, not proof. **This is not a property of "
  "the groups' gradients**; it is a property of how often the verifier accepts "
  "an unparseable answer, which differs by group.\n")

# ---- selection: are survivors systematically low-f_inv?
W("### Is the survival SELECTIVE on the verifier defect?\n")
f_inv = {q: (sum(1 for v in by_q[q]["bad"] if int(v.get("score", 0)) == 1)
             / max(len(by_q[q]["bad"]), 1)) for q in by_q}
nat_pp = json.load(open(os.path.join(EXP, "analysis",
                                     "figure5_natural_per_prompt.json")))
easy_elig = sorted(q for q in by_q if group_of.get(q) == "easy"
                   and len(by_q[q]["ok"]) >= 4 and len(by_q[q]["bad"]) >= 4)
drop = [q for q in easy_elig if q in ever["easy"]]
keep = [q for q in easy_elig if q not in ever["easy"]]
W(f"Among the {len(easy_elig)} eligible easy prompts, guard-rejected share of "
  f"the incorrect pool (`f_inv`):\n")
W(f"* dropped ({len(drop)}): " + ", ".join(f"{f_inv[q]:.2f}" for q in drop))
W(f"* retained ({len(keep)}): median {np.median([f_inv[q] for q in keep]):.2f}, "
  f"max {max(f_inv[q] for q in keep):.2f}\n")
if drop and keep:
    u, p = mannwhitneyu([f_inv[q] for q in drop], [f_inv[q] for q in keep],
                        alternative="greater")
    W(f"Mann-Whitney, dropped > retained on `f_inv`: p = **{p:.4f}** "
      f"(n = {len(drop)} vs {len(keep)}; with n={len(drop)} dropped this is "
      "descriptive, not a test with meaningful power).\n")

sim = {q: nat_pp["easy"][str(q)] for q in easy_elig if str(q) in nat_pp["easy"]}
qs = sorted(sim)
rho, prho = spearmanr([f_inv[q] for q in qs], [sim[q] for q in qs])
W(f"**The question that decides whether the attrition biases the easy mean:** "
  f"does `f_inv` correlate with the per-prompt similarity it would remove? "
  f"Spearman rho = **{rho:+.3f}**, p = {prho:.3f}, n = {len(qs)}.\n")
if prho < 0.05:
    W("It does. The attrition removes prompts non-randomly with respect to the "
      "quantity being measured, so it is a **selection effect on the easy "
      "mean**, and its direction is the sign of rho.\n")
else:
    W("**It does not, at this n.** The prompts this path can remove are not "
      "detectably different in similarity from those it cannot, so the "
      "attrition is a loss of precision rather than a demonstrated bias. That "
      "is the weaker claim and it is the one the data supports — with the "
      "caveat that the correlation is estimated on "
      f"{len(qs)} prompts and only {len(drop)} were actually lost.\n")

# ================= part 2: the cohort-B numbers ==============================
W("## 2. The three numbers, all on cohort B (28/28/38)\n")


def cells(g, K):
    return [e for e in att[g][str(K)] if "per_prompt" in e and e.get("sub") in SUBS]


cohB = {g: sorted(set.intersection(*[set(e["per_prompt"]) for K in Ks
                                     for e in cells(g, K)]), key=int)
        for g in GROUPS}
cohA_src = {g: sorted(set.intersection(
    *[set(e["per_prompt"]) for K in Ks
      for e in att[g][str(K)]
      if "per_prompt" in e and e.get("sub") in ({1} if K == 6 else {1, 2, 3})]),
    key=int) for g in GROUPS}

rng = np.random.default_rng(SEED)


def nat_ratio(cohort):
    e = np.array([nat_pp["easy"][q] for q in cohort["easy"] if q in nat_pp["easy"]])
    d = np.array([nat_pp["du71"][q] for q in cohort["du71"] if q in nat_pp["du71"]])
    b = []
    for _ in range(NBOOT):
        ee = e[rng.integers(0, len(e), len(e))].mean()
        dd = d[rng.integers(0, len(d), len(d))].mean()
        if dd > 1e-9:
            b.append(ee / dd)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return e.mean() / d.mean(), lo, hi, len(e), len(d)


full = {g: sorted(nat_pp[g]) for g in GROUPS}
W("### (a) the natural-count ratio\n")
W("| cohort | easy n | D_u n | natural easy/D_u | 95 % CI |")
W("|---|---|---|---|---|")
for tag, c in [("full included set (55/35/40)", full),
               ("cohort A (28/28/39)", cohA_src),
               ("**cohort B (28/28/38)**", cohB)]:
    r, lo, hi, ne, nd = nat_ratio(c)
    W(f"| {tag} | {ne} | {nd} | **{r:.3f}x** | [{lo:.2f}x, {hi:.2f}x] |")
    if "cohort B" in tag:
        NAT_B = r
W("")

W("### (b) the share of the gap that matching removes, log scale\n")
MATCH_B = (np.mean([np.mean([e["per_prompt"][q] for q in cohB["easy"]])
                    for e in cells("easy", 11)])
           / np.mean([np.mean([e["per_prompt"][q] for q in cohB["du71"]])
                      for e in cells("du71", 11)]))
W("Both terms on cohort B, so no cohort is mixed (A28.1).\n")
W("| quantity | value |")
W("|---|---|")
W(f"| natural-count ratio, cohort B | {NAT_B:.3f}x |")
W(f"| matched ratio at K=11, cohort B | {MATCH_B:.3f}x |")
share_new = (np.log(NAT_B) - np.log(MATCH_B)) / np.log(NAT_B)
share_old = (np.log(NAT_B) - np.log(1.453)) / np.log(NAT_B)
W(f"| share of the log gap removed by matching | **{100 * share_new:.1f} %** |")
W(f"| the same with the superseded 1.453x | {100 * share_old:.1f} % |")
W("")
W("**The sentence must change.** " +
  (f"At {MATCH_B:.3f}x the share is {100 * share_new:.1f} %, which is **just "
   "UNDER half**, not just over. 'More than half the gap is an artefact of "
   "rollout count' is no longer supported; 'about half' is."
   if share_new < 0.5 else
   f"At {MATCH_B:.3f}x the share is {100 * share_new:.1f} %, still above half.")
  + "\n")

W("### (c) the four levels to print — six subsamples, cohort B (28/28/38)\n")
W("| K | easy/D_u | 95 % CI | excludes 1.0 |")
W("|---|---|---|---|")
rd = json.load(open(os.path.join(EXP, "analysis", "ratio_drift_check.json")))
for K in Ks:
    e = rd["ratio_by_K"][str(K)]
    r, lo, hi = e["point"], e["lo"], e["hi"]
    W(f"| {K} | **{r:.3f}x** | [{lo:.2f}x, {hi:.2f}x] | "
      f"{'YES' if lo > 1.0 else 'no'} |")
W("")
W("Cohort label to attach, verbatim: **fixed cohort of 28 D_u / 28 learnable / "
  "38 easy prompts, intersected over six subsamples at each of K = 4, 6, 8, "
  "11**.\n")

json.dump({"ever_dropped": {g: sorted(ever[g]) for g in GROUPS},
           "mismatched_cells": len(mismatch),
           "rho_finv_similarity": float(rho), "p_rho": float(prho),
           "nat_ratio_cohortB": float(NAT_B),
           "matched_K11_cohortB": float(MATCH_B),
           "share_log_gap_removed": float(share_new)},
          open(os.path.join(EXP, "analysis", "guard_attrition.json"), "w"),
          indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
