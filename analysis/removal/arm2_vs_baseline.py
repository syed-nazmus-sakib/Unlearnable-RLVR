"""E3 — arm 2 (union rule) against baseline, reported BEFORE arm 3 exists.

Registered under A32.3 as the comparison most likely to carry information, and
run at the moment arm 2's fifth seed landed so that it is read clean, without
knowledge of arm 3.

What this can and cannot say is fixed in advance by A32:
  * arm 2 vs BASELINE is a legitimate test at this power -- arm 2 removes
    27.6 % of gradient signal, close to arm 4's 39.9 % which did separate.
  * arm 2 vs ARM 3 is NOT read here and is not implied by anything below.
    A32.3 outcome C explicitly forbids inferring it: a difference in
    significance is not a significant difference.

  $HOME/miniconda3/envs/vllm/bin/python arm2_vs_baseline.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_ind, t as tdist

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
OUT = os.path.join(EXP, "analysis", "arm2_vs_baseline.md")
SEEDS = [1, 2, 3, 4, 5]
EVALS = ["e3_math500", "e3_pool2000"]

sets = json.load(open(os.path.join(EXP, "analysis", "removal_sets.json")))
SIG = sets["share_pct"]
ARMS = {"arm1_base": ("baseline", 0.0),
        "e3arm5_rand218": ("random 218 (control)", SIG["random218"]),
        "e3arm2_union": ("union 404 (LABELLED)", SIG["union"]),
        "e3arm4_rand404": ("random 404 (control)", SIG["random404"])}


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def acc(arm, seed, ev):
    p = glob.glob(os.path.join(RES, f"e3_{arm}_seed{seed}_{ev}@test*.json"))
    if not p:
        return None
    per = defaultdict(list)
    for r in json.load(open(p[0])):
        per[r["question_id"]].append(guarded(r.get("verification", {}) or {}))
    return float(np.mean([np.mean(v) for v in per.values()]))


A = {ev: {} for ev in EVALS}
for ev in EVALS:
    for a in ARMS:
        v = [acc(a, s, ev) for s in SEEDS]
        v = [x for x in v if x is not None]
        if v:
            A[ev][a] = np.array(v)

L = []
W = L.append
W("# E3 — arm 2 (union rule) against baseline\n")
W("**Read before arm 3 had five seeds.** Arm 2's fifth seed landed at "
  "2026-08-11 21:18; this is the first time any arm-2 number has been "
  "examined.\n")

for ev in EVALS:
    W(f"## {ev}\n")
    W("| arm | signal removed | seeds | accuracy | vs baseline | 95 % CI | p |")
    W("|---|---|---|---|---|---|---|")
    base = A[ev]["arm1_base"]
    for a, (label, sig) in ARMS.items():
        if a not in A[ev]:
            continue
        v = A[ev][a]
        if a == "arm1_base":
            W(f"| {label} | 0.0 % | {len(v)} | **{100 * v.mean():.2f} %** | — | — | — |")
            continue
        d = v.mean() - base.mean()
        se = np.sqrt(v.var(ddof=1) / len(v) + base.var(ddof=1) / len(base))
        df = len(v) + len(base) - 2
        tc = tdist.ppf(0.975, df)
        p = ttest_ind(v, base, equal_var=False).pvalue
        star = " **SEPARATES**" if (d + tc * se < 0 or d - tc * se > 0) else ""
        W(f"| {label} | {sig:.1f} % | {len(v)} | {100 * v.mean():.2f} % | "
          f"**{100 * d:+.2f} pts**{star} | [{100 * (d - tc * se):+.2f}, "
          f"{100 * (d + tc * se):+.2f}] | {p:.4f} |")
    W("")

ev = "e3_math500"
base = A[ev]["arm1_base"]
v2 = A[ev]["e3arm2_union"]
d2 = v2.mean() - base.mean()
se2 = np.sqrt(v2.var(ddof=1) / len(v2) + base.var(ddof=1) / len(base))
tc = tdist.ppf(0.975, 8)
sep2 = (d2 + tc * se2 < 0) or (d2 - tc * se2 > 0)

W("## Does arm 2 separate from baseline?\n")
W(f"**{'YES' if sep2 else 'NO'}** on the primary eval: {100 * d2:+.2f} points, "
  f"CI [{100 * (d2 - tc * se2):+.2f}, {100 * (d2 + tc * se2):+.2f}].\n")

W("### Against the prediction registered in A32.1\n")
slope = json.load(open(os.path.join(EXP, "analysis", "signal_mde.json")))["slope_pts_per_signal_pt"]
pred = SIG["union"] * slope
W("The crude random-removal calibration predicted what arm 2 would cost, and "
  "that prediction was recorded before arm 2 was read:\n")
W("| | value |")
W("|---|---|")
W(f"| predicted from the random calibration ({SIG['union']:.1f} % signal) | "
  f"{pred:+.2f} pts |")
W(f"| observed | **{100 * d2:+.2f} pts** |")
W(f"| difference (observed - predicted) | {100 * d2 - pred:+.2f} pts |")
W("")
if abs(100 * d2 - pred) < 100 * tc * se2:
    W("**The observed cost is within sampling error of what removing the same "
      "amount of RANDOM signal would cost.** On this evidence the labelled set "
      "behaves like random data of equal signal content — which is the "
      "null-shaped result A33.1 predicted, and it is reported as a prediction "
      "borne out rather than as a discovery.\n")
else:
    W("**The observed cost DEPARTS from the random calibration.** Per A32.3 "
      "this is the SURPRISE branch: labelled data would be behaving "
      "differently from random data of equal signal content. That is a "
      "different and more interesting claim than the one we set out to make, "
      "and it must not be reported as though we had predicted it.\n")

W("## What is NOT claimed here (A32.3, registered in advance)\n")
W("* **Nothing about arm 2 versus arm 3.** Arm 3 is not read in this document. "
  "Whatever it shows, 'arm 2 separates from baseline and arm 3 does not' will "
  "NOT be reported as 'arm 2 differs from arm 3'. **A difference in "
  "significance is not a significant difference.**")
W("* **Nothing about the aggregation rule mattering.** That claim requires "
  "outcome A, a direct arm-2-vs-arm-3 separation, which A32.1 predicts is "
  "below the detection floor.")

json.dump({"arm2_diff_pts": 100 * d2,
           "arm2_ci": [100 * (d2 - tc * se2), 100 * (d2 + tc * se2)],
           "separates": bool(sep2), "predicted_pts": pred,
           "by_eval": {ev: {a: float(A[ev][a].mean()) for a in A[ev]} for ev in EVALS}},
          open(os.path.join(EXP, "analysis", "arm2_vs_baseline.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
