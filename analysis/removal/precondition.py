"""A31.2 — the GOVERNING PRECONDITION, run before any labelled-arm data exists.

If arms 4 and 5 (the random, size-matched positive controls) do not separate
from arm 1 (baseline), the verdict is NO POWER and the arm-2-vs-3 comparison is
uninterpretable regardless of what it later shows.

This is run at the moment both controls completed and BEFORE arms 2 and 3 have
any seeds on disk, so it cannot be influenced by knowing the answer.

Accuracy is the guarded pass rate used throughout this project: score == 1 AND
extracted_answer != "[invalid]". The unguarded `is_correct` rate is reported
beside it, because the guard is our deviation and its effect should be visible.

  $HOME/miniconda3/envs/vllm/bin/python precondition.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_ind, t as tdist

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
OUT = os.path.join(EXP, "analysis", "precondition.md")
SEEDS = [1, 2, 3, 4, 5]
ARMS = {"arm1_base": "arm 1 baseline (remove nothing)",
        "e3arm4_rand404": "arm 4 random 404 (positive control)",
        "e3arm5_rand218": "arm 5 random 218 (positive control)"}
EVALS = ["e3_math500", "e3_pool2000"]
MDE = 0.91          # A31.4, registered in advance


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def load(arm, seed, ev):
    p = glob.glob(os.path.join(RES, f"e3_{arm}_seed{seed}_{ev}@test*.json"))
    if not p:
        return None
    rows = json.load(open(p[0]))
    per = defaultdict(list)
    for r in rows:
        per[r["question_id"]].append(guarded(r.get("verification", {}) or {}))
    ung = np.mean([bool((r.get("verification") or {}).get("is_correct"))
                   for r in rows])
    return {"acc": float(np.mean([np.mean(v) for v in per.values()])),
            "unguarded": float(ung), "n_prompts": len(per), "n_rows": len(rows)}


data = {ev: {a: {} for a in ARMS} for ev in EVALS}
for ev in EVALS:
    for a in ARMS:
        for s in SEEDS:
            r = load(a, s, ev)
            if r:
                data[ev][a][s] = r

L = []
W = L.append
W("# A31.2 — GOVERNING PRECONDITION: do the positive controls separate?\n")
W("**Run before arms 2 and 3 had any data on disk.** Their result cannot have "
  "influenced this check, which is the whole point of running the controls "
  "first.\n")

# prompt-set consistency: all arms must be scored on the same prompts
W("## Prompt-set consistency\n")
W("| eval set | prompts | identical across all arm-seeds |")
W("|---|---|---|")
consistent = True
for ev in EVALS:
    ns = {r["n_prompts"] for a in ARMS for r in data[ev][a].values()}
    ok = len(ns) == 1
    consistent &= ok
    W(f"| {ev} | {sorted(ns)} | {'YES' if ok else '**NO**'} |")
W("")
W("The comparison is paired on a common evaluation set, so prompt difficulty is "
  "common and cancels; that is what the registered MDE assumes.\n")

results = {}
for ev in EVALS:
    W(f"## {ev}\n")
    W("| arm | seeds | mean accuracy | sd | vs baseline | 95 % CI on the difference |")
    W("|---|---|---|---|---|---|")
    base = np.array([data[ev]["arm1_base"][s]["acc"] for s in sorted(data[ev]["arm1_base"])])
    for a, label in ARMS.items():
        v = np.array([data[ev][a][s]["acc"] for s in sorted(data[ev][a])])
        if a == "arm1_base":
            W(f"| {label} | {len(v)} | **{100 * v.mean():.2f} %** | "
              f"{100 * v.std(ddof=1):.2f} | — | — |")
            continue
        diff = v.mean() - base.mean()
        se = np.sqrt(v.var(ddof=1) / len(v) + base.var(ddof=1) / len(base))
        df = len(v) + len(base) - 2
        tc = tdist.ppf(0.975, df)
        lo, hi = diff - tc * se, diff + tc * se
        tt = ttest_ind(v, base, equal_var=False)
        results[(ev, a)] = (diff, lo, hi, tt.pvalue)
        W(f"| {label} | {len(v)} | **{100 * v.mean():.2f} %** | "
          f"{100 * v.std(ddof=1):.2f} | **{100 * diff:+.2f} pts** | "
          f"[{100 * lo:+.2f}, {100 * hi:+.2f}] |")
    W("")

W("## Verdict\n")
prim = [(a, results[("e3_math500", a)]) for a in ARMS if a != "arm1_base"]
sep = {a: (hi < 0 or lo > 0) for a, (d, lo, hi, p) in prim}
W("| control | difference (MATH500) | 95 % CI | p | separates from baseline |")
W("|---|---|---|---|---|")
for a, (d, lo, hi, p) in prim:
    W(f"| {ARMS[a]} | {100 * d:+.2f} pts | [{100 * lo:+.2f}, {100 * hi:+.2f}] | "
      f"{p:.4f} | {'**YES**' if sep[a] else '**NO**'} |")
W("")

if all(sep.values()):
    W("**PRECONDITION MET.** Both positive controls separate from baseline, so "
      "the design can detect a difference of the size it was built to detect. "
      "The arm-2-vs-3 comparison is therefore interpretable when it lands, and "
      "branches A, B and C of A31.3 all apply as written.\n")
elif any(sep.values()):
    W("**PRECONDITION PARTIALLY MET.** One control separates and the other does "
      "not. Per A31.2 the weaker verdict governs: the design demonstrably "
      "detects the LARGER manipulation only, so an arm-2-vs-3 null is "
      "interpretable only against effects at least as large as the control that "
      "did separate. This is stated as a bound, not waved through.\n")
else:
    W("**PRECONDITION FAILED — the verdict is NO POWER.** Neither control "
      "separates from baseline despite removing 4-8x more gradient signal than "
      "the labelled sets. The experiment cannot detect what it was built to "
      "detect, and **none of branches A, B or C applies**. The arm-2-vs-3 "
      "result must NOT be read as evidence either way.\n")

W("### Direction check, registered separately\n")
W("The controls remove 39.87 % (arm 4) and 21.82 % (arm 5) of the training "
  "signal, so both are expected to fall BELOW baseline, and arm 4 further than "
  "arm 5. A separation in the wrong direction, or arm 5 falling further than "
  "arm 4, would indicate something other than signal removal is driving the "
  "result.\n")
d4 = results[("e3_math500", "e3arm4_rand404")][0]
d5 = results[("e3_math500", "e3arm5_rand218")][0]
W(f"* arm 4 (39.87 % signal removed): **{100 * d4:+.2f} pts**")
W(f"* arm 5 (21.82 % signal removed): **{100 * d5:+.2f} pts**")
W(f"* both negative: {'YES' if d4 < 0 and d5 < 0 else '**NO**'}; "
  f"arm 4 below arm 5: {'YES' if d4 < d5 else '**NO**'}\n")

json.dump({"mde_registered": MDE,
           "results": {f"{ev}|{a}": {"diff_pts": 100 * v[0], "lo": 100 * v[1],
                                     "hi": 100 * v[2], "p": v[3]}
                       for (ev, a), v in results.items()},
           "separates": {a: bool(s) for a, s in sep.items()},
           "precondition_met": bool(all(sep.values()))},
          open(os.path.join(EXP, "analysis", "precondition.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
