"""A12.2 matched-K: write a rollouts file with EXACTLY K correct and K incorrect
guarded rollouts per prompt.

Chen's sampler takes min(K, available) (checkpoint_gradient_analysis.py:186), so
feeding it a file that already holds exactly K of each produces matched-n
averages **using his unmodified script** -- no code change, and no per-rollout
gradient storage, which A12.2 rejected at 147 GB.

  python build_matchedK.py <K> <subsample_index>

Prompts with fewer than K correct or K incorrect are dropped and counted; the
retained counts per group are printed and are the numbers A12.2 registered
(K=8 -> 38/30/40, K=4 -> 55/35/40).
"""
import json
import os
import random
import sys
from collections import defaultdict

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
SRC = os.path.join(REPO, "results", "e1_gradsim_initial_GUARDED.json")

K = int(sys.argv[1])
SUB = int(sys.argv[2])
DST = os.path.join(REPO, "results", f"e1_gradsim_K{K}_s{SUB}_GUARDED.json")


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


if os.path.exists(DST):
    print(f"{DST} exists, leaving it")
    sys.exit(0)

coh = json.load(open(os.path.join(EXP, "adaprefix", "gradsim_cohorts.json")))
group_of = {q: g for g, qs in coh.items() for q in qs}

by_q = defaultdict(lambda: {"ok": [], "bad": []})
for r in json.load(open(SRC)):
    q = int(r["question_id"])
    by_q[q]["ok" if guarded(r.get("verification", {}) or {}) else "bad"].append(r)

rng = random.Random(1000 * K + SUB)
out, kept = [], defaultdict(int)
for q in sorted(by_q):
    d = by_q[q]
    if len(d["ok"]) < K or len(d["bad"]) < K:
        continue
    out += rng.sample(d["ok"], K) + rng.sample(d["bad"], K)
    kept[group_of.get(q, "?")] += 1

json.dump(out, open(DST, "w"))
print(f"K={K} sub={SUB}: {sum(kept.values())} prompts retained "
      f"(D_u {kept['du71']}, learnable {kept['learnable']}, easy {kept['easy']}), "
      f"{len(out)} rollouts -> {DST}")
