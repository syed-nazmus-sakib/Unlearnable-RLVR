"""A49 — the 5120 result, judged against the rule registered before it ran.

PRIMARY: Jaccard between the guarded flagged set at 5120 and at 1024, all 1023
prompts, tau = 0.1, against the MEASURED same-budget floor from A36,
0.8770 [0.8397, 0.9127] (two independent 1024 passes of the same checkpoint).

  A49.1 BRANCH A fires if J < 0.8397 -- below what two same-budget passes
        already disagree by. The paper's numbers are then budget-dependent.
  A49.2 BRANCH B fires if J >= 0.8397.

The flagging rule is copied from deep_vs_shallow.py's DEEP rule verbatim:
one pass of 128, flag if the GUARDED pass rate <= tau. Guard is A24:
score == 1 AND extracted_answer != '[invalid]'.

Sold as this and never as more (A49.5): this tests whether OUR central object
is stable to the EVALUATION budget. It does NOT reproduce Chen's
configuration, because our training also ran at 1024 and their policy was
trained to use 5120.

  $HOME/miniconda3/envs/vllm/bin/python budget_jaccard.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
REPO = os.path.join(ROOT, "unlearnability-rlvr")
RES = os.path.join(REPO, "results")
TAU, SEED, NBOOT = 0.1, 20260812, 5000
FLOOR, FLOOR_LO, FLOOR_HI = 0.8770, 0.8397, 0.9127

PASS1024 = ["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
            "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"]
PASS5120 = ["qwen_0.5b_ckpt1_n128_cap5120_simplelr_qwen_level1to4_sub1k@train_0.7_128_5120_None_0_qwen.json"]
CKPT = os.path.join(REPO, "checkpoints")


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def load(files, cap):
    ok, inv, n = defaultdict(int), defaultdict(int), defaultdict(int)
    texts, index = [], []
    for f in files:
        for r in json.load(open(os.path.join(RES, f))):
            q = int(r["question_id"])
            v = r.get("verification", {}) or {}
            ok[q] += guarded(v)
            inv[q] += v.get("extracted_answer") == "[invalid]"
            n[q] += 1
            texts.append(r["response"])
            index.append(q)
    run = sorted(__import__("glob").glob(os.path.join(CKPT, "*_seed1")))[0]
    tok = AutoTokenizer.from_pretrained(
        os.path.join(run, "global_step_120", "actor", "huggingface"),
        trust_remote_code=True)
    tr = defaultdict(int)
    for q, e in zip(index, tok(texts, add_special_tokens=False)["input_ids"]):
        tr[q] += len(e) >= cap
    return ok, inv, n, tr


A = load(PASS1024, 1024)
B = load(PASS5120, 5120)
Q = sorted(set(A[0]) | set(B[0]))
Q = [q for q in Q if A[2].get(q) and B[2].get(q)]
print(f"{len(Q)} prompts present in both passes\n")

rate = {"1024": np.array([A[0][q] / A[2][q] for q in Q]),
        "5120": np.array([B[0][q] / B[2][q] for q in Q])}
flag = {k: v <= TAU for k, v in rate.items()}
inter = flag["1024"] & flag["5120"]
union = flag["1024"] | flag["5120"]
J = inter.sum() / union.sum()

rng = np.random.default_rng(SEED)
BI = rng.integers(0, len(Q), size=(NBOOT, len(Q)))
jb = np.empty(NBOOT)
for i, row in enumerate(BI):
    a, b = flag["1024"][row], flag["5120"][row]
    u = (a | b).sum()
    jb[i] = (a & b).sum() / u if u else np.nan
lo, hi = np.nanpercentile(jb, [2.5, 97.5])

exits = int((flag["1024"] & ~flag["5120"]).sum())
entries = int((~flag["1024"] & flag["5120"]).sum())
print(f"flagged at 1024: {int(flag['1024'].sum())}   at 5120: {int(flag['5120'].sum())}")
print(f"stayed {int(inter.sum())}   exited {exits}   entered {entries}   union {int(union.sum())}")
print(f"\nJACCARD 5120 vs 1024 = {J:.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"A36 same-budget floor = {FLOOR:.4f}  [{FLOOR_LO:.4f}, {FLOOR_HI:.4f}]")
branch = "A" if J < FLOOR_LO else "B"
print(f"\nA49 BRANCH {branch} FIRES "
      f"({'below' if branch == 'A' else 'at or above'} the floor's lower bound "
      f"{FLOOR_LO})")

for k, (d, cap) in {"1024": (A, 1024), "5120": (B, 5120)}.items():
    n = sum(d[2][q] for q in Q)
    print(f"\n{k}: guarded mean pass rate {rate[k].mean():.4f}   "
          f"[invalid] share {sum(d[1][q] for q in Q)/n:.4f}   "
          f"truncated at {cap} {sum(d[3][q] for q in Q)/n:.4f}")

json.dump({"n_prompts": len(Q), "tau": TAU, "seed": SEED, "nboot": NBOOT,
           "jaccard": float(J), "lo": float(lo), "hi": float(hi),
           "floor": FLOOR, "floor_lo": FLOOR_LO, "floor_hi": FLOOR_HI,
           "branch": branch, "flag_1024": int(flag["1024"].sum()),
           "flag_5120": int(flag["5120"].sum()), "stayed": int(inter.sum()),
           "exited": exits, "entered": entries,
           "pass_1024": float(rate["1024"].mean()), "pass_5120": float(rate["5120"].mean()),
           "trunc_1024": sum(A[3][q] for q in Q) / sum(A[2][q] for q in Q),
           "trunc_5120": sum(B[3][q] for q in Q) / sum(B[2][q] for q in Q),
           "inv_1024": sum(A[1][q] for q in Q) / sum(A[2][q] for q in Q),
           "inv_5120": sum(B[1][q] for q in Q) / sum(B[2][q] for q in Q)},
          open(os.path.join(EXP, "analysis", "budget_jaccard.json"), "w"), indent=2)
print("\nwrote analysis/budget_jaccard.json")
