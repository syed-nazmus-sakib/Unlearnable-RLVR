"""A74 stage 1 — the PREDICTION, from pass 3 and pass 3 ONLY.

This file never opens the pass-1 or pass-2 results. That is structural, not a
promise: the only result file it names is the p3G one. It writes a timestamped
prediction which gsm8k_measure.py then reads and cannot alter.

  $HOME/miniconda3/envs/vllm/bin/python gsm8k_predict.py
"""
import glob
import json
import os
import subprocess
from collections import defaultdict

import numpy as np
from scipy.stats import binom

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results"))
AN = os.path.join(EXP, "analysis")
TAU, SEED, NMC, GRID, EM_ITERS = 0.1, 20260816, 4000, 1001, 3000
N = 64

# NOTE the '@'. evaluate_model.py writes <name>_<data>@<split>_<temp>_... and the
# data tag here is "gsm8k_budget@test", so the separator after "gsm8k_budget" is
# '@', not '_'. Writing '_' here is the A74.12 bug a third time; caught at 16:50
# on 2026-08-16, before this script ever ran, by listing the directory.
hits = sorted(glob.glob(os.path.join(RES, "gsm8k_n64_p3G_gsm8k_budget@*.json")))
assert len(hits) == 1, f"expected exactly one p3G file, found {hits}"
P3 = hits[0]


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


per, inval = defaultdict(list), 0
for r in json.load(open(P3)):
    v = r.get("verification", {}) or {}
    per[int(r["question_id"])].append(guarded(v))
    inval += v.get("extracted_answer") == "[invalid]"
Q = sorted(per)
k = np.array([sum(per[q]) for q in Q])
n = np.array([len(per[q]) for q in Q])
assert set(n) == {N}, f"pass 3 is not all N={N}: {sorted(set(n))[:5]}"

pgrid = np.linspace(0, 1, GRID)
LIK = binom.pmf(k[:, None], n[:, None], pgrid[None, :])


def em(lik, iters=EM_ITERS):
    w = np.full(lik.shape[1], 1.0 / lik.shape[1])
    for _ in range(iters):
        post = lik * w
        s = post.sum(1, keepdims=True)
        s[s == 0] = 1.0
        wn = (post / s).mean(0)
        if np.abs(wn - w).max() < 1e-12:
            return wn
        w = wn
    return w


W = em(LIK)
phat = k / n


def qflag(p, nn):
    return binom.cdf(np.floor(TAU * nn), nn, p)


r = np.random.default_rng(SEED)
p = r.choice(pgrid, size=(NMC, len(Q)), p=W / W.sum())
q = qflag(p, N)
a, b = r.random(q.shape) < q, r.random(q.shape) < q
inter, union = (a & b).sum(1), (a | b).sum(1)
ratio = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
qm = qflag(pgrid, N)
wm = W / W.sum()
roe = (wm * qm ** 2).sum() / (wm * (2 * qm - qm ** 2)).sum()

near = float(np.mean(np.abs(phat - TAU) <= 0.05))
out = {
    "frozen_at": subprocess.check_output(["date", "--iso-8601=seconds"]).decode().strip(),
    "source_file": os.path.basename(P3), "n_prompts": len(Q), "N": N, "tau": TAU,
    "seed": SEED, "nmc": NMC,
    "pred": float(np.nanmean(ratio)),
    "lo": float(np.nanpercentile(ratio, 2.5)),
    "hi": float(np.nanpercentile(ratio, 97.5)),
    "roe": float(roe),
    "pred_flag_size": float((wm * qm).sum() * len(Q)),
    "pred_union_size": float(np.mean(union)),
    "mean_p": float(phat.mean()), "sd_p": float(phat.std()),
    "share_near_tau": near,
    "share_p_zero": float(np.mean(k == 0)),
    "share_p_one": float(np.mean(k == n)),
    "invalid_share": float(inval / (len(Q) * N)),
    "eff_threshold": float(np.floor(TAU * N) / N),
    "support_points": int((W > 1e-6).sum()),
}
json.dump(out, open(os.path.join(AN, "gsm8k_prediction.json"), "w"), indent=2)
for kk, vv in out.items():
    print(f"  {kk:20s} {vv}")
print("\nFROZEN. gsm8k_measure.py reads this file and cannot change it.")
