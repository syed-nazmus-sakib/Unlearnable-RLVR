"""A78 stage 1 — the PREDICTION, from the fresh-train pass 3 and ONLY that pass.

Run when pass 3 exists and passes 1 and 2 DO NOT. The only result file this
script names is the p3G one; it cannot open a measurement pass even if one is
present. Estimator identical to A74's: Kiefer-Wolfowitz EM on a 1001-point
grid, seeded Monte Carlo.

It also performs the PRE-FLIGHT POWER CHECK, which uses prediction-side data
only and therefore cannot leak the answer:

  proceed iff projected union >= 655, the union needed for half-width <= 0.0295
  (= delta - |offset|, the 5.00 GPU-h design). Below that this test would be
  weaker than the design already declined as a coin flip, and the remaining
  4.8 GPU-h should not be spent without a decision.

  the 948 target (half-width <= delta - 2|offset|, the design authorised) is
  reported separately as met or not met.

  $HOME/miniconda3/envs/vllm/bin/python gsm8k_train_predict.py
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
HW0, U0 = 0.0386, 382         # A74's realised half-width and union, for scaling
U_PROCEED, U_TARGET = 655, 948

# '@', not '_'.
hits = sorted(glob.glob(os.path.join(RES, "gsm8k_tr_p3G_gsm8k_train_fresh@*.json")))
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

pred_union = float(np.mean(union))
proj_hw = HW0 * (U0 / pred_union) ** 0.5

out = {
    "frozen_at": subprocess.check_output(["date", "--iso-8601=seconds"]).decode().strip(),
    "source_file": os.path.basename(P3), "n_prompts": len(Q), "N": N, "tau": TAU,
    "seed": SEED, "nmc": NMC, "w_reused_from_a74": 0.0,
    "pred": float(np.nanmean(ratio)),
    "lo": float(np.nanpercentile(ratio, 2.5)),
    "hi": float(np.nanpercentile(ratio, 97.5)),
    "roe": float(roe),
    "pred_flag_size": float((wm * qm).sum() * len(Q)),
    "pred_union_size": pred_union,
    "projected_half_width": float(proj_hw),
    "u_proceed": U_PROCEED, "u_target": U_TARGET,
    "power_proceed": bool(pred_union >= U_PROCEED),
    "power_target_met": bool(pred_union >= U_TARGET),
    "mean_p": float(phat.mean()), "sd_p": float(phat.std()),
    "share_near_tau": float(np.mean(np.abs(phat - TAU) <= 0.05)),
    "share_p_zero": float(np.mean(k == 0)),
    "share_p_one": float(np.mean(k == n)),
    "invalid_share": float(inval / (len(Q) * N)),
    "eff_threshold": float(np.floor(TAU * N) / N),
    "support_points": int((W > 1e-6).sum()),
}
json.dump(out, open(os.path.join(AN, "gsm8k_train_prediction.json"), "w"), indent=2)
for kk, vv in out.items():
    print(f"  {kk:22s} {vv}")

print("\nPRE-FLIGHT POWER CHECK (prediction-side data only)")
print(f"  projected union {pred_union:.0f}   projected half-width {proj_hw:.4f}")
print(f"  proceed floor  union >= {U_PROCEED} (hw <= 0.0295): "
      f"{'MET' if out['power_proceed'] else 'NOT MET -- STOP'}")
print(f"  authorised target union >= {U_TARGET} (hw <= 0.0245): "
      f"{'MET' if out['power_target_met'] else 'NOT MET'}")
print("\nFROZEN. gsm8k_train_measure.py reads this file and cannot change it.")
raise SystemExit(0 if out["power_proceed"] else 4)
