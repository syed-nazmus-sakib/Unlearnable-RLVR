"""A65 — a sampling-budget rule for threshold-defined sets, and its validation.

Every rule here was fixed in A65.1-A65.4 before this ran. Guarded throughout.
Validation comes FIRST; the table is printed only if a branch allows it.

  $HOME/miniconda3/envs/vllm/bin/python budget_rule.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binom

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
TAU, SEED, NBOOT, NMC = 0.1, 20260812, 2000, 4000
GRID = 1001                       # p-grid for the deconvolution
EM_ITERS = 3000

P128_A = ["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
          "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"]
P128_B = ["qwen_0.5b_ckpt1_n128_rep2_simplelr_qwen_level1to4_sub1k@train_0.7_128_1024_None_1_qwen.json"]
P32_A = ["qwen_0.5b_run1_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json"]
P32_B = ["qwen_0.5b_ckpt1_rep2_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_1_qwen.json"]


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def load(files):
    per = defaultdict(list)
    for f in files:
        for r in json.load(open(os.path.join(RES, f))):
            per[int(r["question_id"])].append(guarded(r.get("verification", {}) or {}))
    return {q: np.array(v, dtype=bool) for q, v in per.items()}


A128, B128 = load(P128_A), load(P128_B)
A32, B32 = load(P32_A), load(P32_B)
Q = sorted(set(A128) & set(B128) & set(A32) & set(B32))
print(f"prompts present in all four passes: {len(Q)}")
kA = np.array([A128[q].sum() for q in Q])
kB = np.array([B128[q].sum() for q in Q])
nA = np.array([len(A128[q]) for q in Q])
nB = np.array([len(B128[q]) for q in Q])
K256, N256 = kA + kB, nA + nB
assert set(nA) == {128} and set(nB) == {128}, "N=128 passes are not all 128"
print(f"pooled: N={N256[0]} per prompt, mean pass rate {K256.sum()/N256.sum():.4f}")

# ---------------------------------------------------------------- measurements
rng = np.random.default_rng(SEED)


def jac(flagA, flagB):
    u = (flagA | flagB).sum()
    return (flagA & flagB).sum() / u if u else np.nan


def boot_ci(flagA, flagB):
    idx = rng.integers(0, len(flagA), size=(NBOOT, len(flagA)))
    v = np.array([jac(flagA[i], flagB[i]) for i in idx])
    return np.nanpercentile(v, [2.5, 97.5])


def deep_flag(passdict, n):
    return np.array([passdict[q].sum() <= np.floor(TAU * n) for q in Q])


def shallow_flag(passdict, block, nblock):
    out = []
    for q in Q:
        v = passdict[q]
        out.append(all(v[b*block:(b+1)*block].sum() <= np.floor(TAU * block)
                       for b in range(nblock)))
    return np.array(out)


MEAS = {}
f = (deep_flag(A32, 32), deep_flag(B32, 32))
MEAS["N=32 deep"] = (jac(*f), *boot_ci(*f))
f = (deep_flag(A128, 128), deep_flag(B128, 128))
MEAS["N=128 deep"] = (jac(*f), *boot_ci(*f))
f = (shallow_flag(A128, 32, 4), shallow_flag(B128, 32, 4))
MEAS["N=128 shallow 4x32"] = (jac(*f), *boot_ci(*f))
print("\nMEASURED (guarded, recomputed here from rollouts, one bootstrap method):")
for k, (v, lo, hi) in MEAS.items():
    print(f"  {k:22s} {v:.4f}  [{lo:.4f}, {hi:.4f}]")

# ---------------------------------------------------------------- G(p)
pgrid = np.linspace(0, 1, GRID)
LIK = binom.pmf(K256[:, None], N256[:, None], pgrid[None, :])   # n_prompts x GRID


def em(lik, iters=EM_ITERS):
    """Nonparametric MLE of the mixing distribution (Kiefer-Wolfowitz, EM)."""
    w = np.full(lik.shape[1], 1.0 / lik.shape[1])
    for _ in range(iters):
        post = lik * w
        s = post.sum(1, keepdims=True)
        s[s == 0] = 1.0
        w_new = (post / s).mean(0)
        if np.abs(w_new - w).max() < 1e-12:
            w = w_new
            break
        w = w_new
    return w


W = em(LIK)
print(f"\ndeconvolved G: {int((W > 1e-6).sum())} support points with mass > 1e-6")
phat = K256 / N256
print(f"  plug-in mean {phat.mean():.4f} sd {phat.std():.4f}   "
      f"deconvolved mean {(W*pgrid).sum():.4f} "
      f"sd {np.sqrt((W*pgrid**2).sum() - (W*pgrid).sum()**2):.4f}")


def qflag(p, n, block=None, nblock=1):
    """P(flagged) at budget n; shallow if block given."""
    if block is None:
        return binom.cdf(np.floor(TAU * n), n, p)
    return binom.cdf(np.floor(TAU * block), block, p) ** nblock


def predict(n, block=None, nblock=1, mode="deconv", nmc=NMC, seed=SEED):
    """E[ratio] by Monte Carlo, plus the closed-form ratio of expectations."""
    r = np.random.default_rng(seed)
    if mode == "deconv":
        p = r.choice(pgrid, size=(nmc, len(Q)), p=W / W.sum())
    else:
        p = np.tile(phat, (nmc, 1))
    q = qflag(p, n, block, nblock)
    a = r.random(q.shape) < q
    b = r.random(q.shape) < q
    inter = (a & b).sum(1)
    union = (a | b).sum(1)
    ratio = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    qm = qflag(pgrid if mode == "deconv" else phat, n, block, nblock)
    wm = W / W.sum() if mode == "deconv" else np.full(len(phat), 1 / len(phat))
    roe = (wm * qm**2).sum() / (wm * (2*qm - qm**2)).sum()
    return np.nanmean(ratio), np.nanpercentile(ratio, [2.5, 97.5]), roe


print("\n=== VALIDATION (A65.4) ===")
print(f"{'target':22s} {'measured':>22s} {'predicted (deconv)':>26s} "
      f"{'plug-in':>9s} {'RoE':>7s}  overlap?")
TARGETS = [("N=32 deep", dict(n=32)),
           ("N=128 deep", dict(n=128)),
           ("N=128 shallow 4x32", dict(n=128, block=32, nblock=4))]
verdict = {}
for name, kw in TARGETS:
    m, mlo, mhi = MEAS[name]
    pm, (plo, phi), roe = predict(**kw, mode="deconv")
    pin, _, _ = predict(**kw, mode="plugin")
    ok = not (phi < mlo or plo > mhi)
    verdict[name] = ok
    print(f"{name:22s} {m:6.4f} [{mlo:.4f},{mhi:.4f}] "
          f"{pm:8.4f} [{plo:.4f},{phi:.4f}] {pin:9.4f} {roe:7.4f}  "
          f"{'YES' if ok else 'NO'}   err {pm-m:+.4f}")

npass = sum(verdict.values())
branch = "A" if npass == 3 else ("C" if npass == 0 or not verdict["N=128 shallow 4x32"]
                                 else "B")
print(f"\n{npass} of 3 within uncertainty.  shallow predicted: "
      f"{verdict['N=128 shallow 4x32']}")
print(f"A65.4 BRANCH {branch} FIRES")

out = {"tau": TAU, "n_prompts": len(Q), "seed": SEED,
       "measured": {k: list(v) for k, v in MEAS.items()},
       "verdict": verdict, "branch": branch,
       "plugin_sd": float(phat.std()),
       "deconv_sd": float(np.sqrt((W*pgrid**2).sum() - (W*pgrid).sum()**2))}

if branch == "C":
    print("\nBRANCH C: no table is printed. A65.4 forbids it.")
else:
    label = "PRESCRIPTIVE" if branch == "A" else "INDICATIVE ONLY (branch B)"
    print(f"\n=== TABLE ({label}) — tau={TAU}, Qwen2.5-0.5B on "
          f"simplelr_qwen_level1to4_sub1k, cap 1024 ===")
    print(f"{'N':>5s} {'thresh':>8s} {'deep E[J]':>11s} {'shallow (K=4)':>15s}")
    rows = {}
    for n in (8, 16, 32, 64, 128, 256, 512):
        d, dci, _ = predict(n=n)
        thr = np.floor(TAU*n)/n
        if n >= 16 and n // 4 >= 4:
            s, _, _ = predict(n=n, block=n//4, nblock=4)
        else:
            s = float("nan")
        rows[n] = {"deep": d, "deep_lo": dci[0], "deep_hi": dci[1], "shallow": s,
                   "eff_threshold": thr}
        print(f"{n:5d} {thr:8.4f} {d:11.4f} {s:15.4f}")
    print("\n  N required for a target expected Jaccard (deep rule):")
    need = {}
    for tgt in (0.80, 0.90, 0.95):
        n = 8
        while n <= 65536 and predict(n=n)[0] < tgt:
            n *= 2
        need[tgt] = n if n <= 65536 else None
        print(f"    J >= {tgt:.2f}:  N = {need[tgt]}" if need[tgt]
              else f"    J >= {tgt:.2f}:  not reached by N=65536")
    out["table"] = rows
    out["n_required"] = {str(k): v for k, v in need.items()}

json.dump(out, open(os.path.join(EXP, "analysis", "budget_rule.json"), "w"), indent=2)
print("\nwrote analysis/budget_rule.json")
