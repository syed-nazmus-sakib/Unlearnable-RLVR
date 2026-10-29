"""A66-B — cross-model validation of the A65 sampling-budget rule. CPU only.

For each model with an N=32 screening pass:
  split 32 into two DISJOINT blocks of 16 (file order = sampling order),
  measure the Jaccard between the two block-flag sets at tau = 0.1,
  deconvolve that model's own pass-rate distribution from the same 32 draws,
  predict the N=16 Jaccard and compare.

At N=16 the effective threshold is floor(0.1*16)/16 = 1/16 = 0.0625, not 0.1.

  $HOME/miniconda3/envs/vllm/bin/python budget_rule_crossmodel.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binom

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
TAU, SEED, NBOOT, NMC, GRID = 0.1, 20260812, 2000, 6000, 1001
BLOCK = 16

MODELS = {
    "Qwen2.5-0.5B ckpt1 (reference)":
        "qwen_0.5b_run1_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json",
    "Llama-3.2-3B (screen800)":
        "e4_gate_llama3b_e4_screen800@test_0.7_32_1024_None_0_qwen.json",
    "Llama-3.2-3B (screen800 v2)":
        "e4_gate_llama3b_v2_e4_screen800_v2@test_0.7_32_1024_None_0_e4v2.json",
    "Qwen2.5-1.5B base (E5 gate)":
        "e5_gate_qwen15b_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json",
}
pg = np.linspace(0, 1, GRID)


def load(f):
    per, inv = defaultdict(list), defaultdict(int)
    for r in json.load(open(os.path.join(RES, f))):
        v = r.get("verification", {}) or {}
        q = int(r["question_id"])
        per[q].append(int(v.get("score", 0)) == 1
                      and v.get("extracted_answer") != "[invalid]")
        inv[q] += v.get("extracted_answer") == "[invalid]"
    qs = sorted(q for q in per if len(per[q]) >= 2 * BLOCK)
    return qs, {q: np.array(per[q][:2 * BLOCK], dtype=bool) for q in qs}, inv


def em(k, n):
    lik = binom.pmf(k[:, None], n[:, None], pg[None, :])
    w = np.full(GRID, 1.0 / GRID)
    for _ in range(3000):
        post = lik * w
        s = post.sum(1, keepdims=True)
        s[s == 0] = 1.0
        nw = (post / s).mean(0)
        if np.abs(nw - w).max() < 1e-12:
            w = nw
            break
        w = nw
    return w / w.sum()


def jac(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


rng = np.random.default_rng(SEED)
print(f"threshold at N={BLOCK}: flag iff correct <= {int(np.floor(TAU*BLOCK))} "
      f"(effective tau = {np.floor(TAU*BLOCK)/BLOCK:.4f}, not {TAU})\n")
rows = {}
for name, f in MODELS.items():
    qs, per, inv = load(f)
    A = np.array([per[q][:BLOCK].sum() <= np.floor(TAU * BLOCK) for q in qs])
    B = np.array([per[q][BLOCK:2*BLOCK].sum() <= np.floor(TAU * BLOCK) for q in qs])
    m = jac(A, B)
    idx = rng.integers(0, len(qs), size=(NBOOT, len(qs)))
    mci = np.nanpercentile([jac(A[i], B[i]) for i in idx], [2.5, 97.5])

    k = np.array([per[q].sum() for q in qs])
    n = np.full(len(qs), 2 * BLOCK)
    w = em(k, n)
    q16 = binom.cdf(np.floor(TAU * BLOCK), BLOCK, pg)
    roe = float((w * q16**2).sum() / (w * (2*q16 - q16**2)).sum())
    r = np.random.default_rng(SEED)
    p = r.choice(pg, size=(NMC, len(qs)), p=w)
    qq = binom.cdf(np.floor(TAU * BLOCK), BLOCK, p)
    a, b = r.random(qq.shape) < qq, r.random(qq.shape) < qq
    u, i = (a | b).sum(1), (a & b).sum(1)
    v = np.where(u > 0, i / np.maximum(u, 1), np.nan)
    pm, pci = float(np.nanmean(v)), np.nanpercentile(v, [2.5, 97.5])

    phat = k / n
    flag_share = (A.sum() + B.sum()) / (2 * len(qs))
    inv_share = sum(inv[q] for q in qs) / (2 * BLOCK * len(qs))
    ok = not (pci[1] < mci[0] or pci[0] > mci[1])
    rows[name] = dict(n=len(qs), measured=m, mlo=mci[0], mhi=mci[1], pred=pm,
                      plo=pci[0], phi=pci[1], roe=roe, err=pm - m, inside=bool(ok),
                      flag_share=float(flag_share), invalid=float(inv_share),
                      p_zero=float((phat == 0).mean()), mean_p=float(phat.mean()),
                      pred_size=float((w * q16).sum() * len(qs)),
                      meas_size=float((A.sum() + B.sum()) / 2))
    print(f"=== {name} ===")
    print(f"  prompts {len(qs)}   mean pass rate {phat.mean():.4f}   "
          f"p_hat==0 {100*(phat==0).mean():.1f}%   [invalid] share {inv_share:.4f}")
    print(f"  FLAGGED SHARE {flag_share:.4f}   "
          f"set size measured {(A.sum()+B.sum())/2:.0f} predicted "
          f"{(w*q16).sum()*len(qs):.0f}")
    print(f"  measured  J = {m:.4f}  [{mci[0]:.4f}, {mci[1]:.4f}]")
    print(f"  predicted J = {pm:.4f}  [{pci[0]:.4f}, {pci[1]:.4f}]   "
          f"(closed form {roe:.4f})")
    print(f"  error {pm-m:+.4f}   inside: {'YES' if ok else 'NO'}\n")

json.dump(rows, open(os.path.join(EXP, "analysis", "budget_crossmodel.json"), "w"),
          indent=2)
print("wrote analysis/budget_crossmodel.json")
