"""Emit analysis/budget_numbers_rule.tex — the A65 sampling-budget rule.

Deterministic: the closed form E|A and B| / E|A or B| is used for every table
cell and every N-required search, because it matched the Monte Carlo E[ratio] to
four decimals at all three validation points and the Monte Carlo search was
noise-dominated (it produced "N = 1917" where the curve is flat to +-0.003).
Monte Carlo is kept only for the validation intervals.

  $HOME/miniconda3/envs/vllm/bin/python emit_budget_numbers.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binom
import sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tools")))
from texguard import check as texcheck  # noqa: E402


EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
TAU, SEED, NBOOT, NMC, GRID = 0.1, 20260812, 2000, 6000, 1001
TABLE_N = (8, 16, 32, 64, 128, 256, 512)
TARGETS = (0.80, 0.90, 0.95)

P128A = ["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
         "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"]
P128B = ["qwen_0.5b_ckpt1_n128_rep2_simplelr_qwen_level1to4_sub1k@train_0.7_128_1024_None_1_qwen.json"]
P32A = ["qwen_0.5b_run1_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json"]
P32B = ["qwen_0.5b_ckpt1_rep2_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_1_qwen.json"]


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def load(files):
    per = defaultdict(list)
    for f in files:
        for r in json.load(open(os.path.join(RES, f))):
            per[int(r["question_id"])].append(guarded(r.get("verification", {}) or {}))
    return {q: np.array(v, dtype=bool) for q, v in per.items()}


A128, B128, A32, B32 = load(P128A), load(P128B), load(P32A), load(P32B)
Q = sorted(set(A128) & set(B128) & set(A32) & set(B32))
K = np.array([A128[q].sum() + B128[q].sum() for q in Q])
NN = np.full(len(Q), 256)

pg = np.linspace(0, 1, GRID)
lik = binom.pmf(K[:, None], NN[:, None], pg[None, :])
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
w /= w.sum()


def qgrid(n, block=None, nblock=1):
    if block is None:
        return binom.cdf(np.floor(TAU * n), n, pg)
    return binom.cdf(np.floor(TAU * block), block, pg) ** nblock


def roe(n, block=None, nblock=1):
    q = qgrid(n, block, nblock)
    return float((w * q ** 2).sum() / (w * (2 * q - q ** 2)).sum())


def size(n, block=None, nblock=1):
    return float((w * qgrid(n, block, nblock)).sum() * len(Q))


def mc(n, block=None, nblock=1):
    r = np.random.default_rng(SEED)
    p = r.choice(pg, size=(NMC, len(Q)), p=w)
    q = binom.cdf(np.floor(TAU * n), n, p) if block is None else \
        binom.cdf(np.floor(TAU * block), block, p) ** nblock
    a, b = r.random(q.shape) < q, r.random(q.shape) < q
    u, i = (a | b).sum(1), (a & b).sum(1)
    v = np.where(u > 0, i / np.maximum(u, 1), np.nan)
    return float(np.nanmean(v)), np.nanpercentile(v, [2.5, 97.5])


rng = np.random.default_rng(SEED)


def jac(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def measured(fa, fb):
    idx = rng.integers(0, len(fa), size=(NBOOT, len(fa)))
    v = np.array([jac(fa[i], fb[i]) for i in idx])
    return jac(fa, fb), *np.nanpercentile(v, [2.5, 97.5])


df = lambda d, n: np.array([d[q].sum() <= np.floor(TAU * n) for q in Q])
sf = lambda d, b, k: np.array([all(d[q][i*b:(i+1)*b].sum() <= np.floor(TAU * b)
                                   for i in range(k)) for q in Q])
M = {"ThirtyTwo": measured(df(A32, 32), df(B32, 32)),
     "OneTwentyEight": measured(df(A128, 128), df(B128, 128)),
     "Shallow": measured(sf(A128, 32, 4), sf(B128, 32, 4))}
P = {"ThirtyTwo": mc(32), "OneTwentyEight": mc(128), "Shallow": mc(128, 32, 4)}
SZ = {"ThirtyTwo": (int(df(A32, 32).sum()), int(df(B32, 32).sum()), size(32)),
      "OneTwentyEight": (int(df(A128, 128).sum()), int(df(B128, 128).sum()), size(128)),
      "Shallow": (int(sf(A128, 32, 4).sum()), int(sf(B128, 32, 4).sum()), size(128, 32, 4))}

# within-pass block overdispersion
ob, ex = [], []
for q in Q:
    v = A128[q]
    c = np.array([v[i*32:(i+1)*32].sum() for i in range(4)])
    p = v.mean()
    if p * (1 - p) > 0:
        ob.append(c.var(ddof=1))
        ex.append(32 * p * (1 - p))
OD = float(np.mean(ob) / np.mean(ex))


def need(tgt, block=False):
    step = 4 if block else 1
    rs = range(8 if block else 4, 20001, step)
    first = next((n for n in rs if roe(n, n // 4 if block else None, 4 if block else 1) >= tgt), None)
    stay = next((n for n in rs
                 if all(roe(m, m // 4 if block else None, 4 if block else 1) >= tgt
                        for m in range(n, min(n + 200, 20001), step))), None)
    return first, stay


L = []
a = L.append
a("% budget_numbers_rule.tex -- A65 sampling-budget rule for threshold sets.")
a("% Generated by EXPERIMENTS/E1_rescue/emit_budget_numbers.py. Do not hand-edit.")
a("%")
a("% WHAT TRANSFERS AND WHAT DOES NOT (A65.3.1). The METHOD is general: model the")
a("% threshold test as binomial, integrate over the pass-rate distribution, invert")
a("% for N. THE NUMBERS ARE NOT. Every N below is specific to tau = 0.1 and to the")
a("% pass-rate distribution of Qwen2.5-0.5B on simplelr_qwen_level1to4_sub1k at")
a("% response cap 1024. A different model, dataset or tau moves them.")
a("%")
a("% VALIDATION (A65.4): BRANCH A -- all three measured Jaccards fall inside the")
a("% prediction interval. But the residuals are not alike, and the caption must")
a("% not imply they are: deep is predicted to within 0.002, the SHALLOW rule to")
a("% within 0.032, one-directional (the model over-predicts shallow agreement).")
a("% The model reproduces the deep-over-shallow ORDERING and most of its size")
a("% (predicted gap 0.053 against a measured 0.086), so it earns branch A, but it")
a("% understates how much worse the shallow rule is.")
a("%")
a("% WHY IT IS NOT BLOCK DEPENDENCE. Within-pass block variance is 0.985 of the")
a("% binomial expectation -- blocks behave independently -- and the model's")
a("% predicted set sizes match the measured ones to within 1.8 percent for all")
a("% three rules. The flag probability is right; the residual is in the")
a("% between-pass agreement structure, which we have not identified.")
a("%")
a("% THE CURVE IS NOT MONOTONE IN N. The effective threshold is")
a("% floor(tau*N)/N, which oscillates below tau (0.0000 at N=8, 0.0625 at 16,")
a("% 0.0938 at 32, 0.1000 at 40, 0.0833 at 48). J can FALL as N rises. Quote the")
a("% stable-from N, never the first-crossing N; both are given below.")
a("%")
a("% BIAS DIRECTIONS, both stated because they do not cancel (A65.3.2, A65.3.3):")
a("%   plug-in p_hat inflates the spread of G and OVERSTATES agreement; the")
a("%   deconvolution removes it (sd 0.3923 -> 0.3918, a small correction here).")
a("%   A shared deterministic verifier correlates the two draws positively, which")
a("%   makes the independent model UNDERSTATE agreement.")
a("")
a("% ---- validation, measured (guarded, A24) against predicted ----")
for k in ("ThirtyTwo", "OneTwentyEight", "Shallow"):
    m, mlo, mhi = M[k]
    pm, (plo, phi) = P[k]
    a(f"\\newcommand{{\\bgMeas{k}}}{{{m:.4f}}}")
    a(f"\\newcommand{{\\bgMeas{k}Lo}}{{{mlo:.4f}}}")
    a(f"\\newcommand{{\\bgMeas{k}Hi}}{{{mhi:.4f}}}")
    a(f"\\newcommand{{\\bgPred{k}}}{{{pm:.4f}}}")
    a(f"\\newcommand{{\\bgPred{k}Lo}}{{{plo:.4f}}}")
    a(f"\\newcommand{{\\bgPred{k}Hi}}{{{phi:.4f}}}")
    a(f"\\newcommand{{\\bgErr{k}}}{{{pm - m:+.4f}}}")
    a(f"\\newcommand{{\\bgSizeMeas{k}}}{{{(SZ[k][0]+SZ[k][1])/2:.0f}}}")
    a(f"\\newcommand{{\\bgSizePred{k}}}{{{SZ[k][2]:.0f}}}")
a(f"\\newcommand{{\\bgBlockOverdispersion}}{{{OD:.3f}}}")
a(f"\\newcommand{{\\bgGapMeasured}}{{{M['OneTwentyEight'][0]-M['Shallow'][0]:.4f}}}")
a(f"\\newcommand{{\\bgGapPredicted}}{{{P['OneTwentyEight'][0]-P['Shallow'][0]:.4f}}}")
a("")
a("% ---- expected Jaccard by budget (deterministic closed form) ----")
WORD = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour",
        128: "OneTwentyEight", 256: "TwoFiftySix", 512: "FiveTwelve"}
for n in TABLE_N:
    a(f"\\newcommand{{\\bgDeepN{WORD[n]}}}{{{roe(n):.4f}}}")
    a(f"\\newcommand{{\\bgThreshN{WORD[n]}}}{{{np.floor(TAU*n)/n:.4f}}}")
    if n >= 16:
        a(f"\\newcommand{{\\bgShallowN{WORD[n]}}}{{{roe(n, n//4, 4):.4f}}}")
a("")
a("% ---- budget required for a target reproducibility ----")
TW = {0.80: "Eighty", 0.90: "Ninety", 0.95: "NinetyFive"}
for t in TARGETS:
    df_, ds = need(t, False)
    sf_, ss = need(t, True)
    a(f"\\newcommand{{\\bgNeedDeep{TW[t]}}}{{{ds}}}")
    a(f"\\newcommand{{\\bgNeedDeepFirst{TW[t]}}}{{{df_}}}")
    a(f"\\newcommand{{\\bgNeedShallow{TW[t]}}}{{{ss}}}")
    a(f"\\newcommand{{\\bgNeedShallowFirst{TW[t]}}}{{{sf_}}}")
    a(f"\\newcommand{{\\bgCostRatio{TW[t]}}}{{{ss/ds:.2f}}}")
a("")
a("% ---- the two rules do NOT flag the same set, so equal J is not equal answer ----")
a(f"\\newcommand{{\\bgSetDeepAtOneTwentyEight}}{{{size(128):.0f}}}")
a(f"\\newcommand{{\\bgSetShallowAtOneTwentyEight}}{{{size(128, 32, 4):.0f}}}")
a(f"\\newcommand{{\\bgTau}}{{{TAU:.1f}}}")
a(f"\\newcommand{{\\bgNPrompts}}{{{len(Q)}}}")

out = os.path.join(EXP, "analysis", "budget_numbers_rule.tex")
texcheck("\n".join(L) + "\n", path=out)
open(out, "w").write("\n".join(L) + "\n")
nm = sum(1 for x in L if x.startswith("\\newcommand"))
print("\n".join(x for x in L if not x.startswith("%")))
print(f"\nwrote {out}  ({nm} macros)")
