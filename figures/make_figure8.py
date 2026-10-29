"""Figure 8 — the budget curve on a dense integer grid, with the threshold below.

A79.4. Authorised 2026-08-16 as a COMPUTATION, not a new measurement: the A65
closed form evaluated against the same mixing distribution, at every integer N
instead of at seven powers of two.

WHAT HAD TO BE RE-DERIVED, AND WHY THAT IS SAFE
-----------------------------------------------
G IS NOT STORED. `analysis/budget_rule.json` keeps the seven table rows and two
summary sd's, but not the 1001-point weight vector. So G is recomputed here by
the identical code path in budget_rule.py -- same four result files, same
guard, same 1001-point grid, same 3000 EM iterations, same seed 20260812.

THE GUARANTEE IS NOT "same code", IT IS REPRODUCTION. This script asserts that
all seven stored rows -- deep, deep_lo, deep_hi, shallow, eff_threshold --
reproduce to 1e-12 before it draws anything. If G had drifted, the assert fires
and no figure is produced.

WHY THE MONTE CARLO IS NOT RE-DRAWN PER N. predict() in budget_rule.py reseeds
to SEED on every call and the draw shapes do not depend on N, so the p-matrix
and the two uniform matrices are IDENTICAL at every budget. They are therefore
drawn once and reused, which is exactly equivalent, makes the dense grid cheap,
and has the property that matters for this figure: the Monte Carlo error is
perfectly correlated across N, so it cannot masquerade as the oscillation. The
dips in the curve are the threshold, not noise.

THE SHALLOW RULE IS ONLY DEFINED WHEN 4 BLOCKS TILE N, so it is drawn on
N divisible by 4 only. It has NO interval anywhere in storage and none is
invented: it is a bare line and the caption says so.

  $HOME/miniconda3/envs/vllm/bin/python make_figure8.py
"""
import json
import os
from collections import defaultdict

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
AN = os.path.join(EXP, "analysis")
OUT = os.path.join(EXP, "figures_pub")
os.makedirs(OUT, exist_ok=True)

# ---- constants, copied from budget_rule.py and asserted against its output --
TAU, SEED, NMC, GRID, EM_ITERS = 0.1, 20260812, 4000, 1001, 3000
N_LO, N_HI = 8, 260                      # dense grid: EVERY integer in [8, 260] (covers N=242 target)
BAND = (8, 16)                           # the budget range this literature uses
MARK = 128
MARK_TARGET = 242

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


A128, B128, A32, B32 = load(P128_A), load(P128_B), load(P32_A), load(P32_B)
Q = sorted(set(A128) & set(B128) & set(A32) & set(B32))
kA = np.array([A128[q].sum() for q in Q])
kB = np.array([B128[q].sum() for q in Q])
K256 = kA + kB
N256 = np.array([len(A128[q]) for q in Q]) + np.array([len(B128[q]) for q in Q])

pgrid = np.linspace(0, 1, GRID)
LIK = binom.pmf(K256[:, None], N256[:, None], pgrid[None, :])


def em(lik, iters=EM_ITERS):
    w = np.full(lik.shape[1], 1.0 / lik.shape[1])
    for _ in range(iters):
        post = lik * w
        s = post.sum(1, keepdims=True)
        s[s == 0] = 1.0
        w_new = (post / s).mean(0)
        if np.abs(w_new - w).max() < 1e-12:
            return w_new
        w = w_new
    return w


W = em(LIK)
WN = W / W.sum()

# ---- the draws, made ONCE because budget_rule.py's are N-independent --------
_r = np.random.default_rng(SEED)
_p = _r.choice(pgrid, size=(NMC, len(Q)), p=WN)
IDX = np.rint(_p * (GRID - 1)).astype(np.int32)      # exact: pgrid is a linspace
assert np.allclose(pgrid[IDX], _p), "grid index recovery is not exact"
UA = _r.random(IDX.shape)
UB = _r.random(IDX.shape)
del _p


def qm_of(n, block=None, nblock=1):
    if block is None:
        return binom.cdf(np.floor(TAU * n), n, pgrid)
    return binom.cdf(np.floor(TAU * block), block, pgrid) ** nblock


def predict(n, block=None, nblock=1):
    """Mean and 95 % interval of the ratio, identical to budget_rule.predict."""
    q = qm_of(n, block, nblock)[IDX]
    a, b = UA < q, UB < q
    inter = (a & b).sum(1)
    union = (a | b).sum(1)
    ratio = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return (float(np.nanmean(ratio)),
            float(np.nanpercentile(ratio, 2.5)),
            float(np.nanpercentile(ratio, 97.5)))


# ---- REPRODUCTION CHECK: the seven stored rows, before anything is drawn ----
STORED = json.load(open(os.path.join(AN, "budget_rule.json")))["table"]
for ns, row in STORED.items():
    n = int(ns)
    d, dlo, dhi = predict(n)
    assert abs(d - row["deep"]) < 1e-12, (n, d, row["deep"])
    assert abs(dlo - row["deep_lo"]) < 1e-12 and abs(dhi - row["deep_hi"]) < 1e-12, n
    assert abs(np.floor(TAU * n) / n - row["eff_threshold"]) < 1e-12, n
    if n >= 16 and n // 4 >= 4:
        s, _, _ = predict(n, block=n // 4, nblock=4)
        assert abs(s - row["shallow"]) < 1e-12, (n, s, row["shallow"])
print(f"reproduction check: all {len(STORED)} stored rows reproduce to 1e-12")

# ---- the dense grid --------------------------------------------------------
NS = np.arange(N_LO, N_HI + 1)
deep = np.empty(len(NS))
dlo = np.empty(len(NS))
dhi = np.empty(len(NS))
for i, n in enumerate(NS):
    deep[i], dlo[i], dhi[i] = predict(int(n))
thr = np.floor(TAU * NS) / NS

sh_n = np.array([n for n in NS if n % 4 == 0 and n >= 16])
shallow = np.array([predict(int(n), block=int(n) // 4, nblock=4)[0] for n in sh_n])

dips = [int(NS[i]) for i in range(1, len(NS)) if deep[i] < deep[i - 1]]
print(f"dense grid N={N_LO}..{N_HI}  ({len(NS)} integers)")
print(f"  budgets at which the DEEP curve falls as N rises: {len(dips)}")
print(f"  first ten: {dips[:10]}")

json.dump({"grid": [int(N_LO), int(N_HI)], "tau": TAU, "seed": SEED,
           "nmc": NMC, "em_iters": EM_ITERS, "p_grid": GRID,
           "n_prompts": len(Q), "descending_steps": dips,
           "deep": deep.tolist(), "deep_lo": dlo.tolist(), "deep_hi": dhi.tolist(),
           "eff_threshold": thr.tolist(),
           "shallow_n": sh_n.tolist(), "shallow": shallow.tolist()},
          open(os.path.join(AN, "budget_curve_dense.json"), "w"), indent=1)

# ---- draw ------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.6, 1.0],
                                          "hspace": 0.12})

for a in (ax, bx):
    a.grid(True, color="#eaeaea", linewidth=0.6, zorder=0)
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
    a.axvspan(*BAND, color="#f0f0f0", zorder=1)
    a.axvline(MARK, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=2)
    a.axvline(MARK_TARGET, color="#8c1515", lw=0.9, ls=(0, (2, 2)), zorder=2)

# Use Okabe-Ito colors: Vermilion for deep, Blue for shallow
C_DEEP = "#D55E00"
C_SHALLOW = "#0072B2"

ax.fill_between(NS, dlo, dhi, color=C_DEEP, alpha=0.15, lw=0, zorder=3)
ax.plot(NS, deep, color=C_DEEP, lw=1.6, zorder=5, label="deep (one pass of $N$)")
ax.plot(sh_n, shallow, color=C_SHALLOW, lw=1.4, ls=(0, (4, 2)), zorder=4,
        label="shallow (4 blocks of $N/4$) — no interval in storage")
ax.axhline(0.90, color="#8c1515", lw=0.7, ls=":", alpha=0.6, zorder=2)
ax.text(BAND[0] + 1.0, 0.955, "budgets in use\n(8–16)", fontsize=7.0,
        color="#555555", va="top", zorder=6)
ax.text(MARK + 3.0, 0.942, "$N=128$", fontsize=7.4, color="0.35",
        va="center", ha="left", zorder=6)
ax.text(MARK_TARGET - 3.0, 0.942, "$N=242$\n($J\\geq 0.90$)", fontsize=7.2,
        color="#8c1515", va="center", ha="right", zorder=6)
ax.set_ylabel("expected Jaccard between\ntwo independent passes")
ax.set_ylim(0.56, 0.96)
ax.legend(loc="lower right", frameon=False, borderpad=0.2)

bx.plot(NS, thr, color="0.25", lw=1.2, zorder=4)
bx.axhline(TAU, color=C_DEEP, lw=1.0, ls=(0, (2, 2)), zorder=3)
bx.text(N_HI - 1, TAU + 0.004, r"nominal $\tau=0.1$", fontsize=7.0,
        color=C_DEEP, ha="right", va="bottom", zorder=5)
bx.set_ylabel("effective\n" + r"$\lfloor \tau N \rfloor / N$")
bx.set_xlabel("per-prompt sample budget $N$")
bx.set_ylim(0, 0.115)
bx.set_xlim(N_LO, N_HI)

fig.align_ylabels((ax, bx))
for ext in ("pdf", "png"):
    p = os.path.join(OUT, f"figure8.{ext}")
    fig.savefig(p, bbox_inches="tight", dpi=300)
print("wrote figure8.png and figure8.pdf")
