"""A79.5 — dense-grid summary macros for the budget curve.

Recomputes the dense curve over N in [8, 520] through the SAME code path as
make_figure8.py (same four result files, same guard, same 1001-point grid, same
3000 EM iterations, same seed 20260812), reproduces the seven stored rows of
budget_rule.json to 1e-12 first, and emits the summary as macros.

TWO RANGES ARE REPORTED AND BOTH ARE LABELLED:
  the FIGURE range, N in [8, 160], which is what a reader of figure 8 sees;
  the FULL range, N in [8, 520], which is needed to classify N = 256 and 512,
  two of the seven rows of the published table.

  $HOME/miniconda3/envs/vllm/bin/python emit_budget_curve_numbers.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binom

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
AN = os.path.join(EXP, "analysis")
OUT_TEX = os.path.abspath(os.path.join(EXP, "..", "..", "TRANSFER_OUT",
                                       "budget_curve_numbers.tex"))

TAU, SEED, NMC, GRID, EM_ITERS = 0.1, 20260812, 4000, 1001, 3000
TOL = 1e-12
FIG_HI, FULL_HI, N_LO = 160, 520, 8
TABLE_NS = [8, 16, 32, 64, 128, 256, 512]

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
K256 = np.array([A128[q].sum() for q in Q]) + np.array([B128[q].sum() for q in Q])
N256 = np.array([len(A128[q]) for q in Q]) + np.array([len(B128[q]) for q in Q])

pgrid = np.linspace(0, 1, GRID)
LIK = binom.pmf(K256[:, None], N256[:, None], pgrid[None, :])


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
_r = np.random.default_rng(SEED)
_p = _r.choice(pgrid, size=(NMC, len(Q)), p=W / W.sum())
IDX = np.rint(_p * (GRID - 1)).astype(np.int32)
assert np.allclose(pgrid[IDX], _p)
UA, UB = _r.random(IDX.shape), _r.random(IDX.shape)
del _p


def predict(n, block=None, nblock=1):
    q = (binom.cdf(np.floor(TAU * n), n, pgrid) if block is None
         else binom.cdf(np.floor(TAU * block), block, pgrid) ** nblock)[IDX]
    a, b = UA < q, UB < q
    inter, union = (a & b).sum(1), (a | b).sum(1)
    ratio = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return (float(np.nanmean(ratio)), float(np.nanpercentile(ratio, 2.5)),
            float(np.nanpercentile(ratio, 97.5)))


# ---- reproduction check, before any summary is formed ----------------------
STORED = json.load(open(os.path.join(AN, "budget_rule.json")))["table"]
FIELDS = ["deep", "deep_lo", "deep_hi", "shallow", "eff_threshold"]
for ns, row in STORED.items():
    n = int(ns)
    d, lo, hi = predict(n)
    assert abs(d - row["deep"]) < TOL and abs(lo - row["deep_lo"]) < TOL
    assert abs(hi - row["deep_hi"]) < TOL
    assert abs(np.floor(TAU * n) / n - row["eff_threshold"]) < TOL
    if n >= 16 and n // 4 >= 4:
        assert abs(predict(n, n // 4, 4)[0] - row["shallow"]) < TOL
print(f"reproduction: {len(STORED)} rows x {len(FIELDS)} fields to {TOL:g}")

NS = np.arange(N_LO, FULL_HI + 1)
deep = np.array([predict(int(n))[0] for n in NS])
print(f"evaluated {len(NS)} budgets")


def summarise(hi):
    m = NS <= hi
    ns, d = NS[m], deep[m]
    step = np.diff(d)
    falls = step < 0
    i = int(np.argmin(step))
    # worst pair: largest drop from some N=b to a LARGER N=a
    suff = np.minimum.accumulate(d[::-1])[::-1]
    drops = d[:-1] - suff[1:]
    j = int(np.argmax(drops))
    k = int(j + 1 + np.argmin(d[j + 1:]))
    return {
        "lo": int(ns[0]), "hi": int(ns[-1]), "budgets": int(len(ns)),
        "steps": int(len(step)), "falls": int(falls.sum()),
        "falls_pct": 100.0 * falls.sum() / len(step),
        "worst_step": float(-step[i]), "worst_step_from": int(ns[i]),
        "worst_step_to": int(ns[i + 1]),
        "worst_pair_drop": float(drops[j]), "worst_pair_small": int(ns[j]),
        "worst_pair_large": int(ns[k]),
        "worst_pair_small_j": float(d[j]), "worst_pair_large_j": float(d[k]),
    }


FIG, FULL = summarise(FIG_HI), summarise(FULL_HI)

# ---- are the seven published rows on a rise or a descent? ------------------
pos = {int(n): i for i, n in enumerate(NS)}
rows = []
for n in TABLE_NS:
    i = pos[n]
    # N=8 is the FIRST grid point: it has no step in. Indexing deep[i-1] there
    # would wrap to the last element of the array, which is N=520.
    into = deep[i] - deep[i - 1] if i > 0 else float("nan")
    out = deep[i + 1] - deep[i] if i + 1 < len(deep) else float("nan")
    rows.append((n, into, out))
    where = ("undefined (first grid point)" if np.isnan(into)
             else "RISING" if into > 0 else "ON A DESCENT")
    print(f"  N={n:4d}  step in {into:+.5f}   step out {out:+.5f}   {where}")
defined = [(n, a, b) for n, a, b in rows if not np.isnan(a)]
n_desc = sum(1 for _, a, _ in defined if a < 0)

_st = np.diff(deep)
_last_fall = int(NS[1:][_st < 0].max())
_fall_share_beyond = float(100.0 * (_st[NS[1:] > FIG_HI] < 0).mean())
print(f"  falls beyond N={FIG_HI}: {_fall_share_beyond:.0f}% of steps")
print(f"  last budget the curve falls INTO: N={_last_fall}")
json.dump({"fig": FIG, "full": FULL, "tol": TOL,
           "last_descending_budget": _last_fall,
           "fall_share_beyond_fig": _fall_share_beyond,
           "table_rows": [{"n": n, "step_in": a, "step_out": b} for n, a, b in rows],
           "n_on_descent": n_desc},
          open(os.path.join(AN, "budget_curve_summary.json"), "w"), indent=1)

W_ = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour",
      128: "OneTwentyEight", 256: "TwoFiftySix", 512: "FiveTwelve"}
L = []
A = L.append
A("% budget_curve_numbers.tex -- A79.5. Dense-grid summary of the A65 budget")
A("% curve. Generated by EXPERIMENTS/E1_rescue/emit_budget_curve_numbers.py.")
A("%")
A("% THE HEADLINE IS A DIRECTION, NOT A WOBBLE, AND IT IS A SMALL-N FACT.")
A("% Over N = \\bcLo to \\bcHi the curve FALLS on \\bcFallsPct percent of its")
A("% steps. \"Not monotone\" reads as \"rises with wobbles\" and that is close to")
A("% the opposite: between multiples of 10 the effective threshold")
A("% floor(tau N)/N decays as 1/N and agreement DECREASES with more sampling,")
A("% recovering only when floor(tau N) increments.")
A("%")
A("% THE RANGE MUST TRAVEL WITH THE NUMBER. Beyond N = \\bcHi the precision")
A("% gain outweighs the threshold decay and only \\bcFallsPctBeyondFig percent")
A("% of steps fall, which is why the full-range share is \\bcFullFallsPct and")
A("% not \\bcFallsPct. QUOTING \\bcFallsPct WITHOUT ITS RANGE IS A MISQUOTE.")
A("% The effect is concentrated exactly where this literature operates.")
A("%")
A("% \\bcLastDescending IS GRID-LIMITED, NOT A CROSSOVER: it is the largest")
A("% budget evaluated, and the curve still falls into it. Descents do not stop,")
A("% they become rare.")
A("%")
A("% N=128 IS AT THE CROSSOVER AND \\bcLocalOneTwentyEight OVERSTATES IT: the")
A("% step in is \\bcStepInOneTwentyEight and the step out \\bcStepOutOneTwentyEight,")
A("% so it is flat to four decimals rather than meaningfully falling. It is")
A("% counted in \\bcRowsOnDescent because the sign is negative; do not lean on")
A("% that one row.")
A("%")
A("% TWO RANGES, BOTH LABELLED. \\bcLo--\\bcHi is what figure 8 plots.")
A("% \\bcFullLo--\\bcFullHi is the range needed to classify N=256 and N=512,")
A("% two of the seven rows of the published table. Quote the figure range")
A("% alongside the figure; quote the full range only with its own name.")
A("%")
A("% CONDITIONAL ON THE A65 BINOMIAL MODEL, like everything derived from it.")
A("% G is not stored; it is recomputed by the identical code path and the")
A("% seven stored rows are reproduced to \\bcTol across \\bcCheckedFields")
A("% fields before any of this is formed.")
A("")
A(f"\\newcommand{{\\bcTol}}{{{TOL:g}}}")
A(f"\\newcommand{{\\bcCheckedRows}}{{{len(STORED)}}}")
A(f"\\newcommand{{\\bcCheckedFields}}{{{len(FIELDS)}}}")
_fieldlist = ", ".join(FIELDS).replace("_", "\\_")
A(f"\\newcommand{{\\bcCheckedFieldList}}{{{_fieldlist}}}")
A(f"\\newcommand{{\\bcSeed}}{{{SEED}}}")
A(f"\\newcommand{{\\bcNMC}}{{{NMC}}}")
A(f"\\newcommand{{\\bcPGrid}}{{{GRID}}}")
A("")
A("% ---- the range figure 8 plots ----")
for k, v in [("Lo", FIG["lo"]), ("Hi", FIG["hi"]), ("Budgets", FIG["budgets"]),
             ("Steps", FIG["steps"]), ("Falls", FIG["falls"])]:
    A(f"\\newcommand{{\\bc{k}}}{{{v}}}")
A(f"\\newcommand{{\\bcFallsPct}}{{{FIG['falls_pct']:.0f}}}")
A(f"\\newcommand{{\\bcWorstStep}}{{{FIG['worst_step']:.4f}}}")
A(f"\\newcommand{{\\bcWorstStepFrom}}{{{FIG['worst_step_from']}}}")
A(f"\\newcommand{{\\bcWorstStepTo}}{{{FIG['worst_step_to']}}}")
A(f"\\newcommand{{\\bcWorstPairDrop}}{{{FIG['worst_pair_drop']:.4f}}}")
A(f"\\newcommand{{\\bcWorstPairSmall}}{{{FIG['worst_pair_small']}}}")
A(f"\\newcommand{{\\bcWorstPairLarge}}{{{FIG['worst_pair_large']}}}")
A(f"\\newcommand{{\\bcWorstPairSmallJ}}{{{FIG['worst_pair_small_j']:.4f}}}")
A(f"\\newcommand{{\\bcWorstPairLargeJ}}{{{FIG['worst_pair_large_j']:.4f}}}")
A("")
A("% ---- the extended range, needed only to classify N=256 and N=512 ----")
A(f"\\newcommand{{\\bcFullLo}}{{{FULL['lo']}}}")
A(f"\\newcommand{{\\bcFullHi}}{{{FULL['hi']}}}")
A(f"\\newcommand{{\\bcFullBudgets}}{{{FULL['budgets']}}}")
A(f"\\newcommand{{\\bcFullFalls}}{{{FULL['falls']}}}")
A(f"\\newcommand{{\\bcFullSteps}}{{{FULL['steps']}}}")
A(f"\\newcommand{{\\bcFullFallsPct}}{{{FULL['falls_pct']:.0f}}}")
A(f"\\newcommand{{\\bcFallsPctBeyondFig}}{{{_fall_share_beyond:.0f}}}")
A(f"\\newcommand{{\\bcLastDescending}}{{{_last_fall}}}")
A("")
A("% ---- the seven published rows: rising, or sitting on a descent? ----")
A(f"\\newcommand{{\\bcRowsOnDescent}}{{{n_desc}}}")
A(f"\\newcommand{{\\bcRowsClassifiable}}{{{len(defined)}}}")
A(f"\\newcommand{{\\bcRowsTotal}}{{{len(TABLE_NS)}}}")
for n, into, out in rows:
    if np.isnan(into):
        A(f"\\newcommand{{\\bcStepIn{W_[n]}}}{{n/a}}")
        A(f"\\newcommand{{\\bcLocal{W_[n]}}}{{unclassifiable, first grid point}}")
    else:
        A(f"\\newcommand{{\\bcStepIn{W_[n]}}}{{{into:+.5f}}}")
        A(f"\\newcommand{{\\bcLocal{W_[n]}}}"
          f"{{{'rising' if into > 0 else 'on a descent'}}}")
    A(f"\\newcommand{{\\bcStepOut{W_[n]}}}{{{out:+.5f}}}")
open(OUT_TEX, "w").write("\n".join(L) + "\n")
print(f"\nwrote {OUT_TEX}  ({sum(1 for x in L if x.startswith(chr(92)+'newcommand'))} macros)")
print(f"figure range  {FIG['lo']}-{FIG['hi']}: falls on {FIG['falls']}/{FIG['steps']} "
      f"= {FIG['falls_pct']:.0f}%")
print(f"full range    {FULL['lo']}-{FULL['hi']}: falls on {FULL['falls']}/{FULL['steps']} "
      f"= {FULL['falls_pct']:.0f}%")
print(f"worst single step: -{FIG['worst_step']:.4f} from N={FIG['worst_step_from']} "
      f"to N={FIG['worst_step_to']}")
print(f"worst pair: N={FIG['worst_pair_small']} ({FIG['worst_pair_small_j']:.4f}) beats "
      f"N={FIG['worst_pair_large']} ({FIG['worst_pair_large_j']:.4f}) by "
      f"{FIG['worst_pair_drop']:.4f}")
print(f"published rows sitting on a descent: {n_desc} of {len(TABLE_NS)}")
