"""A34.2 gate result + the A35.4 prediction, made BEFORE any main generation.

Reports, per A35.2, the screening span labelled as OPTIMISTIC, and predicts the
realised span by empirical-Bayes deconvolution of the binomial sampling noise at
N=32 (Beta-Binomial marginal likelihood, fitted by MLE).

A35.5: if the screening span is below 6x the run STOPS and the cuts are
reconsidered. That check is applied here and printed as a verdict.

CPU only.

  $HOME/miniconda3/envs/vllm/bin/python gate_report.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import binom

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
OUT = os.path.join(EXP, "analysis", "gate_report.md")
os.makedirs(os.path.join(EXP, "analysis"), exist_ok=True)
FILE = "e4_gate_llama3b_e4_screen800@test_0.7_32_1024_None_0_qwen.json"
N = 32
PER_TIER = 50           # main pass is 150 prompts = 3 x 50 (SCOPING 3b)


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


per, invalid, tot = defaultdict(list), 0, 0
inv_q = defaultdict(int)
for r in json.load(open(os.path.join(RES, FILE))):
    v = r.get("verification", {}) or {}
    tot += 1
    if v.get("extracted_answer") == "[invalid]":
        invalid += 1
        inv_q[int(r["question_id"])] += 1
    per[int(r["question_id"])].append(guarded(v))

qids = sorted(per)
C = np.array([sum(per[q]) for q in qids])          # guarded correct out of 32
P = C / N

L = []
W = L.append
W("# E4 gate — screening result and the pre-main-run prediction\n")
W(f"800 MATH-Hard (all Level 5) prompts x {N} rollouts, "
  "Llama-3.2-3B-Instruct, temperature 0.7. Guarded correctness (A24).\n")
W(f"Rollouts: {tot}; `[invalid]` extractions removed by the guard: **{invalid}** "
  f"({100 * invalid / tot:.2f} %).\n")
W("| quantity | value |")
W("|---|---|")
W(f"| prompts | {len(qids)} |")
W(f"| mean pass rate | {P.mean():.4f} |")
W(f"| median pass rate | {np.median(P):.4f} |")
W(f"| prompts with ZERO correct | **{int((C == 0).sum())}** ({100 * (C == 0).mean():.1f} %) |")
W(f"| prompts with >= 50 % correct | {int((P >= 0.5).sum())} ({100 * (P >= 0.5).mean():.1f} %) |")
W(f"| max correct out of {N} | {int(C.max())} |")
W("")

# ---- empirical Bayes: Beta-Binomial MLE (A35.4) -----------------------------
def nll(par):
    a, b = np.exp(par)
    return -np.sum(betaln(C + a, N - C + b) - betaln(a, b))


fit = minimize(nll, [0.0, 0.0], method="Nelder-Mead")
A_, B_ = np.exp(fit.x)
POST = (C + A_) / (N + A_ + B_)          # posterior mean true rate per prompt
W(f"**Empirical-Bayes fit:** Beta({A_:.3f}, {B_:.3f}) prior on the true pass "
  f"rate, fitted by Beta-Binomial MLE on the {len(qids)} screening counts. "
  f"Implied prior mean {A_ / (A_ + B_):.4f}. Shrinkage weight toward the prior: "
  f"{(A_ + B_) / (N + A_ + B_):.3f}.\n")

# ---- tier constructions -----------------------------------------------------
order = np.argsort(P, kind="stable")


def tiers_extreme(k):
    """bottom k, middle k, top k by screening pass rate"""
    mid = len(order) // 2
    return {"hard": order[:k], "mid": order[mid - k // 2: mid - k // 2 + k],
            "easy": order[-k:]}


def tiers_tercile():
    t = len(order) // 3
    return {"hard": order[:t], "mid": order[t:2 * t], "easy": order[2 * t:]}


def tiers_outer(pct):
    k = int(round(pct * len(order)))
    mid = len(order) // 2
    return {"hard": order[:k], "mid": order[mid - k // 2: mid - k // 2 + k],
            "easy": order[-k:]}


CONS = [("bottom/middle/top 50 (main pass is 150 = 3 x 50)", tiers_extreme(PER_TIER)),
        ("terciles of all 800", tiers_tercile()),
        ("outer 15 % (the A35.5 widening lever)", tiers_outer(0.15))]

W("## Tier spans — SCREENING, and therefore OPTIMISTIC (A35.2)\n")
W("**Every span in this table is measured on the N=32 screening pass that also "
  "defined the tiers, so it is an upper estimate of what the independent main "
  "generation will deliver.** The predicted realised span is beside it.\n")
W("| construction | n/tier | median correct, hard / mid / easy | **screening span** | "
  "EB-predicted realised span | predicted shrinkage |")
W("|---|---|---|---|---|---|")
res = {}
for name, T in CONS:
    med = {g: float(np.median(C[T[g]])) for g in ("hard", "mid", "easy")}
    pm = {g: float(np.mean(POST[T[g]])) for g in ("hard", "mid", "easy")}
    span = med["easy"] / med["hard"] if med["hard"] > 0 else float("inf")
    pspan = pm["easy"] / pm["hard"]
    res[name] = {"n_per_tier": int(len(T["hard"])), "median": med,
                 "post_mean": pm, "screening_span": span, "predicted_span": pspan}
    sh = "n/a" if not np.isfinite(span) else f"{100 * (1 - pspan / span):+.1f} %"
    W(f"| {name} | {len(T['hard'])} | "
      f"{med['hard']:.1f} / {med['mid']:.1f} / {med['easy']:.1f} | "
      f"**{'inf' if not np.isfinite(span) else f'{span:.2f}x'}** | "
      f"{pspan:.2f}x | {sh} |")
W("")
W("The screening span uses MEDIAN correct-rollout counts, as A34.2 specifies. "
  "Where the hard tier's median is 0 the ratio is undefined and printed as "
  "`inf`; the EB column is computed on mean posterior true rates and stays "
  "finite, which is why both are shown.\n")

# ---- the check the gate did NOT contain (A34.2 tested span only) -----------
W("## Matched-K feasibility — the check A34.2 does not contain\n")
W("The gate tests SPAN. The design also needs every cohort prompt to supply "
  "**K correct AND K incorrect** rollouts at K = 4, 6, 8, 11 (protocol item 1). "
  "A tier can pass the span test by having no correct rollouts at all, which "
  "maximises the span and destroys the design. Expected usable prompts at "
  "N=128, from the EB posterior:\n")
W("| construction, hard tier | n | K=4 | K=6 | K=8 | K=11 |")
W("|---|---|---|---|---|---|")
feas = {}
for name, T in CONS:
    idx = T["hard"]
    row, p = [], POST[idx]
    for K in (4, 6, 8, 11):
        pr = np.clip(binom.sf(K - 1, 128, p) - binom.sf(128 - K, 128, p), 0, 1)
        row.append(float(pr.sum()))
    feas[name] = row
    W(f"| {name} | {len(idx)} | " + " | ".join(f"**{x:.1f}**" for x in row) + " |")
W("")

# ---- extraction failure, and whether it covaries with the tier variable -----
IV = np.array([inv_q[q] for q in qids]) / N
r_iv = float(np.corrcoef(IV, P)[0, 1])
# 0.5B reference recomputed here, NOT quoted (A29). The 0.50 % figure that
# appeared in an earlier draft of this table was [invalid] among rollouts the
# verifier had scored CORRECT -- a different denominator entirely.
QF = ["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
      "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"]
qper, qinv, qtot = defaultdict(list), defaultdict(int), 0
for f in QF:
    for r in json.load(open(os.path.join(RES, f))):
        v = r.get("verification", {}) or {}
        x = int(r["question_id"]); qtot += 1
        bad = v.get("extracted_answer") == "[invalid]"
        qper[x].append(int(v.get("score", 0)) == 1 and not bad)
        qinv[x] += bad
qq = sorted(qper); qn = len(qper[qq[0]])
QC = np.array([sum(qper[x]) for x in qq]) / qn
QIV = np.array([qinv[x] for x in qq]) / qn
q_share, q_corr = float(np.mean(QIV)), float(np.corrcoef(QIV, QC)[0, 1])

W("## Extraction failure, and why it is a confound here\n")
W("Both columns are the SAME statistic on the SAME denominator: the share of "
  "ALL rollouts whose answer could not be extracted.\n")
W("| quantity | this run (Llama-3.2-3B, N=32) | Qwen-0.5B reference (N=128 pooled) |")
W("|---|---|---|")
W(f"| `[invalid]` share of all rollouts | **{100 * invalid / tot:.2f} %** | **{100 * q_share:.2f} %** |")
W(f"| prompts with >50 % of rollouts invalid | {int((IV > 0.5).sum())} / {len(qids)} | {int((QIV > 0.5).sum())} / {len(qq)} |")
W(f"| corr(invalid rate, guarded pass rate) | **{r_iv:+.3f}** | **{q_corr:+.3f}** |")
W("")
W("**The artefact is not peculiar to Llama.** It is present in the 0.5B model "
  "this entire paper is built on, and the difficulty correlation there is "
  "slightly STRONGER. Llama's rate is higher in level "
  f"({100 * invalid / tot:.2f} % against {100 * q_share:.2f} %) but the "
  "structure is the same in both.")
W("")
W("The rate is **not uniform across difficulty**: it correlates negatively with "
  "pass rate, so extraction fails most on exactly the hard tier the confound "
  "test is built around. Correct-rollout count would then differ across tiers "
  "partly for FORMATTING reasons, which is the variable under test being "
  "contaminated by a measurement artefact.\n")

# ---- verdict ----------------------------------------------------------------
PRIMARY = CONS[0][0]
r = res[PRIMARY]
scr, pred = r["screening_span"], r["predicted_span"]
W("## Verdict against the registered rules\n")
W(f"Primary construction: **{PRIMARY}**, {r['n_per_tier']} prompts per tier.\n")
W(f"* A34.2 gate (screening span >= 4x): "
  f"**{'PASS' if scr >= 4 else 'FAIL'}** — {('inf' if not np.isfinite(scr) else f'{scr:.2f}x')}")
W(f"* A35.5 consult trigger (screening span < 6x stops the run): "
  f"**{'NOT TRIGGERED' if scr >= 6 else 'TRIGGERED'}**")
W(f"* A35.4 prediction, recorded BEFORE any main generation: the realised span "
  f"from an independent N=128 pass will be **{pred:.2f}x**\n")
W("**This prediction is recorded now and scored later (A35.3). It is not "
  "revised after the main pass.**\n")

json.dump({"n_prompts": len(qids), "invalid_share": invalid / tot,
           "feasibility_K": feas, "corr_invalid_passrate": r_iv,
           "qwen_invalid_share": q_share, "qwen_corr": q_corr,
           "zero_correct": int((C == 0).sum()),
           "beta": [float(A_), float(B_)], "constructions": res,
           "primary": PRIMARY, "screening_span": float(scr),
           "predicted_span": float(pred)},
          open(os.path.join(EXP, "analysis", "gate_report.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
