"""A15: the interval on c_easy / c_D_u, computed three ways.

(a) is the procedure the paper currently prints. It resamples ONE subsample mean
per level, which propagates the spread of a single subsample instead of the
standard error of the level mean -- and omits prompt-sampling variance
altogether. Its signature defect: adding subsamples makes it WIDER.

(c) is registered (A15.3) as the primary interval BEFORE it was computed, and
goes in the paper whatever it says.

    (a) one subsample per level          -- superseded, reported with its defect
    (b) mean over subsamples per level   -- fixes half the problem, intermediate
    (c) joint over prompts AND subsamples -- PRIMARY

In (c) the prompt set is resampled ONCE per draw and reused at every level,
because the same fixed cohort is used at every level; resampling it per level
would destroy the across-level correlation the curve is fitted through.

  $HOME/miniconda3/envs/vllm/bin/python joint_bootstrap.py
"""
import json
import os

import numpy as np
from scipy.optimize import least_squares

EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP, "analysis", "joint_bootstrap.md")
GROUPS = ["du71", "learnable", "easy"]
LABEL = {"du71": "D_u", "learnable": "learnable", "easy": "easy"}
SEED = 20260808
NBOOT = 2000            # same as constrained_fit.py, so only the scheme differs
NO_POWER_WIDTH = 3.99   # A15.3, unchanged; also the width of [0.5x, 2.0x]

att = json.load(open(os.path.join(EXP, "analysis", "attenuation.json")))
Ks = sorted({int(k) for g in att for k in att[g]})


def entries(g, K):
    return [e for e in att[g][str(K)] if "per_prompt" in e]


fixed = {g: sorted(set.intersection(*[set(e["per_prompt"]) for K in Ks
                                      for e in entries(g, K)]), key=int)
         for g in GROUPS}
# per-prompt matrices: pp[g][K] is (n_sub, n_prompt)
pp = {g: {K: np.array([[e["per_prompt"][q] for q in fixed[g]]
                       for e in entries(g, K)]) for K in Ks} for g in GROUPS}
xs = np.array(Ks, float)


def fit_unc(ys):
    def r(p):
        return np.concatenate([p[2 * i] / (1 + p[2 * i + 1] / xs) - ys[i]
                               for i in range(3)])
    return least_squares(r, [0.5, 50] * 3, bounds=([0, 0] * 3, [1, 1e4] * 3)).x


def run(mode):
    r = np.random.default_rng(SEED)
    out = []
    for _ in range(NBOOT):
        ys = np.empty((3, len(Ks)))
        for gi, g in enumerate(GROUPS):
            # (c) only: one prompt resample per draw, shared across levels
            qi = (r.integers(0, len(fixed[g]), len(fixed[g]))
                  if mode == "joint" else None)
            for ki, K in enumerate(Ks):
                M = pp[g][K]                       # (n_sub, n_prompt)
                if mode == "one":
                    ys[gi, ki] = M[r.integers(0, M.shape[0])].mean()
                elif mode == "levelmean":
                    si = r.integers(0, M.shape[0], M.shape[0])
                    ys[gi, ki] = M[si].mean(axis=1).mean()
                else:
                    si = r.integers(0, M.shape[0], M.shape[0])
                    ys[gi, ki] = M[np.ix_(si, qi)].mean()
        p = fit_unc(ys)
        if p[0] > 1e-6:
            out.append(p[4] / p[0])
    a = np.array(out)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi), float(np.median(a)), len(a)


L = []
W = L.append
W("# A15 — the interval on c_easy / c_D_u, three procedures\n")
W(f"Fixed cohort " + ", ".join(f"{LABEL[g]} {len(fixed[g])}" for g in GROUPS)
  + f"; levels K = {Ks}; {NBOOT} resamples each, seed {SEED}. The three differ "
  "ONLY in what is resampled.\n")
W("| | procedure | 95 % interval | multiplicative width | median |")
W("|---|---|---|---|---|")
res = {}
for tag, mode, name in [
        ("a", "one", "one subsample per level — **superseded**"),
        ("b", "levelmean", "mean over subsamples per level — intermediate"),
        ("c", "joint", "**joint over prompts AND subsamples — PRIMARY**")]:
    lo, hi, med, n = run(mode)
    res[tag] = (lo, hi, med)
    W(f"| ({tag}) | {name} | [{lo:.2f}x, {hi:.2f}x] | **{hi / lo:.1f}x** | "
      f"{med:.2f}x |")
W("")

lo, hi, med = res["c"]
width = hi / lo
W("## The primary interval and what it decides\n")
W(f"**(c) = [{lo:.2f}x, {hi:.2f}x], width {width:.1f}x.** Registered as primary "
  "before it was computed (A15.3).\n")
W(f"NO POWER rule, unchanged: width > {NO_POWER_WIDTH}x. "
  + (f"**{width:.1f}x > {NO_POWER_WIDTH}x — the verdict of section 7 STANDS: "
     "the data cannot distinguish equal true similarity from a factor-of-2 "
     "difference, and 'no evidence of a difference' continues to mean no "
     "power rather than evidence of equality.**"
     if width > NO_POWER_WIDTH else
     f"**{width:.1f}x < {NO_POWER_WIDTH}x — the verdict of section 7 CHANGES.** "
     "The interval is now narrower than the factor-of-2 question it was built "
     "to answer, so the data DO constrain the true-similarity ratio, and the "
     "no-power wording must be rewritten rather than retained.") + "\n")
W("| test | result |")
W("|---|---|")
for lab, v in [("excludes equality (1.0x)", 1.0),
               ("excludes a 2x easy advantage", 2.0),
               ("excludes a 2x D_u advantage", 0.5)]:
    W(f"| {lab} | {'YES' if (v < lo or v > hi) else 'no'} |")
W("")

W("## What changed and why (a) is wrong, not merely conservative\n")
a_lo, a_hi, _ = res["a"]
b_lo, b_hi, _ = res["b"]
W(f"(a) resamples one subsample mean per level, so it propagates the spread of "
  "a SINGLE subsample where the level's information is the mean over "
  "subsamples. At K = 6, which has one subsample, it propagates zero "
  "variability. Its signature is that **adding subsamples makes it wider**, "
  "which no correct procedure does.\n")
W(f"(b) repairs that and gives [{b_lo:.2f}x, {b_hi:.2f}x], but every subsample "
  "at every level uses the same fixed cohort, so (b) still contains **no "
  "prompt-sampling variance at all** — the variance Figure 6's error bars "
  "display. (c) adds it by resampling prompts once per draw and reusing that "
  "set at every level, which preserves the across-level correlation the "
  "attenuation curve is fitted through.\n")
W(f"Direction of travel: {a_hi / a_lo:.1f}x -> {b_hi / b_lo:.1f}x -> "
  f"{width:.1f}x. "
  + ("Removing the inflation dominates: the interval narrows even after real "
     "prompt variance is added."
     if width < a_hi / a_lo else
     "Adding prompt variance dominates: the interval widens once the variance "
     "the published procedure omitted is included.") + "\n")

W("## Limitation, registered before the run (A15.4)\n")
W("The per-prompt value is a cosine against the **leave-one-out group mean**, "
  "so prompts are not independent draws and the prompt bootstrap is "
  "approximate, in the direction of understating uncertainty. It is still "
  "strictly better than omitting prompt variance altogether, which is what (a) "
  "does. Recorded rather than corrected.\n")
W("**(a) is superseded, not deleted.** It is reported wherever the corrected "
  "number appears, with its defect named, so the change is visible rather than "
  "silent.\n")

json.dump({"primary": {"lo": res["c"][0], "hi": res["c"][1],
                       "median": res["c"][2], "width": res["c"][1] / res["c"][0]},
           "superseded_a": {"lo": res["a"][0], "hi": res["a"][1]},
           "intermediate_b": {"lo": res["b"][0], "hi": res["b"][1]},
           "nboot": NBOOT, "seed": SEED},
          open(os.path.join(EXP, "analysis", "joint_bootstrap.json"), "w"),
          indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nwritten to {OUT}")
