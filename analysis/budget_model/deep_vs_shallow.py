"""A36 — is one deep pass more reproducible than several shallow ones, at equal
sampling budget? Rules fixed in A36.3 BEFORE this was run.

Two independent N=128 passes over 1023 prompts, same checkpoint. At a budget of
128 draws per prompt:

    DEEP    : one pass of 128, flag if pass rate <= tau
    SHALLOW : the SAME 128 split into 4 disjoint blocks of 32, flag if flagged
              in all four

Each rule is applied to pass A and pass B; the Jaccard between the two passes is
that rule's reproducibility. No GPU.

  $HOME/miniconda3/envs/vllm/bin/python deep_vs_shallow.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
OUT = os.path.join(EXP, "analysis", "deep_vs_shallow.md")
TAU = 0.1
NBOOT = 5000
SEED = 20260812
BLOCK = 32
NBLOCK = 4

PASS_A = ["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
          "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"]
PASS_B = ["qwen_0.5b_ckpt1_n128_rep2_simplelr_qwen_level1to4_sub1k@train_0.7_128_1024_None_1_qwen.json"]


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def load(files):
    per = defaultdict(list)
    for f in files:
        for r in json.load(open(os.path.join(RES, f))):
            per[int(r["question_id"])].append(guarded(r.get("verification", {}) or {}))
    return {q: np.array(v, dtype=bool) for q, v in per.items()}


A, B = load(PASS_A), load(PASS_B)
qids = sorted(set(A) & set(B))
assert all(len(A[q]) == 128 and len(B[q]) == 128 for q in qids), "not all prompts have 128"
print(f"prompts with both passes at N=128: {len(qids)}")


def deep(v):
    return v.mean() <= TAU


def shallow(v):
    # all four disjoint 32-blocks must flag; block k is rollouts [32k, 32k+32)
    return all(v[i * BLOCK:(i + 1) * BLOCK].mean() <= TAU for i in range(NBLOCK))


def jaccard(sa, sb):
    u = len(sa | sb)
    return len(sa & sb) / u if u else np.nan


nr = pd.read_csv(os.path.join(EXP, "..", "E0b_eval_noise", "analysis", "true_rates_all.csv"))
NOREWARD = set(nr.qid[nr.no_reward == 1])

L = []
W = L.append
W("# A36 — deep single pass vs shallow intersected passes, at equal budget\n")
W(f"Two independent N=128 passes, same checkpoint, {len(qids)} prompts. "
  f"tau = {TAU}. **Both rules spend exactly 128 draws per prompt.** No GPU.\n")

results = {}
for excl_label, excl in [("flagging only (no exclusion)", set()),
                         ("after the no_reward exclusion", NOREWARD)]:
    keep = [q for q in qids if q not in excl]
    flags = {}
    for rule, fn in (("deep", deep), ("shallow", shallow)):
        flags[rule] = ({q for q in keep if fn(A[q])}, {q for q in keep if fn(B[q])})

    # ---- the bootstrap, PAIRED by construction (A36.3) ----------------------
    # One index matrix is drawn per keep-set, at that set's own length, and the
    # SAME matrix is used for both rules. Bootstrap draw i therefore resamples
    # prompts once and both Jaccards are computed on that one resampled set, so
    # the difference distribution is paired and the prompt-level correlation
    # between the rules is retained.
    #
    # It is drawn per keep-set rather than once globally on purpose: reusing a
    # length-1023 index matrix under `% 759` would give the first 264 prompts
    # twice the sampling weight of the rest. That is a weighted bootstrap, not a
    # uniform one, and it is wrong even though it is paired.
    rng = np.random.default_rng(SEED)
    BI = rng.integers(0, len(keep), size=(NBOOT, len(keep)))
    memb = {r: (np.array([q in flags[r][0] for q in keep]),
                np.array([q in flags[r][1] for q in keep])) for r in flags}
    jb = {r: np.empty(NBOOT) for r in flags}
    for i, row in enumerate(BI):                 # one resample -> both rules
        for r in ("deep", "shallow"):
            a, b = memb[r][0][row], memb[r][1][row]
            u = (a | b).sum()
            jb[r][i] = (a & b).sum() / u if u else np.nan

    W(f"## {excl_label}  (n = {len(keep)} prompts)\n")
    W("| rule | budget | flagged in A | flagged in B | shared | **Jaccard** | 95 % CI |")
    W("|---|---|---|---|---|---|---|")
    for rule in ("deep", "shallow"):
        sa, sb = flags[rule]
        lo, hi = np.nanpercentile(jb[rule], [2.5, 97.5])
        W(f"| {rule} | 128 | {len(sa)} | {len(sb)} | {len(sa & sb)} | "
          f"**{jaccard(sa, sb):.4f}** | [{lo:.4f}, {hi:.4f}] |")
    W("")
    d = jb["deep"] - jb["shallow"]
    dobs = jaccard(*flags["deep"]) - jaccard(*flags["shallow"])
    dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
    clears = dlo > 0 or dhi < 0
    W(f"**Difference (deep - shallow): {dobs:+.4f}, paired bootstrap 95 % CI "
      f"[{dlo:+.4f}, {dhi:+.4f}] — clears zero: "
      f"{'**YES**' if clears else '**NO**'}.**\n")
    results[excl_label] = {"deep": jaccard(*flags["deep"]),
                           "shallow": jaccard(*flags["shallow"]),
                           "ci": {r: [float(x) for x in
                                      np.nanpercentile(jb[r], [2.5, 97.5])] for r in jb},
                           "diff": dobs, "lo": dlo, "hi": dhi, "clears": bool(clears),
                           "n": len(keep),
                           "sizes": {r: [len(flags[r][0]), len(flags[r][1])]
                                     for r in flags}}
    W("")

W("## Verdict against the registered branches\n")
prim = results["flagging only (no exclusion)"]
if prim["clears"] and prim["diff"] > 0:
    W("**BRANCH A (A36.4).** The single deep pass is more reproducible than the "
      "shallow intersected rule at equal budget, and the difference clears its "
      f"interval: **{prim['deep']:.4f} against {prim['shallow']:.4f}**, "
      f"margin {prim['diff']:+.4f} with CI [{prim['lo']:+.4f}, {prim['hi']:+.4f}]. "
      "We propose the deep pass as a replacement rule and report this margin "
      "with the budget at which it was measured.\n")
elif prim["clears"] and prim["diff"] < 0:
    W("**BRANCH B (A36.5), in its strongest form.** The shallow intersected rule "
      "is MORE reproducible, and the difference clears its interval. Our own "
      "recommendation is contradicted, not merely unsupported. The Discussion "
      "sentence is removed and the contradiction reported in the main text.\n")
else:
    W("**BRANCH B (A36.5).** The difference does not clear its interval, so the "
      "deep pass is not shown to be more reproducible. **Our own recommendation "
      "is unsupported by our own data**, reported in the main text, and the "
      "Discussion sentence recommending it is REMOVED — not softened, not moved "
      "to future work.\n")

W("## Limitation, in the same place as the result (A36.2)\n")
_ns = json.load(open(os.path.join(EXP, "analysis", "noise_share_recompute.json")))
SHARE32 = _ns["guarded (score == 1 and extracted_answer != '[invalid]')"]["share"]["32"]
W("Both passes come from ONE checkpoint, so this measures the aggregation rule "
  "against **evaluation noise only**. Chen's rule aggregates over TRAINING "
  "seeds, and a true repeat of it needs five more training runs (~12.5 h) which "
  f"we did not buy. E0b measured that **{100 * SHARE32:.1f} %** of the apparent "
  "cross-seed disagreement at N=32, under the same verifier guard used "
  "throughout this document, is reproduced with no training difference at all "
  "(`analysis/noise_share_recompute.md`), so evaluation noise is most of what "
  "the five-seed rule is fighting — but not all of it.\n")
W("Where the comparison is unequal it favours the SHALLOW rule: Chen spends 160 "
  "draws (5 x 32) against the deep pass's 128, while here both spend 128 and "
  "the shallow rule gets 4 blocks rather than 5.\n")

json.dump(results, open(os.path.join(EXP, "analysis", "deep_vs_shallow.json"), "w"),
          indent=2, default=float)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------------------------------------------------------------------
# A36.7 -- machine-generated macros. Nothing here is hand-typed; every value is
# the same object the tables above printed. \input this and a rerun updates the
# paper instead of leaving a stale number behind.
# ---------------------------------------------------------------------------
PRIM = "flagging only (no exclusion)"
SEC = "after the no_reward exclusion"
T = ["% GENERATED BY EXPERIMENTS/E1_rescue/deep_vs_shallow.py -- DO NOT EDIT BY HAND.",
     "% Rerun that script to regenerate. A36.7.",
     f"% tau = {TAU}, bootstrap resamples = {NBOOT}, seed = {SEED}.",
     ""]
T.append(r"% Formatting contract (A36.9): Jaccards and their bounds are four")
T.append(r"% decimals with NO sign; all six difference macros carry an explicit")
T.append(r"% leading + or - because they print inside a bracketed interval and")
T.append(r"% the surrounding text adds no sign; counts are bare integers; no")
T.append(r"% macro contains a unit, a trailing percent sign or a dollar sign.")
T.append("")
T.append(r"\newcommand{\dvsBudget}{%d}" % (BLOCK * NBLOCK))
T.append(r"\newcommand{\dvsBlocks}{%d}" % NBLOCK)
T.append(r"\newcommand{\dvsBlockSize}{%d}" % BLOCK)
T.append(r"\newcommand{\dvsTau}{%.2f}" % TAU)
T.append(r"\newcommand{\dvsNboot}{%d}" % NBOOT)
T.append("")
for tag, key in (("Primary", PRIM), ("Secondary", SEC)):
    r = results[key]
    T.append(f"% --- {key} ---")
    T.append(r"\newcommand{\dvsN%s}{%d}" % (tag, r["n"]))
    for rule, rt in (("deep", "Deep"), ("shallow", "Shallow")):
        T.append(r"\newcommand{\dvs%s%s}{%.4f}" % (rt, tag, r[rule]))
        T.append(r"\newcommand{\dvs%s%sLo}{%.4f}" % (rt, tag, r["ci"][rule][0]))
        T.append(r"\newcommand{\dvs%s%sHi}{%.4f}" % (rt, tag, r["ci"][rule][1]))
        T.append(r"\newcommand{\dvs%s%sFlagA}{%d}" % (rt, tag, r["sizes"][rule][0]))
        T.append(r"\newcommand{\dvs%s%sFlagB}{%d}" % (rt, tag, r["sizes"][rule][1]))
    T.append(r"\newcommand{\dvsDiff%s}{%+.4f}" % (tag, r["diff"]))
    T.append(r"\newcommand{\dvsDiff%sLo}{%+.4f}" % (tag, r["lo"]))
    T.append(r"\newcommand{\dvsDiff%sHi}{%+.4f}" % (tag, r["hi"]))
    T.append(r"\newcommand{\dvsClears%s}{%s}" % (tag, "yes" if r["clears"] else "no"))
    T.append("")
T.append(r"\newcommand{\dvsNoiseShareThirtyTwo}{%.1f}" % (100 * SHARE32))
T.append(r"\newcommand{\dvsBranch}{%s}" % ("A" if (results[PRIM]["clears"]
                                                  and results[PRIM]["diff"] > 0) else "B"))
TEX = os.path.join(EXP, "analysis", "deep_vs_shallow_numbers.tex")
open(TEX, "w").write("\n".join(T) + "\n")
print(f"\nwrote {TEX}")
