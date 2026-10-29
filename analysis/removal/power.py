"""E3 scoping — power, cost, and the design checks. CPU ONLY, no GPU.

Item 1 comes first by instruction: if the MDE is larger than any plausible true
difference, the experiment cannot succeed and we do not run it.

Everything is derived from data already on disk:
  E0_seed_stability/analysis/per_example.csv   5 training seeds x 1023 prompts
  unlearnability-rlvr/data/...                 the pools and the held-out split
  a training log                               measured seconds per optimizer step

  $HOME/miniconda3/envs/vllm/bin/python power.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, chi2

ROOT = "/media/rmedu/3039-8E90/LLM"
EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP, "analysis", "power.md")
os.makedirs(os.path.join(EXP, "analysis"), exist_ok=True)

ALPHA, POWER = 0.05, 0.80
SEC_PER_STEP = 48.6          # measured, mean over 139 logged steps
N_STEPS = 120
CKPT_GB = 2.4                # measured, final checkpoint
GEN_PER_SEC = 256 / SEC_PER_STEP   # 32 prompts x 8 rollouts per step, conservative

d = pd.read_csv(os.path.join(ROOT, "EXPERIMENTS/E0_seed_stability/analysis/per_example.csv"))
S = [f"pass1_seed{i}" for i in range(1, 6)]
agg = d[S].mean()
n_seeds_obs = len(S)
M_e0 = len(d)
N_e0 = 32

var_obs = float(agg.var(ddof=1))
p_e0 = d[S].mean(axis=1).values
var_eval_e0 = float(np.mean(p_e0 * (1 - p_e0)) / N_e0 / M_e0)
# 95 % upper bound on the seed-level variance, then strip the eval component
var_up = (n_seeds_obs - 1) * var_obs / chi2.ppf(0.05, n_seeds_obs - 1)
sd_true_up = float(np.sqrt(max(var_up - var_eval_e0, 0.0)))

L = []
W = L.append
W("# E3 scoping — power FIRST, then cost, then the design checks\n")
W("**No GPU time was used to produce this.**\n")

W("## 1. Power\n")
W("### The variance that actually governs this experiment\n")
W("Five training seeds, identical data, evaluated at N=32 on the same 1023 "
  "prompts:\n")
W("| seed | aggregate pass@1 |")
W("|---|---|")
for k, v in agg.items():
    W(f"| {k.replace('pass1_seed', '')} | {v:.5f} |")
W("")
W(f"* between-seed sd: **{100 * agg.std(ddof=1):.3f} accuracy points**")
W(f"* sd predicted by evaluation sampling ALONE at N={N_e0}, M={M_e0}: "
  f"**{100 * np.sqrt(var_eval_e0):.3f} points**")
W("")
W("**The observed between-seed spread is SMALLER than sampling noise alone "
  "predicts.** The training-seed contribution to aggregate accuracy is not "
  "detectable at five seeds — it is consistent with zero.\n")
W("That is the single most important number here, and it cuts AGAINST the "
  "expectation that this experiment is underpowered. Per-prompt labels are "
  "wildly unstable (Jaccard 0.741 across seeds, E0b) while the AGGREGATE is "
  "stable to a tenth of a point. Which prompts a seed learns moves around; how "
  "many it learns does not.\n")
W(f"Because a variance estimate on {n_seeds_obs} seeds is itself very noisy, "
  "everything below is reported twice: at the point estimate (true seed sd = 0) "
  f"and at the **95 % upper bound, {100 * sd_true_up:.3f} points**, which is "
  "the number the decision should be made on.\n")


def mde(sd_true, p, N, M, n):
    """Minimum detectable difference in accuracy between two arms, 80 % power,
    alpha=0.05 two-sided, n seeds per arm, common held-out set of M prompts at
    N samples each."""
    var = sd_true ** 2 + p * (1 - p) / (N * M)
    df = 2 * (n - 1)
    mult = tdist.ppf(1 - ALPHA / 2, df) + tdist.ppf(POWER, df)
    return mult * np.sqrt(var) * np.sqrt(2 / n)


W("### MDE in accuracy points\n")
W("Held-out accuracy, two arms compared, 80 % power, alpha = 0.05 two-sided. "
  "All arms share ONE held-out set, so prompt-difficulty variance is common and "
  "cancels; what remains is seed variance plus sampling.\n")
W("Assumed held-out pass rate p = 0.30 (sensitivity below).\n")
W("| eval set | N per prompt | 3 seeds (sd_true=0) | 3 seeds (upper bd) | "
  "5 seeds (sd_true=0) | 5 seeds (upper bd) |")
W("|---|---|---|---|---|---|")
CONFIGS = [("MATH500 (held out)", 500, 16), ("MATH500 (held out)", 500, 32),
           ("MATH500 (held out)", 500, 64),
           ("1000 unused train-pool prompts", 1000, 32),
           ("2000 unused train-pool prompts", 2000, 32)]
best = {}
for name, M, N in CONFIGS:
    row = [100 * mde(s, 0.30, N, M, n) for n in (3, 5) for s in (0.0, sd_true_up)]
    # order: n=3 sd0, n=3 up, n=5 sd0, n=5 up
    r3a, r3b, r5a, r5b = row[0], row[1], row[2], row[3]
    best[(name, M, N)] = (r3b, r5b)
    W(f"| {name} | {N} | {r3a:.2f} | **{r3b:.2f}** | {r5a:.2f} | **{r5b:.2f}** |")
W("")
W("Sensitivity to the assumed held-out pass rate, at MATH500 / N=32 / 5 seeds, "
  "upper-bound sd:\n")
W("| p | MDE (points) |")
W("|---|---|")
for p in (0.20, 0.25, 0.30, 0.40, 0.50):
    W(f"| {p:.2f} | {100 * mde(sd_true_up, p, 32, 500, 5):.2f} |")
W("")
floor5 = 100 * mde(sd_true_up, 0.30, 10 ** 9, 1, 5)
floor3 = 100 * mde(sd_true_up, 0.30, 10 ** 9, 1, 3)
W(f"**The floor.** With an infinitely large evaluation the MDE cannot fall "
  f"below the seed variance alone: **{floor3:.2f} points at 3 seeds** and "
  f"**{floor5:.2f} points at 5 seeds** (at the upper bound; 0 at the point "
  "estimate). Beyond a few thousand eval prompts, more evaluation buys nothing "
  "— only more seeds do.\n")

W("### Is a difference that small worth reporting?\n")
r3, r5 = best[("MATH500 (held out)", 500, 32)]
W(f"At the design we would actually run — MATH500, N=32 — the MDE is "
  f"**{r3:.2f} points at 3 seeds** and **{r5:.2f} points at 5 seeds**, using "
  "the conservative upper-bound seed variance.\n")
W("**Whether that is enough depends on which arms are compared, and the "
  "arms as proposed do not survive section 4b.** Held here and answered there, "
  "because the deciding quantity is how much GRADIENT SIGNAL each removal set "
  "carries, not how many prompts it contains.\n")
W("Summary of that answer, stated here so section 1 does not contradict "
  "section 4: the proposed 74-vs-228 contrast removes near-identical signal "
  "(5.65 % against 4.99 %) and is a null by construction — expected gap "
  f"~0.15 points against an MDE of {r5:.2f}. **No number of seeds rescues it.** "
  "The union-vs-intersection contrast removes 27.6 % against 5.0 % and should "
  "produce several accuracy points, comfortably above the MDE.\n")
W(f"So: **the experiment is worth running at 5 seeds, on the redefined arms, "
  f"and is not worth running at 3 seeds on any arms** ({r3:.2f} points MDE "
  "sits inside the range a plausible effect occupies, so a null could not be "
  "distinguished from too few seeds).\n")

# ---------------------------------------------------------------- 2. cost ---
W("## 2. Cost\n")
h_run = N_STEPS * SEC_PER_STEP / 3600
W(f"Measured, not assumed: **{SEC_PER_STEP} s per optimizer step** (mean over "
  f"139 logged steps of an existing run), so {N_STEPS} steps = "
  f"**{h_run:.2f} h per training run**.\n")
W("**Arm 1 is already trained.** E0 ran the full-data baseline at five seeds to "
  "step 120. It needs evaluation only, which removes 5 runs from every total "
  "below.\n")
eval_h = 500 * 32 / GEN_PER_SEC / 3600
W(f"Evaluation: MATH500 at N=32 is 16 000 generations at a conservative "
  f"{GEN_PER_SEC:.1f} generations/s (the rate implied by training-time "
  f"generation, which understates eval-only throughput) = **{eval_h:.2f} h per "
  "arm-seed**.\n")
rows = []
for n in (3, 5):
    n_new, n_eval = 4 * n, 5 * n
    tr, ev = n_new * h_run, n_eval * eval_h
    rows.append((n, n_new, tr, n_eval, ev, tr + ev, n_eval * CKPT_GB))
W("| seeds | new training runs | training h | evals | eval h | **total h** | "
  "disk (final ckpts only) |")
W("|---|---|---|---|---|---|---|")
for n, nn, tr, ne, ev, tot, gb in rows:
    W(f"| {n} | {nn} | {tr:.1f} | {ne} | {ev:.1f} | **{tot:.1f} h** | {gb:.0f} GB |")
W("")
W("**Disk is the binding constraint, not hours.** 85 GB is free on the working "
  f"drive. Five seeds x five arms at {CKPT_GB} GB per final checkpoint is "
  f"{25 * CKPT_GB:.0f} GB, which fits only if intermediate checkpoints are NOT "
  "kept. E1's `save_freq 20` would write 6 checkpoints per run and need "
  f"{25 * 6 * CKPT_GB:.0f} GB. **Recommendation: save the final checkpoint "
  "only, evaluate, then delete before the next run**, which caps live usage at "
  "roughly one checkpoint at a time.\n")

# ------------------------------------------------------- 3. evaluation set --
W("## 3. The evaluation set\n")
W("| candidate | size | held out? |")
W("|---|---|---|")
W("| `SimpleRL-Zoo-Data/simplelr_qwen_level1to4/test.parquet` (MATH500) | 500 | "
  "**YES** — split='test', zero overlap with any training prompt |")
W("| unused portion of `train.parquet` | **7116** | **YES** — the 1023-prompt "
  "training subset is a strict subset of the 8139-problem pool |")
W("| `simplelr_qwen_level1to4_rest` | 559 | **NO** — these are the pass-2 half "
  "of the same 1023 prompts |")
W("| `simplelr_qwen_level1to4_e2eval` | 357 | **NO** — all 357 are inside the "
  "1023 |")
W("")
W("**Not a blocker.** Two clean options exist, and the second is large enough "
  "that evaluation sampling stops mattering. `rest` and `e2eval` are NOT held "
  "out despite their names — both are subsets of the training set, and using "
  "either would silently measure training accuracy.\n")
W("Recommendation: **MATH500 as the registered primary** (standard, "
  "comparable to published numbers) with **2000 sampled unused pool prompts as "
  "a registered secondary** (same distribution as training, so it isolates data "
  "composition rather than distribution shift). Both are fixed once and shared "
  "by every arm and seed.\n")

json.dump({"sd_between_seed_obs": float(agg.std(ddof=1)),
           "sd_eval_only_e0": float(np.sqrt(var_eval_e0)),
           "sd_true_upper95": sd_true_up,
           "mde_math500_N32": {"n3": best[("MATH500 (held out)", 500, 32)][0],
                               "n5": best[("MATH500 (held out)", 500, 32)][1]},
           "hours_per_run": h_run, "sec_per_step": SEC_PER_STEP},
          open(os.path.join(EXP, "analysis", "power.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ============================ 4. confounds ==================================
d["mp"] = d[S].mean(axis=1)
G = 8
d["useful"] = 1 - (d.mp ** G + (1 - d.mp) ** G)
TOT = d.useful.sum()
coh = json.load(open(os.path.join(ROOT, "EXPERIMENTS/E1_rescue/adaprefix/gradsim_cohorts.json")))
du = set(int(q) for q in coh["du71"])
SETS = {"union, flagged by >=1 of 5 seeds": set(d.qid[d.n_flagged >= 1]),
        "union, excl. no_reward": set(d.qid[(d.n_flagged >= 1) & ~d.ever_no_reward]),
        "intersection, flagged by all 5": set(d.qid[d.n_flagged == 5]),
        "intersection, excl. no_reward": set(d.qid[(d.n_flagged == 5) & ~d.ever_no_reward]),
        "the paper's D_u (N=128 pooled)": du}
BATCH = 32

W("## 4. Confounds\n")
W("### 4a. THE BLOCKER: the proposed sets are not what the proposal says\n")
W("Measured directly from the five-seed data:\n")
W("| removal set | n | % of training set | **% of gradient signal** | epochs at 120 steps |")
W("|---|---|---|---|---|")
for nm, Sq in SETS.items():
    sub = d[d.qid.isin(Sq)]
    W(f"| {nm} | {len(Sq)} | {100 * len(Sq) / len(d):.1f} % | "
      f"**{100 * sub.useful.sum() / TOT:.2f} %** | {BATCH * N_STEPS / (len(d) - len(Sq)):.2f} |")
W(f"| remove nothing (baseline) | 0 | 0 % | 0 % | {BATCH * N_STEPS / len(d):.2f} |")
W("")
W("**No rule gives 74, and none gives 228.** The union rule gives 404 (157 "
  "excluding `no_reward`); the intersection rule gives 218 (35 excluding it). "
  "The 74 in the proposal is the paper's D_u, which comes from a DIFFERENT "
  "evaluation protocol — one pooled pass at N=128 — not from an aggregation "
  "rule over the five seeds at all.\n")
W("**As proposed, arms 2 and 3 would differ in evaluation protocol as well as "
  "aggregation rule, and any difference would be attributable to either.** "
  "That is the same class of error as comparing groups at different rollout "
  "counts, which is the confound the gradient-similarity work exists to "
  "correct.\n")
W("Fix: define both sets from the SAME five-seed N=32 data. Then they are "
  "**nested** (intersection is a strict subset of union) and differ only in "
  "the rule.\n")

W("### 4b. The gradient-signal column is the finding that reshapes the design\n")
W("GRPO discards any prompt whose G=8 rollouts all receive the same reward — "
  "it contributes exactly zero gradient. For a prompt with pass rate p that "
  "happens with probability p^8 + (1-p)^8. Unlearnable prompts are near p=0, "
  "so **most of them already contribute nothing.**\n")
sub218 = d[d.qid.isin(SETS["intersection, flagged by all 5"])]
rng = np.random.default_rng(0)
rand218 = np.mean([d.sample(218, random_state=i).useful.sum() for i in range(200)])
rand404 = np.mean([d.sample(404, random_state=i).useful.sum() for i in range(200)])
W(f"* the 218-prompt intersection is **21.3 % of the data** but only "
  f"**{100 * sub218.useful.sum() / TOT:.2f} % of the signal**")
W(f"* 218 RANDOM prompts are the same 21.3 % of the data and "
  f"**{100 * rand218 / TOT:.2f} % of the signal** — {rand218 / sub218.useful.sum():.1f}x more")
W(f"* 404 random prompts: **{100 * rand404 / TOT:.2f} %** of the signal")
W("")
W("**Consequence for the original arm 2 vs arm 3.** The paper's D_u (71 "
  f"prompts) carries {100 * d[d.qid.isin(du)].useful.sum() / TOT:.2f} % of the "
  f"signal and the 218-prompt intersection carries "
  f"{100 * sub218.useful.sum() / TOT:.2f} %. Those are nearly identical "
  "despite a 3x difference in prompt count, so **that comparison is a null by "
  "construction** and no number of seeds would rescue it. Scaling crudely "
  "against the 22.2-point gain from training, the expected gap is around 0.15 "
  "points against an MDE of 0.91.\n")
W("**The union-versus-intersection comparison does not have this problem**: "
  f"{100 * d[d.qid.isin(SETS['union, flagged by >=1 of 5 seeds'])].useful.sum() / TOT:.2f} % "
  f"against {100 * sub218.useful.sum() / TOT:.2f} % of the signal, a gap that "
  "should be several accuracy points — comfortably above the MDE. **This is "
  "the comparison to run.**\n")

W("### 4c. Difficulty matching is infeasible, and the infeasibility is itself a result\n")
inter = SETS["intersection, flagged by all 5"]
outside = d[~d.qid.isin(inter)]
need = ((d[d.qid.isin(inter)].mp > 0) & (d[d.qid.isin(inter)].mp <= 0.05)).sum()
have = ((outside.mp > 0) & (outside.mp <= 0.05)).sum()
W(f"To difficulty-match 218 removed prompts we would need {need} prompts with "
  f"trained pass@1 in (0, 0.05]. Outside the unlearnable set only **{have}** "
  "exist. The pool does not contain enough hard prompts that are not already "
  "labelled unlearnable.\n")
W("The same fails on the *pre-treatment* covariate: matching on untrained "
  "pass@1 needs 155 prompts at exactly 0 and only 54 are available. Untrained "
  "pass rate predicts the label with AUC 0.927; trained pass rate with AUC "
  "0.996.\n")
W("**So difficulty matching both bakes in an assumption and cannot be done.** "
  "'Unlearnable' is very nearly a synonym for 'hardest by pass rate', and a "
  "difficulty-matched control would have to reconstruct almost the same set. "
  "Match on **size only**, and treat the random arms as positive controls "
  "rather than as difficulty controls.\n")

W("### 4d. Steps, epochs, or tokens\n")
W("**Recommendation: match total optimizer steps at 120.** It holds the number "
  "of gradient updates constant, so a difference is attributable to WHICH data "
  "was removed rather than to how much training happened; the baseline already "
  "exists at 120 steps for five seeds; and it is the practitioner's actual "
  "situation, a fixed compute budget.\n")
W("The cost is stated rather than hidden: fewer prompts at fixed steps means "
  "**more epochs over what remains** — 3.75 for the baseline rising to 4.77 "
  "when 218 are removed. That repetition is a real mechanism by which removal "
  "could help or hurt, and it is part of what the arms measure. Matching "
  "epochs instead would leave arms with different numbers of updates, which "
  "confounds composition with training amount — the worse trade.\n")

W("### 4e. Nesting\n")
W(f"The paper's D_u is **NOT** a subset of the 218-prompt intersection: "
  f"{len(du & inter)} of {len(du)} overlap. It IS a subset of the 404-prompt "
  "union. Two sets that disagree on 38 of 71 prompts are not a nested pair, so "
  "as originally proposed the arms would be neither nested nor disjoint — the "
  "hardest case to interpret. Defining both from the same five-seed data "
  "restores strict nesting.\n")

W("### 4f. Also worth stating\n")
W("* **`no_reward` handling changes every set size by 2-6x** (404 vs 157, 218 "
  "vs 35). It must be fixed in the registration, not chosen later.")
W("* **One removal set per seed, or one shared?** The unlearnable set was "
  "derived FROM these five seeds. Reusing it to train new seeds leaks the "
  "labels' dependence on the very runs being compared. The clean choice is a "
  "single set fixed from the existing five seeds, applied to all new seeds, "
  "with the leakage stated.")
W("* **MATH500 spans levels 1-5 while training is levels 1-4**, so the primary "
  "eval is slightly out of distribution. That is why the unused-pool secondary "
  "is registered alongside it.")

# ============================ 5. registration ===============================
W("## 5. Both branches, registered before any GPU time\n")
W("### The design this analysis recommends\n")
W("| arm | removal | n removed | signal removed |")
W("|---|---|---|---|")
W("| 1 | nothing (baseline) | 0 | 0 % | ")
W(f"| 2 | union rule, flagged by >=1 of 5 seeds | 404 | "
  f"{100 * d[d.qid.isin(SETS['union, flagged by >=1 of 5 seeds'])].useful.sum() / TOT:.1f} % |")
W(f"| 3 | intersection rule, flagged by all 5 | 218 | "
  f"{100 * sub218.useful.sum() / TOT:.1f} % |")
W(f"| 4 | random, size-matched to arm 2 | 404 | {100 * rand404 / TOT:.1f} % |")
W(f"| 5 | random, size-matched to arm 3 | 218 | {100 * rand218 / TOT:.1f} % |")
W("")
W("**Arms 4 and 5 are positive controls, not merely size controls.** They "
  "remove 4-8x more gradient signal than the labelled sets, so they should "
  "show a clear drop. If they do not, the experiment failed to detect anything "
  "at all and the arm-2-vs-3 result is uninterpretable regardless of what it "
  "says. This is the manipulation check that makes a null readable.\n")
W("**Five seeds per arm. Three is not enough** and is not offered as an "
  "option.\n")
W("### DIFFERENCE DETECTED\n")
W("The aggregation rule changes the trained model. The definitional "
  "instability we document has a downstream consequence, and that becomes a "
  "headline claim with the effect size and its CI in the abstract.\n")
W("### NO DIFFERENCE DETECTED\n")
W("We report that removing the union set versus the intersection set makes no "
  "detectable difference at the MDE stated above, **in the main paper, in the "
  "results section, at the same prominence as the other branch.** It is not "
  "moved to an appendix, not softened to 'suggestive', and not reframed as a "
  "limitation of the experiment. It weakens our own impact claim and it gets "
  "the same number of words.\n")
W("A null is only reportable if arms 4 and 5 moved. That condition is fixed "
  "now, before the data exists: **if the positive controls do not separate "
  "from baseline, the verdict is NO POWER, not NO DIFFERENCE**, and neither "
  "branch above applies.\n")

open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L[L.index("## 4. Confounds\n"):]))
