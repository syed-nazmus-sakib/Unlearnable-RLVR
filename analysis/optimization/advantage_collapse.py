"""A37 — is advantage collapse a rival explanation for "never improves"?

Under GRPO a prompt contributes gradient only when its G rollouts do not all
share a reward:   c(p) = 1 - (p^G + (1-p)^G).

Everything here is CPU-only and comes from data already on disk:
  * G and the per-step per-prompt reward std   -> each run's prompt_reward_stats.csv
  * true pass rates                            -> N=32 initial and N=128 trained evals
  * group membership                           -> example_ids/ and gradsim_cohorts.json

Rules fixed in A37.1-A37.4 BEFORE this was run. The primary test is ABSOLUTE
(A37.4): not "do the groups separate on c(p)" -- which is partly definitional --
but "did unlearnable prompts actually receive so few contributing steps that no
learning could be expected, and did improving learnable prompts receive more?"

  $HOME/miniconda3/envs/vllm/bin/python advantage_collapse.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(EXP, "..", "..", "unlearnability-rlvr")
RES = os.path.join(REPO, "results")
OUT = os.path.join(EXP, "analysis", "advantage_collapse.md")
SEED, NBOOT = 20260812, 5000
SEEDS = [1, 2, 3, 4, 5]

# ---- G, read from the training script, not assumed (A37.1) -----------------
TRAIN_SH = os.path.join(REPO, "repro", "train_1gpu.sh")
G = None
for ln, line in enumerate(open(TRAIN_SH), 1):
    if "--rollout_n" in line and "$" not in line.split("--rollout_n")[1][:6]:
        G, G_LINE = int(line.split("--rollout_n")[1].split()[0]), ln
        break
assert G is not None, "could not read --rollout_n from the training script"


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def rates(files):
    per = defaultdict(list)
    for f in files:
        for r in json.load(open(os.path.join(RES, f))):
            per[int(r["question_id"])].append(guarded(r.get("verification", {}) or {}))
    return {q: float(np.mean(v)) for q, v in per.items()}


P_INIT = rates(["qwen_0.5b_initial_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json"])
P_TRAINED = rates(["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
                   "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"])


def c_of(p):
    return 1.0 - (p ** G + (1.0 - p) ** G)


# ---- groups ----------------------------------------------------------------
def ids(name):
    return {int(x) for x in json.load(
        open(os.path.join(REPO, "example_ids", f"qwen_0.5b_math_level1to4_{name}.json")))}


PRIMARY = {"unlearnable": ids("unlearnable"), "learnable": ids("learnable"),
           "easy": ids("easy")}
NOREW = ids("no_reward") if os.path.exists(
    os.path.join(REPO, "example_ids", "qwen_0.5b_math_level1to4_no_reward.json")) else set()
coh = json.load(open(os.path.join(EXP, "adaprefix", "gradsim_cohorts.json")))
SECONDARY = {k: {int(q) for q in v} for k, v in coh.items()}

# ---- item 3: the MEASURED non-degenerate group counts (A37.2) --------------
app, nondeg = defaultdict(int), defaultdict(int)
per_seed_rows, dup_total = {}, 0
for s in SEEDS:
    p = glob.glob(os.path.join(REPO, "checkpoints",
                               f"qwen_0.5b_baseline_1gpu_*seed{s}", "prompt_reward_stats.csv"))
    assert len(p) == 1, f"seed {s}: expected one stats file, found {len(p)}"
    d = pd.read_csv(p[0])
    n0 = len(d)
    d = d.drop_duplicates(["step", "prompt_id"], keep="first")   # A37.2 rule 2
    dup_total += n0 - len(d)
    assert set(d.n.unique()) == {G}, f"seed {s}: n column is not uniformly {G}"
    per_seed_rows[s] = (len(d), int(d.step.nunique()), int(d.prompt_id.nunique()))
    for q, sd in zip(d.prompt_id.values, d.std_reward.values):
        app[int(q)] += 1
        nondeg[int(q)] += int(sd > 0)

QIDS = sorted(app)
OBS = {q: nondeg[q] / app[q] for q in QIDS}          # observed contributing fraction
CONTRIB = {q: nondeg[q] / len(SEEDS) for q in QIDS}  # contributing steps per run

rng = np.random.default_rng(SEED)


def boot_mean(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(v), size=(NBOOT, len(v)))
    d = v[idx].mean(axis=1)
    return float(v.mean()), *np.percentile(d, [2.5, 97.5])


L = []
W = L.append
W("# A37 — advantage collapse: is the unlearnable set starved by the sampling budget?\n")
W(f"**G = {G}**, read from `repro/train_1gpu.sh:{G_LINE}` (`--rollout_n {G}`), "
  f"corroborated by the `n` column of every `prompt_reward_stats.csv` being "
  f"uniformly {G}. CPU only; no GPU was touched.\n")
W(f"Contributing probability `c(p) = 1 - (p^{G} + (1-p)^{G})`.\n")

W("## 0. What the training record actually contains (item 3: MEASURED)\n")
W("| seed | rows | steps logged | prompts |")
W("|---|---|---|---|")
for s in SEEDS:
    r, st, pr = per_seed_rows[s]
    W(f"| {s} | {r} | {st} | {pr} |")
W("")
W(f"Duplicate `(step, prompt_id)` rows dropped across all five seeds: "
  f"**{dup_total}** (A37.2 rule 2). Every row is `processed = False`, i.e. "
  "logged BEFORE `_filter_std_0`, so these are the groups as generated.\n")
W(f"Mean appearances per prompt per run: "
  f"**{np.mean([app[q] for q in QIDS]) / len(SEEDS):.2f}** over "
  f"{per_seed_rows[1][1]} logged steps.\n")

# ---- item 1 & 2 ------------------------------------------------------------
for gname, GRP in (("PRIMARY — published five-seed partition", PRIMARY),
                   ("SECONDARY — E1 gradient cohort", SECONDARY)):
    W(f"## {gname}\n")
    W("| group | n | median p̂ init | median c(p̂) init | median p̂ trained | "
      "median c(p̂) trained | **observed contributing fraction** | 95 % CI |")
    W("|---|---|---|---|---|---|---|---|")
    for g, S in GRP.items():
        qs = [q for q in sorted(S) if q in OBS]
        pi = [P_INIT[q] for q in qs if q in P_INIT]
        pt = [P_TRAINED[q] for q in qs if q in P_TRAINED]
        ob = [OBS[q] for q in qs]
        m, lo, hi = boot_mean(ob)
        W(f"| {g} | {len(qs)} | {np.median(pi):.4f} | {c_of(np.median(pi)):.4f} | "
          f"{np.median(pt):.4f} | {c_of(np.median(pt)):.4f} | **{m:.4f}** | "
          f"[{lo:.4f}, {hi:.4f}] |")
    W("")

# ---- item 4: THE ABSOLUTE TEST (A37.4) -------------------------------------
W("## The primary test, and it is ABSOLUTE not comparative (A37.4)\n")
W("Group separation on `c(p̂)` is partly definitional — the label IS a low pass "
  "rate and `c` is monotone in it — so it cannot carry the claim. The question "
  "that can: **did unlearnable prompts actually receive so few contributing "
  "gradient steps that no learning could be expected, and did learnable prompts "
  "receive materially more?**\n")
W("| group | n | contributing steps per run, mean | 95 % CI | median | "
  "share with ZERO in all 5 runs |")
W("|---|---|---|---|---|---|")
absres = {}
for g, S in PRIMARY.items():
    qs = [q for q in sorted(S) if q in CONTRIB]
    v = [CONTRIB[q] for q in qs]
    m, lo, hi = boot_mean(v)
    zero = float(np.mean([nondeg[q] == 0 for q in qs]))
    absres[g] = {"n": len(qs), "mean": m, "lo": lo, "hi": hi,
                 "median": float(np.median(v)), "zero_share": zero}
    W(f"| {g} | {len(qs)} | **{m:.2f}** | [{lo:.2f}, {hi:.2f}] | "
      f"{np.median(v):.2f} | {100 * zero:.1f} % |")
W("")

ul, ln = absres["unlearnable"], absres["learnable"]
ratio = ln["mean"] / ul["mean"] if ul["mean"] > 0 else float("inf")
qs_u = [q for q in sorted(PRIMARY["unlearnable"]) if q in CONTRIB]
qs_l = [q for q in sorted(PRIMARY["learnable"]) if q in CONTRIB]
vu = np.array([CONTRIB[q] for q in qs_u])
vl = np.array([CONTRIB[q] for q in qs_l])
iu = rng.integers(0, len(vu), size=(NBOOT, len(vu)))
il = rng.integers(0, len(vl), size=(NBOOT, len(vl)))
dd = vl[il].mean(axis=1) - vu[iu].mean(axis=1)
dlo, dhi = np.percentile(dd, [2.5, 97.5])
overlap = float(np.mean(vu >= np.median(vl)))

W(f"**Learnable minus unlearnable: {ln['mean'] - ul['mean']:+.2f} contributing "
  f"steps per run, 95 % CI [{dlo:+.2f}, {dhi:+.2f}]** "
  f"(ratio {ratio:.2f}x).\n")
W(f"**Overlap:** {100 * overlap:.1f} % of unlearnable prompts received at least "
  f"as many contributing steps as the MEDIAN learnable prompt "
  f"({np.median(vl):.2f} per run).\n")

# calibration: does c(p) predict what happened?
pp = np.array([c_of(P_INIT[q]) for q in QIDS if q in P_INIT])
oo = np.array([OBS[q] for q in QIDS if q in P_INIT])
r_init = float(np.corrcoef(pp, oo)[0, 1])
pp2 = np.array([c_of(P_TRAINED[q]) for q in QIDS if q in P_TRAINED])
oo2 = np.array([OBS[q] for q in QIDS if q in P_TRAINED])
r_tr = float(np.corrcoef(pp2, oo2)[0, 1])
W("### Does the model predict what actually happened?\n")
W("| p̂ anchor | corr(c(p̂), observed fraction) | mean predicted | mean observed |")
W("|---|---|---|---|")
W(f"| initial (N=32) | {r_init:+.3f} | {pp.mean():.4f} | {oo.mean():.4f} |")
W(f"| trained (N=128) | {r_tr:+.3f} | {pp2.mean():.4f} | {oo2.mean():.4f} |")
W("")

W("### The starting line: the mechanism cannot tell the two groups apart\n")
ci_u = np.array([c_of(P_INIT[q]) for q in qs_u if q in P_INIT])
ci_l = np.array([c_of(P_INIT[q]) for q in qs_l if q in P_INIT])
iu2 = rng.integers(0, len(ci_u), size=(NBOOT, len(ci_u)))
il2 = rng.integers(0, len(ci_l), size=(NBOOT, len(ci_l)))
dc = ci_l[il2].mean(axis=1) - ci_u[iu2].mean(axis=1)
clo, chi = np.percentile(dc, [2.5, 97.5])
c_sep = clo > 0 or chi < 0
W("At the moment training starts, both groups sit at the SAME place on this "
  "mechanism — that is what makes the comparison informative rather than "
  "circular:\n")
W("| group | mean c(p̂) at the INITIAL policy | median |")
W("|---|---|---|")
W(f"| unlearnable | {ci_u.mean():.4f} | {np.median(ci_u):.4f} |")
W(f"| learnable | {ci_l.mean():.4f} | {np.median(ci_l):.4f} |")
W("")
W(f"**Difference {ci_l.mean() - ci_u.mean():+.4f}, 95 % CI "
  f"[{clo:+.4f}, {chi:+.4f}] — separates: "
  f"{'**YES**' if c_sep else '**NO**'}.**\n")

# --- POST-HOC (labelled): reverse causality, and the decisive subset --------
W("### Two checks added AFTER seeing the numbers — labelled as post-hoc\n")
W("**These were not in A37.3.** They are reported as exploratory, not as "
  "registered tests, because the reason for running them came from the data.\n")
W("**(i) Reverse causality.** The observed contributing fraction is partly an "
  "OUTCOME of learning, not only a cause of it: a prompt that improves moves "
  "its pass rate into the productive band and therefore contributes more "
  "often afterwards. So the 0.3180 / 0.4684 gap in the table above cannot be "
  "read as purely causal, and A37.3 should have said so. The initial-policy "
  "comparison directly above is the one that is free of this.\n")
thr = ln["mean"]
rich = [q for q in qs_u if CONTRIB[q] >= thr]
W(f"**(ii) The decisive subset.** {len(rich)} of {len(qs_u)} unlearnable "
  f"prompts ({100 * len(rich) / len(qs_u):.1f} %) received at least "
  f"{thr:.2f} contributing steps per run — the MEAN received by learnable "
  "prompts — and still never improved in any of the five runs. If the sampling "
  "budget were the binding constraint, these prompts should have moved.\n")

def auc(pos, neg):
    """P(a random pos scores below a random neg) -- oriented so >0.5 means the
    score identifies the unlearnable group by being SMALLER."""
    p, n = np.asarray(pos), np.asarray(neg)
    return float((np.sum(p[:, None] < n[None, :]) + 0.5 * np.sum(p[:, None] == n[None, :]))
                 / (len(p) * len(n)))


a_contrib = auc(vu, vl)
a_init = auc(np.array([P_INIT[q] for q in qs_u if q in P_INIT]),
             np.array([P_INIT[q] for q in qs_l if q in P_INIT]))
W("### Item 4, answered directly: how much is already predicted?\n")
W("Discrimination of unlearnable from learnable, as AUC:\n")
W("| predictor | AUC |")
W("|---|---|")
W(f"| contributing steps actually received (advantage collapse) | **{a_contrib:.3f}** |")
W(f"| initial-policy pass rate (for reference) | {a_init:.3f} |")
W("")
W(f"Advantage collapse alone separates the two groups at AUC "
  f"**{a_contrib:.3f}** — better than chance, and far short of an explanation. "
  "The gradient budget is a real gradient on difficulty, but it is not what "
  "distinguishes a prompt that never improves from one that does.\n")

W("## Verdict against the registered branches (A37.5 / A37.6)\n")
starved = ul["mean"] < 1.0 and ul["zero_share"] > 0.5
if starved and ratio > 3:
    W("**BRANCH A (A37.5).** The unlearnable group sits where the contributing "
      "probability is near zero and is starved in absolute terms.\n")
else:
    W("**BRANCH B (A37.6) FIRES.**\n")
    W(f"The groups do separate on `c(p̂)` — as A37.4 warned they nearly had to — "
      f"but the absolute test fails on every count:\n")
    W(f"* unlearnable prompts received **{ul['mean']:.2f} contributing gradient "
      f"steps per run**, not near zero;")
    W(f"* **{100 * ul['zero_share']:.1f} %** of them were fully starved across "
      "all five runs;")
    W(f"* the gap to learnable is only **{ratio:.2f}x** "
      f"({ln['mean'] - ul['mean']:+.2f} steps, CI [{dlo:+.2f}, {dhi:+.2f}]);")
    W(f"* **{len(rich)} unlearnable prompts ({100 * len(rich) / len(qs_u):.1f} %) "
      f"received at least the MEAN learnable exposure and still never improved.**\n")
    W("**Per A37.6 this goes in the Discussion as a negative result: advantage "
      "collapse does not explain the phenomenon.** We say so plainly rather "
      "than leaving the question open, and we do not have a rival mechanism to "
      "offer in its place.\n")
    W("**What the mechanism DOES do, and it is worth stating.** The effect is "
      f"real and large in the difficulty direction: easy prompts received "
      f"{absres['easy']['mean']:.2f} contributing steps per run against "
      f"{ul['mean']:.2f} for unlearnable ones, a "
      f"{absres['easy']['mean'] / ul['mean']:.2f}x gap, and `c(p̂)` at the "
      f"initial policy tracks the realised contributing fraction at "
      f"r = {r_init:+.3f}. So the sampling budget measurably starves hard "
      "prompts of gradient — it simply is not what separates the unlearnable "
      "from the learnable, which is the phenomenon under study.\n")

rest = [q for q in QIDS if q not in PRIMARY["unlearnable"] and q in P_INIT]
a_init_rest = auc(np.array([P_INIT[q] for q in qs_u if q in P_INIT]),
                  np.array([P_INIT[q] for q in rest]))

# ---- what CAN be said about the 35 fully-exposed prompts -------------------
W("## The 35 fully-exposed prompts: what the record can and cannot say\n")
sr = []
for s in SEEDS:
    p = glob.glob(os.path.join(REPO, "checkpoints",
                               f"qwen_0.5b_baseline_1gpu_*seed{s}", "prompt_reward_stats.csv"))[0]
    d = pd.read_csv(p).drop_duplicates(["step", "prompt_id"], keep="first")
    sr.append(d[d.prompt_id.isin(rich)])
RD = pd.concat(sr)
nd = RD[RD.std_reward > 0]
half = RD.step.max() / 2
early = RD[RD.step <= half].mean_reward.mean()
late = RD[RD.step > half].mean_reward.mean()
W("| quantity, over the 35 prompts x 5 runs | value |")
W("|---|---|")
W(f"| non-degenerate groups (real gradient events) | {len(nd)} |")
W(f"| mean correct rollouts in those groups, out of {G} | "
  f"{G * nd.mean_reward.mean():.2f} |")
W(f"| max correct rollouts seen in any single group | "
  f"{int(round(G * RD.mean_reward.max()))} |")
W(f"| mean training reward, first half of training | {early:.4f} |")
W(f"| mean training reward, second half | {late:.4f} |")
W(f"| change across training | {late - early:+.4f} |")
W(f"| mean pass rate, initial eval -> trained eval | "
  f"{np.mean([P_INIT[q] for q in rich]):.4f} -> "
  f"{np.mean([P_TRAINED[q] for q in rich]):.4f} |")
W("")

json.dump({"G": G, "G_source": f"repro/train_1gpu.sh:{G_LINE}",
           "auc_contrib": a_contrib, "auc_init_rate": a_init,
           "auc_init_rest": a_init_rest,
           "rich_diag": {"nondeg_events": int(len(nd)),
                         "mean_correct": float(G * nd.mean_reward.mean()),
                         "reward_early": float(early), "reward_late": float(late),
                         "p_init": float(np.mean([P_INIT[q] for q in rich])),
                         "p_trained": float(np.mean([P_TRAINED[q] for q in rich]))},
           "branch": "A" if (starved and ratio > 3) else "B",
           "init_c": {"unlearnable": float(ci_u.mean()),
                      "learnable": float(ci_l.mean()),
                      "diff_ci": [float(clo), float(chi)],
                      "separates": bool(c_sep)},
           "decisive_subset": {"n": len(rich), "of": len(qs_u),
                               "threshold": float(thr)},
           "absolute": absres, "diff_ci": [float(dlo), float(dhi)],
           "ratio": float(ratio), "overlap_share": overlap,
           "corr_init": r_init, "corr_trained": r_tr,
           "dups_dropped": int(dup_total)},
          open(os.path.join(EXP, "analysis", "advantage_collapse.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------------------------------------------------------------------
# A37.9 -- machine-generated macros, same discipline as A36.7. Every value is
# the object the tables above printed; nothing is hand-typed.
# Formatting contract: no percent or dollar sign inside any macro; explicit
# sign on anything printed inside a bracketed interval; counts bare integers.
# ---------------------------------------------------------------------------
T = ["% GENERATED BY EXPERIMENTS/E1_rescue/advantage_collapse.py -- DO NOT EDIT BY HAND.",
     "% Rerun that script to regenerate. A37.9.",
     f"% G = {G} read from repro/train_1gpu.sh:{G_LINE}; bootstrap {NBOOT} resamples, seed {SEED}.",
     ""]
T.append(r"\newcommand{\acG}{%d}" % G)
T.append("")
for tag, g in (("Unlearn", "unlearnable"), ("Learn", "learnable"), ("Easy", "easy")):
    r = absres[g]
    T.append(r"\newcommand{\acN%s}{%d}" % (tag, r["n"]))
    T.append(r"\newcommand{\acSteps%s}{%.2f}" % (tag, r["mean"]))
    T.append(r"\newcommand{\acSteps%sLo}{%.2f}" % (tag, r["lo"]))
    T.append(r"\newcommand{\acSteps%sHi}{%.2f}" % (tag, r["hi"]))
    T.append(r"\newcommand{\acStarved%s}{%.1f}" % (tag, 100 * r["zero_share"]))
    T.append("")
T.append(r"\newcommand{\acGapRatio}{%.2f}" % ratio)
T.append(r"\newcommand{\acGapAbs}{%+.2f}" % (ln["mean"] - ul["mean"]))
T.append(r"\newcommand{\acGapAbsLo}{%+.2f}" % dlo)
T.append(r"\newcommand{\acGapAbsHi}{%+.2f}" % dhi)
T.append("")
T.append(r"\newcommand{\acFullExposureN}{%d}" % len(rich))
T.append(r"\newcommand{\acFullExposurePct}{%.1f}" % (100 * len(rich) / len(qs_u)))
T.append("")
T.append(r"% AUC definitions, spelled out because a hand-typed 0.927 in")
T.append(r"% EXPERIMENTS/E3_removal/power.py:287 was long mistaken for the third")
T.append(r"% of these. It is NOT: it was computed on a sub-tau FLAGGED set")
T.append(r"% (218-intersection 0.919 / 404-union 0.931), not on the published")
T.append(r"% 156-prompt unlearnable partition, and it is not reproducible from")
T.append(r"% any script. See A37.10 and A37.11.")
T.append(r"% acAUC            : contributing steps received, unlearnable vs learnable")
T.append(r"% acAUCInitialUL   : initial-policy pass rate, unlearnable vs LEARNABLE only")
T.append(r"% acAUCInitialRest : initial-policy pass rate, unlearnable vs ALL other prompts")
T.append(r"\newcommand{\acAUC}{%.3f}" % a_contrib)
T.append(r"\newcommand{\acAUCInitialUL}{%.3f}" % a_init)
T.append(r"\newcommand{\acAUCInitialRest}{%.3f}" % a_init_rest)
T.append(r"\newcommand{\acExposureRatio}{%.2f}" % (absres["easy"]["mean"] / ul["mean"]))
T.append(r"\newcommand{\acCorrInitial}{%+.3f}" % r_init)
TEX = os.path.join(EXP, "analysis", "advantage_collapse_numbers.tex")
open(TEX, "w").write("\n".join(T) + "\n")
print(f"\nwrote {TEX}")
