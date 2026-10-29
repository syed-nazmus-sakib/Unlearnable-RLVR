"""Define D_u at N=128 over all 1023 training prompts, and audit the K=5 set against it.

Both 128-sample passes are pooled here:
  pass 1  464 prompts = 404 ever-flagged (in >=1 of the 5 E0 seeds) + 60 never-flagged controls
  pass 2  559 prompts = the remaining never-flagged prompts
1023 - 404 = 619 never-flagged, of which 60 went into pass 1, leaving 559. So the
two passes split the never-flagged population at random, which is what makes the
poolability check in section 3 meaningful rather than decorative.

Sections:
  1  D_u at N=128 over all 1023
  2  prompts newly below tau that no 32-sample seed ever flagged
  3  poolability: pass-1 controls vs pass-2, KS and Mann-Whitney
  4  precision/recall of the K=5 intersection vs full-coverage truth, with a
     Clopper-Pearson interval on recall
  5  the two same-model Jaccard values, and which population each belongs to
  6  sample size needed to resolve the threshold, under an explicit definition
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
E0 = os.path.abspath(os.path.join(EXP, "..", "E0_seed_stability"))


def load_counts(pattern):
    paths = sorted(glob.glob(os.path.join(REPO, pattern)))
    if not paths:
        return {}
    sc = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for it in json.load(f):
                sc[it["question_id"]].append(int(it["verification"]["score"]))
    return {q: (sum(v), len(v)) for q, v in sc.items()}


def load_no_reward(rel):
    best = {}
    with open(os.path.join(REPO, rel)) as f:
        for row in csv.DictReader(f):
            i, r = int(row["prompt_id"]), float(row["mean_reward"])
            best[i] = max(best.get(i, -1.0), r)
    return {i for i, r in best.items() if r == 0.0}


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def min_samples(p, tau, target=0.95, nmax=4000, stable=50):
    """Smallest N such that P(flagged) >= target for a prompt of true rate p,
    and stays >= target for the next `stable` values of N.

    The stability window matters: `flagged` means X <= floor(tau*N), and that
    floor makes P(flagged) jump around with N, so a bare first-crossing can be a
    lucky N that the next value undoes.
    """
    ok = None
    for n in range(8, nmax + 1):
        k = math.floor(tau * n)
        if stats.binom.cdf(k, n, p) >= target:
            if ok is None:
                ok = n
            elif n - ok >= stable:
                return ok
        else:
            ok = None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--outdir", default=os.path.join(EXP, "analysis"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tau = args.tau
    L = []

    # ---------------- pooled 128-sample truth ----------------
    # NB: "…_hi_*" also matches "…_hi_rest_*", which would silently merge the two
    # passes and make pass 1 look like it covered all 1023. Match pass 1 by its
    # own dataset tag instead.
    hi1 = load_counts("results/qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train*.json")
    hi2 = load_counts("results/qwen_0.5b_ckpt1_hi_rest_*@train*.json")
    overlap = set(hi1) & set(hi2)
    hi = {**hi1, **hi2}
    p_hat = {q: c / n for q, (c, n) in hi.items()}

    # ---------------- 32-sample seed evals ----------------
    seed_pat = {1: "results/qwen_0.5b_run1_*@train*.json"}
    for s in (2, 3, 4, 5):
        seed_pat[s] = f"results/qwen_0.5b_seed{s}_*@train*.json"
    seed_flag, no_reward = {}, set()
    for s, pat in seed_pat.items():
        c = load_counts(pat)
        seed_flag[s] = {q for q, (a, b) in c.items() if a / b <= tau}
        csvp = glob.glob(os.path.join(REPO, f"checkpoints/*_seed{s}/prompt_reward_stats.csv"))
        if csvp:
            no_reward |= load_no_reward(os.path.relpath(csvp[0], REPO))
    ever = set().union(*seed_flag.values())

    # ---------------- 1. D_u at N=128 ----------------
    du = {q for q, v in p_hat.items() if v <= tau} - no_reward
    L.append("# Final D_u at N=128, all 1023 prompts\n")
    L.append(f"tau={tau} | pooled coverage {len(hi)} prompts "
             f"(pass1 {len(hi1)} + pass2 {len(hi2)}, overlap {len(overlap)})\n")
    L.append("## 1. D_u\n")
    L.append(f"- prompts with 128-sample pass@1 <= {tau}: **{len({q for q,v in p_hat.items() if v<=tau})}**")
    L.append(f"- minus no_reward union ({len(no_reward)}): **|D_u| = {len(du)}**\n")

    # ---------------- 2. newly discovered ----------------
    new = du - ever
    L.append("## 2. Prompts no 32-sample seed ever flagged\n")
    L.append(f"- **{len(new)}** of {len(du)} D_u prompts were never flagged in any of the 5 seeds\n")
    # expectation is reported whether or not any were found -- that comparison is
    # the point, and suppressing it when the count is 0 hides the agreement
    exp = sum((1 - stats.binom.cdf(math.floor(tau * 32), 32, p_hat[q])) ** 5 for q in du)
    L.append(f"- binomial expectation for D_u prompts escaping all 5 independent "
             f"32-sample evals: **{exp:.2f}**; observed **{len(new)}**\n")
    if new:
        rows = sorted(((q, p_hat[q]) for q in new), key=lambda t: t[1])
        L.append("| qid | 128-sample pass@1 | P(flagged at N=32) |")
        L.append("|---|---|---|")
        for q, v in rows[:15]:
            L.append(f"| {q} | {v:.4f} | {stats.binom.cdf(math.floor(tau*32), 32, v):.3f} |")
        if len(rows) > 15:
            L.append(f"| … | {len(rows)-15} more | |")
    L.append("")

    # ---------------- 3. poolability ----------------
    groups = json.load(open(os.path.join(EXP, "subset_groups.json")))
    ctrl = [p_hat[q] for q in groups["control"] if q in p_hat]
    rest = [p_hat[q] for q in hi2 if q in p_hat]
    ks = stats.ks_2samp(ctrl, rest)
    mw = stats.mannwhitneyu(ctrl, rest, alternative="two-sided")
    L.append("## 3. Poolability of the two passes\n")
    L.append("Pass-1 controls and pass-2 are a random split of the same never-flagged "
             "population (619 = 60 + 559), so a difference here would indicate a "
             "measurement artifact rather than a real one.\n")
    L.append("| group | n | mean | median |")
    L.append("|---|---|---|---|")
    L.append(f"| pass-1 controls | {len(ctrl)} | {np.mean(ctrl):.4f} | {np.median(ctrl):.4f} |")
    L.append(f"| pass-2 remainder | {len(rest)} | {np.mean(rest):.4f} | {np.median(rest):.4f} |")
    L.append(f"\n- Kolmogorov-Smirnov: D={ks.statistic:.4f}, p={ks.pvalue:.3f}")
    L.append(f"- Mann-Whitney U: U={mw.statistic:.1f}, p={mw.pvalue:.3f}")
    verdict = ("indistinguishable -- safe to pool" if min(ks.pvalue, mw.pvalue) > 0.05
               else "DIFFERENT -- do not pool without investigating")
    L.append(f"- verdict: **{verdict}**\n")

    # ---------------- 4. K=5 audit ----------------
    k5f = glob.glob(os.path.join(E0, "partitions/K5_*/qwen_0.5b_math_level1to4_unlearnable.json"))
    k5 = set(json.load(open(k5f[0]))) if k5f else set()
    tp, fp, fn = len(k5 & du), len(k5 - du), len(du - k5)
    prec = tp / len(k5) if k5 else float("nan")
    rec = tp / len(du) if du else float("nan")
    lo, hi_ci = clopper_pearson(tp, len(du))
    L.append("## 4. K=5 intersection vs full-coverage truth\n")
    L.append(f"| | |")
    L.append(f"|---|---|")
    L.append(f"| K=5 intersection | {len(k5)} |")
    L.append(f"| D_u at N=128 | {len(du)} |")
    L.append(f"| true positives | {tp} |")
    L.append(f"| false positives | {fp} |")
    L.append(f"| false negatives | {fn} |")
    L.append(f"| precision | **{prec:.3f}** |")
    L.append(f"| recall | **{rec:.3f}**  (95% CI {lo:.3f}-{hi_ci:.3f}, Clopper-Pearson) |")
    L.append("\nClopper-Pearson rather than Wilson: it is exact and does not "
             "under-cover near the boundary, and precision here may sit at 1.0.\n")

    # ---------------- 5. Jaccard reconciliation ----------------
    orig, rep = load_counts(seed_pat[1]), load_counts("results/qwen_0.5b_ckpt1_rep2_*@train*.json")
    fo = {q for q, (a, b) in orig.items() if a / b <= tau}
    fr = {q for q, (a, b) in rep.items() if a / b <= tau}
    j_all = len(fo & fr) / len(fo | fr)
    common464 = set(hi1) & set(orig) & set(rep)
    fo4, fr4 = fo & common464, fr & common464
    j_464 = len(fo4 & fr4) / len(fo4 | fr4)
    L.append("## 5. The two same-model Jaccard values\n")
    L.append("| value | population | n |")
    L.append("|---|---|---|")
    L.append(f"| {j_all:.3f} | all prompts both 32-sample evals cover | {len(set(orig)&set(rep))} |")
    L.append(f"| {j_464:.3f} | restricted to pass-1's 128-sample coverage | {len(common464)} |")
    L.append("\nBoth are correct; they differ only in population. The headline table "
             "quotes the full-population value because that is what compares like-for-like "
             "against the cross-seed 0.741, which is also computed on all 1023. The "
             "analytic section quotes the restricted value because the prediction it is "
             "compared against can only be formed where 128-sample rates exist.\n")
    L.append("**Quote the full-population value as the headline**, and label the other "
             "explicitly as restricted. With coverage now complete the distinction "
             "disappears for future runs.\n")

    # ---------------- 6. sample size ----------------
    L.append("## 6. What 'resolving the threshold' requires\n")
    L.append("A prompt whose true rate is exactly tau is a coin flip at every N -- "
             "P(flagged) -> 0.5 as N grows, so no sample size fixes it and any figure "
             "must be stated at an offset from tau.\n")
    L.append(f"Definition used: smallest N such that P(flagged) >= 0.95 for a prompt of "
             f"true rate p, holding for the next 50 values of N (the floor(tau*N) "
             f"boundary makes a bare first crossing unreliable).\n")
    L.append("| true rate p | distance below tau | N needed |")
    L.append("|---|---|---|")
    for p in (0.09, 0.08, 0.07, 0.06, 0.05):
        n = min_samples(p, tau)
        L.append(f"| {p:.2f} | {tau-p:.2f} | {n if n else '>4000'} |")
    L.append(f"\nThe earlier report's \"~491 samples\" was **not** derived this way. It "
             f"came from `32 * (1.96/0.5)**2`, a normal-approximation scaling with no "
             f"stated criterion attached, and should not be quoted. Use this table "
             f"instead, with the offset named.\n")

    path = os.path.join(args.outdir, "final_du_report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(args.outdir, "D_u_N128.json"), "w") as f:
        json.dump(sorted(du), f)
    with open(os.path.join(args.outdir, "true_rates_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "correct", "n", "p_hat", "in_D_u", "ever_flagged_32", "in_K5", "no_reward"])
        for q in sorted(hi):
            c, n = hi[q]
            w.writerow([q, c, n, round(p_hat[q], 4), int(q in du), int(q in ever),
                        int(q in k5), int(q in no_reward)])
    print("\n".join(L))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
