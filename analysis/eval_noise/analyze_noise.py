"""E0b: is the unlearnable set unstable because of TRAINING seeds, or because of
pass@1 MEASUREMENT noise?

E0 found that independently trained seeds agree on only ~0.72-0.76 Jaccard about
which prompts are flagged (pass@1 <= tau), even though aggregate pass@1 agrees to
three decimals. That was interpreted as training-seed instability. But the label is
a thresholded count over 32 samples: at tau=0.1 a prompt is flagged iff it gets
<= 3 correct, and a prompt whose true rate is near 0.1 is close to a coin flip on
sampling alone. So some of the disagreement is guaranteed regardless of training.

Three comparisons separate the two:

  1. SAME MODEL, TWO EVALS. One checkpoint evaluated twice with independent
     sampling. Any disagreement here is pure measurement noise -- there is no
     training difference to appeal to.

  2. ACROSS TRAINING SEEDS. The E0 number, recomputed here on the same prompt
     population so the two are comparable.

  3. ANALYTIC PREDICTION. From a 128-sample estimate of each prompt's true rate p,
     the probability it gets flagged in a 32-sample eval is P = P(X <= floor(tau*32)),
     X ~ Bin(32, p). For two independent evals,
         E|A n B| = sum P_i^2 ,  E|A u B| = sum (2P_i - P_i^2)
     giving an expected Jaccard that depends on nothing but sampling.

If (1) ~ (2) ~ (3), the instability is measurement noise and the fix is more eval
samples, not more training seeds. If (1) and (3) are near 1.0 while (2) is 0.72,
the instability is genuine training variation and E0's reading stands.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
E0 = os.path.join(EXP, "..", "E0_seed_stability")


def load_counts(pattern):
    """qid -> (n_correct, n_samples)"""
    paths = sorted(glob.glob(os.path.join(REPO, pattern)))
    if not paths:
        return None
    scores = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for item in json.load(f):
                scores[item["question_id"]].append(int(item["verification"]["score"]))
    return {q: (sum(v), len(v)) for q, v in scores.items()}


def flagged(counts, tau):
    return {q for q, (c, n) in counts.items() if c / n <= tau}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Bin(n, p). Exact, no scipy."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-seed", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--outdir", default=os.path.join(EXP, "analysis"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tau = args.threshold
    cs = args.ckpt_seed

    orig_pat = ("results/qwen_0.5b_run1_*@train*.json" if cs == 1
                else f"results/qwen_0.5b_seed{cs}_*@train*.json")
    rep_pat = f"results/qwen_0.5b_ckpt{cs}_rep2_*@train*.json"
    hi_pat = f"results/qwen_0.5b_ckpt{cs}_hi_*@train*.json"

    orig = load_counts(orig_pat)
    rep = load_counts(rep_pat)
    hi = load_counts(hi_pat)
    if orig is None:
        sys.exit(f"missing original eval for seed {cs}")
    if rep is None:
        sys.exit("missing repeat eval -- run run_control.sh first")

    n_samp = max(n for _, n in orig.values())
    kmax = math.floor(tau * n_samp)

    f_orig, f_rep = flagged(orig, tau), flagged(rep, tau)
    j_same = jaccard(f_orig, f_rep)

    # --- cross training seeds, same population -------------------------------
    seed_pats = {1: "results/qwen_0.5b_run1_*@train*.json"}
    for s in (2, 3, 4, 5):
        if glob.glob(os.path.join(REPO, f"results/qwen_0.5b_seed{s}_*@train*.json")):
            seed_pats[s] = f"results/qwen_0.5b_seed{s}_*@train*.json"
    seed_flags = {s: flagged(load_counts(p), tau) for s, p in seed_pats.items()}
    cross = [(a, b, jaccard(seed_flags[a], seed_flags[b]))
             for i, a in enumerate(sorted(seed_flags))
             for b in sorted(seed_flags)[i + 1:]]
    j_cross = float(np.mean([j for _, _, j in cross])) if cross else float("nan")

    L = []
    L.append("# E0b - eval noise vs training-seed instability\n")
    L.append(f"Checkpoint: seed {cs}, global_step_120 | tau={tau} | {n_samp} samples "
             f"=> flagged iff correct <= {kmax}\n")
    L.append("## Headline\n")
    L.append(f"| comparison | Jaccard on flagged sets |")
    L.append(f"|---|---|")
    L.append(f"| **same model, two independent evals** | **{j_same:.3f}** |")
    L.append(f"| across training seeds (mean of {len(cross)} pairs) | {j_cross:.3f} |")
    L.append("")
    L.append(f"- same-model flagged: {len(f_orig)} vs {len(f_rep)}, shared {len(f_orig & f_rep)}\n")
    L.append(f"- prompts flagged in exactly one of the two evals: **{len(f_orig ^ f_rep)}**\n")

    ratio = (1 - j_same) / (1 - j_cross) if j_cross < 1 else float("nan")
    L.append(f"\n**Share of the cross-seed disagreement reproduced with no training "
             f"difference at all: {100*ratio:.0f}%.**\n")

    # --- analytic prediction from the 128-sample estimate --------------------
    if hi is not None:
        n_hi = max(n for _, n in hi.values())
        common = sorted(set(hi) & set(orig) & set(rep))
        P = {}
        for q in common:
            c, n = hi[q]
            P[q] = binom_cdf(kmax, n_samp, c / n)
        num = sum(p * p for p in P.values())
        den = sum(2 * p - p * p for p in P.values())
        j_pred = num / den if den else float("nan")

        js_same = jaccard(f_orig & set(common), f_rep & set(common))
        cross_sub = [jaccard(seed_flags[a] & set(common), seed_flags[b] & set(common))
                     for a, b, _ in cross]
        j_cross_sub = float(np.mean(cross_sub)) if cross_sub else float("nan")

        coin = [q for q in common if 0.05 <= P[q] <= 0.95]
        L.append(f"\n## Analytic check ({n_hi}-sample estimate, {len(common)} prompts)\n")
        L.append("| | Jaccard |")
        L.append("|---|---|")
        L.append(f"| predicted from sampling alone | **{j_pred:.3f}** |")
        L.append(f"| observed, same model | {js_same:.3f} |")
        L.append(f"| observed, across training seeds | {j_cross_sub:.3f} |")
        L.append(f"\n- prompts whose flag probability is between 0.05 and 0.95 "
                 f"(i.e. genuinely uncertain at {n_samp} samples): **{len(coin)}** "
                 f"of {len(common)}\n")
        L.append(f"- to push a prompt at the threshold below a 5% flip rate you would "
                 f"need roughly {int(n_samp * (1.96/0.5)**2)} samples, not {n_samp}\n")

        with open(os.path.join(args.outdir, "true_rates.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["qid", f"correct_of_{n_hi}", "p_hat", "P_flagged_at_32",
                        "flagged_orig", "flagged_rep"])
            for q in common:
                c, n = hi[q]
                w.writerow([q, c, round(c / n, 4), round(P[q], 4),
                            int(q in f_orig), int(q in f_rep)])
    else:
        L.append("\n## Analytic check\n\n(skipped: high-sample eval not found)\n")

    L.append("\n## Reading\n")
    L.append("- **same-model Jaccard ~= cross-seed Jaccard** -> the instability is "
             "measurement noise. The correct fix is more eval samples per prompt, not "
             "more training seeds, and E0's decay curve is mostly re-measuring the same "
             "coin flip. D_u must be defined at a higher sample count before E1/E2.\n")
    L.append("- **same-model Jaccard near 1.0 while cross-seed stays ~0.75** -> the "
             "instability is genuine training variation and E0's reading stands.\n")
    L.append("- Anything between: report both, and treat the measurement-noise share as "
             "a correction to the decay curve rather than a refutation of it.\n")

    path = os.path.join(args.outdir, "noise_report.md")
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
