"""E0: how much of the "unlearnable" set is a stable property, and how much is seed noise?

Chen et al. (arXiv:2605.16787) define unlearnable as the *intersection* over three
GRPO seeds of prompts still at pass@1 <= tau after training. Intersection is a
noise filter, but nobody has reported how much it actually filters, and the paper's
own Limitations concede that "a fraction of examples will sit near the boundary".
That fraction is unmeasured, and every downstream claim -- gradient outlierness,
low reasoning quality, resistance to intervention -- is a claim about whichever set
the intersection happened to produce.

This script measures it. Three questions, in order of how much they matter:

  1. DECAY. How does |unlearnable| shrink as seeds are added? Averaged over all
     C(K,j) subsets of size j, so the curve is not an artifact of seed ordering.
     If it is still falling steeply at j=K, the paper's 3-seed number is an
     overestimate and so is yours.

  2. PER-PROMPT STABILITY. For each prompt, in how many of the K seeds is it
     flagged? A prompt flagged K/K is robustly unlearnable. One flagged 3/5 is a
     coin flip that the intersection over an unlucky triple would have kept.
     Reported as a Wilson score interval on the flag probability, which needs no
     scipy and behaves sanely at k=0 and k=K.

  3. BOUNDARY MASS. The label is a thresholded count, not a continuous quantity:
     at 32 samples and tau=0.1 a prompt is flagged iff it gets <= 3 correct
     (0.1*32 = 3.2). Three correct is flagged, four is not. Prompts sitting at 3
     or 4 are one lucky rollout from changing class, and they are counted here
     separately from prompts that are genuinely far below the line.

Outputs into --outdir:
  per_example.csv      one row per prompt, all per-seed pass@1 values and its class
  stability_report.md  the headline numbers with an interpretation guide
  stability.png        decay curve, flag-count histogram, boundary scatter

Usage:
  python analyze_stability.py                 # every seed in manifest.tsv
  python analyze_stability.py --seeds 1 2 3   # a subset
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(EXP_DIR, "..", "..", "unlearnability-rlvr"))


# ---------------------------------------------------------------- loading

def load_manifest(path, want_seeds=None):
    if not os.path.exists(path):
        sys.exit(f"no manifest at {path} -- run run_seed.sh first")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            seed = int(row["seed"])
            if want_seeds and seed not in want_seeds:
                continue
            rows.append({"seed": seed,
                         "results_glob": row["results_glob"],
                         "reward_stats": row["reward_stats"]})
    if not rows:
        sys.exit("no seeds selected from the manifest")
    return sorted(rows, key=lambda r: r["seed"])


def load_pass1(repo, pattern):
    """qid -> (n_correct, n_samples), matching finalize_unlearnable_example_id.py."""
    paths = sorted(glob.glob(os.path.join(repo, pattern)))
    if not paths:
        sys.exit(f"no result files matched {pattern!r} under {repo}")
    scores = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for item in json.load(f):
                scores[item["question_id"]].append(int(item["verification"]["score"]))
    return {q: (sum(s), len(s)) for q, s in scores.items()}


def load_no_reward(repo, rel):
    """Prompts whose max mean_reward over all logged training steps is 0."""
    path = os.path.join(repo, rel)
    if not os.path.exists(path):
        sys.exit(f"reward-stats CSV not found: {path}")
    best = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            pid, r = int(row["prompt_id"]), float(row["mean_reward"])
            if pid not in best or r > best[pid]:
                best[pid] = r
    return {pid for pid, r in best.items() if r == 0.0}


# ---------------------------------------------------------------- stats

def wilson(k, n, z=1.96):
    """Score interval for a binomial proportion. Closed form, no scipy, and
    unlike the normal approximation it stays inside [0,1] at k=0 and k=n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def partition(flagged_by_seed, no_reward_by_seed, initial_flagged, subset):
    """The repo's definitions, restricted to a subset of seeds."""
    inter = set.intersection(*[flagged_by_seed[s] for s in subset])
    union = set.union(*[flagged_by_seed[s] for s in subset])
    nr = set.union(*[no_reward_by_seed[s] for s in subset])
    return inter - nr, initial_flagged - union, nr


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--manifest", default=os.path.join(EXP_DIR, "manifest.tsv"))
    ap.add_argument("--initial-glob", default="results/qwen_0.5b_initial_*@train*.json")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--outdir", default=os.path.join(EXP_DIR, "analysis"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tau = args.threshold
    rows = load_manifest(args.manifest, set(args.seeds) if args.seeds else None)
    seeds = [r["seed"] for r in rows]
    K = len(seeds)

    print(f"E0 stability analysis: {K} seed(s) {seeds}, tau={tau}\n")

    initial = load_pass1(args.repo, args.initial_glob)
    init_p1 = {q: c / n for q, (c, n) in initial.items()}
    initial_flagged = {q for q, p in init_p1.items() if p <= tau}
    all_qids = set(initial)

    pass1, flagged_by_seed, no_reward_by_seed, ncorrect = {}, {}, {}, {}
    for r in rows:
        s = r["seed"]
        counts = load_pass1(args.repo, r["results_glob"])
        pass1[s] = {q: c / n for q, (c, n) in counts.items()}
        ncorrect[s] = {q: c for q, (c, n) in counts.items()}
        flagged_by_seed[s] = {q for q, p in pass1[s].items() if p <= tau}
        no_reward_by_seed[s] = load_no_reward(args.repo, r["reward_stats"])
        print(f"  seed {s}: {len(pass1[s])} prompts, "
              f"{len(flagged_by_seed[s])} flagged, "
              f"{len(no_reward_by_seed[s])} no-reward")

    n_samples = max(n for _, n in initial.values())
    thresh_count = tau * n_samples

    # ---- 1. decay -------------------------------------------------------
    decay = []
    for j in range(1, K + 1):
        sizes_u, sizes_l = [], []
        for sub in combinations(seeds, j):
            u, l, _ = partition(flagged_by_seed, no_reward_by_seed, initial_flagged, sub)
            sizes_u.append(len(u))
            sizes_l.append(len(l))
        decay.append({"j": j, "n_subsets": len(sizes_u),
                      "u_mean": float(np.mean(sizes_u)), "u_min": min(sizes_u), "u_max": max(sizes_u),
                      "l_mean": float(np.mean(sizes_l)), "l_min": min(sizes_l), "l_max": max(sizes_l)})
    full_u, full_l, full_nr = partition(flagged_by_seed, no_reward_by_seed, initial_flagged, seeds)

    # ---- 2. per-prompt stability ---------------------------------------
    flag_count = {q: sum(q in flagged_by_seed[s] for s in seeds) for q in all_qids}
    ever_nr = set.union(*[no_reward_by_seed[s] for s in seeds])

    recs = []
    for q in sorted(all_qids):
        ps = [pass1[s].get(q, float("nan")) for s in seeds]
        k = flag_count[q]
        lo, hi = wilson(k, K)
        if q in ever_nr:
            cls = "no_reward"
        elif k == K:
            cls = "stable_unlearnable"
        elif k == 0:
            cls = "never_flagged"
        else:
            cls = "boundary"
        rec = {"qid": q, "initial_pass1": round(init_p1.get(q, float("nan")), 4),
               "n_flagged": k, "K": K, "flag_frac": round(k / K, 3),
               "wilson_lo": round(lo, 3), "wilson_hi": round(hi, 3),
               "mean_pass1": round(float(np.nanmean(ps)), 4),
               "std_pass1": round(float(np.nanstd(ps)), 4),
               "min_correct_of_%d" % n_samples: int(np.nanmin([ncorrect[s].get(q, np.nan) for s in seeds])),
               "ever_no_reward": q in ever_nr, "class": cls}
        for s, p in zip(seeds, ps):
            rec[f"pass1_seed{s}"] = round(p, 4)
        recs.append(rec)
    df = pd.DataFrame(recs)
    csv_path = os.path.join(args.outdir, "per_example.csv")
    df.to_csv(csv_path, index=False)

    # ---- 3. boundary mass ----------------------------------------------
    # Prompts whose per-seed correct-count straddles the threshold, i.e. one or
    # two rollouts out of n_samples would change the label.
    cand = df[(df.n_flagged > 0) & (~df.ever_no_reward)]
    near = cand[cand.mean_pass1.between(max(0.0, tau - 0.06), tau + 0.06)]
    boundary = cand[cand["class"] == "boundary"]

    # ---- 4. pairwise agreement -----------------------------------------
    jac = np.eye(K)
    for i, a in enumerate(seeds):
        for j2, b in enumerate(seeds):
            if i < j2:
                A, B = flagged_by_seed[a], flagged_by_seed[b]
                v = len(A & B) / len(A | B) if (A | B) else 1.0
                jac[i, j2] = jac[j2, i] = v

    # ---- 5. survival of a single-seed set ------------------------------
    surv = [len(full_u & (flagged_by_seed[s] - ever_nr)) / max(1, len(flagged_by_seed[s] - ever_nr))
            for s in seeds]

    # ---- report ---------------------------------------------------------
    L = []
    L.append(f"# E0 - seed stability of the unlearnable set\n")
    L.append(f"Seeds: {seeds} (K={K}) | tau={tau} | {n_samples} samples/prompt "
             f"| {len(all_qids)} prompts\n")
    L.append("## Headline\n")
    if K == 1:
        L.append("Only one seed. `unlearnable` here is the single-seed upper bound; "
                 "run more seeds before reading anything into it.\n")
    else:
        drop = 100 * (1 - decay[-1]["u_mean"] / decay[0]["u_mean"]) if decay[0]["u_mean"] else 0
        L.append(f"- Single seed: **{decay[0]['u_mean']:.1f}** unlearnable prompts on average.\n")
        if K >= 3:
            L.append(f"- Paper's 3-seed protocol: **{decay[2]['u_mean']:.1f}** "
                     f"(range {decay[2]['u_min']}-{decay[2]['u_max']} over "
                     f"{decay[2]['n_subsets']} triples).\n")
        L.append(f"- All {K} seeds: **{len(full_u)}**.\n")
        L.append(f"- So **{drop:.0f}%** of the single-seed set does not survive to K={K}.\n")
        L.append(f"- Of the prompts ever flagged (excluding no-reward), "
                 f"**{len(df[df['class']=='stable_unlearnable'])}** are flagged in all {K} seeds "
                 f"and **{len(boundary)}** in some but not all.\n")
        L.append(f"- Mean survival of one seed's set into the K={K} intersection: "
                 f"**{100*np.mean(surv):.0f}%**.\n")

    L.append("\n## Decay of |unlearnable| with seed count\n")
    # min/max reported for BOTH series: Figure 1(a) shades a band on each, and a
    # caption describing them needs the numbers to be checkable here.
    L.append("| seeds j | subsets | mean \\|unlearnable\\| | min | max "
             "| mean \\|learnable\\| | min | max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in decay:
        L.append(f"| {d['j']} | {d['n_subsets']} | {d['u_mean']:.1f} | {d['u_min']} "
                 f"| {d['u_max']} | {d['l_mean']:.1f} | {d['l_min']} | {d['l_max']} |")
    if K >= 3:
        last = decay[-1]["u_mean"] - decay[-2]["u_mean"]
        L.append(f"\nMarginal change from the {K-1}th to the {K}th seed: **{last:+.1f}** prompts. "
                 "If this is still large in magnitude, the curve has not converged and even "
                 "K seeds overestimates the stable set.\n")

    L.append("\n## Threshold discreteness\n")
    L.append(f"pass@1 is a count over {n_samples} samples, so tau={tau} means "
             f"flagged iff correct <= {math.floor(thresh_count)} "
             f"(tau*n = {thresh_count:.1f}). One extra correct rollout flips the label.\n")
    L.append(f"- Ever-flagged, non-no-reward prompts: **{len(cand)}**\n")
    L.append(f"- Of those, mean pass@1 within +/-0.06 of tau: **{len(near)}** "
             f"({100*len(near)/max(1,len(cand)):.0f}%) - these are threshold artifacts "
             f"as much as findings.\n")

    if K > 1:
        L.append("\n## Pairwise seed agreement (Jaccard on flagged sets)\n")
        L.append("| | " + " | ".join(f"s{s}" for s in seeds) + " |")
        L.append("|---" * (K + 1) + "|")
        for i, s in enumerate(seeds):
            L.append(f"| s{s} | " + " | ".join(f"{jac[i,j]:.2f}" for j in range(K)) + " |")
        off = jac[~np.eye(K, dtype=bool)]
        L.append(f"\nMean off-diagonal Jaccard: **{off.mean():.2f}**. Low values mean the "
                 "flagged sets are largely seed-specific and the intersection is doing "
                 "most of the definitional work.\n")

    L.append("\n## What to conclude\n")
    L.append("- **Stable set is large and the curve has flattened** -> unlearnability is a real "
             "per-prompt property. Use `class == stable_unlearnable` as D_u for E1/E2 and say "
             "explicitly that it is the K-seed set.\n")
    L.append("- **Stable set is small / curve still falling** -> the published 3-seed number is "
             "substantially seed noise. That is a publishable correction on its own, and it "
             "means every downstream analysis must be re-run on the stable subset.\n")
    L.append("- Either way, report D_u as `stable_unlearnable` and carry `boundary` as a "
             "separate group. Pooling them is what makes the 3-seed number fragile.\n")
    L.append(f"\nFiles: `per_example.csv` ({len(df)} rows), `stability.png`\n")

    rep_path = os.path.join(args.outdir, "stability_report.md")
    with open(rep_path, "w") as f:
        f.write("\n".join(L))

    # ---- figure ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))

        js = [d["j"] for d in decay]
        ax[0].plot(js, [d["u_mean"] for d in decay], "o-", color="#c1272d", label="unlearnable")
        ax[0].fill_between(js, [d["u_min"] for d in decay], [d["u_max"] for d in decay],
                           color="#c1272d", alpha=0.15)
        ax[0].plot(js, [d["l_mean"] for d in decay], "s--", color="#0b6e4f", label="learnable")
        ax[0].set_xlabel("seeds intersected"); ax[0].set_ylabel("prompts")
        ax[0].set_title("Set size vs seed count"); ax[0].set_xticks(js)
        ax[0].legend(); ax[0].grid(alpha=0.3)

        ks = list(range(1, K + 1))
        cnt = [int((cand.n_flagged == k).sum()) for k in ks]
        cols = ["#f4a582"] * (K - 1) + ["#c1272d"]
        ax[1].bar(ks, cnt, color=cols[:K], width=0.6)
        ax[1].set_xlabel(f"seeds flagging the prompt (of {K})")
        ax[1].set_ylabel("prompts"); ax[1].set_xticks(ks)
        ax[1].set_title("Per-prompt flag count (dark = stable)"); ax[1].grid(alpha=0.3, axis="y")

        ax[2].errorbar(cand.mean_pass1, cand.n_flagged, xerr=cand.std_pass1,
                       fmt="o", ms=3, alpha=0.45, color="#2166ac", elinewidth=0.6)
        ax[2].axvline(tau, color="k", ls="--", lw=1)
        ax[2].text(tau, 0.5, f" tau={tau}", fontsize=8, va="bottom")
        ax[2].set_xlabel("mean pass@1 across seeds"); ax[2].set_ylabel("seeds flagging")
        ax[2].set_title("Boundary mass"); ax[2].grid(alpha=0.3)
        ax[2].set_xlim(-0.01, min(0.6, float(cand.mean_pass1.max()) + 0.05) if len(cand) else 0.6)

        fig.suptitle(f"E0: seed stability of the unlearnable set (K={K}, tau={tau})")
        # explicit margins rather than tight_layout: with 3 panels plus a suptitle
        # tight_layout cannot satisfy its constraints and warns.
        fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.13, wspace=0.30)
        fig.savefig(os.path.join(args.outdir, "stability.png"), dpi=150)
        print(f"  wrote {os.path.join(args.outdir, 'stability.png')}")
    except Exception as e:                                    # plotting is optional
        print(f"  (skipped figure: {e})")

    print("\n" + "\n".join(L[:20]))
    print(f"\nwrote {rep_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
