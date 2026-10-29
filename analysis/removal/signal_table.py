"""E3 item 1 — the gradient-signal table, and the single source of truth for
which prompts each arm removes.

**Nothing here is trained.** Every number is computed from rollout statistics
already on disk. That is the point: two aggregation rules over the SAME five
seeds remove 5.0 % and 27.6 % of the training signal, and stating it costs no
GPU time at all.

Method. GRPO computes a group-relative advantage over G rollouts of a prompt.
If all G receive the same reward the advantage is identically zero and the
prompt contributes NO gradient that step. For a prompt with pass rate p, that
happens with probability p^G + (1-p)^G. The expected number of prompt-instances
that DO contribute is therefore

    useful(q) = 1 - (p_q^G + (1-p_q)^G)

and a removal set's share of the training signal is sum(useful) over the set
divided by sum(useful) over all 1023. p_q is the mean pass@1 across the five
E0 training seeds at N=32; G = 8 is the configured rollout_n.

The sets written here are the ONLY definition used downstream; build_arms.py
reads this file rather than recomputing, so the table in the paper and the data
the arms train on cannot disagree.

  $HOME/miniconda3/envs/vllm/bin/python signal_table.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/media/rmedu/3039-8E90/LLM"
EXP = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(EXP, "analysis"), exist_ok=True)
OUT = os.path.join(EXP, "analysis", "signal_table.md")
SETS_JSON = os.path.join(EXP, "analysis", "removal_sets.json")

G = 8                      # rollout_n, from repro/train_1gpu.sh
N_EVAL = 32                # samples per prompt in the E0 seed evals
RANDOM_SEED = 20260809     # fixed here, before any arm is trained

d = pd.read_csv(os.path.join(ROOT, "EXPERIMENTS/E0_seed_stability/analysis/per_example.csv"))
S = [f"pass1_seed{i}" for i in range(1, 6)]
d["p"] = d[S].mean(axis=1)
d["useful"] = 1 - (d.p ** G + (1 - d.p) ** G)
TOT = float(d.useful.sum())
NALL = len(d)

coh = json.load(open(os.path.join(ROOT, "EXPERIMENTS/E1_rescue/adaprefix/gradsim_cohorts.json")))
du = sorted(int(q) for q in coh["du71"])

union = sorted(d.qid[d.n_flagged >= 1])
inter = sorted(d.qid[d.n_flagged == 5])

rng = np.random.default_rng(RANDOM_SEED)
rand404 = sorted(rng.choice(d.qid.values, size=len(union), replace=False).tolist())
rand218 = sorted(rng.choice(d.qid.values, size=len(inter), replace=False).tolist())


def share(qids):
    return 100 * float(d[d.qid.isin(qids)].useful.sum()) / TOT


def pct(qids):
    return 100 * len(qids) / NALL


ROWS = [("published D_u (N=128 pooled)", du, "the set used throughout our Part I"),
        ("intersection: flagged by all 5 seeds", inter, "the strict rule"),
        ("union: flagged by >=1 of 5 seeds", union, "the permissive rule"),
        ("random, size-matched to intersection", rand218, "positive control"),
        ("random, size-matched to union", rand404, "positive control")]

L = []
W = L.append
W("# The gradient-signal cost of an aggregation rule\n")
W("**COMPUTED, NOT TRAINED.** No model was trained to produce this table and no "
  "GPU time was used. Every quantity follows from rollout statistics already "
  "recorded in E0.\n")
W("## Result\n")
W("| removal set | prompts removed | % of training set | **% of gradient signal** |")
W("|---|---|---|---|")
for name, qids, _ in ROWS:
    W(f"| {name} | {len(qids)} | {pct(qids):.1f} % | **{share(qids):.2f} %** |")
W("")
W(f"**Two aggregation rules over the same five seeds remove "
  f"{share(inter):.1f} % and {share(union):.1f} % of the training signal — a "
  f"{share(union) / share(inter):.1f}x difference from a choice the original "
  "work does not present as a choice.**\n")
W("The prompt counts understate it in one direction and overstate it in the "
  "other. The intersection set is 21.3 % of the data but 5.0 % of the signal; "
  "a random set of the same size is 21.3 % of both. Removing 'unlearnable' "
  f"prompts is roughly {share(rand218) / share(inter):.1f}x cheaper, per prompt "
  "removed, than removing arbitrary ones.\n")

W("## Method\n")
W("GRPO forms a group-relative advantage over G rollouts of each prompt. When "
  "all G receive the same reward the advantage is identically zero and the "
  "prompt contributes **no gradient at all** for that step. For a prompt with "
  "pass rate p this happens with probability\n")
W("    P(degenerate) = p^G + (1-p)^G\n")
W("so the expected contribution of prompt q is `useful(q) = 1 - P(degenerate)`, "
  "and a set's share of the signal is `sum(useful over the set) / sum(useful "
  "over all 1023)`.\n")
W(f"* `p_q` — mean pass@1 over the five E0 training seeds, N={N_EVAL} each")
W(f"* `G = {G}` — the configured `rollout_n`")
W(f"* denominator — {TOT:.1f} expected contributing prompt-instances per epoch "
  f"out of {NALL} prompts, i.e. **just under half of the training set "
  "contributes anything on a given pass**")
W("")
W("**Why this matters for anyone filtering data.** Prompts labelled "
  "unlearnable sit near p = 0, so they are already discarded by the "
  "zero-variance filter most of the time. Removing them removes far less "
  "training signal than their count suggests, and a filtering study that "
  "reports only how many prompts it dropped is not reporting the quantity that "
  "governs the outcome.\n")

W("## Limitations\n")
W("* `p_q` is measured at the END of training (step 120), while the degeneracy "
  "that matters occurs throughout. A prompt at p = 0 early and p = 0.3 late "
  "contributes more than this estimate credits. The table is therefore a "
  "**first-order estimate of relative cost, not a training simulation**.")
W("* Independence across the G rollouts is assumed. Rollouts of one prompt "
  "share a policy and a prefix, so they are positively correlated, which makes "
  "degeneracy MORE likely than `p^G + (1-p)^G` implies. The signal shares of "
  "the near-zero-p sets are, if anything, **overstated** here.")
W("* The five seeds that produced `p_q` are the same five that define the "
  "unlearnable sets, so the table shares the leakage stated in A31.5.")

json.dump({"seed": RANDOM_SEED, "G": G, "total_useful": TOT, "n_all": NALL,
           "sets": {"du71": du, "intersection": inter, "union": union,
                    "random218": rand218, "random404": rand404},
           "share_pct": {"du71": share(du), "intersection": share(inter),
                         "union": share(union), "random218": share(rand218),
                         "random404": share(rand404)}},
          open(SETS_JSON, "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nsets written to {SETS_JSON}")
