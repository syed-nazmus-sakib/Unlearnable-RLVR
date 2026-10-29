"""Build the prompt subset for the eval-noise control.

The control needs a high-sample pass@1 estimate for the prompts whose labels are
actually in doubt -- the ones at or near tau. Evaluating all 1023 prompts at 128
samples would be ~131k generations (~2h); restricting to the prompts that were
ever flagged, plus a never-flagged comparison group, is ~27k (~25min) and answers
the same question.

Groups written:
  ever_flagged   flagged (pass@1 <= tau) in at least one of the E0 training seeds.
                 This is the population whose instability we are trying to explain.
  control        a random sample of prompts never flagged in any seed, to show
                 that re-sampling noise is specific to the threshold region and
                 not a property of the eval in general.

extra_info["index"] is preserved verbatim -- it is the join key between
prompt_reward_stats.csv, the eval results, and the classification script.
Re-indexing would silently misalign all three.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-example",
                    default=os.path.join(EXP, "..", "E0_seed_stability", "analysis", "per_example.csv"))
    ap.add_argument("--src", default=f"{REPO}/data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4_sub1k")
    ap.add_argument("--out", default=f"{REPO}/data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4_noise")
    ap.add_argument("--n-control", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.per_example)
    ever = set(df.loc[df.n_flagged > 0, "qid"].astype(int))
    never = df.loc[df.n_flagged == 0, "qid"].astype(int).tolist()

    rng = np.random.RandomState(args.seed)
    n_ctrl = min(args.n_control, len(never))
    control = set(rng.choice(never, size=n_ctrl, replace=False).tolist())
    keep = ever | control

    train = pd.read_parquet(f"{args.src}/train.parquet")
    idx = train["extra_info"].map(lambda e: e["index"])
    sub = train[idx.isin(keep)].copy()

    os.makedirs(args.out, exist_ok=True)
    sub.to_parquet(f"{args.out}/train.parquet", index=False)
    pd.read_parquet(f"{args.src}/test.parquet").to_parquet(f"{args.out}/test.parquet", index=False)

    groups = {"ever_flagged": sorted(int(x) for x in ever),
              "control": sorted(int(x) for x in control)}
    with open(os.path.join(EXP, "subset_groups.json"), "w") as f:
        json.dump(groups, f)

    print(f"wrote {len(sub)} prompts -> {args.out}/train.parquet")
    print(f"  ever_flagged: {len(ever)}   control(never flagged): {len(control)}")
    print(f"  at 128 samples that is {128*len(sub)} generations")
    missing = keep - set(idx)
    if missing:
        print(f"  warning: {len(missing)} ids not found in the source parquet")


if __name__ == "__main__":
    main()
