"""Build the single-GPU training subset of simplelr_qwen_level1to4.

The full train split is 8139 prompts; at batch 32 (what fits one 24GB card) a
paper-faithful ~4 exposures per prompt would need ~1000 steps. Subsetting to
~1k prompts keeps exposures-per-prompt at ~4 in ~124 steps instead.

`extra_info["index"]` is preserved verbatim, because it is the join key used in
three places that must agree:
  - verl's rl_dataset sets prompt_id = extra_info["index"]  (-> prompt_reward_stats.csv)
  - inference/evaluate_model.py sets id  = extra_info["index"]  (-> question_id in results)
  - classification/finalize_unlearnable_example_id.py joins those two
Re-indexing the subset would silently misalign all three.
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4")
    ap.add_argument("--out", default="data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4_sub1k")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = pd.read_parquet(f"{args.src}/train.parquet")
    rng = np.random.RandomState(args.seed)
    frac = args.n / len(train)
    # stratify on the top-level `level` column; extra_info["level"] is uniformly 1
    # in this dataset and is not the MATH difficulty level.
    parts = [g.sample(n=int(round(len(g) * frac)), random_state=rng)
             for _, g in train.groupby("level")]
    sub = pd.concat(parts).sort_index()

    os.makedirs(args.out, exist_ok=True)
    sub.to_parquet(f"{args.out}/train.parquet", index=False)
    # validation file: keep the official 500-prompt test split unchanged
    pd.read_parquet(f"{args.src}/test.parquet").to_parquet(f"{args.out}/test.parquet", index=False)

    ids = [e["index"] for e in sub["extra_info"]]
    print(f"wrote {len(sub)} prompts -> {args.out}/train.parquet")
    print(f"  unique ids: {len(set(ids))}  range: {min(ids)}..{max(ids)}")
    print("  level distribution:")
    print(sub["level"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
