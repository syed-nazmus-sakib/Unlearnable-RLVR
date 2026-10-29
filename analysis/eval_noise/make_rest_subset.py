"""Build the complement subset: every training prompt NOT covered by the first
128-sample eval.

Why this is needed. The first high-sample eval covered only prompts that were
flagged in at least one E0 seed, plus 60 never-flagged controls. Against that
population the K=5 intersection had recall 0.49 -- it missed half the prompts whose
true pass@1 is below tau. By the same logic, prompts that happened never to be
flagged in any of the 5 seeds may still be genuinely below tau: a prompt with true
rate 0.08 escapes flagging in all five 32-sample evals with probability
(1-0.749)^5, small but not zero, and the population of such prompts is large.
Until they are measured, D_u's recall on the full 1023 is unknown.

The complement is derived from the qids actually present in the first eval's
results, not from subset_groups.json -- the written parquet and the eval's
reported support differed slightly, and the results file is the authority on what
was really measured.

extra_info["index"] is preserved verbatim; it is the join key across
prompt_reward_stats.csv, the eval results, and the classification script.
"""
import argparse
import glob
import json
import os

import pandas as pd

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--done-glob", default="results/qwen_0.5b_ckpt1_hi_*@train*.json",
                    help="results of the first 128-sample eval")
    ap.add_argument("--src", default=f"{REPO}/data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4_sub1k")
    ap.add_argument("--out", default=f"{REPO}/data/SimpleRL-Zoo-Data/simplelr_qwen_level1to4_rest")
    args = ap.parse_args()

    done = set()
    paths = sorted(glob.glob(os.path.join(REPO, args.done_glob)))
    if not paths:
        raise SystemExit(f"no first-pass results matched {args.done_glob!r}")
    for p in paths:
        with open(p) as f:
            for item in json.load(f):
                done.add(int(item["question_id"]))

    train = pd.read_parquet(f"{args.src}/train.parquet")
    idx = train["extra_info"].map(lambda e: int(e["index"]))
    sub = train[~idx.isin(done)].copy()

    os.makedirs(args.out, exist_ok=True)
    sub.to_parquet(f"{args.out}/train.parquet", index=False)
    pd.read_parquet(f"{args.src}/test.parquet").to_parquet(f"{args.out}/test.parquet", index=False)

    print(f"already measured at 128 samples : {len(done)}")
    print(f"remaining                       : {len(sub)}")
    print(f"wrote -> {args.out}/train.parquet")
    print(f"  {128*len(sub)} generations, ~{128*len(sub)/15.5/60:.0f} min at the observed 15.5 gen/s")


if __name__ == "__main__":
    main()
