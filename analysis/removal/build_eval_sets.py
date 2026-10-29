"""E3 — build the two held-out evaluation sets in the schema repro/eval.sh wants.

PRIMARY   e3_math500   500 problems, split='test', disjoint from all training
SECONDARY e3_pool2000  2000 problems sampled from the 7116 pool prompts that the
                       1023-prompt training subset never used

Both are fixed once here and shared by every arm and every seed, so the
prompt-difficulty component of variance is common and cancels in every
between-arm comparison (this is what the power analysis assumes).

The eval loader reads data/<tag>/<split>.jsonl. MATH500's parquet carries a
different column set from the training jsonl -- no `answer`, `question` or
`target` -- so those are derived here from `reward_model.ground_truth` and
`extra_info.question` rather than left missing.

  $HOME/miniconda3/envs/vllm/bin/python build_eval_sets.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/media/rmedu/3039-8E90/LLM"
EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "unlearnability-rlvr")
ZOO = os.path.join(REPO, "data", "SimpleRL-Zoo-Data", "simplelr_qwen_level1to4")
SUB1K = os.path.join(REPO, "data", "SimpleRL-Zoo-Data",
                     "simplelr_qwen_level1to4_sub1k", "train.parquet")
SEED = 20260809
N_POOL = 2000


def to_rows(df):
    out = []
    for r in df.to_dict("records"):
        ei = dict(r["extra_info"])
        gt = dict(r["reward_model"]).get("ground_truth")
        q = ei.get("question") or r.get("question")
        out.append({"answer": gt, "gt_answer": gt, "subject": r.get("subject"),
                    "level": str(r.get("level")), "question": q, "target": gt,
                    "data_source": r.get("data_source", "simplelr_qwen"),
                    "prompt": list(r["prompt"]), "ability": r.get("ability", "math"),
                    "reward_model": dict(r["reward_model"]),
                    "extra_info": ei})
    return out


def write(tag, split, rows):
    d = os.path.join(REPO, "data", tag)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{split}.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  {tag}@{split}: {len(rows)} rows -> {p}")
    return p


train_all = pd.read_parquet(os.path.join(ZOO, "train.parquet"))
sub1k = pd.read_parquet(SUB1K)
used = {r["index"] for r in sub1k.extra_info}
train_all["_qid"] = [r["index"] for r in train_all.extra_info]
pool = train_all[~train_all._qid.isin(used)]
assert len(pool) == len(train_all) - len(sub1k), "pool is not the exact complement"

test = pd.read_parquet(os.path.join(ZOO, "test.parquet"))
# MATH500 must not intersect the training pool at all
qs_test = {str(dict(r)["question"]).strip() for r in test.extra_info}
qs_train = {str(dict(r)["question"]).strip() for r in train_all.extra_info}
overlap = qs_test & qs_train
assert not overlap, f"MATH500 overlaps the training pool on {len(overlap)} problems"

print("building held-out evaluation sets")
write("e3_math500", "test", to_rows(test))

rng = np.random.default_rng(SEED)
idx = rng.choice(len(pool), size=min(N_POOL, len(pool)), replace=False)
sample = pool.iloc[np.sort(idx)].drop(columns=["_qid"])
qs_sample = {str(dict(r)["question"]).strip() for r in sample.extra_info}
assert not (qs_sample & {str(dict(r)["question"]).strip() for r in sub1k.extra_info}), \
    "pool sample intersects the training subset"
write("e3_pool2000", "test", to_rows(sample))

json.dump({"seed": SEED, "n_math500": len(test), "n_pool": len(sample),
           "pool_available": len(pool)},
          open(os.path.join(EXP, "analysis", "eval_sets.json"), "w"), indent=2)
print(f"\npool available: {len(pool)}; sampled {len(sample)}")
