"""E3 — write the four arm datasets by removing prompts from sub1k.

Sets are READ from analysis/removal_sets.json (written by signal_table.py), not
recomputed, so the published signal table and the data actually trained on
cannot disagree.

Arm 1 is the untouched sub1k and already has five trained seeds from E0; it is
not built here.

  $HOME/miniconda3/envs/vllm/bin/python build_arms.py
"""
import json
import os
import shutil

import pandas as pd

ROOT = "/media/rmedu/3039-8E90/LLM"
EXP = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "unlearnability-rlvr", "data", "SimpleRL-Zoo-Data")
SRC = os.path.join(DATA, "simplelr_qwen_level1to4_sub1k")

sets = json.load(open(os.path.join(EXP, "analysis", "removal_sets.json")))
ARMS = {"e3arm2_union": "union", "e3arm3_inter": "intersection",
        "e3arm4_rand404": "random404", "e3arm5_rand218": "random218"}

train = pd.read_parquet(os.path.join(SRC, "train.parquet"))
train["_qid"] = [r["index"] for r in train.extra_info]
assert len(train) == 1023, f"source is {len(train)} rows, expected 1023"

print(f"source: {len(train)} prompts")
manifest = {}
for tag, key in ARMS.items():
    remove = set(sets["sets"][key])
    kept = train[~train._qid.isin(remove)].drop(columns=["_qid"])
    # every removed qid must actually be present, or the arm silently removes
    # fewer prompts than it claims
    n_removed = len(train) - len(kept)
    assert n_removed == len(remove), \
        f"{tag}: removed {n_removed} but set has {len(remove)}"
    dst = os.path.join(DATA, f"simplelr_qwen_level1to4_{tag}")
    os.makedirs(dst, exist_ok=True)
    kept.to_parquet(os.path.join(dst, "train.parquet"), index=False)
    shutil.copy(os.path.join(SRC, "test.parquet"), os.path.join(dst, "test.parquet"))
    epochs = 120 * 32 / len(kept)
    manifest[tag] = {"removed": len(remove), "kept": len(kept),
                     "signal_removed_pct": sets["share_pct"][key],
                     "epochs_at_120_steps": epochs}
    print(f"  {tag:16s} removed {len(remove):4d}  kept {len(kept):4d}  "
          f"signal -{sets['share_pct'][key]:5.2f}%  epochs {epochs:.2f}  -> {dst}")

json.dump(manifest, open(os.path.join(EXP, "analysis", "arm_manifest.json"), "w"),
          indent=2)
print("\nmanifest written")
