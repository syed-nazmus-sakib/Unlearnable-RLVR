"""Measurements for the 1.5B-at-5120 costing. CPU only; the GPU is running A49.

Three things are MEASURED here rather than extrapolated, because the last
estimate missed by 8.9x and the miss was in verification:

  1. The token-length distribution of the 1.5B's actual 1024-cap responses,
     and what fraction are degenerate repeaters (R2: some 12-gram occurring
     >=30 times). Degenerate repeaters do not terminate, so at 5120 they run
     to the new cap -- that is what sets generation cost.

  2. Verification cost as a function of response LENGTH, on real responses,
     within the 1024 range. Gives the scaling exponent where we have data.

  3. Verification cost on responses SYNTHETICALLY EXTENDED to 2048/3072/4096/
     5120 tokens, which is the only way to see ANTLR's adaptive-prediction cost
     at lengths we have never generated. THIS IS A PROXY, and labelled one:
     extension repeats the response text, which reproduces a longer pathological
     token stream but not the exact string a 5120 run would emit.

  $HOME/miniconda3/envs/vllm/bin/python cost_5120_measure.py
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from transformers import AutoTokenizer

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
RES = os.path.join(ROOT, "unlearnability-rlvr", "results")
sys.path.insert(0, os.path.join(ROOT, "unlearnability-rlvr", "inference"))
from math_verifier import evaluate  # noqa: E402

SRC = "e5_gate_qwen15b_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json"
MODEL = os.path.join(ROOT, "models", "Qwen2.5-1.5B")
SEED, CAP, WORKERS = 20260812, 1024, 6      # 6 workers: leave the GPU job room
K, R2_MIN = 12, 30
TARGETS = [1024, 2048, 3072, 4096, 5120]
NPER = 24                                   # responses timed per length bucket


def max_kgram(s, k=K):
    if len(s) < k:
        return 0
    return Counter(s[i:i + k] for i in range(len(s) - k + 1)).most_common(1)[0][1]


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
recs = json.load(open(os.path.join(RES, SRC)))
resp = [r["response"] for r in recs]
gt = [r.get("ground_truth", "") for r in recs]
data = json.loads(open(os.path.join(
    ROOT, "unlearnability-rlvr", "data", "simplelr_qwen_level1to4_sub1k",
    "train_parsed.jsonl")).readline())  # ground truth is not stored in results
GT = {}
for line in open(os.path.join(ROOT, "unlearnability-rlvr", "data",
                              "simplelr_qwen_level1to4_sub1k", "train_parsed.jsonl")):
    d = json.loads(line)
    GT[int(d.get("id", d.get("extra_info", {}).get("index")))] = (
        d.get("ground_truth") or d.get("gt_answer") or d.get("target"))
gts = [GT[int(r["question_id"])] for r in recs]

lens = np.array([len(e) for e in tok(resp, add_special_tokens=False)["input_ids"]])
trunc = lens >= CAP
deg = np.array([max_kgram(s) >= R2_MIN for s in resp])

print("=== 1. LENGTH AND DEGENERACY, measured on all "
      f"{len(lens)} responses at cap {CAP} ===")
print(f"mean tokens {lens.mean():.1f}   median {np.median(lens):.0f}   "
      f"truncated {trunc.mean():.4f}")
print(f"degenerate (R2) overall {deg.mean():.4f}")
print(f"  R2 among TRUNCATED   {deg[trunc].mean():.4f}  (n={int(trunc.sum())})")
print(f"  R2 among UNtruncated {deg[~trunc].mean():.4f}  (n={int((~trunc).sum())})")
print(f"untruncated length: mean {lens[~trunc].mean():.1f}  "
      f"p90 {np.percentile(lens[~trunc], 90):.0f}  max {lens[~trunc].max()}")
nd_trunc = float((trunc & ~deg).mean())
print(f"truncated AND NOT degenerate (these may finish under 5120): {nd_trunc:.4f}")
print(f"truncated AND degenerate (these run to ANY cap): "
      f"{float((trunc & deg).mean()):.4f}")


def _t(a):
    g, s = a
    t0 = time.perf_counter()
    try:
        evaluate(answers=[g], completions=[s])
    except Exception:
        pass
    return time.perf_counter() - t0


rng = np.random.default_rng(SEED)

print("\n=== 2. VERIFY COST vs LENGTH, real responses, within the 1024 range ===")
edges = [(0, 256), (256, 512), (512, 900), (900, 1025)]
for lo, hi in edges:
    idx = np.where((lens >= lo) & (lens < hi))[0]
    if len(idx) == 0:
        continue
    pick = rng.choice(idx, size=min(200, len(idx)), replace=False)
    with ProcessPoolExecutor(WORKERS) as ex:
        ts = np.array(list(ex.map(_t, [(gts[i], resp[i]) for i in pick])))
    print(f"  {lo:4d}-{hi:4d} tok  n={len(pick):3d}  mean {ts.mean()*1000:9.2f} ms  "
          f"median {np.median(ts)*1000:8.2f} ms  max {ts.max():6.2f} s")

print("\n=== 3. SYNTHETIC EXTENSION (PROXY, not a generated 5120 response) ===")
print("Degenerate truncated responses, text-repeated to each target length.")
pool = np.where(trunc & deg)[0]
pick = rng.choice(pool, size=NPER, replace=False)
rows = {}
for T in TARGETS:
    items = []
    for i in pick:
        s, need = resp[i], T
        reps = max(1, int(np.ceil(need / max(lens[i], 1))))
        items.append((gts[i], (s * reps)))
    with ProcessPoolExecutor(WORKERS) as ex:
        ts = np.array(list(ex.map(_t, items)))
    rows[T] = {"mean_s": float(ts.mean()), "median_s": float(np.median(ts)),
               "max_s": float(ts.max()), "n": len(ts)}
    print(f"  target {T:5d} tok  n={len(ts)}  mean {ts.mean():7.3f} s  "
          f"median {np.median(ts):7.3f} s  max {ts.max():7.3f} s")

r = rows[TARGETS[0]]["mean_s"]
print("\n  growth vs the 1024 bucket:")
for T in TARGETS:
    print(f"    {T:5d}: {rows[T]['mean_s']/max(r,1e-9):6.2f}x")

json.dump({"n": int(len(lens)), "mean_tokens": float(lens.mean()),
           "trunc_share": float(trunc.mean()), "deg_share": float(deg.mean()),
           "deg_given_trunc": float(deg[trunc].mean()),
           "deg_given_untrunc": float(deg[~trunc].mean()),
           "untrunc_mean_tokens": float(lens[~trunc].mean()),
           "trunc_not_deg": nd_trunc,
           "trunc_and_deg": float((trunc & deg).mean()),
           "synthetic": rows},
          open(os.path.join(EXP, "cost_5120_measure.json"), "w"), indent=2)
print("\nwrote cost_5120_measure.json")
