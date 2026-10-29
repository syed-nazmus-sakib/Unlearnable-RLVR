"""Within-group cosine means for one matched-K gradient run, appended to
analysis/attenuation.json (A12.3's curve).

  python matchedK_similarity.py <grad_dir> <K> <subsample_index>

Same similarity as Figure 5: Chen's key filter, his flatten, cosine over the
leave-one-out group mean, accumulated through memmaps in feature chunks so peak
RAM stays a few hundred MB.

Results are stored as group means per K so the raw .npz and flattened .npy of a
K-run can be deleted immediately afterwards -- ten K-runs would otherwise need
~350 GB. The attenuation panel is rebuilt from this JSON, not from the
directories.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
sys.path.insert(0, REPO)
from common.grad_utils import flatten_gradient  # noqa: E402

GRADS, K, SUB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
FLAT = os.path.join(EXP, f"gradsim_flat_K{K}_s{SUB}")
STORE = os.path.join(EXP, "analysis", "attenuation.json")
CHUNK = int(os.environ.get("GRADSIM_CHUNK", 600_000))
MIN_CORRECT = MIN_INCORRECT = 4
GROUPS = ["du71", "learnable", "easy"]
os.makedirs(FLAT, exist_ok=True)


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


coh = json.load(open(os.path.join(EXP, "adaprefix", "gradsim_cohorts.json")))
n_ok, n_bad = defaultdict(int), defaultdict(int)
for r in json.load(open(os.path.join(REPO, "results",
                                     "e1_gradsim_initial_GUARDED.json"))):
    q = int(r["question_id"])
    if guarded(r.get("verification", {}) or {}):
        n_ok[q] += 1
    else:
        n_bad[q] += 1
included = {g: [q for q in coh[g] if n_ok[q] >= MIN_CORRECT and n_bad[q] >= MIN_INCORRECT]
            for g in GROUPS}

keys = None
qs, paths = [], []
for g in GROUPS:
    for q in included[g]:
        src = os.path.join(GRADS, f"average_gradients_prompt_correct_{q}.npz")
        if not os.path.exists(src):
            continue
        dst = os.path.join(FLAT, f"correct_{q}.npy")
        if not os.path.exists(dst):
            gz = np.load(src)
            if keys is None:
                keys = [k for k in gz.keys() if "base_layer" not in k and "mlp" not in k]
            np.save(dst, flatten_gradient(gz, keys))
        qs.append(q)
        paths.append(dst)

if len(qs) < 2:
    print(f"K={K} s={SUB}: only {len(qs)} gradients present, nothing to do")
    sys.exit(0)

mms = [np.load(p, mmap_mode="r") for p in paths]
n, D = len(qs), mms[0].shape[0]
print(f"K={K} s={SUB}: {n} prompts, accumulating cosines "
      f"(~{n * CHUNK * 4 / 1e9:.2f} GB peak)")
S = np.zeros((n, n), np.float64)
sq = np.zeros(n, np.float64)
for a in range(0, D, CHUNK):
    X = np.stack([np.asarray(m[a:a + CHUNK], dtype=np.float32) for m in mms])
    S += X @ X.T
    sq += np.einsum("ij,ij->i", X, X)
d = np.sqrt(sq)
C = S / np.outer(d, d)
idx = {q: i for i, q in enumerate(qs)}

store = json.load(open(STORE)) if os.path.exists(STORE) else {}
for g in GROUPS:
    here = [q for q in included[g] if q in idx]
    if len(here) < 2:
        continue
    sub = C[np.ix_([idx[q] for q in here], [idx[q] for q in here])].copy()
    np.fill_diagonal(sub, np.nan)
    per_prompt = np.nanmean(sub, axis=1)
    m = float(np.nanmean(per_prompt))
    store.setdefault(g, {}).setdefault(str(K), [])
    # a re-run of the same (K, subsample) REPLACES its entry rather than
    # appending a duplicate -- K=4 and K=6 are re-run to recover per-prompt values
    store[g][str(K)] = [e for e in store[g][str(K)] if e.get("sub") != SUB]
    # PER-PROMPT values are stored, not just the group mean. Without them a
    # fixed-cohort curve cannot be built after the fact, and the gradients are
    # deleted immediately after this script runs (350 GB otherwise). Storing only
    # means cost a re-run of K=4 and K=6, which had already been measured when
    # the cohort-drift problem was noticed.
    store[g][str(K)].append({
        "sub": SUB, "mean": m, "n_prompts": len(here),
        "per_prompt": {str(q): float(v) for q, v in zip(here, per_prompt)},
    })
    print(f"  {g:10s} K={K:>3} sub={SUB}: mean cosine {m:.4f} over {len(here)} prompts "
          f"(per-prompt values stored)")
json.dump(store, open(STORE, "w"), indent=1)
print(f"appended to {STORE}")
