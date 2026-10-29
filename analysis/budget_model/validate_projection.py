"""Does a random projection preserve our gradient cosines?

Gate for the second-model run: at 3B the lm_head gradient is 788 MB per prompt
and one matched-K pass needs ~204 GB of disk against 104 GB free. Cosine
similarity needs only inner products, so a projection to a few tens of thousands
of dimensions removes the disk problem entirely -- IF it preserves the numbers.

VALIDATION, and note what it can and cannot cover: the matched-K gradient
directories were deleted after each run by design (ten runs would have needed
~350 GB), so the matched-K curve cannot be recomputed under projection without
re-running the GPU passes. The NATURAL-N vectors survive -- 130 prompts, 136 M
dimensions each -- and the full-vector cosine matrix over them is cached. That is
the same operation the matched-K curve is built from, on the same real
gradients, so it tests the projection where it matters.

Sketch: CountSketch (a sparse random projection). Each coordinate is hashed to
one of D_PROJ buckets with a random sign and accumulated. Inner products are
preserved in expectation with relative error ~1/sqrt(D_PROJ), and it never
materialises a projection matrix -- the hashes are regenerated per feature chunk
from SEED, so the whole thing is reproducible from that one number.

  $HOME/miniconda3/envs/vllm/bin/python validate_projection.py
"""
import json
import os

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(EXP, "gradsim_flat")
CACHE = os.path.join(EXP, "analysis", "cosine_cache_gradsim_grads.npz")
OUT = os.path.join(EXP, "analysis", "projection_validation.md")
SEED = 20260808                 # THE projection seed; recorded, never regenerated
D_PROJ = int(os.environ.get("D_PROJ", 65536))
CHUNK = int(os.environ.get("GRADSIM_CHUNK", 600_000))
GROUPS = ["du71", "learnable", "easy"]
LABEL = {"du71": "D_u", "learnable": "learnable", "easy": "easy"}

z = np.load(CACHE)
qids = [int(q) for q in z["qids"]]
C_full = z["C"]

coh = json.load(open(os.path.join(EXP, "adaprefix", "gradsim_cohorts.json")))
group_of = {q: g for g, qs in coh.items() for q in qs}

mms = [np.load(os.path.join(FLAT, f"correct_{q}.npy"), mmap_mode="r") for q in qids]
n, D = len(qids), mms[0].shape[0]
print(f"{n} prompts x {D} dims -> CountSketch to {D_PROJ} (seed {SEED})")

Y = np.zeros((n, D_PROJ), np.float64)
for a in range(0, D, CHUNK):
    b = min(a + CHUNK, D)
    rng = np.random.default_rng(SEED + a)          # chunk-local, reproducible
    h = rng.integers(0, D_PROJ, size=b - a)
    s = rng.integers(0, 2, size=b - a).astype(np.float64) * 2 - 1
    for i, m in enumerate(mms):
        x = np.asarray(m[a:b], dtype=np.float64) * s
        Y[i] += np.bincount(h, weights=x, minlength=D_PROJ)

norm = np.linalg.norm(Y, axis=1)
C_proj = (Y @ Y.T) / np.outer(norm, norm)
np.fill_diagonal(C_proj, np.nan)
C_ref = C_full.copy()
np.fill_diagonal(C_ref, np.nan)

sim_full = {g: {} for g in GROUPS}
sim_proj = {g: {} for g in GROUPS}
for g in GROUPS:
    idx = [i for i, q in enumerate(qids) if group_of.get(q) == g]
    if len(idx) < 2:
        continue
    sub_f = C_ref[np.ix_(idx, idx)]
    sub_p = C_proj[np.ix_(idx, idx)]
    sim_full[g] = np.nanmean(np.nanmean(sub_f, axis=1))
    sim_proj[g] = np.nanmean(np.nanmean(sub_p, axis=1))

pair_mask = ~np.isnan(C_ref)
abs_err = np.abs(C_proj - C_ref)[pair_mask]
rel_err = abs_err / np.abs(C_ref[pair_mask])

L = []
W = L.append
W("# Random-projection validation, on real gradients\n")
W(f"CountSketch, **seed {SEED}**, {D} dims -> **{D_PROJ}** buckets, over the "
  f"{n} natural-n prompts whose full-vector cosine matrix is cached. Storage per "
  f"prompt falls from {D * 2 / 1e6:.0f} MB to {D_PROJ * 4 / 1e6:.2f} MB "
  f"({D * 2 / (D_PROJ * 4):.0f}x).\n")
W("**Scope, stated:** the matched-K gradient directories were deleted after each "
  "run by design, so the matched-K curve itself cannot be recomputed under "
  "projection without re-running those GPU passes. This validates the same "
  "operation -- pairwise cosines between real per-prompt gradients -- on the "
  "natural-n set that survives.\n")
W("## Group means, full vectors vs projected\n")
W("| group | full-vector | projected | abs diff | rel diff |")
W("|---|---|---|---|---|")
for g in GROUPS:
    f_, p_ = sim_full[g], sim_proj[g]
    W(f"| {LABEL[g]} | {f_:.5f} | {p_:.5f} | {abs(p_ - f_):.5f} | "
      f"{100 * abs(p_ - f_) / abs(f_):.2f} % |")
W("")
W("## Ratios, which is what the argument rests on\n")
W("| pair | full-vector | projected |")
W("|---|---|---|")
for a, b in [("easy", "du71"), ("easy", "learnable"), ("learnable", "du71")]:
    W(f"| {LABEL[a]} / {LABEL[b]} | {sim_full[a] / sim_full[b]:.3f}x | "
      f"{sim_proj[a] / sim_proj[b]:.3f}x |")
W("")
W("## Per-pair error over every prompt pair\n")
W(f"- pairs compared: **{pair_mask.sum() // 2}**")
W(f"- max absolute error: **{abs_err.max():.5f}**")
W(f"- mean absolute error: {abs_err.mean():.5f}")
W(f"- max relative error: **{100 * rel_err.max():.2f} %**")
W(f"- median relative error: {100 * np.median(rel_err):.2f} %\n")

worst_group = max(abs(sim_proj[g] - sim_full[g]) / abs(sim_full[g]) for g in GROUPS)
ok = worst_group < 0.01
W("## Verdict\n")
if ok:
    W(f"**APPROVED.** Every group mean agrees to within "
      f"{100 * worst_group:.2f} % (< 1 %), and the ratios the argument rests on "
      "are preserved. The projection is registered as a method deviation with "
      f"this validation as its evidence, seed {SEED}, {D_PROJ} buckets.\n")
else:
    W(f"**NOT APPROVED.** Worst group-mean disagreement is "
      f"{100 * worst_group:.2f} %, above the 1 % gate. Raise D_PROJ or abandon "
      "the projection -- learned now at zero cost.\n")
W("Per-pair errors are larger than group-mean errors and that is expected: the "
  "sketch is unbiased per pair with variance ~1/D_PROJ, and averaging over a "
  "group cancels most of it. The curve uses group means, so the group-mean row "
  "is the one the gate is set on; the per-pair row is reported so the noise is "
  "not hidden.\n")

open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nwritten to {OUT}")
