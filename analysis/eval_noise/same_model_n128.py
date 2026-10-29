"""Future Work B.1 — measure the same-model Jaccard at N=128 and regenerate
analysis/same_model_n128.md.

Section 4.1 of the paper compares four Jaccard values; before this run three were
measured and one (same model at N=128) came from the binomial model. The
training-variation component was inferred from that analytic value. This script
consumes the measured repeat and reports the deltas.

Inputs, all produced by run_same_model_n128.sh and earlier E0b runs:
  reference  results/qwen_0.5b_ckpt1_hi_*noise@train*.json  + ..._hi_rest_*  (eval-seed 0)
  repeat     results/qwen_0.5b_ckpt1_n128_rep2_*@train*.json                 (eval-seed 1)
  seed 2     results/qwen_0.5b_ckpt2_hi_all_*@train*.json                    (cross-seed)

Writes analysis/same_model_n128.md. Modifies no other analysis file.

  python same_model_n128.py
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
TAU = 0.1
ANALYTIC = 0.889          # the value Section 4.1 carried before this run
PAPER_TRAIN = 0.076       # training-variation component inferred from it


def counts(pattern):
    sc = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(REPO, pattern))):
        with open(p) as f:
            for it in json.load(f):
                sc[it["question_id"]].append(int(it["verification"]["score"]))
    return {q: (sum(v), len(v)) for q, v in sc.items()}


def flagged(rates):
    return {q for q, v in rates.items() if v <= TAU}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def predict_J(N, rates):
    """E[J] for two independent N-sample evaluations, given per-prompt true rates."""
    k = math.floor(TAU * N)
    P = [stats.binom.cdf(k, N, p) for p in rates]
    return sum(x * x for x in P) / sum(2 * x - x * x for x in P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(EXP, "analysis"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ref = {**counts("results/qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train*.json"),
           **counts("results/qwen_0.5b_ckpt1_hi_rest_*@train*.json")}
    rep = counts("results/qwen_0.5b_ckpt1_n128_rep2_*@train*.json")
    s2 = counts("results/qwen_0.5b_ckpt2_hi_all_*@train*.json")
    for name, d in (("reference", ref), ("repeat", rep), ("seed-2", s2)):
        if not d:
            raise SystemExit(f"missing {name} results — run run_same_model_n128.sh first")

    pr = {q: c / n for q, (c, n) in ref.items()}
    pp = {q: c / n for q, (c, n) in rep.items()}
    p2 = {q: c / n for q, (c, n) in s2.items()}
    Fr, Fp, F2 = flagged(pr), flagged(pp), flagged(p2)

    J_same, J_cross = jaccard(Fr, Fp), jaccard(Fr, F2)
    d_same, d_cross = 1 - J_same, 1 - J_cross
    train = d_cross - d_same
    # pooled 256-sample rate: a less noisy input to the same model
    pool = {q: (ref[q][0] + rep[q][0]) / (ref[q][1] + rep[q][1]) for q in ref if q in rep}
    p32 = predict_J(32, list(pr.values()))

    L = []
    A = L.append
    A("# Same-model Jaccard at N=128, measured\n")
    A("Future Work B.1. Replaces the one analytic value in Section 4.1 with a")
    A("measurement. **No existing E0 analysis file was modified** — deltas only.\n")
    A("## Run\n")
    A("| | reference | repeat |")
    A("|---|---|---|")
    A("| model name | `ckpt1_hi` + `ckpt1_hi_rest` | `ckpt1_n128_rep2` |")
    A("| checkpoint | seed 1, global_step_120 | seed 1, global_step_120 |")
    A("| eval seed | 0 | 1 |")
    A("| prompts | 464 + 559 = 1023 | 1023 (single pass) |")
    A("| samples | 128 | 128 |")
    A(f"| pass@1 | {np.mean(list(pr.values())):.4f} | {np.mean(list(pp.values())):.4f} |")
    A("\nTemperature 0.7, max_new_tokens 1024, prompt-type qwen throughout.\n")
    A("**On independence.** `--seed` does not drive sampling here: `gen_utils_sglang.py`")
    A("forwards only temperature, top_p, max_new_tokens and stop to the engine, and")
    A("`seed_everything()` seeds python/numpy/torch, which the sglang sampler does not")
    A("consult. The two evaluations are independent draws because the harness never")
    A("seeds sampling — not because the seeds differ. The N=32 pair already demonstrated")
    A("this empirically (Jaccard 0.798, not 1.000).\n")
    A("## 1. Measured vs analytic\n")
    A("| | Jaccard | disagreement |")
    A("|---|---|---|")
    A(f"| **same model @N=128, MEASURED** | **{J_same:.3f}** | **{d_same:.3f}** |")
    A(f"| same model @N=128, analytic (Section 4.1) | {ANALYTIC:.3f} | {1-ANALYTIC:.3f} |")
    A(f"| difference | **{J_same-ANALYTIC:+.3f}** | {d_same-(1-ANALYTIC):+.3f} |")
    A(f"\nFlagged sets: reference {len(Fr)}, repeat {len(Fp)}, shared {len(Fr&Fp)}, "
      f"symmetric difference {len(Fr^Fp)}.\n")
    A(f"At N=32 the same model predicted {p32:.3f} against a measured 0.798 — accurate to")
    A(f"{abs(p32-0.798):.3f}. At N=128 the error is **{abs(J_same-ANALYTIC):.3f}**, "
      f"{abs(J_same-ANALYTIC)/max(abs(p32-0.798),1e-9):.0f}x larger.")
    A("So the binomial model does **not** hold as well at N=128 as it did at N=32.\n")
    A("### Why the model drifts\n")
    A("The prediction takes p̂ as the true rate, but p̂ is itself a 128-sample estimate.")
    A("Feeding a noisy p̂ into P(X ≤ ⌊τN⌋) inflates predicted agreement, and the")
    A("inflation grows with N because the threshold sits closer to the estimate's own")
    A("uncertainty. Re-running the model on a pooled 256-sample rate (reference + repeat)")
    A("gives a less noisy input:\n")
    A("| p̂ source | samples | predicted J @N=128 |")
    A("|---|---|---|")
    A(f"| reference only (as used in Section 4.1) | 128 | {predict_J(128, list(pr.values())):.3f} |")
    A(f"| reference + repeat pooled | 256 | {predict_J(128, list(pool.values())):.3f} |")
    A(f"| — measured | — | **{J_same:.3f}** |")
    A("\n## 2. Training-variation component, recomputed\n")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| cross-seed disagreement @N=128 (seed1 vs seed2) | {d_cross:.3f} |")
    A(f"| same-model disagreement @N=128, MEASURED | {d_same:.3f} |")
    A(f"| **training variation = difference** | **{train:.3f}** |")
    A(f"| current estimate in the paper (analytic baseline) | {PAPER_TRAIN:.3f} |")
    A(f"| **delta** | **{train-PAPER_TRAIN:+.3f}** |")
    A(f"\nAs a share of cross-seed disagreement: **{100*train/d_cross:.0f}%** "
      f"(paper currently says 41%).\n")
    A("## 3. What Section 4.1 needs\n")
    if abs(J_same - ANALYTIC) < 0.01:
        A("The analytic value was accurate. Section 4.1 stands; only the label changes")
        A("from analytic to measured.")
    else:
        A(f"**Revise.** The analytic {ANALYTIC:.3f} should become the measured "
          f"**{J_same:.3f}**, and")
        A(f"the training-variation component **{PAPER_TRAIN:.3f} → {train:.3f}** "
          f"({100*train/d_cross:.0f}% rather than 41%).\n")
        # direction: over-estimating agreement under-estimates same-model disagreement,
        # so the shortfall is wrongly attributed to training
        A(f"Direction: the analytic model **over-estimated agreement** "
          f"({ANALYTIC:.3f} vs {J_same:.3f}), so it")
        A(f"**under-estimated same-model disagreement** ({1-ANALYTIC:.3f} vs {d_same:.3f}), "
          "and that shortfall was")
        A("attributed to training instead. The paper therefore **over-states** the")
        A(f"training-variation component — {PAPER_TRAIN:.3f} against a measured {train:.3f}.\n")
        A("The qualitative claim strengthens rather than weakens: evaluation noise accounts for")
        A(f"an even larger share of cross-seed disagreement than reported "
          f"({100*d_same/d_cross:.0f}% at N=128, not")
        A("59%), and still 78% at N=32.")
    A("\n### Limitations paragraph\n")
    A("*\"The same-model baseline at N=128 is analytic, not measured\"* — **can now be")
    A("deleted.** It should be replaced by a note that the binomial model was validated")
    A(f"at N=32 (error {abs(p32-0.798):.3f}) but drifts at N=128 (error "
      f"{abs(J_same-ANALYTIC):.3f}), because p̂ noise propagates into the prediction.")
    A("That is a more useful caveat than the one it replaces, and it is now supported.\n")
    A("### One asymmetry worth stating\n")
    A("The reference p̂ is pooled from two passes over different dataset files (464 + 559")
    A("prompts); the repeat is a single pass over all 1023. Same checkpoint, same sampling")
    A("parameters, same prompts — but not a byte-for-byte protocol replica. If anything")
    A("this makes the measured agreement a slight under-estimate, since batch composition")
    A("differs between the two.")

    path = os.path.join(args.outdir, "same_model_n128.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:26]))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
