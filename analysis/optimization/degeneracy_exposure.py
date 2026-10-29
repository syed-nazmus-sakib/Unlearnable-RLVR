"""A51 -- how much sympy-pathological degeneracy the TRAINED checkpoint emits.

The E5 gate's verification leg is slow because an UNTRAINED base 1.5B emits
degenerate repetition (`E*E*E*...`, `factorial(k)*factorial(k)*...`) which
math_verify hands to sympy's polynomial factoriser, where it blows up. The
5120 re-run uses the TRAINED 0.5B checkpoint. This measures whether that
exposure is real there rather than assuming it either way.

DETECTION RULES, FIXED HERE BEFORE ANY NUMBER WAS LOOKED AT (A29/A40
discipline). Both are applied to the FULL response text, because
math_verifier.evaluate:67 passes `model_completion` -- the whole response --
to _verify_with_math_verify, not the extracted boxed answer.

  R1 TARGETED. >=20 consecutive repeats of the same multiplicative factor,
     i.e. the exact shape observed blowing up in logs_gate.out. Regex below.
  R2 GENERIC.  some 12-character substring occurs >=30 times in the response.
     Catches degeneration that is not multiplicative. O(n) by counting
     12-grams, so no regex backtracking risk.

R1 is the mechanism; R2 is the wider net. Reported separately, never merged.

THE DIRECT MEASUREMENT, which is what the re-cost actually needs: the real
`evaluate` from inference/math_verifier is timed on a random sample of stored
responses. Per-response wall time summed IS the serial verification cost. No
proxy, no assumption about which pattern is expensive.

  $HOME/miniconda3/envs/vllm/bin/python degeneracy_exposure.py
"""
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
RES = os.path.join(REPO, "results")
sys.path.insert(0, os.path.join(REPO, "inference"))
from math_verifier import evaluate  # noqa: E402  the real one, unmodified

SEED, NSAMPLE, WORKERS = 20260812, 6000, 10
K = 12          # R2 substring length
R2_MIN = 30     # R2 repeat count
R1 = re.compile(r"([A-Za-z_][A-Za-z0-9_]{0,15}(?:\([^()]{0,20}\))?)\*(?:\1\*){19,}")

# name -> (files, what it is). The trained checkpoint is the 5120 run's model.
COHORTS = {
    "trained ckpt1 (N=128)": (
        ["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
         "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"],
        "the model the 5120 run will use"),
    "untrained base 0.5B (N=32)": (
        ["qwen_0.5b_initial_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json"],
        "same family, no RL -- the base-model control"),
    "Llama-3B instruct (N=32)": (
        ["e4_gate_llama3b_v2_e4_screen800_v2@test_0.7_32_1024_None_0_e4v2.json"],
        "instruction-tuned, a second trained-model point"),
    # A51.2: R1 has fired ZERO times in every cohort above, so it has been shown
    # only not to fire. The blowups were observed live on THIS cohort, whose
    # output is written only when the gate finishes. Skipped until then; this is
    # the run that decides whether R1 detects anything at all.
    "untrained base 1.5B (N=32)": (
        ["e5_gate_qwen15b_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json"],
        "the cohort whose verification blew up -- R1's only real test"),
}


def max_kgram_repeat(s, k=K):
    if len(s) < k:
        return 0
    c = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    return c.most_common(1)[0][1]


def flags(resp):
    return bool(R1.search(resp)), max_kgram_repeat(resp) >= R2_MIN


def _timed(args):
    """(seconds, r1, r2, chars) for one response through the REAL verifier."""
    gt, resp = args
    t0 = time.perf_counter()
    try:
        evaluate(answers=[gt], completions=[resp])
    except Exception:
        pass
    dt = time.perf_counter() - t0
    a, b = flags(resp)
    return dt, a, b, len(resp)


def load(files):
    out = []
    for f in files:
        for r in json.load(open(os.path.join(RES, f))):
            out.append((r.get("ground_truth") or "", r["response"]))
    return out


rng = np.random.default_rng(SEED)
report, rows = [], {}
for name, (files, what) in COHORTS.items():
    if not all(os.path.exists(os.path.join(RES, f)) for f in files):
        print(f"SKIP {name}: missing file")
        continue
    data = load(files)
    n = len(data)

    # R1/R2 over EVERY response in the cohort, not a sample.
    f1 = f2 = 0
    for _gt, resp in data:
        a, b = flags(resp)
        f1 += a
        f2 += b

    idx = rng.choice(n, size=min(NSAMPLE, n), replace=False)
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_timed, [data[i] for i in idx], chunksize=16))
    t = np.array([r[0] for r in res])

    rows[name] = {
        "what": what, "n_responses": n,
        "r1_n": int(f1), "r1_frac": f1 / n,
        "r2_n": int(f2), "r2_frac": f2 / n,
        "sample": int(len(t)),
        "mean_s": float(t.mean()), "median_s": float(np.median(t)),
        "p99_s": float(np.percentile(t, 99)), "max_s": float(t.max()),
        "slow_1s_frac": float((t > 1.0).mean()),
        "slow_share_of_time": float(t[t > 1.0].sum() / t.sum()) if (t > 1.0).any() else 0.0,
        "est_serial_s_per_1e5": float(t.mean() * 1e5),
        "mean_chars": float(np.mean([r[3] for r in res])),
    }
    print(f"{name}: R1 {f1}/{n} = {f1/n:.5f} | R2 {f2}/{n} = {f2/n:.5f} | "
          f"mean {t.mean()*1000:.2f} ms | max {t.max():.2f} s | "
          f">1s {(t>1.0).mean():.5f}")

json.dump({"seed": SEED, "n_sample": NSAMPLE, "k": K, "r2_min": R2_MIN,
           "r1_regex": R1.pattern, "cohorts": rows},
          open(os.path.join(EXP, "analysis", "degeneracy_exposure.json"), "w"),
          indent=2)
print("\nwrote analysis/degeneracy_exposure.json")
