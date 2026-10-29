"""Evaluate the prospectively frozen N=242 target-budget test (J=0.90 target).

Pass 1 = b2_exec1 (N=64, seed 101) + n242_topup_p1 (N=178, seed 301) -> N=242
Pass 2 = b2_exec2 (N=64, seed 202) + n242_topup_p2 (N=178, seed 402) -> N=242
"""
import datetime
import json
import os
from collections import defaultdict
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
FROZEN = os.path.join(EXP, "analysis", "budget_prospective_n242_prediction.frozen.json")
OUT = os.path.join(EXP, "analysis", "budget_prospective_n242_result.json")


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def load(files):
    per = defaultdict(list)
    for f in files:
        path = os.path.join(RES, f)
        with open(path) as fp:
            for r in json.load(fp):
                per[int(r["question_id"])].append(guarded(r.get("verification", {}) or {}))
    return {q: np.array(v, dtype=bool) for q, v in per.items()}


def main():
    with open(FROZEN) as f:
        pred = json.load(f)

    p1_files = [
        "b2_exec1_simplelr_qwen_level1to4_sub1k@train_0.7_64_1024_None_101_qwen.json",
        "n242_topup_p1_simplelr_qwen_level1to4_sub1k@train_0.7_178_1024_None_301_qwen.json",
    ]
    p2_files = [
        "b2_exec2_simplelr_qwen_level1to4_sub1k@train_0.7_64_1024_None_202_qwen.json",
        "n242_topup_p2_simplelr_qwen_level1to4_sub1k@train_0.7_178_1024_None_402_qwen.json",
    ]

    P1 = load(p1_files)
    P2 = load(p2_files)
    Q = sorted(set(P1) & set(P2))
    assert len(Q) == 1023, f"Expected 1023 prompts, got {len(Q)}"
    for q in Q:
        assert len(P1[q]) == 242, f"Prompt {q} Pass 1 has {len(P1[q])} rollouts != 242"
        assert len(P2[q]) == 242, f"Prompt {q} Pass 2 has {len(P2[q])} rollouts != 242"

    N = 242
    cut = int(np.floor(0.1 * N))
    k1 = np.array([P1[q].sum() for q in Q])
    k2 = np.array([P2[q].sum() for q in Q])
    f1 = k1 <= cut
    f2 = k2 <= cut
    inter = int((f1 & f2).sum())
    union = int((f1 | f2).sum())
    meas = float(inter / union)

    rng = np.random.default_rng(20260812)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(Q), len(Q))
        u = (f1[idx] | f2[idx]).sum()
        boots.append((f1[idx] & f2[idx]).sum() / u if u > 0 else 1.0)
    lo, hi = [float(x) for x in np.percentile(boots, [2.5, 97.5])]

    res = {
        "N": N,
        "cutoff_k": cut,
        "eff_threshold": cut / N,
        "measured": meas,
        "meas_lo": lo,
        "meas_hi": hi,
        "pred_roe": pred["pred_roe"],
        "pred_mc": pred["pred_mc"],
        "pred_lo": pred["pred_lo"],
        "pred_hi": pred["pred_hi"],
        "err_roe": pred["pred_roe"] - meas,
        "err_mc": pred["pred_mc"] - meas,
        "inside_pred_interval": bool(pred["pred_lo"] <= meas <= pred["pred_hi"]),
        "flag1": int(f1.sum()),
        "flag2": int(f2.sum()),
        "inter": inter,
        "union": union,
        "pred_size": pred["pred_size"],
        "mean_p1": float(k1.mean() / N),
        "mean_p2": float(k2.mean() / N),
        "frozen_at": pred["frozen_at"],
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
