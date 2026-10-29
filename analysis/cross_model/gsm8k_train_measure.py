"""A78 stage 2 — the MEASUREMENT, and the branch.

Reads the frozen prediction and passes 1 and 2. It never names the p3 file, so
no pass contributes to both sides. It reuses NOTHING from A74: w = 0.

DISCLOSURE 2, printed with the result and not separable from it: the prompt
count 3,274 was chosen using the offset OBSERVED in A74. Using the first
measurement to size the second is standard practice and it is still using the
measurement.

  $HOME/miniconda3/envs/vllm/bin/python gsm8k_train_measure.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results"))
AN = os.path.join(EXP, "analysis")
TAU, SEED, NBOOT = 0.1, 20260816, 2000
DELTA = 0.0345          # A74.4, unchanged. Derived, not re-chosen.
FLOOR = 100             # A74.6, unchanged.
N = 64

PRED = json.load(open(os.path.join(AN, "gsm8k_train_prediction.json")))


def load(tag):
    # '@', not '_'.
    h = sorted(glob.glob(os.path.join(RES, f"gsm8k_tr_{tag}_gsm8k_train_fresh@*.json")))
    assert len(h) == 1, f"expected one {tag} file, found {h}"
    per, inval = defaultdict(list), 0
    for r in json.load(open(h[0])):
        v = r.get("verification", {}) or {}
        per[int(r["question_id"])].append(
            int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]")
        inval += v.get("extracted_answer") == "[invalid]"
    return {q: np.array(v, bool) for q, v in per.items()}, inval, os.path.basename(h[0])


A, invA, fA = load("p1")
B, invB, fB = load("p2")
Q = sorted(set(A) & set(B))
thr = np.floor(TAU * N)
fa = np.array([A[q].sum() <= thr for q in Q])
fb = np.array([B[q].sum() <= thr for q in Q])
kA = np.array([A[q].sum() for q in Q])
kB = np.array([B[q].sum() for q in Q])


def jac(x, y):
    u = (x | y).sum()
    return (x & y).sum() / u if u else np.nan


m = jac(fa, fb)
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(fa), size=(NBOOT, len(fa)))
bs = np.array([jac(fa[i], fb[i]) for i in idx])
mlo, mhi = np.nanpercentile(bs, [2.5, 97.5])

union = int((fa | fb).sum())
inter = int((fa & fb).sum())
p_lo, p_hi = PRED["pred"] - DELTA, PRED["pred"] + DELTA
floor_ok = union >= FLOOR and PRED["pred_flag_size"] >= FLOOR

if not floor_ok:
    branch, why = "C", (f"flagged set below the floor: union {union}, predicted "
                        f"flag size {PRED['pred_flag_size']:.0f}, floor {FLOOR}")
elif mlo >= p_lo and mhi <= p_hi:
    branch, why = "A", "measured CI lies entirely within pred +/- delta-gsm"
elif mhi < p_lo or mlo > p_hi:
    branch, why = "B", "measured CI lies entirely outside pred +/- delta-gsm"
else:
    branch, why = "C", ("measured CI neither fits inside nor falls outside "
                        "pred +/- delta-gsm")

weak = PRED["lo"] <= m <= PRED["hi"]
out = {"experiment": "A78", "population": "gsm8k train, fresh",
       "w_reused_from_a74": 0.0,
       "sizing_disclosure": ("prompt count 3274 chosen using the offset observed "
                             "in A74 (+0.0050); standard practice, and still "
                             "using the measurement"),
       "branch": branch, "why": why, "measured": float(m),
       "mlo": float(mlo), "mhi": float(mhi), "delta_gsm": DELTA,
       "half_width": float((mhi - mlo) / 2),
       "pred": PRED["pred"], "pred_lo": PRED["lo"], "pred_hi": PRED["hi"],
       "equiv_lo": p_lo, "equiv_hi": p_hi, "err": PRED["pred"] - float(m),
       "flag1": int(fa.sum()), "flag2": int(fb.sum()), "union": union,
       "inter": inter, "floor": FLOOR, "floor_ok": bool(floor_ok),
       "a66_point_inside_predicted_interval": bool(weak),
       "n_prompts": len(Q), "eff_threshold": float(thr / N),
       "mean_p_p1": float(kA.mean() / N), "mean_p_p2": float(kB.mean() / N),
       "share_near_tau_p1": float(np.mean(np.abs(kA / N - TAU) <= 0.05)),
       "invalid_share_p1": float(invA / (len(Q) * N)),
       "invalid_share_p2": float(invB / (len(Q) * N)),
       "files": {"p1": fA, "p2": fB, "p3": PRED["source_file"]},
       "frozen_at": PRED["frozen_at"]}
json.dump(out, open(os.path.join(AN, "gsm8k_train_result.json"), "w"), indent=2)

print("A78 — SECOND GSM8K test, fresh train prompts, w = 0\n")
print("A74 IS NOT OVERWRITTEN BY THIS. A74 stands as reported: inconclusive on")
print("the 1,319-prompt test set, on precision. This is a different population.\n")
print(f"prediction FROZEN at {PRED['frozen_at']} from {PRED['source_file']} alone")
print(f"  pred {PRED['pred']:.4f} [{PRED['lo']:.4f}, {PRED['hi']:.4f}]  "
      f"RoE {PRED['roe']:.4f}   predicted flag size {PRED['pred_flag_size']:.0f}")
print(f"\nREGIME")
print(f"  mean pass rate      p1 {out['mean_p_p1']:.4f}  p2 {out['mean_p_p2']:.4f}  "
      f"p3 {PRED['mean_p']:.4f}")
print(f"  within +/-0.05 of tau                {out['share_near_tau_p1']*100:5.1f} %")
print(f"  prompts with zero correct (p3)       {PRED['share_p_zero']*100:5.1f} %")
print(f"  prompts with all correct (p3)        {PRED['share_p_one']*100:5.1f} %")
print(f"  invalid extractions   p1 {out['invalid_share_p1']*100:.2f} %  "
      f"p2 {out['invalid_share_p2']*100:.2f} %")
print(f"  effective threshold  floor({TAU}*{N})/{N} = {out['eff_threshold']:.4f}")
print(f"\nFLAGGED SETS")
print(f"  pass 1 {out['flag1']}   pass 2 {out['flag2']}   union {union}   "
      f"intersection {inter}   (floor {FLOOR}: {'MET' if floor_ok else 'NOT MET'})")
print(f"\nMEASURED  {m:.4f} [{mlo:.4f}, {mhi:.4f}]   half-width {out['half_width']:.4f}")
print(f"EQUIVALENCE REGION  pred +/- {DELTA} = [{p_lo:.4f}, {p_hi:.4f}]")
print(f"  err (pred - measured) {out['err']:+.4f}")
print(f"  A66 weaker convention (point inside predicted interval): {weak}")
print(f"\nBRANCH {branch} — {why}")
print(f"\nDISCLOSURE (A78, in the same sentence as the result): the decision to run")
print(f"  this test was taken AFTER an inconclusive first result, and the prompt")
print(f"  count 3,274 was chosen using the offset observed in that first test.")
if branch != "A":
    print("\nA74.10 CARRIES OVER: this may not be written as 'the rule failed on")
    print("  GSM8K', and 'four models, two datasets' stays unwritten.")
print(f"\nwrote {os.path.join(AN, 'gsm8k_train_result.json')}")
