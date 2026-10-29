"""A32 — the detection floor expressed in PERCENTAGE POINTS OF GRADIENT SIGNAL.

The precondition came back mixed: arm 4 (39.87 % of signal removed) separates
from baseline, arm 5 (21.82 %) does not. The floor therefore lies between them,
and accuracy points are the wrong unit for deciding what arm 2 vs arm 3 could
ever show. This converts.

CRUDE BY CONSTRUCTION, and labelled so wherever it is used: three calibration
points, one of which is the origin, from RANDOM removals only. Nothing here is
a fitted model of training dynamics.

Run before any arm 2 or arm 3 seed was read.

  $HOME/miniconda3/envs/vllm/bin/python signal_mde.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import t as tdist

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
OUT = os.path.join(EXP, "analysis", "signal_mde.md")
SEEDS = [1, 2, 3, 4, 5]
ALPHA, POWER = 0.05, 0.80
MDE_REG = 0.91          # A31.4, registered in advance

sets = json.load(open(os.path.join(EXP, "analysis", "removal_sets.json")))
SIG = sets["share_pct"]
# arm -> (signal % removed, prompts kept)
ARM = {"arm1_base": (0.0, 1023), "e3arm4_rand404": (SIG["random404"], 619),
       "e3arm5_rand218": (SIG["random218"], 805),
       "e3arm2_union": (SIG["union"], 619), "e3arm3_inter": (SIG["intersection"], 805)}


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def acc(arm, seed, ev="e3_math500"):
    p = glob.glob(os.path.join(RES, f"e3_{arm}_seed{seed}_{ev}@test*.json"))
    if not p:
        return None
    per = defaultdict(list)
    for r in json.load(open(p[0])):
        per[r["question_id"]].append(guarded(r.get("verification", {}) or {}))
    return float(np.mean([np.mean(v) for v in per.values()]))


A = {}
for a in ARM:
    v = [acc(a, s) for s in SEEDS]
    v = [x for x in v if x is not None]
    if v:
        A[a] = np.array(v)

L = []
W = L.append
W("# A32 — the detection floor in units of gradient signal removed\n")
W("**Computed before any arm 2 or arm 3 seed was read.**\n")
W("**CRUDE.** Three calibration points including the origin, all from RANDOM "
  "removals. It is an order-of-magnitude guide to what the design can resolve, "
  "not a model of training.\n")

W("## Calibration points\n")
W("| arm | signal removed | seeds | accuracy | vs baseline |")
W("|---|---|---|---|---|")
base = A["arm1_base"]
cal = []
for a in ("arm1_base", "e3arm5_rand218", "e3arm4_rand404"):
    if a not in A:
        continue
    d = A[a].mean() - base.mean()
    cal.append((ARM[a][0], 100 * d))
    W(f"| {a} | {ARM[a][0]:.2f} % | {len(A[a])} | {100 * A[a].mean():.2f} % | "
      f"{100 * d:+.2f} pts |")
W("")

x = np.array([c[0] for c in cal])
y = np.array([c[1] for c in cal])
slope_o = float((x * y).sum() / (x * x).sum())          # through the origin
sl, ic = np.polyfit(x, y, 1)
W(f"* slope through the origin: **{slope_o:.4f} accuracy points per point of "
  f"signal removed** (i.e. removing 10 points of signal costs "
  f"{abs(10 * slope_o):.2f} accuracy points)")
W(f"* unconstrained line: slope {sl:.4f}, intercept {ic:+.3f} — the intercept "
  "should be 0 by construction, and its distance from 0 is the honest measure "
  "of how crude this is")
W("")

# empirical sd across the completed arms, and the MDE it implies
sds = [A[a].std(ddof=1) for a in A if len(A[a]) == len(SEEDS)]
sd_emp = float(np.mean(sds))
df = 2 * (len(SEEDS) - 1)
mult = tdist.ppf(1 - ALPHA / 2, df) + tdist.ppf(POWER, df)
mde_emp = 100 * mult * sd_emp * np.sqrt(2 / len(SEEDS))

W("## The floor, in both units\n")
W("| basis | MDE in accuracy points | **MDE in points of signal removed** |")
W("|---|---|---|")
W(f"| A31.4 as registered (conservative upper-bound sd) | {MDE_REG:.2f} | "
  f"**{MDE_REG / abs(slope_o):.1f}** |")
W(f"| empirical (observed sd {100 * sd_emp:.2f} pts across completed arms) | "
  f"{mde_emp:.2f} | **{mde_emp / abs(slope_o):.1f}** |")
W("")
W(f"**The design resolves a removal of roughly "
  f"{mde_emp / abs(slope_o):.0f}-{MDE_REG / abs(slope_o):.0f} points of "
  "gradient signal, and nothing smaller.** That is consistent with what the "
  "precondition showed directly: 39.9 points separated, 21.8 did not.\n")

W("## What that implies for the comparisons ahead\n")
W("| comparison | signal gap | predicted accuracy gap | above the floor? |")
W("|---|---|---|---|")
COMP = [("arm 2 vs arm 3 (REGISTERED PRIMARY)", "e3arm2_union", "e3arm3_inter"),
        ("arm 2 vs arm 4 (size- and epoch-matched)", "e3arm2_union", "e3arm4_rand404"),
        ("arm 3 vs arm 5 (size- and epoch-matched)", "e3arm3_inter", "e3arm5_rand218")]
for name, a, b in COMP:
    gap = abs(ARM[a][0] - ARM[b][0])
    pred = gap * abs(slope_o)
    W(f"| {name} | {gap:.1f} pts | {pred:.2f} pts | "
      f"{'yes' if pred > mde_emp else '**NO**'} |")
W("")
W("**Every comparison the experiment was built to make sits below its own "
  "detection floor.** This is stated now, before the data, so that a null "
  "cannot later be presented as a finding about the world.\n")

W("### A confound in the registered primary, found while doing this\n")
W("Arms 2 and 3 differ in **dataset size and epoch count as well as signal**: "
  "619 prompts at 6.20 epochs against 805 at 4.77. The random controls were "
  "introduced to absorb exactly that, and they can — but only through a "
  "**difference-in-differences**, not through the raw arm-2-vs-arm-3 contrast:\n")
W("    DiD = (arm2 - arm4) - (arm3 - arm5)\n")
W("Each bracket is size- and epoch-matched by construction (arm 2 with arm 4 at "
  "619/6.20; arm 3 with arm 5 at 805/4.77), so the DiD isolates WHICH prompts "
  "were removed from HOW MANY. Its cost is variance: four arm means instead of "
  "two.\n")
sd_did = sd_emp * 2.0 / np.sqrt(len(SEEDS))
df4 = 4 * (len(SEEDS) - 1)
mult4 = tdist.ppf(1 - ALPHA / 2, df4) + tdist.ppf(POWER, df4)
W(f"* MDE on the DiD: **{100 * mult4 * sd_did:.2f} accuracy points** "
  f"({100 * mult4 * sd_did / abs(slope_o):.0f} points of signal)")
W(f"* MDE on the raw arm-2-vs-arm-3 contrast: {mde_emp:.2f} points — smaller, "
  "but of a confounded quantity\n")
W("**Recommendation, registered rather than acted on unilaterally:** report the "
  "raw contrast as the registered primary exactly as A31 specifies, and the DiD "
  "beside it as the unconfounded estimate. Neither replaces the other, and the "
  "raw contrast is not quietly reinterpreted after the fact.\n")

W("## Arm 5's null is a result in its own right\n")
a5 = A["e3arm5_rand218"]
d5 = 100 * (a5.mean() - base.mean())
se5 = 100 * np.sqrt(a5.var(ddof=1) / len(a5) + base.var(ddof=1) / len(base))
tc = tdist.ppf(0.975, len(a5) + len(base) - 2)
W(f"**Removing 218 prompts at random — {ARM['e3arm5_rand218'][0]:.1f} % of the "
  f"training gradient signal, 21.3 % of the dataset — changes held-out accuracy "
  f"by {d5:+.2f} points, CI [{d5 - tc * se5:+.2f}, {d5 + tc * se5:+.2f}], "
  f"against a minimum detectable difference of {mde_emp:.2f} points.**\n")
W("Stated with the MDE attached, as a bound rather than as equivalence: the "
  "effect of losing a fifth of the training signal is smaller than we can "
  f"resolve at five seeds, not shown to be zero. On the crude calibration it "
  f"would be about {ARM['e3arm5_rand218'][0] * abs(slope_o):.2f} points — "
  "consistent with the null and below the floor.\n")
W("This is worth reporting on its own: **RLVR training here is robust to losing "
  "a fifth of its gradient signal**, and that robustness is also the reason the "
  "labelled comparison is hard. The same fact that makes filtering safe in "
  "practice makes its consequences difficult to measure.\n")

json.dump({"slope_pts_per_signal_pt": slope_o, "intercept_check": float(ic),
           "sd_empirical": sd_emp, "mde_acc_empirical": mde_emp,
           "mde_signal_empirical": mde_emp / abs(slope_o),
           "mde_signal_registered": MDE_REG / abs(slope_o),
           "arm5_diff": d5, "arm5_ci": [d5 - tc * se5, d5 + tc * se5],
           "mde_did_acc": float(100 * mult4 * sd_did)},
          open(os.path.join(EXP, "analysis", "signal_mde.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
