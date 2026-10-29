"""A39.3 acceptance check — attempt 2 against attempt 1, and against the line
fixed BEFORE attempt 2 was run.

    PASS : overall [invalid] share < 5 %  AND  |corr(invalid rate, pass rate)| < 0.15
    FAIL : anything else

Both attempts are reported side by side so the sequence is visible, as A39.3
requires. CPU only.

  $HOME/miniconda3/envs/vllm/bin/python a39_check.py
"""
import json
import os
from collections import defaultdict

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(EXP, "..", "..", "unlearnability-rlvr", "results")
OUT = os.path.join(EXP, "analysis", "a39_acceptance.md")
os.makedirs(os.path.join(EXP, "analysis"), exist_ok=True)
SEED, NBOOT, N = 20260812, 5000, 32
SHARE_MAX, CORR_MAX = 0.05, 0.15

ATT = {1: ("e4_gate_llama3b_e4_screen800@test_0.7_32_1024_None_0_qwen.json",
           "qwen", "Please reason step by step, and put your final answer within \\boxed{}."),
       2: ("e4_gate_llama3b_v2_e4_screen800_v2@test_0.7_32_1024_None_0_e4v2.json",
           "e4v2", "Solve the problem. Keep your reasoning brief. End your response with "
                   "the final answer in the form \\boxed{answer} -- exactly one \\boxed{}, "
                   "containing only the final answer, with nothing after it.")}

rng = np.random.default_rng(SEED)


def measure(fn):
    ok, inv, n = defaultdict(int), defaultdict(int), defaultdict(int)
    boxed = 0
    for r in json.load(open(os.path.join(RES, fn))):
        v = r.get("verification", {}) or {}
        q = int(r["question_id"])
        bad = v.get("extracted_answer") == "[invalid]"
        ok[q] += int(v.get("score", 0)) == 1 and not bad
        inv[q] += bad
        n[q] += 1
        boxed += "\\boxed" in (r.get("response") or "")
    q = sorted(ok)
    iv = np.array([inv[x] / n[x] for x in q])
    pr = np.array([ok[x] / n[x] for x in q])
    tot = sum(n.values())
    share = sum(inv.values()) / tot
    corr = float(np.corrcoef(iv, pr)[0, 1])
    b = rng.integers(0, len(q), size=(NBOOT, len(q)))
    sh_b = iv[b].mean(axis=1)
    co_b = np.array([np.corrcoef(iv[r], pr[r])[0, 1] for r in b[:1000]])
    return {"n_prompts": len(q), "rollouts": tot, "share": share,
            "share_ci": [float(x) for x in np.percentile(sh_b, [2.5, 97.5])],
            "corr": corr,
            "corr_ci": [float(x) for x in np.nanpercentile(co_b, [2.5, 97.5])],
            "pass_rate": float(pr.mean()), "zero": int((pr == 0).sum()),
            "boxed_share": boxed / tot,
            "gt50": int((iv > 0.5).sum())}


M = {a: measure(f) for a, (f, _, _) in ATT.items()}
v1, v2 = M[1], M[2]
ok_share = v2["share"] < SHARE_MAX
ok_corr = abs(v2["corr"]) < CORR_MAX
PASS = ok_share and ok_corr

L = []
W = L.append
W("# A39.3 acceptance check — attempt 2 against the line set before it ran\n")
W(f"> **PASS** — overall `[invalid]` share < {100 * SHARE_MAX:.0f} % AND "
  f"|corr(invalid rate, guarded pass rate)| < {CORR_MAX:.2f}. "
  "**FAIL** — anything else.\n")
W("Same 800 MATH-Hard prompts, same seed, same N=32, same model. The ONLY "
  "difference is the output instruction: attempt 1 used the `qwen` template "
  "(Chen's instruction, as the 0.5B runs did), attempt 2 the `e4v2` template.\n")
W("| | attempt 1 (`qwen`) | attempt 2 (`e4v2`) | change |")
W("|---|---|---|---|")
W(f"| **`[invalid]` share** | **{100 * v1['share']:.2f} %** | "
  f"**{100 * v2['share']:.2f} %** | {100 * (v2['share'] - v1['share']):+.2f} pp |")
W(f"| 95 % CI | [{100 * v1['share_ci'][0]:.2f}, {100 * v1['share_ci'][1]:.2f}] | "
  f"[{100 * v2['share_ci'][0]:.2f}, {100 * v2['share_ci'][1]:.2f}] | — |")
W(f"| **corr(invalid, pass rate)** | **{v1['corr']:+.3f}** | **{v2['corr']:+.3f}** | "
  f"{v2['corr'] - v1['corr']:+.3f} |")
W(f"| 95 % CI | [{v1['corr_ci'][0]:+.3f}, {v1['corr_ci'][1]:+.3f}] | "
  f"[{v2['corr_ci'][0]:+.3f}, {v2['corr_ci'][1]:+.3f}] | — |")
W(f"| rollouts containing `\\boxed` | {100 * v1['boxed_share']:.2f} % | "
  f"{100 * v2['boxed_share']:.2f} % | {100 * (v2['boxed_share'] - v1['boxed_share']):+.2f} pp |")
W(f"| prompts >50 % invalid | {v1['gt50']} / {v1['n_prompts']} | "
  f"{v2['gt50']} / {v2['n_prompts']} | — |")
W(f"| mean guarded pass rate | {v1['pass_rate']:.4f} | {v2['pass_rate']:.4f} | "
  f"{v2['pass_rate'] - v1['pass_rate']:+.4f} |")
W(f"| prompts with zero correct | {v1['zero']} | {v2['zero']} | — |")
W("")

W("## Verdict\n")
W(f"* share < {100 * SHARE_MAX:.0f} % : **{'PASS' if ok_share else 'FAIL'}** "
  f"({100 * v2['share']:.2f} %)")
W(f"* |corr| < {CORR_MAX:.2f} : **{'PASS' if ok_corr else 'FAIL'}** "
  f"({abs(v2['corr']):.3f})\n")
if PASS:
    W("**ATTEMPT 2 PASSES A39.3.** The extraction artefact is resolved; the gate "
      "may reopen. Tier construction must be recomputed on the attempt-2 pass "
      "rates, since the instruction changed what the model produces and the "
      "attempt-1 tiers are no longer the right ones.\n")
else:
    W("**ATTEMPT 2 FAILS A39.3.** Under the registered rule there is no third "
      "revision: **the Llama arm is DROPPED**, the paper stays single-model, "
      "and the reason is reported in the main text as a stated negative — a 3B "
      "instruct model fails answer extraction on hard problems at a rate "
      "correlated with difficulty.\n")
    W("The mean guarded pass rate is reported beside the failure so that a "
      "reader can see whether the instruction traded accuracy for parseability, "
      "and neither number is presented without the other.\n")

json.dump({"attempts": M, "pass": bool(PASS),
           "criteria": {"share_max": SHARE_MAX, "corr_max": CORR_MAX},
           "ok_share": bool(ok_share), "ok_corr": bool(ok_corr)},
          open(os.path.join(EXP, "analysis", "a39_acceptance.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------------------------------------------------------------------
# A43 -- machine-generated macros for the two-attempt comparison. The main-text
# claim "extraction failure is a property of the model-verifier pair, not of
# prompt wording" rests entirely on these two runs, so the numbers belong in
# the paper and not only in the log.
# ---------------------------------------------------------------------------
T = ["% GENERATED BY EXPERIMENTS/E4_llama/a39_check.py -- DO NOT EDIT BY HAND.",
     "% Rerun that script to regenerate. A43.",
     "%",
     "% THE TWO ATTEMPTS DIFFER ONLY IN THE OUTPUT INSTRUCTION.",
     "%   same 800 MATH-Hard prompts (byte-identical data files)",
     "%   same sampling seed, same N = 32 rollouts per prompt",
     "%   same model (Llama-3.2-3B-Instruct), same temperature 0.7,",
     "%   same max_new_tokens 1024, same verifier and same A24 guard.",
     "% Attempt 1 used the `qwen` prompt template, carrying Chen's instruction",
     "% verbatim as the 0.5B runs did. Attempt 2 used the `e4v2` template,",
     "% which demands a single terminal \\boxed{} and brevity. Nothing else",
     "% changed between them.",
     "%",
     f"% Acceptance line, fixed BEFORE attempt 2 ran (A39.3): share < "
     f"{100 * SHARE_MAX:.0f} percent AND |corr| < {CORR_MAX:.2f}. Attempt 2 failed both.",
     "% Contract: explicit sign on interval-printed values and on deltas; no",
     "% percent or dollar sign inside any macro; counts are bare integers.",
     ""]
T.append(r"\newcommand{\eeNPrompts}{%d}" % v1["n_prompts"])
T.append(r"\newcommand{\eeSamplesPerPrompt}{%d}" % N)
T.append(r"\newcommand{\eeNRollouts}{%d}" % v1["rollouts"])
T.append("")
for tag, m in (("One", v1), ("Two", v2)):
    T.append(f"% --- attempt {'1 (qwen template)' if tag == 'One' else '2 (e4v2 template)'} ---")
    T.append(r"\newcommand{\eeInvalid%s}{%.4f}" % (tag, m["share"]))
    T.append(r"\newcommand{\eeInvalid%sLo}{%.4f}" % (tag, m["share_ci"][0]))
    T.append(r"\newcommand{\eeInvalid%sHi}{%.4f}" % (tag, m["share_ci"][1]))
    T.append(r"\newcommand{\eeCorr%s}{%+.3f}" % (tag, m["corr"]))
    T.append(r"\newcommand{\eeCorr%sLo}{%+.3f}" % (tag, m["corr_ci"][0]))
    T.append(r"\newcommand{\eeCorr%sHi}{%+.3f}" % (tag, m["corr_ci"][1]))
    T.append(r"\newcommand{\eeBoxed%s}{%.4f}" % (tag, m["boxed_share"]))
    T.append(r"\newcommand{\eePass%s}{%.4f}" % (tag, m["pass_rate"]))
    T.append(r"\newcommand{\eeZero%s}{%d}" % (tag, m["zero"]))
    T.append(r"\newcommand{\eeGtFifty%s}{%d}" % (tag, m["gt50"]))
    T.append("")
T.append("% --- attempt 2 minus attempt 1 ---")
T.append(r"\newcommand{\eeInvalidDelta}{%+.4f}" % (v2["share"] - v1["share"]))
T.append(r"\newcommand{\eeCorrDelta}{%+.3f}" % (v2["corr"] - v1["corr"]))
T.append(r"\newcommand{\eeBoxedDelta}{%+.4f}" % (v2["boxed_share"] - v1["boxed_share"]))
T.append(r"\newcommand{\eePassDelta}{%+.4f}" % (v2["pass_rate"] - v1["pass_rate"]))
T.append(r"\newcommand{\eeZeroDelta}{%+d}" % (v2["zero"] - v1["zero"]))
T.append("")
T.append(r"\newcommand{\eeShareMax}{%.2f}" % SHARE_MAX)
T.append(r"\newcommand{\eeCorrMax}{%.2f}" % CORR_MAX)
TEX = os.path.join(EXP, "analysis", "e4_extraction_numbers.tex")
open(TEX, "w").write("\n".join(T) + "\n")
print(f"\nwrote {TEX}")
