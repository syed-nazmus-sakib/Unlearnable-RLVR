"""A80 step one — is any second model viable for the SAMPLE-SIZE CONFOUND test?

CPU only. Reads the four existing N=32, cap-1024 screening passes and answers
four questions per model, with no training run and no Chen labels:

  1. how many prompts are K-eligible at K = 4, 6, 8, 11
  2. the correct-rollout count IMBALANCE between the hard and easy groups,
     which is the quantity the confound is about
  3. the truncation contamination in each pool at the 1024 cap
  4. the same for Qwen2.5-1.5B, whose K=11 pool died at 16 prompts

GROUP DEFINITION, AND WHY tau = 0.1 CANNOT BE USED HERE.
The paper's threshold flags a prompt at <= floor(0.1*32) = 3 correct. A prompt
with <= 3 correct rollouts is K-ineligible at every K we care about, so a
tau-defined hard group has EXACTLY ZERO eligible members by construction. The
threshold has to be raised or there is no test. Two are reported:

  PRIMARY   median split of the model's own initial pass rate. Model-agnostic,
            label-free, pre-specifiable without looking at k, and it maximises
            both group sizes -- which is the binding constraint here.
  SENSITIVITY  bottom quartile against top quartile. More contrast, fewer
            eligible prompts. Reported so the choice is visible, not hidden.

THE CEILING THAT MATTERS. A K-eligible hard prompt has k >= K by definition, so
the attainable imbalance is bounded above by N/K: 8.0 at K=4 but only 2.9 at
K=11. Raising K to reduce estimator noise MECHANICALLY COMPRESSES THE VERY
QUANTITY THE TEST IS ABOUT. That ceiling is reported beside every measured
ratio.

  $HOME/miniconda3/envs/vllm/bin/python second_model_screen.py
"""
import json
import os
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
RES = os.path.join(ROOT, "unlearnability-rlvr", "results")
AN = os.path.join(EXP, "analysis")
CAP, KS = 1024, [4, 6, 8, 11]

MODELS = {
    "Qwen2.5-0.5B ckpt1 (reference)": (
        "qwen_0.5b_run1_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json",
        os.path.join(ROOT, "models", "Qwen2.5-0.5B")),
    "Llama-3.2-3B (screen800)": (
        "e4_gate_llama3b_e4_screen800@test_0.7_32_1024_None_0_qwen.json",
        os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct")),
    "Llama-3.2-3B (screen800 v2)": (
        "e4_gate_llama3b_v2_e4_screen800_v2@test_0.7_32_1024_None_0_e4v2.json",
        os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct")),
    "Qwen2.5-1.5B base (E5 gate)": (
        "e5_gate_qwen15b_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json",
        os.path.join(ROOT, "models", "Qwen2.5-1.5B")),
}


def guarded(v):
    return int(v.get("score", 0)) == 1 and v.get("extracted_answer") != "[invalid]"


def screen(fname, tok_path):
    tok = AutoTokenizer.from_pretrained(tok_path)
    k = defaultdict(int)          # guarded-correct rollouts
    n = defaultdict(int)
    tr_all = defaultdict(int)     # truncated rollouts, any
    tr_ok = defaultdict(int)      # truncated AND guarded-correct
    texts, meta = [], []
    for r in json.load(open(os.path.join(RES, fname))):
        q = int(r["question_id"])
        v = r.get("verification", {}) or {}
        g = guarded(v)
        k[q] += g
        n[q] += 1
        texts.append(r["response"])
        meta.append((q, g))
    lens = np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])
    for (q, g), L in zip(meta, lens):
        t = int(L >= CAP)
        tr_all[q] += t
        tr_ok[q] += t and g
    Q = sorted(k)
    return {
        "Q": Q,
        "k": np.array([k[q] for q in Q]),
        "n": np.array([n[q] for q in Q]),
        "tr_all": np.array([tr_all[q] for q in Q]),
        "tr_ok": np.array([tr_ok[q] for q in Q]),
    }


def analyse(d):
    k, n = d["k"], d["n"]
    N = int(np.median(n))
    p = k / n
    out = {"n_prompts": len(k), "N": N, "mean_p": float(p.mean()),
           "trunc_overall": float(d["tr_all"].sum() / n.sum()),
           "trunc_among_correct": float(d["tr_ok"].sum() / max(k.sum(), 1)),
           "splits": {}}

    for name, (hard_m, easy_m) in {
        "median": (p <= np.median(p), p > np.median(p)),
        "quartile": (p <= np.quantile(p, 0.25), p >= np.quantile(p, 0.75)),
    }.items():
        rows = {}
        for K in KS:
            el = k >= K
            h, e = hard_m & el, easy_m & el
            kh = float(k[h].mean()) if h.sum() else float("nan")
            ke = float(k[e].mean()) if e.sum() else float("nan")
            rows[K] = {
                "eligible_total": int(el.sum()),
                "hard": int(h.sum()), "easy": int(e.sum()),
                "mean_k_hard": kh, "mean_k_easy": ke,
                "imbalance": float(ke / kh) if h.sum() and e.sum() else float("nan"),
                "ceiling": float(N / K),
                "trunc_hard": (float(d["tr_all"][h].sum() / d["n"][h].sum())
                               if h.sum() else float("nan")),
                "trunc_easy": (float(d["tr_all"][e].sum() / d["n"][e].sum())
                               if e.sum() else float("nan")),
                "trunc_correct_hard": (float(d["tr_ok"][h].sum() / max(k[h].sum(), 1))
                                       if h.sum() else float("nan")),
                "trunc_correct_easy": (float(d["tr_ok"][e].sum() / max(k[e].sum(), 1))
                                       if e.sum() else float("nan")),
            }
        out["splits"][name] = rows
    # what a tau=0.1 group would give, to show why it is unusable
    tau_hard = k <= np.floor(0.1 * N)
    out["tau_hard_prompts"] = int(tau_hard.sum())
    out["tau_hard_eligible_at_four"] = int((tau_hard & (k >= 4)).sum())
    return out


results = {}
for label, (fname, tokp) in MODELS.items():
    print(f"\n{'='*74}\n{label}\n  {fname}")
    d = screen(fname, tokp)
    a = analyse(d)
    results[label] = a
    print(f"  {a['n_prompts']} prompts, N={a['N']}, mean pass rate {a['mean_p']:.4f}")
    print(f"  truncation at {CAP}: {a['trunc_overall']*100:.2f} % of all rollouts, "
          f"{a['trunc_among_correct']*100:.2f} % of CORRECT ones")
    print(f"  tau=0.1 hard group: {a['tau_hard_prompts']} prompts, of which "
          f"{a['tau_hard_eligible_at_four']} are eligible at K=4")
    for sp in ("median", "quartile"):
        print(f"  --- {sp} split ---")
        print(f"    {'K':>3} {'elig':>6} {'hard':>6} {'easy':>6} {'k_hard':>8} "
              f"{'k_easy':>8} {'imbal':>7} {'ceil':>6} {'trunc_h':>8} {'trunc_e':>8}")
        for K in KS:
            r = a["splits"][sp][K]
            print(f"    {K:>3} {r['eligible_total']:>6} {r['hard']:>6} {r['easy']:>6} "
                  f"{r['mean_k_hard']:>8.2f} {r['mean_k_easy']:>8.2f} "
                  f"{r['imbalance']:>7.3f} {r['ceiling']:>6.2f} "
                  f"{r['trunc_hard']*100:>7.2f}% {r['trunc_easy']*100:>7.2f}%")

json.dump(results, open(os.path.join(AN, "second_model_screen.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(AN, 'second_model_screen.json')}")
