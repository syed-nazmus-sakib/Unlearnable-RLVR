"""A48 — the E5 gate result, judged against the rule registered BEFORE the run.

Primary (A48.3): FAIL only if the 95 % paired-bootstrap interval on D_share lies
entirely ABOVE zero, or the interval on D_corr lies entirely BELOW zero.
PASS otherwise, including when they straddle zero.

Paired means one draw resamples PROMPTS once and recomputes BOTH models on that
same resampled set (A48.2). 5000 draws, seed 20260812.

Truncation follows A47: a response is truncated if it tokenises to >= 1024
tokens under ITS OWN model's tokenizer, because the cap is applied in that
model's tokens.

  $HOME/miniconda3/envs/vllm/bin/python gate_report.py
"""
import json
import os

import numpy as np
from transformers import AutoTokenizer

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(EXP, "..", ".."))
RES = os.path.join(ROOT, "unlearnability-rlvr", "results")
SEED, NBOOT, CAP = 20260812, 5000, 1024

RUNS = {
    "1.5B": ("e5_gate_qwen15b_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json",
             os.path.join(ROOT, "models", "Qwen2.5-1.5B")),
    "0.5B": ("qwen_0.5b_initial_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json",
             os.path.join(ROOT, "models", "Qwen2.5-0.5B")),
}


def load(fname, model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    recs = json.load(open(os.path.join(RES, fname)))
    per = {}
    texts, index = [], []
    for r in recs:
        q = int(r["question_id"])
        v = r.get("verification", {}) or {}
        bad = v.get("extracted_answer") == "[invalid]"
        d = per.setdefault(q, {"n": 0, "inv": 0, "ok": 0, "tr": 0})
        d["n"] += 1
        d["inv"] += bad
        d["ok"] += int(v.get("score", 0)) == 1 and not bad   # A24 guard
        texts.append(r["response"])
        index.append(q)
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    for q, e in zip(index, enc):
        per[q]["tr"] += len(e) >= CAP
    return per


D = {k: load(*v) for k, v in RUNS.items()}
Q = sorted(set(D["1.5B"]) & set(D["0.5B"]))
print(f"{len(Q)} prompts common to both runs\n")


def vec(model, key):
    return np.array([D[model][q][key] for q in Q], dtype=float)


N = {m: vec(m, "n") for m in D}
INV = {m: vec(m, "inv") for m in D}
OK = {m: vec(m, "ok") for m in D}
TR = {m: vec(m, "tr") for m in D}


def stats(m, idx):
    n, inv, ok, tr = N[m][idx], INV[m][idx], OK[m][idx], TR[m][idx]
    share = inv.sum() / n.sum()
    trunc = tr.sum() / n.sum()
    resid = (inv - np.minimum(inv, tr)).sum() / max((n - tr).sum(), 1)
    pr, ir = ok / n, inv / n
    corr = float(np.corrcoef(ir, pr)[0, 1])
    return share, corr, trunc, resid


base = {m: stats(m, np.arange(len(Q))) for m in D}
rng = np.random.default_rng(SEED)
BI = rng.integers(0, len(Q), size=(NBOOT, len(Q)))
dif = np.empty((NBOOT, 4))
for i, row in enumerate(BI):                    # ONE resample -> BOTH models
    a, b = stats("1.5B", row), stats("0.5B", row)
    dif[i] = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]

names = ["D_share", "D_corr", "D_trunc", "D_resid"]
L = ["| quantity | 1.5B | 0.5B initial | difference | 95 % CI |",
     "|---|---|---|---|---|"]
for j, (nm, lab) in enumerate(zip(names, ["[invalid] share", "corr(invalid, pass)",
                                          "truncation share", "residual failure"])):
    lo, hi = np.percentile(dif[:, j], [2.5, 97.5])
    L.append(f"| {lab} | {base['1.5B'][j]:+.4f} | {base['0.5B'][j]:+.4f} | "
             f"{base['1.5B'][j]-base['0.5B'][j]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
print("\n".join(L))

s_lo, s_hi = np.percentile(dif[:, 0], [2.5, 97.5])
c_lo, c_hi = np.percentile(dif[:, 1], [2.5, 97.5])
fail_share, fail_corr = s_lo > 0, c_hi < 0
verdict = "FAIL" if (fail_share or fail_corr) else "PASS"
print(f"\nD_share entirely above zero? {fail_share}")
print(f"D_corr  entirely below zero? {fail_corr}")
print(f"A48.3 VERDICT: {verdict}")

# Tier construction and matched-K feasibility, on the 1.5B screening pass.
out = {}
for m in D:
    pr = OK[m] / N[m]
    o = np.argsort(pr, kind="stable")
    t = len(o) // 3
    bot, mid, top = pr[o[:t]], pr[o[t:2 * t]], pr[o[2 * t:]]
    span = float(top.mean() / bot.mean()) if bot.mean() > 0 else float("inf")
    inc = N[m] - OK[m]
    feas = {K: int(((OK[m] >= K) & (inc >= K)).sum()) for K in (4, 6, 8, 11)}
    out[m] = {"tier_means": [float(bot.mean()), float(mid.mean()), float(top.mean())],
              "span": span, "feasible": feas,
              "bottom_zero": int((OK[m][o[:t]] == 0).sum()), "n_tier": t}
    print(f"\n{m} terciles on guarded pass rate (N=32 screening, OPTIMISTIC per A35.2)")
    print(f"  bottom {bot.mean():.4f}  mid {mid.mean():.4f}  top {top.mean():.4f}  "
          f"span {span:.2f}x   ({t} prompts per tier, "
          f"{out[m]['bottom_zero']} bottom-tier prompts at zero correct)")
    print(f"  usable prompts (>=K correct AND >=K incorrect): "
          + "  ".join(f"K={K}: {v}" for K, v in feas.items()))

json.dump({"n_prompts": len(Q), "seed": SEED, "nboot": NBOOT,
           "base": {m: dict(zip(names, map(float, base[m]))) for m in D},
           "diff_ci": {nm: [float(x) for x in np.percentile(dif[:, j], [2.5, 97.5])]
                       for j, nm in enumerate(names)},
           "verdict": verdict, "tiers": out},
          open(os.path.join(EXP, "gate_report.json"), "w"), indent=2)
print("\nwrote gate_report.json")
