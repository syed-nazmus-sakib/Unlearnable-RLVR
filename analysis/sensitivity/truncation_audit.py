"""A47 — how much of our "extraction failure" is truncation at OUR token cap?

`repro/eval.sh:41` sets MAX_LENGTH=1024. Chen's `inference/test.sh:25` defaults
to 5120 and their training configs use 5120 (8192 for the 3B arm). We ran every
evaluation at one fifth of their generation budget. This measures what that cost.

A rollout is TRUNCATED if its response tokenises to >= the cap. Truncated
rollouts cannot contain a terminal \\boxed{}, so they are scored incorrect for a
reason that is about our budget, not about the model or the verifier.

Reported as a RANGE, never a point (A44.3): conditioning on non-truncated
rollouts removes the confound but drops exactly the long, messy generations, so
it reads LOW. The truth at 5120 lies between the two, and we did not buy it.

CPU only.

  $HOME/miniconda3/envs/vllm/bin/python truncation_audit.py
"""
import json
import os
import re
import subprocess
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(EXP, "..", "..", "unlearnability-rlvr")
RES = os.path.join(REPO, "results")
MODELS = os.path.join(EXP, "..", "..", "models")
OUT = os.path.join(EXP, "analysis", "truncation_audit.md")
CAP, THEIR_CAP = 1024, 5120
SEED, NBOOT = 20260812, 5000

QWEN_REF = ["qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train_0.7_128_1024_None_0_qwen.json",
            "qwen_0.5b_ckpt1_hi_rest_simplelr_qwen_level1to4_rest@train_0.7_128_1024_None_0_qwen.json"]
QWEN_INIT = ["qwen_0.5b_initial_simplelr_qwen_level1to4_sub1k@train_0.7_32_1024_None_0_qwen.json"]
LLAMA_V1 = ["e4_gate_llama3b_e4_screen800@test_0.7_32_1024_None_0_qwen.json"]


def ids(name):
    return {int(x) for x in json.load(
        open(os.path.join(REPO, "example_ids", f"qwen_0.5b_math_level1to4_{name}.json")))}


def scan(files, tok_path):
    tok = AutoTokenizer.from_pretrained(tok_path)
    q, inv, trunc = [], [], []
    for f in files:
        rs = json.load(open(os.path.join(RES, f)))
        texts = [r.get("response", "") for r in rs]
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for r, e in zip(rs, enc):
            v = r.get("verification", {}) or {}
            q.append(int(r["question_id"]))
            inv.append(v.get("extracted_answer") == "[invalid]")
            trunc.append(len(e) >= CAP)
    return np.array(q), np.array(inv), np.array(trunc)


rng = np.random.default_rng(SEED)


def summarise(inv, tr):
    return {"n": int(len(inv)), "invalid": float(inv.mean()),
            "trunc": float(tr.mean()),
            "inv_given_trunc": float(inv[tr].mean()) if tr.any() else float("nan"),
            "trunc_given_inv": float(tr[inv].mean()) if inv.any() else float("nan"),
            "inv_given_untrunc": float(inv[~tr].mean()) if (~tr).any() else float("nan")}


L = []
W = L.append
W("# A47 — how much of our extraction failure is our own token cap?\n")
W(f"`repro/eval.sh:41` sets **MAX_LENGTH={CAP}**. Chen's `inference/test.sh:25` "
  f"defaults to **{THEIR_CAP}**, and their training configs use {THEIR_CAP} "
  f"(8192 for the 3B arm). **We ran every evaluation at one fifth of their "
  "generation budget, and our harness chose it.** A rollout counted as "
  f"truncated if its response tokenises to >= {CAP} tokens.\n")

RESULTS = {}
for label, files, tokp in (("Qwen2.5-0.5B trained, N=128 pooled (the paper's reference)",
                            QWEN_REF, os.path.join(MODELS, "Qwen2.5-0.5B")),
                           ("Qwen2.5-0.5B initial, N=32", QWEN_INIT,
                            os.path.join(MODELS, "Qwen2.5-0.5B")),
                           ("Llama-3.2-3B-Instruct, E4 gate v1, N=32", LLAMA_V1,
                            os.path.join(MODELS, "Llama-3.2-3B-Instruct"))):
    tokp = tokp if os.path.isdir(tokp) else ("Qwen/Qwen2.5-0.5B" if "Qwen" in label
                                             else "meta-llama/Llama-3.2-3B-Instruct")
    qid, inv, tr = scan(files, tokp)
    RESULTS[label] = {"overall": summarise(inv, tr), "qid": qid, "inv": inv, "tr": tr}
    print(f"scanned {label}: {len(inv)} rollouts", flush=True)

W("## Overall\n")
W("| run | rollouts | `[invalid]` | truncated | of INVALID, at the cap | "
  "of TRUNCATED, invalid | **invalid among NON-truncated** |")
W("|---|---|---|---|---|---|---|")
for label, d in RESULTS.items():
    o = d["overall"]
    W(f"| {label} | {o['n']} | {100 * o['invalid']:.2f} % | {100 * o['trunc']:.2f} % | "
      f"**{100 * o['trunc_given_inv']:.2f} %** | {100 * o['inv_given_trunc']:.2f} % | "
      f"**{100 * o['inv_given_untrunc']:.2f} %** |")
W("")

key = "Qwen2.5-0.5B trained, N=128 pooled (the paper's reference)"
o = RESULTS[key]["overall"]
W(f"**On the checkpoint the paper's D_u is defined from, "
  f"{100 * o['trunc_given_inv']:.1f} % of failed extractions are truncated "
  f"rollouts, and among rollouts that finished, extraction fails "
  f"{100 * o['inv_given_untrunc']:.2f} % of the time.** What we described as a "
  "verifier failing on hard problems is substantially hard problems needing "
  "more than our budget and being cut off mid-working.\n")

# ---- per published group, on the paper's reference ---------------------------
GRP = {g: ids(g) for g in ("unlearnable", "learnable", "easy", "no_reward")}
import pandas as pd  # noqa: E402
_tr = pd.read_csv(os.path.join(EXP, "..", "E0b_eval_noise", "analysis", "true_rates_all.csv"))
GRP["du"] = {int(x) for x in _tr.qid[_tr.in_D_u == 1]}
qid, inv, tr = RESULTS[key]["qid"], RESULTS[key]["inv"], RESULTS[key]["tr"]
per = defaultdict(lambda: {"inv": [], "tr": []})
memb = defaultdict(list)
for g, S in GRP.items():
    for x in S:
        memb[x].append(g)
for x, i, t in zip(qid, inv, tr):
    for g in memb.get(int(x), ()):
        per[g]["inv"].append(i)
        per[g]["tr"].append(t)

W("## By published group, on that same reference\n")
W("This is the table that matters: if truncation varies with difficulty, then "
  "so does everything computed downstream of it.\n")
W("| group | truncated | `[invalid]` | **invalid among NON-truncated** | "
  "95 % CI on the last column |")
W("|---|---|---|---|---|")
GRES = {}
for g in ("unlearnable", "du", "learnable", "easy", "no_reward"):
    i = np.array(per[g]["inv"])
    t = np.array(per[g]["tr"])
    u = i[~t]
    b = u[rng.integers(0, len(u), size=(NBOOT, len(u)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    bt = t[rng.integers(0, len(t), size=(NBOOT, len(t)))].mean(axis=1)
    tlo, thi = np.percentile(bt, [2.5, 97.5])
    bi = i[rng.integers(0, len(i), size=(NBOOT, len(i)))].mean(axis=1)
    ilo, ihi = np.percentile(bi, [2.5, 97.5])
    GRES[g] = {"trunc": float(t.mean()), "trunc_lo": float(tlo), "trunc_hi": float(thi),
               "invalid": float(i.mean()), "inv_lo": float(ilo), "inv_hi": float(ihi),
               "inv_untrunc": float(u.mean()), "lo": float(lo), "hi": float(hi)}
    W(f"| {g} | {100 * t.mean():.2f} % | {100 * i.mean():.2f} % | "
      f"**{100 * u.mean():.2f} %** | [{100 * lo:.2f}, {100 * hi:.2f}] |")
W("")
ru = GRES["unlearnable"]["trunc"] / GRES["easy"]["trunc"]
rv = (GRES["unlearnable"]["inv_untrunc"] / GRES["easy"]["inv_untrunc"]
      if GRES["easy"]["inv_untrunc"] > 0 else float("inf"))
W(f"**Truncation rate, unlearnable / easy: {ru:.1f}x.** The gradient we "
  "attributed to the verifier is present in the truncation rate itself.\n")
W(f"Residual extraction failure among rollouts that finished, unlearnable / "
  f"easy: {rv:.1f}x — on a base of "
  f"{100 * GRES['unlearnable']['inv_untrunc']:.2f} % against "
  f"{100 * GRES['easy']['inv_untrunc']:.2f} %.\n")

W("## What this does and does not license\n")
W("**Does:** every extraction-failure number we report must state the token "
  "cap beside it, and must be given as a RANGE — the raw share at our cap, and "
  "the share conditioned on non-truncated rollouts.\n")
W("**Does not:** conditioning is not a substitute for a run at 5120. It drops "
  "exactly the long, messy generations that would be most at risk of failing to "
  "parse even with more room, so the conditioned figure is a LOWER bound. The "
  "value at Chen's budget lies between the two and we declined to buy it "
  "(A44).\n")
W("**This is a defect in our replication, not in their paper.** Their default "
  f"is {THEIR_CAP}; ours was {CAP}; the choice was made in our harness.\n")

# ---- macros -----------------------------------------------------------------
def git(*a):
    return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True,
                          text=True).stdout.strip()


HEAD = git("rev-parse", "HEAD")
VER_COMMIT = git("log", "-1", "--format=%H", "--", "inference/math_verifier.py")
VER_CLEAN = git("status", "--porcelain", "inference/math_verifier.py") == ""


def prompt_template_keys_at_head():
    """Enumerate inference prompt keys from both registry construction sites."""
    config = git("show", f"{HEAD}:inference/config.py")
    keys = []
    in_initial_registry = False
    for line in config.splitlines():
        if "self.custom_templates = {" in line:
            in_initial_registry = True
        if in_initial_registry:
            match = re.match(r'^\s*"([^"]+)"\s*:', line)
            if match and match.group(1) not in keys:
                keys.append(match.group(1))
            if line.strip() == "}":
                in_initial_registry = False
        for match in re.finditer(
                r'config\.prompt\.custom_templates\["([^"]+)"\]', line):
            if match.group(1) not in keys:
                keys.append(match.group(1))
    if not keys:
        raise AssertionError("no inference prompt-template keys discovered")
    return keys


PROMPT_TEMPLATE_KEYS = prompt_template_keys_at_head()

llk = "Llama-3.2-3B-Instruct, E4 gate v1, N=32"
lo_ = RESULTS[llk]["overall"]
T = ["% GENERATED BY EXPERIMENTS/E1_rescue/truncation_audit.py -- DO NOT EDIT BY HAND.",
     "% Rerun that script to regenerate. A47.",
     "%",
     f"% OUR cap was {CAP} tokens (repro/eval.sh:41). Chen's inference/test.sh:25",
     f"% defaults to {THEIR_CAP}. Every share below is measured AT OUR CAP and is",
     "% meaningless without it. Report shares as a RANGE: raw, and conditioned on",
     "% non-truncated rollouts, which is a LOWER bound (A44.3).",
     "% Contract: no percent or dollar sign inside a macro; counts bare integers.",
     ""]
T.append(r"\newcommand{\trOurCap}{%d}" % CAP)
T.append(r"\newcommand{\trTheirCap}{%d}" % THEIR_CAP)
T.append("")
T.append("% --- Llama-3.2-3B E4 gate v1 (the dropped arm) ---")
T.append(r"\newcommand{\trLlamaInvalid}{%.4f}" % lo_["invalid"])
T.append(r"\newcommand{\trLlamaCapped}{%.4f}" % lo_["trunc"])
T.append(r"\newcommand{\trLlamaInvGivenCapped}{%.4f}" % lo_["inv_given_trunc"])
T.append(r"\newcommand{\trLlamaCappedGivenInv}{%.4f}" % lo_["trunc_given_inv"])
T.append(r"\newcommand{\trLlamaInvGivenUncapped}{%.4f}" % lo_["inv_given_untrunc"])
T.append("")
T.append("% --- Qwen2.5-0.5B trained, N=128 pooled: the paper's own reference ---")
T.append(r"\newcommand{\trQwenInvalid}{%.4f}" % o["invalid"])
T.append(r"\newcommand{\trQwenCapped}{%.4f}" % o["trunc"])
T.append(r"\newcommand{\trQwenCappedGivenInv}{%.4f}" % o["trunc_given_inv"])
T.append(r"\newcommand{\trQwenInvGivenUncapped}{%.4f}" % o["inv_given_untrunc"])
T.append("")
NM = {"unlearnable": "Unlearn", "du": "Du", "learnable": "Learn",
      "easy": "Easy", "no_reward": "Noreward"}
for g, tag in NM.items():
    T.append(r"\newcommand{\trTrunc%s}{%.4f}" % (tag, GRES[g]["trunc"]))
    T.append(r"\newcommand{\trResid%s}{%.4f}" % (tag, GRES[g]["inv_untrunc"]))
    T.append(r"\newcommand{\trResid%sLo}{%.4f}" % (tag, GRES[g]["lo"]))
    T.append(r"\newcommand{\trResid%sHi}{%.4f}" % (tag, GRES[g]["hi"]))
T.append(r"\newcommand{\trTruncRatioUnlearnEasy}{%.1f}" % ru)
T.append("")
T.append("% --- provenance of the released artifact (A46), for checkable claims ---")
T.append(r"\newcommand{\chenHead}{%s}" % HEAD)
T.append(r"\newcommand{\chenHeadShort}{%s}" % HEAD[:7])
T.append(r"\newcommand{\chenReadmeLlamaLine}{%d}" % 246)
T.append(r"\newcommand{\chenReadmeTableLine}{%d}" % 243)
T.append(r"\newcommand{\chenConfigKeyLineFirst}{%d}" % 159)
T.append(r"\newcommand{\chenConfigKeyLineLast}{%d}" % 164)
T.append(r"\newcommand{\chenTemplateKeys}{%s}" % ", ".join(
    key.replace("_", r"\_") for key in PROMPT_TEMPLATE_KEYS))
T.append(r"\newcommand{\chenNumTemplateKeys}{%d}" % len(PROMPT_TEMPLATE_KEYS))
T.append(r"\newcommand{\chenTestShMaxLenLine}{%d}" % 25)
T.append(r"\newcommand{\verifierCommit}{%s}" % VER_COMMIT)
T.append(r"\newcommand{\verifierCommitShort}{%s}" % VER_COMMIT[:7])
T.append(r"\newcommand{\verifierUnmodified}{%s}" % ("true" if VER_CLEAN else "FALSE"))

TEX = os.path.join(EXP, "analysis", "truncation_numbers.tex")
open(TEX, "w").write("\n".join(T) + "\n")
json.dump({"overall": {k: v["overall"] for k, v in RESULTS.items()},
           "by_group": GRES, "our_cap": CAP, "their_cap": THEIR_CAP,
           "head": HEAD, "verifier_commit": VER_COMMIT,
           "verifier_unmodified": VER_CLEAN},
          open(os.path.join(EXP, "analysis", "truncation_audit.json"), "w"), indent=2)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nwrote {TEX}")
