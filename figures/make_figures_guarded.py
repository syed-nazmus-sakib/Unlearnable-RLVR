"""Regenerate Figures 1 and 2 from GUARDED data.

Guard: a rollout counts as correct only when
`verification.extracted_answer != '[invalid]'`, i.e. the verifier actually
extracted a boxed answer. See truncation_check.md for why the raw
`score == 1` label admits answer-less responses.

Styling, fonts, sizes and file formats are identical to
E0_seed_stability/make_figures.py. Output goes to E1_rescue/figures_guarded/
rather than over the E0 figures, so the published versions survive for
comparison -- copy them across if the paper adopts the guarded set.

  python make_figures_guarded.py
"""
import csv, glob, json, os
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(EXP, "..", "..", "unlearnability-rlvr"))
OUT = os.path.join(EXP, "figures_pub")
os.makedirs(OUT, exist_ok=True)
TAU = 0.1

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_UNL, C_LRN = "#D55E00", "#0072B2"

GOOD = lambda i: (int(i["verification"]["score"]) == 1
                  and i["verification"]["extracted_answer"] != "[invalid]")


def rates(pattern):
    d = defaultdict(lambda: [0, 0])
    for p in sorted(glob.glob(os.path.join(REPO, pattern))):
        with open(p) as f:
            for it in json.load(f):
                c = d[it["question_id"]]
                c[0] += GOOD(it); c[1] += 1
    return {q: a / b for q, (a, b) in d.items()}


def no_reward(s):
    b = {}
    with open(glob.glob(f"{REPO}/checkpoints/*_seed{s}/prompt_reward_stats.csv")[0]) as f:
        for row in csv.DictReader(f):
            i, x = int(row["prompt_id"]), float(row["mean_reward"])
            b[i] = max(b.get(i, -1.0), x)
    return {i for i, x in b.items() if x == 0.0}


PAT = {1: "results/qwen_0.5b_run1_*@train*.json"}
PAT.update({s: f"results/qwen_0.5b_seed{s}_*@train*.json" for s in (2, 3, 4, 5)})
R = {s: rates(p) for s, p in PAT.items()}
flag = {s: {q for q, v in R[s].items() if v <= TAU} for s in R}
NR = {s: no_reward(s) for s in R}
NRU = set().union(*NR.values())
initf = {q for q, v in rates("results/qwen_0.5b_initial_*@train*.json").items() if v <= TAU}
K = len(flag)

dec = []
for j in range(1, K + 1):
    us, ls = [], []
    for sub in combinations(sorted(flag), j):
        us.append(len(set.intersection(*[flag[s] for s in sub])
                      - set.union(*[NR[s] for s in sub])))
        ls.append(len(initf - set.union(*[flag[s] for s in sub])))
    dec.append(dict(j=j, um=np.mean(us), ulo=min(us), uhi=max(us),
                    lm=np.mean(ls), llo=min(ls), lhi=max(ls)))

allq = set(R[1])
nflag_map = {q: sum(q in flag[s] for s in flag) for q in allq}
cand = [q for q in allq if nflag_map[q] > 0 and q not in NRU]
nflag = np.array([nflag_map[q] for q in cand])
mp = np.array([np.mean([R[s].get(q, np.nan) for s in R]) for q in cand])
sp = np.array([np.std([R[s].get(q, np.nan) for s in R]) for q in cand])

# ---------------------------------------------------------------- Figure 1
# Authored at FINAL PRINTED SIZE: the paper includes this row at
# ~0.9\textwidth (NeurIPS \textwidth = 5.5in) = 4.95in, so the fontsize=8/9
# values below are the true printed sizes. Fonts are set per-call rather than
# via rcParams so Figure 2 (drawn later in this script) is left untouched.
FIG1_W = 0.9 * 5.5
fig, ax = plt.subplots(1, 3, figsize=(FIG1_W, 2.0))
js = [d["j"] for d in dec]
for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
    a_.grid(color="#eaeaea", lw=0.6, zorder=0)
    a_.set_axisbelow(True)
    a_.tick_params(labelsize=8.5)

ax[0].fill_between(js, [d["ulo"] for d in dec], [d["uhi"] for d in dec],
                   color=C_UNL, alpha=0.20, linewidth=0, zorder=1)
ax[0].fill_between(js, [d["llo"] for d in dec], [d["lhi"] for d in dec],
                   facecolor="none", edgecolor=C_LRN, hatch="////",
                   linewidth=0.0, alpha=0.55, zorder=1)
ax[0].plot(js, [d["um"] for d in dec], "o-", color=C_UNL, ms=3.4, lw=1.3, zorder=3, label="unlearnable")
ax[0].plot(js, [d["lm"] for d in dec], "s--", color=C_LRN, ms=3.4, lw=1.3, zorder=3, label="learnable")
ax[0].set_xlabel("seeds intersected", fontsize=9)
ax[0].set_ylabel("prompts", fontsize=9)
ax[0].set_title("Set size vs seed count", fontsize=9, fontweight="bold")
ax[0].set_xticks(js)
# Inline labels instead of a legend: at this panel width an 8pt legend box
# cannot avoid the curves. Placed in regions both curves stay clear of.
ax[0].text(0.55, 0.80, "learnable", color=C_LRN, fontsize=8.5,
           transform=ax[0].transAxes, ha="left", va="bottom")
ax[0].text(0.04, 0.04, "unlearnable", color=C_UNL, fontsize=8.5,
           transform=ax[0].transAxes, ha="left", va="bottom")

ks = list(range(1, K + 1))
cnt = [int((nflag == k).sum()) for k in ks]
cols = ["#cccccc"] * (K - 1) + [C_UNL]  # Neutral grey for background, Okabe-Ito Vermilion for highlight
ax[1].bar(ks, cnt, color=cols, width=0.62, edgecolor="black", linewidth=0.4, zorder=3)
ax[1].set_xlabel(f"seeds flagging the prompt (of {K})", fontsize=9)
ax[1].set_ylabel("prompts", fontsize=9)
ax[1].set_title("Per-prompt flag count", fontsize=9, fontweight="bold")
ax[1].set_xticks(ks)
ax[1].grid(False, axis="x")  # Remove vertical gridlines for categorical bars

ax[2].errorbar(mp, nflag, xerr=sp, fmt="o", ms=2.4, alpha=0.6,
               color=C_LRN, elinewidth=0.5, capsize=0, zorder=3)
ax[2].axvline(TAU, color="black", ls="--", lw=0.9, zorder=4, label=r"$\tau=0.1$")
ax[2].set_xlabel("mean pass@1", fontsize=9)
ax[2].set_ylabel("seeds flagging", fontsize=9)
ax[2].set_title("Boundary mass", fontsize=9, fontweight="bold")
ax[2].set_xlim(-0.01, min(0.6, float(mp.max()) + 0.05)); ax[2].set_yticks(ks)
# larger fonts + narrow panel: fewer x ticks instead of smaller text
ax[2].set_xticks([0.0, 0.2, 0.4])
ax[2].legend(frameon=False, loc="upper right", fontsize=8.5,
             handlelength=1.6, borderpad=0.2)

fig.subplots_adjust(left=0.075, right=0.97, top=0.85, bottom=0.26, wspace=0.42)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figure1.{ext}"), dpi=300,
                bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
ref = {}
for pat in ("results/qwen_0.5b_ckpt1_hi_simplelr_qwen_level1to4_noise@train*.json",
            "results/qwen_0.5b_ckpt1_hi_rest_*@train*.json"):
    ref.update(rates(pat))
sub = {q for q, v in ref.items() if v <= TAU}
I = set.intersection(*NR.values())
per = [len(sub - NR[s]) for s in sorted(NR)]
rules = [
    ("union of no_reward\n(5 seeds, published)", float(len(sub - NRU)), None),
    ("single-seed no_reward", float(np.mean(per)), sorted(per)),
    ("zero correct in 128", float(sum(1 for q in sub if ref[q] > 0)), None),
    ("intersection of no_reward\n(5 seeds)", float(len(sub - I)), None),
]
fig2, ax2 = plt.subplots(figsize=(7.0, 2.3))
ax2.spines[["top", "right"]].set_visible(False)
y = np.arange(len(rules))[::-1]
ax2.barh(y, [r[1] for r in rules], height=0.55,
         color=[C_UNL, "#b8b8b8", "#7f7f7f", "#4d4d4d"],
         edgecolor="black", linewidth=0.5, zorder=3)
for yi, (lab, v, pts) in zip(y, rules):
    if pts:
        ax2.plot(pts, [yi] * len(pts), "o", ms=4.5, color="black",
                 markerfacecolor="white", markeredgewidth=0.9, zorder=4)
        ax2.text(max(pts) + 16, yi, f"{v:g}  (seeds {min(pts)}–{max(pts)})",
                 va="center", fontsize=9, zorder=5)
    else:
        ax2.text(v + 7, yi, f"{v:g}", va="center", fontsize=9, zorder=5)
ax2.axvline(len(sub), color="black", ls=":", lw=0.9, zorder=2)
ax2.text(len(sub) - 7, 1.5, f"{len(sub)} sub-$\\tau$ prompts", fontsize=8,
         ha="center", va="center", rotation=90, zorder=5)
ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rules])
ax2.set_xlabel(r"$|D_u|$ after exclusion")
ax2.set_xlim(0, 320); ax2.set_ylim(-0.55, len(rules) - 0.45)
ax2.grid(True, axis="x", color="#eaeaea", lw=0.6, zorder=0); ax2.set_axisbelow(True)
fig2.subplots_adjust(left=0.27, right=0.99, top=0.97, bottom=0.22)
for ext in ("pdf", "png"):
    fig2.savefig(os.path.join(OUT, f"figure2.{ext}"), dpi=300,
                 bbox_inches="tight", pad_inches=0.06)
plt.close(fig2)

print("Figure 1 (guarded)")
print(f"  decay unlearnable: {[round(d['um'],1) for d in dec]}")
print(f"  decay learnable  : {[round(d['lm'],1) for d in dec]}")
print(f"  flag counts      : {cnt}  (n={len(cand)})")
print(f"  panel (c) points : {len(mp)}  x-range {mp.min():.3f}..{mp.max():.3f}")
print("Figure 2 (guarded)")
print(f"  bars: {[round(r[1],1) for r in rules]}   seeds {min(per)}-{max(per)}   ref line {len(sub)}")
for f in ("figure1.pdf", "figure1.png", "figure2.pdf", "figure2.png"):
    print(f"  {os.path.join(OUT,f)}  ({os.path.getsize(os.path.join(OUT,f))/1024:.0f} KB)")
