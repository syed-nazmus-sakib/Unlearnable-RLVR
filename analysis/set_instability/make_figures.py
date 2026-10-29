"""Publication figures for E0. Post-processing only — reads existing results,
runs nothing.

Figure 1  (a) set size vs seed count, min-max bands on BOTH curves
          (b) per-prompt flag count
          (c) boundary mass
Figure 2  |D_u| under each aggregation rule

Two-column paper settings: ~7in wide, axis labels 10pt, ticks 9pt, panel
titles 10pt, legend 9pt. Saved as vector PDF and 300-dpi PNG.

Greyscale: the two series differ in lightness (checked and printed at the end),
in line style (solid vs dashed), in marker (circle vs square), and the bands
differ in fill (solid tint vs hatch), so the figure survives mono printing.
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
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,      # embed TrueType, not Type-3 — required by most venues
    "ps.fonttype": 42,
})

C_UNL, C_LRN = "#8c1515", "#2b6a99"     # dark red / mid blue: distinct lightness


def counts(pat):
    sc = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(REPO, pat))):
        with open(p) as f:
            for it in json.load(f):
                sc[it["question_id"]].append(int(it["verification"]["score"]))
    return {q: (sum(v), len(v)) for q, v in sc.items()}


def no_reward(s):
    b = {}
    with open(glob.glob(f"{REPO}/checkpoints/*_seed{s}/prompt_reward_stats.csv")[0]) as f:
        for row in csv.DictReader(f):
            i, x = int(row["prompt_id"]), float(row["mean_reward"])
            b[i] = max(b.get(i, -1.0), x)
    return {i for i, x in b.items() if x == 0.0}


def luminance(hexcol):
    r, g, b = (int(hexcol[i:i+2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


# ---------------------------------------------------------------- data
pats = {1: "results/qwen_0.5b_run1_*@train*.json"}
pats.update({s: f"results/qwen_0.5b_seed{s}_*@train*.json" for s in (2, 3, 4, 5)})
c32 = {s: counts(p) for s, p in pats.items()}
flag = {s: {q for q, (a, b) in c.items() if a / b <= TAU} for s, c in c32.items()}
NR = {s: no_reward(s) for s in pats}
init = counts("results/qwen_0.5b_initial_*@train*.json")
initf = {q for q, (a, b) in init.items() if a / b <= TAU}
K = len(flag)

dec = []
for j in range(1, K + 1):
    us, ls = [], []
    for sub in combinations(sorted(flag), j):
        inter = set.intersection(*[flag[s] for s in sub])
        un = set.union(*[flag[s] for s in sub])
        nru = set.union(*[NR[s] for s in sub])
        us.append(len(inter - nru))
        ls.append(len(initf - un))
    dec.append(dict(j=j, um=np.mean(us), ulo=min(us), uhi=max(us),
                    lm=np.mean(ls), llo=min(ls), lhi=max(ls)))

df = list(csv.DictReader(open(os.path.join(EXP, "analysis", "per_example.csv"))))
cand = [r for r in df if int(r["n_flagged"]) > 0 and r["ever_no_reward"] == "False"]
nflag = np.array([int(r["n_flagged"]) for r in cand])
mp = np.array([float(r["mean_pass1"]) for r in cand])
sp = np.array([float(r["std_pass1"]) for r in cand])

# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.75))
js = [d["j"] for d in dec]

# (a) both curves carry a min-max band
ax[0].fill_between(js, [d["ulo"] for d in dec], [d["uhi"] for d in dec],
                   color=C_UNL, alpha=0.20, linewidth=0)
ax[0].fill_between(js, [d["llo"] for d in dec], [d["lhi"] for d in dec],
                   facecolor="none", edgecolor=C_LRN, hatch="////",
                   linewidth=0.0, alpha=0.55)
ax[0].plot(js, [d["um"] for d in dec], "o-", color=C_UNL, ms=4, lw=1.4,
           label="unlearnable")
ax[0].plot(js, [d["lm"] for d in dec], "s--", color=C_LRN, ms=4, lw=1.4,
           label="learnable")
ax[0].set_xlabel("seeds intersected")
ax[0].set_ylabel("prompts")
ax[0].set_title("Set size vs seed count")
ax[0].set_xticks(js)
ax[0].legend(frameon=False, loc="upper right", handlelength=2.2, borderpad=0.2)
ax[0].grid(alpha=0.25, lw=0.5)

# (b)
ks = list(range(1, K + 1))
cnt = [int((nflag == k).sum()) for k in ks]
cols = ["#d9a3a3"] * (K - 1) + [C_UNL]
ax[1].bar(ks, cnt, color=cols, width=0.62, edgecolor="black", linewidth=0.4)
ax[1].set_xlabel(f"seeds flagging the prompt (of {K})")
ax[1].set_ylabel("prompts")
ax[1].set_title("Per-prompt flag count")
ax[1].set_xticks(ks)
ax[1].grid(alpha=0.25, axis="y", lw=0.5)

# (c) tau marker moved into the legend so it cannot collide with the 0.1 tick
ax[2].errorbar(mp, nflag, xerr=sp, fmt="o", ms=2.6, alpha=0.45,
               color="#31688e", elinewidth=0.5, capsize=0)
ax[2].axvline(TAU, color="black", ls="--", lw=0.9, label=r"$\tau=0.1$")
# "across seeds" moves to the caption: the full label is 1.92in against a 1.39in
# panel, and as the rightmost panel that overhang ran off the page.
ax[2].set_xlabel("mean pass@1")
ax[2].set_ylabel("seeds flagging")
ax[2].set_title("Boundary mass")
ax[2].set_xlim(-0.01, min(0.6, mp.max() + 0.05))
ax[2].set_yticks(ks)
ax[2].legend(frameon=False, loc="upper right", handlelength=1.8, borderpad=0.2)
ax[2].grid(alpha=0.25, lw=0.5)

# Panel letters sit just outside the top-left corner rather than inside it:
# inside, (a) collided with the legend and (c) landed on the y=5 data points.
for a, lab in zip(ax, "abc"):
    a.text(-0.16, 1.12, f"({lab})", transform=a.transAxes,
           fontsize=10, fontweight="bold", va="top", ha="left")

fig.subplots_adjust(left=0.075, right=0.96, top=0.86, bottom=0.24, wspace=0.34)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figure1.{ext}"), dpi=300,
                bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
rules = [
    ("union of no_reward\n(5 seeds, published)", 72.0, None),
    ("single-seed no_reward", 145.4, [146, 146, 142, 145, 148]),
    ("zero correct in 128", 184.0, None),
    ("intersection of no_reward\n(5 seeds)", 225.0, None),
]
fig2, ax2 = plt.subplots(figsize=(7.0, 2.3))
y = np.arange(len(rules))[::-1]
vals = [r[1] for r in rules]
ax2.barh(y, vals, height=0.55,
         color=["#8c1515", "#b8b8b8", "#7f7f7f", "#4d4d4d"],
         edgecolor="black", linewidth=0.5)
# No legend: the per-seed spread is written into the value label instead, which
# keeps it off the bars. A legend box anywhere in this axes overlapped either the
# 225 bar or the 289 reference line.
for yi, (lab, v, pts) in zip(y, rules):
    if pts:
        ax2.plot(pts, [yi] * len(pts), "o", ms=4.5, color="black",
                 markerfacecolor="white", markeredgewidth=0.9, zorder=3)
        # +16, not +7: the marker at 148 is ~4 data units wide at this scale, so a
        # +7 offset left only ~3 units of clearance and the text crowded the markers.
        ax2.text(max(pts) + 16, yi, f"{v:g}  (seeds {min(pts)}–{max(pts)})",
                 va="center", fontsize=9)
    else:
        ax2.text(v + 7, yi, f"{v:g}", va="center", fontsize=9)
ax2.axvline(289, color="black", ls=":", lw=0.9)
ax2.text(289 - 7, 1.5, "289 sub-$\\tau$ prompts", fontsize=8,
         ha="center", va="center", rotation=90)
ax2.set_yticks(y)
ax2.set_yticklabels([r[0] for r in rules])
ax2.set_xlabel(r"$|D_u|$ after exclusion")
ax2.set_xlim(0, 320)
ax2.set_ylim(-0.55, len(rules) - 0.45)
ax2.grid(alpha=0.25, axis="x", lw=0.5)
ax2.set_axisbelow(True)
fig2.subplots_adjust(left=0.27, right=0.99, top=0.97, bottom=0.22)
for ext in ("pdf", "png"):
    fig2.savefig(os.path.join(OUT, f"figure2.{ext}"), dpi=300,
                 bbox_inches="tight", pad_inches=0.06)
plt.close(fig2)

print("greyscale check (WCAG relative luminance):")
lu, ll = luminance(C_UNL), luminance(C_LRN)
print(f"  unlearnable {C_UNL}: {lu:.3f}")
print(f"  learnable   {C_LRN}: {ll:.3f}")
print(f"  ratio {(max(lu,ll)+0.05)/(min(lu,ll)+0.05):.2f}:1  "
      f"({'OK' if (max(lu,ll)+0.05)/(min(lu,ll)+0.05) >= 1.5 else 'TOO CLOSE'})")
print("  bands also differ by fill type (solid tint vs hatch) and lines by "
      "style/marker, so they separate without colour.")
for f in ("figure1.pdf", "figure1.png", "figure2.pdf", "figure2.png"):
    p = os.path.join(OUT, f)
    print(f"  {p}  ({os.path.getsize(p)/1024:.0f} KB)")
