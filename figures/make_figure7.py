"""Figure 7 — reproducibility of the two aggregation rules at equal budget (A36).

One panel. x-axis: the two rules. y-axis: Jaccard between two independent
evaluation passes. Primary (flagging only) solid; the post-exclusion set in a
lighter shade beside it. Bars are the 95 % bootstrap intervals from the same
paired resample that produced the difference CI.

EVERY number is read from analysis/deep_vs_shallow.json, which deep_vs_shallow.py
writes. Nothing is typed in, so the figure cannot drift from the tables or from
deep_vs_shallow_numbers.tex.

  $HOME/miniconda3/envs/vllm/bin/python make_figure7.py
"""
import json
import os

# As with figures 5 and 6: matplotlib stamps /CreationDate into the PDF, so
# without this an md5 certifies "same file" and never "same figure".
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP, "figures_pub")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

R = json.load(open(os.path.join(EXP, "analysis", "deep_vs_shallow.json")))
PRIM = "flagging only (no exclusion)"
SEC = "after the no_reward exclusion"
RULES = ["deep", "shallow"]
LABEL = {"deep": "DEEP\none pass, $N=128$",
         "shallow": "SHALLOW\n4 blocks of 32,\nintersected"}
C_SOLID, C_LIGHT = "#D55E00", "#56B4E9"

fig, ax = plt.subplots(figsize=(4.2, 3.2))
x = np.arange(len(RULES))
w = 0.32

for off, key, col, name in ((-w / 2, PRIM, C_SOLID, "flagging only"),
                            (+w / 2, SEC, C_LIGHT, "after $no\\_reward$ exclusion")):
    r = R[key]
    y = np.array([r[k] for k in RULES])
    lo = np.array([r["ci"][k][0] for k in RULES])
    hi = np.array([r["ci"][k][1] for k in RULES])
    ax.bar(x + off, y, w, color=col, edgecolor="black", linewidth=0.6,
           label=f"{name} ($n={r['n']}$)", zorder=2)
    ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt="none", ecolor="black",
                elinewidth=0.9, capsize=3, capthick=0.9, zorder=3)
    # labels sit INSIDE the bars so they cannot collide with the interval caps
    for xi, yi in zip(x + off, y):
        ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, -4),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=6.8, color="white", zorder=4)

# the registered primary comparison, annotated with its paired-bootstrap CI
p = R[PRIM]
ax.annotate(f"primary $\\Delta$ = {p['diff']:+.4f}\n[{p['lo']:+.4f}, {p['hi']:+.4f}]",
            xy=(0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
            fontsize=7.0, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels([LABEL[k] for k in RULES])
ax.set_ylabel("Jaccard, two independent passes")
ax.set_ylim(0, 1.28)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.legend(frameon=False, loc="upper left", fontsize=6.8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(True, axis="y", color="#eaeaea", lw=0.6, zorder=0)
ax.set_title("Equal budget: 128 draws per prompt for both rules", fontsize=8.2)

fig.tight_layout()
p = os.path.join(OUT, "figure7.pdf")
fig.savefig(p, dpi=300, bbox_inches="tight")
fig.savefig(p.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"wrote {p}")
