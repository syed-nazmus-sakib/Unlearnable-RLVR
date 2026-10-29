"""Figure 10 — what set instability costs, as a function of the sample budget.

x: N, the per-prompt sample budget, log scale.
y: the 95 % width of the similarity ratio induced by REDRAWING THE FLAGGED SET
   under the A65 binomial model. This is not a sampling interval; it is the
   spread a reader would see across equally valid draws of the same experiment.

Two anchorings are plotted because A69 reports two and they agree as widths:
  solid   D_u-anchored (primary)
  dashed  label-faithful (secondary)
Their LEVELS are not comparable and levels are not plotted here at all.

Two horizontal references, both already printed in the paper:
  the sampling CI width on the same ratio, 0.50
  the materiality threshold fixed before computing, 0.25 (= half of it)

N=32 IS MARKED UNRESOLVED: the two anchorings straddle the threshold there
(0.2295 against 0.2767) and A69 forbids calling it either way.

Every number is read from the analysis JSONs and asserted against the emitted
macro file, so the figure cannot drift from the text.

  $HOME/miniconda3/envs/vllm/bin/python make_figure10.py
"""
import json
import os
import re

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(EXP, "analysis")
OUT = os.path.join(EXP, "figures_pub")
TEX = os.path.abspath(os.path.join(EXP, "..", "..", "TRANSFER_OUT",
                                   "instability_numbers.tex"))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

PRIM = json.load(open(os.path.join(AN, "a69_du_anchored.json")))
ALT = json.load(open(os.path.join(AN, "a69_set_instability.json")))

# ---- the figure must agree with the macro file, digit for digit ------------
src = open(TEX).read()


def macro(name):
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", src)
    assert m, f"macro {name} not found in {TEX}"
    return m.group(1)


MATERIAL = float(macro("insMaterial"))
PUB_W = float(macro("insPubCiWidth"))
WORDS = {"8": "Eight", "16": "Sixteen", "32": "ThirtyTwo",
         "128": "OneTwentyEight", "200": "TwoHundred"}
for n, w in WORDS.items():
    assert abs(PRIM[n]["width"] - float(macro("insWidth" + w))) < 5e-5, n
    assert abs(ALT[n]["width"] - float(macro("insWidthAlt" + w))) < 5e-5, n

NS = sorted(PRIM, key=int)
xs = [int(n) for n in NS]
yp = [PRIM[n]["width"] for n in NS]
ya = [ALT[n]["width"] for n in NS]

fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.spines[['top', 'right']].set_visible(False)

ax.axhspan(0, MATERIAL, color="#eef2f6", zorder=0)
ax.axhline(PUB_W, color="0.35", lw=0.9, ls=(0, (5, 4)), zorder=1)
ax.axhline(MATERIAL, color="#D55E00", lw=0.9, ls=(0, (2, 2)), zorder=1)
ax.text(8.4, PUB_W + 0.013, "sampling CI width already reported (0.50)",
        fontsize=7.2, color="0.35", va="bottom")
ax.text(205, MATERIAL - 0.014, "materiality (0.25)",
        fontsize=7.2, color="#D55E00", va="top", ha="right")
ax.text(205, 0.020, "immaterial", fontsize=7.2, color="0.45", ha="right")

ax.plot(xs, yp, marker="o", ms=4.5, lw=1.3, color="#D55E00", zorder=3,
        label="$D_u$-anchored (primary)")
ax.plot(xs, ya, marker="s", ms=4.0, lw=1.1, color="#0072B2",
        ls=(0, (4, 2)), zorder=3, label="label-faithful (secondary)")

ax.annotate("$N{=}32$: anchorings straddle\nthe threshold — UNRESOLVED",
            xy=(32, 0.2531), xytext=(44, 0.372), fontsize=7.0, color="0.25",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))

ax.set_xscale("log", base=2)
ax.set_xticks(xs)
ax.set_xticklabels([str(n) for n in xs])
ax.set_xlim(7.4, 220)
ax.set_ylim(0, 0.56)
ax.set_xlabel("per-prompt sample budget $N$")
ax.set_ylabel("95 % width of the similarity ratio\ninduced by redrawing the set")
ax.legend(loc="lower left", frameon=False, borderpad=0.2)
fig.tight_layout()
p = os.path.join(OUT, "figure10.pdf")
fig.savefig(p)
fig.savefig(p.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print("wrote", p)
print("  widths asserted against instability_numbers.tex for all five budgets")
