"""Figure 9 (redesigned) — predicted vs measured Jaccard, NeurIPS style.

Two-panel layout:
  Top:    scatter with shaded agreement band around the diagonal.
  Bottom: residual strip (measured − predicted) with zero line.

Direct inline labels replace the legend box.  Every number is read from
the analysis JSONs — nothing is typed in.

  $HOME/miniconda3/envs/vllm/bin/python make_figure_9.py
"""
import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────
EXP = os.path.dirname(os.path.abspath(__file__))
AN  = os.path.join(EXP, "analysis")
OUT = os.path.join(EXP, "figures_pub")
os.makedirs(OUT, exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────
RULE = json.load(open(os.path.join(AN, "budget_rule.json")))
XM   = json.load(open(os.path.join(AN, "budget_crossmodel.json")))
PP   = json.load(open(os.path.join(AN, "budget_prospective_result.json")))
GS   = json.load(open(os.path.join(AN, "gsm8k_result.json")))

# ── NeurIPS house style ───────────────────────────────────────────────
# Authored at FINAL PRINTED SIZE: the paper includes this as subfigure (b)
# at 0.40\textwidth (NeurIPS \textwidth = 5.5in) = 2.2in wide, so every
# font below is its true printed size (>= 8pt).
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "cm",
    "font.size":         8.5,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.dpi":        300,
    "savefig.dpi":       300,
})

# ── colour palette (colorblind-safe, muted) ───────────────────────────
C_RETRO  = "#4878A8"     # steel blue   — retrospective
C_CROSS  = "#D4963A"     # warm amber   — cross-model
C_OOS    = "#C0392B"     # deep red     — out-of-sample
C_GSM    = "#8E44AD"     # plum         — GSM8K
C_EXCL   = "#7F8C8D"     # gray         — excluded 1.5B
C_BAND   = "#2C3E50"     # dark band edge
C_GRID   = "#E8E8E8"     # subtle grid

# ── assemble point lists ──────────────────────────────────────────────
# Each tuple: (label, pred, plo, phi, meas, mlo, mhi)

retro = []
for key, npred, field in [("$N{=}32$ deep",  "32",  "deep"),
                          ("$N{=}128$ deep", "128", "deep"),
                          ("$N{=}128$ shallow", "128", "shallow")]:
    m, mlo, mhi = RULE["measured"][
        {"$N{=}32$ deep": "N=32 deep",
         "$N{=}128$ deep": "N=128 deep",
         "$N{=}128$ shallow": "N=128 shallow 4x32"}[key]
    ]
    t = RULE["table"][npred]
    f = {"$N{=}128$ shallow": "shallow"}.get(key, field)
    p   = t[f]
    plo = t.get(f"{f}_lo")
    phi = t.get(f"{f}_hi")
    retro.append((key, p, plo, phi, m, mlo, mhi))

XM_EXCLUDED = "Qwen2.5-1.5B base (E5 gate)"
cross = []
for k, v in XM.items():
    if k == XM_EXCLUDED:
        continue
    short = (k.replace(" (reference)", "")
              .replace(" (screen800)", "")
              .replace(" (screen800 v2)", " v2"))
    cross.append((short, v["pred"], v["plo"], v["phi"],
                  v["measured"], v["mlo"], v["mhi"]))
xv = XM[XM_EXCLUDED]

pros = []
for n in ("40", "64"):
    pros.append(("$N\\!=" + n + "$",
                 PP[n]["pred"], PP[n]["plo"], PP[n]["phi"],
                 PP[n]["measured"], PP[n]["lo"], PP[n]["hi"]))
p242_path = os.path.join(AN, "budget_prospective_n242_result.json")
if os.path.exists(p242_path):
    P242 = json.load(open(p242_path))
    pros.append(("$N\\!=242$",
                 P242["pred_mc"], P242["pred_lo"], P242["pred_hi"],
                 P242["measured"], P242["meas_lo"], P242["meas_hi"]))

gsm = ("GSM8K", GS["pred"], GS["pred_lo"], GS["pred_hi"],
       GS["measured"], GS["mlo"], GS["mhi"])

# ── figure layout ─────────────────────────────────────────────────────
# 2.2in wide (0.40\textwidth); extra height for the legend moved below
# the axes — at this width a 5-entry 8pt legend cannot sit inside them.
fig = plt.figure(figsize=(2.2, 4.55))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1], hspace=0.08,
                        left=0.235, right=0.97, top=0.905, bottom=0.36)
ax_main = fig.add_subplot(gs[0])
ax_res  = fig.add_subplot(gs[1], sharex=ax_main)

lim = (0.58, 0.96)
diag = np.linspace(lim[0], lim[1], 200)

# ── shaded agreement band ────────────────────────────────────────────
BAND = 0.03   # ± tolerance
ax_main.fill_between(diag, diag - BAND, diag + BAND,
                     color=C_BAND, alpha=0.06, zorder=1,
                     label=None)
ax_main.plot(diag, diag, color=C_BAND, lw=0.7, ls="--", alpha=0.45,
             zorder=2)

# ── helper: draw one class of points ─────────────────────────────────
def draw(ax, pts, color, marker, size, filled=True, lw=0.8,
         zorder=4, alpha=1.0):
    """Scatter with error bars. Returns list of (x, y) for residual."""
    coords = []
    for _, p, plo, phi, m, mlo, mhi in pts:
        xerr = None if plo is None else [[p - plo], [phi - p]]
        yerr = [[m - mlo], [mhi - m]]
        ax.errorbar(p, m, xerr=xerr, yerr=yerr,
                    fmt="none", elinewidth=0.55, capsize=1.5,
                    ecolor=color, alpha=alpha * 0.55, zorder=zorder - 1)
        coords.append((p, m))
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    fc = color if filled else "white"
    ax.scatter(xs, ys, marker=marker, s=size, facecolor=fc,
               edgecolor=color, linewidths=lw, zorder=zorder, alpha=alpha)
    return coords


# ── main panel: scatter ───────────────────────────────────────────────
rc = draw(ax_main, retro, C_RETRO, "o", 20, zorder=4)
cc = draw(ax_main, cross, C_CROSS, "s", 20, zorder=4)
oc = draw(ax_main, pros,  C_OOS,   "D", 38, lw=1.0, zorder=6)

# excluded 1.5B — show as dashed ring
ax_main.scatter(xv["pred"], xv["measured"], marker="o", s=110,
                facecolor="none", edgecolor=C_EXCL, linewidths=1.0,
                zorder=5, alpha=0.7)

# GSM8K
gc = draw(ax_main, [gsm], C_GSM, "^", 34, filled=False, lw=1.2, zorder=5)

# ── legend: below the axes — a 5-entry 8pt legend cannot fit inside a
# 2.2in-wide panel without covering data ──────────────────────────────
from matplotlib.lines import Line2D

def leg_handle(marker, color, ms, filled=True):
    return Line2D([], [], marker=marker, linestyle="none", markersize=ms,
                  markerfacecolor=(color if filled else "none"),
                  markeredgecolor=color,
                  markeredgewidth=0.6 if filled else 1.0)

fig.legend(
    [leg_handle("o", C_RETRO, 4.5),
     leg_handle("s", C_CROSS, 4.5),
     leg_handle("D", C_OOS, 5.0),
     leg_handle("^", C_GSM, 5.5, filled=False),
     leg_handle("o", C_EXCL, 6.5, filled=False)],
    # long entries wrapped to two lines: one line at 8pt exceeds 2.2in
    ["retrospective, 0.5B (3)\n— in-sample",
     "cross-model, $N{=}16$ (3)\n— consistency check",
     f"out-of-sample, frozen ({len(pros)})\n— prospective tests",
     "GSM8K (1) — inconclusive",
     "Qwen 1.5B (1) — excluded"],
    loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1,
    frameon=False, fontsize=8.5, handlelength=1.0, handletextpad=0.5,
    labelspacing=0.45, borderaxespad=0.0)

# ── main panel: styling ───────────────────────────────────────────────
ax_main.set_xlim(*lim)
ax_main.set_ylim(*lim)
ax_main.set_aspect("equal")
ax_main.set_ylabel("Measured Jaccard")
ax_main.set_yticks([0.6, 0.7, 0.8, 0.9])
ax_main.set_xticks([0.6, 0.7, 0.8, 0.9])
ax_main.tick_params(labelbottom=False)

# subtle grid
ax_main.set_axisbelow(True)
ax_main.grid(True, which="major", color=C_GRID, linewidth=0.4)

# remove top and right spines
ax_main.spines["top"].set_visible(False)
ax_main.spines["right"].set_visible(False)

# band label
ax_main.text(0.95, 0.16, "$y = x$", transform=ax_main.transAxes,
             fontsize=8.5, color=C_BAND, alpha=0.75, ha="right", va="top",
             rotation=0)
ax_main.text(0.95, 0.085, "$\\pm 0.03$", transform=ax_main.transAxes,
             fontsize=8.5, color=C_BAND, alpha=0.55, ha="right", va="top")

# title (two lines: one line does not fit a 2.2in-wide figure at >= 8pt)
ax_main.set_title("Binomial budget model:\npredicted vs. measured\nset agreement",
                   fontsize=9, pad=5, fontweight="bold")

# ── residual panel ────────────────────────────────────────────────────
ax_res.axhline(0, color=C_BAND, lw=0.7, ls="--", alpha=0.45, zorder=1)
ax_res.axhspan(-BAND, BAND, color=C_BAND, alpha=0.05, zorder=0)

def plot_resid(pts, color, marker, size, filled=True, lw=0.8, alpha=1.0, zorder=4):
    for _, p, plo, phi, m, mlo, mhi in pts:
        resid = m - p
        yerr = [[m - mlo], [mhi - m]]
        ax_res.errorbar(p, resid, yerr=yerr,
                        fmt="none", elinewidth=0.5, capsize=1.2,
                        ecolor=color, alpha=alpha * 0.5, zorder=zorder - 1)
    xs = [t[1] for t in pts]
    rs = [t[4] - t[1] for t in pts]
    fc = color if filled else "white"
    ax_res.scatter(xs, rs, marker=marker, s=size, facecolor=fc,
                   edgecolor=color, linewidths=lw, zorder=zorder, alpha=alpha)

plot_resid(retro, C_RETRO, "o", 16, zorder=4)
plot_resid(cross, C_CROSS, "s", 16, zorder=4)
plot_resid(pros,  C_OOS,   "D", 30, lw=1.0, zorder=6)
plot_resid([gsm], C_GSM,   "^", 26, filled=False, lw=1.2, zorder=5)

# excluded 1.5B
ax_res.scatter(xv["pred"], xv["measured"] - xv["pred"], marker="o", s=75,
               facecolor="none", edgecolor=C_EXCL, linewidths=0.9,
               zorder=5, alpha=0.7)

ax_res.set_xlim(*lim)
ax_res.set_ylim(-0.055, 0.055)
ax_res.set_xlabel("Predicted expected Jaccard")
ax_res.set_ylabel("Residual", fontsize=8.5)
# fewer ticks instead of smaller text: the strip is ~0.6in tall in print
ax_res.set_yticks([-0.04, 0, 0.04])
ax_res.set_axisbelow(True)
ax_res.grid(True, which="major", color=C_GRID, linewidth=0.4)
ax_res.spines["top"].set_visible(False)
ax_res.spines["right"].set_visible(False)

# ── save ──────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    p = os.path.join(OUT, f"figure_9.{ext}")
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, dpi=300 if ext == "png" else None)
    print(f"wrote {p}")

counts = (f"retrospective {len(retro)}   cross-model {len(cross)}   "
          f"excluded-1.5B 1   out-of-sample {len(pros)}   gsm8k 1")
print(counts)
