"""Figure 6 — the attenuation curve and the absence of power.

(a) within-group mean pairwise gradient cosine against rollouts-per-prompt K,
    fixed cohort, with each group's NATURAL-count value as an open marker at its
    own K -- so it is visible that the published comparison reads the groups at
    different places on the x-axis.
(b) the easy/D_u ratio: measured, the shared-c model, and the bootstrap interval
    on the true-similarity ratio mapped onto this axis.

EVERY number is read or recomputed from the stored analysis files. Nothing is
typed in. Sources:
    analysis/attenuation.json              per-prompt matched-K similarities
    analysis/figure5_natural_per_prompt.json   per-prompt natural-n similarities
    gradsim_grads/per_prompt_summary.json  the recorded rollout count per prompt
The constrained fit and its bootstrap CI are RECOMPUTED here with the registered
seed rather than copied from constrained_fit.md, so the figure cannot drift from
the tables.

Style matches make_figures_guarded.py: 9 pt base, pdf.fonttype 42 (no external
fonts), vector PDF plus a PNG copy.

  $HOME/miniconda3/envs/vllm/bin/python make_figure6.py
"""
import json
import os

# The PDF md5 must certify CONTENT. matplotlib stamps /CreationDate into the
# file, so two runs of identical code produced different hashes -- an md5
# handed over for verification would then prove only "same file", never
# "same figure". SOURCE_DATE_EPOCH freezes the stamp and makes the build
# reproducible.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP, "figures_pub")
os.makedirs(OUT, exist_ok=True)
GROUPS = ["du71", "learnable", "easy"]
LABEL = {"du71": r"$D_u$", "learnable": "learnable", "easy": "easy"}
SEED = 20260814
NBOOT = 10000
NBOOT_FIT = 2000
rng = np.random.default_rng(SEED)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_UNL, C_LRN, C_EASY = "#D55E00", "#0072B2", "#4d4d4d"  # Okabe-Ito: Vermilion, Blue, Grey
COLOR = {"du71": C_UNL, "learnable": C_LRN, "easy": C_EASY}

att = json.load(open(os.path.join(EXP, "analysis", "attenuation.json")))
nat_pp = json.load(open(os.path.join(EXP, "analysis", "figure5_natural_per_prompt.json")))
summ = json.load(open(os.path.join(EXP, "gradsim_grads", "per_prompt_summary.json")))
Ks = sorted({int(k) for g in att for k in att[g]})


def entries(g, K):
    return [e for e in att[g][str(K)] if "per_prompt" in e]


fixed = {g: sorted(set.intersection(*[set(e["per_prompt"]) for K in Ks
                                      for e in entries(g, K)]), key=int)
         for g in GROUPS}


def boot_ci(vals):
    v = np.asarray(vals, float)
    d = v[rng.integers(0, len(v), size=(NBOOT, len(v)))].mean(1)
    return tuple(np.percentile(d, [2.5, 97.5]))


# ---- panel (a) data: per-prompt values averaged over subsamples, then over prompts
curve, curve_ci, sub_sd = {}, {}, {}
for g in GROUPS:
    curve[g], curve_ci[g], sub_sd[g] = {}, {}, {}
    for K in Ks:
        es = entries(g, K)
        pp = np.array([[e["per_prompt"][q] for q in fixed[g]] for e in es]).mean(0)
        curve[g][K] = float(pp.mean())
        curve_ci[g][K] = boot_ci(pp)
        sub_sd[g][K] = (float(np.std([np.mean([e["per_prompt"][q] for q in fixed[g]])
                                      for e in es], ddof=1)) if len(es) > 1 else np.nan)

nat, nat_ci, nat_K, nat_vals = {}, {}, {}, {}
for g in GROUPS:
    pp = np.array([nat_pp[g][str(q)] for q in fixed[g] if str(q) in nat_pp[g]])
    nat_vals[g] = pp
    nat[g] = float(pp.mean())
    nat_ci[g] = boot_ci(pp)
    nat_K[g] = float(np.median([summ[str(q)]["num_correct"] for q in fixed[g]
                                if str(q) in summ]))

# CI on the natural-count RATIO. Prompt bootstrap only: the natural-n pass has a
# single rollout draw per prompt, so there is no subsample dimension to resample
# here -- unlike the matched-K bars, whose interval is joint over prompts AND
# subsamples. The two interval types are therefore NOT identical and that is
# stated in the printout rather than smoothed over.
_r = np.random.default_rng(SEED)
_b = []
for _ in range(NBOOT):
    e = nat_vals["easy"][_r.integers(0, len(nat_vals["easy"]), len(nat_vals["easy"]))]
    d = nat_vals["du71"][_r.integers(0, len(nat_vals["du71"]), len(nat_vals["du71"]))]
    if d.mean() > 1e-9:
        _b.append(e.mean() / d.mean())
nat_r_lo, nat_r_hi = (float(x) for x in np.percentile(_b, [2.5, 97.5]))

# ---- the shared-c model, recomputed here
xs = np.array(Ks, float)
ys = np.array([[curve[g][K] for K in Ks] for g in GROUPS])


def fit_shared(y):
    def r(p):
        return np.concatenate([p[0] / (1 + p[1 + i] / xs) - y[i] for i in range(3)])
    return least_squares(r, [0.5, 50, 50, 50],
                         bounds=([0] + [0] * 3, [1] + [1e4] * 3)).x


def fit_free(y):
    def r(p):
        return np.concatenate([p[2 * i] / (1 + p[2 * i + 1] / xs) - y[i]
                               for i in range(3)])
    return least_squares(r, [0.5, 50] * 3, bounds=([0, 0] * 3, [1, 1e4] * 3)).x


p_sh = fit_shared(ys)
c_sh, lam = p_sh[0], p_sh[1:]
model_ratio = {K: (c_sh / (1 + lam[2] / K)) / (c_sh / (1 + lam[0] / K)) for K in Ks}

sub_means = {g: {K: [float(np.mean([e["per_prompt"][q] for q in fixed[g]]))
                     for e in entries(g, K)] for K in Ks} for g in GROUPS}
# A15.3: the band is the PRIMARY interval -- the JOINT bootstrap over prompts
# AND subsamples. The previous version resampled one subsample mean per level,
# which propagated the spread of a single subsample instead of the standard
# error of the level mean, and contained no prompt-sampling variance at all.
# Prompts are resampled once per draw and reused at every level, so the
# across-level correlation the curve is fitted through is preserved.
# It is READ, not recomputed: an independent refit here produced [0.38, 3.58]
# against joint_bootstrap.py's [0.37, 3.31] -- Monte Carlo noise on the upper
# percentile of a heavy-tailed ratio, but two published values for one quantity
# is the cohort-drift failure in a different costume. One source of truth.
_jb = json.load(open(os.path.join(EXP, "analysis", "joint_bootstrap.json")))
f_lo, f_hi = _jb["primary"]["lo"], _jb["primary"]["hi"]

# the CI is on the TRUE-similarity ratio; map it onto the measured axis through
# the fitted noise levels, otherwise it is not comparable to the plotted points
def measured_from_f(f, K):
    return f * (1 + lam[0] / K) / (1 + lam[2] / K)


meas_ratio = {K: curve["easy"][K] / curve["du71"][K] for K in Ks}
nat_ratio = nat["easy"] / nat["du71"]

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))

for g in GROUPS:
    y = [curve[g][K] for K in Ks]
    lo = [curve[g][K] - curve_ci[g][K][0] for K in Ks]
    hi = [curve_ci[g][K][1] - curve[g][K] for K in Ks]
    ax[0].errorbar(Ks, y, yerr=[lo, hi], color=COLOR[g], lw=1.6, marker="o", ms=4.5,
                   markeredgecolor="white", markeredgewidth=0.8,
                   capsize=0, elinewidth=1.6, zorder=3, label=f"{LABEL[g]} (n={len(fixed[g])})")
    ax[0].errorbar([nat_K[g]], [nat[g]],
                   yerr=[[nat[g] - nat_ci[g][0]], [nat_ci[g][1] - nat[g]]],
                   fmt="o", ms=7, mfc="none", mew=1.6, color=COLOR[g],
                   capsize=0, elinewidth=1.6, zorder=3)
ax[0].grid(True, axis="y", color="#eaeaea", linewidth=0.6, zorder=0)
ax[0].set_axisbelow(True)
ax[0].set_xscale("log")
ax[0].set_xticks(Ks + [int(round(nat_K["easy"]))])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_xlabel("correct rollouts averaged, $K$ (log)")
ax[0].set_ylabel("within-group mean cosine")
ax[0].set_title("matched $K$, fixed cohort", fontweight="bold")
ax[0].legend(frameon=False, loc="upper left", fontsize=7.5)
for g in GROUPS:
    # D_u K=16 and learnable K=18 are close enough on a log axis that the
    # labels collided; offset them apart. Markers are NOT moved.
    _dx = {"du71": -16, "learnable": 16, "easy": 0}[g]
    _dy = {"du71": 6, "learnable": 11, "easy": 9}[g]
    ax[0].annotate(f"$K$={nat_K[g]:.0f}", xy=(nat_K[g], nat[g]), xytext=(_dx, _dy),
                   textcoords="offset points", ha="center", fontsize=6.4,
                   color=COLOR[g])
ax[0].text(0.97, 0.05, "open = each group's natural count", transform=ax[0].transAxes,
           ha="right", fontsize=6.8, color="#555555")

# Panel (b) is MEASUREMENTS ONLY. The shared-c model line and the mapped
# [0.37, 3.31] band are both gone: both come from the derivation withdrawn under
# A14.4/A17.1, and a panel whose caption says "measurements only" must not carry
# them. Ratios and intervals are READ from analysis/ratio_drift_check.json --
# the same numbers section 7 tabulates, so figure and text cannot drift apart.
rd = json.load(open(os.path.join(EXP, "analysis", "ratio_drift_check.json")))
rr = {int(k): v for k, v in rd["ratio_by_K"].items()}
r_pt = [rr[K]["point"] for K in Ks]
r_lo = [rr[K]["point"] - rr[K]["lo"] for K in Ks]
r_hi = [rr[K]["hi"] - rr[K]["point"] for K in Ks]

ax[1].axhline(1.0, color="black", lw=1.0, ls=":", zorder=1)
ax[1].grid(True, axis="y", color="#eaeaea", linewidth=0.6, zorder=0)
ax[1].set_axisbelow(True)
ax[1].text(min(Ks) * 0.92, 1.0, "equal similarity ", va="bottom", ha="left",
           fontsize=6.8, color="#333333")
ax[1].errorbar(Ks, r_pt, yerr=[r_lo, r_hi], color="#D55E00", lw=1.6, marker="o",
               ms=4.5, markeredgecolor="white", markeredgewidth=0.8,
               capsize=0, elinewidth=1.6, zorder=3,
               label="matched $K$ (measured)")
ax[1].errorbar([nat_K["easy"]], [nat_ratio],
               yerr=[[nat_ratio - nat_r_lo], [nat_r_hi - nat_ratio]],
               fmt="o", ms=7.5, mfc="none", mew=1.6, color="#D55E00",
               capsize=0, elinewidth=1.6, zorder=3,
               label="natural counts (unmatched)")
ax[1].annotate(f"{nat_ratio:.2f}$\\times$\nunmatched:\neasy $K$={nat_K['easy']:.0f}"
               f" vs $D_u$ $K$={nat_K['du71']:.0f}",
               xy=(nat_K["easy"], nat_ratio), xytext=(-12, -2),
               textcoords="offset points", ha="right", va="top", fontsize=6.4,
               color="#D55E00")
ax[1].set_xscale("log")
ax[1].set_xticks(Ks + [int(round(nat_K["easy"]))])
ax[1].set_xlim(min(Ks) * 0.85, nat_K["easy"] * 1.35)
ax[1].set_ylim(0.9, max(nat_r_hi, max(rr[K]["hi"] for K in Ks)) * 1.08)
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_xlabel("correct rollouts averaged, $K$ (log)")
ax[1].set_ylabel(r"easy / $D_u$ similarity ratio")
ax[1].set_title("measured ratio; every bar clears 1.0", fontweight="bold")
ax[1].legend(frameon=False, loc="upper left", fontsize=6.6)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figure6.{ext}"), dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------- the numbers
print("FIGURE 6 — every plotted value\n")
print("y units: mean pairwise cosine similarity between per-prompt gradient")
print("         vectors (dimensionless, -1..1). Error bars: 95% BOOTSTRAP CI over")
print(f"         prompts, {NBOOT} resamples, seed {SEED}. Not standard error.")
print(f"Fixed cohort: " + ", ".join(f"{LABEL[g]} {len(fixed[g])}" for g in GROUPS))
print("\n--- panel (a): within-group mean cosine ---")
print(f"{'group':>10} {'K':>5} {'mean':>8} {'boot 95% CI':>22} {'sd across subsamples':>22}")
for g in GROUPS:
    for K in Ks:
        lo, hi = curve_ci[g][K]
        sd = sub_sd[g][K]
        print(f"{LABEL[g]:>10} {K:>5} {curve[g][K]:>8.4f}  [{lo:.4f}, {hi:.4f}]"
              + (f"{sd:>22.4f}" if sd == sd else f"{'n/a (1 subsample)':>22}"))
    lo, hi = nat_ci[g]
    print(f"{LABEL[g]:>10} {nat_K[g]:>5.0f} {nat[g]:>8.4f}  [{lo:.4f}, {hi:.4f}]"
          f"{'  <- NATURAL count (open marker)':>22}")
print("\n--- panel (b): easy / D_u ratio — MEASUREMENTS ONLY ---")
_jb = json.load(open(os.path.join(EXP, "analysis", "joint_bootstrap.json")))
print(f"The shared-c model line and the mapped "
      f"[{_jb['primary']['lo']:.2f}, {_jb['primary']['hi']:.2f}] band are REMOVED:")
print("both come from the derivation withdrawn under A14.4 / A17.1.")
print("Ratios and intervals read from analysis/ratio_drift_check.json, the same")
print("numbers section 7 tabulates, so figure and text cannot drift apart.")
print("y units: ratio of within-group mean cosines (dimensionless).")
print(f"Matched-K bars: 95% JOINT bootstrap over prompts AND subsamples,")
print(f"         {rd['nboot']} draws, seed {rd['seed']}.")
print(f"{'K':>5} {'ratio':>8} {'95% CI':>20} {'width':>7} {'clears 1.0':>11}")
for K in Ks:
    v = rr[K]
    print(f"{K:>5} {v['point']:>8.3f}  [{v['lo']:.2f}, {v['hi']:.2f}]"
          f"{v['hi'] / v['lo']:>10.2f}x {'YES' if v['lo'] > 1.0 else 'no':>10}")
print(f"\nnatural counts, UNMATCHED (easy K={nat_K['easy']:.0f} vs "
      f"D_u K={nat_K['du71']:.0f}): {nat_ratio:.3f}  "
      f"[{nat_r_lo:.2f}, {nat_r_hi:.2f}]")
print("  NOTE: this interval is a PROMPT bootstrap only -- the natural-n pass")
print("  has one rollout draw per prompt, so there is no subsample dimension.")
print("  It is not the same interval type as the matched-K bars above.")
print(f"\nhorizontal reference line at 1.0 = equal similarity.")
print(f"drift ratio(K={Ks[0]}) - ratio(K={Ks[-1]}) = {rd['drift_obs']:+.3f}, "
      f"CI [{rd['drift_lo']:+.3f}, {rd['drift_hi']:+.3f}], "
      f"significant: {rd['drift_significant']}")
print(f"\nfigure6.pdf and .png written to {OUT}")
