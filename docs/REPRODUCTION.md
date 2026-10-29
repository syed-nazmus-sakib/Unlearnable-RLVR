# Reproduction guide

Every number in the paper is produced by a script in this repository.

**Figures regenerate from this repository alone** — the figure scripts read the derived summary
files shipped in each directory's `analysis/` folder.

**Most analysis scripts do not.** 46 of the 69 Python files read raw rollout generations from
`unlearnability-rlvr/results/`, a corpus of 8.7 GB across 115 files (individual files up to
307 MB) that exceeds GitHub's limits and is not hosted here. Those scripts are included because
they document exactly how each number was computed, not because they run unaided. Contact the
corresponding author for the rollout corpus.

## Requirements

```bash
pip install numpy scipy pandas matplotlib
```

Training and generation additionally require the RLVR trainer — see [Training](#training).

## Figures

| Paper figure | Script | Output |
|---|---|---|
| Figure 1 — training reward by cohort | *not included* (see note) | `paper/figures/figure3_baseline.png` |
| Figure 2 — set instability | `figures/make_figures_guarded.py` | `figure1.{png,pdf}`, `figure2.{png,pdf}` |
| Figure 3a — reproducibility vs budget | `figures/make_figure8.py` | `figure8.{png,pdf}` |
| Figure 3b — predicted vs measured | `figures/make_figure_9.py` | `figure_9.{png,pdf}` |
| Figure 4 — matched-K gradient similarity | `figures/make_figure6.py` | `figure6.{png,pdf}` |
| Figure 5 — uncertainty propagation | `figures/make_figure7.py` | `figure7.pdf` |
| Figure 6 — budget-model validation | `figures/make_figure_9.py` | `figure_9.{png,pdf}` |
| Figure 7 — interval width vs budget | `figures/make_figure10.py` | `figure10.pdf` |

> **Note on Figure 1.** This figure was produced outside the analysis pipeline during an early
> stage of the project and its generating script is not part of this release. The rendered figure
> is included in `paper/figures/`, and the underlying slopes it plots are reproducible from
> `analysis/optimization/`. We flag this rather than imply full coverage.

Figure scripts write both PNG and vector PDF. Figures 3b and 6 are the same plot at different
sizes, so both come from `make_figure_9.py`.

## Tables and numbers

| Paper element | Script |
|---|---|
| §3.2 within-prompt slopes, exposure, 1/σ normalization | `analysis/optimization/` |
| §3.3 set sizes under each aggregation rule | `analysis/set_instability/` |
| §3.3 same-checkpoint evaluation control | `analysis/eval_noise/` |
| §3.4 budget model, Jaccard curve, Table 2 | `analysis/budget_model/budget_rule.py`, `budget_rule_crossmodel.py` |
| §3.4 equal-cost deep vs split allocation | `analysis/budget_model/deep_vs_shallow.py` |
| §3.4 frozen N=40 / N=64 predictions | `analysis/budget_model/validate_projection.py` |
| §3.5 matched-K gradient ratios | `analysis/gradients/matchedK_similarity.py`, `joint_bootstrap.py` |
| §3.6 / Table 4 removal arms | `analysis/removal/` |
| Appendix — sequential certification routing | `analysis/routing/` |
| Appendix — 5,120-token sensitivity | `analysis/sensitivity/pool_5120_numbers.py` |
| Table 5 — configuration audit | `analysis/sensitivity/constants_audit.py` |
| Cross-model consistency checks | `analysis/cross_model/`, `cross_model_llama/`, `cross_model_qwen/` |

## Quick start

Regenerate every paper figure (needs only numpy/scipy/matplotlib):

```bash
cd figures
for f in make_figures_guarded make_figure8 make_figure_9 make_figure6 make_figure7 make_figure10; do
    python "$f.py"
done
```

Outputs land in `figures/figures_pub/` as both PNG and vector PDF.

## Training

See the Training section of the [main README](../README.md). Summary:

- **Model** Qwen2.5-0.5B · **Algorithm** GRPO · **Group size** `G = 8`
- **Data** 1,023-prompt MATH levels 1–4 subset · **Threshold** `τ = 0.1`
- **Hardware** one RTX 3090 Ti (24 GB) · **Wall clock** ~16–24 h per run

Apply `training/upstream-modifications.diff` to upstream commit `1dce7fe`, then run
`training/repro/train_1gpu.sh`. Configuration differences from the original are enumerated in
[`constants_audit.md`](constants_audit.md).

## Scripts that need the trainer

A few analyses import `common` or `math_verifier`, which live inside the trainer repository rather
than here. Run those from inside the patched trainer checkout. All budget-model, gradient, and
figure code is self-contained.

## Caveats

- Required rollout budgets are specific to this model, dataset, and threshold. The procedure
  transfers; the constants do not.
- The 0.90 and 0.95 budgets (242 and 2,111) are projections beyond the `N = 128` fitting range.
- Results are a reproduction of Chen et al. (2026), not a byte-exact replication.
