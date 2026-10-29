<div align="center">

# Unlearnable, or Unmeasured?
### On the Reliability of Difficulty Labels in RLVR

[![arXiv](https://img.shields.io/badge/arXiv-2609.40115-b31b1b.svg)](https://arxiv.org/abs/2609.40115)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://syed-nazmus-sakib.github.io/Unlearnable-RLVR/)
[![Venue](https://img.shields.io/badge/NeurIPS%202026-Workshop-6f42c1.svg)](https://neurips.cc/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Chandak Chakma](https://github.com/)<sup>1</sup> · [Syed Nazmus Sakib](https://github.com/syed-nazmus-sakib)<sup>1</sup> · [Nafiul Haque](https://github.com/)<sup>1</sup> · [Shifat E. Arman](mailto:shifatearman@du.ac.bd)<sup>1,2,\*</sup>**

<sup>1</sup>Department of Robotics and Mechatronics Engineering, University of Dhaka, Bangladesh
<sup>2</sup>Department of Computer Science, University of Oxford, UK

<sub><sup>\*</sup>Work done at University of Dhaka. Corresponding author.</sub>

**[📄 Paper](https://arxiv.org/abs/2609.40115)** · **[🌐 Project Page](https://syed-nazmus-sakib.github.io/Unlearnable-RLVR/)** · **[📊 Reproduction Guide](docs/REPRODUCTION.md)**

</div>

---

## Overview

Recent work identifies a set of RLVR prompts as **"unlearnable"** — hard problems that occasionally
produce correct rollouts yet barely improve during training. We revisit that claim and find a more
nuanced picture.

The prompts *do* learn, at roughly **one third** of the learnable rate. But the set used to define
them is far less reproducible than the label implies, and the gradient evidence offered to explain
them is partly an artifact of **unequal estimator sample counts**.

> **The core message.** Difficulty labels are estimated from a finite number of sampled responses.
> At the budgets commonly used, they are not stable enough to classify an individual prompt as
> unlearnable, nor to support mechanistic claims built on that classification.

## Key findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Slow learning, not non-learning.** The affected cohort improves, just slowly. | Within-prompt reward slope per 100 steps: `+0.0422 [+0.0259, +0.0590]` (D<sub>u</sub>) vs `+0.1119` (learnable) vs `+0.3800` (easy), on a fixed cohort with zero prompts dropped |
| 2 | **The aggregation rule moves the set more than the training seed does.** | Holding 292 candidates fixed: the published rule returns **74** prompts, an alternative aggregation returns **228**, individual seeds return **145–150** |
| 3 | **The published construction is not a consistent estimator.** Requiring the label in every run drives the set toward `p(x)=0`, and the `no_reward` exclusion then removes exactly those prompts. | Of 218 prompts excluded as `no_reward`, **116 are solved at least once** under a later `N=128` evaluation |
| 4 | **Most cross-seed disagreement is evaluation noise.** | Re-evaluating one fixed checkpoint reproduces ~**81%** of the apparent cross-seed effect (Jaccard `0.798` vs `0.751`) |
| 5 | **Reproducibility is predictable before the budget is spent.** | Two *prospectively frozen* predictions both landed: `N=40` predicted `0.8131 [0.7699, 0.8533]`, measured `0.8450`; `N=64` predicted `0.8397`, measured `0.8742` |
| 6 | **The gradient gap shrinks once sample count is matched.** | Easy-to-D<sub>u</sub> cosine ratio falls from `2.327×` to `1.539× [1.26, 1.88]` at matched `K=11` — weakened, not eliminated |

### How much evaluation does a difficulty label need?

| Rollouts per prompt | Predicted Jaccard agreement |
|---:|---:|
| 8 | 0.6322 |
| 16 | 0.7265 |
| 128 | 0.8759 |
| **~40** | **0.80** (first budget that holds) |
| 242 *(projected)* | 0.90 |
| 2,111 *(projected)* | 0.95 |

Budgets are specific to this model, dataset, and threshold — the contribution is the **procedure**
for computing them, not the particular numbers. The 0.90/0.95 figures lie beyond the `N=128`
fitting range and should not be acted on without direct measurement.

## Repository layout

```
.
├── analysis/            Analysis code, grouped by paper section
│   ├── set_instability/   §3.3  seed/aggregation instability
│   ├── eval_noise/        §3.3  same-checkpoint evaluation control
│   ├── budget_model/      §3.4  sampling-budget model + frozen predictions
│   ├── gradients/         §3.5  matched-K gradient similarity
│   ├── optimization/      §3.5  exposure and GRPO normalization diagnostics
│   ├── removal/           §3.6  difficulty-removal training arms
│   ├── routing/           App.  sequential certification for difficulty routing
│   ├── cohort/                  guarded cohort construction
│   ├── cross_model/             GSM8K and second-model consistency checks
│   ├── cross_model_llama/       Llama-3.2-3B checks
│   └── cross_model_qwen/        Qwen2.5-1.5B diagnostic
├── figures/             Figure-generation scripts (matplotlib → PNG + PDF)
├── artifacts/           Small JSON/CSV results the analyses consume
├── training/            GRPO training: repro scripts, configs, upstream diff
├── paper/               Camera-ready PDF and the figures it uses
├── docs/                Reproduction guide, preregistrations, audits
├── lib/                 Shared helpers
│
└── Each code directory carries its own analysis/ folder holding the
    derived JSON/CSV inputs that its scripts read.
└── index.html, static/, scripts/     Project page (GitHub Pages)
```

## Setup

```bash
git clone https://github.com/syed-nazmus-sakib/Unlearnable-RLVR.git
cd Unlearnable-RLVR
pip install numpy scipy pandas matplotlib
```

Analysis and figure code needs only the scientific Python stack. Re-running **training** or
**generation** additionally requires the RLVR trainer — see [`training/`](training/).

## Reproducing the paper

### What runs out of the box

**All seven paper figures regenerate from this repository alone.** The figure scripts read the
derived summary files shipped in each directory's `analysis/` folder:

```bash
cd figures
python make_figures_guarded.py   # Figure 2
python make_figure8.py           # Figure 3a
python make_figure_9.py          # Figures 3b and 6
python make_figure6.py           # Figure 4
python make_figure7.py           # Figure 5
python make_figure10.py          # Figure 7
```

Each writes both a PNG and a vector PDF into `figures_pub/`. Regenerated figures are identical in
content to those in the paper; pixel dimensions may differ slightly, since the embedded versions
were exported at different bbox/DPI settings.

### What needs the rollout corpus

Be aware of a real limitation: **46 of the 69 analysis scripts read raw rollout generations**
(`unlearnability-rlvr/results/`) rather than derived summaries. That corpus is **8.7 GB across 115
files, with individual files up to 307 MB** — past GitHub's per-file limit, so it is not hosted
here.

What this means in practice:

| You want to | Possible with this repo alone? |
|---|---|
| Regenerate every paper figure | **Yes** |
| Inspect exactly how each number was computed | **Yes** — the code is all here |
| Re-derive numbers from raw rollouts | No — needs the 8.7 GB corpus |
| Re-run training or generation | No — needs a GPU and the trainer |

A few scripts also import `common` or `math_verifier`, which live in the trainer repository; run
those from inside a patched trainer checkout.

We are glad to share the rollout corpus — please open an issue or contact the corresponding author.

### Training

Experiments use **Qwen2.5-0.5B** trained with **GRPO** on a 1,023-prompt MATH levels 1–4 subset,
`G = 8` rollouts per prompt, on a single RTX 3090 Ti (24 GB). Each run takes roughly 16–24 hours.

Training builds on [`yulinchen99/unlearnability-rlvr`](https://github.com/yulinchen99/unlearnability-rlvr)
at commit [`1dce7fe`](https://github.com/yulinchen99/unlearnability-rlvr/commit/1dce7fec92e7685929a5efbe73ff2091e7d474a1).
Rather than vendoring that repository, we ship our modifications as a patch:

```bash
git clone https://github.com/yulinchen99/unlearnability-rlvr.git
cd unlearnability-rlvr && git checkout 1dce7fec92e7685929a5efbe73ff2091e7d474a1
git apply ../training/upstream-modifications.diff
bash ../training/repro/setup_env.sh
bash ../training/repro/train_1gpu.sh
```

Our configuration differs from the original in response caps, epochs, and dataset subset. Every
differing constant is enumerated in [`docs/constants_audit.md`](docs/constants_audit.md) and in the
paper's appendix — we report a **reproduction**, not a byte-exact replication.

## Research practice

Experiments in this project were **preregistered**: protocols, thresholds, branches, and stopping
rules were frozen before outcomes were computed, and registered verdicts were reported whether or
not they were convenient.

[`docs/PREREGISTRATIONS.md`](docs/PREREGISTRATIONS.md) reproduces the frozen registration for the
difficulty-routing experiment reported in the appendix, including its recorded verdicts and the
provisional gates it was judged against. [`docs/B1_REPORT.md`](docs/B1_REPORT.md) and
[`docs/B2_REPORT.md`](docs/B2_REPORT.md) are the corresponding generated reports.

## Citation

```bibtex
@article{chakma2026unlearnable,
  title={Unlearnable, or Unmeasured? On the Reliability of Difficulty Labels in RLVR},
  author={Chakma, Chandak and Sakib, Syed Nazmus and Haque, Nafiul and Arman, Shifat E},
  journal={arXiv preprint arXiv:2609.40115},
  year={2026}
}
```

## Acknowledgements

This work builds directly on the unlearnability analysis of Chen et al. (2026) and on
[verl](https://github.com/volcengine/verl). We thank the authors for releasing their code, without
which this reanalysis would not have been possible.

## License

Released under the [MIT License](LICENSE). The paper PDF is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
