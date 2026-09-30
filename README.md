# Unlearnable, or Unmeasured? project page

Project page for *Unlearnable, or Unmeasured? On the Reliability of Difficulty Labels in RLVR*
(NeurIPS 2026 Workshop: Transitioning from Pre-Training to Post-Training).
Plain static HTML/CSS, served by GitHub Pages. No build step is needed to serve it.

## Layout
- `index.html`: the page. Blocks between `<!-- @build:NAME -->` markers are generated; edit everything else by hand.
- `static/css/style.css`: styles.
- `static/img/figures/`: paper figures (WebP).
- `scripts/build.py`: regenerates the marked blocks and the figures.

## Regenerating
```
python3 scripts/build.py
```
It reads the tables straight from `../neurips_2026.tex` (robustness controls, retrospective
budget-model checks, predicted-Jaccard budget array, configuration audit) and converts the PNG
figures in `../experimental_results_bundle/` and `../NeurIPS_fig/` to WebP.

## Preview
```
python3 -m http.server 8000   # then open http://localhost:8000
```
