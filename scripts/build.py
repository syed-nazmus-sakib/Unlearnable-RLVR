#!/usr/bin/env python3
"""Build script for the "Unlearnable, or Unmeasured?" project page.

Reads the camera-ready paper source, then:
  1. converts paper figures to WebP under static/img/figures/
  2. renders the paper's tables as HTML fragments and injects them into
     index.html between <!-- @build:NAME --> markers.

Every table number is parsed from the paper's LaTeX, so nothing in the
tables is retyped by hand.

Usage:  python3 scripts/build.py
"""
import html
import re
from pathlib import Path

from PIL import Image

SITE = Path(__file__).resolve().parent.parent
PAPER_DIR = SITE.parent
PAPER_TEX = PAPER_DIR / "neurips_2026.tex"

FIG_OUT = SITE / "static/img/figures"

# (source relative to the paper folder, published name)
FIGURES = [
    ("experimental_results_bundle/figure2.png", "aggregation"),
    ("experimental_results_bundle/figure3_baseline.png", "slow_learning"),
    ("experimental_results_bundle/figure1.png", "set_construction"),
    ("experimental_results_bundle/figure8.png", "budget_curve"),
    ("experimental_results_bundle/figure9.png", "predicted_measured"),
    ("experimental_results_bundle/figure6.png", "matched_gradient"),
    ("NeurIPS_fig/figure10.png", "propagation"),
]


# ----------------------------------------------------------------------------
# LaTeX parsing
# ----------------------------------------------------------------------------
def body_only(tex):
    """Drop the \\iffalse-wrapped template instructions after the appendix."""
    return tex.split(r"\iffalse % Remaining NeurIPS template", 1)[0]


def table_block(tex, label):
    """Return the tabular(x) body of the table carrying \\label{label}.

    Captions in this paper sit below the tabular, so search backwards from
    the label to the enclosing table environment."""
    at = tex.index(rf"\label{{{label}}}")
    start = tex.rindex(r"\begin{table", 0, at)
    end = tex.index(r"\end{tabular", start)
    return tex[start:end]


def latex_to_html(s):
    s = re.sub(r"(?<!\\)%.*$", "", s, flags=re.M)
    s = re.sub(r"\\rowcolor\{[^}]*\}", "", s)
    s = s.replace(r"\%", "%").replace(r"\_", "_").replace("~", " ")
    s = s.replace(r"vs.\ ", "vs. ").replace(r"\ ", " ")
    s = s.replace(r"\(", "").replace(r"\)", "").replace(r"\times", "×")
    s = s.replace("---", "—").replace("--", "–")
    s = html.escape(s, quote=False)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\path\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\bN=(\d+)", r"<i>N</i>&nbsp;=&nbsp;\1", s)
    s = re.sub(r"^-(?=\d)", "−", s.strip())
    return s.replace("$", "").strip()


def parse_table(tex, label):
    """Parse a booktabs table into (header, rows). A row that is a single
    \\multicolumn becomes a section row."""
    block = table_block(tex, label)
    body = block.split(r"\toprule", 1)[1].split(r"\bottomrule", 1)[0]
    header, rows = None, []
    for line in re.split(r"\\\\", body.replace(r"\midrule", "")):
        line = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", line)
        line = re.sub(r"(?<!\\)%.*$", "", line, flags=re.M)
        line = re.sub(r"\\rowcolor\{[^}]*\}", "", line).strip()
        if not line:
            continue
        m = re.match(r"\\multicolumn\{\d+\}\{(?:[^{}]|\{[^{}]*\})*\}\{(.*)\}$", line, re.S)
        if m:
            rows.append(("section", latex_to_html(m.group(1))))
            continue
        cells = [latex_to_html(c) for c in line.split("&")]
        if header is None:
            header = cells
        else:
            rows.append(("row", cells))
    return header, rows


def parse_budget_array(tex):
    """Predicted Jaccard at representative budgets (appendix B.1 array)."""
    at = tex.index(r"\begin{array}{c|ccccccc}")
    block = tex[at:tex.index(r"\end{array}", at)]
    ns = re.search(r"N\s*&(.*?)\\\\", block, re.S).group(1)
    js = re.search(r"\\mathrm\{Deep\}\s*&(.*)$", block, re.S).group(1)
    ns = [int(x) for x in re.findall(r"\d+", ns)]
    js = [x.strip() for x in js.split("&")]
    assert len(ns) == len(js), (ns, js)
    return list(zip(ns, js))


# ----------------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------------
def to_webp(src, dst, max_w, quality=88):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return Image.open(dst).size
    im = Image.open(src)
    im = im.convert("RGBA") if im.mode in ("P", "LA", "RGBA") else im.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(dst, "WEBP", quality=quality, method=6)
    return im.size


def build_figures():
    return {name: to_webp(PAPER_DIR / src, FIG_OUT / f"{name}.webp", 1800) for src, name in FIGURES}


# ----------------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------------
def render_table(header, rows, cls="data", row_class=None):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = []
    for kind, cells in rows:
        if kind == "section":
            body.append(f'<tr class="section"><td colspan="{len(header)}">{cells}</td></tr>')
            continue
        attr = f' class="{row_class(cells)}"' if row_class and row_class(cells) else ""
        body.append(f"<tr{attr}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def render_budget(pairs):
    head = "<th><i>N</i> rollouts per prompt</th>" + "".join(f"<th>{n}</th>" for n, _ in pairs)
    cells = "".join(f'<td class="heat" style="--a:{(float(j) - 0.6) / 0.35 * 0.7:.2f}">{j}</td>'
                    for _, j in pairs)
    return (f'<table class="data budget"><thead><tr>{head}</tr></thead>'
            f'<tbody><tr><td>Predicted Jaccard</td>{cells}</tr></tbody></table>')


def inject(page, name, fragment):
    pat = re.compile(rf"(<!-- @build:{name} -->)(.*?)(<!-- /@build:{name} -->)", re.S)
    if not pat.search(page):
        raise KeyError(f"marker {name} missing from index.html")
    return pat.sub(lambda m: m.group(1) + "\n" + fragment + "\n" + m.group(3), page)


def main():
    tex = body_only(PAPER_TEX.read_text())
    sizes = build_figures()

    index = SITE / "index.html"
    page = index.read_text()

    header, rows = parse_table(tex, "tab:controls")
    page = inject(page, "controls", render_table(header, rows, "data wrap"))

    header, rows = parse_table(tex, "tab:appendix-retrospective")
    page = inject(page, "retrospective", render_table(
        header, rows, row_class=lambda c: "excluded" if "1.5B" in c[0] else None))

    header, rows = parse_table(tex, "tab:config-audit")
    page = inject(page, "config-audit", render_table(header, rows, "data wrap"))

    budget = parse_budget_array(tex)
    page = inject(page, "budget", render_budget(budget))
    index.write_text(page)

    print(f"figures={len(sizes)} budget={budget}")


if __name__ == "__main__":
    main()
