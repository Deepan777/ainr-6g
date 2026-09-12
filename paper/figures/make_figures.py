"""Compile every figures/fig_*.tex into a standalone vector PDF (figures/pdf/).

The same fig_*.tex sources are \\input by main.tex, so the separately submitted
figure files and the in-text figures are identical. Run from anywhere:

    python FINAL_OJCOMS/figures/make_figures.py [fig_name ...]
"""
import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                   # FINAL_OJCOMS (data/ paths are relative to it)
BUILD = os.path.join(HERE, "build")
OUTDIR = os.path.join(HERE, "pdf")
# Two-column (figure*) sources; everything else is set at the 3.5-in column width.
WIDE = {"fig_severity", "fig_detect", "fig_adapt", "fig_architecture", "fig_graphical_abstract",
        "fig_budget"}

WRAPPER = r"""\documentclass[10pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\renewcommand{\rmdefault}{ptm}
\input{figures/ieee_plot_style.tex}
\input{figures/notation.tex}
\setlength{\textwidth}{%(tw)s}\setlength{\columnwidth}{%(cw)s}
\begin{document}
\setlength{\linewidth}{%(lw)s}%% set after \begin{document}, which resets \linewidth
\input{figures/%(name)s.tex}
\end{document}
"""


def build(name):
    src_tex = open(os.path.join(HERE, f"{name}.tex"), encoding="utf-8").read()
    missing = [p for p in re.findall(r"data/[\w.-]+\.csv", src_tex) if not os.path.exists(os.path.join(PKG, p))]
    if missing:
        print(f"skip   {name} (data not yet available: {', '.join(sorted(set(missing)))})")
        return True
    wide = name in WIDE
    tex = WRAPPER % {"tw": "7.16in", "cw": "3.5in", "lw": "7.16in" if wide else "3.5in", "name": name}
    os.makedirs(BUILD, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)
    src = os.path.join(BUILD, f"{name}_standalone.tex")
    with open(src, "w", encoding="utf-8") as f:
        f.write(tex)
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={BUILD}", src]
    r = subprocess.run(cmd, cwd=PKG, capture_output=True, text=True, errors="replace")
    pdf = os.path.join(BUILD, f"{name}_standalone.pdf")
    if r.returncode != 0 or not os.path.exists(pdf):
        log = r.stdout[-3000:]
        print(f"FAILED {name}\n{log}")
        return False
    shutil.copy(pdf, os.path.join(OUTDIR, f"{name}.pdf"))
    print(f"ok     {name} -> figures/pdf/{name}.pdf")
    return True


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(os.path.splitext(os.path.basename(p))[0]
                                   for p in glob.glob(os.path.join(HERE, "fig_*.tex")))
    ok = all([build(n) for n in names])
    sys.exit(0 if ok else 1)
