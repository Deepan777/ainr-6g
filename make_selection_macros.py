"""Macros for the two constructive selection results added on 2026-09-21.

  FINAL_OJCOMS/tables/selection_macros.tex
  FINAL_OJCOMS/data/fig_soup.csv          frontier points of the averaging schemes

Sources (raw, never edited by hand):
  results/ojcoms/soup.csv        checkpoint averaging (exp_soup.py)
  results/ojcoms/sourceonly.csv  source-only scores (exp_sourceonly.py)
  results/ojcoms/dense_eval.csv  discovery sweep, for the score selections
  results/ojcoms/highprec.csv    independent streams, for the selected checkpoints
  results/ojcoms/traj_extra.csv  validation optimum per trajectory

Protocols fixed before the runs: audits/ANALYSIS_PROTOCOL_SOUP.md and
audits/ANALYSIS_PROTOCOL_SOURCEONLY.md.
"""
import csv
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "FINAL_OJCOMS")
OJ = os.path.join(ROOT, "results", "ojcoms")
CAP = 0.30
DEV_SEEDS, HELD_OUT = [2, 3, 4], [5, 6, 7, 8]
M = {}


def num(x, d=2):
    s = f"{x:.{d}f}"
    return "\\ensuremath{" + s + "}" if s.startswith("-") else s


# ------------------------------------------------------------------ checkpoint averaging
rows = list(csv.DictReader(open(os.path.join(OJ, "soup.csv"))))
by = {}
for r in rows:
    by.setdefault(int(r["seed"]), {})[r["model"]] = r

stats_by_scheme = {}
for scheme in ("ALL", "EARLY", "MID", "TAIL10", "TAIL20", "TAIL35", "t_star"):
    ds, ks, succ = [], [], 0
    for s, m in sorted(by.items()):
        if scheme not in m or "t_val" not in m:
            continue
        best = min(float(v["snr_at_0p1"]) for v in m.values())
        d = float(m[scheme]["snr_at_0p1"]) - best
        p, pv = float(m[scheme]["bler_S2"]), float(m["t_val"]["bler_S2"])
        k = pv / p if p > 0 else pv / (3.0 / 960)
        ds.append(d)
        ks.append(k)
        succ += (d <= CAP + 1e-9) and (p <= pv / 10)
    stats_by_scheme[scheme] = (np.array(ds), np.array(ks), succ)

for scheme, tag in (("MID", "Mid"), ("EARLY", "Early"), ("ALL", "All"), ("TAIL35", "Tail"),
                    ("TAIL10", "Tailten"), ("TAIL20", "Tailtwenty")):
    ds, ks, succ = stats_by_scheme[scheme]
    M[f"sp{tag}DeltaMed"] = num(float(np.median(ds)), 2)
    M[f"sp{tag}DeltaMin"] = num(float(ds.min()), 2)
    M[f"sp{tag}DeltaMax"] = num(float(ds.max()), 2)
    M[f"sp{tag}KappaMed"] = num(float(np.median(ks)), 1)
    M[f"sp{tag}KappaMin"] = num(float(ks.min()), 1)
    M[f"sp{tag}KappaMax"] = num(float(ks.max()), 0)
    M[f"sp{tag}Succ"] = str(int(succ))
M["spSeeds"] = str(len(by))
M["spSchemes"] = "6"
M["spEndpoint"] = "4"
M["spBlocksMatched"] = "512"
M["spBlocksShift"] = "960"
M["spMidLo"] = "$1.0\\times10^{4}$"
M["spMidHi"] = "$2.5\\times10^{4}$"
ds, ks, _ = stats_by_scheme["t_star"]
M["spStarDeltaMed"] = num(float(np.median(ds)), 2)
M["spStarKappaMed"] = num(float(np.median(ks)), 1)

with open(os.path.join(PKG, "data", "fig_soup.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scheme", "delta", "S2"])
    for scheme in ("ALL", "EARLY", "MID", "TAIL10", "TAIL20", "TAIL35", "t_star", "t_val"):
        for s, m in sorted(by.items()):
            if scheme not in m:
                continue
            best = min(float(v["snr_at_0p1"]) for v in m.values())
            w.writerow([scheme, round(float(m[scheme]["snr_at_0p1"]) - best, 4),
                        max(1e-4, float(m[scheme]["bler_S2"]))])
print("wrote data/fig_soup.csv")

# ------------------------------------------------------------------ source-only scores
SC = {}
for r in csv.DictReader(open(os.path.join(OJ, "sourceonly.csv"))):
    SC.setdefault(int(r["seed"]), []).append(
        {k: float(r[k]) for k in ("step", "effrank", "logdet", "featnorm", "sharpness")})
DE = {}
for r in csv.DictReader(open(os.path.join(OJ, "dense_eval.csv"))):
    if r["receiver"] == "ainr":
        DE.setdefault(int(r["seed"]), {})[int(float(r["step"]))] = (
            float(r["snr_at_0p1"]), float(r["bler_S2"]))
VAL = {}
tv = {}
for r in csv.DictReader(open(os.path.join(OJ, "traj_extra.csv"))):
    tv.setdefault(int(r["seed"]), []).append((float(r["val_bce"]), int(r["step"])))
for s, v in tv.items():
    VAL[s] = min(v)[1]
STAR = {}
for r in csv.DictReader(open(os.path.join(OJ, "highprec.csv"))):
    if "t_star" in r["roles"]:
        STAR[int(r["seed"])] = int(r["step"])

SCORES = ["effrank", "logdet", "featnorm", "sharpness"]
dev_rho = {}
for name in SCORES:
    rs = []
    for s in DEV_SEEDS:
        x = [c[name] for c in SC[s]]
        y = [DE[s][int(c["step"])][1] for c in SC[s]]
        rs.append(stats.spearmanr(x, y).statistic)
    dev_rho[name] = np.array(rs)

succ, same = {}, {}
for name in SCORES:
    direction = np.sign(dev_rho[name].mean())        # +1 -> pick the minimum inside the cap
    n_ok, n_same = 0, 0
    for s in sorted(SC):
        best = min(DE[s][int(c["step"])][0] for c in SC[s])
        feas = [c for c in SC[s] if DE[s][int(c["step"])][0] - best <= CAP + 1e-9]
        pick = min(feas, key=lambda c: c[name]) if direction > 0 else max(feas, key=lambda c: c[name])
        step = int(pick["step"])
        if s in HELD_OUT:
            n_ok += DE[s][step][1] <= DE[s][VAL[s]][1] / 10
        n_same += step == STAR.get(s, -1)
    succ[name], same[name] = n_ok, n_same

M["soScores"] = str(len(SCORES))
M["soHeldOut"] = str(len(HELD_OUT))
M["soSeeds"] = str(len(SC))
M["soCap"] = num(CAP, 2)
M["soSlots"] = "256"
M["soFeatRhoMin"] = num(float(dev_rho["featnorm"].min()), 2)
M["soFeatRhoMax"] = num(float(dev_rho["featnorm"].max()), 2)
M["soFeatSucc"] = str(succ["featnorm"])
M["soFeatSame"] = str(same["featnorm"])
M["soLogdetSucc"] = str(succ["logdet"])
M["soEffrankSucc"] = str(succ["effrank"])
M["soSharpSucc"] = str(succ["sharpness"])
M["soFeatLogdetRho"] = num(float(np.mean([
    stats.spearmanr([c["featnorm"] for c in SC[s]], [c["logdet"] for c in SC[s]]).statistic
    for s in SC])), 2)

out = os.path.join(PKG, "tables", "selection_macros.tex")
with open(out, "w", encoding="utf-8") as f:
    f.write("% Generated by make_selection_macros.py -- do not edit by hand.\n")
    for k in sorted(M):
        f.write("\\newcommand{\\" + k + "}{" + M[k] + "}\n")
print(f"wrote {out}: {len(M)} macros")
for k in sorted(M):
    print(f"  {k:20s} {M[k]}")

# ------------------------------------------------------------------ supplementary tables
# Added 2026-11-06. These reuse the quantities computed above, so the tables and the macros
# cannot drift apart: kappa is printed with the same digits the macros use (median and minimum
# to one decimal, maximum to none). Outputs are mirrored into every package directory present.
# The window boundaries are the experiment's own definitions: read them out of exp_soup.py
# rather than retyping them, so the published table cannot drift from what was run.
def _schemes_from_exp_soup():
    import ast
    src = open(os.path.join(ROOT, "exp_soup.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "SCHEMES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("SCHEMES not found in exp_soup.py")


_SCH = _schemes_from_exp_soup()
ALL_SCHEMES = list(_SCH)
WINDOW = {k: f"{lo // 1000}--{hi // 1000}" for k, (lo, hi) in _SCH.items()}
FLOOR = 3.0 / 960.0


def per_seed(scheme):
    """(seed, Delta, kappa, censored) for one scheme, in seed order."""
    out = []
    for s, m in sorted(by.items()):
        if scheme not in m or "t_val" not in m:
            continue
        best = min(float(v["snr_at_0p1"]) for v in m.values())
        d = float(m[scheme]["snr_at_0p1"]) - best
        p, pv = float(m[scheme]["bler_S2"]), float(m["t_val"]["bler_S2"])
        out.append((s, d, (pv / p if p > 0 else pv / FLOOR), p == 0))
    return out


def mirror(relpath, text):
    for pkg in ("FINAL_OJCOMS", "FINAL_ARRAY"):
        base = os.path.join(ROOT, pkg)
        if not os.path.isdir(base):
            continue
        dst = os.path.join(base, relpath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        print("wrote", os.path.join(pkg, relpath))


n_cens = sum(1 for sc in ALL_SCHEMES for r in per_seed(sc) if r[3])

# --- Table: every averaging window, including the two the main text does not report -------
L = [r"% Generated by make_selection_macros.py -- do not edit by hand.",
     r"\begin{table}[H]\centering",
     r"\caption{Checkpoint averaging: all \spSchemes{} pre-registered windows over \spSeeds{} "
     r"trajectories. $n$ is the number of checkpoints averaged, $\Delta$ the matched cost in "
     r"decibels, and $\kappa$ the ratio of $P_{\mathrm{S2}}$ at the validation-selected "
     r"checkpoint to that of the averaged model, so $\kappa>1$ is an improvement. The last "
     r"column counts trajectories meeting the pre-registered endpoint, $\Delta\le\soCap{}$~dB "
     r"together with a tenfold reduction in $P_{\mathrm{S2}}$. One averaged model (the early "
     r"window of the first trajectory) produced no block error; its $\kappa$ is computed with a "
     r"censoring floor of three errors in \spBlocksShift{} blocks and is therefore a lower "
     r"bound. The oracle checkpoint $t^{\star}_{\epsilon}$ and the validation optimum "
     r"$t_{\mathrm{val}}$ of the same trajectories are listed for "
     r"reference.}\label{tab:s-soup}",
     r"\begin{tabular}{llrrrrrrrc}",
     r"\toprule",
     r"& Window & & \multicolumn{3}{c}{$\Delta$ (dB)} & \multicolumn{3}{c}{$\kappa$} & \\",
     r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
     r"Scheme & ($\times10^{3}$) & $n$ & med. & min & max & med. & min & max & Succ.\\",
     r"\midrule"]


def scheme_row(label, scheme, window, show_succ=True):
    rowsps = per_seed(scheme)
    if not rowsps:
        return None
    ds = np.array([r[1] for r in rowsps])
    ks = np.array([r[2] for r in rowsps])
    n_avg = int(by[sorted(by)[0]][scheme]["n_avg"])
    succ_n = sum((r[1] <= CAP + 1e-9) and (r[2] >= 10.0) for r in rowsps)
    tail = f"{succ_n}/{len(rowsps)}" if show_succ else "--"
    return (f"{label} & {window} & {n_avg} & "
            f"{np.median(ds):.2f} & {ds.min():.2f} & {ds.max():.2f} & "
            f"{np.median(ks):.1f} & {ks.min():.1f} & "
            # the macros print maxima with no decimals; every maximum the main text quotes is
            # >= 10, so one decimal below 10 adds precision without contradicting the text
            + (f"{ks.max():.0f}" if ks.max() >= 10 else f"{ks.max():.1f}")
            + f" & {tail}" + r"\\")


for scheme in ALL_SCHEMES:
    L.append(scheme_row(scheme.replace("TAIL", "TAIL-"), scheme, WINDOW[scheme]))
L.append(r"\midrule")
for scheme, label in (("t_star", r"$t^{\star}_{\epsilon}$"), ("t_val", r"$t_{\mathrm{val}}$")):
    row = scheme_row(label, scheme, "--", show_succ=False)
    if row:
        L.append(row)
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
mirror(os.path.join("tables", "tab_supp_soup.tex"), "\n".join(L))

# --- Table: the four source-only scores ---------------------------------------------------
NICE = {"effrank": "Effective rank", "logdet": "Log-determinant of feature covariance",
        "featnorm": "Mean feature norm", "sharpness": "Matched-loss sharpness"}
L = [r"% Generated by make_selection_macros.py -- do not edit by hand.",
     r"\begin{table}[H]\centering",
     r"\caption{Source-only checkpoint scores, each computed from \soSlots{} matched slots per "
     r"checkpoint. The rule takes the extreme-scoring checkpoint among those within "
     r"$\soCap{}$~dB of the best matched accuracy of the trajectory; which extreme is fixed by "
     r"the sign of the score's rank correlation with $P_{\mathrm{S2}}$ on the development "
     r"trajectories (seeds 2--4) and is then applied unchanged to the held-out ones. Successes "
     r"are held-out trajectories meeting the pre-registered tenfold endpoint. The last column "
     r"counts trajectories, of all \soSeeds{}, in which the rule selects exactly the oracle "
     r"checkpoint. No shifted data enter any score.}\label{tab:s-sourceonly}",
     r"\begin{tabular}{lcrrcc}",
     r"\toprule",
     r"Score & Rule & \multicolumn{2}{c}{Dev.\ $\rho$ with $P_{\mathrm{S2}}$} & Succ. & Oracle\\",
     r"\cmidrule(lr){3-4}",
     r"& & min & max & (/\soHeldOut{}) & (/\soSeeds{})\\",
     r"\midrule"]
for name in SCORES:
    rho = dev_rho[name]
    rule = "min" if np.sign(rho.mean()) > 0 else "max"
    L.append(f"{NICE[name]} & {rule} & {rho.min():.2f} & {rho.max():.2f} & "
             f"{succ[name]} & {same[name]}" + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
mirror(os.path.join("tables", "tab_supp_sourceonly.tex"), "\n".join(L))

# --- Table: per-trajectory detail for the two constructions -------------------------------
direction_fn = np.sign(dev_rho["featnorm"].mean())
mid = {s: (d, k) for s, d, k, _ in per_seed("MID")}
L = [r"% Generated by make_selection_macros.py -- do not edit by hand.",
     r"\begin{table}[H]\centering",
     r"\caption{Per-trajectory selections. $t_{\mathrm{val}}$ is the validation optimum over "
     r"the training distribution, $t^{\star}_{\epsilon}$ the oracle of the main text on the "
     r"discovery sweep, and $t_{\mathrm{norm}}$ the checkpoint chosen by the mean feature norm "
     r"from matched data alone; the latter two coincide in every trajectory. The last two "
     r"columns give the matched cost and the robustness ratio of the mid-window average of the "
     r"same trajectory. Iteration counts are rounded to the nearest "
     r"thousand.}\label{tab:s-selseed}",
     r"\begin{tabular}{crrrcrr}",
     r"\toprule",
     r"& \multicolumn{3}{c}{Checkpoint ($\times10^{3}$ iterations)} & & "
     r"\multicolumn{2}{c}{MID average}\\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
     r"Seed & $t_{\mathrm{val}}$ & $t^{\star}_{\epsilon}$ & $t_{\mathrm{norm}}$ & "
     r"$t_{\mathrm{norm}}=t^{\star}_{\epsilon}$ & $\Delta$ (dB) & $\kappa$\\",
     r"\midrule"]
for s in sorted(SC):
    best = min(DE[s][int(c["step"])][0] for c in SC[s])
    feas = [c for c in SC[s] if DE[s][int(c["step"])][0] - best <= CAP + 1e-9]
    pick = (min(feas, key=lambda c: c["featnorm"]) if direction_fn > 0
            else max(feas, key=lambda c: c["featnorm"]))
    step = int(pick["step"])
    star = STAR.get(s)
    agree = r"\checkmark" if star is not None and step == star else "--"
    d, k = mid.get(s, (float("nan"), float("nan")))
    L.append(f"{s} & {VAL[s] / 1000:.0f} & "
             + (f"{star / 1000:.0f}" if star is not None else "--")
             + f" & {step / 1000:.0f} & {agree} & {d:.2f} & {k:.1f}" + r"\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
mirror(os.path.join("tables", "tab_supp_selseed.tex"), "\n".join(L))

# --- One column pair per scheme, so the figure needs no row filtering ----------------------
cols, data = [], {}
for scheme in ALL_SCHEMES + ["t_star", "t_val"]:
    rowsps = per_seed(scheme)
    if not rowsps:
        continue
    cols.append(scheme)
    data[scheme] = [(r[1], max(1e-4, float(by[r[0]][scheme]["bler_S2"]))) for r in rowsps]
lines = [",".join(f"d_{c},p_{c}" for c in cols)]
for i in range(max(len(data[c]) for c in cols)):
    cells = []
    for c in cols:
        cells += ([f"{data[c][i][0]:.4f}", f"{data[c][i][1]:.6f}"]
                  if i < len(data[c]) else ["nan", "nan"])
    lines.append(",".join(cells))
mirror(os.path.join("data", "fig_soup_wide.csv"), "\n".join(lines) + "\n")
print(f"censored entries among the averaging schemes: {n_cens}")
