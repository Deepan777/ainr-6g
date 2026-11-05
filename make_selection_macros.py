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

for scheme, tag in (("MID", "Mid"), ("EARLY", "Early"), ("ALL", "All"), ("TAIL35", "Tail")):
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
