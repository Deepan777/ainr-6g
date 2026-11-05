"""Generate the macros for the independent revalidation added in the audit cycle.

  FINAL_OJCOMS/tables/reval_macros.tex

Sources (raw, never edited by hand):
  results/ojcoms/highprec.csv     independent high-precision re-evaluation (exp_highprec.py)
  results/ojcoms/traj_extra.csv   D_tr validation and Doppler along the trajectories (exp_traj_extra.py)
  results/ojcoms/traj_doppler.csv Doppler dose-response along the trajectories (exp_traj_doppler.py)

Every checkpoint measured in highprec.csv was *chosen* on the discovery sweep and *measured* on
streams disjoint from it, so the factors below are unbiased for the chosen checkpoints.
"""
import csv
import math
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "FINAL_OJCOMS")
OJ = os.path.join(ROOT, "results", "ojcoms")
N_SHIFT_BLOCKS = 1920
M = {}


def num(x, d=2):
    s = f"{x:.{d}f}"
    return "\\ensuremath{" + s + "}" if s.startswith("-") else s


def pfmt(p):
    return "\\ensuremath{<0.001}" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ high-precision frontier
rows = list(csv.DictReader(open(os.path.join(OJ, "highprec.csv"))))
by_seed = {}
for r in rows:
    by_seed.setdefault(int(r["seed"]), []).append(r)

kap = {"S2": [], "DS1000": []}
bound = {"S2": 0, "DS1000": 0}
deltas, over_cap = [], 0
for s, rs in sorted(by_seed.items()):
    rs = sorted(rs, key=lambda r: int(r["step"]))
    best = min(float(r["snr_at_0p1"]) for r in rs)
    ref = ([r for r in rs if "t_val" in r["roles"]] or [r for r in rs if "t_final" in r["roles"]])[0]
    star = [r for r in rs if "t_star" in r["roles"]][0]
    d = float(star["snr_at_0p1"]) - best
    deltas.append(d)
    over_cap += d > 0.30
    for c in kap:
        m, mf = float(star[f"bler_{c}"]), float(ref[f"bler_{c}"])
        if m > 0:
            kap[c].append(mf / m)
        else:                                   # no block error: 95% one-sided bound, 3/n
            kap[c].append(mf / (3.0 / N_SHIFT_BLOCKS))
            bound[c] += 1

M["hpSeeds"] = str(len(by_seed))
M["hpBlocksMatched"] = "1024"
M["hpBlocksShift"] = "1920"
for c, tag in (("S2", "Stwo"), ("DS1000", "DS")):
    k = np.array(kap[c])
    M[f"hpKappa{tag}Min"] = num(k.min(), 1)
    M[f"hpKappa{tag}Max"] = num(k.max(), 0)
    M[f"hpKappa{tag}Median"] = num(np.median(k), 1)
    M[f"hpKappa{tag}Bounds"] = str(bound[c])
M["hpDeltaMin"] = num(min(deltas), 2)
M["hpDeltaMax"] = num(max(deltas), 2)
M["hpDeltaMedian"] = num(float(np.median(deltas)), 2)
M["hpOverCap"] = str(over_cap)
M["hpWithinCap"] = str(len(deltas) - over_cap)

# Robustness cost relative to the matched-condition optimum t_match, on the same streams.
km = []
for s_, rs in sorted(by_seed.items()):
    ref = [r for r in rs if "t_match" in r["roles"]][0]
    star = [r for r in rs if "t_star" in r["roles"]][0]
    m, mf = float(star["bler_S2"]), float(ref["bler_S2"])
    km.append(mf / m if m > 0 else mf / (3.0 / N_SHIFT_BLOCKS))
km = np.array(km)
M["hpKappaMatchStwoMin"] = num(km.min(), 1)
M["hpKappaMatchStwoMax"] = num(km.max(), 0)
M["hpKappaMatchStwoMedian"] = num(np.median(km), 1)

# Rules R1/R2 re-measured on the independent streams. The pre-registered test applies the rules to
# the held-out trajectories only (analyze_budget.HELD_OUT); the count within the cap is reported
# over all trajectories so that the dev seeds are not silently dropped.
HELD_OUT = [5, 6, 7, 8]
CAP_DB = 0.30
for role, tag in (("t_R1", "Rone"), ("t_R2", "Rtwo")):
    ds_ho, n_in, n_all = [], 0, 0
    for s, rs in sorted(by_seed.items()):
        sel = [r for r in rs if role in r["roles"]]
        if not sel:
            continue
        best = min(float(r["snr_at_0p1"]) for r in rs)
        d = float(sel[0]["snr_at_0p1"]) - best
        n_all += 1
        n_in += d <= CAP_DB + 1e-9
        if s in HELD_OUT:
            ds_ho.append(d)
    M[f"hp{tag}DeltaMin"] = num(min(ds_ho), 2)
    M[f"hp{tag}DeltaMax"] = num(max(ds_ho), 2)
    M[f"hp{tag}WithinCap"] = str(n_in)
    M[f"hp{tag}NAll"] = str(n_all)
M["hpHeldOut"] = str(len([s for s in by_seed if s in HELD_OUT]))

# ------------------------------------------------------------------ dense DiscNRX control
# Full trajectories of the discriminative baseline (same schedule and snapshot grid as the AINR
# runs), completed after the audit cycle. Source: results/ojcoms/dense_eval.csv, receiver=discnrx.
DD = {}
for r in csv.DictReader(open(os.path.join(OJ, "dense_eval.csv"))):
    if r["receiver"] != "discnrx":
        continue
    DD.setdefault(int(r["seed"]), []).append(
        (int(r["step"]), float(r["bler_S2"]), float(r["snr_at_0p1"])))
if DD:
    rho_s2, rho_g = [], []
    at = {5000: [], 50000: []}
    for s_, v in sorted(DD.items()):
        v.sort()
        step = np.array([x[0] for x in v], float)
        s2 = np.array([x[1] for x in v])
        gm = np.array([x[2] for x in v])
        m = step >= 5000
        rho_s2.append(stats.spearmanr(step[m], s2[m]).statistic)
        rho_g.append(stats.spearmanr(step[m], gm[m]).statistic)
        for k in at:
            sel = step == k
            if sel.any():
                at[k].append(float(s2[sel][0]))
    M["ddnSeeds"] = str(len(DD))
    M["ddnSnaps"] = str(len(next(iter(DD.values()))))
    M["ddnRhoStwoMin"] = num(min(rho_s2), 2)
    M["ddnRhoStwoMax"] = num(max(rho_s2), 2)
    M["ddnRhoSnrMin"] = num(min(rho_g), 2)
    M["ddnRhoSnrMax"] = num(max(rho_g), 2)
    M["ddnStwoEarly"] = num(float(np.mean(at[5000])), 4)
    M["ddnStwoLate"] = num(float(np.mean(at[50000])), 3)

# ------------------------------------------------------------------ D_tr validation metric
V = {}
for r in csv.DictReader(open(os.path.join(OJ, "traj_extra.csv"))):
    V.setdefault(int(r["seed"]), []).append(
        (int(r["step"]), float(r["val_bce"]), float(r["val_bler"]),
         float(r["bler_V250"]), float(r["bler_V400"])))

rho, frac, fracb, tailgap, tailsd, early_drop = [], [], [], [], [], []
for s, v in sorted(V.items()):
    v.sort()
    step = np.array([x[0] for x in v], float)
    bce = np.array([x[1] for x in v])
    bler = np.array([x[2] for x in v])
    rho.append(stats.spearmanr(step, bce).statistic)
    T = step.max()
    frac.append(step[int(np.argmin(bce))] / T)
    fracb.append(step[int(np.argmin(bler))] / T)
    tailgap.append(bce[-1] - bce.min())
    tailsd.append(np.diff(bce[step >= 20000], 2).std())
    early_drop.append(bce[0] - bce.min())

M["vlSeeds"] = str(len(V))
M["vlSlots"] = "1280"
M["vlRhoMin"] = num(min(rho), 3)
M["vlRhoMax"] = num(max(rho), 3)
M["vlFracMin"] = num(100 * min(frac), 0)
M["vlFracBlerMin"] = num(100 * min(fracb), 0)
M["vlTailGapMax"] = num(1e3 * max(tailgap), 1)
M["vlTailSdMin"] = num(1e3 * min(tailsd), 1)
M["vlTailSdMax"] = num(1e3 * max(tailsd), 1)
M["vlEarlyDropMin"] = num(1e3 * min(early_drop), 0)
M["vlEarlyDropMax"] = num(1e3 * max(early_drop), 0)

# ------------------------------------------------------------------ Doppler along the trajectory
dop = {"V250": [], "V400": []}
for s, v in sorted(V.items()):
    v.sort()
    step = np.array([x[0] for x in v], float)
    for j, c in ((3, "V250"), (4, "V400")):
        y = np.array([x[j] for x in v])
        dop[c].append((stats.spearmanr(step, y).statistic,
                       y[step <= 5000].mean(), y[step >= 35000].mean()))
for c, tag in (("V250", "Vtwofifty"), ("V400", "Vfourhundred")):
    r = np.array([x[0] for x in dop[c]])
    e = np.array([x[1] for x in dop[c]])
    l = np.array([x[2] for x in dop[c]])
    M[f"dp{tag}RhoMin"] = num(r.min(), 2)
    M[f"dp{tag}RhoMax"] = num(r.max(), 2)
    M[f"dp{tag}NPos"] = str(int((r > 0).sum()))
    M[f"dp{tag}Early"] = num(e.mean(), 3)
    M[f"dp{tag}Late"] = num(l.mean(), 3)
    M[f"dp{tag}P"] = pfmt(float(stats.ttest_rel(l, e).pvalue))
M["dpSeeds"] = str(len(V))
M["dpBlocks"] = "480"

# ------------------------------------------------------------------ Doppler dose-response
dd = os.path.join(OJ, "traj_doppler.csv")
if os.path.exists(dd):
    D = {}
    for r in csv.DictReader(open(dd)):
        D.setdefault(int(r["seed"]), []).append(
            (int(r["step"]), float(r["bler_V30"]), float(r["bler_V90"]), float(r["bler_V150"])))
    if len(D) == 7 and all(len(v) >= 9 for v in D.values()):
        for j, c, tag in ((1, "V30", "Vthirty"), (2, "V90", "Vninety"), (3, "V150", "Vonefifty")):
            rs, es, ls = [], [], []
            for s, v in sorted(D.items()):
                v.sort()
                step = np.array([x[0] for x in v], float)
                y = np.array([x[j] for x in v])
                # a condition with no block error at any checkpoint has no rank to correlate
                rs.append(0.0 if y.std() == 0 else stats.spearmanr(step, y).statistic)
                es.append(y[step <= 5000].mean())
                ls.append(y[step >= 20000].mean())
            rs, es, ls = np.array(rs), np.array(es), np.array(ls)
            flat = (es.std() == 0 and ls.std() == 0 and es.mean() == ls.mean())
            M[f"dd{tag}RhoMin"] = num(rs.min(), 2)
            M[f"dd{tag}RhoMax"] = num(rs.max(), 2)
            M[f"dd{tag}NPos"] = str(int((rs > 0).sum()))
            M[f"dd{tag}NNeg"] = str(int((rs < 0).sum()))
            M[f"dd{tag}Early"] = num(es.mean(), 3)
            M[f"dd{tag}Late"] = num(ls.mean(), 3)
            M[f"dd{tag}P"] = "--" if flat else pfmt(float(stats.ttest_rel(ls, es).pvalue))
        M["ddSeeds"] = str(len(D))
        M["ddBlocks"] = "960"

# ------------------------------------------------------------------ genie attribution
ap = os.path.join(OJ, "attribution.csv")
if os.path.exists(ap):
    A = {}
    for r in csv.DictReader(open(ap)):
        A.setdefault(r["cond"], []).append(
            {k: float(r[k]) for k in ("gg", "gq", "qg", "qq")})
    for c, tag in (("S1", "Sone"), ("DS1000", "DS"), ("S2", "Stwo"), ("V250", "Vtwofifty")):
        if c not in A:
            continue
        gg = np.array([x["gg"] for x in A[c]])
        gq = np.array([x["gq"] for x in A[c]])
        qg = np.array([x["qg"] for x in A[c]])
        qq = np.array([x["qq"] for x in A[c]])
        M[f"at{tag}Genie"] = num(gg.mean(), 1)
        M[f"at{tag}Chan"] = num((gq - gg).mean(), 1)
        M[f"at{tag}Bits"] = num((qg - gg).mean(), 1)
        M[f"at{tag}Inter"] = num((qq - gq - qg + gg).mean(), 1)
        M[f"at{tag}Full"] = num(qq.mean(), 1)
    M["atSeeds"] = str(len(A[list(A)[0]]))

# ------------------------------------------------------------------ figure data: shift families
# Mean over trajectories of the shifted-channel BLER against the training iteration, for the
# frequency-domain shifts (5 dB, from the discovery sweep) and the Doppler shifts (10 dB).
DATA = os.path.join(PKG, "data")


def _write(name, header, rows):
    with open(os.path.join(DATA, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote data/" + name)


de = {}
for r in csv.DictReader(open(os.path.join(OJ, "dense_eval.csv"))):
    if r["receiver"] != "ainr":
        continue
    de.setdefault(int(r["step"]), []).append((float(r["bler_S2"]), float(r["bler_DS1000"])))
FLOOR = 1e-4
_write("fig_family_freq.csv", ["step", "S2", "DS"],
       [[k, max(FLOOR, float(np.mean([x[0] for x in v]))),
         max(FLOOR, float(np.mean([x[1] for x in v])))] for k, v in sorted(de.items())])

dv = {}
for r in csv.DictReader(open(os.path.join(OJ, "traj_extra.csv"))):
    dv.setdefault(int(r["step"]), []).append((float(r["bler_V250"]), float(r["bler_V400"])))
_write("fig_family_dop.csv", ["step", "V250", "V400"],
       [[k, float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))]
        for k, v in sorted(dv.items())])

if os.path.exists(os.path.join(OJ, "traj_doppler.csv")):
    d1 = {}
    for r in csv.DictReader(open(os.path.join(OJ, "traj_doppler.csv"))):
        d1.setdefault(int(r["step"]), []).append(float(r["bler_V150"]))
    _write("fig_family_v150.csv", ["step", "V150"],
           [[k, max(FLOOR, float(np.mean(v)))] for k, v in sorted(d1.items())])

# ------------------------------------------------------------------ figure data: DiscNRX trajectories
if DD:
    rows_ = []
    steps_ = sorted({x[0] for v in DD.values() for x in v})
    S2m, GAM = {}, {}
    for r in csv.DictReader(open(os.path.join(OJ, "dense_eval.csv"))):
        if r["receiver"] != "discnrx":
            continue
        S2m.setdefault(int(r["step"]), []).append(float(r["bler_S2"]))
        GAM.setdefault(int(r["step"]), []).append(float(r["snr_at_0p1"]))
    for k in steps_:
        rows_.append([k, max(1e-4, float(np.mean(S2m[k]))), float(np.mean(GAM[k]))])
    _write("fig_disc_traj.csv", ["step", "S2", "gamma"], rows_)

out = os.path.join(PKG, "tables", "reval_macros.tex")
with open(out, "w", encoding="utf-8") as f:
    f.write("% Generated by make_revalidation_macros.py -- do not edit by hand.\n")
    for k in sorted(M):
        f.write("\\newcommand{\\" + k + "}{" + M[k] + "}\n")
print(f"wrote {out}: {len(M)} macros")
for k in sorted(M):
    print(f"  {k:24s} {M[k]}")
