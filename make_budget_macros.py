"""Generate the numbers and figure data for the training-budget study and the two
adaptation probes quoted in the manuscript.

  FINAL_OJCOMS/tables/budget_macros.tex      every number the text quotes (LaTeX macros)
  FINAL_OJCOMS/data/fig_frontier_traj.csv    trajectories over training steps (all seeds)
  FINAL_OJCOMS/data/fig_frontier_front.csv   frontier: matched cost vs S2 BLER (all seeds)
  FINAL_OJCOMS/data/fig_frontier_sel_{fe,mbler}.csv   selections on the held-out seeds

Sources (raw, never edited by hand):
  results/ojcoms/dense_eval.csv          dense snapshots (exp_dense_eval.py)
  results/ojcoms/scramble_probe.csv      NR-style scrambling test (exp_scramble_probe.py)
  results/ojcoms/vfe_probe_S2_snr3*.csv  label-free free-energy adaptation (exp_vfe_probe.py)
The pre-registered test itself is implemented in analyze_budget.py and reused here.
"""
import csv
import os
import statistics as st

import numpy as np
from scipy import stats

from analyze_budget import load, evaluate, oracle, SIGNALS, PRIMARY, HELD_OUT, DEV_SEEDS, CAP_DB, holm

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "FINAL_OJCOMS")
M = {}
EARLY, LATE, FLOOR = 5000, 50000, 1e-4


def num(x, d=2):
    s = f"{x:.{d}f}"
    return "\\ensuremath{" + s + "}" if s.startswith("-") else s


def pfmt(p):
    return "\\ensuremath{<0.001}" if p < 0.001 else f"{p:.3f}"


def onesided(a, b):
    """Paired one-sided p-value for mean(a) < mean(b)."""
    t, p2 = stats.ttest_rel(a, b)
    return p2 / 2 if t < 0 else 1 - p2 / 2


# ------------------------------------------------------------------ dense study
R = load(os.path.join(ROOT, "results", "ojcoms", "dense_eval.csv"))
SEEDS = sorted(s for (s, rx) in R if rx == "ainr")
steps = R[(SEEDS[0], "ainr")]["step"]
if not all(R[(s, "ainr")]["step"] == steps for s in SEEDS):
    raise SystemExit("unequal snapshot grids across seeds")
if not all(s in SEEDS for s in HELD_OUT):
    raise SystemExit("a held-out seed is missing")


def at(s, t, k):
    r = R[(s, "ainr")]
    return r[k][r["step"].index(t)]


def mean_at(t, k, seeds=None):
    return float(np.nanmean([at(s, t, k) for s in (seeds or SEEDS)]))


M["dnSeeds"] = str(len(SEEDS))
M["dnHeldOut"] = str(len(HELD_OUT))
M["dnDev"] = str(len(DEV_SEEDS))
M["dnSnaps"] = str(len(steps))
M["dnCap"] = f"{CAP_DB:.2f}"
M["dnSnrEarly"] = num(mean_at(EARLY, "snr_at_0p1"))
M["dnSnrLate"] = num(mean_at(LATE, "snr_at_0p1"))
M["dnCostEarly"] = num(mean_at(EARLY, "snr_at_0p1") - mean_at(LATE, "snr_at_0p1"))
M["dnStwoEarly"] = num(mean_at(EARLY, "bler_S2"), 4)
M["dnStwoLate"] = num(mean_at(LATE, "bler_S2"), 3)
M["dnDSEarly"] = num(mean_at(EARLY, "bler_DS1000"), 4)
M["dnDSLate"] = num(mean_at(LATE, "bler_DS1000"), 3)
M["dnStwoFactor"] = f"{mean_at(LATE, 'bler_S2') / mean_at(EARLY, 'bler_S2'):.0f}"

rs2, rsnr, rpt = [], [], []
for s in SEEDS:
    r = R[(s, "ainr")]
    idx = [i for i, t in enumerate(r["step"]) if t >= EARLY]
    x = [r["step"][i] for i in idx]
    rs2.append(stats.spearmanr(x, [r["bler_S2"][i] for i in idx])[0])
    rsnr.append(stats.spearmanr(x, [r["snr_at_0p1"][i] for i in idx])[0])
    rpt.append(stats.spearmanr(x, [r["ptol"][i] for i in idx])[0])
M["dnRhoStwoMin"] = num(min(rs2))
M["dnRhoStwoMax"] = num(max(rs2))
M["dnRhoSnrStrong"] = num(min(rsnr))      # most negative
M["dnRhoSnrWeak"] = num(max(rsnr))        # least negative
M["dnPtolRhoMin"] = num(min(rpt))
M["dnPtolRhoMax"] = num(max(rpt))
M["dnPtolEarly"] = num(mean_at(EARLY, "ptol"))
M["dnPtolLate"] = num(mean_at(LATE, "ptol"))
best_steps = [R[(s, "ainr")]["step"][int(np.nanargmin(R[(s, "ainr")]["snr_at_0p1"]))] for s in SEEDS]
M["dnBestStepMin"] = f"{min(best_steps) / 1e4:.1f}"
M["dnBestStepMax"] = f"{max(best_steps) / 1e4:.1f}"
# What conventional early stopping on matched validation accuracy would select: the snapshot with
# the best matched accuracy of each run (training data are fresh in every batch, so matched accuracy
# keeps improving and this snapshot lies near the end of training).
es_s2 = [R[(s, "ainr")]["bler_S2"][int(np.nanargmin(R[(s, "ainr")]["snr_at_0p1"]))] for s in SEEDS]
M["dnEsStwoMean"] = num(np.mean(es_s2), 3)
M["dnEsStwoMin"] = num(min(es_s2), 3)
M["dnEsStwoMax"] = num(max(es_s2), 3)
# Robustness cost of convergence-based selection (descriptive, all seeds): S2 BLER of the best-matched
# checkpoint divided by the lowest S2 BLER among checkpoints within the matched-cost cap.
kappa = [R[(s, "ainr")]["bler_S2"][int(np.nanargmin(R[(s, "ainr")]["snr_at_0p1"]))] / oracle(R[(s, "ainr")])
         for s in SEEDS]
M["dnKappaMin"] = f"{min(kappa):.0f}"
M["dnKappaMax"] = f"{max(kappa):.0f}"
M["dnKappaMedian"] = f"{np.median(kappa):.0f}"
M["dnOracleAllMean"] = num(np.mean([oracle(R[(s, 'ainr')]) for s in SEEDS]), 4)

# pilot tolerance versus S2: pooled over snapshots, and within a fixed training step
xs, ys, gs = [], [], []
for s in SEEDS:
    r = R[(s, "ainr")]
    for i, t in enumerate(r["step"]):
        if t >= EARLY and np.isfinite(r["ptol"][i]):
            xs.append(r["ptol"][i]); ys.append(r["bler_S2"][i]); gs.append(t)
xs, ys, gs = np.asarray(xs, float), np.asarray(ys, float), np.asarray(gs)
M["dnPtolPooledRho"] = num(stats.spearmanr(xs, ys)[0])
rx, ry = xs.copy(), ys.copy()
for g in set(gs.tolist()):
    m = gs == g
    rx[m] -= xs[m].mean(); ry[m] -= ys[m].mean()
rho_w, p_w = stats.spearmanr(rx, ry)
M["dnPtolWithinRho"] = num(rho_w)
M["dnPtolWithinP"] = pfmt(p_w)
M["dnPtolN"] = str(len(xs))

# best snapshot within the matched-cost cap (held-out seeds)
fac = [at(s, LATE, "bler_S2") / oracle(R[(s, "ainr")]) for s in HELD_OUT]
M["dnOracleFactorMin"] = f"{min(fac):.0f}"
M["dnOracleFactorMax"] = f"{max(fac):.0f}"


def scored(sig, seeds):
    """Selections of one signal; `a` applies the pre-registered cap rule (analyze_budget.report)."""
    a, b, raw, cost, step, ok = [], [], [], [], [], []
    for s in seeds:
        sel, c, s2, within = evaluate(R[(s, "ainr")], sig)
        d = at(s, LATE, "bler_S2")
        a.append(s2 if within else d); b.append(d); raw.append(s2)
        cost.append(c); step.append(sel); ok.append(within)
    return a, b, raw, cost, step, ok


praw, labels = [], []
for sig in SIGNALS:
    a, b, *_ = scored(sig, HELD_OUT)
    praw.append(1.0 if np.allclose(a, b) else onesided(a, b))
    labels.append(sig)
M["dnPrimaryP"] = f"{praw[labels.index(PRIMARY)]:.3f}"
sec = [(l, p) for l, p in zip(labels, praw) if l != PRIMARY]
M["dnSecMinP"] = f"{min(holm([p for _, p in sec], [l for l, _ in sec]).values()):.3f}"

a, b, raw, cost, stp, ok = scored("fe", HELD_OUT)
M["dnFeWithinN"] = str(sum(ok))
M["dnFeCostMin"] = num(min(cost)); M["dnFeCostMax"] = num(max(cost)); M["dnFeCostMean"] = num(np.mean(cost))
M["dnFeStepMin"] = f"{min(stp) / 1e3:.0f}"; M["dnFeStepMax"] = f"{max(stp) / 1e3:.0f}"
M["dnFeSelMean"] = num(np.mean(raw), 4); M["dnFeDefMean"] = num(np.mean(b), 3)
M["dnFeExpP"] = pfmt(onesided(raw, b))
f = [d / max(x, 1e-12) for x, d in zip(raw, b)]
M["dnFeFactorMin"] = f"{min(f):.0f}"; M["dnFeFactorMax"] = f"{max(f):.0f}"
_, _, rawm, costm, _, okm = scored("mbler", HELD_OUT)
M["dnMbWithinN"] = str(sum(okm))
M["dnMbCostMin"] = num(min(costm)); M["dnMbCostMax"] = num(max(costm))
M["dnMbSelMean"] = num(np.mean(rawm), 4)
fm = [at(s, LATE, "bler_S2") / max(x, 1e-12) for s, x in zip(HELD_OUT, rawm)]
M["dnMbFactorMin"] = f"{min(fm):.0f}"; M["dnMbFactorMax"] = f"{max(fm):.0f}"
M["dnSelCostMin"] = num(min(cost + costm)); M["dnSelCostMax"] = num(max(cost + costm))
M["dnSelFactorMin"] = f"{min(f + fm):.0f}"; M["dnSelFactorMax"] = f"{max(f + fm):.0f}"
M["dnFeMbHeldP"] = f"{stats.ttest_rel(raw, rawm)[1]:.3f}"
_, bd, rawd, costd, _, _ = scored("fe", DEV_SEEDS)
M["dnDevSel"] = num(np.mean(rawd), 4); M["dnDevDef"] = num(np.mean(bd), 3); M["dnDevCost"] = num(np.mean(costd))
fa = [evaluate(R[(s, "ainr")], "fe")[2] for s in SEEDS]
ma = [evaluate(R[(s, "ainr")], "mbler")[2] for s in SEEDS]
M["dnFeAll"] = num(np.mean(fa), 4); M["dnMbAll"] = num(np.mean(ma), 4)
M["dnFeMbAllP"] = "1.000" if np.allclose(fa, ma) else f"{stats.ttest_rel(fa, ma)[1]:.3f}"

# ------------------------------------------------------------------ figure data
os.makedirs(os.path.join(PKG, "data"), exist_ok=True)


def band(v):
    v = np.asarray(v, float)
    m = float(np.nanmean(v))
    h = float(stats.t.ppf(0.975, len(v) - 1) * np.nanstd(v, ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h


with open(os.path.join(PKG, "data", "fig_frontier_traj.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["step", "snr", "snr_lo", "snr_hi", "S2", "S2_lo", "S2_hi",
                "DS", "DS_lo", "DS_hi", "ptol", "ptol_lo", "ptol_hi"])
    for t in steps:
        row = [t]
        for k, logy in (("snr_at_0p1", False), ("bler_S2", True), ("bler_DS1000", True), ("ptol", False)):
            m, lo, hi = band([at(s, t, k) for s in SEEDS])
            if logy:   # clip to the log axis floor and to valid probabilities
                m, lo, hi = max(m, FLOOR), max(lo, FLOOR), min(max(hi, FLOOR), 1.0)
            row += [f"{m:.6g}", f"{lo:.6g}", f"{hi:.6g}"]
        w.writerow(row)
with open(os.path.join(PKG, "data", "fig_frontier_front.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["step", "cost", "S2"])
    for t in steps:
        c = np.mean([at(s, t, "snr_at_0p1") - np.nanmin(R[(s, "ainr")]["snr_at_0p1"]) for s in SEEDS])
        w.writerow([t, f"{c:.4f}", f"{max(mean_at(t, 'bler_S2'), FLOOR):.6g}"])
# per-seed trajectories and the convergence-selected checkpoint of every run
with open(os.path.join(PKG, "data", "fig_frontier_seeds.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["step"] + [f"{k}_s{s}" for s in SEEDS for k in ("snr", "S2", "cost")])
    for t in steps:
        row = [t]
        for s in SEEDS:
            snr = at(s, t, "snr_at_0p1")
            row += [f"{snr:.5g}", f"{max(at(s, t, 'bler_S2'), FLOOR):.6g}",
                    f"{snr - np.nanmin(R[(s, 'ainr')]['snr_at_0p1']):.4f}"]
        w.writerow(row)
with open(os.path.join(PKG, "data", "fig_frontier_conv.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["seed", "step", "cost", "S2"])
    for s in SEEDS:
        r = R[(s, "ainr")]
        i = int(np.nanargmin(r["snr_at_0p1"]))
        w.writerow([s, r["step"][i], "0", f"{max(r['bler_S2'][i], FLOOR):.6g}"])
for sig in ("fe", "mbler"):
    with open(os.path.join(PKG, "data", f"fig_frontier_sel_{sig}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "step", "cost", "S2"])
        for s in HELD_OUT:
            sel, c, s2, _ = evaluate(R[(s, "ainr")], sig)
            w.writerow([s, int(sel), f"{c:.4f}", f"{max(s2, FLOOR):.6g}"])

# per-seed selections for the supplementary material (generated table, no typed numbers)
tab = os.path.join(PKG, "tables", "tab_supp_budget.tex")
with open(tab, "w", encoding="utf-8") as fh:
    fh.write("% Generated by make_budget_macros.py -- do not edit by hand.\n")
    fh.write("\\begin{table}[H]\\centering\\small\n")
    fh.write("\\caption{Stopping-rule selections per training seed (AINR). Cost: matched SNR at BLER 0.1 "
             "above the run's best; S2: BLER at 5~dB. Seeds 2--4 are development data, seeds 5--8 held out.}"
             "\\label{tab:s-budget}\n")
    fh.write("\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n")
    fh.write("& & \\multicolumn{3}{c}{Free-energy rule R1} & \\multicolumn{3}{c}{Matched-accuracy knee R2}\\\\\n")
    fh.write("\\cmidrule(lr){3-5}\\cmidrule(l){6-8}\n")
    fh.write("Seed & S2 at $5\\times10^{4}$ & Step & Cost (dB) & S2 & Step & Cost (dB) & S2\\\\\\midrule\n")
    for s in SEEDS:
        r = R[(s, "ainr")]
        fs, fc, f2, _ = evaluate(r, "fe")
        ms, mc, m2, _ = evaluate(r, "mbler")
        fh.write(f"{s}{'' if s in HELD_OUT else ' (dev.)'} & {at(s, LATE, 'bler_S2'):.3f} & {int(fs)} & "
                 f"{fc:.2f} & {f2:.3f} & {int(ms)} & {mc:.2f} & {m2:.3f}\\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ NR-style scrambling
arms = {}
for x in csv.DictReader(open(os.path.join(ROOT, "results", "ojcoms", "scramble_probe.csv"))):
    arms.setdefault(x["mode"], []).append(x)
M["scrSlots"] = str(len(arms["none"]))
for arm, key in (("none", "None"), ("fixed", "Fixed"), ("perslot", "Slot")):
    v = arms[arm]
    M[f"scrFp{key}"] = str(sum(int(float(x["false_pass"])) for x in v))
    M[f"scrOnes{key}"] = num(st.mean(float(x["ones"]) for x in v[-100:]), 2)

# ------------------------------------------------------------------ label-free free-energy adaptation
byseed = {}
for x in csv.DictReader(open(os.path.join(ROOT, "results", "ojcoms", "vfe_probe_S2_snr3.csv"))):
    byseed.setdefault((x["seed"], x["mode"]), []).append(x)
vseeds = sorted({s for s, _ in byseed})


def last(v, k):
    return st.mean(float(x[k]) for x in v[-200:])


M["vfeSeeds"] = str(len(vseeds))
M["vfeSlots"] = str(len(byseed[(vseeds[0], "vfe")]))
M["vfeBlerMin"] = num(min(last(byseed[(s, "vfe")], "err") for s in vseeds), 3)
M["vfeNoneBlerMin"] = num(min(last(byseed[(s, "none")], "err") for s in vseeds), 3)
M["vfeNonePassMax"] = num(max(st.mean(float(x["crc_pass"]) for x in byseed[(s, "none")]) for s in vseeds), 3)
M["vfePassMax"] = num(max(last(byseed[(s, "vfe")], "crc_pass") for s in vseeds), 3)
M["vfeLlrBefore"] = num(st.mean(float(byseed[(s, "vfe")][0]["abs_llr"]) for s in vseeds), 2)
M["vfeLlrAfter"] = num(st.mean(last(byseed[(s, "vfe")], "abs_llr") for s in vseeds), 2)
fe_fell_every = all(last(byseed[(s, "vfe")], "free_energy") < float(byseed[(s, "vfe")][0]["free_energy"])
                    for s in vseeds)
vfe_all_fail = all(last(byseed[(s, "vfe")], "err") >= 0.999 for s in vseeds)   # text: "in every seed"
scr_fail_every = vfe_all_fail and all(st.mean(float(x["err"]) for x in arms[a][-100:]) >= 0.999 for a in arms)
lr_fail = True
for tag in ("lr3e4", "lr1e3"):
    g = {}
    for x in csv.DictReader(open(os.path.join(ROOT, "results", "ojcoms", f"vfe_probe_S2_snr3_{tag}.csv"))):
        g.setdefault(x["seed"], []).append(x)
    lr_fail &= all(last(v, "err") >= 0.999 for v in g.values())
# the text states both facts; stop rather than print a sentence the data contradicts
if not (fe_fell_every and lr_fail and scr_fail_every):
    raise SystemExit("probe data no longer support the text (free energy fell in every seed; higher "
                     "learning rates also failed; every scrambling arm still fails) -- revise the text")

out = os.path.join(PKG, "tables", "budget_macros.tex")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("% Generated by make_budget_macros.py -- do not edit by hand.\n")
    for k in sorted(M):
        fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"{len(M)} macros -> {out}")
for k in sorted(M):
    print(f"  {k:20s} {M[k]}")
