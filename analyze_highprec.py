"""Analysis of the independent high-precision re-evaluation (results/ojcoms/highprec.csv).

Every checkpoint in that file was *chosen* on the discovery sweep and *measured* on fresh streams,
so the numbers below are unbiased estimates of the chosen checkpoints' performance.

Reported per trajectory and pooled:
  gamma_0.1(t)   matched SNR at BLER 0.1, with a block bootstrap CI
  Delta(t)       matched cost relative to the best matched checkpoint among the candidates
  kappa(t)       shifted-BLER ratio of the convergence-selected checkpoint to t
The kappa interval combines the two bootstrap intervals on the log scale. The two checkpoints are
evaluated on the same block stream, so their errors are positively correlated and treating them as
independent widens the interval; the interval is therefore conservative.

    python analyze_highprec.py            (writes results/ojcoms/highprec_summary.csv)
"""
import csv
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "results", "ojcoms", "highprec.csv")
OUT = os.path.join(ROOT, "results", "ojcoms", "highprec_summary.csv")
SHIFTED = ["S2", "DS1000", "V250"]
N_SHIFT_BLOCKS = 1920           # exp_highprec.py: NB_SHIFT * BATCH
Z = 1.959963985


def se_log(lo, hi):
    """SD of log(BLER) implied by a 95% bootstrap interval."""
    if not (lo > 0 and hi > 0):
        return float("nan")
    return (math.log(hi) - math.log(lo)) / (2 * Z)


rows = list(csv.DictReader(open(SRC)))
by_seed = {}
for r in rows:
    by_seed.setdefault(int(r["seed"]), []).append(r)

out = []
print(f"{'seed':>4} {'step':>6} {'roles':<26} {'gamma':>18} {'Delta':>7} "
      + " ".join(f"{c + ' BLER':>22}" for c in SHIFTED))
for s in sorted(by_seed):
    rs = sorted(by_seed[s], key=lambda r: int(r["step"]))
    gam = np.array([float(r["snr_at_0p1"]) for r in rs])
    best = np.nanmin(gam)
    # Reference = the checkpoint a genuine validation criterion on D_tr returns (t_val);
    # t_final is kept as a robustness check because the two are close.
    fin = [r for r in rs if "t_val" in r["roles"]] or [r for r in rs if "t_final" in r["roles"]]
    fin = fin[0] if fin else rs[-1]
    for r in rs:
        d = float(r["snr_at_0p1"]) - best
        cells = []
        rec = {"seed": s, "step": int(r["step"]), "roles": r["roles"],
               "gamma": r["snr_at_0p1"], "gamma_lo": r["snr_lo"], "gamma_hi": r["snr_hi"],
               "delta_db": f"{d:.4f}"}
        for c in SHIFTED:
            m, lo, hi = float(r[f"bler_{c}"]), float(r[f"lo_{c}"]), float(r[f"hi_{c}"])
            cells.append(f"{m:.4f}[{lo:.4f},{hi:.4f}]")
            rec[f"bler_{c}"], rec[f"lo_{c}"], rec[f"hi_{c}"] = m, lo, hi
            mf, lf, hf = float(fin[f"bler_{c}"]), float(fin[f"lo_{c}"]), float(fin[f"hi_{c}"])
            if m > 0 and mf > 0:
                k = mf / m
                sd = math.hypot(se_log(lo, hi), se_log(lf, hf))
                rec[f"kappa_{c}"] = f"{k:.2f}"
                rec[f"kappa_lo_{c}"] = f"{k * math.exp(-Z * sd):.2f}"
                rec[f"kappa_hi_{c}"] = f"{k * math.exp(Z * sd):.2f}"
                rec[f"kappa_bound_{c}"] = ""
            elif mf > 0 and m == 0:
                # no block error at this checkpoint: report a one-sided bound using the
                # 95% upper limit of a zero count over n blocks (Clopper-Pearson, 3/n).
                rec[f"kappa_{c}"] = f"{mf / (3.0 / N_SHIFT_BLOCKS):.2f}"
                rec[f"kappa_lo_{c}"] = rec[f"kappa_hi_{c}"] = ""
                rec[f"kappa_bound_{c}"] = "lower"
            else:
                rec[f"kappa_{c}"] = rec[f"kappa_lo_{c}"] = rec[f"kappa_hi_{c}"] = ""
                rec[f"kappa_bound_{c}"] = ""
        out.append(rec)
        print(f"{s:>4} {int(r['step']):>6} {r['roles']:<26} "
              f"{float(r['snr_at_0p1']):.3f}[{float(r['snr_lo']):.3f},{float(r['snr_hi']):.3f}] "
              f"{d:>7.3f} " + " ".join(f"{c:>22}" for c in cells))
    print()

if out:
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

# ------------------------------------------------------------------ pooled per role
print("=" * 110)
print("Pooled over trajectories, kappa = BLER(t_val) / BLER(t), evaluated on held-out streams")
roles = ["t_final", "t_match", "t_R1", "t_R2", "t_star"]
for role in roles:
    for c in ["S2", "DS1000"]:
        ks, ds, nb = [], [], 0
        for s in sorted(by_seed):
            sel = [r for r in out if r["seed"] == s and role in r["roles"]]
            if not sel or not sel[0][f"kappa_{c}"]:
                continue
            ks.append(float(sel[0][f"kappa_{c}"]))
            ds.append(float(sel[0]["delta_db"]))
            nb += 1 if sel[0].get(f"kappa_bound_{c}") == "lower" else 0
        if ks:
            ks = np.array(ks)
            tag = f" ({nb} of {len(ks)} are lower bounds: no block error)" if nb else ""
            print(f"  {role:<8} {c:<7} n={len(ks)}  kappa median {np.median(ks):7.1f}  "
                  f"range {ks.min():6.1f}-{ks.max():7.1f}  Delta median {np.median(ds):.3f} dB "
                  f"(max {np.max(ds):.3f}){tag}")
    print()

print("Doppler (V250) for the same checkpoints, for contrast:")
for role in roles:
    vals = [(r["seed"], r["bler_V250"]) for r in out if role in r["roles"]]
    if vals:
        v = np.array([x[1] for x in vals])
        print(f"  {role:<8} n={len(v)}  mean {v.mean():.3f}  range {v.min():.3f}-{v.max():.3f}")
