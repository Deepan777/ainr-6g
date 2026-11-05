"""High-precision, independent re-evaluation of the checkpoints that carry the central claim.

Candidates are identified from the existing trajectory sweep (discovery data) and are then
re-evaluated here on FRESH streams whose seeds are disjoint from every earlier evaluation. The
oracle checkpoint is therefore selected on one data set and reported on another, which removes the
selection bias of reporting an oracle on the data used to choose it.

Per trajectory the candidates are
    t_val    lowest validation loss over the full training distribution (traj_extra.csv)
    t_match  lowest matched-condition SNR at BLER 0.1 (discovery)
    t_star   lowest S2 BLER among checkpoints with Delta <= eps (discovery oracle)
    t_R1     free-energy rule, t_R2 matched-accuracy rule
    t_final  last checkpoint
and, for the frontier, the checkpoints adjacent to t_star.

Measured: matched BLER on a dense SNR grid (1024 blocks per point) from which gamma_0.1 is
interpolated, and BLER on S2, 1000-ns delay spread (5 dB) and 250 km/h (10 dB), 1920 blocks each.
Uncertainties are nonparametric bootstraps over blocks.

Output: results/ojcoms/highprec.csv (resumable, one row per checkpoint).
"""
import csv
import os

import numpy as np
import torch

from exp_common import ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel, decode
from analyze_budget import load as load_disc, evaluate, oracle, CAP_DB
from src.ainr import AINR

SEEDS = [2, 3, 4, 5, 6, 7, 8]
GRID = [round(0.4 + 0.1 * i, 2) for i in range(21)]        # 0.4 .. 2.4 dB
NB_MATCHED, NB_SHIFT, BATCH = 32, 60, 32
SHIFTED = [("S2", 5.0), ("DS1000", 5.0), ("V250", 10.0)]
# Fixed, process-independent stream offsets (str hashing is randomized per process).
OFFSET = {"S2": 96001, "DS1000": 96002, "V250": 96003}
BOOT = 400
OUT = os.path.join(ROOT, "results", "ojcoms", "highprec.csv")
FIELDS = (["seed", "step", "roles", "snr_at_0p1", "snr_lo", "snr_hi"]
          + [f"{k}{c}" for c, _ in SHIFTED for k in ("bler_", "lo_", "hi_")])

cfg = load_cfg(16)
COND = {c[0]: c for c in CONDITIONS}
ch_m = make_channel(cfg, *MATCHED[1:])
CH = {c: (ch_m if c == "S1" else make_channel(cfg, *COND[c][1:])) for c, _ in SHIFTED}


def errors(rx, ch, snr, nbatch, seed):
    """Per-block error indicators on a fresh stream."""
    torch.manual_seed(seed)
    out = []
    for _ in range(nbatch):
        b = ch.generate_batch(BATCH, snr_db=snr)
        with torch.no_grad():
            out.append((decode(rx, b, ch) != b.bits).any(1).float().cpu().numpy())
    return np.concatenate(out)


def crossing(snrs, blers, target=0.1):
    for (s0, b0), (s1, b1) in zip(zip(snrs, blers), zip(snrs[1:], blers[1:])):
        if b0 > target >= b1 and b0 > 0 and b1 > 0:
            f = (np.log10(b0) - np.log10(target)) / (np.log10(b0) - np.log10(b1))
            return s0 + f * (s1 - s0)
    return float("nan")


def gamma_with_ci(err_by_snr, rng):
    blers = [e.mean() for e in err_by_snr]
    point = crossing(GRID, blers)
    boot = []
    for _ in range(BOOT):
        bl = [e[rng.integers(0, len(e), len(e))].mean() for e in err_by_snr]
        g = crossing(GRID, bl)
        if np.isfinite(g):
            boot.append(g)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 20 else (np.nan, np.nan))
    return point, lo, hi


def bler_with_ci(e, rng):
    m = e.mean()
    boot = np.array([e[rng.integers(0, len(e), len(e))].mean() for _ in range(BOOT)])
    return m, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ---------------------------------------------------------------- candidates (discovery data)
R = load_disc(os.path.join(ROOT, "results", "ojcoms", "dense_eval.csv"))
val = {}
vp = os.path.join(ROOT, "results", "ojcoms", "traj_extra.csv")
if os.path.exists(vp):
    for r in csv.DictReader(open(vp)):
        val.setdefault(int(r["seed"]), []).append((float(r["val_bce"]), int(r["step"])))

cands = {}
for s in SEEDS:
    r = R[(s, "ainr")]
    steps, snr, s2 = r["step"], np.asarray(r["snr_at_0p1"], float), np.asarray(r["bler_S2"], float)
    roles = {}
    roles.setdefault(steps[int(np.nanargmin(snr))], []).append("t_match")
    ok = [i for i in range(len(steps)) if snr[i] - np.nanmin(snr) <= CAP_DB + 1e-9]
    i_star = min(ok, key=lambda i: s2[i])
    roles.setdefault(steps[i_star], []).append("t_star")
    for j in (i_star - 1, i_star + 1):                      # frontier neighbours
        if 0 <= j < len(steps):
            roles.setdefault(steps[j], []).append("neighbour")
    for sig, name in (("fe", "t_R1"), ("mbler", "t_R2")):
        sel = evaluate(r, sig)[0]
        if sel is not None:
            roles.setdefault(int(sel), []).append(name)
    roles.setdefault(50000, []).append("t_final")
    if s in val:
        roles.setdefault(min(val[s])[1], []).append("t_val")
    cands[s] = roles

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(int(r["seed"]), int(r["step"])) for r in csv.DictReader(f)}
new = not os.path.exists(OUT)
rng = np.random.default_rng(97531)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for s in SEEDS:
        for step, roles in sorted(cands[s].items()):
            if (s, step) in done:
                continue
            p = os.path.join(ROOT, "results", "dense", f"seed{s}", "checkpoints", f"ainr_step{step}.pt")
            if not os.path.exists(p):
                continue
            rx = AINR(cfg, ch_m).to(DEV)
            rx.load_state_dict(torch.load(p, map_location=DEV))
            rx.eval()
            eb = [errors(rx, ch_m, g, NB_MATCHED, 95000 + int(100 * g)) for g in GRID]
            g0, glo, ghi = gamma_with_ci(eb, rng)
            row = {"seed": s, "step": step, "roles": "+".join(sorted(set(roles))),
                   "snr_at_0p1": f"{g0:.4f}", "snr_lo": f"{glo:.4f}", "snr_hi": f"{ghi:.4f}"}
            for c, snr in SHIFTED:
                m, lo, hi = bler_with_ci(errors(rx, CH[c], snr, NB_SHIFT, OFFSET[c]), rng)
                row[f"bler_{c}"], row[f"lo_{c}"], row[f"hi_{c}"] = f"{m:.6f}", f"{lo:.6f}", f"{hi:.6f}"
            w.writerow(row); f.flush()
            print(f"seed{s} {step:6d} [{row['roles']}] gamma={g0:.3f} "
                  f"({glo:.3f},{ghi:.3f})  S2={row['bler_S2']}  V250={row['bler_V250']}", flush=True)
            del rx
            torch.cuda.empty_cache()
print("DONE")
