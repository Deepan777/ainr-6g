"""Margin-controlled pilot-reliance test.

exp_budget_curve.py measured pilot sensitivity at a fixed 5 dB, where the
longer-trained snapshots hold a small accuracy advantage. This script removes
that confound: each snapshot is evaluated at ITS OWN SNR for a matched block
error rate of 0.1 (from budget_curve.csv), so every snapshot starts from the
same error rate. A finer pilot-noise grid gives the multiplier at which the
block error rate reaches 0.5.

Output: results/ojcoms/pilot_reliance_margin.csv
"""
import csv, os
import numpy as np
import torch

from exp_common import ROOT, DEV, MATCHED, load_cfg, make_channel, atomic_write_csv
from exp_budget_curve import load, bler

MULTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

cr = {}
with open(os.path.join(ROOT, "results", "ojcoms", "budget_curve.csv")) as f:
    for r in csv.DictReader(f):
        cr[(int(r["seed"]), int(r["step"]), r["receiver"])] = float(r["snr_at_0p1"])

cfg = load_cfg(16)
ch = make_channel(cfg, *MATCHED[1:])
rows = []
for (seed, step, kind), snr in sorted(cr.items()):
    rx = load(seed, step, kind)
    vals = [bler(rx, ch, snr, 10, pilot_noise=a, seed=4000) for a in MULTS]
    # multiplier (in dB of extra pilot noise) at which BLER first reaches 0.5
    thr = float("nan")
    for (a0, b0), (a1, b1) in zip(zip(MULTS, vals), zip(MULTS[1:], vals[1:])):
        if b0 < 0.5 <= b1 and a0 > 0:
            f_ = (0.5 - b0) / (b1 - b0)
            thr = 10 * np.log10(a0) + f_ * (10 * np.log10(a1) - 10 * np.log10(a0))
            break
    rows.append({"seed": seed, "step": step, "receiver": kind, "snr_db": round(snr, 3),
                 **{f"bler_x{a:g}": v for a, v in zip(MULTS, vals)}, "pilot_noise_db_at_0p5": thr})
    print(f"seed{seed} {step:6d} {kind:8s} @{snr:.2f}dB  "
          + " ".join(f"x{a:g}={v:.3f}" for a, v in zip(MULTS, vals))
          + f"   thr={thr:.2f} dB", flush=True)
    del rx
    torch.cuda.empty_cache()

atomic_write_csv(os.path.join(ROOT, "results", "ojcoms", "pilot_reliance_margin.csv"), rows)
print("DONE")
