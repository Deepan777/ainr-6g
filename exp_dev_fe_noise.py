"""Development-seed check behind the R1 amendment (ANALYSIS_PROTOCOL_BUDGET.md section 7).

Is the post-convergence variation of the free energy along a training run larger than the Monte Carlo
noise of its estimate? One fixed checkpoint (seed 2, step 25000) is evaluated under six evaluation seeds
with the same 12-batch estimator as exp_dev_fe.py; the spread is compared with the spread over the
post-convergence snapshots (step >= 2000) of the same run in results/ojcoms/dev_fe_trajectory.csv.

Output: results/ojcoms/dev_fe_noise.csv
"""
import csv
import os

import numpy as np
import torch

from exp_common import ROOT, DEV, MATCHED, load_cfg, make_channel
from src.ainr import AINR

cfg = load_cfg(16)
ch = make_channel(cfg, *MATCHED[1:])
p = os.path.join(ROOT, "results", "dense", "seed2", "checkpoints", "ainr_step25000.pt")
m = AINR(cfg, ch).to(DEV)
m.load_state_dict(torch.load(p, map_location=DEV))
m.eval()
rows = []
for es in range(7000, 7006):
    torch.manual_seed(es)
    tot = 0.0
    for _ in range(12):
        b = ch.generate_batch(32, snr_db=5.0)
        with torch.no_grad():
            tot += float(m.compute_free_energy(b.received_grid).total)
    rows.append({"eval_seed": es, "fe": tot / 12})
traj = [float(r["fe"]) for r in csv.DictReader(open(os.path.join(ROOT, "results", "ojcoms", "dev_fe_trajectory.csv")))
        if int(r["seed"]) == 2 and int(r["step"]) >= 2000]
noise_sd = float(np.std([r["fe"] for r in rows], ddof=1))
traj_sd = float(np.std(traj, ddof=1))
out = os.path.join(ROOT, "results", "ojcoms", "dev_fe_noise.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["eval_seed", "fe"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"eval-noise sd {noise_sd:.1f}; across-step sd {traj_sd:.1f}; ratio {traj_sd / noise_sd:.2f} -> {out}")
