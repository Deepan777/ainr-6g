"""Dose-response of the Doppler shift along the training trajectory.

traj_extra.csv showed no trend at 250 and 400 km/h, but at those speeds every checkpoint --
including the barely trained one at 1000 iterations -- is already far from the matched operating
point, so the absence of a trend could be a dynamic-range artifact rather than a property of the
shift. This script adds milder speeds (30, 90, 150 km/h; the training range is 3-30 km/h) at the
same SNR, so that the question "does the tradeoff appear at any Doppler level?" is answered on a
condition where an early checkpoint still has room to be better than a converged one.

Streams use seeds disjoint from every earlier evaluation. Output: results/ojcoms/traj_doppler.csv
(resumable).
"""
import csv
import os

import torch

from exp_common import ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel
from exp_budget_curve import bler
from src.ainr import AINR

SEEDS = [2, 3, 4, 5, 6, 7, 8]
STEPS = [1000, 2000, 3000, 5000, 7000, 10000, 20000, 35000, 50000]
SPEEDS = ["V30", "V90", "V150"]
SNR_DB = 10.0
NB = 30                                       # 960 blocks per cell
STREAM = {"V30": 9401, "V90": 9402, "V150": 9403}

OUT = os.path.join(ROOT, "results", "ojcoms", "traj_doppler.csv")
FIELDS = ["seed", "step"] + [f"bler_{s}" for s in SPEEDS]

cfg = load_cfg(16)
COND = {c[0]: c for c in CONDITIONS}
ch_m = make_channel(cfg, *MATCHED[1:])
CH = {s: make_channel(cfg, *COND[s][1:]) for s in SPEEDS}

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(int(r["seed"]), int(r["step"])) for r in csv.DictReader(f)}
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for seed in SEEDS:
        for step in STEPS:
            if (seed, step) in done:
                continue
            p = os.path.join(ROOT, "results", "dense", f"seed{seed}", "checkpoints", f"ainr_step{step}.pt")
            if not os.path.exists(p):
                continue
            rx = AINR(cfg, ch_m).to(DEV)
            rx.load_state_dict(torch.load(p, map_location=DEV))
            rx.eval()
            row = {"seed": seed, "step": step}
            for s in SPEEDS:
                row[f"bler_{s}"] = f"{bler(rx, CH[s], SNR_DB, NB, seed=STREAM[s]):.6f}"
            w.writerow(row); f.flush()
            print(f"seed{seed} {step:6d} " + "  ".join(f"{s}={row['bler_' + s]}" for s in SPEEDS), flush=True)
            del rx
            torch.cuda.empty_cache()
print("DONE")
