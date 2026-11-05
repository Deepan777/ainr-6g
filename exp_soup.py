"""Checkpoint averaging along the optimization trajectory.

Does averaging weights over a window of snapshots reach the robust region of the frontier without
using shifted data? The schemes, streams, sample sizes and decision rule are fixed in
FINAL_OJCOMS/audits/ANALYSIS_PROTOCOL_SOUP.md, written before any averaged model was evaluated.

The networks use GroupNorm, so an averaged model is simply the elementwise mean of the selected
checkpoints and no normalization statistics need recomputation.

Output: results/ojcoms/soup.csv (resumable, one row per model).
"""
import csv
import os

import numpy as np
import torch

from exp_common import ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel, decode
from src.ainr import AINR

SEEDS = [2, 3, 4, 5, 6, 7, 8]
SNAPS = list(range(1000, 10001, 1000)) + list(range(12500, 50001, 2500))
GRID = [round(0.4 + 0.1 * i, 2) for i in range(21)]
NB_MATCHED, NB_SHIFT, BATCH = 16, 30, 32          # 512 and 960 blocks
SHIFTED = [("S2", 5.0), ("DS1000", 5.0)]
OFFSET = {"S2": 98001, "DS1000": 98002}

SCHEMES = {                                        # name -> (first step, last step)
    "ALL": (1000, 50000),
    "EARLY": (5000, 15000),
    "MID": (10000, 25000),
    "TAIL10": (10000, 50000),
    "TAIL20": (20000, 50000),
    "TAIL35": (35000, 50000),
}

OUT = os.path.join(ROOT, "results", "ojcoms", "soup.csv")
FIELDS = ["seed", "model", "n_avg", "snr_at_0p1"] + [f"bler_{c}" for c, _ in SHIFTED]

cfg = load_cfg(16)
COND = {c[0]: c for c in CONDITIONS}
ch_m = make_channel(cfg, *MATCHED[1:])
CH = {c: make_channel(cfg, *COND[c][1:]) for c, _ in SHIFTED}


def ckpt(seed, step):
    return os.path.join(ROOT, "results", "dense", f"seed{seed}", "checkpoints", f"ainr_step{step}.pt")


def average(seed, lo, hi):
    """Elementwise mean of the checkpoints of one trajectory within [lo, hi]."""
    steps = [s for s in SNAPS if lo <= s <= hi and os.path.exists(ckpt(seed, s))]
    acc, n = None, 0
    for s in steps:
        sd = torch.load(ckpt(seed, s), map_location="cpu")
        if acc is None:
            acc = {k: v.double().clone() for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].double()
        n += 1
    return {k: (v / n).to(torch.load(ckpt(seed, steps[0]), map_location="cpu")[k].dtype)
            for k, v in acc.items()}, n


def build(state):
    m = AINR(cfg, ch_m).to(DEV)
    m.load_state_dict(state)
    m.eval()
    return m


def bler(rx, ch, snr, nbatch, seed):
    torch.manual_seed(seed)
    err = tot = 0
    for _ in range(nbatch):
        b = ch.generate_batch(BATCH, snr_db=snr)
        with torch.no_grad():
            err += int((decode(rx, b, ch) != b.bits).any(1).sum())
        tot += BATCH
    return err / tot


def crossing(snrs, blers, target=0.1):
    for (s0, b0), (s1, b1) in zip(zip(snrs, blers), zip(snrs[1:], blers[1:])):
        if b0 > target >= b1 and b0 > 0 and b1 > 0:
            f = (np.log10(b0) - np.log10(target)) / (np.log10(b0) - np.log10(b1))
            return s0 + f * (s1 - s0)
    return float("nan")


def evaluate(rx):
    bl = [bler(rx, ch_m, g, NB_MATCHED, 97000 + int(100 * g)) for g in GRID]
    row = {"snr_at_0p1": f"{crossing(GRID, bl):.4f}"}
    for c, snr in SHIFTED:
        row[f"bler_{c}"] = f"{bler(rx, CH[c], snr, NB_SHIFT, OFFSET[c]):.6f}"
    return row


# reference single checkpoints, taken from the high-precision run's roles
REF = {}
hp = os.path.join(ROOT, "results", "ojcoms", "highprec.csv")
if os.path.exists(hp):
    for r in csv.DictReader(open(hp)):
        for role in ("t_val", "t_star"):
            if role in r["roles"]:
                REF.setdefault(int(r["seed"]), {})[role] = int(r["step"])

done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(int(r["seed"]), r["model"]) for r in csv.DictReader(f)}
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for seed in SEEDS:
        jobs = [(name, None) for name in SCHEMES] + [(role, step) for role, step in REF.get(seed, {}).items()]
        for name, step in jobs:
            if (seed, name) in done:
                continue
            if step is None:
                state, n = average(seed, *SCHEMES[name])
            else:
                if not os.path.exists(ckpt(seed, step)):
                    continue
                state, n = torch.load(ckpt(seed, step), map_location="cpu"), 1
            rx = build(state)
            row = {"seed": seed, "model": name, "n_avg": n}
            row.update(evaluate(rx))
            w.writerow(row); f.flush()
            print(f"seed{seed} {name:7s} n={n:2d} gamma={row['snr_at_0p1']} "
                  f"S2={row['bler_S2']} DS={row['bler_DS1000']}", flush=True)
            del rx, state
            torch.cuda.empty_cache()
print("DONE")
