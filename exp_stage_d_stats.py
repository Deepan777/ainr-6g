"""Multi-seed online-adaptation study (Stage D) + gating and LR ablations.

Protocol per run: the receiver starts from the trained matched checkpoint and
faces the S2 delay-drift channel (CDL-A, 1000 ns) at a fixed SNR, one slot at a
time, for N_SLOTS slots.  A slot is "CRC-passed" when the decoded block matches
the transmitted block (an emulated CRC; a real CRC-24 differs only by a 2^-24
false-pass probability).

Runs:
  gated       — CRC-gated adaptation, AINR and DiscNRX, seeds 1..N_SEEDS
  ungated     — adapt on EVERY slot with the receiver's own (possibly wrong)
                decisions, AINR, seeds 1..2   (the failure mode CRC gating avoids)
  none        — no adaptation at all, AINR, seed 1 (reference floor)
  lr sweep    — gated AINR, seed 1, lr in {3e-5, 3e-4, 1e-3}
                (1e-4 is the default, covered by the gated run at seed 1)

Outputs:
  results/stage_d_stats.csv    (mode, receiver, lr, seed, slot, bler, crc_pass)
  results/stage_d_summary.csv  (mode, receiver, lr, seed, recovery_slot,
                                final300_bler, crc_rate_final300)

Usage:  python exp_stage_d_stats.py [--quick]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from src.ainr import AINR
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.scenarios import build_scenario_channel
from src.evaluation.metrics import hard_bits, compute_bler

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(_ROOT, "results", "checkpoints")

N_SLOTS = 20 if args.quick else 1500
N_SEEDS = 2 if args.quick else 5
SNR = 5.0
DEFAULT_LR = float(cfg.eval.adaptation_lr)   # 1e-4
LR_SWEEP = [3e-5, 3e-4, 1e-3]

ch = build_scenario_channel(cfg, "s2_delay_shift", dev)
no = float(ch.snr_db_to_no(torch.tensor(SNR)).item())


def fresh(name):
    if name == "AINR":
        m = AINR(cfg, ch).to(dev)
        m.load_state_dict(torch.load(f"{CKPT}/ainr_final.pt", map_location=dev))
    else:
        m = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(dev)
        m.load_state_dict(torch.load(f"{CKPT}/discnrx_final.pt", map_location=dev))
    m.eval()
    return m


def run(mode, name, lr, seed):
    """One adaptation run; returns per-slot rows."""
    rx = fresh(name)
    torch.manual_seed(seed)          # channel stream identical across receivers
    rows = []
    t0 = time.time()
    for slot in range(N_SLOTS):
        b = ch.generate_batch(1, snr_db=SNR)
        with torch.no_grad():
            if isinstance(rx, AINR):
                llr = rx(b.received_grid, record_fe=False)
            else:
                llr = rx(b.received_grid)
        decoded = hard_bits(llr)
        bler = compute_bler(decoded, b.bits)
        crc = bool((decoded > 0.5).eq(b.bits > 0.5).all().item())
        if mode == "gated" and crc:
            rx.adapt_online(b.received_grid, decoded, lr=lr)
        elif mode == "ungated":
            rx.adapt_online(b.received_grid, decoded, lr=lr)
        rows.append({"mode": mode, "receiver": name, "lr": lr, "seed": seed,
                     "slot": slot, "bler": bler, "crc_pass": int(crc)})
    blers = np.array([r["bler"] for r in rows])
    k = min(300, N_SLOTS)
    # Recovery slot: first slot from which the trailing-50 rolling BLER < 0.2.
    rec = -1
    w = min(50, N_SLOTS)
    roll = np.convolve(blers, np.ones(w) / w, mode="valid")
    idx = np.where(roll < 0.2)[0]
    if idx.size:
        rec = int(idx[0] + w - 1)
    crc_rate = np.mean([r["crc_pass"] for r in rows[-k:]])
    print(f"[{mode}|{name}|lr={lr:g}|seed={seed}] "
          f"final-{k} BLER={blers[-k:].mean():.3f} recovery_slot={rec} "
          f"crc_rate={crc_rate:.3f} ({time.time()-t0:.0f}s)", flush=True)
    summary = {"mode": mode, "receiver": name, "lr": lr, "seed": seed,
               "recovery_slot": rec, "final300_bler": float(blers[-k:].mean()),
               "crc_rate_final300": float(crc_rate)}
    del rx
    torch.cuda.empty_cache()
    return rows, summary


all_rows, summaries = [], []

for seed in range(1, N_SEEDS + 1):
    for name in ["AINR", "DiscNRX"]:
        r, s = run("gated", name, DEFAULT_LR, seed)
        all_rows += r; summaries.append(s)

for seed in [1, 2][: max(1, N_SEEDS // 2)]:
    r, s = run("ungated", "AINR", DEFAULT_LR, seed)
    all_rows += r; summaries.append(s)

r, s = run("none", "AINR", 0.0, 1)
all_rows += r; summaries.append(s)

for lr in LR_SWEEP:
    r, s = run("gated", "AINR", lr, 1)
    all_rows += r; summaries.append(s)

with open(os.path.join(_ROOT, "results", "stage_d_stats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "receiver", "lr", "seed", "slot",
                                      "bler", "crc_pass"])
    w.writeheader(); w.writerows(all_rows)
with open(os.path.join(_ROOT, "results", "stage_d_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "receiver", "lr", "seed",
                                      "recovery_slot", "final300_bler",
                                      "crc_rate_final300"])
    w.writeheader(); w.writerows(summaries)
print("-> results/stage_d_stats.csv")
print("-> results/stage_d_summary.csv")
print("STAGED_DONE", flush=True)
