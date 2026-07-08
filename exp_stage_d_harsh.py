"""Gating ablation at a harsh post-drift operating point (S2, low SNR).

At 5 dB the drifted receiver still decodes ~2/3 of blocks, so even ungated
decision-directed adaptation has mostly-correct self-labels and recovers.  The
regime that motivates CRC gating is a harsher one, where most self-labels are
wrong.  This script repeats the Stage-D protocol at SNR = HARSH_SNR dB (drifted
BLER well above 0.5) with:

  gated    — AINR, CRC-gated, seeds 1..3
  ungated  — AINR, adapt on every slot with its own decisions, seeds 1..3
  none     — AINR, no adaptation, seed 1 (reference)

Outputs: results/stage_d_harsh.csv, results/stage_d_harsh_summary.csv
Usage:   python exp_stage_d_harsh.py [--quick]
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
from src.evaluation.scenarios import build_scenario_channel
from src.evaluation.metrics import hard_bits, compute_bler

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(_ROOT, "results", "checkpoints")

N_SLOTS = 20 if args.quick else 1500
HARSH_SNR = 3.0
LR = float(cfg.eval.adaptation_lr)

ch = build_scenario_channel(cfg, "s2_delay_shift", dev)


def run(mode, seed):
    rx = AINR(cfg, ch).to(dev)
    rx.load_state_dict(torch.load(f"{CKPT}/ainr_final.pt", map_location=dev))
    rx.eval()
    torch.manual_seed(seed)
    rows = []
    t0 = time.time()
    for slot in range(N_SLOTS):
        b = ch.generate_batch(1, snr_db=HARSH_SNR)
        with torch.no_grad():
            llr = rx(b.received_grid, record_fe=False)
        decoded = hard_bits(llr)
        bler = compute_bler(decoded, b.bits)
        crc = bool((decoded > 0.5).eq(b.bits > 0.5).all().item())
        if mode == "gated" and crc:
            rx.adapt_online(b.received_grid, decoded, lr=LR)
        elif mode == "ungated":
            rx.adapt_online(b.received_grid, decoded, lr=LR)
        rows.append({"mode": mode, "seed": seed, "slot": slot,
                     "bler": bler, "crc_pass": int(crc)})
    blers = np.array([r["bler"] for r in rows])
    k = min(300, N_SLOTS)
    print(f"[{mode}|seed={seed}] first-{k} BLER={blers[:k].mean():.3f} "
          f"final-{k} BLER={blers[-k:].mean():.3f} ({time.time()-t0:.0f}s)",
          flush=True)
    summary = {"mode": mode, "seed": seed,
               "first300_bler": float(blers[:k].mean()),
               "final300_bler": float(blers[-k:].mean()),
               "crc_rate_final300": float(np.mean(
                   [r["crc_pass"] for r in rows[-k:]]))}
    del rx
    torch.cuda.empty_cache()
    return rows, summary


all_rows, summaries = [], []
seeds = [1] if args.quick else [1, 2, 3]
for mode in ["gated", "ungated"]:
    for seed in seeds:
        r, s = run(mode, seed)
        all_rows += r; summaries.append(s)
r, s = run("none", 1)
all_rows += r; summaries.append(s)

with open(os.path.join(_ROOT, "results", "stage_d_harsh.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "seed", "slot", "bler", "crc_pass"])
    w.writeheader(); w.writerows(all_rows)
with open(os.path.join(_ROOT, "results", "stage_d_harsh_summary.csv"), "w",
          newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "seed", "first300_bler",
                                      "final300_bler", "crc_rate_final300"])
    w.writeheader(); w.writerows(summaries)
print("-> results/stage_d_harsh.csv")
print("HARSH_DONE", flush=True)
