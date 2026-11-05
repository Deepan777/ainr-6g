"""Probe 2: does the physics likelihood substitute for pilot overhead?

Trains-once comparison on the matched channel with a SINGLE DMRS symbol
(13 data symbols, K = 1976) instead of two (12 data symbols, K = 1824):
8.3 % more payload per slot, but a far weaker channel estimate. The question is
whether the AINR, which has an explicit likelihood and code constraints, loses
less than the parameter-matched discriminative receiver, which has no model.

Both learned receivers see identical slots at every SNR. LS-LIN and perfect CSI
are shown for context (LS-LMMSE is skipped: its cached covariances belong to the
two-pilot grid).

Output: results/ojcoms/pilot1_matched.csv
"""
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from exp_common import ROOT, DEV, MATCHED, make_channel, decode, atomic_write_csv
from src.ainr import AINR
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.baselines.classical import ClassicalReceiver

cfg = OmegaConf.load(os.path.join(ROOT, "config", "config.yaml"))
cfg.phy.n_bits_per_symbol = 4
cfg.phy.pilot_ofdm_symbol_indices = [7]
ch = make_channel(cfg, *MATCHED[1:])
print(f"single-pilot link: K={ch.k}  n={ch.n}  (two-pilot: K=1824, n=3648)", flush=True)

ck = os.path.join(ROOT, "results", "pilot1", "checkpoints")
ainr = AINR(cfg, ch).to(DEV)
ainr.load_state_dict(torch.load(os.path.join(ck, "ainr_final.pt"), map_location=DEV))
ainr.eval()
disc = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(DEV)
disc.load_state_dict(torch.load(os.path.join(ck, "discnrx_final.pt"), map_location=DEV))
disc.eval(); disc._ensure_ldpc(DEV)
RX = {"AINR": ainr, "DiscNRX": disc,
      "LS-LIN": ClassicalReceiver(cfg, ch, "ls_lin"),
      "PCSI": ClassicalReceiver(cfg, ch, "perfect")}

B, NB = 32, 15
grid = [round(x, 2) for x in np.arange(-1.0, 6.01, 0.25)]
rows = []
for i, snr in enumerate(grid):
    torch.manual_seed(60_000 + 100 * i)
    err = {k: 0 for k in RX}
    for _ in range(NB):
        bt = ch.generate_batch(B, snr_db=float(snr))
        with torch.no_grad():
            for k, rx in RX.items():
                err[k] += int((decode(rx, bt, ch) != bt.bits).any(1).sum())
    n = NB * B
    rows += [{"receiver": k, "snr_db": snr, "block_errors": err[k], "n_blocks": n,
              "bler": err[k] / n} for k in RX]
    print(f"{snr:5.2f} dB  " + "  ".join(f"{k}={err[k]/n:.3f}" for k in RX), flush=True)
atomic_write_csv(os.path.join(ROOT, "results", "ojcoms", "pilot1_matched.csv"), rows)


def crossing(rx, target=0.1):
    pts = sorted((r["snr_db"], r["bler"]) for r in rows if r["receiver"] == rx)
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if b0 > target >= b1 and b0 > 0 and b1 > 0:
            f = (np.log10(b0) - np.log10(target)) / (np.log10(b0) - np.log10(b1))
            return s0 + f * (s1 - s0)
    return float("nan")


print("\n=== SNR at BLER 0.1, single pilot ===")
for k in RX:
    print(f"{k:8s} {crossing(k):.2f} dB")
print("two-pilot reference (seed 0): AINR 0.87, DiscNRX 0.86 dB")
