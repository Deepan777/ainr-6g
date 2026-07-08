"""S4 (ray-traced Munich) evaluation with confidence intervals.

Repeats the S4 BLER evaluation K times with independent payload bits and noise
(the ray-traced channel set itself is fixed), for both 16-QAM and 64-QAM, and
reports mean +/- std per (modulation, receiver, SNR).

Outputs: results/stage_s4_ci.csv
Usage:   python exp_s4_ci.py [--quick]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from src.channel.sionna_channel import SionnaChannel
from src.ainr import AINR
from src.baselines.lmmse import LMMSEReceiver
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.metrics import hard_bits, count_block_errors

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
K_PASSES = 2 if args.quick else 5
SNRS = [0, 5] if args.quick else [0, 5, 10, 15, 20, 25, 30]
BS = 32

Hs = np.load(os.path.join(_ROOT, "results", "s4_channels.npy"))
rows = []

for qam, ckpt_sub in [(16, "results/checkpoints"), (64, "results/qam64/checkpoints")]:
    cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
    cfg.phy.n_bits_per_symbol = 4 if qam == 16 else 6
    ckpt = os.path.join(_ROOT, ckpt_sub)

    ch = SionnaChannel(cfg, device=dev)
    N_sc, N_sym, N_rx = ch.n_subcarriers, ch.n_symbols, ch.n_rx
    Hs_t = torch.tensor(Hs, dtype=torch.complex64, device=dev)
    N_CH = Hs_t.shape[0]

    ainr = AINR(cfg, ch).to(dev)
    ainr.load_state_dict(torch.load(f"{ckpt}/ainr_final.pt", map_location=dev))
    ainr.eval()
    disc = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(dev)
    disc.load_state_dict(torch.load(f"{ckpt}/discnrx_final.pt", map_location=dev))
    disc.eval()
    lmmse = LMMSEReceiver(cfg, ch).to(dev)
    RX = {"AINR": ainr, "DiscNRX": disc, "LMMSE": lmmse}
    print(f"=== {qam}-QAM | {N_CH} RT channels ===", flush=True)

    def make_hfreq(idx):
        H = Hs_t[idx]
        return H[:, None, :, None, None, None, :].expand(
            H.shape[0], 1, N_rx, 1, 1, N_sym, N_sc).contiguous()

    for name, m in RX.items():
        for snr in SNRS:
            no = float(ch.snr_db_to_no(torch.tensor(float(snr))).item())
            no_t = torch.full((1, 1, 1, 1, 1), no, device=dev)
            blers = []
            for k in range(K_PASSES):
                torch.manual_seed(2000 + k)
                te = tot = 0
                with torch.no_grad():
                    for s in range(0, N_CH, BS):
                        idx = list(range(s, min(s + BS, N_CH)))
                        B = len(idx)
                        info = ch.binary_source([B, 1, ch.n_tx, ch.k])
                        bits = info.reshape(B, ch.num_info_bits)
                        tx = ch.transmit(bits)
                        y = ch.apply_channel(tx, make_hfreq(idx), no_t)
                        Y = ch._received_to_claude_layout(y)
                        if isinstance(m, AINR):
                            llr = m(Y, record_fe=False)
                        elif isinstance(m, LMMSEReceiver):
                            llr = m(Y, no=no)
                        else:
                            llr = m(Y)
                        e, t = count_block_errors(hard_bits(llr), bits)
                        te += e; tot += t
                blers.append(te / tot)
            blers = np.array(blers)
            rows.append({"qam": qam, "receiver": name, "snr_db": snr,
                         "bler_mean": blers.mean(), "bler_std": blers.std(),
                         "n_blocks": N_CH * K_PASSES, "n_passes": K_PASSES})
            print(f"  {qam}-QAM {name:8s} SNR={snr:3d} | "
                  f"BLER={blers.mean():.4f}+/-{blers.std():.4f}", flush=True)
    del ainr, disc, lmmse, ch, RX
    torch.cuda.empty_cache()

out = os.path.join(_ROOT, "results", "stage_s4_ci.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["qam", "receiver", "snr_db", "bler_mean",
                                      "bler_std", "n_blocks", "n_passes"])
    w.writeheader(); w.writerows(rows)
print(f"-> {out}")
print("S4CI_DONE", flush=True)
