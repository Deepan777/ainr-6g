"""Evaluate trained receivers on the S4 ray-traced (Munich) channel set.

Loads results/s4_channels.npy, applies each RT channel through the OFDM transmit
chain (with the standard pilots) + AWGN, and measures BLER vs SNR for AINR,
DiscNRX, and LMMSE — an out-of-distribution realism test (training was CDL-C).

    python s4_eval.py                         # 16-QAM (results/checkpoints)
    python s4_eval.py phy.n_bits_per_symbol=6 paths.checkpoint_dir=results/qam64/checkpoints
"""
import os, sys, csv
import numpy as np
import torch
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _ROOT)
from src.channel.sionna_channel import SionnaChannel
from src.ainr import AINR
from src.baselines.lmmse import LMMSEReceiver
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.metrics import hard_bits, count_block_errors

cfg = OmegaConf.merge(OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml")),
                      OmegaConf.from_cli())
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt = cfg.paths.checkpoint_dir
ckpt = ckpt if os.path.isabs(ckpt) else os.path.join(_ROOT, ckpt)

ch = SionnaChannel(cfg, device=dev)
N_sc, N_sym, N_rx = ch.n_subcarriers, ch.n_symbols, ch.n_rx
Hs = np.load(os.path.join(_ROOT, "results", "s4_channels.npy"))      # (N_CH, N_rx, N_sc)
Hs_t = torch.tensor(Hs, dtype=torch.complex64, device=dev)
N_CH = Hs_t.shape[0]
print(f"Loaded {N_CH} RT channels; QAM={2**ch.n_bits_per_symbol}; ckpt={ckpt}", flush=True)

ainr = AINR(cfg, ch).to(dev); ainr.load_state_dict(torch.load(f"{ckpt}/ainr_final.pt", map_location=dev)); ainr.eval()
disc = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(dev); disc.load_state_dict(torch.load(f"{ckpt}/discnrx_final.pt", map_location=dev)); disc.eval()
lmmse = LMMSEReceiver(cfg, ch).to(dev)
RX = {"AINR": ainr, "DiscNRX": disc, "LMMSE": lmmse}

SNRS = [0, 5, 10, 15, 20, 25, 30]
BS = 32


def make_hfreq(idx):
    H = Hs_t[idx]                                   # (B, N_rx, N_sc)
    return H[:, None, :, None, None, None, :].expand(
        H.shape[0], 1, N_rx, 1, 1, N_sym, N_sc).contiguous()


rows = []
for name, m in RX.items():
    for snr in SNRS:
        no = float(ch.snr_db_to_no(torch.tensor(float(snr))).item())
        no_t = torch.full((1, 1, 1, 1, 1), no, device=dev)
        te = tot = 0
        with torch.no_grad():
            for s in range(0, N_CH, BS):
                idx = list(range(s, min(s + BS, N_CH))); B = len(idx)
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
                e, t = count_block_errors(hard_bits(llr), bits); te += e; tot += t
        bler = te / tot
        rows.append({"receiver": name, "scenario": "S4", "snr_db": snr, "bler": bler})
        print(f"  {name:8s} SNR={snr:3d} dB | BLER={bler:.4f}", flush=True)

tag = f"qam{2**ch.n_bits_per_symbol}"
out = os.path.join(_ROOT, "results", f"stage_s4_{tag}.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["receiver", "scenario", "snr_db", "bler"]); w.writeheader(); w.writerows(rows)
print(f"-> {out}")
print("S4EVAL_DONE", flush=True)
