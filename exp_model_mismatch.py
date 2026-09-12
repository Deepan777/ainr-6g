"""Representation error of the AINR likelihood's channel model (no neural network).

The generative model represents the channel of each receive antenna by L = 16 delay
taps on a DFT basis (tap spacing 1/(N_sc * df) = 438.6 ns, span 7.0 us) and holds it
constant over the 14 OFDM symbols of a slot. For every test condition this script
measures how much of the true frequency-domain channel H[k, l, r] these two
assumptions cannot represent, even with a perfect (genie) choice of taps:

  e_time  = ||H - Hbar||^2 / ||H||^2           Hbar = mean over OFDM symbols
  e_taps  = ||Hbar - P Hbar||^2 / ||H||^2      P = least-squares projection on the
                                               first L DFT delay bins
  e_total = ||H - P Hbar||^2 / ||H||^2  (= e_time + e_taps, orthogonal parts)

At SNR s the noise-to-signal ratio is 10^(-s/10), so a representation error above
that level is a structural mismatch that no posterior can remove.

Output: results/ojcoms/model_mismatch.csv
"""
import argparse
import math
import os

import torch

from exp_common import ROOT, DEV, CONDITIONS, load_cfg, make_channel, atomic_write_csv

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, default=20)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.batches = 1

B = 32
cfg = load_cfg(16)
L = int(cfg.model.n_channel_taps)
rows = []
for ci, (name, cdl, ds, v) in enumerate(CONDITIONS):
    ch = make_channel(cfg, cdl, ds, v)
    N = ch.n_subcarriers if hasattr(ch, "n_subcarriers") else int(cfg.phy.n_subcarriers)
    k = torch.arange(N, device=DEV, dtype=torch.float32)
    l = torch.arange(L, device=DEV, dtype=torch.float32)
    F = torch.exp(-2j * math.pi * torch.outer(k, l) / N).to(torch.complex64)   # (N, L)
    P = F @ torch.linalg.pinv(F)                                               # (N, N) projector
    torch.manual_seed(55_000 + ci)
    num_t = num_l = den = 0.0
    for _ in range(args.batches):
        bt = ch.generate_batch(B, snr_db=10.0)
        H = bt.h_freq
        # Sionna layout: [..., num_ofdm_symbols, fft_size]; flatten everything else.
        H = H.reshape(-1, H.shape[-2], H.shape[-1]).to(torch.complex64)         # (M, N_sym, N_sc)
        Hbar = H.mean(dim=1, keepdim=True)
        PH = (P @ Hbar.transpose(1, 2)).transpose(1, 2)                         # project over subcarriers
        num_t += float((H - Hbar).abs().pow(2).sum())
        num_l += float((Hbar - PH).abs().pow(2).sum()) * H.shape[1]
        den += float(H.abs().pow(2).sum())
    et, el = num_t / den, num_l / den
    rows.append({"cond": name, "cdl": cdl, "ds_ns": ds, "speed_kmh": v, "e_time": et, "e_taps": el,
                 "e_total": et + el, "e_total_db": 10 * math.log10(max(et + el, 1e-12)),
                 "n_slots": args.batches * B})
    print(f"{name:7s} e_time={et:.2e} e_taps={el:.2e} total={10*math.log10(max(et+el,1e-12)):6.1f} dB",
          flush=True)
atomic_write_csv(os.path.join(ROOT, "results", "ojcoms",
                              f"model_mismatch{'_quick' if args.quick else ''}.csv"), rows)
print("MISMATCH_DONE", flush=True)
