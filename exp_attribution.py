"""Attribution of the reconstruction error to its sources.

The model-fit residual depends jointly on the bit posterior, the channel posterior, and the
representational limits of the physical model. This script separates them by evaluating the
normalized reconstruction error

    e = E|Y - Yhat|^2 / E|Y|^2

for four combinations of genie and inferred quantities:

    (c*, h*)   both genie          -> representation error of the physical model alone
    (c*, q(h)) genie bits          -> adds the channel-posterior error
    (q(c), h*) genie channel       -> adds the bit-posterior error
    (q(c), q(h))                   -> the residual the monitor actually computes

h* is the least-squares projection of the true, slot-averaged channel onto the L delay taps of
the likelihood, i.e. the best channel the model can represent. Conditions: matched, 1000-ns delay
spread and S2 at 5 dB, and 250 km/h at 10 dB. Converged receivers of the main study (seeds 0-4).

Output: results/ojcoms/attribution.csv
"""
import csv
import os

import torch

from exp_common import ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel, load_neural
from src.ainr import AINR

SEEDS = [0, 1, 2, 3, 4]
CONDS = [("S1", 5.0), ("DS1000", 5.0), ("S2", 5.0), ("V250", 10.0)]
NB, BATCH = 20, 32
COND = {c[0]: c for c in CONDITIONS}
cfg = load_cfg(16)
ch_m = make_channel(cfg, *MATCHED[1:])


def genie_taps(gm, h_freq):
    """Least-squares projection of the slot-averaged true channel onto the L delay taps."""
    # h_freq: (B,1,N_rx,1,N_tx,N_sym,N_sc) -> average over OFDM symbols
    H = h_freq.mean(dim=-2)                                   # (B,1,N_rx,1,N_tx,N_sc)
    B = H.shape[0]
    H = H.reshape(B, gm.N_rx, gm.N_tx, gm.N_sc)
    Dp = torch.linalg.pinv(gm.dft.to(H.dtype))                # (N_sc, L), dft is (L, N_sc)
    taps = H @ Dp                                             # (B, N_rx, N_tx, L)
    return taps.unsqueeze(0)


def err_db(Y, Yhat):
    num = (Y - Yhat).abs().pow(2).sum()
    den = Y.abs().pow(2).sum()
    return float(10.0 * torch.log10(num / den))


rows = []
for seed in SEEDS:
    ainr, _ = load_neural(cfg, ch_m, seed)
    gm = ainr.generative_model
    for cname, snr in CONDS:
        ch = ch_m if cname == "S1" else make_channel(cfg, *COND[cname][1:])
        torch.manual_seed(9404)
        acc = {k: 0.0 for k in ("gg", "gq", "qg", "qq")}
        for _ in range(NB):
            b = ch.generate_batch(BATCH, snr_db=snr)
            Y = b.received_grid
            with torch.no_grad():
                p = ainr.posterior.forward(Y)
                s = ainr.posterior.sample_from_params(p, 1)
                cstar = ch.encoder(b.bits.to(Y.device, torch.float32)
                                   .reshape(BATCH, 1, ainr.n_tx, ainr.k)).reshape(1, BATCH, ainr.posterior.n_coded)
                hstar = genie_taps(gm, b.h_freq).to(s.h_samples.dtype)
                qh = p.h_mean.unsqueeze(0).to(s.h_samples.dtype)
                acc["gg"] += err_db(Y, gm.predict_grid(cstar, hstar)[0])
                acc["gq"] += err_db(Y, gm.predict_grid(cstar, qh)[0])
                acc["qg"] += err_db(Y, gm.predict_grid(s.bits_soft, hstar)[0])
                acc["qq"] += err_db(Y, gm.predict_grid(s.bits_soft, qh)[0])
        rows.append({"seed": seed, "cond": cname, "snr_db": snr,
                     **{k: round(v / NB, 3) for k, v in acc.items()}})
        print(f"seed{seed} {cname:7s} {snr:4.1f} dB  "
              + "  ".join(f"{k}={rows[-1][k]:7.2f}" for k in ("gg", "gq", "qg", "qq")), flush=True)
    del ainr
    torch.cuda.empty_cache()

out = os.path.join(ROOT, "results", "ojcoms", "attribution.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "cond", "snr_db", "gg", "gq", "qg", "qq"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("DONE", out)
