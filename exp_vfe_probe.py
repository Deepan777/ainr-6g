"""Probe 1: label-free (unsupervised) free-energy adaptation at test time.

Question: in the regime where the CRC gate is dead (S2 at 3 dB the receiver
passes no CRC, so gated adaptation never starts and a discriminative receiver
has no labels at all), can the AINR still recover by minimising its variational
free energy from the received signal alone?

The update uses no bit labels, no CRC and no ground truth: one Adam step per
slot on F(phi; Y) = KL_bits + KL_h + KL_sigma - E_q[log p(Y | c, h, sigma^2)].
Ground truth is read only to *report* block errors; the CRC verdict is recorded
for reporting and never gates the update. Decoded-bit statistics (ones fraction,
mean |LLR|) are logged to detect the degenerate all-zero attractor that ungated
decision-directed training falls into.

DiscNRX cannot run this mode: it has no generative model and hence no label-free
objective. That asymmetry is the point of the probe.

Output: results/ojcoms/vfe_probe_{cond}_snr{snr}.csv (one row per slot)
Usage : python exp_vfe_probe.py [--seeds 0 1 2 3 4] [--slots 1000]
"""
import argparse
import os
import time

import numpy as np
import torch

from exp_common import (ROOT, CONDITIONS, MATCHED, load_cfg, make_channel,
                        load_neural, atomic_write_csv)
from src.evaluation.metrics import hard_bits

ap = argparse.ArgumentParser()
ap.add_argument("--cond", default="S2")
ap.add_argument("--snr", type=float, default=3.0)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--slots", type=int, default=1000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--modes", nargs="+", default=["none", "vfe"])
ap.add_argument("--tag", default="")
args = ap.parse_args()

cfg = load_cfg(16)
cond = {c[0]: c for c in CONDITIONS}[args.cond]
out_dir = os.path.join(ROOT, "results", "ojcoms")
os.makedirs(out_dir, exist_ok=True)

ch_m = make_channel(cfg, *MATCHED[1:])
ch_d = make_channel(cfg, *cond[1:])
torch.manual_seed(777)                      # same drifted stream as exp_adapt.py
stream = [ch_d.generate_batch(1, snr_db=args.snr) for _ in range(args.slots)]

rows = []
for seed in args.seeds:
    for mode in args.modes:
        ainr, disc = load_neural(cfg, ch_m, seed)
        del disc
        rx = ainr
        t0 = time.time()
        for t, b in enumerate(stream):
            Y = b.received_grid
            with torch.no_grad():
                llr = rx(Y, record_fe=False)
            hb = hard_bits(llr)
            err = bool((hb != b.bits).any())
            crc = bool(ch_d.crc_check(hb)[0])
            fe = float("nan")
            if mode == "vfe":
                opt = rx._ensure_optimizer(args.lr)
                rx.train()
                opt.zero_grad(set_to_none=True)
                out = rx.compute_free_energy(Y)
                out.total.backward()
                torch.nn.utils.clip_grad_norm_(rx.posterior.parameters(), 1.0)
                opt.step()
                rx.eval()
                fe = float(out.total.detach())
            rows.append({"seed": seed, "mode": mode, "slot": t, "err": int(err),
                         "crc_pass": int(crc), "ones": float(hb.float().mean()),
                         "abs_llr": float(llr.abs().mean()), "free_energy": fe})
        sel = [r for r in rows if r["seed"] == seed and r["mode"] == mode]
        e = np.array([r["err"] for r in sel])
        k = max(1, len(e) // 5)
        print(f"[seed {seed}] {mode:5s} first-{k}={e[:k].mean():.3f} last-{k}={e[-k:].mean():.3f} "
              f"crc={np.mean([r['crc_pass'] for r in sel]):.3f} "
              f"ones={sel[-1]['ones']:.3f} |llr|={sel[-1]['abs_llr']:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del ainr, rx
        torch.cuda.empty_cache()

path = os.path.join(out_dir, f"vfe_probe_{args.cond}_snr{args.snr:g}{args.tag}.csv")
atomic_write_csv(path, rows)
print(f"-> {path} ({len(rows)} rows)")
print("NOTE: DiscNRX cannot run the label-free mode (it has no generative model).")
