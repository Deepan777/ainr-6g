"""Training budget: the robustness-accuracy trade-off, and its mechanism.

(A) Trade-off curve. For each snapshot (12k, 25k, 50k steps) measure the matched
    waterfall (SNR at BLER 0.1) and out-of-distribution BLER at 5 dB (S2, DS1000).

(B) Mechanism test. Hypothesis: longer training shifts the receiver from using
    pilot evidence to using memorised channel statistics. Prediction: later
    snapshots are LESS sensitive to pilot degradation (they lean on the prior)
    while being MORE sensitive to channel shift. Extra noise is injected on the
    pilot OFDM symbols only, at multiples of the data-RE noise variance.

Output: results/ojcoms/budget_curve.csv, results/ojcoms/pilot_reliance.csv
"""
import os
import numpy as np
import torch

from exp_common import (ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel,
                        decode, atomic_write_csv)
from src.ainr import AINR
from src.baselines.discriminative_nrx import DiscriminativeNRX

cfg = load_cfg(16)
SEEDS, STEPS = [2, 3, 4], [12000, 25000, 50000]
COND = {c[0]: c for c in CONDITIONS}
ch_m = make_channel(cfg, *MATCHED[1:])
PIDX = torch.tensor(list(cfg.phy.pilot_ofdm_symbol_indices), device=DEV)


def load(seed, step, kind):
    d = os.path.join(ROOT, "results", "seeds", f"seed{seed}", "checkpoints")
    p = os.path.join(d, f"{kind}_final.pt" if step == 50000 else f"{kind}_step{step}.pt")
    if kind == "ainr":
        m = AINR(cfg, ch_m).to(DEV)
    else:
        m = DiscriminativeNRX(cfg, pilot_grid=ch_m.pilot_grid).to(DEV)
    m.load_state_dict(torch.load(p, map_location=DEV))
    m.eval()
    if kind != "ainr":
        m._ensure_ldpc(DEV)
    return m


def bler(rx, ch, snr, nb, pilot_noise=0.0, seed=0):
    torch.manual_seed(seed)
    e = t = 0
    for _ in range(nb):
        b = ch.generate_batch(32, snr_db=snr)
        Y = b.received_grid
        if pilot_noise > 0:
            no = float(b.no.reshape(-1)[0]) * pilot_noise
            z = torch.randn(Y.shape[0], Y.shape[1], len(PIDX), Y.shape[3], dtype=torch.cfloat, device=Y.device)
            Y = Y.clone().index_add_(2, PIDX, z * (no / 2) ** 0.5)
            b = b._replace(received_grid=Y) if hasattr(b, "_replace") else b
            if not hasattr(b, "_replace"):
                b.received_grid = Y
        with torch.no_grad():
            e += int((decode(rx, b, ch) != b.bits).any(1).sum())
        t += 32
    return e / t


def crossing(pts, target=0.1):
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if b0 > target >= b1 and b0 > 0 and b1 > 0:
            f = (np.log10(b0) - np.log10(target)) / (np.log10(b0) - np.log10(b1))
            return s0 + f * (s1 - s0)
    return float("nan")


if __name__ == "__main__":
    grid = [round(0.25 * i, 2) for i in range(0, 11)]        # 0.00 .. 2.50 dB
    rowsA, rowsB = [], []
    for seed in SEEDS:
        for step in STEPS:
            for kind in ("ainr", "discnrx"):
                rx = load(seed, step, kind)
                pts = [(s, bler(rx, ch_m, s, 8, seed=1000 + int(100 * s))) for s in grid]
                cr = crossing(pts)
                ood = {c: bler(rx, make_channel(cfg, *COND[c][1:]), 5.0, 15, seed=2000) for c in ("S2", "DS1000")}
                rowsA.append({"seed": seed, "step": step, "receiver": kind, "snr_at_0p1": cr,
                              **{f"bler_{c}": v for c, v in ood.items()}})
                print(f"[A] seed{seed} {step:6d} {kind:8s} waterfall={cr:.2f} dB  "
                      f"S2={ood['S2']:.3f}  DS1000={ood['DS1000']:.3f}", flush=True)
                for a in (0.0, 1.0, 4.0, 16.0, 64.0):
                    v = bler(rx, ch_m, 5.0, 8, pilot_noise=a, seed=3000)
                    rowsB.append({"seed": seed, "step": step, "receiver": kind,
                                  "pilot_noise_mult": a, "bler": v})
                pr = [r["bler"] for r in rowsB if r["seed"] == seed and r["step"] == step and r["receiver"] == kind]
                print(f"[B] seed{seed} {step:6d} {kind:8s} pilot-noise x(0,1,4,16,64): "
                      + " ".join(f"{v:.3f}" for v in pr), flush=True)
                del rx
                torch.cuda.empty_cache()

    atomic_write_csv(os.path.join(ROOT, "results", "ojcoms", "budget_curve.csv"), rowsA)
    atomic_write_csv(os.path.join(ROOT, "results", "ojcoms", "pilot_reliance.csv"), rowsB)
    print("DONE")
