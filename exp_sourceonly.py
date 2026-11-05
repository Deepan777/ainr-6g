"""Source-only checkpoint scores, computed from matched data alone.

Protocol (fixed before any score was computed):
FINAL_OJCOMS/audits/ANALYSIS_PROTOCOL_SOURCEONLY.md

For every checkpoint of every trajectory this records four scores derived from the encoder
representation and from the flatness of the matched loss. No shifted data are used.

Output: results/ojcoms/sourceonly.csv (resumable).
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from exp_common import ROOT, DEV, MATCHED, load_cfg, make_channel
from src.ainr import AINR
from src.channel.crc_channel import CRCSionnaChannel

SEEDS = [2, 3, 4, 5, 6, 7, 8]
STEPS = list(range(1000, 10001, 1000)) + list(range(12500, 50001, 2500))
N_FEAT_BATCH, BATCH = 8, 32          # 256 slots for the representation statistics
N_LOSS_BATCH = 4                     # 128 slots for the loss and its perturbed versions
N_PERTURB, REL_SIGMA = 5, 0.01
OUT = os.path.join(ROOT, "results", "ojcoms", "sourceonly.csv")
FIELDS = ["seed", "step", "effrank", "logdet", "featnorm", "sharpness", "bce"]

cfg = load_cfg(16)
ch_m = make_channel(cfg, *MATCHED[1:])
ch_tr = CRCSionnaChannel(cfg, device=DEV)


def fixed_batches(n, snr=None, seed=99001):
    torch.manual_seed(seed)
    ch = ch_m if snr is not None else ch_tr
    return [ch.generate_batch(BATCH, snr_db=snr) if snr is not None else ch.generate_batch(BATCH)
            for _ in range(n)]


FEAT_B = fixed_batches(N_FEAT_BATCH, snr=5.0, seed=99001)
LOSS_B = fixed_batches(N_LOSS_BATCH, snr=5.0, seed=99002)


def coded_target(rx, b):
    ib = b.bits.to(device=b.received_grid.device, dtype=torch.float32)
    return ch_m.encoder(ib.reshape(BATCH, 1, rx.n_tx, rx.k)).reshape(BATCH, rx.posterior.n_coded)


def matched_bce(rx):
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in LOSS_B:
            p = rx.posterior.forward(b.received_grid)
            t = coded_target(rx, b)
            tot += float(F.binary_cross_entropy_with_logits(p.bit_logits, t, reduction="sum"))
            n += t.numel()
    return tot / n


def representation(rx):
    """Pooled backbone features on a fixed matched set."""
    feats = []
    with torch.no_grad():
        for b in FEAT_B:
            f = rx.posterior._backbone(b.received_grid).mean(dim=(2, 3))
            feats.append(f.double().cpu())
    X = torch.cat(feats, 0)
    Xc = X - X.mean(0, keepdim=True)
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    ev = torch.linalg.eigvalsh(C).clamp_min(0)
    effrank = float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30))
    d = C.shape[0]
    reg = C + 1e-3 * torch.eye(d, dtype=C.dtype) * float(torch.diagonal(C).mean())
    logdet = float(torch.linalg.slogdet(reg).logabsdet)
    featnorm = float(X.norm(dim=1).mean())
    return effrank, logdet, featnorm


def sharpness(rx, base):
    """Mean increase of the matched loss under relative Gaussian weight noise."""
    clean = {k: v.detach().clone() for k, v in rx.state_dict().items()}
    out = []
    g = torch.Generator(device="cpu").manual_seed(99003)
    for _ in range(N_PERTURB):
        sd = {}
        for k, v in clean.items():
            if v.dtype.is_floating_point and v.dim() > 0:
                scale = REL_SIGMA * v.detach().abs().mean()
                noise = torch.randn(v.shape, generator=g).to(v.device, v.dtype) * scale
                sd[k] = v + noise
            else:
                sd[k] = v
        rx.load_state_dict(sd)
        out.append(matched_bce(rx))
    rx.load_state_dict(clean)
    return float(np.mean(out) - base)


done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(int(r["seed"]), int(r["step"])) for r in csv.DictReader(f)}
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for seed in SEEDS:
        for step in STEPS:
            if (seed, step) in done:
                continue
            p = os.path.join(ROOT, "results", "dense", f"seed{seed}", "checkpoints", f"ainr_step{step}.pt")
            if not os.path.exists(p):
                continue
            rx = AINR(cfg, ch_m).to(DEV)
            rx.load_state_dict(torch.load(p, map_location=DEV))
            rx.eval()
            base = matched_bce(rx)
            er, ld, fn = representation(rx)
            sh = sharpness(rx, base)
            w.writerow({"seed": seed, "step": step, "effrank": f"{er:.4f}", "logdet": f"{ld:.4f}",
                        "featnorm": f"{fn:.4f}", "sharpness": f"{sh:.6f}", "bce": f"{base:.6f}"})
            f.flush()
            print(f"seed{seed} {step:6d} effrank={er:7.3f} logdet={ld:9.3f} "
                  f"featnorm={fn:8.3f} sharp={sh:.5f}", flush=True)
            del rx
            torch.cuda.empty_cache()
print("DONE")
