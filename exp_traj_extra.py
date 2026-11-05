"""Per-checkpoint quantities that the trajectory analysis was missing.

(1) Validation over the FULL training distribution D_tr. The channel is built without a fixed
    speed, so the speed is drawn per batch from the training range and the SNR per slot from the
    training range, exactly as during training. Reported: mean coded-bit BCE (the training loss)
    and BLER over the same fixed validation draws. This gives t_val = argmin BCE, the checkpoint a
    conventional validation-based selection would return, as distinct from the matched-condition
    selection used previously.

(2) Robustness under an orthogonal physical shift: high mobility at 250 and 400 km/h, 10 dB.

Streams use seeds disjoint from every earlier evaluation (9xxx) and are identical for all
checkpoints. Output: results/ojcoms/traj_extra.csv (resumable).
"""
import csv
import os

import torch
import torch.nn.functional as F

from exp_common import ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel
from exp_budget_curve import bler
from src.ainr import AINR
from src.channel.crc_channel import CRCSionnaChannel

SEEDS = [2, 3, 4, 5, 6, 7, 8]
STEPS = list(range(1000, 10001, 1000)) + list(range(12500, 50001, 2500))
OUT = os.path.join(ROOT, "results", "ojcoms", "traj_extra.csv")
FIELDS = ["seed", "step", "val_bce", "val_bler", "bler_V250", "bler_V400"]
N_VAL_BATCHES, BATCH = 40, 32

cfg = load_cfg(16)
COND = {c[0]: c for c in CONDITIONS}
ch_m = make_channel(cfg, *MATCHED[1:])                  # matched, for model construction
ch_tr = CRCSionnaChannel(cfg, device=DEV)               # full training distribution D_tr
ch_v250 = make_channel(cfg, *COND["V250"][1:])
ch_v400 = make_channel(cfg, *COND["V400"][1:])


def load(seed, step):
    p = os.path.join(ROOT, "results", "dense", f"seed{seed}", "checkpoints", f"ainr_step{step}.pt")
    if not os.path.exists(p):
        return None
    m = AINR(cfg, ch_m).to(DEV)
    m.load_state_dict(torch.load(p, map_location=DEV))
    m.eval()
    return m


def validation(rx):
    """Mean coded-bit BCE and BLER on a fixed draw from D_tr."""
    torch.manual_seed(9101)
    bce_sum, n_bits, err, n_blk = 0.0, 0, 0, 0
    for _ in range(N_VAL_BATCHES):
        b = ch_tr.generate_batch(BATCH)
        with torch.no_grad():
            params = rx.posterior.forward(b.received_grid)
            # same coded-bit target as the training loss (AINR.hybrid_objective)
            ib = b.bits.to(device=b.received_grid.device, dtype=torch.float32)
            target = ch_tr.encoder(ib.reshape(BATCH, 1, rx.n_tx, rx.k)).reshape(BATCH, rx.posterior.n_coded)
            bce = F.binary_cross_entropy_with_logits(params.bit_logits, target, reduction="sum")
            bce_sum += float(bce)
            n_bits += target.numel()
            hb = ch_tr.decoder(params.bit_logits.detach().reshape(BATCH, 1, rx.n_tx, -1)).reshape(BATCH, -1)
            err += int((hb != b.bits).any(1).sum())
            n_blk += BATCH
    return bce_sum / n_bits, err / n_blk


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
            rx = load(seed, step)
            if rx is None:
                continue
            bce, vbler = validation(rx)
            row = {"seed": seed, "step": step, "val_bce": f"{bce:.6f}", "val_bler": f"{vbler:.6f}",
                   "bler_V250": f"{bler(rx, ch_v250, 10.0, 15, seed=9202):.6f}",
                   "bler_V400": f"{bler(rx, ch_v400, 10.0, 15, seed=9303):.6f}"}
            w.writerow(row); f.flush()
            print(f"seed{seed} {step:6d} BCE={bce:.4f} val_BLER={vbler:.3f} "
                  f"V250={row['bler_V250']} V400={row['bler_V400']}", flush=True)
            del rx
            torch.cuda.empty_cache()
print("DONE")
