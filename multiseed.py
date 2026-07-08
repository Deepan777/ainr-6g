"""Multi-seed verification of the AINR-vs-DiscNRX comparison.

For each seed: train AINR (hybrid) and DiscNRX (CE) from scratch at an identical
reduced budget, then measure (a) matched-condition BLER parity and (b) the S2
delay-drift robustness gap. Prints per-seed numbers and the mean +/- std across
seeds, so we can tell whether AINR's drift robustness is a real effect or seed
variance.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _ROOT)
from src.channel.sionna_channel import SionnaChannel
from src.ainr import AINR
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.scenarios import build_scenario_channel
from src.evaluation.metrics import hard_bits, count_block_errors

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
SEEDS = [1, 2, 3]
STEPS = 12000
WARMUP = 200
B = 32


def bler(model, ch, snr, nb=20):
    model.eval(); te = tot = 0
    with torch.no_grad():
        for _ in range(nb):
            b = ch.generate_batch(B, snr_db=snr)
            e, t = count_block_errors(hard_bits(model(b.received_grid, record_fe=False)
                                                if isinstance(model, AINR) else model(b.received_grid)),
                                      b.bits)
            te += e; tot += t
    return te / tot


def train_seed(seed):
    torch.manual_seed(seed)
    ch = SionnaChannel(cfg, device=dev)
    ainr = AINR(cfg, ch).to(dev)
    disc = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(dev)
    oa = torch.optim.AdamW(ainr.posterior.parameters(), lr=1e-3)
    od = torch.optim.AdamW(disc.parameters(), lr=1e-3)
    sa = torch.optim.lr_scheduler.CosineAnnealingLR(oa, T_max=STEPS)
    sd = torch.optim.lr_scheduler.CosineAnnealingLR(od, T_max=STEPS)
    ns = int(cfg.model.n_vfe_samples)
    ainr.train(); disc.train()
    for step in range(1, STEPS + 1):
        b = ch.generate_batch(B)
        so = b.no.sqrt() if step <= WARMUP else None
        oa.zero_grad(set_to_none=True)
        out = ainr.hybrid_objective(b.received_grid, b.bits, n_samples=ns, sigma_override=so)
        out.total.backward(); nn.utils.clip_grad_norm_(ainr.posterior.parameters(), 1.0)
        oa.step(); sa.step()
        od.zero_grad(set_to_none=True)
        ld = disc.loss(b.received_grid, b.bits)
        ld.backward(); nn.utils.clip_grad_norm_(disc.parameters(), 1.0); od.step(); sd.step()
    return ch, ainr, disc


rows = {}
for seed in SEEDS:
    t0 = time.time()
    ch_m, ainr, disc = train_seed(seed)
    ch2 = build_scenario_channel(cfg, "s2_delay_shift", dev)
    r = {
        "matched@2": (bler(ainr, ch_m, 2.0), bler(disc, ch_m, 2.0)),
        "S2@3": (bler(ainr, ch2, 3.0), bler(disc, ch2, 3.0)),
        "S2@5": (bler(ainr, ch2, 5.0), bler(disc, ch2, 5.0)),
        "S2@10": (bler(ainr, ch2, 10.0), bler(disc, ch2, 10.0)),
    }
    rows[seed] = r
    print(f"[seed {seed}] ({time.time()-t0:.0f}s)  "
          + "  ".join(f"{k}: AINR={a:.3f}/Disc={d:.3f}" for k, (a, d) in r.items()),
          flush=True)
    del ainr, disc, ch_m, ch2; torch.cuda.empty_cache()

print("\n=== AGGREGATE (mean +/- std across seeds) ===")
for k in ["matched@2", "S2@3", "S2@5", "S2@10"]:
    A = np.array([rows[s][k][0] for s in SEEDS]); D = np.array([rows[s][k][1] for s in SEEDS])
    print(f"  {k:10s}: AINR={A.mean():.3f}+/-{A.std():.3f}   DiscNRX={D.mean():.3f}+/-{D.std():.3f}")
print("MULTISEED_DONE")
