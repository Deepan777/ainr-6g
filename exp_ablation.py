"""Ablation study: budget-matched training of AINR design variants.

Each variant is trained from scratch for an identical reduced budget
(ABLATION_STEPS steps, same seed, same data stream) and then evaluated on
(a) matched CDL-C BLER in the waterfall region and (b) the S2 delay-drift
reconstruction-residual AUC.  Variants:

  full        — the proposed design (hybrid objective, pilot reference,
                detached aux heads, noise-freeze warm-up)
  no_pilot    — raw grid only: no pilot-referenced LS input to the encoder
  pure_vfe    — unsupervised VFE only (no supervised coded-bit CE term)
  no_detach   — VFE reconstruction gradient also flows into the shared backbone
  no_freeze   — no noise-freeze warm-up (sigma^2 free from step 0)

Resume-safe: each trained variant is checkpointed under
results/ablation/<variant>.pt and skipped on re-run.  Outputs:
  results/ablation_bler.csv   (variant, snr_db, bler, n_blocks)
  results/ablation_summary.csv (variant, params, auc_s2, ber_5db, sigma_post)

Usage:  python exp_ablation.py            # full run
        python exp_ablation.py --quick    # smoke test (tiny budgets)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from src.channel.sionna_channel import SionnaChannel
from src.ainr import AINR
from src.variational_posterior import VariationalPosterior
from src.evaluation.scenarios import build_scenario_channel
from src.evaluation.metrics import hard_bits, count_block_errors, compute_ber

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 1
STEPS = 30 if args.quick else 12000
WARMUP = 5 if args.quick else 200
B = 32
EVAL_SNRS = [0.0, 1.0, 3.0, 5.0]
EVAL_BATCHES = 2 if args.quick else 40          # 40*32 = 1280 blocks / point
AUC_SLOTS = 5 if args.quick else 200            # per condition
VARIANTS = ["full", "no_pilot", "pure_vfe", "no_detach", "no_freeze"]

ABL_DIR = os.path.join(_ROOT, "results", "ablation")
os.makedirs(ABL_DIR, exist_ok=True)


def build_model(variant, ch):
    m = AINR(cfg, ch).to(dev)
    if variant == "no_pilot":
        m.posterior = VariationalPosterior(cfg, pilot_grid=None).to(dev)
    return m


def train_variant(variant):
    """Train one variant for STEPS steps (identical seed/budget) or load it."""
    ckpt = os.path.join(ABL_DIR, f"{variant}.pt")
    torch.manual_seed(SEED)
    ch = SionnaChannel(cfg, device=dev)
    model = build_model(variant, ch)
    if os.path.exists(ckpt) and not args.quick:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"[{variant}] loaded existing {ckpt}", flush=True)
        return ch, model
    opt = torch.optim.AdamW(model.posterior.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    ns = int(cfg.model.n_vfe_samples)
    model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        batch = ch.generate_batch(B)
        freeze = (variant != "no_freeze") and step <= WARMUP
        so = batch.no.sqrt() if freeze else None
        opt.zero_grad(set_to_none=True)
        if variant == "pure_vfe":
            out = model.compute_free_energy(batch.received_grid, n_samples=ns,
                                            sigma_override=so)
            loss = out.total
        else:
            out = model.hybrid_objective(batch.received_grid, batch.bits,
                                         n_samples=ns, sigma_override=so,
                                         detach_aux=(variant != "no_detach"))
            loss = out.total
        loss.backward()
        nn.utils.clip_grad_norm_(model.posterior.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 2000 == 0 or step == STEPS:
            print(f"[{variant}] step {step}/{STEPS} loss={loss.item():.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), ckpt)
    return ch, model


@torch.no_grad()
def eval_bler(model, ch, snr, n_batches):
    model.eval(); te = tot = 0
    for _ in range(n_batches):
        b = ch.generate_batch(B, snr_db=snr)
        llr = model(b.received_grid, record_fe=False)
        e, t = count_block_errors(hard_bits(llr), b.bits)
        te += e; tot += t
    return te / tot, tot


@torch.no_grad()
def eval_ber(model, ch, snr, n_batches=8):
    """Raw info-bit BER (diagnostic: 0.5 == chance)."""
    model.eval(); vals = []
    for _ in range(n_batches):
        b = ch.generate_batch(B, snr_db=snr)
        llr = model(b.received_grid, record_fe=False)
        vals.append(compute_ber(hard_bits(llr), b.bits))
    return float(np.mean(vals))


@torch.no_grad()
def drift_auc(model, ch_matched, ch_drift, n_slots, snr=10.0):
    model.eval()
    scores, labels = [], []
    for lab, ch in [(0, ch_matched), (1, ch_drift)]:
        for _ in range(n_slots):
            b = ch.generate_batch(1, snr_db=snr)
            scores.append(model.reconstruction_error(b.received_grid))
            labels.append(lab)
    scores, labels = np.array(scores), np.array(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    # AUC = P(drift score > matched score) — Mann-Whitney formulation.
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


bler_rows, summary_rows = [], []
for variant in VARIANTS:
    print(f"\n===== VARIANT: {variant} =====", flush=True)
    ch, model = train_variant(variant)
    ch_s2 = build_scenario_channel(cfg, "s2_delay_shift", dev)

    torch.manual_seed(12345)          # identical eval draws across variants
    for snr in EVAL_SNRS:
        bler, tot = eval_bler(model, ch, snr, EVAL_BATCHES)
        bler_rows.append({"variant": variant, "snr_db": snr,
                          "bler": bler, "n_blocks": tot})
        print(f"[{variant}] SNR={snr:4.1f} dB  BLER={bler:.4f} (n={tot})", flush=True)

    torch.manual_seed(54321)
    ber5 = eval_ber(model, ch, 5.0)
    auc = drift_auc(model, ch, ch_s2, AUC_SLOTS)
    sigma_post = float(torch.exp(0.5 * model.posterior.sigma_post_logstd
                                 .detach()).item())
    n_par = int(model.n_parameters)
    summary_rows.append({"variant": variant, "params": n_par, "auc_s2": auc,
                         "ber_5db": ber5, "sigma_post_std": sigma_post})
    print(f"[{variant}] AUC(S2)={auc:.3f}  BER@5dB={ber5:.4f}  params={n_par}",
          flush=True)
    del model, ch, ch_s2
    torch.cuda.empty_cache()

with open(os.path.join(_ROOT, "results", "ablation_bler.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant", "snr_db", "bler", "n_blocks"])
    w.writeheader(); w.writerows(bler_rows)
with open(os.path.join(_ROOT, "results", "ablation_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["variant", "params", "auc_s2",
                                      "ber_5db", "sigma_post_std"])
    w.writeheader(); w.writerows(summary_rows)
print("\n-> results/ablation_bler.csv")
print("-> results/ablation_summary.csv")
print("ABLATION_DONE", flush=True)
