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


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--variants", nargs="+",
                    default=["full", "no_pilot", "pure_vfe", "no_detach", "no_freeze"])
args = parser.parse_args()

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = args.seed
STEPS = 30 if args.quick else 12000
WARMUP = 5 if args.quick else 200
B = 32
EVAL_SNRS = [0.0, 1.0, 3.0, 5.0]
EVAL_BATCHES = 2 if args.quick else 40          # 40*32 = 1280 blocks / point
AUC_SLOTS = 5 if args.quick else 200            # per condition
VARIANTS = args.variants

# Seed 1 reuses the original checkpoints; other seeds get their own directory.
ABL_DIR = os.path.join(_ROOT, "results", "ablation") if SEED == 1 else \
    os.path.join(_ROOT, "results", "ablation", f"seed{SEED}")
os.makedirs(ABL_DIR, exist_ok=True)
OUT_DIR = os.path.join(_ROOT, "results", "ojcoms")   # never overwrite the original CSVs
os.makedirs(OUT_DIR, exist_ok=True)


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
    # Resumable: full training state every 1000 steps (atomic write).
    state = ckpt + ".state"
    start = 0
    if os.path.exists(state) and not args.quick:
        st = torch.load(state, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optim"])
        sch.load_state_dict(st["sched"]); start = int(st["step"])
        torch.manual_seed(SEED + start)      # fresh data after a resume
        print(f"[{variant}] resuming at step {start}", flush=True)
    model.train()
    t0 = time.time()
    for step in range(start + 1, STEPS + 1):
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
        if step % 1000 == 0 and not args.quick:
            atomic_torch_save({"model": model.state_dict(), "optim": opt.state_dict(),
                               "sched": sch.state_dict(), "step": step}, state)
    if not args.quick:                # never let a smoke test masquerade as a checkpoint
        atomic_torch_save(model.state_dict(), ckpt)
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


from exp_common import CONDITIONS, make_channel  # noqa: E402
COND_BY_NAME = {c[0]: c for c in CONDITIONS}
SHIFT_EVAL = ["S1", "DS600", "DS1000", "S2", "V250"]

bler_rows, summary_rows, shift_rows = [], [], []
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

    # BLER under distribution shift (5 dB). With detach_aux=True the bit path of
    # the "full" variant is trained by coded-bit CE alone; "no_detach" and
    # "pure_vfe" let the generative objective shape the backbone. Comparing them
    # here is the budget- and seed-matched test of whether the generative
    # objective itself changes robustness to shift.
    for cname in SHIFT_EVAL:
        _, cdl, ds, v = COND_BY_NAME[cname]
        ch_sh = make_channel(cfg, cdl, ds, v)
        torch.manual_seed(777)                       # identical draws across variants
        bl, tot = eval_bler(model, ch_sh, 5.0, EVAL_BATCHES)
        shift_rows.append({"variant": variant, "cond": cname, "snr_db": 5.0,
                           "bler": bl, "n_blocks": tot})
        print(f"[{variant}] shift {cname:6s} @5 dB BLER={bl:.4f} (n={tot})", flush=True)
        del ch_sh

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

for r in bler_rows + summary_rows + shift_rows:
    r["seed"] = SEED
sfx = f"_seed{SEED}" + ("_quick" if args.quick else "")
p_shift = os.path.join(OUT_DIR, f"ablation_shift{sfx}.csv")
with open(p_shift, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "variant", "cond", "snr_db", "bler", "n_blocks"])
    w.writeheader(); w.writerows(shift_rows)
print(f"-> {p_shift}")
p_bler = os.path.join(OUT_DIR, f"ablation_bler{sfx}.csv")
p_sum = os.path.join(OUT_DIR, f"ablation_summary{sfx}.csv")
with open(p_bler, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "variant", "snr_db", "bler", "n_blocks"])
    w.writeheader(); w.writerows(bler_rows)
with open(p_sum, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "variant", "params", "auc_s2",
                                      "ber_5db", "sigma_post_std"])
    w.writeheader(); w.writerows(summary_rows)
print(f"\n-> {p_bler}")
print(f"-> {p_sum}")
print("ABLATION_DONE", flush=True)
