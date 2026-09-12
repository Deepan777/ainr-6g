"""Training budget versus robustness to distribution shift.

For each training seed with snapshots, both receivers are evaluated at 12k, 25k and
50k (final) training steps on the matched channel and on shifted channels, at 5 dB.
Every (receiver, step, condition) cell uses the same block stream across seeds and
checkpoints (fixed stream seed per condition), so differences between checkpoints
are not sampling noise of the channel draws.

Motivation: the 12k-step ablation reference (seed 1) decodes DS1000 and S2 with
BLER <= 0.003 at 5 dB, whereas the 50k-step seed-0 checkpoints reach 0.10-0.59.

Output: results/ojcoms/budget_shift_seed{s}.csv (parts in results/ojcoms/budget_parts/)
Usage : python exp_budget.py --seeds 2 3 4 [--quick]
"""
import argparse
import os

import torch

from exp_common import (ROOT, DEV, CONDITIONS, load_cfg, make_channel, ckpt_dir,
                        decode, atomic_write_csv, read_csv_rows)
from src.ainr import AINR
from src.baselines.discriminative_nrx import DiscriminativeNRX

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
ap.add_argument("--snr", type=float, default=5.0)
ap.add_argument("--batches", type=int, default=30)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.batches = 1

B = 32
EVAL = ["S1", "DS600", "DS1000", "S2", "V250"]
STEPS = [("12000", "_step12000.pt"), ("25000", "_step25000.pt"), ("50000", "_final.pt")]
cfg = load_cfg(16)
cond = {c[0]: c for c in CONDITIONS}
out_dir = os.path.join(ROOT, "results", "ojcoms")
part_dir = os.path.join(out_dir, "budget_parts")
os.makedirs(part_dir, exist_ok=True)

ch0 = make_channel(cfg)
chans = {c: make_channel(cfg, *cond[c][1:]) for c in EVAL}

for seed in args.seeds:
    rows = []
    d = ckpt_dir(seed)
    for rx_name, prefix in (("AINR", "ainr"), ("DiscNRX", "discnrx")):
        for step, suffix in STEPS:
            path = os.path.join(d, prefix + suffix)
            part = os.path.join(part_dir, f"seed{seed}_{rx_name}_{step}{'_quick' if args.quick else ''}.csv")
            if os.path.exists(part) and not args.quick:
                rows += read_csv_rows(part, text_cols=("receiver", "cond"))
                continue
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing checkpoint {path}")
            if rx_name == "AINR":
                rx = AINR(cfg, ch0).to(DEV)
            else:
                rx = DiscriminativeNRX(cfg, pilot_grid=ch0.pilot_grid).to(DEV)
            rx.load_state_dict(torch.load(path, map_location=DEV))
            rx.eval()
            if rx_name == "DiscNRX":
                rx._ensure_ldpc(DEV)
            piece = []
            for ci, c in enumerate(EVAL):
                torch.manual_seed(90_000 + 1000 * ci + int(10 * args.snr))  # same stream for all
                err = tot = 0
                for _ in range(args.batches):
                    bt = chans[c].generate_batch(B, snr_db=args.snr)
                    hb = decode(rx, bt, chans[c])
                    err += int((hb != bt.bits).any(1).sum()); tot += B
                piece.append({"seed": seed, "receiver": rx_name, "step": int(step), "cond": c,
                              "snr_db": args.snr, "block_errors": err, "n_blocks": tot,
                              "bler": err / tot})
                print(f"[seed {seed}] {rx_name:7s} step {step:>5s} {c:6s} BLER={err / tot:.4f} (n={tot})",
                      flush=True)
            if not args.quick:
                atomic_write_csv(part, piece)
            rows += piece
            del rx
            torch.cuda.empty_cache()
    atomic_write_csv(os.path.join(out_dir, f"budget_shift_seed{seed}{'_quick' if args.quick else ''}.csv"), rows)
print("BUDGET_DONE", flush=True)
