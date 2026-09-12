"""BLER versus SNR on a dense grid for every receiver (matched channel by default).

Every receiver and every training seed sees the same block streams (one fixed stream
seed per SNR point), so seed-to-seed differences are due to training alone. Classical
receivers do not depend on the training seed; they are evaluated once per stream.

Output: results/ojcoms/matched_qam{Q}_{cond}_seed{s}.csv (neural receivers)
        results/ojcoms/matched_qam{Q}_{cond}_classical.csv
Usage : python exp_matched.py --qam 16 --seeds 0 [--cond S1] [--quick]
"""
import argparse
import os

import numpy as np
import torch

from exp_common import (ROOT, CONDITIONS, load_cfg, make_channel, load_neural, load_classical,
                        decode, atomic_write_csv, read_csv_rows)

ap = argparse.ArgumentParser()
ap.add_argument("--qam", type=int, default=16, choices=[16, 64])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--cond", default="S1")
ap.add_argument("--snrs", type=float, nargs="+", default=None)
ap.add_argument("--batches", type=int, default=40)          # 40 x 32 = 1280 blocks per point
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.snrs is None:
    if args.qam == 16:
        # 0.1-dB steps across the LDPC waterfalls (learned/LMMSE/PCSI: 0-2 dB; LS-LIN: 4-6 dB),
        # coarse elsewhere. A first run on a uniform 0.5-dB grid (-2..8 dB) placed only two
        # points on each learned receiver's waterfall; it is kept as *_coarse.csv.
        args.snrs = sorted({round(float(x), 2) for x in
                            [-1.0, -0.5, *np.arange(0.0, 2.001, 0.1), 2.5, 3.0, 3.5, *np.arange(4.0, 6.001, 0.1)]})
    else:
        # Same design for 64-QAM (learned/LMMSE/PCSI waterfalls 4.5-6 dB, LS-LIN 7.5-9 dB); the
        # first run on 4..14 dB in 0.5-dB steps is kept as *_coarse.csv.
        args.snrs = sorted({round(float(x), 2) for x in
                            [3.0, 3.5, *np.arange(4.0, 6.501, 0.1), 7.0, 7.5, *np.arange(7.6, 9.501, 0.1), 10.0]})
if args.quick:
    args.batches, args.snrs = 1, args.snrs[::6]

B = 32
cfg = load_cfg(args.qam)
cond = {c[0]: c for c in CONDITIONS}[args.cond]
out = os.path.join(ROOT, "results", "ojcoms")
sfx = "_quick" if args.quick else ""
ch = make_channel(cfg, *cond[1:])


def stream_seed(i):
    return 70_000 + 1000 * args.qam + 100 * i + (0 if args.cond == "S1" else 7 * len(args.cond))


def run(receivers, tag_seed):
    rows = []
    for i, snr in enumerate(args.snrs):
        torch.manual_seed(stream_seed(i))
        err = {k: 0 for k in receivers}; tot = 0
        for _ in range(args.batches):
            bt = ch.generate_batch(B, snr_db=float(snr))
            with torch.no_grad():
                for k, rx in receivers.items():
                    err[k] += int((decode(rx, bt, ch) != bt.bits).any(1).sum())
            tot += B
        for k in receivers:
            rows.append({"seed": tag_seed, "receiver": k, "cond": args.cond, "qam": args.qam,
                         "snr_db": float(snr), "block_errors": err[k], "n_blocks": tot, "bler": err[k] / tot})
        print(f"[{tag_seed}] SNR={snr:5.1f} " + " ".join(f"{k}={err[k] / tot:.4f}" for k in receivers), flush=True)
    return rows


cpath = os.path.join(out, f"matched_qam{args.qam}_{args.cond}_classical{sfx}.csv")
if args.quick or not os.path.exists(cpath):
    atomic_write_csv(cpath, run(load_classical(cfg, ch, args.qam), -1))
for s in args.seeds:
    p = os.path.join(out, f"matched_qam{args.qam}_{args.cond}_seed{s}{sfx}.csv")
    if os.path.exists(p) and not args.quick:
        print(f"seed {s}: exists, skipped"); continue
    ainr, disc = load_neural(cfg, ch, s, args.qam)
    atomic_write_csv(p, run({"AINR": ainr, "DiscNRX": disc}, s))
    del ainr, disc
    torch.cuda.empty_cache()
print("MATCHED_DONE", flush=True)
