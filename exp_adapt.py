"""Online adaptation with a real receiver-side CRC gate.

A fixed stream of single-slot transmissions is pre-generated under a drifted
channel (identical for every receiver and mode). Each receiver starts from its
trained checkpoint and processes the stream slot by slot:

  none      no adaptation
  ungated   one gradient step on every slot, using its own decoded bits
  gated     one step only when the receiver's own CRC check passes
  trig      one step only when a drift flag is raised AND the CRC passes
            (asynchronous adaptation). AINR flags with its reconstruction
            residual; DiscNRX flags with its mean soft-output confidence
            (posterior-based detector). Thresholds are set on matched-channel
            validation slots at the same SNR (99th percentile / 1st percentile).

The CRC verdict used for gating is computed from the receiver's decoded bits
only. Ground truth is used solely to *report* block errors and false CRC passes.

Output: results/ojcoms/adapt_{cond}_snr{snr}_seed{s}.csv
Usage : python exp_adapt.py --cond S2 --snr 3 --seeds 0 1 2 [--slots 1000] [--quick]
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from exp_common import (ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel,
                        load_neural, ainr_monitor, atomic_write_csv, read_csv_rows)
from src.ainr import AINR
from src.evaluation.metrics import hard_bits

ap = argparse.ArgumentParser()
ap.add_argument("--cond", default="S2")
ap.add_argument("--snr", type=float, default=3.0)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--slots", type=int, default=1000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--modes", nargs="+", default=["none", "ungated", "gated", "trig"])
ap.add_argument("--stream_seed", type=int, default=777)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.slots = 12

cfg = load_cfg(16)
cond = {c[0]: c for c in CONDITIONS}[args.cond]
out_dir = os.path.join(ROOT, "results", "ojcoms")
os.makedirs(out_dir, exist_ok=True)


def pregenerate(ch, n, snr, seed):
    torch.manual_seed(seed)
    return [ch.generate_batch(1, snr_db=snr) for _ in range(n)]


@torch.no_grad()
def score(rx, Y):
    """Drift score, oriented so that larger = more drift."""
    if isinstance(rx, AINR):
        return float(ainr_monitor(rx, Y, n_mc=1)[0])
    return -float(torch.tanh(rx.coded_llrs(Y).abs() / 2).mean())


def sync():
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()


ch_m = make_channel(cfg, *MATCHED[1:])
ch_d = make_channel(cfg, *cond[1:])
stream = pregenerate(ch_d, args.slots, args.snr, args.stream_seed)
val = pregenerate(ch_m, 32 if args.quick else 256, args.snr, 4242)

part_dir = os.path.join(out_dir, "adapt_parts")
os.makedirs(part_dir, exist_ok=True)

for seed in args.seeds:
    rows = []
    tag = f"{args.cond}_snr{args.snr:g}_seed{seed}{'_quick' if args.quick else ''}"
    for rx_name in ("AINR", "DiscNRX"):
        parts = {m: os.path.join(part_dir, f"{tag}_{rx_name}_{m}.csv") for m in args.modes}
        todo = [m for m in args.modes if args.quick or not os.path.exists(parts[m])]
        for m in args.modes:
            if m not in todo:           # resumable: finished (receiver, mode) pieces are reused
                rows += read_csv_rows(parts[m], text_cols=("receiver", "mode"))
                print(f"[seed {seed}] {rx_name:7s} {m:7s} loaded from {parts[m]}", flush=True)
        if not todo:
            continue
        # Threshold for the triggered mode from matched validation slots.
        a0, d0 = load_neural(cfg, ch_m, seed)
        base = a0 if rx_name == "AINR" else d0
        vs = np.array([score(base, b.received_grid) for b in val])
        tau = float(np.quantile(vs, 0.99))
        del a0, d0
        for mode in todo:
            n_before = len(rows)
            ainr, disc = load_neural(cfg, ch_m, seed)
            rx = ainr if rx_name == "AINR" else disc
            del ainr, disc
            t0 = time.time()
            n_upd = n_false = 0
            for t, b in enumerate(stream):
                Y = b.received_grid
                with torch.no_grad():
                    llr = rx(Y, record_fe=False) if isinstance(rx, AINR) else rx(Y)
                hb = hard_bits(llr)
                err = bool((hb != b.bits).any())
                crc = bool(ch_d.crc_check(hb)[0])
                s = score(rx, Y) if mode == "trig" else float("nan")
                flag = (s > tau) if mode == "trig" else False
                do = (mode == "ungated") or (mode == "gated" and crc) or (mode == "trig" and crc and flag)
                upd_ms = float("nan")
                if do:
                    sync(); u0 = time.perf_counter()
                    rx.adapt_online(Y, hb, lr=args.lr)
                    sync(); upd_ms = 1e3 * (time.perf_counter() - u0)
                    rx.eval()
                    n_upd += 1
                    n_false += int(crc and err)
                rows.append({"seed": seed, "receiver": rx_name, "mode": mode, "slot": t,
                             "err": int(err), "crc_pass": int(crc), "false_pass": int(crc and err),
                             "updated": int(do), "update_ms": upd_ms, "score": s, "tau": tau})
            blers = np.array([r["err"] for r in rows if r["receiver"] == rx_name and r["mode"] == mode])
            k = max(1, len(blers) // 5)
            print(f"[seed {seed}] {rx_name:7s} {mode:7s} first-{k}={blers[:k].mean():.3f} "
                  f"last-{k}={blers[-k:].mean():.3f} updates={n_upd} false_pass_updates={n_false} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if not args.quick:
                atomic_write_csv(parts[mode], rows[n_before:])
            del rx
            torch.cuda.empty_cache()
    path = os.path.join(out_dir, f"adapt_{tag}.csv")
    atomic_write_csv(path, rows)
    print(f"-> {path}", flush=True)
print("ADAPT_DONE", flush=True)
