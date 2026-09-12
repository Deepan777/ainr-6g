"""Shift-severity sweep: BLER of every receiver and per-slot drift statistics.

For each training seed, condition (see exp_common.CONDITIONS) and SNR, two
independent slot streams are generated: a small *validation* stream (used
only to set detector thresholds) and a larger *test* stream. For every slot
we store the block-error indicator of each receiver, the receiver-side CRC
verdicts of the neural receivers, and all drift statistics:

  generative (intrinsic to the AINR):
    resid   reconstruction residual E(Y)                  (higher = drift)
    tdisp   temporal dispersion of the per-symbol residual (higher = drift)
  soft-output (posterior-based, Uzlaner et al. style):
    conf_a  mean |tanh(LLR/2)| of the AINR coded-bit posterior (lower = drift)
    conf_d  same for the discriminative receiver
  classical pilot statistics (receiver-agnostic):
    cf      lag-1 frequency correlation of LS pilot estimates (lower = drift)
    ct      correlation between the two pilot symbols        (lower = drift)
    ymag    mean |Y| over the slot (PHT-style first moment)

Output: results/ojcoms/shift_slots_seed{s}.csv (one row per slot).

Usage: python exp_shift_detect.py --seeds 0 [1 2 3 4] [--quick]
"""
import argparse
import csv
import os
import time

import torch

from exp_common import (ROOT, DEV, CONDITIONS, load_cfg, make_channel, load_neural,
                        load_classical, decode, ainr_monitor, pilot_statistics,
                        atomic_write_csv, read_csv_rows)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--snrs", type=float, nargs="+", default=[5.0, 10.0])
ap.add_argument("--test_batches", type=int, default=30)
ap.add_argument("--val_batches", type=int, default=4)
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.test_batches, args.val_batches = 1, 1

B = 32
cfg = load_cfg(16)
out_dir = os.path.join(ROOT, "results", "ojcoms")
os.makedirs(out_dir, exist_ok=True)
pilot_syms = list(cfg.phy.pilot_ofdm_symbol_indices)

for seed in args.seeds:
    t_seed = time.time()
    ch0 = make_channel(cfg)
    ainr, disc = load_neural(cfg, ch0, seed)
    classical = load_classical(cfg, ch0)
    rows = []
    part_dir = os.path.join(out_dir, "shift_parts")
    os.makedirs(part_dir, exist_ok=True)
    for ci, (name, cdl, ds, v) in enumerate(CONDITIONS):
        # Resumable: each finished condition is stored as its own part file.
        part = os.path.join(part_dir, f"seed{seed}_{name}.csv")
        if os.path.exists(part) and not args.quick:
            rows += read_csv_rows(part, text_cols=("cond", "cdl", "stream"))
            print(f"[seed {seed}] {name:7s} loaded from {part}", flush=True)
            continue
        n_before = len(rows)
        ch = make_channel(cfg, cdl, ds, v)
        for snr in args.snrs:
            for stream, nb, off in (("val", args.val_batches, 10_000),
                                    ("test", args.test_batches, 20_000)):
                torch.manual_seed(off + 1000 * ci + int(10 * snr) + 7 * seed)
                for bi in range(nb):
                    bt = ch.generate_batch(B, snr_db=snr)
                    Y = bt.received_grid
                    with torch.no_grad():
                        resid, tdisp, logits = ainr_monitor(ainr, Y)
                        conf_a = torch.tanh(logits.abs() / 2).mean(1)
                        dl = disc.coded_llrs(Y)
                        conf_d = torch.tanh(dl.abs() / 2).mean(1)
                        cf, ct = pilot_statistics(Y, ch.pilot_grid, pilot_syms)
                        ymag = Y.abs().mean(dim=(1, 2, 3))
                        hb = {"AINR": decode(ainr, bt, ch), "DiscNRX": decode(disc, bt, ch)}
                        for cn, crx in classical.items():
                            hb[cn] = decode(crx, bt, ch)
                        err = {k: (v_ != bt.bits).any(1) for k, v_ in hb.items()}
                        crc_a = ch.crc_check(hb["AINR"])
                        crc_d = ch.crc_check(hb["DiscNRX"])
                    cols = {
                        "resid": resid, "tdisp": tdisp, "conf_a": conf_a, "conf_d": conf_d,
                        "cf": cf, "ct": ct, "ymag": ymag,
                        "crc_a": crc_a.float(), "crc_d": crc_d.float(),
                    }
                    cols.update({f"err_{k}": v_.float() for k, v_ in err.items()})
                    cols = {k: v_.detach().cpu().numpy() for k, v_ in cols.items()}
                    for i in range(B):
                        r = {"seed": seed, "cond": name, "cdl": cdl, "ds_ns": ds,
                             "speed_kmh": v, "snr_db": snr, "stream": stream,
                             "batch": bi, "slot": i}
                        r.update({k: float(v_[i]) for k, v_ in cols.items()})
                        rows.append(r)
        done = [r for r in rows if r["cond"] == name and r["stream"] == "test"]
        msg = "  ".join(
            f"{k[4:]}={sum(r[k] for r in done if r['snr_db'] == args.snrs[0]) / max(1, sum(1 for r in done if r['snr_db'] == args.snrs[0])):.3f}"
            for k in ("err_AINR", "err_DiscNRX", "err_LS-LIN", "err_LS-LMMSE"))
        print(f"[seed {seed}] {name:7s} BLER@{args.snrs[0]:.0f}dB {msg}  ({time.time() - t_seed:.0f}s)",
              flush=True)
        if not args.quick:
            atomic_write_csv(part, rows[n_before:])
    path = os.path.join(out_dir, f"shift_slots_seed{seed}{'_quick' if args.quick else ''}.csv")
    atomic_write_csv(path, rows)
    print(f"-> {path} ({len(rows)} rows, {time.time() - t_seed:.0f}s)", flush=True)
    del ainr, disc, classical
    torch.cuda.empty_cache()
print("SHIFT_DONE", flush=True)
