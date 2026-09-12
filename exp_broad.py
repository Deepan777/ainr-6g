"""Evaluation of the broadly trained DiscNRX (train_broad.py) against the
narrowly trained receivers (seed 0) on identical slots.

  1. shift : 15 conditions x SNR {5, 10} dB; validation (4 batches) and test (30
             batches) streams; per slot the block error of every receiver, the
             broad receiver's CRC verdict, and the soft-output confidence
             statistics conf_b (broad) and conf_d (narrow DiscNRX).
  2. matched: S1 BLER of the three learned receivers on an SNR grid (-1..2.5 dB step 0.1,
             2.75..5 dB step 0.25), 40 batches (1280 blocks) per point.
  3. s4    : ray-traced S4 evaluation set, same SNRs and passes as exp_s4.py.

Output: results/ojcoms/broad_shift_slots.csv, broad_matched.csv, broad_s4_bler.csv
Usage : python exp_broad.py [--quick]
"""
import argparse
import os
import time

import numpy as np
import torch

from exp_common import (ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel, load_neural,
                        load_classical, decode, atomic_write_csv, read_csv_rows)
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.metrics import hard_bits

ap = argparse.ArgumentParser()
ap.add_argument("--snrs", type=float, nargs="+", default=[5.0, 10.0])
ap.add_argument("--test_batches", type=int, default=30)
ap.add_argument("--val_batches", type=int, default=4)
ap.add_argument("--ckpt", default=os.path.join(ROOT, "results", "broad", "checkpoints", "discnrx_final.pt"))
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.test_batches, args.val_batches = 1, 1

B = 32
cfg = load_cfg(16)
OUT = os.path.join(ROOT, "results", "ojcoms")
PARTS = os.path.join(OUT, "broad_parts")
os.makedirs(PARTS, exist_ok=True)
sfx = "_quick" if args.quick else ""

ch0 = make_channel(cfg)
ainr, disc = load_neural(cfg, ch0, 0)
broad = DiscriminativeNRX(cfg, pilot_grid=ch0.pilot_grid).to(DEV)
broad.load_state_dict(torch.load(args.ckpt, map_location=DEV))
broad.eval(); broad._ensure_ldpc(DEV)
classical = load_classical(cfg, ch0)
classical = {k: classical[k] for k in ("LS-LIN", "LS-LMMSE")}
RX = {"Broad": broad, "DiscNRX": disc, "AINR": ainr, **classical}
t0 = time.time()

# ---------------------------------------------------------------- 1. shift
rows = []
for ci, (name, cdl, ds, v) in enumerate(CONDITIONS):
    part = os.path.join(PARTS, f"{name}.csv")
    if os.path.exists(part) and not args.quick:
        rows += read_csv_rows(part, text_cols=("cond", "cdl", "stream"))
        continue
    n0 = len(rows)
    ch = make_channel(cfg, cdl, ds, v)
    for snr in args.snrs:
        for stream, nb, off in (("val", args.val_batches, 30_000), ("test", args.test_batches, 40_000)):
            torch.manual_seed(off + 1000 * ci + int(10 * snr))
            for bi in range(nb):
                bt = ch.generate_batch(B, snr_db=snr)
                with torch.no_grad():
                    conf_b = torch.tanh(broad.coded_llrs(bt.received_grid).abs() / 2).mean(1)
                    conf_d = torch.tanh(disc.coded_llrs(bt.received_grid).abs() / 2).mean(1)
                    hb = {k: decode(rx, bt, ch) for k, rx in RX.items()}
                    cols = {f"err_{k}": (h != bt.bits).any(1).float() for k, h in hb.items()}
                    cols.update(conf_b=conf_b, conf_d=conf_d, crc_b=ch.crc_check(hb["Broad"]).float())
                cols = {k: c.detach().cpu().numpy() for k, c in cols.items()}
                for i in range(B):
                    r = {"cond": name, "cdl": cdl, "ds_ns": ds, "speed_kmh": v, "snr_db": snr,
                         "stream": stream, "batch": bi, "slot": i}
                    r.update({k: float(c[i]) for k, c in cols.items()})
                    rows.append(r)
    test = [r for r in rows[n0:] if r["stream"] == "test" and r["snr_db"] == args.snrs[0]]
    print(f"[broad] {name:7s} BLER@{args.snrs[0]:.0f}dB " + "  ".join(
        f"{k}={np.mean([r['err_' + k] for r in test]):.3f}" for k in RX) + f"  ({time.time() - t0:.0f}s)",
        flush=True)
    if not args.quick:
        atomic_write_csv(part, rows[n0:])
atomic_write_csv(os.path.join(OUT, f"broad_shift_slots{sfx}.csv"), rows)

# ---------------------------------------------------------------- 2. matched grid
mpath = os.path.join(OUT, f"broad_matched{sfx}.csv")
if args.quick or not os.path.exists(mpath):
    ch = make_channel(cfg, *MATCHED[1:])
    NRX = {k: RX[k] for k in ("Broad", "DiscNRX", "AINR")}     # classical receivers: see exp_matched.py
    grid = sorted({round(float(x), 2) for x in [*np.arange(-1.0, 2.501, 0.1), *np.arange(2.75, 5.001, 0.25)]})
    mrows = []
    for i, snr in enumerate(grid if not args.quick else grid[::15]):
        torch.manual_seed(50_000 + 100 * i)
        err = {k: 0 for k in NRX}; n = 0
        for _ in range(40 if not args.quick else 1):                 # 1280 blocks per point
            bt = ch.generate_batch(B, snr_db=float(snr))
            for k, rx in NRX.items():
                err[k] += int((decode(rx, bt, ch) != bt.bits).any(1).sum())
            n += B
        mrows += [{"receiver": k, "snr_db": float(snr), "block_errors": err[k], "n_blocks": n,
                   "bler": err[k] / n} for k in NRX]
        print(f"[broad matched] {snr:5.2f} dB " + " ".join(f"{k}={err[k] / n:.4f}" for k in NRX), flush=True)
    atomic_write_csv(mpath, mrows)

# ---------------------------------------------------------------- 3. ray-traced S4
spath = os.path.join(OUT, f"broad_s4_bler{sfx}.csv")
if args.quick or not os.path.exists(spath):
    Hs = torch.tensor(np.load(os.path.join(ROOT, "results", "s4_channels.npy")), dtype=torch.complex64, device=DEV)
    EVAL_IDX = list(range(1, Hs.shape[0], 2))                # same split as exp_s4.py
    N_sym, N_sc, N_rx = ch0.n_symbols, ch0.n_subcarriers, ch0.n_rx
    srows = []
    for i, snr in enumerate([0, 5, 10, 15, 20, 25] if not args.quick else [25]):
        torch.manual_seed(80_000 + 100 * i)
        no = float(ch0.snr_db_to_no(torch.tensor(float(snr))).item())
        err = {"Broad": 0, "DiscNRX": 0}; n = 0
        for _ in range(5 if not args.quick else 1):
            for s in range(0, len(EVAL_IDX), 25):
                idx = EVAL_IDX[s:s + 25]
                H = Hs[idx][:, None, :, None, None, None, :].expand(len(idx), 1, N_rx, 1, 1, N_sym, N_sc)
                bits = ch0.binary_source([len(idx), 1, ch0.n_tx, ch0.k]).reshape(len(idx), ch0.num_info_bits)
                y = ch0.apply_channel(ch0.transmit(bits), H.contiguous(),
                                      torch.full((1, 1, 1, 1, 1), no, device=DEV))
                Y = ch0._received_to_claude_layout(y)
                with torch.no_grad():
                    for k in err:
                        err[k] += int((hard_bits(RX[k](Y)) != bits).any(1).sum())
                n += len(idx)
        srows += [{"receiver": k, "snr_db": snr, "block_errors": err[k], "n_blocks": n, "bler": err[k] / n}
                  for k in err]
        print(f"[broad S4] {snr:3d} dB " + " ".join(f"{k}={err[k] / n:.3f}" for k in err), flush=True)
    atomic_write_csv(spath, srows)
print(f"BROAD_DONE ({time.time() - t0:.0f}s)", flush=True)
