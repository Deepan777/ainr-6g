"""Experiment 1: does NR-style scrambling remove the all-zero CRC-passing attractor?

Ungated decision-directed adaptation drives a receiver onto the all-zero codeword.
The transport-block CRC of an all-zero payload is also all zero, so the receiver's
own CRC accepts every erroneous block. In NR the coded bits are scrambled after
CRC attachment and coding, so a receiver that saturates in the *transmitted* bit
domain no longer descrambles into a valid codeword.

Three arms, identical channel draws, same receiver (the seed that collapsed):
  none    no scrambling (reproduces the vulnerability)
  fixed   one scrambling sequence reused every slot (adversarial best case:
          the network could in principle learn it)
  perslot new sequence every slot (NR-like: c_init depends on RNTI/slot)

Reported per arm: block errors, CRC passes, and FALSE PASSES (CRC accepted an
erroneous block) - the quantity that makes code-aided adaptation unsafe.

Output: results/ojcoms/scramble_probe.csv
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from exp_common import ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel, load_neural, atomic_write_csv

ap = argparse.ArgumentParser()
ap.add_argument("--cond", default="S2")
ap.add_argument("--snr", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--slots", type=int, default=400)
ap.add_argument("--lr", type=float, default=1e-4)
args = ap.parse_args()

cfg = load_cfg(16)
cond = {c[0]: c for c in CONDITIONS}[args.cond]
ch_m = make_channel(cfg, *MATCHED[1:])
ch = make_channel(cfg, *cond[1:])
K, n, ntx = ch.k, ch.n, ch.n_tx
no = float(ch.snr_db_to_no(torch.tensor(float(args.snr))).item())
no_b = torch.full((1, 1, 1, 1, 1), no, device=DEV)

gfix = torch.Generator(device="cpu").manual_seed(20260914)
S_FIXED = torch.randint(0, 2, (n,), generator=gfix).float().to(DEV)


def seq_for(mode, t):
    if mode == "none":
        return None
    if mode == "fixed":
        return S_FIXED
    g = torch.Generator(device="cpu").manual_seed(90000 + t)      # NR-like: varies per slot
    return torch.randint(0, 2, (n,), generator=g).float().to(DEV)


rows = []
for mode in ("none", "fixed", "perslot"):
    _ainr, disc = load_neural(cfg, ch_m, args.seed)
    del _ainr
    disc._ensure_ldpc(DEV)
    opt = disc._ensure_optimizer(args.lr)
    torch.manual_seed(777)                     # identical channel/data draws per arm
    n_err = n_crc = n_false = 0
    for t in range(args.slots):
        info = ch.binary_source([1, 1, ntx, K])            # payload + CRC
        cw = ch.encoder(info)                              # (1,1,ntx,n)
        s = seq_for(mode, t)
        cw_tx = cw if s is None else torch.remainder(cw + s.view(1, 1, 1, n), 2.0)
        y = ch.apply_channel(ch.rg_mapper(ch.mapper(cw_tx)), ch.generate_channel(1), no_b)
        Y = ch._received_to_claude_layout(y)
        bits_true = info.reshape(1, K)

        llr_tx = disc.coded_llrs(Y)                        # logits of TRANSMITTED bits
        llr_orig = llr_tx if s is None else llr_tx * (1.0 - 2.0 * s).view(1, n)
        with torch.no_grad():
            hb = ch.decoder(llr_orig.detach().reshape(1, 1, ntx, n)).reshape(1, K)
        err = bool((hb != bits_true).any())
        crc = bool(ch.crc_check(hb)[0])
        n_err += err; n_crc += crc; n_false += (crc and err)

        tgt = disc._coded_targets(hb)                      # pseudo-label, original domain
        if s is not None:
            tgt = torch.remainder(tgt + s.view(1, n), 2.0)  # ... in the transmitted domain
        disc.train(); opt.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(llr_tx, tgt, reduction="none").sum(1).mean().backward()
        opt.step(); disc.eval()

        rows.append({"mode": mode, "slot": t, "err": int(err), "crc_pass": int(crc),
                     "false_pass": int(crc and err), "ones": float(hb.float().mean()),
                     "abs_llr": float(llr_tx.abs().mean())})
    last = [r for r in rows if r["mode"] == mode][-50:]
    print(f"{mode:8s} BLER={n_err/args.slots:.3f}  crc_pass={n_crc:4d}/{args.slots}"
          f"  FALSE_PASS={n_false:4d}  ones(last50)={np.mean([r['ones'] for r in last]):.3f}"
          f"  |llr|(last50)={np.mean([r['abs_llr'] for r in last]):.2f}", flush=True)
    del disc
    torch.cuda.empty_cache()

atomic_write_csv(os.path.join(ROOT, "results", "ojcoms", "scramble_probe.csv"), rows)
print("\nfalse_pass = the receiver's own CRC accepted a block that was in error")
