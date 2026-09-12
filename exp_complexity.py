"""Complexity and latency of every receiver stage (run with the GPU otherwise idle).

Reported per stage and batch size (1, 32): median and interquartile range of 50
CUDA-synchronised calls after 10 warm-up calls.
  detection   full receiver call (network or classical estimator + LDPC BP, 20 it.)
  cnn         network forward pass only (coded-bit logits)
  ldpc        LDPC BP decoding of given LLRs
  monitor     reconstruction residual with S = 1 and S = 4 posterior samples
  crc         receiver-side CRC16 check of decoded blocks
  adapt       one online update step (batch 1)
Also: trainable parameters, multiply-accumulate operations (MACs) per slot counted
from Conv2d/Linear layer shapes, checkpoint sizes, peak GPU memory, and the
hardware/software environment.

Output: results/ojcoms/complexity.csv, results/ojcoms/complexity_env.json
"""
import json
import os
import platform
import subprocess
import time

import numpy as np
import psutil
import torch
import torch.nn as nn

from exp_common import (ROOT, DEV, load_cfg, make_channel, load_neural, load_classical, ainr_monitor,
                        ckpt_dir, env_record, atomic_write_csv)
from src.evaluation.metrics import hard_bits

import sys
WARM, REP = (2, 3) if "--quick" in sys.argv else (10, 50)
SFX = "_quick" if "--quick" in sys.argv else ""
cfg = load_cfg(16)
ch = make_channel(cfg)
ainr, disc = load_neural(cfg, ch, 0)
classical = load_classical(cfg, ch)
rows = []


def sync():
    if DEV.startswith("cuda"):
        torch.cuda.synchronize()


def timeit(fn):
    for _ in range(WARM):
        fn()
    sync()
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter(); fn(); sync(); ts.append(1e3 * (time.perf_counter() - t0))
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return med, q1, q3


def macs(module, *inputs):
    total = [0]

    def hook(m, inp, outp):
        if isinstance(m, nn.Conv2d):
            total[0] += outp.numel() // outp.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += outp.numel() // outp.shape[0] * m.in_features
    hs = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        module(*inputs)
    for h in hs:
        h.remove()
    return total[0]


def add(stage, receiver, bs, t, extra=None):
    r = {"stage": stage, "receiver": receiver, "batch": bs, "median_ms": t[0], "q1_ms": t[1], "q3_ms": t[2],
         "ms_per_slot": t[0] / bs}
    r.update(extra or {})
    rows.append(r)
    print(f"{stage:10s} {receiver:9s} B={bs:2d} {t[0]:8.2f} ms  ({t[0] / bs:6.3f} ms/slot)", flush=True)


for bs in (1, 32):
    torch.manual_seed(1234)
    bt = ch.generate_batch(bs, snr_db=5.0)
    Y, no = bt.received_grid, float(bt.no.reshape(-1)[0])
    with torch.no_grad():
        add("detection", "AINR", bs, timeit(lambda: ainr(Y, record_fe=False)))
        add("detection", "DiscNRX", bs, timeit(lambda: disc(Y)))
        for k, rx in classical.items():
            add("detection", k, bs, timeit(lambda rx=rx: rx(Y, no=no, h_freq=bt.h_freq)))
        add("cnn", "AINR", bs, timeit(lambda: ainr.posterior(Y)))
        add("cnn", "DiscNRX", bs, timeit(lambda: disc.coded_llrs(Y)))
        cl = disc.coded_llrs(Y)
        add("ldpc", "shared", bs, timeit(lambda: disc._ldpc["dec"](cl.reshape(bs, 1, disc.n_tx, disc.n))))
        add("monitor", "AINR S=1", bs, timeit(lambda: ainr_monitor(ainr, Y, n_mc=1)))
        add("monitor", "AINR S=4", bs, timeit(lambda: ainr_monitor(ainr, Y, n_mc=4)))
        hb = hard_bits(ainr(Y, record_fe=False))
        add("crc", "shared", bs, timeit(lambda: ch.crc_check(hb)))
    if bs == 1:
        for name, rx in (("AINR", ainr), ("DiscNRX", disc)):
            def step(rx=rx):
                rx.adapt_online(Y, hb, lr=1e-4)
            add("adapt", name, 1, timeit(step))
            rx.eval()

# peak memory (batch 32, detection) and static properties
ainr, disc = load_neural(cfg, ch, 0)             # fresh weights after the adaptation timing
bt = ch.generate_batch(32, snr_db=5.0)
for name, fn in (("AINR", lambda: ainr(bt.received_grid, record_fe=False)), ("DiscNRX", lambda: disc(bt.received_grid)),
                 ("AINR monitor S=4", lambda: ainr_monitor(ainr, bt.received_grid, n_mc=4))):
    torch.cuda.reset_peak_memory_stats(); sync()
    with torch.no_grad():
        fn()
    sync()
    rows.append({"stage": "peak_memory", "receiver": name, "batch": 32,
                 "peak_mib": torch.cuda.max_memory_allocated() / 2 ** 20})
y1 = ch.generate_batch(1, snr_db=5.0).received_grid
p_all = sum(p.numel() for p in ainr.posterior.parameters())
p_det = sum(p.numel() for n, p in ainr.posterior.named_parameters()
            if not n.startswith(("h_mean", "h_logvar", "sigma")))
rows += [
    {"stage": "static", "receiver": "AINR", "params": p_all, "params_detection_path": p_det,
     "macs_per_slot": macs(ainr.posterior, y1),
     "checkpoint_bytes": os.path.getsize(os.path.join(ckpt_dir(0), "ainr_final.pt"))},
    {"stage": "static", "receiver": "DiscNRX", "params": sum(p.numel() for p in disc.parameters()),
     "params_detection_path": sum(p.numel() for p in disc.parameters()),
     "macs_per_slot": None,          # filled below (coded_llrs is not the module's forward)
     "checkpoint_bytes": os.path.getsize(os.path.join(ckpt_dir(0), "discnrx_final.pt"))},
]
# DiscNRX MACs: hook the whole module while calling coded_llrs
tot = [0]
def _h(m, i, o):
    if isinstance(m, nn.Conv2d):
        tot[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
hs = [m.register_forward_hook(_h) for m in disc.modules() if isinstance(m, nn.Conv2d)]
with torch.no_grad():
    disc.coded_llrs(y1)
for h in hs:
    h.remove()
rows[-1]["macs_per_slot"] = tot[0]

env = env_record()
env.update({"cpu": platform.processor(), "cpu_cores_logical": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1)})
try:
    env["gpu_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,temperature.gpu,clocks.sm",
                                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except OSError:
    pass
out = os.path.join(ROOT, "results", "ojcoms")
atomic_write_csv(os.path.join(out, f"complexity{SFX}.csv"), rows,
                 fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in ("stage", "receiver", "batch"), k)))
json.dump(env, open(os.path.join(out, f"complexity_env{SFX}.json"), "w"), indent=1)
print("COMPLEXITY_DONE", flush=True)
