"""Rigorous inference-latency benchmark.

For each receiver and batch size in {1, 32}: 10 warm-up calls, then 50 timed
calls (CUDA-synchronized); reports the median and IQR in ms per call and ms per
slot.  Also times the AINR generative monitor (reconstruction residual, S=4)
separately, since it runs beside — not inside — the decoding path.

Outputs: results/latency.csv     (receiver, batch, median_ms, iqr_ms, ms_per_slot)
         results/latency_env.txt (GPU / library versions)
Usage:   python exp_latency.py [--quick]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from src.ainr import AINR
from src.baselines.lmmse import LMMSEReceiver
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.scenarios import build_scenario_channel

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(_ROOT, "results", "checkpoints")
N_WARM = 2 if args.quick else 10
N_ITER = 5 if args.quick else 50

ch = build_scenario_channel(cfg, "s1_matched", dev)
ainr = AINR(cfg, ch).to(dev)
ainr.load_state_dict(torch.load(f"{CKPT}/ainr_final.pt", map_location=dev))
ainr.eval()
disc = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(dev)
disc.load_state_dict(torch.load(f"{CKPT}/discnrx_final.pt", map_location=dev))
disc.eval()
lmmse = LMMSEReceiver(cfg, ch).to(dev)


def timed(fn):
    with torch.no_grad():
        for _ in range(N_WARM):
            fn()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        ts = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            fn()
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 75) - np.percentile(ts, 25))


rows = []
for B in [1, 32]:
    Y = ch.generate_batch(B, snr_db=10.0).received_grid
    no = float(ch.snr_db_to_no(torch.tensor(10.0)).item())
    for name, fn in [
        ("AINR", lambda: ainr(Y, record_fe=False)),
        ("DiscNRX", lambda: disc(Y)),
        ("LMMSE", lambda: lmmse(Y, no=no)),
        ("AINR-monitor", lambda: ainr.reconstruction_error(Y, n_samples=4)),
    ]:
        med, iqr = timed(fn)
        rows.append({"receiver": name, "batch": B, "median_ms": med,
                     "iqr_ms": iqr, "ms_per_slot": med / B})
        print(f"B={B:2d} {name:14s} median={med:8.2f} ms  iqr={iqr:6.2f}  "
              f"({med/B:.2f} ms/slot)", flush=True)

with open(os.path.join(_ROOT, "results", "latency.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["receiver", "batch", "median_ms",
                                      "iqr_ms", "ms_per_slot"])
    w.writeheader(); w.writerows(rows)
with open(os.path.join(_ROOT, "results", "latency_env.txt"), "w") as f:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    f.write(f"GPU: {gpu}\ntorch: {torch.__version__}\n"
            f"params AINR: {ainr.n_parameters}\nparams DiscNRX: {disc.n_parameters}\n")
print("-> results/latency.csv")
print("LATENCY_DONE", flush=True)
