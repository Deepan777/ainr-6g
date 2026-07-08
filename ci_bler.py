"""Confidence intervals on matched BLER: K independent eval passes -> mean +/- std.

Produces results/stage_b_ci.csv and results/plots/bler_stage_b_ci.pdf with error
bars (1 std over K passes). Focused on the waterfall region where variance matters.
"""
import os, sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _ROOT)
from src.ainr import AINR
from src.baselines.lmmse import LMMSEReceiver
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.scenarios import build_scenario_channel
from src.evaluation.metrics import hard_bits, count_block_errors

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt = os.path.join(_ROOT, "results", "checkpoints")

ch = build_scenario_channel(cfg, "s1_matched", dev)
ainr = AINR(cfg, ch).to(dev); ainr.load_state_dict(torch.load(f"{ckpt}/ainr_final.pt", map_location=dev)); ainr.eval()
disc = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(dev); disc.load_state_dict(torch.load(f"{ckpt}/discnrx_final.pt", map_location=dev)); disc.eval()
lmmse = LMMSEReceiver(cfg, ch).to(dev)
RX = {"AINR": ainr, "DiscNRX": disc, "LMMSE": lmmse}

SNRS = [-2, 0, 1, 2, 3, 4, 5, 6]
K = 8            # independent passes
NB = 40          # batches per pass (40*32 = 1280 blocks)

def one_pass(model, snr, no):
    te = tot = 0
    with torch.no_grad():
        for _ in range(NB):
            b = ch.generate_batch(32, snr_db=snr)
            llr = model(b.received_grid, record_fe=False) if isinstance(model, AINR) else \
                  (model(b.received_grid, no=no) if isinstance(model, LMMSEReceiver) else model(b.received_grid))
            e, t = count_block_errors(hard_bits(llr), b.bits); te += e; tot += t
    return te / tot

rows = []
for name, model in RX.items():
    for snr in SNRS:
        no = float(ch.snr_db_to_no(torch.tensor(float(snr))).item())
        vals = []
        for k in range(K):
            torch.manual_seed(1000 + k)   # independent draw per pass
            vals.append(one_pass(model, snr, no))
        vals = np.array(vals)
        rows.append({"receiver": name, "snr_db": snr,
                     "bler_mean": vals.mean(), "bler_std": vals.std()})
        print(f"{name:8s} SNR={snr:3d} | BLER = {vals.mean():.4f} +/- {vals.std():.4f}", flush=True)

with open("results/stage_b_ci.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["receiver","snr_db","bler_mean","bler_std"]); w.writeheader(); w.writerows(rows)

# Figure with error bars
plt.figure(figsize=(5.2, 4))
mk = {"AINR":"o","DiscNRX":"s","LMMSE":"^"}
floor = 5e-4
for name in RX:
    rs = [r for r in rows if r["receiver"]==name]
    x = [r["snr_db"] for r in rs]
    y = np.array([max(r["bler_mean"], floor) for r in rs])
    e = np.array([r["bler_std"] for r in rs])
    plt.errorbar(x, y, yerr=e, marker=mk[name], capsize=3, label=name)
plt.yscale("log"); plt.ylim(floor, 1.5); plt.xlabel("SNR (dB)"); plt.ylabel("BLER")
plt.title(f"Matched BLER with 1-sigma CI (K={K} passes)")
plt.grid(True, which="both", ls=":", alpha=0.5); plt.legend(); plt.tight_layout()
plt.savefig("results/plots/bler_stage_b_ci.pdf"); plt.close()
print("-> results/plots/bler_stage_b_ci.pdf")
print("-> results/stage_b_ci.csv")
print("DONE")
