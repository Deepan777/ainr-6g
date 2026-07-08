"""Proper ROC curve: TPR vs FPR for the reconstruction-error drift detector.

Sweeps the detection threshold across the empirical distribution of matched and
drift per-slot reconstruction errors, computing TPR and FPR at each threshold.
Produces results/plots/roc_drift_detection.pdf and prints the AUC.
"""
import os, sys, csv, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _ROOT)
from src.ainr import AINR
from src.evaluation.scenarios import build_scenario_channel, scenario_label

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

ch_m = build_scenario_channel(cfg, "s1_matched", dev)
m = AINR(cfg, ch_m).to(dev)
m.load_state_dict(torch.load("results/checkpoints/ainr_final.pt", map_location=dev))
m.eval()

N_SLOTS = 500   # per condition — enough for a smooth ROC
SNR = 10.0

def collect_errors(ch, n):
    errs = []
    with torch.no_grad():
        for _ in range(n):
            b = ch.generate_batch(1, snr_db=SNR)
            errs.append(m.reconstruction_error(b.received_grid))
    return np.array(errs)

results = {}
for scen in ["s2_delay_shift", "s3_speed_shift"]:
    ch_d = build_scenario_channel(cfg, scen, dev)
    matched = collect_errors(ch_m, N_SLOTS)
    drift   = collect_errors(ch_d, N_SLOTS)
    labels  = np.concatenate([np.zeros(N_SLOTS), np.ones(N_SLOTS)])
    scores  = np.concatenate([matched, drift])
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [], []
    for t in thresholds:
        pred = scores >= t
        tprs.append(pred[labels==1].mean())
        fprs.append(pred[labels==0].mean())
    tprs, fprs = np.array(tprs), np.array(fprs)
    order = np.argsort(fprs)                  # integrate w.r.t. increasing FPR
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tprs[order], fprs[order]))
    results[scen] = (fprs, tprs, auc, matched, drift)
    print(f"{scenario_label(scen)}: AUC = {auc:.4f} | "
          f"matched {matched.mean():.3f}±{matched.std():.3f} "
          f"drift {drift.mean():.3f}±{drift.std():.3f}")

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (scen, (fprs, tprs, auc, matched, drift)) in zip(axes, results.items()):
    ax.plot(fprs, tprs, lw=2, label=f"AINR (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Drift Detection ROC — {scenario_label(scen)}")
    ax.legend(); ax.grid(True, ls=":", alpha=0.5); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.tight_layout()
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/roc_drift_detection.pdf")
plt.close()
print("-> results/plots/roc_drift_detection.pdf")

# --- Save ROC data as CSV ---
with open("results/roc_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "fpr", "tpr", "auc"])
    for scen, (fprs, tprs, auc, _, __) in results.items():
        for fpr, tpr in zip(fprs, tprs):
            w.writerow([scen, round(float(fpr),4), round(float(tpr),4), round(auc,4)])
print("-> results/roc_data.csv")
print("DONE")
