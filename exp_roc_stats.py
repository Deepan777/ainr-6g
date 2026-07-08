"""Statistical strengthening of the drift-detection results.

For each drift scenario (S2 delay-spread, S3 Doppler):
  * R independent runs of N slots per condition -> per-run AUC, mean +/- std;
  * pooled scores -> bootstrap 95% CI on the AUC;
  * operating points: TPR at FPR = 1%, 5%, 10% (threshold from matched scores);
  * sensitivity of the AUC and of the per-slot monitor cost to the number of
    Monte-Carlo posterior samples S in {1, 2, 4, 8, 16}.

Outputs:
  results/roc_stats.csv         (scenario, run, auc, n_matched, n_drift)
  results/roc_summary.csv       (scenario, auc_mean, auc_std, ci_lo, ci_hi,
                                 tpr_fpr01, tpr_fpr05, tpr_fpr10, n_total)
  results/roc_sensitivity.csv   (scenario, n_samples, auc, ms_per_slot)
  results/roc_scores_pooled.csv (scenario, label, score)  — for figures

Usage:  python exp_roc_stats.py [--quick]
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
from src.evaluation.scenarios import build_scenario_channel, scenario_label

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

R_RUNS = 2 if args.quick else 5
N_SLOTS = 10 if args.quick else 400        # per condition per run
SENS_SLOTS = 8 if args.quick else 300      # per condition, sensitivity sweep
SNR = 10.0
MC_GRID = [1, 2, 4, 8, 16]
N_BOOT = 200 if args.quick else 2000

ch_m = build_scenario_channel(cfg, "s1_matched", dev)
model = AINR(cfg, ch_m).to(dev)
model.load_state_dict(torch.load(os.path.join(_ROOT, "results", "checkpoints",
                                              "ainr_final.pt"), map_location=dev))
model.eval()


@torch.no_grad()
def collect(chan, n, n_samples=4):
    out = []
    for _ in range(n):
        b = chan.generate_batch(1, snr_db=SNR)
        out.append(model.reconstruction_error(b.received_grid, n_samples=n_samples))
    return np.array(out)


def auc_mw(pos, neg):
    """Mann-Whitney AUC = P(drift score > matched score)."""
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def bootstrap_ci(pos, neg, n_boot, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))]
        vals[i] = auc_mw(p, n)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


stat_rows, summary_rows, pooled_rows, sens_rows = [], [], [], []
for scen in ["s2_delay_shift", "s3_speed_shift"]:
    label = scenario_label(scen)
    ch_d = build_scenario_channel(cfg, scen, dev)

    pos_all, neg_all = [], []
    for r in range(R_RUNS):
        torch.manual_seed(100 + r)
        neg = collect(ch_m, N_SLOTS)      # matched
        pos = collect(ch_d, N_SLOTS)      # drift
        a = auc_mw(pos, neg)
        stat_rows.append({"scenario": label, "run": r, "auc": a,
                          "n_matched": len(neg), "n_drift": len(pos)})
        print(f"[{label}] run {r}: AUC={a:.4f}", flush=True)
        pos_all.append(pos); neg_all.append(neg)

    pos_all = np.concatenate(pos_all); neg_all = np.concatenate(neg_all)
    aucs = np.array([row["auc"] for row in stat_rows if row["scenario"] == label])
    lo, hi = bootstrap_ci(pos_all, neg_all, N_BOOT)
    # Operating points: threshold at the (1 - FPR) quantile of matched scores.
    ops = {}
    for fpr in [0.01, 0.05, 0.10]:
        thr = np.quantile(neg_all, 1.0 - fpr)
        ops[fpr] = float((pos_all > thr).mean())
    summary_rows.append({"scenario": label,
                         "auc_mean": aucs.mean(), "auc_std": aucs.std(),
                         "ci_lo": lo, "ci_hi": hi,
                         "tpr_fpr01": ops[0.01], "tpr_fpr05": ops[0.05],
                         "tpr_fpr10": ops[0.10],
                         "n_total": len(pos_all) + len(neg_all)})
    print(f"[{label}] AUC={aucs.mean():.4f}+/-{aucs.std():.4f} "
          f"CI95=[{lo:.4f},{hi:.4f}] TPR@1%FPR={ops[0.01]:.3f} "
          f"TPR@5%={ops[0.05]:.3f} TPR@10%={ops[0.10]:.3f}", flush=True)

    for s in neg_all:
        pooled_rows.append({"scenario": label, "label": 0, "score": float(s)})
    for s in pos_all:
        pooled_rows.append({"scenario": label, "label": 1, "score": float(s)})

    # --- MC-sample sensitivity -------------------------------------------
    for S in MC_GRID:
        torch.manual_seed(999)
        t0 = time.perf_counter()
        neg = collect(ch_m, SENS_SLOTS, n_samples=S)
        pos = collect(ch_d, SENS_SLOTS, n_samples=S)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / (2 * SENS_SLOTS) * 1000.0
        a = auc_mw(pos, neg)
        sens_rows.append({"scenario": label, "n_samples": S, "auc": a,
                          "ms_per_slot": ms})
        print(f"[{label}] S={S:2d}: AUC={a:.4f} ({ms:.1f} ms/slot)", flush=True)


def dump(name, rows, fields):
    with open(os.path.join(_ROOT, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"-> results/{name}")


dump("roc_stats.csv", stat_rows, ["scenario", "run", "auc", "n_matched", "n_drift"])
dump("roc_summary.csv", summary_rows,
     ["scenario", "auc_mean", "auc_std", "ci_lo", "ci_hi",
      "tpr_fpr01", "tpr_fpr05", "tpr_fpr10", "n_total"])
dump("roc_sensitivity.csv", sens_rows, ["scenario", "n_samples", "auc", "ms_per_slot"])
dump("roc_scores_pooled.csv", pooled_rows, ["scenario", "label", "score"])
print("ROCSTATS_DONE", flush=True)
