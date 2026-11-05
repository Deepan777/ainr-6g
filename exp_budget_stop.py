"""Feasibility: can the generative model-fit statistic pick the stopping point
WITHOUT any out-of-distribution data?

For each AINR snapshot, compute the free energy and its components on held-out
MATCHED validation slots only, then ask whether they track the S2 degradation
that the same snapshot suffers. If they are flat, a validation-only stopping
criterion is not available from this statistic and the dense-snapshot study is
not worth running.

Output: results/ojcoms/budget_stop_signal.csv
"""
import csv, os
import torch
from exp_common import ROOT, DEV, MATCHED, load_cfg, make_channel, atomic_write_csv
from exp_budget_curve import load

cfg = load_cfg(16)
ch = make_channel(cfg, *MATCHED[1:])
rows = []
for seed in (2, 3, 4):
    for step in (12000, 25000, 50000):
        rx = load(seed, step, "ainr")
        torch.manual_seed(7000)
        acc = {}
        n = 0
        for _ in range(12):
            b = ch.generate_batch(32, snr_db=5.0)
            with torch.no_grad():
                o = rx.compute_free_energy(b.received_grid)
            for k in ("total", "expected_ll", "kl_bits", "kl_channel", "kl_noise"):
                v = getattr(o, k, None)
                if v is not None:
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
            n += 1
        r = {"seed": seed, "step": step, **{k: v / n for k, v in acc.items()}}
        rows.append(r)
        print(f"seed{seed} {step:6d}  " + "  ".join(f"{k}={v:12.2f}" for k, v in r.items() if k not in ("seed", "step")), flush=True)
        del rx
        torch.cuda.empty_cache()

atomic_write_csv(os.path.join(ROOT, "results", "ojcoms", "budget_stop_signal.csv"), rows)
print("DONE")
