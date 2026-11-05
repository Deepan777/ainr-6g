"""Development-seed free-energy trajectories only (seeds 2-4).

Purpose: check whether the pre-registered R1 rule degenerates (fires at the first
snapshot in every run) BEFORE the held-out seeds are evaluated. Uses only the
development seeds named in ANALYSIS_PROTOCOL_BUDGET.md section 1. Computes the
free energy alone -- no waterfall, no OOD, no pilot sweep -- so it costs ~1.5 s
per snapshot and barely disturbs the training running alongside it.

Output: results/ojcoms/dev_fe_trajectory.csv  (development data, not the endpoint)
"""
import csv, os
import torch
from exp_common import ROOT, DEV, MATCHED, load_cfg, make_channel
from exp_budget_curve import load as _unused   # noqa: F401  (guarded import check)
from src.ainr import AINR

STEPS = list(range(1000, 10001, 1000)) + list(range(12500, 50001, 2500))
cfg = load_cfg(16)
ch = make_channel(cfg, *MATCHED[1:])
out = os.path.join(ROOT, "results", "ojcoms", "dev_fe_trajectory.csv")
rows = []
for seed in (2, 3, 4):
    for step in STEPS:
        p = os.path.join(ROOT, "results", "dense", f"seed{seed}", "checkpoints", f"ainr_step{step}.pt")
        if not os.path.exists(p):
            continue
        m = AINR(cfg, ch).to(DEV)
        m.load_state_dict(torch.load(p, map_location=DEV))
        m.eval()
        torch.manual_seed(7000)
        acc, n = {}, 0
        for _ in range(12):
            b = ch.generate_batch(32, snr_db=5.0)
            with torch.no_grad():
                o = m.compute_free_energy(b.received_grid)
            for s_, d_ in (("total", "fe"), ("expected_ll", "ell"),
                           ("kl_channel", "klh"), ("kl_noise", "kls")):
                v = getattr(o, s_, None)
                if v is not None:
                    acc[d_] = acc.get(d_, 0.0) + float(v.detach())
            n += 1
        rows.append({"seed": seed, "step": step, **{k: v / n for k, v in acc.items()}})
        del m
        torch.cuda.empty_cache()
    print(f"seed{seed}: {len([r for r in rows if r['seed']==seed])} snapshots", flush=True)

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "step", "fe", "ell", "klh", "kls"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("DONE", out)
