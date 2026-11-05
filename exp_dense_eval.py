"""Evaluate every dense snapshot: trade-off frontier + all candidate stopping signals.

Implements the measurements required by
`FINAL_OJCOMS/audits/ANALYSIS_PROTOCOL_BUDGET.md` (frozen 2026-09-14).

Per snapshot (seed, step, receiver) it records
  * snr_at_0p1   matched SNR at BLER 0.1          -> trade-off frontier, and rule R2
  * bler_S1_5db  matched BLER at 5 dB             -> signal `mbler`
  * bler_S2, bler_DS1000   out-of-distribution    -> endpoint (never used by any criterion)
  * fe, ell, klh, kls      matched free energy    -> signals (AINR only)
  * ptol         margin-controlled pilot tolerance-> signal `ptol`

Resumable: rows are appended to results/ojcoms/dense_eval.csv and already-completed
(seed, step, receiver) triples are skipped.
"""
import csv, os, sys
import numpy as np
import torch

from exp_common import ROOT, DEV, CONDITIONS, MATCHED, load_cfg, make_channel, decode
from exp_budget_curve import bler, crossing
from src.ainr import AINR
from src.baselines.discriminative_nrx import DiscriminativeNRX

SEEDS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else [2, 3, 4, 5, 6, 7, 8])]
STEPS = list(range(1000, 10001, 1000)) + list(range(12500, 50001, 2500))
if os.environ.get("DENSE_STEPS"):   # smoke-testing / partial reruns
    STEPS = [int(x) for x in os.environ["DENSE_STEPS"].split(",")]
MULTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
GRID = [round(0.25 * i, 2) for i in range(0, 11)]
OUT = os.path.join(ROOT, "results", "ojcoms", "dense_eval.csv")
FIELDS = ["seed", "step", "receiver", "snr_at_0p1", "bler_S1_5db", "bler_S2", "bler_DS1000",
          "fe", "ell", "klh", "kls", "ptol"]

cfg = load_cfg(16)
COND = {c[0]: c for c in CONDITIONS}
ch_m = make_channel(cfg, *MATCHED[1:])
ch_s2 = make_channel(cfg, *COND["S2"][1:])
ch_ds = make_channel(cfg, *COND["DS1000"][1:])


def load_dense(seed, step, kind):
    p = os.path.join(ROOT, "results", "dense", f"seed{seed}", "checkpoints", f"{kind}_step{step}.pt")
    if not os.path.exists(p):
        return None
    m = AINR(cfg, ch_m).to(DEV) if kind == "ainr" else DiscriminativeNRX(cfg, pilot_grid=ch_m.pilot_grid).to(DEV)
    m.load_state_dict(torch.load(p, map_location=DEV))
    m.eval()
    if kind != "ainr":
        m._ensure_ldpc(DEV)
    return m


def free_energy(rx):
    torch.manual_seed(7000)
    acc, n = {}, 0
    for _ in range(12):
        b = ch_m.generate_batch(32, snr_db=5.0)
        with torch.no_grad():
            o = rx.compute_free_energy(b.received_grid)
        for src, dst in (("total", "fe"), ("expected_ll", "ell"),
                         ("kl_channel", "klh"), ("kl_noise", "kls")):
            v = getattr(o, src, None)
            if v is not None:
                acc[dst] = acc.get(dst, 0.0) + float(v.detach())
        n += 1
    return {k: v / n for k, v in acc.items()}


def pilot_tolerance(rx, snr):
    """dB of extra pilot noise at which matched BLER reaches 0.5, at this snapshot's own
    BLER-0.1 operating point (margin-controlled)."""
    if not np.isfinite(snr):
        return float("nan")
    vals = [bler(rx, ch_m, snr, 10, pilot_noise=a, seed=4000) for a in MULTS]
    for (a0, b0), (a1, b1) in zip(zip(MULTS, vals), zip(MULTS[1:], vals[1:])):
        if b0 < 0.5 <= b1 and a0 > 0:
            f = (0.5 - b0) / (b1 - b0)
            return 10 * np.log10(a0) + f * (10 * np.log10(a1) - 10 * np.log10(a0))
    return float("nan")


done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {(int(r["seed"]), int(r["step"]), r["receiver"]) for r in csv.DictReader(f)}
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for seed in SEEDS:
        for step in STEPS:
            for kind in ("ainr", "discnrx"):
                if (seed, step, kind) in done:
                    continue
                rx = load_dense(seed, step, kind)
                if rx is None:
                    continue
                pts = [(s, bler(rx, ch_m, s, 8, seed=1000 + int(100 * s))) for s in GRID]
                cr = crossing(pts)
                row = {"seed": seed, "step": step, "receiver": kind, "snr_at_0p1": cr,
                       "bler_S1_5db": bler(rx, ch_m, 5.0, 10, seed=2500),
                       "bler_S2": bler(rx, ch_s2, 5.0, 15, seed=2000),
                       "bler_DS1000": bler(rx, ch_ds, 5.0, 15, seed=2000),
                       "ptol": pilot_tolerance(rx, cr)}
                row.update(free_energy(rx) if kind == "ainr" else
                           {k: "" for k in ("fe", "ell", "klh", "kls")})
                w.writerow(row); f.flush()
                print(f"seed{seed} {step:6d} {kind:8s} wf={cr:.2f} S2={row['bler_S2']:.3f} "
                      f"ptol={row['ptol']:.2f} fe={row.get('fe','')}", flush=True)
                del rx
                torch.cuda.empty_cache()
print("DONE")
