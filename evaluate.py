"""
evaluate.py — Stages B, C, D: evaluation and online adaptation.

Usage:
    python evaluate.py --stage B        # matched conditions (S1)
    python evaluate.py --stage C        # distribution shift (S2, S3) + F drift timeline
    python evaluate.py --stage D        # online adaptation after drift
    python evaluate.py --stage all      # B, C, D in sequence
    python evaluate.py --stage all --quick   # tiny budgets (pipeline smoke test)

All receivers emit info-bit LLRs ``(B, K)`` (positive => bit 1), so decoding is a
hard threshold — no separate LDPC pass (AINR/DiscNRX infer info bits directly;
LMMSE already LDPC-decodes internally). Inference runs under ``torch.no_grad()``;
the only gradient step is AINR/DiscNRX ``adapt_online`` in Stage D.

Key fixes vs. the scaffold: each scenario's :class:`SionnaChannel` is built
**once** (not per SNR point / per slot — critical under tight RAM), and LMMSE is
given the correct noise variance for the SNR under test.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.ainr import AINR
from src.baselines.lmmse import LMMSEReceiver
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.evaluation.metrics import (
    hard_bits, count_block_errors, compute_bler, measure_latency,
    free_energy_drift_correlation,
)
from src.evaluation.scenarios import build_scenario_channel, scenario_label

_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_ROOT, "config", "config.yaml")


# --------------------------------------------------------------------------- #
#  Receiver construction / loading
# --------------------------------------------------------------------------- #
def _results_dir(config) -> str:
    d = config.paths.results_dir
    d = d if os.path.isabs(d) else os.path.join(_ROOT, d)
    os.makedirs(d, exist_ok=True)
    return d


def _ckpt_dir(config) -> str:
    d = config.paths.checkpoint_dir
    return d if os.path.isabs(d) else os.path.join(_ROOT, d)


def _write_csv(path: str, rows: List[dict]) -> None:
    """Write a list-of-dicts to CSV (header from the first row's keys)."""
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_receivers(config, device) -> Dict[str, object]:
    """Build all three receivers, loading trained weights where available."""
    ckpt_dir = _ckpt_dir(config)
    # A channel is needed by AINR (for its generative model) and LMMSE.
    ch = build_scenario_channel(config, "s1_matched", device)

    ainr = AINR(config, ch).to(device)
    _maybe_load(ainr, os.path.join(ckpt_dir, "ainr_final.pt"), device, "AINR")
    ainr.eval()

    disc = DiscriminativeNRX(config, pilot_grid=ch.pilot_grid).to(device)
    _maybe_load(disc, os.path.join(ckpt_dir, "discnrx_final.pt"), device, "DiscNRX")
    disc.eval()

    lmmse = LMMSEReceiver(config, ch).to(device)

    return {"AINR": ainr, "DiscNRX": disc, "LMMSE": lmmse}


def _maybe_load(model, path, device, name) -> None:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  loaded {name} <- {os.path.basename(path)}")
    else:
        print(f"  WARNING: no checkpoint for {name} ({path}); using random init.")


# --------------------------------------------------------------------------- #
#  Core inference helpers
# --------------------------------------------------------------------------- #
def _forward_llrs(receiver, Y, no=None):
    """Uniform LLR forward: set LMMSE noise, disable AINR F-recording (speed)."""
    if no is not None and hasattr(receiver, "default_no"):
        receiver.default_no = float(no)
    if isinstance(receiver, AINR):
        return receiver(Y, record_fe=False)
    return receiver(Y)


def evaluate_one_snr(receiver, ch_eval, snr_db, n_batches, batch_size) -> float:
    """BLER for one receiver at one SNR under a prebuilt channel."""
    no = float(ch_eval.snr_db_to_no(torch.tensor(float(snr_db))).item())
    tot_err = tot = 0
    for _ in range(n_batches):
        batch = ch_eval.generate_batch(batch_size, snr_db=snr_db)
        with torch.no_grad():
            llrs = _forward_llrs(receiver, batch.received_grid, no)
        e, n = count_block_errors(hard_bits(llrs), batch.bits)
        tot_err += e
        tot += n
    return tot_err / tot if tot else float("nan")


# --------------------------------------------------------------------------- #
#  Stage B — matched conditions (S1)
# --------------------------------------------------------------------------- #
def evaluate_stage_b(config, receivers, device) -> list:
    print("\n=== Stage B: Matched Conditions (S1) ===")
    ch = build_scenario_channel(config, "s1_matched", device)
    snr_points = list(config.eval.snr_db_points)
    n_batches = int(config.eval.n_batches)
    bs = int(config.train.batch_size)

    sample_Y = ch.generate_batch(min(8, bs)).received_grid
    rows = []
    for name, rx in receivers.items():
        lat = measure_latency(rx, sample_Y)
        for snr in snr_points:
            bler = evaluate_one_snr(rx, ch, snr, n_batches, bs)
            rows.append({"receiver": name, "scenario": "S1", "snr_db": snr,
                         "bler": bler, "latency_ms": lat})
            print(f"  {name:8s} | SNR={snr:4.0f} dB | BLER={bler:.4f} | lat={lat:.2f} ms")

    _write_csv(os.path.join(_results_dir(config), "stage_b.csv"), rows)
    print("  -> results/stage_b.csv")
    return rows


# --------------------------------------------------------------------------- #
#  Stage C — distribution shift (S2, S3) + free-energy drift timeline (H2)
# --------------------------------------------------------------------------- #
def evaluate_stage_c(config, receivers, device) -> Dict[str, list]:
    print("\n=== Stage C: Distribution Shift ===")
    snr_points = list(config.eval.snr_db_points)
    n_batches = int(config.eval.n_batches)
    bs = int(config.train.batch_size)
    out = {}

    for scen in ["s2_delay_shift", "s3_speed_shift"]:
        ch = build_scenario_channel(config, scen, device)
        label = scenario_label(scen)
        rows = []
        for name, rx in receivers.items():
            for snr in snr_points:
                bler = evaluate_one_snr(rx, ch, snr, n_batches, bs)
                fe = float("nan")
                if hasattr(rx, "compute_free_energy"):
                    b = ch.generate_batch(bs, snr_db=snr)
                    with torch.no_grad():
                        fe = rx.compute_free_energy(b.received_grid).total.item()
                rows.append({"receiver": name, "scenario": label, "snr_db": snr,
                             "bler": bler, "free_energy": fe})
                print(f"  [{label}] {name:8s} | SNR={snr:4.0f} dB | "
                      f"BLER={bler:.4f} | F={fe:.1f}")
        short = scen.split("_")[0]            # s2 / s3
        _write_csv(os.path.join(_results_dir(config), f"stage_c_{short}.csv"), rows)
        print(f"  -> results/stage_c_{short}.csv")
        out[short] = rows

    # Free-energy drift timeline (Hypothesis H2): matched -> drift transition.
    out["drift_timeline"] = _free_energy_timeline(config, receivers.get("AINR"), device)
    return out


def _free_energy_timeline(config, ainr, device) -> list:
    """Record AINR's per-slot free energy across a matched->drift transition."""
    if ainr is None or not hasattr(ainr, "compute_free_energy"):
        return []
    print("  -- free-energy drift timeline (H2) --")
    n_each = int(config.eval.get("drift_timeline_slots", 200))
    snr = 10.0
    ch_matched = build_scenario_channel(config, "s1_matched", device)
    rows = []
    for scen in ["s2_delay_shift", "s3_speed_shift"]:
        ch_drift = build_scenario_channel(config, scen, device)
        fe_series, re_series = [], []
        for slot in range(2 * n_each):
            ch = ch_matched if slot < n_each else ch_drift
            b = ch.generate_batch(1, snr_db=snr)
            fe = ainr.compute_free_energy(b.received_grid).total.item()
            re = ainr.reconstruction_error(b.received_grid)   # the working detector
            fe_series.append(fe); re_series.append(re)
            rows.append({"scenario": scenario_label(scen), "slot": slot,
                         "phase": "matched" if slot < n_each else "drift",
                         "free_energy": fe, "recon_error": re})
        re_corr = free_energy_drift_correlation(re_series, n_each)
        f_corr = free_energy_drift_correlation(fe_series, n_each)
        mm = float(np.mean(re_series[:n_each])); dd = float(np.mean(re_series[n_each:]))
        print(f"     {scenario_label(scen)}: recon-error drift corr = {re_corr:+.3f} "
              f"(matched {mm:.3f} -> drift {dd:.3f}, {dd/max(mm,1e-9):.1f}x) | F corr {f_corr:+.3f}")
    _write_csv(os.path.join(_results_dir(config), "stage_c_drift_timeline.csv"), rows)
    print("  -> results/stage_c_drift_timeline.csv")
    return rows


# --------------------------------------------------------------------------- #
#  Stage D — online adaptation after a drift event
# --------------------------------------------------------------------------- #
def evaluate_stage_d(config, receivers, device) -> list:
    print("\n=== Stage D: Online Adaptation (decision-directed) ===")
    n_slots = int(config.eval.n_adaptation_slots)
    lr = float(config.eval.adaptation_lr)
    # Adapt at an SNR where the *drifted* receiver actually fails, so there is
    # something to recover (10 dB is too easy under S2 — BLER already ~0).
    snr = float(config.eval.get("adapt_snr_db", 5.0))
    seed = int(config.train.get("seed", 42))
    ckpt_dir = _ckpt_dir(config)

    ch = build_scenario_channel(config, "s2_delay_shift", device)  # the drift
    no = float(ch.snr_db_to_no(torch.tensor(snr)).item())
    rows = []

    for name, rx in receivers.items():
        # Fresh start from the trained checkpoint so D is independent of B/C.
        if name == "AINR":
            _maybe_load(rx, os.path.join(ckpt_dir, "ainr_final.pt"), device, "AINR")
        elif name == "DiscNRX":
            _maybe_load(rx, os.path.join(ckpt_dir, "discnrx_final.pt"), device, "DiscNRX")
        if hasattr(rx, "reset_free_energy_series"):
            rx.reset_free_energy_series()

        torch.manual_seed(seed)  # identical channel realisations across receivers
        for slot in tqdm(range(n_slots), desc=f"{name:8s}"):
            b = ch.generate_batch(int(config.eval.adaptation_batch), snr_db=snr)
            with torch.no_grad():
                llrs = _forward_llrs(rx, b.received_grid, no)
            decoded = hard_bits(llrs)
            bler = compute_bler(decoded, b.bits)
            crc_pass = bool((decoded > 0.5).eq(b.bits > 0.5).all().item())
            # Adapt ONLY on CRC-passed blocks (correct labels). Adapting on every
            # slot — including wrongly-decoded ones — feeds wrong self-labels and
            # catastrophically degrades the receiver (verified). CRC gating is the
            # standard HARQ/CRC-driven online-adaptation setup (CLAUDE.md Phase 5).
            if crc_pass and hasattr(rx, "adapt_online"):
                rx.adapt_online(b.received_grid, decoded, lr=lr)
            rows.append({"receiver": name, "slot": slot, "bler": bler,
                         "crc_pass": crc_pass})

    _write_csv(os.path.join(_results_dir(config), "stage_d.csv"), rows)
    # Recovery summary: mean BLER over first vs last 20% of slots.
    for name in receivers:
        sub = np.array([r["bler"] for r in rows if r["receiver"] == name], dtype=float)
        if sub.size:
            k = max(1, sub.size // 5)
            print(f"  {name:8s} | BLER first-{k}={sub[:k].mean():.3f} "
                  f"-> last-{k}={sub[-k:].mean():.3f}")
    print("  -> results/stage_d.csv")
    return rows


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["B", "C", "D", "all"], default="all")
    parser.add_argument("--quick", action="store_true",
                        help="tiny budgets for a pipeline smoke test")
    args, extra = parser.parse_known_args()

    # OmegaConf-style CLI overrides, e.g. eval.snr_db_points=[-5,0,2,4,6] eval.n_batches=400
    config = OmegaConf.merge(OmegaConf.load(_CONFIG_PATH), OmegaConf.from_dotlist(extra))
    if args.quick:
        config.eval.n_batches = 2
        config.eval.snr_db_points = [0, 20]
        config.eval.n_adaptation_slots = 8
        config.eval.drift_timeline_slots = 5

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | stage={args.stage} | quick={args.quick}")

    receivers = load_receivers(config, device)

    if args.stage in ("B", "all"):
        evaluate_stage_b(config, receivers, device)
    if args.stage in ("C", "all"):
        evaluate_stage_c(config, receivers, device)
    if args.stage in ("D", "all"):
        evaluate_stage_d(config, receivers, device)

    print("\n=== Evaluation complete. Results in results/ ===")
