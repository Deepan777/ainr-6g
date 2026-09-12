"""Shared infrastructure for the OJ-COMS experiment suite.

Training seeds: 0 is the original main checkpoint (trained with seed 42 in
results/checkpoints); 1-4 are independently trained replicas in
results/seeds/seed{s}/checkpoints. All evaluation channels carry a real
transport-block CRC (CRC16, 3GPP TS 38.212 Sec. 7.2.1) that the receiver
checks on its own decoded bits.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from src.channel.crc_channel import CRCSionnaChannel
from src.ainr import AINR
from src.baselines.discriminative_nrx import DiscriminativeNRX
from src.baselines.classical import ClassicalReceiver, estimate_covariances
from src.evaluation.metrics import hard_bits

ROOT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SEEDS_16QAM = [0, 1, 2, 3, 4]

# Evaluation conditions: name -> (CDL model, delay spread [ns], UE speed [km/h]).
MATCHED = ("S1", "C", 100.0, 3.0)
CONDITIONS = [
    MATCHED,
    ("DS200", "C", 200.0, 3.0), ("DS300", "C", 300.0, 3.0),
    ("DS600", "C", 600.0, 3.0), ("DS1000", "C", 1000.0, 3.0),
    ("CDLA", "A", 100.0, 3.0), ("CDLB", "B", 100.0, 3.0),
    ("CDLD", "D", 100.0, 3.0), ("CDLE", "E", 100.0, 3.0),
    ("S2", "A", 1000.0, 3.0),
    ("V30", "C", 100.0, 30.0), ("V90", "C", 100.0, 90.0),
    ("V150", "C", 100.0, 150.0), ("V250", "C", 100.0, 250.0),
    ("V400", "C", 100.0, 400.0),
]


def load_cfg(qam: int = 16):
    cfg = OmegaConf.load(os.path.join(ROOT, "config", "config.yaml"))
    cfg.phy.n_bits_per_symbol = 4 if qam == 16 else 6
    return cfg


def ckpt_dir(seed: int, qam: int = 16) -> str:
    if qam == 64:
        if seed != 0:
            raise ValueError("only one 64-QAM training seed exists")
        return os.path.join(ROOT, "results", "qam64", "checkpoints")
    if seed == 0:
        return os.path.join(ROOT, "results", "checkpoints")
    return os.path.join(ROOT, "results", "seeds", f"seed{seed}", "checkpoints")


def make_channel(cfg, cdl="C", ds_ns=100.0, speed_kmh=3.0) -> CRCSionnaChannel:
    return CRCSionnaChannel(cfg, device=DEV, cdl_model=cdl,
                            delay_spread_ns=float(ds_ns), ue_speed_kmh=float(speed_kmh))


def load_neural(cfg, ch, seed: int, qam: int = 16):
    d = ckpt_dir(seed, qam)
    ainr = AINR(cfg, ch).to(DEV)
    ainr.load_state_dict(torch.load(os.path.join(d, "ainr_final.pt"), map_location=DEV))
    ainr.eval()
    disc = DiscriminativeNRX(cfg, pilot_grid=ch.pilot_grid).to(DEV)
    disc.load_state_dict(torch.load(os.path.join(d, "discnrx_final.pt"), map_location=DEV))
    disc.eval()
    disc._ensure_ldpc(DEV)
    return ainr, disc


def matched_covariances(cfg, qam: int = 16):
    """S1 (training-distribution) channel covariances, cached on disk."""
    path = os.path.join(ROOT, "results", f"cov_s1_qam{qam}.pt")
    if os.path.exists(path):
        return tuple(t.to(DEV) for t in torch.load(path, map_location=DEV))
    g = torch.random.get_rng_state()
    torch.manual_seed(314159)
    ch = make_channel(cfg, *MATCHED[1:])
    cov = estimate_covariances(ch, n_batches=200, batch_size=64)
    torch.random.set_rng_state(g)
    torch.save(tuple(c.cpu() for c in cov), path)
    return cov


def load_classical(cfg, ch, qam: int = 16):
    cov = matched_covariances(cfg, qam)
    return {
        "LS-LIN": ClassicalReceiver(cfg, ch, "ls_lin"),
        "LS-LMMSE": ClassicalReceiver(cfg, ch, "ls_lmmse", covariances=cov),
        "PCSI": ClassicalReceiver(cfg, ch, "perfect"),
    }


@torch.no_grad()
def decode(rx, batch, ch):
    """Hard info-bit decisions (B, K) of any receiver."""
    Y = batch.received_grid
    if isinstance(rx, AINR):
        llr = rx(Y, record_fe=False)
    elif isinstance(rx, DiscriminativeNRX):
        llr = rx(Y)
    else:
        no = float(batch.no.reshape(-1)[0])
        llr = rx(Y, no=no, h_freq=batch.h_freq)
    return hard_bits(llr)


@torch.no_grad()
def ainr_monitor(ainr, Y, n_mc: int = 4):
    """Reconstruction residual and its per-OFDM-symbol profile.

    Returns (residual (B,), temporal dispersion (B,), bit logits (B, n)).
    residual  = E_q ||Y - G(c, h)||^2 / (N_sc N_sym N_rx)
    dispersion = Var_l(r_l) / Mean_l(r_l)^2 over the per-symbol residual r_l.
    """
    p = ainr.posterior(Y)
    s = ainr.posterior.sample_from_params(p, n_mc)
    yp = ainr.generative_model.predict_grid(s.bits_soft, s.h_samples)
    r = (Y.unsqueeze(0) - yp).abs() ** 2                 # (S,B,N_sc,N_sym,N_rx)
    resid = r.mean(dim=(0, 2, 3, 4))
    prof = r.mean(dim=(0, 2, 4))                         # (B, N_sym)
    disp = prof.var(dim=1) / prof.mean(dim=1).clamp_min(1e-12) ** 2
    return resid, disp, p.bit_logits


@torch.no_grad()
def pilot_statistics(Y, pilot_grid, pilot_syms):
    """Classical, receiver-agnostic pilot statistics from LS pilot estimates.

    c_f : normalised lag-1 frequency correlation (falls with delay spread).
    c_t : normalised correlation between the two pilot symbols (falls with
          Doppler / time selectivity).
    """
    pg = pilot_grid.to(Y.device)
    idx = torch.tensor(pilot_syms, device=Y.device)
    H = Y.index_select(2, idx) * torch.conj(pg.index_select(1, idx))[None, :, :, None]
    num_f = (H[:, 1:] * torch.conj(H[:, :-1])).sum(dim=1).abs()      # (B,P,R)
    den_f = (H.abs() ** 2).sum(dim=1).clamp_min(1e-12)
    c_f = (num_f / den_f).mean(dim=(1, 2))
    h0, h1 = H[:, :, 0, :], H[:, :, -1, :]
    num_t = (h0 * torch.conj(h1)).sum(dim=(1, 2)).abs()
    den_t = torch.sqrt((h0.abs() ** 2).sum(dim=(1, 2)) * (h1.abs() ** 2).sum(dim=(1, 2)))
    c_t = num_t / den_t.clamp_min(1e-12)
    return c_f, c_t


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    """Exact binomial confidence interval for k errors in n blocks."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def auc(pos, neg) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5 P(tie)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    allv = np.concatenate([neg, pos])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    r = np.empty(len(allv)); r[order] = ranks
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def atomic_write_csv(path: str, rows, fieldnames=None) -> None:
    """Write CSV to a temporary file, then rename. An interrupted write never
    leaves a partial file that a resumed run would mistake for a finished one."""
    import csv
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def read_csv_rows(path: str, text_cols=()) -> list:
    """Read a CSV written by atomic_write_csv; non-text columns become float."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k not in text_cols:
                r[k] = float(r[k])
    return rows


def atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def env_record() -> dict:
    import platform, sys, sionna
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "sionna": sionna.__version__, "gpu": gpu, "platform": platform.platform()}
