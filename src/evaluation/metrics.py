"""
src/evaluation/metrics.py

Phase 8 — evaluation metrics shared by all stages.

All receivers in this project emit **info-bit LLRs** of shape ``(B, K)`` with the
convention *positive logit => bit 1* (see CLAUDE.md Phase 2 / Phase 6). A block
(transport block) is the full K-bit information word for one slot, so a block
error is "any of the K decoded bits is wrong".

Provides:
  * ``hard_bits``                     — LLR -> {0,1} decision.
  * ``count_block_errors`` / ``compute_bler`` — block error rate (BLER).
  * ``compute_ber``                   — raw bit error rate (diagnostic).
  * ``measure_latency``               — mean inference latency (ms/slot), CUDA-synced,
                                        measuring only the LLR path (F-monitoring off).
  * ``free_energy_drift_correlation`` — Pearson corr. between a per-slot free-energy
                                        series and a post-drift indicator (tests H2).
"""

from __future__ import annotations

import inspect
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Decisions / error rates
# --------------------------------------------------------------------------- #
def hard_bits(llrs: torch.Tensor) -> torch.Tensor:
    """LLRs (positive => bit 1) -> hard bits {0,1} as float."""
    return (llrs > 0).to(torch.float32)


def count_block_errors(decoded_bits: torch.Tensor,
                       true_bits: torch.Tensor) -> Tuple[int, int]:
    """Return (num_block_errors, num_blocks) for a ``(B, K)`` pair."""
    d = decoded_bits > 0.5
    t = true_bits.to(d.device) > 0.5
    block_err = (d != t).any(dim=1)
    return int(block_err.sum().item()), int(block_err.numel())


def compute_bler(decoded_bits: torch.Tensor, true_bits: torch.Tensor) -> float:
    """Block error rate over a ``(B, K)`` batch."""
    errs, n = count_block_errors(decoded_bits, true_bits)
    return errs / n if n else float("nan")


def compute_ber(decoded_bits: torch.Tensor, true_bits: torch.Tensor) -> float:
    """Raw bit error rate over a ``(B, K)`` batch (diagnostic)."""
    d = decoded_bits > 0.5
    t = true_bits.to(d.device) > 0.5
    return float((d != t).to(torch.float32).mean().item())


# --------------------------------------------------------------------------- #
#  Latency
# --------------------------------------------------------------------------- #
def _sync(device) -> None:
    if torch.is_tensor(device):
        device = device.device
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(receiver, Y: torch.Tensor,
                    n_warmup: int = 3, n_iters: int = 20) -> float:
    """Mean inference latency in **ms per call** for ``receiver(Y)``.

    Measures only the LLR path: if the receiver's ``forward`` accepts
    ``record_fe`` (AINR), it is disabled so the (optional) free-energy monitoring
    does not inflate the latency or pollute ``free_energy_series``.
    """
    kwargs = {}
    try:
        if "record_fe" in inspect.signature(receiver.forward).parameters:
            kwargs["record_fe"] = False
    except (ValueError, TypeError):
        pass

    with torch.no_grad():
        for _ in range(n_warmup):
            receiver(Y, **kwargs)
        _sync(Y)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            receiver(Y, **kwargs)
        _sync(Y)
        t1 = time.perf_counter()
    return (t1 - t0) / max(n_iters, 1) * 1000.0


# --------------------------------------------------------------------------- #
#  Free-energy drift detection (Hypothesis H2)
# --------------------------------------------------------------------------- #
def free_energy_drift_correlation(fe_series: Sequence[float],
                                  drift_slot: int) -> float:
    """Pearson correlation between the free-energy series and a step indicator
    that turns on at ``drift_slot`` (1 after drift, 0 before).

    A strong positive value means F rose at the drift event — evidence for H2
    (the free energy is a usable drift detector). Returns NaN when undefined.
    """
    fe = np.asarray(list(fe_series), dtype=float)
    if fe.size < 2:
        return float("nan")
    indicator = (np.arange(fe.size) >= drift_slot).astype(float)
    if fe.std() == 0.0 or indicator.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(fe, indicator)[0, 1])


def moving_average(x: Sequence[float], window: int) -> np.ndarray:
    """Causal moving average for smoothing per-slot series in plots."""
    x = np.asarray(list(x), dtype=float)
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")
