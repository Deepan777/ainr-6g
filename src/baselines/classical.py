"""Classical OFDM receivers used as reference baselines.

Three variants share the demapper and a soft-output 5G LDPC decoder:

* ``ls_lin``  -- LS estimation at pilots + linear interpolation, LMMSE
                 equalization (the conventional low-complexity chain).
* ``ls_lmmse`` -- LS estimation + LMMSE (Wiener) interpolation/smoothing in
                 time, frequency and space, using channel covariances estimated
                 empirically from the *training* (matched) channel distribution.
                 Under drift the covariances are mismatched, exactly as the
                 neural receivers' training distribution is.
* ``perfect`` -- LMMSE equalization with the true channel (upper reference).

All variants receive the true noise variance, which favours the classical
chain relative to the neural receivers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer, LMMSEInterpolator
from sionna.phy.mimo import StreamManagement
from sionna.phy.fec.ldpc import LDPC5GDecoder


def estimate_covariances(channel, n_batches: int = 50, batch_size: int = 64):
    """Empirical time/frequency/space covariances of ``channel``'s h_freq.

    Returns (cov_time [N_sym,N_sym], cov_freq [N_sc,N_sc], cov_space [N_rx,N_rx]),
    each complex64 and normalised to unit mean diagonal.
    """
    N_sym, N_sc, N_rx = channel.n_symbols, channel.n_subcarriers, channel.n_rx
    ct = torch.zeros(N_sym, N_sym, dtype=torch.complex128, device=channel.device)
    cf = torch.zeros(N_sc, N_sc, dtype=torch.complex128, device=channel.device)
    cs = torch.zeros(N_rx, N_rx, dtype=torch.complex128, device=channel.device)
    for _ in range(n_batches):
        h = channel.generate_channel(batch_size)          # (B,1,N_rx,1,1,N_sym,N_sc)
        h = h[:, 0, :, 0, 0].to(torch.complex128)          # (B,N_rx,N_sym,N_sc)
        cf += torch.einsum("bris,brit->st", h.conj(), h) / (batch_size * N_rx * N_sym)
        ct += torch.einsum("bris,brjs->ij", h.conj(), h) / (batch_size * N_rx * N_sc)
        cs += torch.einsum("bris,bqis->rq", h.conj(), h) / (batch_size * N_sym * N_sc)
    out = []
    for c in (ct, cf, cs):
        c = c / n_batches
        c = c / torch.real(torch.diagonal(c)).mean()
        c = 0.5 * (c + c.conj().T)                         # enforce Hermitian
        out.append(c.to(torch.complex64))
    return tuple(out)


class ClassicalReceiver(nn.Module):
    """LS/LMMSE-interpolation/perfect-CSI receivers with LMMSE equalization."""

    def __init__(self, config, channel, mode: str = "ls_lin",
                 covariances: Optional[tuple] = None):
        super().__init__()
        if mode not in ("ls_lin", "ls_lmmse", "perfect"):
            raise ValueError(f"unknown mode {mode}")
        self.mode = mode
        self.channel = channel
        dev = channel.device
        rg = channel.resource_grid
        self.sm = StreamManagement(np.array([[1]]), channel.n_tx)
        if mode == "ls_lin":
            self.est = LSChannelEstimator(rg, interpolation_type="lin", device=dev)
        elif mode == "ls_lmmse":
            if covariances is None:
                raise ValueError("ls_lmmse requires covariances")
            ct, cf, cs = covariances
            interp = LMMSEInterpolator(rg.pilot_pattern, ct, cf, cs, order="t-f-s")
            self.est = LSChannelEstimator(rg, interpolator=interp, device=dev)
        self.eq = LMMSEEqualizer(rg, self.sm, device=dev)
        self.demapper = channel.demapper
        self.decoder = LDPC5GDecoder(channel.encoder, hard_out=False,
                                     return_infobits=True, num_iter=20, device=dev)
        self.K = channel.num_info_bits
        self.default_no = 0.1

    def _y_sionna(self, Y):
        return Y.permute(0, 3, 2, 1).unsqueeze(1).contiguous()

    def forward(self, Y: torch.Tensor, no=None, h_freq: Optional[torch.Tensor] = None):
        B = Y.shape[0]
        y = self._y_sionna(Y)
        no = self.default_no if no is None else no
        no_t = torch.as_tensor(float(no) if not torch.is_tensor(no) else no,
                               device=Y.device, dtype=torch.float32).reshape(-1)[0]
        if self.mode == "perfect":
            if h_freq is None:
                raise ValueError("perfect-CSI receiver needs h_freq")
            h_hat = h_freq.to(y.dtype)
            err = torch.zeros((), device=Y.device)
        else:
            h_hat, err = self.est(y, no_t)
        x_hat, no_eff = self.eq(y, h_hat, err, no_t)
        llr = self.demapper(x_hat, no_eff)
        return self.decoder(llr).reshape(B, self.K)

    def adapt_online(self, *args, **kwargs) -> float:
        return 0.0
