"""
src/baselines/lmmse.py

Phase 6 — the classical LMMSE receiver baseline.

A fully classical, parameter-free OFDM receiver built from Sionna 2.x blocks:

    Y --(LS estimate on pilots)--> H_hat
    Y, H_hat --(LMMSE equalise)--> X_hat, no_eff
    X_hat --(QAM demap)--> coded LLRs
    coded LLRs --(LDPC decode, soft)--> info-bit LLRs (B, K)

It exposes the same interface as :class:`~src.ainr.AINR`
(``forward(Y) -> LLRs (B, K)``) so the evaluation harness can treat all three
receivers uniformly. It has **no trainable parameters**, so ``adapt_online`` is
a no-op (LMMSE cannot learn from data).

Notes
-----
* The received grid arrives in CLAUDE.md layout ``(B, N_sc, N_sym, N_rx)`` and is
  converted back to Sionna layout ``(B, num_rx=1, N_rx, N_sym, N_sc)`` for the
  Sionna blocks.
* LMMSE needs the noise variance ``no``. ``forward(Y, no=...)`` accepts it; if
  omitted, a stored ``default_no`` is used (the evaluation loop sets ``no`` per
  SNR point). Within a batch all slots share the same SNR, so a scalar ``no`` is
  exact.
* A *soft-output* LDPC decoder (``hard_out=False``) is used so the receiver emits
  info-bit LLRs (positive => bit 1, matching the project convention) rather than
  hard bits.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mimo import StreamManagement
from sionna.phy.fec.ldpc import LDPC5GDecoder

from src.utils import infer_dims
from src.channel.sionna_channel import SionnaChannel


class LMMSEReceiver(nn.Module):
    """Classical LMMSE-equalisation receiver (no learnable parameters)."""

    def __init__(self, config, sionna_channel: SionnaChannel):
        super().__init__()
        self.config = config
        self.channel = sionna_channel
        dev = sionna_channel.device
        self.device = dev

        dims = infer_dims(config)
        self.N_sc = dims.n_subcarriers
        self.N_sym = dims.n_symbols
        self.N_rx = dims.n_rx
        self.n_tx = dims.n_tx
        self.K = dims.K

        rg = sionna_channel.resource_grid

        # rx_tx_association[i, j] = 1  <=>  receiver i decodes transmitter j.
        rx_tx_association = np.array([[1]])  # 1 receiver, 1 transmitter
        self.stream_management = StreamManagement(rx_tx_association, self.n_tx)

        self.ls_estimator = LSChannelEstimator(
            rg, interpolation_type="lin", device=dev
        )
        self.lmmse_equalizer = LMMSEEqualizer(
            rg, self.stream_management, device=dev
        )
        # Reuse the channel's demapper (same QAM order / convention).
        self.demapper = sionna_channel.demapper
        # Soft-output decoder -> info-bit LLRs (the channel's decoder is hard-out).
        self.decoder = LDPC5GDecoder(
            sionna_channel.encoder,
            hard_out=False,
            return_infobits=True,
            num_iter=20,
            device=dev,
        )

        # Default noise variance (mid of the training SNR range); evaluation
        # overrides this per SNR point via forward(Y, no=...).
        mid_snr = 0.5 * (float(config.channel.snr_db_min)
                         + float(config.channel.snr_db_max))
        self.default_no = float(10.0 ** (-mid_snr / 10.0))

    # ------------------------------------------------------------------ #
    def _format_no(self, no: Optional[Union[float, torch.Tensor]]) -> torch.Tensor:
        """Return a scalar noise-variance tensor on the right device."""
        if no is None:
            no = self.default_no
        if torch.is_tensor(no):
            no = no.to(self.device).float()
            # All slots in an eval batch share one SNR -> collapse to a scalar.
            return no.reshape(-1)[0] if no.ndim > 0 else no
        return torch.tensor(float(no), device=self.device, dtype=torch.float32)

    def _to_sionna_layout(self, Y: torch.Tensor) -> torch.Tensor:
        """(B, N_sc, N_sym, N_rx) -> (B, num_rx=1, N_rx, N_sym, N_sc)."""
        # Inverse of SionnaChannel._received_to_claude_layout.
        y = Y.permute(0, 3, 2, 1)            # (B, N_rx, N_sym, N_sc)
        return y.unsqueeze(1).contiguous()   # (B, 1, N_rx, N_sym, N_sc)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        Y: torch.Tensor,
        no: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Classical receive chain -> info-bit LLRs.

        Args:
            Y:  received grid, complex ``(B, N_sc, N_sym, N_rx)``.
            no: AWGN variance (scalar). Defaults to ``self.default_no``.
        Returns:
            llrs: ``(B, K)`` — logits of P(bit = 1).
        """
        B = Y.shape[0]
        y_sionna = self._to_sionna_layout(Y)
        no_t = self._format_no(no)

        h_hat, err_var = self.ls_estimator(y_sionna, no_t)
        x_hat, no_eff = self.lmmse_equalizer(y_sionna, h_hat, err_var, no_t)
        coded_llr = self.demapper(x_hat, no_eff)     # (B, 1, n_tx, n)
        info_llr = self.decoder(coded_llr)           # (B, 1, n_tx, k) soft
        return info_llr.reshape(B, self.K)

    # ------------------------------------------------------------------ #
    def adapt_online(self, Y: torch.Tensor, decoded_bits: torch.Tensor,
                     lr: Optional[float] = None) -> float:
        """No-op: a classical LMMSE receiver has nothing to adapt."""
        return 0.0

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
