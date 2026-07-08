"""
src/baselines/discriminative_nrx.py

Phase 6 — the discriminative neural receiver baseline (coded-bit design).

Direct discriminative counterpart of the AINR: same CNN backbone and
convolutional bit-head as :class:`~src.variational_posterior.VariationalPosterior`
(so the trainable parameter count matches), but trained by supervised
cross-entropy on the **coded** bits and with no channel/noise heads. Like the
AINR it emits coded-bit LLRs and runs a standard LDPC decoder internally, so
``forward(Y)`` returns info-bit LLRs ``(B, K)`` — the common receiver interface.

This mirrors the literature neural receiver (DeepRx/NRX): the CNN learns
equalisation + demapping to coded-bit LLRs (a local, learnable map), and the
LDPC code is inverted by belief propagation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

from src.utils import infer_dims, data_symbol_indices
from src.variational_posterior import ResidualBlock, build_cnn_input  # shared arch


class DiscriminativeNRX(nn.Module):
    """Feed-forward CNN receiver trained by supervised coded-bit cross-entropy."""

    def __init__(self, config, pilot_grid=None):
        super().__init__()
        self.config = config
        self.use_pilot = pilot_grid is not None
        self.register_buffer(
            "pilot_grid",
            pilot_grid.detach().clone() if pilot_grid is not None else None,
            persistent=False,
        )

        dims = infer_dims(config)
        self.dims = dims
        self.N_sc = dims.n_subcarriers
        self.N_sym = dims.n_symbols
        self.N_rx = dims.n_rx
        self.n_tx = dims.n_tx
        self.k = dims.k_info                       # info bits / stream
        self.n = dims.n_coded                      # coded bits / stream
        self.K = dims.K                            # total info bits
        self.n_coded = dims.n_coded * dims.n_tx    # total coded bits (output dim)

        data_syms = data_symbol_indices(config)
        self.register_buffer(
            "data_sym_idx", torch.tensor(data_syms, dtype=torch.long), persistent=False
        )
        self.n_data_sym = len(data_syms)

        num_data_re = self.N_sc * self.n_data_sym
        if self.n_coded % num_data_re != 0:
            raise ValueError(
                f"n_coded={self.n_coded} not divisible by num_data_re={num_data_re}."
            )
        self.coded_bits_per_re = self.n_coded // num_data_re

        # ---- Backbone (identical to VariationalPosterior) -----------------
        F_ch = int(config.model.n_filters)
        ks = int(config.model.kernel_size)
        n_blocks = int(config.model.n_conv_blocks)
        in_ch = 2 * self.N_rx * (2 if self.use_pilot else 1)  # Y (+ pilot LS est.)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, F_ch, ks, padding=ks // 2),
            nn.GroupNorm(math.gcd(8, F_ch) or 1, F_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualBlock(F_ch, ks) for _ in range(n_blocks)])
        self.bit_head = nn.Conv2d(F_ch, self.coded_bits_per_re, kernel_size=1)

        # Lazily-built (parameter-free) LDPC encoder/decoder. Stored in a plain
        # dict so they are NOT registered as nn.Module submodules — otherwise
        # their (deterministic) buffers would land in state_dict and break
        # resume (the freshly-constructed model hasn't built them yet).
        self._ldpc: dict = {}
        self._adapt_opt: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------ #
    def _ensure_ldpc(self, device) -> None:
        if not self._ldpc:
            # Sionna expects a device *string* ('cpu' / 'cuda:0'), not a
            # torch.device object (and 'cuda' must be 'cuda:0').
            dev = str(device)
            if dev.startswith("cuda"):
                dev = "cuda:0"
            enc = LDPC5GEncoder(self.k, self.n, device=dev)
            self._ldpc["enc"] = enc
            self._ldpc["dec"] = LDPC5GDecoder(
                enc, hard_out=False, return_infobits=True, num_iter=20, device=dev,
            )

    def coded_llrs(self, Y: torch.Tensor) -> torch.Tensor:
        """Map the received grid to coded-bit LLRs (B, n_coded)."""
        B = Y.shape[0]
        x = build_cnn_input(Y, self.pilot_grid if self.use_pilot else None)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        re_logits = self.bit_head(x)                   # (B, cpr, N_sc, N_sym)
        idx = self.data_sym_idx.to(re_logits.device)
        re_logits = re_logits.index_select(3, idx)     # (B, cpr, N_sc, n_data_sym)
        # Reorder to codeword order (see VariationalPosterior.forward): Sionna
        # places symbols OFDM-symbol-major, subcarrier-minor; bits are consecutive.
        re_logits = re_logits.permute(0, 3, 2, 1).contiguous()  # (B, n_data_sym, N_sc, cpr)
        return re_logits.reshape(B, -1)                # (B, n_coded), codeword order

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Return info-bit LLRs (B, K) after LDPC-decoding the coded LLRs."""
        self._ensure_ldpc(Y.device)
        cl = self.coded_llrs(Y)                         # (B, n_coded)
        B = cl.shape[0]
        info = self._ldpc["dec"](cl.reshape(B, 1, self.n_tx, self.n))
        return info.reshape(B, self.K)                  # (B, K) soft

    # ------------------------------------------------------------------ #
    def _coded_targets(self, info_bits: torch.Tensor) -> torch.Tensor:
        """Re-encode info bits (B, K) -> coded-bit targets (B, n_coded)."""
        self._ensure_ldpc(info_bits.device)
        B = info_bits.shape[0]
        with torch.no_grad():
            ib = info_bits.to(torch.float32).reshape(B, 1, self.n_tx, self.k)
            return self._ldpc["enc"](ib).reshape(B, self.n_coded)

    def loss(self, Y: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """Supervised training loss: mean coded-bit BCE (sum over bits, mean batch)."""
        logits = self.coded_llrs(Y)
        targets = self._coded_targets(bits)
        return F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).sum(dim=1).mean()

    # ------------------------------------------------------------------ #
    def _ensure_optimizer(self, lr: Optional[float]) -> torch.optim.Optimizer:
        if lr is None:
            lr = float(self.config.eval.adaptation_lr)
        if self._adapt_opt is None:
            self._adapt_opt = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            for group in self._adapt_opt.param_groups:
                group["lr"] = lr
        return self._adapt_opt

    def adapt_online(
        self, Y: torch.Tensor, decoded_bits: torch.Tensor, lr: Optional[float] = None
    ) -> float:
        """One supervised cross-entropy step using CRC-validated decoded bits."""
        opt = self._ensure_optimizer(lr)
        self.train()
        opt.zero_grad(set_to_none=True)
        loss = self.loss(Y, decoded_bits)
        loss.backward()
        opt.step()
        return float(loss.detach().item())

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
