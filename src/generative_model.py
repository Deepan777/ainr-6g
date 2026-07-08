"""
src/generative_model.py

Phase 3 (coded-bit design) — the physics-based likelihood p(Y | c, h, sigma^2),
where ``c`` are the **coded** bits (the LDPC codeword), NOT the information bits.

Why coded bits? Inferring *information* bits by minimising reconstruction error is
ill-posed: the LDPC encoder spreads each info bit across ~half the codeword, so
the reconstruction-vs-info-bits landscape is combinatorial and has no usable
gradient (verified empirically — even with perfect CSI, gradient descent on info
bits cannot decode). The *coded* bits, by contrast, map **locally** to the
transmitted signal (``coded bit -> one QAM symbol -> one resource element``), so
the reconstruction gradient w.r.t. each coded bit is smooth and informative. The
LDPC code is therefore inverted afterwards by a standard belief-propagation
decoder (in the receiver), not by gradient descent.

Forward model (fully differentiable in the soft coded bits):

    Y_pred = ApplyChannel( RG( QAM(c) ), H(h) )
    log p(Y | c, h, sigma^2) = log CN(Y; Y_pred, sigma^2 I)

  * **QAM map** — expected constellation point under the independent soft coded
    bits (exact for hard bits).
  * **Resource grid** — Sionna's ``ResourceGridMapper`` (inserts pilots).
  * **Channel** — ``L`` posterior delay taps -> per-subcarrier frequency response
    via a fixed DFT, applied with ``ApplyOFDMChannel`` (noiseless mean; the noise
    enters through the likelihood). A known ``h_freq`` may be supplied to bypass
    the tap estimate (perfect-CSI experiments).

No learnable parameters — only precomputed buffers (QAM constellation + bit
table, and the delay->frequency DFT matrix). It *is* the physics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.utils import infer_dims


class GenerativeModel(nn.Module):
    """Differentiable physical forward model / likelihood (coded-bit latent)."""

    def __init__(self, config, channel):
        super().__init__()
        self.config = config
        self.channel = channel  # plain SionnaChannel; not an nn.Module submodule

        dims = infer_dims(config)
        self.N_sc = dims.n_subcarriers
        self.N_sym = dims.n_symbols
        self.N_rx = dims.n_rx
        self.N_tx = dims.n_tx
        self.nbps = dims.n_bits_per_symbol
        self.L = int(config.model.n_channel_taps)

        self.n = int(channel.n)                       # coded bits / stream
        self.n_data = int(channel.num_data_symbols)   # QAM symbols / stream
        self.n_coded_total = self.n * self.N_tx       # total coded bits (latent dim)
        self.M = 2 ** self.nbps
        self.device = channel.device

        if self.n != self.n_data * self.nbps:
            raise ValueError(
                f"Coded length n={self.n} != n_data*nbps={self.n_data * self.nbps}."
            )
        if dims.n_coded != self.n:
            raise ValueError(
                f"infer_dims n_coded={dims.n_coded} != channel n={self.n}."
            )

        # ---- QAM constellation + MSB-first bit table ----------------------
        idx = torch.arange(self.M, device=self.device)
        table = torch.zeros(self.M, self.nbps, device=self.device)
        for j in range(self.nbps):
            table[:, j] = ((idx >> (self.nbps - 1 - j)) & 1).to(torch.float32)
        points = channel.mapper(table.reshape(1, self.M * self.nbps)).reshape(self.M)
        self.register_buffer("qam_table", table, persistent=False)        # (M, nbps)
        self.register_buffer(
            "qam_points", points.to(torch.complex64), persistent=False
        )                                                                  # (M,)

        # ---- Delay -> frequency DFT matrix (L x N_sc) ---------------------
        l = torch.arange(self.L, device=self.device).reshape(self.L, 1)
        sc = torch.arange(self.N_sc, device=self.device).reshape(1, self.N_sc)
        ang = -2.0 * math.pi * (l * sc) / float(self.N_sc)
        dft = torch.complex(torch.cos(ang), torch.sin(ang)).to(torch.complex64)
        self.register_buffer("dft", dft, persistent=False)                 # (L, N_sc)

    # ------------------------------------------------------------------ #
    #  Differentiable QAM (coded bits -> symbols)
    # ------------------------------------------------------------------ #
    def _soft_qam(self, coded_soft: torch.Tensor) -> torch.Tensor:
        """Expected QAM symbols under independent soft coded bits.

        Args:
            coded_soft: (S, B, N_tx, n) in [0, 1].
        Returns:
            symbols: (S, B, N_tx, n_data) complex.
        """
        S, B, ntx, _ = coded_soft.shape
        c = coded_soft.reshape(S, B, ntx, self.n_data, self.nbps).unsqueeze(-2)
        tbl = self.qam_table.view(1, 1, 1, 1, self.M, self.nbps)
        factor = tbl * c + (1.0 - tbl) * (1.0 - c)                 # (...,M,nbps)
        probs = factor.prod(dim=-1)                                # (...,n_data,M)
        return torch.einsum(
            "...m,m->...", probs.to(self.qam_points.dtype), self.qam_points
        )

    def _taps_to_h_freq(self, h_samples: torch.Tensor, SB: int) -> torch.Tensor:
        """Delay taps (S,B,N_rx,N_tx,L) -> Sionna-layout freq channel."""
        H = h_samples.reshape(SB, self.N_rx, self.N_tx, self.L)
        H = H @ self.dft                                   # (SB, N_rx, N_tx, N_sc)
        H = H[:, None, :, None, :, None, :]                # (SB,1,N_rx,1,N_tx,1,N_sc)
        return H.expand(
            SB, 1, self.N_rx, 1, self.N_tx, self.N_sym, self.N_sc
        ).contiguous()

    # ------------------------------------------------------------------ #
    #  Forward / likelihood (coded-bit latent)
    # ------------------------------------------------------------------ #
    def predict_grid(
        self,
        coded_soft: torch.Tensor,
        h_samples: torch.Tensor = None,
        h_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """Reconstruct the (noiseless) received grid Y_pred from coded bits.

        Args:
            coded_soft: (S, B, N_coded) in [0, 1], N_coded = n * N_tx.
            h_samples:  (S, B, N_rx, N_tx, L) complex delay taps (used unless
                        ``h_freq`` is provided).
            h_freq:     optional known channel in Sionna layout
                        ``(S*B, 1, N_rx, 1, N_tx, N_sym, N_sc)`` (perfect CSI).
        Returns:
            Y_pred: (S, B, N_sc, N_sym, N_rx) complex.
        """
        S, B, _ = coded_soft.shape
        SB = S * B
        coded = coded_soft.reshape(S, B, self.N_tx, self.n)
        symbols = self._soft_qam(coded)                    # (S,B,N_tx,n_data)
        sym_f = symbols.reshape(SB, self.N_tx, self.n_data).unsqueeze(1)
        tx_grid = self.channel.rg_mapper(sym_f)            # (SB,1,N_tx,N_sym,N_sc)
        if h_freq is None:
            h_freq = self._taps_to_h_freq(h_samples, SB)
        y = self.channel.apply_channel(tx_grid, h_freq, None)
        y = self.channel._received_to_claude_layout(y)
        return y.reshape(S, B, self.N_sc, self.N_sym, self.N_rx)

    def log_prob(
        self,
        Y: torch.Tensor,
        coded_soft: torch.Tensor,
        h_samples: torch.Tensor,
        sigma_samples: torch.Tensor,
        h_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """Complex-Gaussian log-likelihood log p(Y | c, h, sigma^2) -> (S, B).

        Args:
            Y:             (B, N_sc, N_sym, N_rx) complex — observed grid.
            coded_soft:    (S, B, N_coded) in [0, 1] — soft coded bits.
            h_samples:     (S, B, N_rx, N_tx, L) complex (ignored if h_freq given).
            sigma_samples: (S, B, 1) noise *std*.
            h_freq:        optional known channel (perfect CSI) — see predict_grid.
        """
        if not torch.is_complex(Y):
            raise TypeError("GenerativeModel.log_prob expects a complex grid Y.")
        y_pred = self.predict_grid(coded_soft, h_samples, h_freq=h_freq)
        resid = Y.unsqueeze(0) - y_pred
        sum_sq = (resid.real ** 2 + resid.imag ** 2).sum(dim=(2, 3, 4))   # (S,B)
        sigma_sq = sigma_samples.squeeze(-1) ** 2
        n_re = self.N_sc * self.N_sym * self.N_rx
        return (-n_re * math.log(math.pi) - n_re * torch.log(sigma_sq)
                - sum_sq / sigma_sq)

    def expected_log_likelihood(
        self,
        Y: torch.Tensor,
        coded_soft: torch.Tensor,
        h_samples: torch.Tensor,
        sigma_samples: torch.Tensor,
        h_freq: torch.Tensor = None,
    ) -> torch.Tensor:
        """Monte-Carlo E_q[log p(Y|c,h,sigma^2)] (mean over the S samples)."""
        return self.log_prob(
            Y, coded_soft, h_samples, sigma_samples, h_freq=h_freq
        ).mean(dim=0)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())  # == 0 (physics only)
