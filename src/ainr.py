"""
src/ainr.py

Phase 5 — AINR: the Active Inference Neural Receiver (top-level class).

Combines the :class:`VariationalPosterior` (inference network), the
:class:`GenerativeModel` (physics likelihood) and the variational free energy
into a single receiver that exposes the *same* interface as the baseline
receivers (``forward(Y) -> LLRs``), plus the active-inference extras:

  * ``compute_free_energy(Y)``  — full VFE breakdown (training / analysis).
  * ``adapt_online(Y, decoded_bits)`` — exactly one gradient step of the
    posterior on the VFE, using CRC-validated ``decoded_bits`` as a supervision
    signal for the bit term (Stage D online adaptation).
  * ``free_energy_series`` — per-slot F recorded during evaluation, used to test
    Hypothesis H2 (F rises at a drift event).

Design notes
------------
* Only the posterior carries trainable parameters; the generative model is pure
  physics (0 params). ``n_parameters`` therefore counts the posterior alone,
  which is what the discriminative-NRX baseline must match (CLAUDE.md rule 7).
* ``adapt_online`` keeps a *persistent* Adam optimiser over the posterior so
  momentum accumulates across slots; the supervised bit term is the binary
  cross-entropy between the bit logits and ``decoded_bits`` — i.e. it replaces
  the uniform-prior bit KL with a data-informed one, while the channel/noise KLs
  and the expected log-likelihood keep the physics consistent.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sionna.phy.fec.ldpc import LDPC5GDecoder

from src.variational_posterior import VariationalPosterior
from src.generative_model import GenerativeModel
from src.vfe import variational_free_energy, VFEOutput
from src.channel.sionna_channel import SionnaChannel


# Hybrid training objective: supervised coded-bit CE + VFE for channel/noise.
HybridOutput = namedtuple(
    "HybridOutput", ["total", "ce", "kl_channel", "kl_noise", "expected_ll"]
)


class AINR(nn.Module):
    """Active Inference Neural Receiver.

    The variational posterior infers the **coded** bits (its logits are coded-bit
    LLRs), the generative model scores reconstruction over coded bits + channel +
    noise, and a standard LDPC belief-propagation decoder turns the inferred coded
    LLRs into information bits — so ``forward`` returns info-bit LLRs ``(B, K)``,
    the common receiver interface.
    """

    def __init__(self, config, sionna_channel: SionnaChannel):
        super().__init__()
        self.config = config
        self.channel = sionna_channel
        self.posterior = VariationalPosterior(config, pilot_grid=sionna_channel.pilot_grid)
        self.generative_model = GenerativeModel(config, sionna_channel)

        # Code dimensions (for reshaping LLRs into the LDPC decoder layout).
        self.n_tx = sionna_channel.n_tx
        self.n = sionna_channel.n                  # coded bits / stream
        self.k = sionna_channel.k                  # info bits / stream
        self.K = sionna_channel.num_info_bits      # total info bits

        # Soft-output BP decoder: coded-bit LLRs -> info-bit LLRs (positive=>1).
        self._ldpc_decoder = LDPC5GDecoder(
            sionna_channel.encoder, hard_out=False, return_infobits=True,
            num_iter=20, device=sionna_channel.device,
        )

        # Per-slot free energy recorded during evaluation (drift monitoring, H2).
        self.free_energy_series: List[float] = []

        # Lazily-built optimiser for online adaptation (Stage D).
        self._adapt_opt: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------ #
    #  Receiver interface
    # ------------------------------------------------------------------ #
    def forward(self, Y: torch.Tensor, record_fe: bool = True) -> torch.Tensor:
        """Main receiver pass.

        Args:
            Y:         received grid, complex ``(B, N_sc, N_sym, N_rx)``.
            record_fe: if True (default), also compute the per-slot free energy
                       and append it to ``free_energy_series`` (no gradient).
                       Set False during a pure BLER sweep to skip the (costly)
                       generative forward.

        Returns:
            info-bit LLRs, shape ``(B, K)`` (positive => bit 1), after LDPC decode.
        """
        coded_llrs = self.posterior.get_llrs(Y)            # (B, n_coded)
        info_llrs = self._ldpc_decode(coded_llrs)          # (B, K)
        if record_fe:
            with torch.no_grad():
                vfe = variational_free_energy(
                    Y, self.posterior, self.generative_model,
                    n_samples=int(self.config.model.n_vfe_samples),
                )
                self.free_energy_series.append(float(vfe.total.item()))
        return info_llrs

    def _ldpc_decode(self, coded_llrs: torch.Tensor) -> torch.Tensor:
        """Coded-bit LLRs (B, n_coded) -> info-bit LLRs (B, K) via BP decoding."""
        B = coded_llrs.shape[0]
        c = coded_llrs.reshape(B, 1, self.n_tx, self.n)
        info = self._ldpc_decoder(c)                       # (B,1,n_tx,k) soft
        return info.reshape(B, self.K)

    def compute_free_energy(
        self,
        Y: torch.Tensor,
        n_samples: Optional[int] = None,
        sigma_override: Optional[torch.Tensor] = None,
    ) -> VFEOutput:
        """Full VFE breakdown (used by the training loop and analysis).

        Returns the :class:`VFEOutput` namedtuple ``(total, expected_ll,
        kl_bits, kl_channel, kl_noise)``; ``total`` is the free energy F.

        ``sigma_override`` (noise std, ``(B,)`` or ``(B,1)``) freezes the noise
        used in the likelihood — see :func:`variational_free_energy` (Stage-A
        warmup anti-collapse).
        """
        if n_samples is None:
            n_samples = int(self.config.model.n_vfe_samples)
        return variational_free_energy(
            Y, self.posterior, self.generative_model, n_samples=n_samples,
            sigma_override=sigma_override,
        )

    @torch.no_grad()
    def reconstruction_error(self, Y: torch.Tensor, n_samples: int = 4) -> float:
        """Mean per-RE generative reconstruction error  E||Y - Y_pred||^2.

        A principled distribution-shift / anomaly signal: the generative model is
        tuned to the training channel statistics, so when the channel distribution
        drifts (e.g. delay spread / model type), its reconstruction of Y degrades
        sharply. Unlike the free energy, it is not masked by the (inflated) noise
        variance. The discriminative receiver has no analogous self-monitoring
        signal. (Doppler-only drift that preserves the per-slot channel structure
        is not captured — see paper discussion.)
        """
        params = self.posterior.forward(Y)
        samples = self.posterior.sample_from_params(params, n_samples)
        y_pred = self.generative_model.predict_grid(samples.bits_soft, samples.h_samples)
        return float(((Y.unsqueeze(0) - y_pred).abs() ** 2).mean().item())

    # ------------------------------------------------------------------ #
    #  Hybrid training objective
    # ------------------------------------------------------------------ #
    def hybrid_objective(
        self,
        Y: torch.Tensor,
        info_bits: torch.Tensor,
        n_samples: Optional[int] = None,
        sigma_override: Optional[torch.Tensor] = None,
        detach_aux: bool = True,
    ) -> HybridOutput:
        """Hybrid loss: supervised coded-bit CE + VFE for channel & noise.

        The coded-bit head is trained by **supervised** cross-entropy against the
        true codeword (the reliably-trainable part). The channel/noise posteriors
        are trained by the VFE reconstruction term, evaluated with **teacher-forced
        true coded bits** so the reconstruction signal for ``(h, sigma^2)`` is clean
        and independent of how well the bit head currently decodes — eliminating the
        bit/channel chicken-and-egg.

            L = CE(coded_logits, true_coded)
              + KL[q(h)||p(h)] + KL[q(sigma^2)||p(sigma^2)]
              - E_{q(h,sigma^2)}[ log p(Y | true_coded, h, sigma^2) ]

        Args:
            Y:          received grid, complex ``(B, N_sc, N_sym, N_rx)``.
            info_bits:  true info bits ``(B, K)`` (training supervision).
            sigma_override: optional frozen noise std for the reconstruction
                        (Stage-A warmup; lets the channel bootstrap).
        """
        if n_samples is None:
            n_samples = int(self.config.model.n_vfe_samples)
        B = Y.shape[0]

        # True codeword (supervision target / teacher for reconstruction).
        with torch.no_grad():
            ib = info_bits.to(device=Y.device, dtype=torch.float32)
            true_coded = self.channel.encoder(
                ib.reshape(B, 1, self.n_tx, self.k)
            ).reshape(B, self.posterior.n_coded)

        # detach_aux: CE owns the backbone; the VFE term trains only the
        # channel/noise heads (decoupled — see VariationalPosterior.forward).
        # Exposed as a flag so the ablation study can disable the decoupling.
        params = self.posterior.forward(Y, detach_aux=detach_aux)

        ce = F.binary_cross_entropy_with_logits(
            params.bit_logits, true_coded, reduction="none"
        ).sum(dim=1).mean()

        kl_h = self.posterior.kl_channel(params)
        kl_n = self.posterior.kl_noise(params)

        samples = self.posterior.sample_from_params(params, n_samples)
        sigma_for_ll = samples.sigma_samples
        if sigma_override is not None:
            so = sigma_override.to(sigma_for_ll.dtype).to(sigma_for_ll.device)
            if so.ndim == 1:
                so = so.unsqueeze(-1)
            sigma_for_ll = so.unsqueeze(0).expand_as(sigma_for_ll)

        coded_tf = true_coded.unsqueeze(0).expand(n_samples, B, self.posterior.n_coded)
        ell = self.generative_model.expected_log_likelihood(
            Y, coded_tf, samples.h_samples, sigma_for_ll
        ).mean()

        total = ce + kl_h + kl_n - ell
        return HybridOutput(
            total=total, ce=ce, kl_channel=kl_h, kl_noise=kl_n, expected_ll=ell
        )

    # ------------------------------------------------------------------ #
    #  Online adaptation (Stage D)
    # ------------------------------------------------------------------ #
    def _ensure_optimizer(self, lr: Optional[float]) -> torch.optim.Optimizer:
        """Create (once) or re-rate the persistent online-adaptation optimiser."""
        if lr is None:
            lr = float(self.config.eval.adaptation_lr)
        if self._adapt_opt is None:
            self._adapt_opt = torch.optim.Adam(self.posterior.parameters(), lr=lr)
        else:
            for group in self._adapt_opt.param_groups:
                group["lr"] = lr
        return self._adapt_opt

    def adapt_online(
        self,
        Y: torch.Tensor,
        decoded_bits: torch.Tensor,
        lr: Optional[float] = None,
    ) -> float:
        """One gradient step of the posterior on the (supervised) VFE.

        Called after a CRC-pass / HARQ-ACK so that ``decoded_bits`` can be trusted
        as a label. Exactly one optimiser step is taken (no accumulation), and
        only the posterior parameters are updated.

        Args:
            Y:            received grid, complex ``(B, N_sc, N_sym, N_rx)``.
            decoded_bits: CRC-validated info bits, ``(B, K)`` in {0, 1}.
            lr:           learning-rate override (defaults to
                          ``config.eval.adaptation_lr``).

        Returns:
            The scalar adaptation loss (float) for logging.
        """
        opt = self._ensure_optimizer(lr)
        n_samples = int(self.config.model.n_vfe_samples)

        # decoded_bits are CRC-validated *info* bits (B, K); re-encode them to the
        # coded bitstream to supervise the (coded-bit) posterior.
        with torch.no_grad():
            di = decoded_bits.to(device=Y.device, dtype=torch.float32)
            B = di.shape[0]
            coded_target = self.channel.encoder(
                di.reshape(B, 1, self.n_tx, self.k)
            ).reshape(B, self.posterior.n_coded)

        self.train()
        opt.zero_grad(set_to_none=True)

        params = self.posterior.forward(Y)

        # Supervised coded-bit term: pull q(c|Y) toward the re-encoded codeword
        # (replaces the uniform-prior bit KL). Sum over bits, mean over batch —
        # same reduction as the closed-form KL terms, so the scales are comparable.
        bit_term = F.binary_cross_entropy_with_logits(
            params.bit_logits, coded_target, reduction="none"
        ).sum(dim=1).mean()

        kl_h = self.posterior.kl_channel(params)
        kl_n = self.posterior.kl_noise(params)

        samples = self.posterior.sample_from_params(params, n_samples)
        ell = self.generative_model.expected_log_likelihood(
            Y, samples.bits_soft, samples.h_samples, samples.sigma_samples
        ).mean()

        loss = bit_term + kl_h + kl_n - ell
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().item())
        self.free_energy_series.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def reset_free_energy_series(self) -> None:
        """Clear the recorded free-energy series (call before each eval run)."""
        self.free_energy_series = []

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters (posterior only; generative model = 0)."""
        return sum(p.numel() for p in self.posterior.parameters())
