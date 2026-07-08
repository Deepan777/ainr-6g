"""
src/vfe.py

Phase 4 — the variational free energy (VFE) objective.

The AINR receiver is trained by minimising the per-example variational free
energy (a.k.a. negative ELBO):

    F = KL[q(b|Y) || p(b)]              (Bernoulli, closed form)
      + KL[q(h|Y) || p(h)]              (complex Gaussian, closed form)
      + KL[q(sigma^2|Y) || p(sigma^2)]  (log-normal, closed form)
      - E_q[ log p(Y | b, h, sigma^2) ] (Monte-Carlo, reparameterised)

with the priors

    p(b)        = Bernoulli(0.5)                    (equiprobable code bits)
    p(h)        = CN(0, channel_prior_variance * I) (zero-mean complex Gaussian)
    p(sigma^2)  = LogNormal(noise_prior_mu, noise_prior_std^2)

All three KL divergences have closed forms and are computed *exactly* by the
posterior network (:class:`~src.variational_posterior.VariationalPosterior`),
so we never Monte-Carlo them. Only the expected log-likelihood is estimated with
``n_samples`` reparameterised draws, so gradients flow into the posterior
parameters through both the (straight-through Gumbel) bit samples and the
(reparameterised) channel / noise samples.

Sign convention: ``total`` is the free energy F to be **minimised**. Each KL
term and the expected log-likelihood are reported separately so the training
loop can log every component (see CLAUDE.md Phase 7).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import torch


VFEOutput = namedtuple(
    "VFEOutput",
    ["total", "expected_ll", "kl_bits", "kl_channel", "kl_noise"],
)


def variational_free_energy(
    Y: torch.Tensor,
    posterior,
    generative_model,
    n_samples: Optional[int] = None,
    priors=None,
    sigma_override: Optional[torch.Tensor] = None,
) -> VFEOutput:
    """Compute the variational free energy F and its components.

    Args:
        Y:                received OFDM grid, complex ``(B, N_sc, N_sym, N_rx)``.
        posterior:        a :class:`VariationalPosterior` — supplies the posterior
                          parameters, reparameterised samples, and the closed-form
                          KL terms.
        generative_model: a :class:`GenerativeModel` — supplies
                          ``log p(Y | b, h, sigma^2)``.
        n_samples:        number of Monte-Carlo samples for the expected
                          log-likelihood. Defaults to ``config.model.n_vfe_samples``.
        priors:           accepted for API symmetry but unused — the prior
                          hyper-parameters live in the posterior's ``config``.
        sigma_override:   optional noise **std** to use in the likelihood instead
                          of the sampled ``sigma`` (shape ``(B,)`` or ``(B,1)``).
                          Used during the Stage-A warmup to *freeze* sigma^2 to the
                          known noise so the channel/bit posteriors must explain the
                          signal (rather than the model attributing everything to an
                          inflated noise variance). ``kl_noise`` is still computed
                          from the posterior, so the noise head is still regularised;
                          only the reconstruction term uses the override.

    Returns:
        :class:`VFEOutput` ``(total, expected_ll, kl_bits, kl_channel, kl_noise)``
        — all scalar tensors. ``total`` is differentiable w.r.t. the posterior
        parameters.
    """
    if n_samples is None:
        n_samples = int(posterior.config.model.n_vfe_samples)

    # Single forward pass -> posterior parameters, reused for both the
    # closed-form KLs and the reparameterised sampling (avoids a second pass).
    params = posterior.forward(Y)

    # --- Closed-form KL terms (per-example, averaged over the batch) ---
    kl_b = posterior.kl_bits(params)
    kl_h = posterior.kl_channel(params)
    kl_n = posterior.kl_noise(params)

    # --- Monte-Carlo expected log-likelihood E_q[log p(Y|b,h,sigma^2)] ---
    samples = posterior.sample_from_params(params, n_samples)
    sigma_for_ll = samples.sigma_samples                       # (S, B, 1) noise std
    if sigma_override is not None:
        so = sigma_override.to(sigma_for_ll.dtype).to(sigma_for_ll.device)
        if so.ndim == 1:
            so = so.unsqueeze(-1)                              # (B,) -> (B,1)
        sigma_for_ll = so.unsqueeze(0).expand_as(sigma_for_ll)  # (S, B, 1)
    ell = generative_model.expected_log_likelihood(
        Y, samples.bits_soft, samples.h_samples, sigma_for_ll
    )                                   # (B,) — per-example, averaged over samples
    expected_ll = ell.mean()            # average over the batch -> scalar

    total = kl_b + kl_h + kl_n - expected_ll

    return VFEOutput(
        total=total,
        expected_ll=expected_ll,
        kl_bits=kl_b,
        kl_channel=kl_h,
        kl_noise=kl_n,
    )
