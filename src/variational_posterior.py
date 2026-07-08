"""
src/variational_posterior.py

Phase 2 — the inference network q_phi(b, h, sigma^2 | Y).

A residual CNN encoder that consumes the received OFDM grid ``Y`` and emits the
parameters of a factorised variational posterior:

    q_phi(b, h, sigma^2 | Y) = q(b | Y) * q(h | Y) * q(sigma^2 | Y)

with
  * q(b | Y)        : independent Bernoulli over the K information bits
                      (logits == LLRs for the LDPC decoder),
  * q(h | Y)        : circularly-symmetric complex Gaussian over L channel taps
                      per (N_rx, N_tx) antenna pair,
  * q(sigma^2 | Y)  : log-normal over the noise variance, i.e. log sigma^2 is
                      Gaussian with a per-example mean head and a learnable
                      (shared) log-std.

Design notes
------------
* Input ``Y`` is complex ``(B, N_sc, N_sym, N_rx)``; real/imag of each receive
  antenna become separate channels -> ``2 * N_rx`` input channels for the CNN.
* GroupNorm (not BatchNorm) is used so the network behaves identically for the
  batch-size-1 online-adaptation regime (Phase 5 / Stage D).
* The bit head is fully convolutional: it predicts a small number of logits per
  *data* resource element and gathers the data-carrying OFDM symbols, yielding
  exactly ``K`` logits without a giant dense layer. ``K`` and the data-symbol
  layout are derived from ``config`` via :func:`src.utils.infer_dims`, keeping
  this module consistent with :class:`~src.channel.sionna_channel.SionnaChannel`
  even though it only receives ``config``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import infer_dims, data_symbol_indices


# --------------------------------------------------------------------------- #
#  Output containers
# --------------------------------------------------------------------------- #
@dataclass
class PosteriorParams:
    """Parameters of the variational posterior produced by ``forward``."""

    bit_logits: torch.Tensor   # (B, n_coded)            real — CODED-bit LLRs (logit P(coded bit=1))
    h_mean: torch.Tensor       # (B, N_rx, N_tx, L)      complex — posterior mean of taps
    h_logvar: torch.Tensor     # (B, N_rx, N_tx, L)      real — log of (total complex) tap var
    sigma_logvar: torch.Tensor # (B, 1)                  real — posterior mean of log sigma^2


@dataclass
class PosteriorSamples:
    """Reparameterised draws produced by ``sample``."""

    bits_soft: torch.Tensor      # (S, B, n_coded)         in [0,1], straight-through {0,1} (coded bits)
    h_samples: torch.Tensor      # (S, B, N_rx, N_tx, L)   complex
    sigma_samples: torch.Tensor  # (S, B, 1)               noise *std* (sigma, not sigma^2)


# --------------------------------------------------------------------------- #
#  CNN input builder (received grid + optional pilot-based channel estimate)
# --------------------------------------------------------------------------- #
def build_cnn_input(Y: torch.Tensor, pilot_grid=None) -> torch.Tensor:
    """Format the received grid (and an optional pilot reference) for the CNN.

    Args:
        Y:          received grid, complex ``(B, N_sc, N_sym, N_rx)``.
        pilot_grid: known transmitted pilots, complex ``(N_sc, N_sym)`` (zeros off
                    the pilot REs), or None.

    Returns:
        real tensor ``(B, C, N_sc, N_sym)`` with ``C = 2*N_rx`` (Y real/imag),
        and ``+2*N_rx`` more if ``pilot_grid`` is given: a least-squares channel
        estimate ``H_ls = Y * conj(pilot)`` averaged over the pilot symbols and
        broadcast across time (the standard neural-receiver pilot reference, so
        the network can *measure* the channel instead of guessing it).
    """
    if not torch.is_complex(Y):
        raise TypeError("Expected a complex received grid Y.")
    yr = Y.real.permute(0, 3, 1, 2)                     # (B, N_rx, N_sc, N_sym)
    yi = Y.imag.permute(0, 3, 1, 2)
    chans = [yr, yi]
    if pilot_grid is not None:
        pg = pilot_grid.to(Y.device)                   # (N_sc, N_sym) complex
        mask = (pg.abs() > 1e-6).to(Y.real.dtype)      # (N_sc, N_sym)
        h_raw = Y * torch.conj(pg)[None, :, :, None]   # (B, N_sc, N_sym, N_rx)
        denom = mask.sum(dim=1).clamp_min(1.0)         # (N_sc,)
        h_avg = (h_raw * mask[None, :, :, None]).sum(dim=2) / denom[None, :, None]
        h_avg = h_avg[:, :, None, :].expand(-1, -1, Y.shape[2], -1)  # broadcast time
        chans += [h_avg.real.permute(0, 3, 1, 2), h_avg.imag.permute(0, 3, 1, 2)]
    return torch.cat(chans, dim=1).contiguous()        # (B, C, N_sc, N_sym)


# --------------------------------------------------------------------------- #
#  Residual CNN block (GroupNorm, batch-size-1 safe)
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        pad = kernel_size // 2
        groups = math.gcd(n_groups, channels) or 1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


# --------------------------------------------------------------------------- #
#  Variational posterior network
# --------------------------------------------------------------------------- #
class VariationalPosterior(nn.Module):
    LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0  # numerical guards on log-variances

    def __init__(self, config, pilot_grid=None):
        super().__init__()
        self.config = config
        # Pilot reference (optional): doubles the CNN input channels with a
        # least-squares channel estimate so the network can measure the channel.
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
        self.N_tx = dims.n_tx
        self.K = dims.K                                   # info bits (after decode)
        self.n_coded = dims.n_coded * dims.n_tx           # CODED bits == network output
        self.L = int(config.model.n_channel_taps)

        # Data-symbol indices (the OFDM symbols whose REs carry data bits).
        data_syms = data_symbol_indices(config)
        self.register_buffer(
            "data_sym_idx", torch.tensor(data_syms, dtype=torch.long), persistent=False
        )
        self.n_data_sym = len(data_syms)

        # Bit head emits `coded_bits_per_re` logits per data RE; gathering the
        # data symbols yields exactly n_coded LLRs. Coded bits map locally to the
        # QAM symbol on each RE (coded_bits_per_re == nbps for n_tx=1), so this is
        # a physically-grounded, learnable head (unlike inferring info bits).
        num_data_re = self.N_sc * self.n_data_sym
        if self.n_coded % num_data_re != 0:
            raise ValueError(
                f"n_coded={self.n_coded} is not divisible by num_data_re="
                f"{num_data_re}; the convolutional bit head cannot produce "
                "exactly n_coded logits."
            )
        self.coded_bits_per_re = self.n_coded // num_data_re

        # ---- Gumbel-softmax / prior hyper-parameters -------------------
        self.gumbel_temperature = float(config.model.gumbel_temperature)
        self.channel_prior_var = float(config.priors.channel_prior_variance)
        self.noise_prior_mu = float(config.priors.noise_prior_mu)
        self.noise_prior_std = float(config.priors.noise_prior_std)
        # Upper clamp on the noise log-variance (safety rail against the
        # "explain everything as noise" degeneracy). sigma^2 <= exp(this).
        self.noise_logvar_max = float(
            config.priors.get("noise_logvar_max", self.LOGVAR_MAX)
        )

        # ---- Backbone --------------------------------------------------
        F_ch = int(config.model.n_filters)
        ks = int(config.model.kernel_size)
        n_blocks = int(config.model.n_conv_blocks)
        in_ch = 2 * self.N_rx * (2 if self.use_pilot else 1)  # Y (+ pilot LS est.)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, F_ch, ks, padding=ks // 2),
            nn.GroupNorm(math.gcd(8, F_ch) or 1, F_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(F_ch, ks) for _ in range(n_blocks)]
        )

        # ---- Heads -----------------------------------------------------
        # Bit head: per-RE coded-bit logits (fully convolutional).
        self.bit_head = nn.Conv2d(F_ch, self.coded_bits_per_re, kernel_size=1)

        # Channel head: from globally-pooled features to per-tap params.
        n_tap_params = self.N_rx * self.N_tx * self.L
        self.h_mean_real = nn.Linear(F_ch, n_tap_params)
        self.h_mean_imag = nn.Linear(F_ch, n_tap_params)
        self.h_logvar = nn.Linear(F_ch, n_tap_params)

        # Noise head: posterior mean of log sigma^2.
        self.sigma_logvar_head = nn.Linear(F_ch, 1)
        # Shared, learnable posterior log-std of log sigma^2 (init small).
        self.sigma_post_logstd = nn.Parameter(torch.tensor(-1.0))

    # ------------------------------------------------------------------ #
    #  Input formatting
    # ------------------------------------------------------------------ #
    def _backbone(self, Y: torch.Tensor) -> torch.Tensor:
        x = build_cnn_input(Y, self.pilot_grid if self.use_pilot else None)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, F, N_sc, N_sym)

    # ------------------------------------------------------------------ #
    #  Forward: posterior parameters
    # ------------------------------------------------------------------ #
    def forward(self, Y: torch.Tensor, detach_aux: bool = False) -> PosteriorParams:
        """Posterior parameters.

        ``detach_aux``: if True, the channel & noise heads read the backbone
        features *detached*, so the VFE reconstruction gradient trains only those
        small heads — not the shared backbone. This lets the supervised coded-bit
        CE fully own the backbone (clean bit decoding) in the hybrid objective,
        removing the bit/channel gradient conflict on the shared trunk.
        """
        B = Y.shape[0]
        feat = self._backbone(Y)                       # (B, F, N_sc, N_sym)

        # --- Coded-bit logits (gather data-carrying OFDM symbols) ---
        re_logits = self.bit_head(feat)                # (B, cpr, N_sc, N_sym)
        idx = self.data_sym_idx.to(re_logits.device)
        re_logits = re_logits.index_select(3, idx)     # (B, cpr, N_sc, n_data_sym)
        # Reorder to the CODEWORD order so logit i lines up with coded bit i:
        # Sionna places QAM symbols OFDM-symbol-major, subcarrier-minor, and each
        # symbol's nbps bits are consecutive -> (n_data_sym, N_sc, cpr=bitpos).
        re_logits = re_logits.permute(0, 3, 2, 1).contiguous()  # (B, n_data_sym, N_sc, cpr)
        bit_logits = re_logits.reshape(B, -1)          # (B, n_coded), codeword order

        # --- Pooled features for channel & noise heads ---
        pooled = feat.mean(dim=(2, 3))                 # (B, F)
        if detach_aux:
            pooled = pooled.detach()

        h_mean_r = self.h_mean_real(pooled).view(B, self.N_rx, self.N_tx, self.L)
        h_mean_i = self.h_mean_imag(pooled).view(B, self.N_rx, self.N_tx, self.L)
        h_mean = torch.complex(h_mean_r, h_mean_i)
        h_logvar = self.h_logvar(pooled).view(B, self.N_rx, self.N_tx, self.L)
        h_logvar = h_logvar.clamp(self.LOGVAR_MIN, self.LOGVAR_MAX)

        sigma_logvar = self.sigma_logvar_head(pooled)  # (B, 1)
        sigma_logvar = sigma_logvar.clamp(self.LOGVAR_MIN, self.noise_logvar_max)

        return PosteriorParams(
            bit_logits=bit_logits,
            h_mean=h_mean,
            h_logvar=h_logvar,
            sigma_logvar=sigma_logvar,
        )

    # ------------------------------------------------------------------ #
    #  LLRs
    # ------------------------------------------------------------------ #
    def get_llrs(self, Y: torch.Tensor) -> torch.Tensor:
        """Return coded-bit logits (B, n_coded) — the LLRs fed to the LDPC decoder."""
        return self.forward(Y).bit_logits

    # ------------------------------------------------------------------ #
    #  Reparameterised sampling
    # ------------------------------------------------------------------ #
    def sample(self, Y: torch.Tensor, n_samples: int) -> PosteriorSamples:
        params = self.forward(Y)
        return self.sample_from_params(params, n_samples)

    def sample_from_params(
        self, params: PosteriorParams, n_samples: int
    ) -> PosteriorSamples:
        S = int(n_samples)
        z = params.bit_logits                          # (B, K)
        B, K = z.shape
        dev = z.device

        # --- Bits: binary Gumbel-softmax, straight-through hard samples ---
        z_exp = z.unsqueeze(0).expand(S, B, K)         # (S, B, K)
        # Two-class logits: class-1 == "bit is 1".
        logits2 = torch.stack([torch.zeros_like(z_exp), z_exp], dim=-1)  # (S,B,K,2)
        y = F.gumbel_softmax(
            logits2, tau=self.gumbel_temperature, hard=True, dim=-1
        )
        bits_soft = y[..., 1]                          # (S, B, K), {0,1} w/ ST grad

        # --- Channel: reparameterised complex Gaussian ---
        std = torch.exp(0.5 * params.h_logvar)         # per-real/imag std = sqrt(var)/?
        # 'var' here is the TOTAL complex variance; split equally over re/im.
        per_axis_std = std / math.sqrt(2.0)            # (B, N_rx, N_tx, L)
        per_axis_std = per_axis_std.unsqueeze(0)       # (1, B, ...)
        mean = params.h_mean.unsqueeze(0)              # (1, B, ...) complex
        shape = (S,) + tuple(params.h_logvar.shape)
        eps_r = torch.randn(shape, device=dev)
        eps_i = torch.randn(shape, device=dev)
        h_samples = torch.complex(
            mean.real + per_axis_std * eps_r,
            mean.imag + per_axis_std * eps_i,
        )                                              # (S, B, N_rx, N_tx, L)

        # --- Noise: reparameterised log-normal over sigma^2 ---
        mu_q = params.sigma_logvar.unsqueeze(0)        # (1, B, 1)
        s_q = torch.exp(self.sigma_post_logstd)        # scalar
        eps_s = torch.randn((S, B, 1), device=dev)
        u = mu_q + s_q * eps_s                         # log sigma^2  (S, B, 1)
        sigma_samples = torch.exp(0.5 * u)             # sigma (std), (S, B, 1)

        return PosteriorSamples(
            bits_soft=bits_soft,
            h_samples=h_samples,
            sigma_samples=sigma_samples,
        )

    # ------------------------------------------------------------------ #
    #  Closed-form KL terms (scalars: summed over latents, averaged over B)
    # ------------------------------------------------------------------ #
    def kl_bits(self, params: PosteriorParams) -> torch.Tensor:
        """KL[ q(b|Y) || Bernoulli(0.5) ], summed over bits, mean over batch."""
        z = params.bit_logits
        q = torch.sigmoid(z)
        # q*log q + (1-q)*log(1-q) + log 2   (stable via logsigmoid)
        neg_entropy = q * F.logsigmoid(z) + (1.0 - q) * F.logsigmoid(-z)
        kl_per_bit = neg_entropy + math.log(2.0)
        return kl_per_bit.sum(dim=1).mean()

    def kl_channel(self, params: PosteriorParams) -> torch.Tensor:
        """KL[ q(h|Y) || CN(0, vp) ] over taps, mean over batch.

        Per circularly-symmetric complex tap:
            KL = (|mu|^2 + var)/vp - 1 - log(var) + log(vp)
        with var = exp(h_logvar) and vp = channel_prior_variance.
        """
        vp = self.channel_prior_var
        var = torch.exp(params.h_logvar)
        mu2 = params.h_mean.real ** 2 + params.h_mean.imag ** 2
        kl = (mu2 + var) / vp - 1.0 - params.h_logvar + math.log(vp)
        # Sum over (N_rx, N_tx, L); mean over batch.
        return kl.sum(dim=(1, 2, 3)).mean()

    def kl_noise(self, params: PosteriorParams) -> torch.Tensor:
        """KL[ q(log sigma^2) || N(mu0, std0^2) ], mean over batch.

        q(u) = N(mu_q, s_q^2), with mu_q = sigma_logvar and s_q = exp(sigma_post_logstd).
        """
        mu0 = self.noise_prior_mu
        s0 = self.noise_prior_std
        mu_q = params.sigma_logvar                      # (B, 1)
        s_q = torch.exp(self.sigma_post_logstd)         # scalar
        kl = (
            math.log(s0) - torch.log(s_q)
            + (s_q ** 2 + (mu_q - mu0) ** 2) / (2.0 * s0 ** 2)
            - 0.5
        )
        return kl.sum(dim=1).mean()

    # ------------------------------------------------------------------ #
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
