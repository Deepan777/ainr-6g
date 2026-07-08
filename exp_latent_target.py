"""Controlled inversion experiment: coded-bit vs information-bit latent target.

Isolates the well-posedness claim of the paper (Sec. "Coded-Bit Inference"):
with the channel known exactly (perfect CSI) and the true noise level given,
can gradient descent on the *bit* latent alone invert the generative model?

  * coded-bit latent  — free logits over the n coded bits; the QAM map is local
    and smooth, so descent should recover the codeword (BER -> ~0).
  * info-bit latent   — free logits over the K information bits, pushed through
    a differentiable relaxation of the LDPC encoder (exact soft-XOR product
    form over the generator matrix G).  Each coded bit mixes hundreds of
    information bits, so the relaxed encoder output saturates at 1/2 and the
    gradient vanishes: descent should stay at chance (BER ~ 0.5).

Both problems share the same received grids, the same perfect CSI, the same
true noise level, the same optimizer, and the same step budget: the only
difference is the latent parameterization.

Outputs: results/latent_target.csv  (step, coded_ber, info_ber)
         results/latent_target_summary.csv

Usage:  python exp_latent_target.py [--quick]
"""
import argparse
import csv
import os
import sys

import torch
from omegaconf import OmegaConf

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from src.channel.sionna_channel import SionnaChannel
from src.generative_model import GenerativeModel

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()

cfg = OmegaConf.load(os.path.join(_ROOT, "config", "config.yaml"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(7)

STEPS = 20 if args.quick else 1500
LOG_EVERY = 5 if args.quick else 25
BATCH = 8
SNR = 10.0
LR = 0.05

ch = SionnaChannel(cfg, device=dev)
gen = GenerativeModel(cfg, ch)
batch = ch.generate_batch(BATCH, snr_db=SNR)
Y = batch.received_grid                                # (B, N_sc, N_sym, N_rx)
B = Y.shape[0]

# Perfect CSI: the batch keeps the channel in native Sionna layout
# (B, num_rx, N_rx, num_tx, N_tx, N_sym, N_sc) — exactly what predict_grid wants.
h_freq = batch.h_freq.contiguous()

# True noise std, shape (S=1, B, 1).
no = batch.no.reshape(-1)[:B].float()
sigma = no.sqrt().reshape(1, B, 1).to(dev)

# True bits for scoring.
info_true = batch.bits.to(dev)                         # (B, K) in {0,1}
with torch.no_grad():
    coded_true = ch.encoder(
        info_true.reshape(B, 1, ch.n_tx, ch.k)
    ).reshape(B, ch.n * ch.n_tx)

n_coded = ch.n * ch.n_tx
K = ch.num_info_bits


# --------------------------------------------------------------------------- #
#  Generator matrix of the (linear) LDPC encoder: G[i] = encode(e_i).
# --------------------------------------------------------------------------- #
def build_generator():
    rows = []
    chunk = 256
    with torch.no_grad():
        for s in range(0, K, chunk):
            e = torch.zeros(min(chunk, K - s), 1, ch.n_tx, ch.k, device=dev)
            for j in range(e.shape[0]):
                e[j, 0, 0, s + j] = 1.0
            rows.append(ch.encoder(e).reshape(e.shape[0], n_coded))
    G = torch.cat(rows, dim=0)                          # (K, n_coded) in {0,1}
    # Sanity: the encoder must be linear over GF(2) for G to be exact.
    with torch.no_grad():
        z = ch.encoder(torch.zeros(1, 1, ch.n_tx, ch.k, device=dev)).abs().sum()
        if z.item() > 0:
            raise RuntimeError("encode(0) != 0 — encoder not linear, G invalid")
        for _ in range(3):
            b = (torch.rand(1, 1, ch.n_tx, ch.k, device=dev) > 0.5).float()
            c_direct = ch.encoder(b).reshape(-1)
            c_G = (b.reshape(-1) @ G) % 2.0
            if not torch.equal(c_direct, c_G):
                raise RuntimeError("superposition check failed — G invalid")
    return G


print("Building LDPC generator matrix G ...", flush=True)
G = build_generator()
col_weight = G.sum(dim=0)
print(f"G: {tuple(G.shape)}, mean column weight = {col_weight.mean():.1f} "
      f"(max {int(col_weight.max())})", flush=True)


def soft_encode(p_info):
    """Exact soft-XOR LDPC encoding of independent info-bit probabilities.

    For c_j = XOR of info bits in the support of G[:, j]:
        1 - 2 P(c_j = 1) = prod_i (1 - 2 p_i)^{G_ij}.
    Computed in sign/log-magnitude form; the sign path carries no gradient
    (it is piecewise constant), the magnitude path is differentiable.
    """
    m = 1.0 - 2.0 * p_info                              # (B, K) in (-1, 1)
    logmag = torch.log(m.abs().clamp_min(1e-12)) @ G    # (B, n_coded)
    with torch.no_grad():
        neg = (m < 0).float() @ G                       # count of negative factors
        sign = 1.0 - 2.0 * (neg % 2.0)                  # (+1 / -1), no grad
    m_c = sign * torch.exp(logmag)
    return ((1.0 - m_c) / 2.0).clamp(1e-6, 1.0 - 1e-6)  # P(c_j = 1)


# --------------------------------------------------------------------------- #
#  Exact expected negative log-likelihood under independent soft bits.
#
#  E_q[|y - h s|^2] = |y - h E[s]|^2 + |h|^2 Var[s]; the variance term makes the
#  objective vertex-seeking (the deterministic mean-symbol relaxation is flat
#  along bit mixtures that share the same expected symbol).  This is the exact
#  closed form of the expectation the AINR estimates by posterior sampling.
# --------------------------------------------------------------------------- #
qam_abs2 = (gen.qam_points.real ** 2 + gen.qam_points.imag ** 2)     # (M,)
pilot_mask = (ch.pilot_grid.abs() > 1e-6)                            # (N_sc, N_sym)
h_abs2 = (batch.channel_response.real ** 2
          + batch.channel_response.imag ** 2)[:, :, 0]               # (B,N_rx,N_sc,N_sym)
sigma2 = (sigma.reshape(B, 1) ** 2)                                  # (B,1)


def symbol_moments(coded_soft):
    """Per-symbol constellation probabilities -> E[s], E[|s|^2]."""
    c = coded_soft.reshape(1, B, gen.N_tx, gen.n_data, gen.nbps).unsqueeze(-2)
    tbl = gen.qam_table.view(1, 1, 1, 1, gen.M, gen.nbps)
    probs = (tbl * c + (1.0 - tbl) * (1.0 - c)).prod(dim=-1)         # (1,B,ntx,nd,M)
    Es = torch.einsum("...m,m->...", probs.to(gen.qam_points.dtype),
                      gen.qam_points)                                # (1,B,ntx,nd)
    Es2 = torch.einsum("...m,m->...", probs, qam_abs2)               # (1,B,ntx,nd)
    return Es, Es2


def neg_log_lik(coded_soft):
    """Exact E_q[-log p(Y | c, h_true, sigma_true)], summed over the batch."""
    Es, Es2 = symbol_moments(coded_soft)
    var_s = (Es2 - (Es.real ** 2 + Es.imag ** 2)).clamp_min(0.0)     # (1,B,ntx,nd)
    # Mean reconstruction through the exact forward operators (pilots inserted).
    mean_grid = ch.rg_mapper(Es.reshape(B, gen.N_tx, gen.n_data).unsqueeze(1))
    y_mean = ch.apply_channel(mean_grid, h_freq, None)
    y_mean = ch._received_to_claude_layout(y_mean)                   # (B,N_sc,N_sym,N_rx)
    sq_mean = ((Y - y_mean).real ** 2 + (Y - y_mean).imag ** 2).sum(dim=(1, 2, 3))
    # Variance term: place Var[s] on the grid (zero at pilot REs), weight by |h|^2.
    var_grid = ch.rg_mapper(
        var_s.reshape(B, gen.N_tx, gen.n_data).unsqueeze(1).to(gen.qam_points.dtype))
    var_grid = ch._received_to_claude_layout(
        var_grid.reshape(B, 1, 1, gen.N_sym, gen.N_sc).expand(
            B, 1, gen.N_rx, gen.N_sym, gen.N_sc)).real               # (B,N_sc,N_sym,N_rx)
    var_grid = var_grid * (~pilot_mask).float()[None, :, :, None]
    var_term = (var_grid.permute(0, 3, 1, 2) * h_abs2).sum(dim=(1, 2, 3))
    return ((sq_mean + var_term) / sigma2.squeeze(-1)).sum()


# --------------------------------------------------------------------------- #
#  Two optimizations, identical in everything but the latent.
# --------------------------------------------------------------------------- #
z_coded = torch.zeros(B, n_coded, device=dev, requires_grad=True)
u_info = torch.zeros(B, K, device=dev, requires_grad=True)
opt_c = torch.optim.Adam([z_coded], lr=LR)
opt_i = torch.optim.Adam([u_info], lr=LR)

rows = []
for step in range(1, STEPS + 1):
    opt_c.zero_grad(set_to_none=True)
    loss_c = neg_log_lik(torch.sigmoid(z_coded))
    loss_c.backward(); opt_c.step()

    opt_i.zero_grad(set_to_none=True)
    loss_i = neg_log_lik(soft_encode(torch.sigmoid(u_info)))
    loss_i.backward(); opt_i.step()

    if step % LOG_EVERY == 0 or step == 1:
        with torch.no_grad():
            ber_c = ((z_coded > 0).float() != coded_true).float().mean().item()
            ber_i = ((u_info > 0).float() != info_true).float().mean().item()
        rows.append({"step": step, "coded_ber": ber_c, "info_ber": ber_i})
        if step % (LOG_EVERY * 10) == 0 or step == 1 or step == STEPS:
            print(f"step {step:5d} | coded-latent BER={ber_c:.4f} | "
                  f"info-latent BER={ber_i:.4f}", flush=True)

# BP decode of the final coded-latent LLRs -> information-bit error rate.
from sionna.phy.fec.ldpc import LDPC5GDecoder
with torch.no_grad():
    dec = LDPC5GDecoder(ch.encoder, hard_out=True, return_infobits=True,
                        num_iter=20, device=dev)
    info_hat = dec(z_coded.reshape(B, 1, ch.n_tx, ch.n)).reshape(B, K)
    ber_post_bp = float((info_hat != info_true).float().mean().item())
    bler_post_bp = float((info_hat != info_true).any(dim=1).float().mean().item())
print(f"coded latent after BP decode: info BER={ber_post_bp:.5f} "
      f"BLER={bler_post_bp:.3f}", flush=True)

# Gradient-magnitude diagnostic at the final point (the mechanism).
opt_c.zero_grad(set_to_none=True)
neg_log_lik(torch.sigmoid(z_coded)).backward()
g_c = z_coded.grad.abs().mean().item()
opt_i.zero_grad(set_to_none=True)
neg_log_lik(soft_encode(torch.sigmoid(u_info))).backward()
g_i = u_info.grad.abs().mean().item()

os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
with open(os.path.join(_ROOT, "results", "latent_target.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step", "coded_ber", "info_ber"])
    w.writeheader(); w.writerows(rows)
with open(os.path.join(_ROOT, "results", "latent_target_summary.csv"), "w",
          newline="") as f:
    w = csv.writer(f)
    w.writerow(["latent", "final_ber", "mean_abs_grad", "n_latents",
                "mean_col_weight", "post_bp_ber", "post_bp_bler"])
    w.writerow(["coded", rows[-1]["coded_ber"], g_c, n_coded, 1,
                ber_post_bp, bler_post_bp])
    w.writerow(["info", rows[-1]["info_ber"], g_i, K,
                float(col_weight.mean().item()), "", ""])
print(f"\nfinal: coded BER={rows[-1]['coded_ber']:.4f} (grad {g_c:.2e}) | "
      f"info BER={rows[-1]['info_ber']:.4f} (grad {g_i:.2e})")
print("-> results/latent_target.csv")
print("LATENT_DONE", flush=True)
