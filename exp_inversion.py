"""Controlled inversion: coded-bit vs information-bit latent, multi-seed, multi-code.

Setting identical for both arms: perfect CSI, true noise variance, no neural
network; free logits optimised by Adam on the exact expected negative
log-likelihood under a factorised Bernoulli posterior. The information-bit arm
passes through the exact soft-XOR relaxation of the (linear) LDPC encoder,
1 - 2P(c_j=1) = prod_{i in S_j} (1 - 2 p_i).

Configurations (all rate-1/2 5G NR LDPC as instantiated by Sionna):
  16QAM-K1824 : 76 subcarriers, 16-QAM   (n = 3648)
  64QAM-K2736 : 76 subcarriers, 64-QAM   (n = 5472)
  16QAM-K576  : 24 subcarriers, 16-QAM   (n = 1152)
Seeds 0-4 (independent received batches and noise).

Outputs: results/ojcoms/inversion_trace.csv, results/ojcoms/inversion_summary.csv
"""
import argparse
import csv
import os
import time

import torch

from exp_common import ROOT, DEV, load_cfg, atomic_write_csv, read_csv_rows
from src.channel.sionna_channel import SionnaChannel
from src.generative_model import GenerativeModel
from sionna.phy.fec.ldpc import LDPC5GDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--inits", nargs="+", default=["zero", "rand", "warm"])
ap.add_argument("--configs", nargs="+", default=None, help="subset of config names")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
if args.quick:
    args.steps, args.seeds = 20, [0]

CONFIGS = [("16QAM-K1824", 4, 76), ("64QAM-K2736", 6, 76), ("16QAM-K576", 4, 24)]
BATCH, SNR, LR, LOG = 8, 10.0, 0.05, 25
out_dir = os.path.join(ROOT, "results", "ojcoms")
os.makedirs(out_dir, exist_ok=True)


def run(cfg_name, nbps, nsc, seed, init_mode):
    cfg = load_cfg(16)
    cfg.phy.n_bits_per_symbol = nbps
    cfg.phy.n_subcarriers = nsc
    torch.manual_seed(1000 + seed)
    ch = SionnaChannel(cfg, device=DEV)
    gen = GenerativeModel(cfg, ch)
    b = ch.generate_batch(BATCH, snr_db=SNR)
    Y, h_freq = b.received_grid, b.h_freq.contiguous()
    B = Y.shape[0]
    sigma2 = b.no.reshape(-1)[:B].float().to(DEV).reshape(B, 1)
    info_true = b.bits.to(DEV)
    n_coded, K = ch.n * ch.n_tx, ch.num_info_bits
    with torch.no_grad():
        coded_true = ch.encoder(info_true.reshape(B, 1, ch.n_tx, ch.k)).reshape(B, n_coded)
        rows_g = []
        for s in range(0, K, 256):
            e = torch.zeros(min(256, K - s), 1, ch.n_tx, ch.k, device=DEV)
            e[torch.arange(e.shape[0]), 0, 0, s + torch.arange(e.shape[0])] = 1.0
            rows_g.append(ch.encoder(e).reshape(e.shape[0], n_coded))
        G = torch.cat(rows_g)
        bt = (torch.rand(1, 1, ch.n_tx, ch.k, device=DEV) > 0.5).float()
        if not torch.equal(ch.encoder(bt).reshape(-1), (bt.reshape(-1) @ G) % 2.0):
            raise RuntimeError("encoder not linear over GF(2); generator invalid")
    colw = G.sum(0)

    def soft_encode(p):
        m = 1.0 - 2.0 * p
        logmag = torch.log(m.abs().clamp_min(1e-12)) @ G
        with torch.no_grad():
            sign = 1.0 - 2.0 * (((m < 0).float() @ G) % 2.0)
        return ((1.0 - sign * torch.exp(logmag)) / 2.0).clamp(1e-6, 1 - 1e-6)

    qam_abs2 = gen.qam_points.real ** 2 + gen.qam_points.imag ** 2
    pmask = (ch.pilot_grid.abs() > 1e-6)
    h_abs2 = (b.channel_response.real ** 2 + b.channel_response.imag ** 2)[:, :, 0]

    def nll(cs):
        c = cs.reshape(1, B, gen.N_tx, gen.n_data, gen.nbps).unsqueeze(-2)
        tbl = gen.qam_table.view(1, 1, 1, 1, gen.M, gen.nbps)
        probs = (tbl * c + (1 - tbl) * (1 - c)).prod(-1)
        Es = torch.einsum("...m,m->...", probs.to(gen.qam_points.dtype), gen.qam_points)
        Es2 = torch.einsum("...m,m->...", probs, qam_abs2)
        var_s = (Es2 - Es.abs() ** 2).clamp_min(0.0)
        ym = ch._received_to_claude_layout(
            ch.apply_channel(ch.rg_mapper(Es.reshape(B, gen.N_tx, gen.n_data).unsqueeze(1)), h_freq, None))
        sq = ((Y - ym).abs() ** 2).sum(dim=(1, 2, 3))
        vg = ch.rg_mapper(var_s.reshape(B, gen.N_tx, gen.n_data).unsqueeze(1).to(gen.qam_points.dtype))
        vg = ch._received_to_claude_layout(
            vg.reshape(B, 1, 1, gen.N_sym, gen.N_sc).expand(B, 1, gen.N_rx, gen.N_sym, gen.N_sc)).real
        vg = vg * (~pmask).float()[None, :, :, None]
        vt = (vg.permute(0, 3, 1, 2) * h_abs2).sum(dim=(1, 2, 3))
        return ((sq + vt) / sigma2.squeeze(-1)).sum()

    # Initialisation (identical rule for both arms, applied to each arm's own latent):
    #   zero : logits 0 (p = 1/2). Exact stationary point of the soft-XOR map
    #          (every factor 1-2p = 0), so the info-bit gradient is 0 by construction.
    #   rand : logits ~ N(0, 1), uninformative but off the saddle.
    #   warm : logits = +-1 with the sign of the true bit flipped w.p. 0.3
    #          (a partially informative start, e.g. from a weak detector).
    g = torch.Generator(device=DEV).manual_seed(5000 + seed)

    def init(truth):
        if init_mode == "zero":
            return torch.zeros_like(truth)
        if init_mode == "rand":
            return torch.randn(truth.shape, generator=g, device=DEV)
        flip = (torch.rand(truth.shape, generator=g, device=DEV) < 0.3).float()
        return 2.0 * ((truth + flip) % 2.0) - 1.0

    z = init(coded_true).clone().requires_grad_(True)
    u = init(info_true).clone().requires_grad_(True)
    with torch.no_grad():
        init_coded_ber = ((z > 0).float() != coded_true).float().mean().item()
        init_info_ber = ((u > 0).float() != info_true).float().mean().item()
        init_soft_coded_ber = ((soft_encode(torch.sigmoid(u)) > 0.5).float()
                               != coded_true).float().mean().item()
    oz, ou = torch.optim.Adam([z], lr=LR), torch.optim.Adam([u], lr=LR)
    trace = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        oz.zero_grad(set_to_none=True); nll(torch.sigmoid(z)).backward()
        gz = z.grad.abs().mean().item(); oz.step()
        ou.zero_grad(set_to_none=True); nll(soft_encode(torch.sigmoid(u))).backward()
        gu = u.grad.abs().mean().item(); ou.step()
        if step == 1 or step % LOG == 0:
            with torch.no_grad():
                trace.append({"config": cfg_name, "init": init_mode, "seed": seed, "step": step,
                              "coded_ber": ((z > 0).float() != coded_true).float().mean().item(),
                              "info_ber": ((u > 0).float() != info_true).float().mean().item(),
                              "grad_coded": gz, "grad_info": gu})
    with torch.no_grad():
        dec = LDPC5GDecoder(ch.encoder, hard_out=True, return_infobits=True, num_iter=20, device=DEV)
        ih = dec(z.reshape(B, 1, ch.n_tx, ch.n)).reshape(B, K)
    summ = {"config": cfg_name, "init": init_mode, "seed": seed, "K": K, "n": n_coded,
            "colw_mean": colw.mean().item(), "colw_min": colw.min().item(), "colw_max": colw.max().item(),
            "init_coded_ber": init_coded_ber, "init_info_ber": init_info_ber,
            "init_info_implied_coded_ber": init_soft_coded_ber,
            "final_coded_ber": trace[-1]["coded_ber"], "final_info_ber": trace[-1]["info_ber"],
            "post_bp_info_ber": (ih != info_true).float().mean().item(),
            "post_bp_bler": (ih != info_true).any(1).float().mean().item(),
            "final_grad_coded": trace[-1]["grad_coded"], "final_grad_info": trace[-1]["grad_info"],
            "seconds": time.time() - t0}
    print(f"{cfg_name} {init_mode:4s} seed {seed}: coded BER {summ['final_coded_ber']:.4f} -> BP BLER "
          f"{summ['post_bp_bler']:.3f} | info BER {summ['final_info_ber']:.4f} "
          f"| grad info {summ['final_grad_info']:.2e} | colw {summ['colw_mean']:.1f} "
          f"({summ['seconds']:.0f}s)", flush=True)
    return trace, summ


part_dir = os.path.join(out_dir, "inversion_parts")
os.makedirs(part_dir, exist_ok=True)
traces, summs = [], []
for name, nbps, nsc in CONFIGS:
    if args.configs and name not in args.configs:
        continue
    for init_mode in args.inits:
        for s in args.seeds:
            # Resumable: every (code, init, seed) run is stored as its own pair of parts.
            pt = os.path.join(part_dir, f"{name}_{init_mode}_s{s}_trace.csv")
            ps = os.path.join(part_dir, f"{name}_{init_mode}_s{s}_summary.csv")
            if not args.quick and os.path.exists(pt) and os.path.exists(ps):
                traces += read_csv_rows(pt, text_cols=("config", "init"))
                summs += read_csv_rows(ps, text_cols=("config", "init"))
                print(f"{name} {init_mode} seed {s}: loaded from parts", flush=True)
                continue
            tr, sm = run(name, nbps, nsc, s, init_mode)
            if not args.quick:
                atomic_write_csv(pt, tr)
                atomic_write_csv(ps, [sm])
            traces += tr; summs.append(sm)
            torch.cuda.empty_cache()
sfx = "_quick" if args.quick else ""
for fn, rows in ((f"inversion_trace{sfx}.csv", traces), (f"inversion_summary{sfx}.csv", summs)):
    atomic_write_csv(os.path.join(out_dir, fn), rows)
print("INVERSION_DONE", flush=True)
