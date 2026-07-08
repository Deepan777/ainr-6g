"""Render all figures for the AINR_IEEE_Access_Paper manuscript (300-dpi PNG).

Six result figures come straight from the verified result CSVs in results/;
two design figures (architecture schematic, training convergence) are drawn
from the source layout and the real training log. Nothing is fabricated.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 1.8, "lines.markersize": 6, "figure.dpi": 300,
})
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "AINR_IEEE_Access_Paper", "figures")
os.makedirs(OUT, exist_ok=True)
FLOOR = 5e-4
MK = {"AINR": "o", "DiscNRX": "s", "LMMSE": "^"}
CO = {"AINR": "#1f77b4", "DiscNRX": "#d62728", "LMMSE": "#2ca02c"}
LB = {"AINR": "AINR (proposed)", "DiscNRX": "Discriminative NRX", "LMMSE": "LMMSE"}


def rd(p):
    with open(os.path.join(ROOT, p)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig)
    print("->", name)


# ---- Fig: system architecture schematic -----------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 60)

    def box(x, y, w, h, text, fc="#eaf2fb", ec="#1f77b4", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.2, color="#444"))

    box(1, 26, 14, 10, "Received grid\n$\\mathbf{Y}$ + pilots", fc="#f5f5f5", ec="#666")
    box(18, 26, 13, 10, "LS pilot\nestimate\n$\\hat{\\mathbf{h}}_{LS}$")
    box(34, 22, 17, 18, "Residual CNN\nencoder $f_\\phi$\n(4 blocks,\nGroupNorm)", fc="#dcecfb")
    # three heads
    box(55, 44, 17, 9, "Bit head\n$q(\\mathbf{c})$", fc="#fde8e8", ec="#d62728")
    box(55, 30, 17, 9, "Channel head\n$q(\\mathbf{h})$")
    box(55, 16, 17, 9, "Noise head\n$q(\\sigma^2)$")
    # downstream
    box(76, 44, 22, 9, "LDPC BP decode\n$\\rightarrow$ bits + CRC", fc="#fde8e8", ec="#d62728")
    box(76, 23, 22, 12, "Generative model\n$\\mathcal{G}(\\mathbf{c},\\mathbf{h})$\n$\\rightarrow$ VFE / residual")

    arrow(15, 31, 18, 31)
    arrow(31, 31, 34, 31)
    arrow(15, 31, 34, 33)   # Y also straight to encoder
    arrow(51, 35, 55, 48); arrow(51, 31, 55, 34); arrow(51, 28, 55, 21)
    arrow(72, 48, 76, 48)
    arrow(72, 34, 76, 31); arrow(72, 20, 76, 27)
    # feedback (adaptation + drift)
    ax.add_patch(FancyArrowPatch((87, 23), (87, 14), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.2, color="#888", ls="--"))
    ax.text(87, 11, "drift residual $\\mathcal{E}$ / CRC-gated adaptation",
            ha="center", va="center", fontsize=8, color="#555")
    ax.text(64, 56, "Variational posterior $q_\\phi(\\mathbf{c},\\mathbf{h},\\sigma^2\\mid\\mathbf{Y})$",
            ha="center", fontsize=9, style="italic")
    save(fig, "fig_architecture.png")


# ---- Fig: training convergence (real log) ---------------------------------
def fig_training():
    rows = rd("results/ainr_training_log.csv")
    step = np.array([float(r["step"]) for r in rows])
    ce = np.array([float(r["ce"]) for r in rows])
    klc = np.array([float(r["kl_channel"]) for r in rows])
    kln = np.array([float(r["kl_noise"]) for r in rows])
    nll = -np.array([float(r["expected_ll"]) for r in rows])  # negative expected LL
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(step, ce, label="Coded-bit CE", color="#d62728")
    ax.plot(step, klc, label=r"KL channel $q(\mathbf{h})\|p$", color="#1f77b4")
    ax.plot(step, kln, label=r"KL noise $q(\sigma^2)\|p$", color="#2ca02c")
    ax.plot(step, np.clip(nll, 1, None), label=r"$-\mathbb{E}_q[\log p(\mathbf{Y}|\cdot)]$",
            color="#9467bd", ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Training step"); ax.set_ylabel("Objective component")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(loc="upper right", ncol=1)
    save(fig, "fig_training.png")


def fig_matched_ci():
    rows = rd("results/stage_b_ci.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for rx in ["LMMSE", "DiscNRX", "AINR"]:
        rs = [r for r in rows if r["receiver"] == rx]
        x = [float(r["snr_db"]) for r in rs]
        y = np.array([max(float(r["bler_mean"]), FLOOR) for r in rs])
        e = np.array([float(r["bler_std"]) for r in rs])
        ax.errorbar(x, y, yerr=e, marker=MK[rx], color=CO[rx], capsize=3, label=LB[rx])
    ax.set_yscale("log"); ax.set_ylim(FLOOR, 1.5); ax.set_xlim(-2.5, 6.5)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BLER")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(loc="lower left")
    save(fig, "fig_matched_ci.png")


def fig_64qam():
    rows = rd("results/qam64/stage_b.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for rx in ["LMMSE", "DiscNRX", "AINR"]:
        rs = [r for r in rows if r["receiver"] == rx]
        x = [float(r["snr_db"]) for r in rs]
        y = [max(float(r["bler"]), FLOOR) for r in rs]
        ax.plot(x, y, marker=MK[rx], color=CO[rx], label=LB[rx])
    ax.set_yscale("log"); ax.set_ylim(FLOOR, 1.5); ax.set_xlim(-0.5, 13)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BLER")
    ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(loc="lower left")
    save(fig, "fig_64qam.png")


def fig_roc():
    rows = rd("results/roc_data.csv")
    scen = {"s2_delay_shift": "S2: CDL-A delay drift",
            "s3_speed_shift": "S3: 90 km/h Doppler drift"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, (sk, title) in zip(axes, scen.items()):
        rs = [r for r in rows if r["scenario"] == sk]
        fpr = [float(r["fpr"]) for r in rs]; tpr = [float(r["tpr"]) for r in rs]
        auc = rs[0]["auc"]
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AINR recon-error (AUC={float(auc):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance (AUC=0.500)")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(True, ls=":", alpha=0.5); ax.legend(loc="lower right")
    save(fig, "fig_roc.png")


def fig_drift_timeline():
    rows = rd("results/stage_c_drift_timeline.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = {"S2": "#1f77b4", "S3": "#ff7f0e"}
    drift_slot = None
    for sk in ["S2", "S3"]:
        rs = [r for r in rows if r["scenario"] == sk]
        slot = [int(r["slot"]) for r in rs]; re = [float(r["recon_error"]) for r in rs]
        ax.plot(slot, re, color=colors[sk], lw=1.3, alpha=0.9, label=f"{sk} reconstruction error")
        for r in rs:
            if r["phase"] != "matched":
                drift_slot = int(r["slot"]); break
    if drift_slot is not None:
        ax.axvline(drift_slot, color="k", ls="--", lw=1.2, alpha=0.7)
        ax.text(drift_slot + 4, ax.get_ylim()[1] * 0.92, "drift onset", fontsize=8.5)
    ax.set_xlabel("Slot index"); ax.set_ylabel(r"Reconstruction error $\;\mathbb{E}\,\|Y-\hat Y\|^2$")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(loc="upper left")
    save(fig, "fig_drift_timeline.png")


def fig_s4():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, (csvp, title) in zip(
            axes, [("results/stage_s4_qam16.csv", "S4 ray-traced (Munich), 16-QAM"),
                   ("results/stage_s4_qam64.csv", "S4 ray-traced (Munich), 64-QAM")]):
        rows = rd(csvp)
        for rx in ["LMMSE", "DiscNRX", "AINR"]:
            rs = [r for r in rows if r["receiver"] == rx]
            x = [float(r["snr_db"]) for r in rs]; y = [max(float(r["bler"]), FLOOR) for r in rs]
            ax.plot(x, y, marker=MK[rx], color=CO[rx], label=LB[rx])
        ax.set_yscale("log"); ax.set_ylim(FLOOR, 1.5)
        ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BLER"); ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(loc="lower left")
    save(fig, "fig_s4.png")


def fig_stage_d():
    rows = rd("results/stage_d.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); win = 25
    for rx in ["AINR", "DiscNRX"]:
        rs = [r for r in rows if r["receiver"] == rx]
        slot = np.array([int(r["slot"]) for r in rs]); bler = np.array([float(r["bler"]) for r in rs])
        roll = np.convolve(bler, np.ones(win) / win, mode="valid")
        ax.plot(slot[win - 1:], roll, color=CO[rx], lw=1.8, label=f"{LB[rx]} (rolling BLER, w={win})")
    ax.set_xlabel("Slot index (post-drift, online adaptation active)")
    ax.set_ylabel("Block error rate"); ax.set_ylim(-0.03, 1.05)
    ax.grid(True, ls=":", alpha=0.5); ax.legend(loc="center right")
    save(fig, "fig_stage_d.png")


if __name__ == "__main__":
    fig_architecture(); fig_training()
    fig_matched_ci(); fig_64qam(); fig_roc()
    fig_drift_timeline(); fig_s4(); fig_stage_d()
    print("ALL FIGURES DONE")
