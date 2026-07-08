"""Render the new/updated result figures for the Elsevier manuscript (300 dpi).

New:      fig_latent.png     — controlled inversion, coded vs info-bit latent
Updated:  fig_stage_d.png    — multi-seed adaptation with mean +/- std band,
                               plus ungated and no-adaptation references
          fig_s4.png         — S4 BLER with error bars (5 passes)
          fig_roc.png        — ROC from pooled multi-run scores, AUC +/- std

All data come from the verified CSVs written by the exp_*.py scripts.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 1.8, "lines.markersize": 6, "figure.dpi": 300,
})
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Elsevier_Manuscript", "figures")
os.makedirs(OUT, exist_ok=True)
FLOOR = 5e-4
MK = {"AINR": "o", "DiscNRX": "s", "LMMSE": "^"}
CO = {"AINR": "#1f77b4", "DiscNRX": "#d62728", "LMMSE": "#2ca02c"}
LB = {"AINR": "AINR (proposed)", "DiscNRX": "Discriminative NRX", "LMMSE": "LMMSE"}


def rd(p):
    with open(os.path.join(ROOT, p)) as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("->", name)


# ---- Fig: controlled inversion (latent-target experiment) ------------------
def fig_latent():
    rows = rd("results/latent_target.csv")
    step = np.array([int(r["step"]) for r in rows])
    cb = np.array([float(r["coded_ber"]) for r in rows])
    ib = np.array([float(r["info_ber"]) for r in rows])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(step, ib, color="#d62728", lw=2,
            label="Information-bit latent (relaxed LDPC encoder)")
    ax.plot(step, cb, color="#1f77b4", lw=2, label="Coded-bit latent (proposed)")
    ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.6)
    ax.text(step[-1], 0.515, "chance", ha="right", fontsize=8, color="#444")
    ax.set_xlabel("Gradient-descent iteration")
    ax.set_ylabel("Bit error rate of the latent estimate")
    ax.set_ylim(-0.02, 0.6)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="center right")
    save(fig, "fig_latent.png")


# ---- Fig: multi-seed online adaptation (two operating points) ---------------
def _rolling_curves(rows, win, mode, rx=None, lr=None):
    key_rx = (lambda r: r.get("receiver", "AINR") == rx) if rx else (lambda r: True)
    curves = []
    seeds = sorted({r["seed"] for r in rows if r["mode"] == mode and key_rx(r)
                    and (lr is None or float(r.get("lr", 0)) == lr)})
    for sd in seeds:
        rs = [r for r in rows if r["mode"] == mode and key_rx(r)
              and r["seed"] == sd and (lr is None or float(r.get("lr", 0)) == lr)]
        rs.sort(key=lambda r: int(r["slot"]))
        b = np.array([float(r["bler"]) for r in rs])
        curves.append(np.convolve(b, np.ones(win) / win, mode="valid"))
    return np.array(curves)


def fig_stage_d():
    win = 50
    rows5 = rd("results/stage_d_stats.csv")
    rows3 = rd("results/stage_d_harsh.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    # Panel (a): moderate drift, 5 dB.
    ax = axes[0]
    for rx in ["AINR", "DiscNRX"]:
        c = _rolling_curves(rows5, win, "gated", rx, 1e-4)
        x = np.arange(win - 1, win - 1 + c.shape[1])
        m, s = c.mean(0), c.std(0)
        ax.plot(x, m, color=CO[rx], lw=1.9,
                label=f"{LB[rx]}, gated ({c.shape[0]} seeds)")
        ax.fill_between(x, m - s, m + s, color=CO[rx], alpha=0.18)
    c = _rolling_curves(rows5, win, "ungated", "AINR")
    ax.plot(x[:c.shape[1]], c.mean(0), color="#9467bd", lw=1.6, ls="--",
            label=f"AINR, ungated ({c.shape[0]} seeds)")
    c = _rolling_curves(rows5, win, "none", "AINR")
    ax.plot(x[:c.shape[1]], c.mean(0), color="#7f7f7f", lw=1.4, ls=":",
            label="AINR, no adaptation")
    ax.set_title("(a) Moderate drift: S2 at 5 dB")
    ax.legend(loc="upper right", fontsize=8)

    # Panel (b): harsh drift, 3 dB (AINR only).
    ax = axes[1]
    c = _rolling_curves(rows3, win, "gated")
    x = np.arange(win - 1, win - 1 + c.shape[1])
    m, s = c.mean(0), c.std(0)
    ax.plot(x, m, color=CO["AINR"], lw=1.9,
            label=f"AINR, gated ({c.shape[0]} seeds)")
    ax.fill_between(x, m - s, m + s, color=CO["AINR"], alpha=0.18)
    c = _rolling_curves(rows3, win, "ungated")
    m, s = c.mean(0), c.std(0)
    ax.plot(x[:len(m)], m, color="#9467bd", lw=1.9, ls="--",
            label=f"AINR, ungated ({c.shape[0]} seeds)")
    ax.fill_between(x[:len(m)], m - s, m + s, color="#9467bd", alpha=0.18)
    c = _rolling_curves(rows3, win, "none")
    ax.plot(x[:c.shape[1]], c.mean(0), color="#7f7f7f", lw=1.4, ls=":",
            label="AINR, no adaptation")
    ax.set_title("(b) Harsh drift: S2 at 3 dB")
    ax.legend(loc="center right", fontsize=8)

    for ax in axes:
        ax.set_xlabel("Slot index after the drift event")
        ax.set_ylabel(f"Rolling BLER (window {win})")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, ls=":", alpha=0.5)
    save(fig, "fig_stage_d.png")


# ---- Fig: S4 with error bars -------------------------------------------------
def fig_s4():
    rows = rd("results/stage_s4_ci.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, qam in zip(axes, [16, 64]):
        for rx in ["LMMSE", "DiscNRX", "AINR"]:
            rs = [r for r in rows if r["receiver"] == rx and int(r["qam"]) == qam]
            rs.sort(key=lambda r: float(r["snr_db"]))
            xx = [float(r["snr_db"]) for r in rs]
            y = np.array([max(float(r["bler_mean"]), FLOOR) for r in rs])
            e = np.array([float(r["bler_std"]) for r in rs])
            ax.errorbar(xx, y, yerr=e, marker=MK[rx], color=CO[rx],
                        capsize=2.5, label=LB[rx])
        ax.set_yscale("log")
        ax.set_ylim(FLOOR, 1.5)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BLER")
        ax.set_title(f"S4 ray-traced (Munich), {qam}-QAM")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(loc="lower left")
    save(fig, "fig_s4.png")


# ---- Fig: ROC from pooled multi-run scores -----------------------------------
def fig_roc():
    scores = rd("results/roc_scores_pooled.csv")
    summ = {r["scenario"]: r for r in rd("results/roc_summary.csv")}
    titles = {"S2": "S2: CDL-A delay drift", "S3": "S3: 90 km/h Doppler drift"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, sk in zip(axes, ["S2", "S3"]):
        rs = [r for r in scores if r["scenario"] == sk]
        lab = np.array([int(r["label"]) for r in rs])
        sc = np.array([float(r["score"]) for r in rs])
        thr = np.sort(np.unique(sc))[::-1]
        tpr = [(sc[lab == 1] >= t).mean() for t in thr]
        fpr = [(sc[lab == 0] >= t).mean() for t in thr]
        s = summ[sk]
        ax.plot(fpr, tpr, color="#1f77b4", lw=2,
                label=(f"AINR residual (AUC={float(s['auc_mean']):.3f}"
                       f"$\\pm${float(s['auc_std']):.3f})"))
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance (AUC=0.500)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(titles[sk])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower right")
    save(fig, "fig_roc.png")


if __name__ == "__main__":
    fig_latent()
    fig_stage_d()
    fig_s4()
    fig_roc()
    print("NEW FIGURES DONE")
