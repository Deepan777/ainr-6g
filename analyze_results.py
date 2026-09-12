"""Analysis of the OJ-COMS experiments, implementing FINAL_OJCOMS/audits/ANALYSIS_PROTOCOL.md.

Reads raw per-slot / per-run outputs in results/ojcoms/ and writes figure/table data to
FINAL_OJCOMS/data/. Uses only the seeds whose files exist and records n_seeds.

Usage: python analyze_results.py [--parts shift adapt inversion ablation budget mismatch]
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "ojcoms")
OUT = os.path.join(ROOT, "FINAL_OJCOMS", "data")
os.makedirs(OUT, exist_ok=True)

RX = ["AINR", "DiscNRX", "LS-LIN", "LS-LMMSE", "PCSI"]
# statistic -> (direction, receiver whose dBLER it should track); direction:
# +1 high = shift, -1 low = shift, 0 two-sided (|z|)
STATS = {"resid": (1, "AINR"), "tdisp": (1, "AINR"), "comb": (1, "AINR"),
         "conf_a": (-1, "AINR"), "conf_d": (-1, "DiscNRX"),
         "crcfail_a": (1, "AINR"), "crcfail_d": (1, "DiscNRX"),
         "cf": (0, "AINR"), "ct": (0, "AINR"), "ymag": (0, "AINR")}
ALPHAS = [0.01, 0.05, 0.10]
WINDOWS = [1, 8, 32]
DS_SEV = [("S1", 100), ("DS200", 200), ("DS300", 300), ("DS600", 600), ("DS1000", 1000)]
V_SEV = [("S1", 3), ("V30", 30), ("V90", 90), ("V150", 150), ("V250", 250), ("V400", 400)]
NBOOT = 4000
RNG = np.random.default_rng(20260911)


def write(name, rows):
    if not rows:
        print(f"  (no rows for {name})"); return
    path = os.path.join(OUT, name)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    os.replace(path + ".tmp", path)
    print(f"  -> FINAL_OJCOMS/data/{name} ({len(rows)} rows)")


def tci(x):
    """Mean and 95 % t-interval over seeds (NaN interval for n < 2)."""
    x = np.asarray([v for v in x if v == v], float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan"), n
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n))
    return m, m - h, m + h, n


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    r = stats.rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def seeds_of(pattern):
    out = {}
    for f in glob.glob(os.path.join(RES, pattern)):
        m = re.search(r"seed(\d+)\.csv$", f)
        if m and "_quick" not in f:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def read(f):
    with open(f, newline="") as fh:
        return list(csv.DictReader(fh))


# ============================================================================ shift
def analyze_shift():
    files = seeds_of("shift_slots_seed*.csv")
    if not files:
        print("shift: no data"); return
    print(f"shift: seeds {list(files)}")
    per_seed = {}          # seed -> dict[(stream, cond, snr)] -> dict col -> np.array
    for s, f in files.items():
        D = defaultdict(lambda: defaultdict(list))
        for r in read(f):
            key = (r["stream"], r["cond"], float(r["snr_db"]))
            for k, v in r.items():
                if k not in ("cond", "cdl", "stream"):
                    D[key][k].append(float(v))
        D2 = {}
        for key, cols in D.items():
            c = {k: np.asarray(v) for k, v in cols.items()}
            c["crcfail_a"] = 1.0 - c["crc_a"]; c["crcfail_d"] = 1.0 - c["crc_d"]
            D2[key] = c
        per_seed[s] = D2
    seeds = list(per_seed)
    conds = list(dict.fromkeys(k[1] for k in per_seed[seeds[0]]))
    snrs = sorted({k[2] for k in per_seed[seeds[0]]})
    shifted = [c for c in conds if c != "S1"]

    # --- z-standardisation from the S1 validation stream (same seed, SNR) ---
    def z_of(s, snr, stat, x):
        v = per_seed[s][("val", "S1", snr)]
        if stat == "comb":
            return np.maximum(z_of(s, snr, "resid", x[0]), z_of(s, snr, "tdisp", x[1]))
        mu, sd = v[stat].mean(), v[stat].std(ddof=1)
        z = (x - mu) / (sd if sd > 0 else 1.0)
        d = STATS[stat][0]
        return np.abs(z) if d == 0 else d * z

    def oriented(s, snr, stat, stream, cond):
        c = per_seed[s][(stream, cond, snr)]
        if stat == "comb":
            return z_of(s, snr, "comb", (c["resid"], c["tdisp"]))
        return z_of(s, snr, stat, c[stat])

    def windows(x, W):
        n = len(x) // W
        return x[: n * W].reshape(n, W).mean(1) if W > 1 else x

    def boot_windows(x, W):
        return RNG.choice(x, size=(NBOOT, W), replace=True).mean(1) if W > 1 else x

    # --- BLER ---
    bler_rows, paired_rows = [], []
    for snr in snrs:
        for c in conds:
            for rx in RX:
                vals, ks, ns = [], 0, 0
                for s in seeds:
                    e = per_seed[s][("test", c, snr)][f"err_{rx}"]
                    vals.append(e.mean()); ks += int(e.sum()); ns += len(e)
                m, lo, hi, n = tci(vals)
                plo, phi = clopper(ks, ns)
                bler_rows.append({"snr_db": snr, "cond": c, "receiver": rx, "n_seeds": n,
                                  "bler_mean": m, "ci_lo": lo, "ci_hi": hi,
                                  "pooled_errors": ks, "pooled_blocks": ns, "cp_lo": plo, "cp_hi": phi,
                                  **{f"seed{s}": v for s, v in zip(seeds, vals)}})
    write("shift_bler_summary.csv", bler_rows)

    # --- Family F1: AINR - DiscNRX paired over seeds, shifted conditions, 5 dB ---
    if len(seeds) >= 2 and 5.0 in snrs:
        tests = []
        for c in shifted:
            d = [per_seed[s][("test", c, 5.0)]["err_AINR"].mean() - per_seed[s][("test", c, 5.0)]["err_DiscNRX"].mean()
                 for s in seeds]
            m, lo, hi, n = tci(d)
            p = float(stats.ttest_1samp(d, 0.0).pvalue) if np.std(d) > 0 else (1.0 if np.mean(d) == 0 else 0.0)
            tests.append({"cond": c, "snr_db": 5.0, "n_seeds": n, "mean_diff_AINR_minus_Disc": m,
                          "ci_lo": lo, "ci_hi": hi, "n_ainr_better": int(sum(x < 0 for x in d)),
                          "n_disc_better": int(sum(x > 0 for x in d)), "p_raw": p})
        adj = holm([t["p_raw"] for t in tests])
        for t, a in zip(tests, adj):
            t["p_holm"] = float(a)
        write("paired_F1_ainr_vs_disc.csv", tests)

    # --- Detection: AUC and TPR at validation thresholds, windows ---
    auc_rows, tpr_rows, rel_rows = [], [], []
    for snr in snrs:
        for W in WINDOWS:
            for stat in STATS:
                for c in shifted:
                    aucs, tprs = [], {a: [] for a in ALPHAS}
                    fprs = {a: [] for a in ALPHAS}
                    for s in seeds:
                        neg = windows(oriented(s, snr, stat, "test", "S1"), W)
                        pos = windows(oriented(s, snr, stat, "test", c), W)
                        aucs.append(auc(pos, neg))
                        valw = boot_windows(oriented(s, snr, stat, "val", "S1"), W)
                        for a in ALPHAS:
                            tau = np.quantile(valw, 1 - a)
                            tprs[a].append(float((pos > tau).mean())); fprs[a].append(float((neg > tau).mean()))
                    m, lo, hi, n = tci(aucs)
                    auc_rows.append({"snr_db": snr, "window": W, "stat": stat, "cond": c, "n_seeds": n,
                                     "auc_mean": m, "ci_lo": lo, "ci_hi": hi,
                                     **{f"seed{s}": v for s, v in zip(seeds, aucs)}})
                    for a in ALPHAS:
                        mt, lot, hit, _ = tci(tprs[a]); mf, _, _, _ = tci(fprs[a])
                        tpr_rows.append({"snr_db": snr, "window": W, "stat": stat, "cond": c, "alpha": a,
                                         "n_seeds": n, "tpr_mean": mt, "tpr_lo": lot, "tpr_hi": hit,
                                         "realised_fpr_mean": mf})
        # relevance: Spearman between per-condition median oriented z and dBLER (single slots)
        for stat, (_, rxname) in STATS.items():
            for s in seeds:
                zmed, dbl = [], []
                base = per_seed[s][("test", "S1", snr)][f"err_{rxname}"].mean()
                for c in shifted:
                    zmed.append(float(np.median(oriented(s, snr, stat, "test", c))))
                    dbl.append(per_seed[s][("test", c, snr)][f"err_{rxname}"].mean() - base)
                rho = stats.spearmanr(zmed, dbl).correlation if np.std(dbl) > 0 else float("nan")
                harmful = [c for c, d in zip(shifted, dbl) if d >= 0.01]
                rel_rows.append({"snr_db": snr, "stat": stat, "tracks": rxname, "seed": s,
                                 "spearman_rho": rho, "n_harmful": len(harmful), "harmful": " ".join(harmful)})
    write("detector_auc_summary.csv", auc_rows)
    write("detector_tpr_summary.csv", tpr_rows)
    write("detector_relevance.csv", rel_rows)

    # --- Severity curves ---
    for tag, sev in (("ds", DS_SEV), ("v", V_SEV)):
        rows = []
        for snr in snrs:
            for c, x in sev:
                r = {"snr_db": snr, "cond": c, "severity": x}
                for rx in RX:
                    m, lo, hi, n = tci([per_seed[s][("test", c, snr)][f"err_{rx}"].mean() for s in seeds])
                    r.update({f"bler_{rx}": m, f"bler_{rx}_lo": lo, f"bler_{rx}_hi": hi})
                for stat in ("resid", "tdisp", "conf_a", "conf_d", "comb"):
                    zm = [float(np.median(oriented(s, snr, stat, "test", c))) for s in seeds]
                    m, lo, hi, n = tci(zm)
                    r.update({f"z_{stat}": m, f"z_{stat}_lo": lo, f"z_{stat}_hi": hi})
                    if c != "S1":
                        a = [auc(oriented(s, snr, stat, "test", c), oriented(s, snr, stat, "test", "S1")) for s in seeds]
                        m, lo, hi, _ = tci(a)
                    else:
                        m = lo = hi = 0.5
                    r.update({f"auc_{stat}": m, f"auc_{stat}_lo": lo, f"auc_{stat}_hi": hi})
                r["n_seeds"] = len(seeds)
                rows.append(r)
        write(f"severity_{tag}.csv", rows)


# ============================================================================ adapt
def rolling(e, w=50):
    c = np.cumsum(np.insert(e, 0, 0.0))
    out = np.full(len(e), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def analyze_adapt():
    files = sorted(f for f in glob.glob(os.path.join(RES, "adapt_*_seed*.csv")) if "_quick" not in f)
    if not files:
        print("adapt: no data"); return
    summ, curves = [], defaultdict(list)
    for f in files:
        m = re.search(r"adapt_(\w+?)_snr([\d.]+)_seed(\d+)\.csv$", os.path.basename(f))
        cond, snr, seed = m.group(1), float(m.group(2)), int(m.group(3))
        G = defaultdict(list)
        for r in read(f):
            G[(r["receiver"], r["mode"])].append(r)
        init_pass = {rx: float(np.mean([float(r["crc_pass"]) for r in G[(rx, "none")][:100]])) if (rx, "none") in G else float("nan")
                     for rx in ("AINR", "DiscNRX")}
        for (rx, mode), L in G.items():
            e = np.array([float(r["err"]) for r in L])
            up = np.array([float(r["updated"]) for r in L]); cp = np.array([float(r["crc_pass"]) for r in L])
            fp = np.array([float(r["false_pass"]) for r in L])
            ms = [float(r["update_ms"]) for r in L if r["update_ms"] not in ("", "nan") and float(r["update_ms"]) == float(r["update_ms"])]
            roll = rolling(e)
            below = roll < 0.10
            rec = float("inf")
            for t in range(len(e)):
                if not np.isnan(roll[t]) and below[t:][~np.isnan(roll[t:])].all():
                    rec = t + 1; break          # 1-based slot index (50 = first full window)
            summ.append({"cond": cond, "snr_db": snr, "seed": seed, "receiver": rx, "mode": mode,
                         "first200": e[:200].mean(), "last200": e[-200:].mean(), "all": e.mean(),
                         "recovery_slot": rec, "accepted_updates": int(up.sum()),
                         "rejected_crc_fail": int(((1 - cp) * (mode in ("gated", "trig"))).sum()),
                         "false_pass_updates": int((fp * up).sum()), "crc_passes": int(cp.sum()),
                         "median_update_ms": float(np.median(ms)) if ms else float("nan"),
                         "initial_pass_rate": init_pass.get(rx, float("nan")), "n_slots": len(e)})
            curves[(cond, snr, rx, mode)].append(roll)
    write("adapt_summary_per_seed.csv", summ)
    agg = []
    keys = sorted({(r["cond"], r["snr_db"], r["receiver"], r["mode"]) for r in summ})
    for k in keys:
        L = [r for r in summ if (r["cond"], r["snr_db"], r["receiver"], r["mode"]) == k]
        row = {"cond": k[0], "snr_db": k[1], "receiver": k[2], "mode": k[3], "n_seeds": len(L)}
        for col in ("first200", "last200", "accepted_updates", "false_pass_updates", "median_update_ms", "initial_pass_rate"):
            m, lo, hi, _ = tci([r[col] for r in L])
            row.update({col: m, f"{col}_lo": lo, f"{col}_hi": hi})
        rec = [r["recovery_slot"] for r in L]
        row["recovery_slots"] = " ".join("inf" if x == float("inf") else str(int(x)) for x in rec)
        agg.append(row)
    write("adapt_summary.csv", agg)
    rows = []
    for (cond, snr, rx, mode), R in curves.items():
        A = np.vstack(R)
        for t in range(49, A.shape[1], 10):
            m, lo, hi, n = tci(A[:, t])
            rows.append({"cond": cond, "snr_db": snr, "receiver": rx, "mode": mode, "slot": t + 1,
                         "rolling_bler": m, "ci_lo": lo, "ci_hi": hi, "n_seeds": n})
    write("adapt_rolling.csv", rows)


# ============================================================================ inversion
def analyze_inversion():
    f = os.path.join(RES, "inversion_summary.csv")
    if not os.path.exists(f):
        print("inversion: no data"); return
    R = read(f)
    G = defaultdict(list)
    for r in R:
        G[(r["config"], r["init"])].append(r)
    rows = []
    for (c, i), L in G.items():
        row = {"config": c, "init": i, "n_seeds": len(L), "K": L[0]["K"], "n": L[0]["n"], "colw_mean": L[0]["colw_mean"]}
        for col in ("final_coded_ber", "post_bp_bler", "final_info_ber", "init_info_implied_coded_ber",
                    "final_grad_info", "final_grad_coded"):
            v = [float(x[col]) for x in L]
            row.update({f"{col}_mean": float(np.mean(v)), f"{col}_std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")})
        rows.append(row)
    write("inversion_summary.csv", rows)
    T = read(os.path.join(RES, "inversion_trace.csv"))
    G = defaultdict(list)
    for r in T:
        G[(r["config"], r["init"], int(float(r["step"])))].append(r)
    rows = []
    for (c, i, st), L in sorted(G.items()):
        rows.append({"config": c, "init": i, "step": st, "n_seeds": len(L),
                     **{f"{col}_mean": float(np.mean([float(x[col]) for x in L])) for col in ("coded_ber", "info_ber", "grad_coded", "grad_info")}})
    write("inversion_trace.csv", rows)


# ============================================================================ ablation / budget / mismatch
def analyze_ablation():
    rows = []
    for kind in ("bler", "shift", "summary"):
        for f in sorted(glob.glob(os.path.join(RES, f"ablation_{kind}_seed*.csv"))):
            if "_quick" in f or "_noshift" in f:
                continue
            for r in read(f):
                r["kind"] = kind; rows.append(r)
    if not rows:
        print("ablation: no data"); return
    write("ablation_all.csv", rows)
    shift = [r for r in rows if r["kind"] == "shift"]
    seeds = sorted({r["seed"] for r in shift})
    agg = []
    for v in sorted({r["variant"] for r in shift}):
        for c in sorted({r["cond"] for r in shift}):
            vals = [float(r["bler"]) for r in shift if r["variant"] == v and r["cond"] == c]
            m, lo, hi, n = tci(vals)
            agg.append({"variant": v, "cond": c, "snr_db": 5.0, "n_seeds": n, "bler_mean": m, "ci_lo": lo, "ci_hi": hi})
    write("ablation_shift_summary.csv", agg)

    # Paired comparison against the reference variant over the seeds that ran every variant
    # (protocol: paired t-tests over training seeds, Holm-corrected within each family).
    by = {(r["variant"], r["cond"], r["seed"]): float(r["bler"]) for r in shift}
    conds = [c for c in ("DS600", "DS1000", "S2", "V250") if any(k[1] == c for k in by)]
    paired = []
    for v in ("no_detach", "pure_vfe"):
        common = [s for s in seeds if all((v, c, s) in by and ("full", c, s) in by for c in conds)]
        if len(common) < 2:
            continue
        tests = []
        for c in conds:
            d = [by[(v, c, s)] - by[("full", c, s)] for s in common]      # variant minus reference
            m, lo, hi, n = tci(d)
            p = float(stats.ttest_1samp(d, 0.0).pvalue) if np.std(d) > 0 else (1.0 if np.mean(d) == 0 else 0.0)
            tests.append({"variant": v, "cond": c, "n_seeds": n, "mean_diff_vs_full": m,
                          "ci_lo": lo, "ci_hi": hi, "p_raw": p})
        for t, a in zip(tests, holm([t["p_raw"] for t in tests])):
            t["p_holm"] = float(a)
        paired += tests
    if paired:
        write("ablation_paired.csv", paired)


def analyze_budget():
    files = seeds_of("budget_shift_seed*.csv")
    if not files:
        print("budget: no data"); return
    rows = [r for f in files.values() for r in read(f)]
    agg = []
    for rx in ("AINR", "DiscNRX"):
        for st in sorted({int(float(r["step"])) for r in rows}):
            for c in sorted({r["cond"] for r in rows}):
                v = [float(r["bler"]) for r in rows if r["receiver"] == rx and int(float(r["step"])) == st and r["cond"] == c]
                m, lo, hi, n = tci(v)
                agg.append({"receiver": rx, "step": st, "cond": c, "n_seeds": n, "bler_mean": m, "ci_lo": lo, "ci_hi": hi})
    write("budget_summary.csv", agg)


def snr_at(snrs, bler, target=0.1):
    """SNR where BLER crosses `target`, by linear interpolation of log10(BLER); NaN if never."""
    pts = [(s, b) for s, b in zip(snrs, bler)]
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if b0 >= target > b1:
            if b1 <= 0:
                return s0 + (s1 - s0) * (b0 - target) / (b0 - b1)   # linear when the next point is 0
            f = (math.log10(b0) - math.log10(target)) / (math.log10(b0) - math.log10(b1))
            return s0 + f * (s1 - s0)
    return float("nan")


def analyze_matched():
    rows, crossing, wide_all = [], [], {}
    for qam in (16, 64):
        neural = {int(m.group(1)): f for f in glob.glob(os.path.join(RES, f"matched_qam{qam}_S1_seed*.csv"))
                  for m in [re.search(r"seed(\d+)\.csv$", f)] if m}          # skips *_quick, *_coarse
        cls = os.path.join(RES, f"matched_qam{qam}_S1_classical.csv")
        if not neural:
            continue
        per = defaultdict(lambda: defaultdict(dict))            # rx -> seed -> snr -> (err, n)
        for s, f in neural.items():
            for r in read(f):
                per[r["receiver"]][s][float(r["snr_db"])] = (int(r["block_errors"]), int(r["n_blocks"]))
        if os.path.exists(cls):
            for r in read(cls):
                per[r["receiver"]][-1][float(r["snr_db"])] = (int(r["block_errors"]), int(r["n_blocks"]))
        snrs = sorted({snr for rx in per for s in per[rx] for snr in per[rx][s]})
        wide = {snr: {"snr_db": snr} for snr in snrs}
        for rx, seeds in per.items():
            for snr in snrs:
                vals = [seeds[s][snr][0] / seeds[s][snr][1] for s in seeds if snr in seeds[s]]
                k = sum(seeds[s][snr][0] for s in seeds if snr in seeds[s]); n = sum(seeds[s][snr][1] for s in seeds if snr in seeds[s])
                m, lo, hi, ns = tci(vals)
                plo, phi = clopper(k, n)
                rows.append({"qam": qam, "receiver": rx, "snr_db": snr, "n_seeds": ns, "bler_mean": m, "ci_lo": lo,
                             "ci_hi": hi, "pooled_errors": k, "pooled_blocks": n, "cp_lo": plo, "cp_hi": phi})
                key = rx.replace("-", "")
                # log axis (ymin 5e-4): points without errors are not drawn; a band edge that
                # falls outside the axis is clamped to it so the band still renders
                if m != m or m <= 0:
                    wide[snr][key] = wide[snr][key + "_lo"] = wide[snr][key + "_hi"] = "nan"
                else:
                    wide[snr][key] = f"{m:.6g}"
                    wide[snr][key + "_lo"] = f"{max(lo if lo == lo else m, 5e-4):.6g}"
                    wide[snr][key + "_hi"] = f"{min(hi if hi == hi else m, 1.0):.6g}"
            for s in seeds:
                b = [seeds[s][snr][0] / seeds[s][snr][1] for snr in snrs if snr in seeds[s]]
                crossing.append({"qam": qam, "receiver": rx, "seed": s, "snr_at_bler_0p1": snr_at(snrs, b)})
        wide_all[qam] = [wide[s] for s in snrs]
        write(f"fig_matched_{qam}.csv", wide_all[qam])
    if rows:
        write("matched_summary.csv", rows)
        write("matched_snr01.csv", crossing)


def analyze_s4():
    rows, arows = [], []
    files = {int(re.search(r"seed(\d+)\.csv$", f).group(1)): f for f in glob.glob(os.path.join(RES, "s4_bler_seed*.csv"))
             if "_quick" not in f}
    if not files:
        print("s4: no data"); return
    per = defaultdict(lambda: defaultdict(dict))
    for s, f in files.items():
        for r in read(f):
            per[r["receiver"]][s][float(r["snr_db"])] = float(r["bler"])
    cls = os.path.join(RES, "s4_bler_classical.csv")
    if os.path.exists(cls):
        for r in read(cls):
            per[r["receiver"]][-1][float(r["snr_db"])] = float(r["bler"])
    snrs = sorted({x for rx in per for s in per[rx] for x in per[rx][s]})
    wide = []
    for snr in snrs:
        w = {"snr_db": snr}
        for rx, seeds in per.items():
            m, lo, hi, n = tci([seeds[s][snr] for s in seeds if snr in seeds[s]])
            rows.append({"receiver": rx, "snr_db": snr, "n_seeds": n, "bler_mean": m, "ci_lo": lo, "ci_hi": hi})
            key = rx.replace("-", "")
            if m != m or m <= 0:                      # log axis (ymin 1e-3), as in analyze_matched
                w[key] = w[key + "_lo"] = w[key + "_hi"] = "nan"
            else:
                w[key] = f"{m:.6g}"
                w[key + "_lo"] = f"{max(lo if lo == lo else m, 1e-3):.6g}"
                w[key + "_hi"] = f"{min(hi if hi == hi else m, 1.0):.6g}"
        wide.append(w)
    write("s4_summary.csv", rows)
    write("fig_s4.csv", wide)
    for f in glob.glob(os.path.join(RES, "s4_adapt_seed*.csv")):
        if "_quick" not in f:
            arows += read(f)
    if arows:
        agg = []
        for key in sorted({(r["receiver"], r["snr_db"]) for r in arows}):
            L = [r for r in arows if (r["receiver"], r["snr_db"]) == key]
            b, lo_b, hi_b, n = tci([float(r["bler_before"]) for r in L])
            a, lo_a, hi_a, _ = tci([float(r["bler_after"]) for r in L])
            agg.append({"receiver": key[0], "snr_db": key[1], "n_seeds": n, "bler_before": b, "before_lo": lo_b,
                        "before_hi": hi_b, "bler_after": a, "after_lo": lo_a, "after_hi": hi_a,
                        "updates_mean": np.mean([float(r["updates"]) for r in L]),
                        "false_pass_total": int(sum(float(r["false_pass_updates"]) for r in L))})
        write("s4_adapt_summary.csv", agg)


def analyze_mismatch():
    f = os.path.join(RES, "model_mismatch.csv")
    if os.path.exists(f):
        write("model_mismatch.csv", read(f))


# ============================================================================ broad control
BROAD_RX = ["Broad", "DiscNRX", "AINR", "LS-LIN", "LS-LMMSE"]


def analyze_broad():
    """Broadly trained DiscNRX (one training seed) vs the seed-0 receivers on identical slots
    (protocol deviation log, 2026-09-11): descriptive only."""
    f = os.path.join(RES, "broad_shift_slots.csv")
    if not os.path.exists(f):
        print("broad: no data"); return
    D = defaultdict(lambda: defaultdict(list))
    for r in read(f):
        key = (r["stream"], r["cond"], float(r["snr_db"]))
        for k in [f"err_{x}" for x in BROAD_RX] + ["conf_b", "conf_d", "crc_b"]:
            D[key][k].append(float(r[k]))
    D = {k: {c: np.asarray(v) for c, v in cols.items()} for k, cols in D.items()}
    conds = list(dict.fromkeys(k[1] for k in D))
    rows = []
    for snr in sorted({k[2] for k in D}):
        s1t, s1v = D[("test", "S1", snr)], D[("val", "S1", snr)]
        for c in conds:
            t = D[("test", c, snr)]
            r = {"snr_db": snr, "cond": c, "n_blocks": len(t["err_Broad"])}
            for rx in BROAD_RX:
                e = t[f"err_{rx}"]; lo, hi = clopper(int(e.sum()), len(e))
                r.update({f"bler_{rx}": float(e.mean()), f"cp_lo_{rx}": lo, f"cp_hi_{rx}": hi})
            b_only = int(((t["err_Broad"] == 1) & (t["err_DiscNRX"] == 0)).sum())
            d_only = int(((t["err_Broad"] == 0) & (t["err_DiscNRX"] == 1)).sum())
            r.update({"broad_only_errors": b_only, "narrow_only_errors": d_only,
                      "mcnemar_p": float(stats.binomtest(min(b_only, d_only), b_only + d_only, 0.5).pvalue)
                      if b_only + d_only else 1.0,
                      "crc_false_pass_broad": int(((t["crc_b"] == 1) & (t["err_Broad"] == 1)).sum())})
            if c != "S1":
                for stat in ("conf_b", "conf_d"):             # lower confidence = drift
                    r[f"auc_{stat}"] = auc(-t[stat], -s1t[stat])
                    tau = np.quantile(-s1v[stat], 0.95)          # validation-only threshold, alpha = 5 %
                    r[f"tpr05_{stat}"] = float((-t[stat] > tau).mean())
                    r[f"fpr05_{stat}"] = float((-s1t[stat] > tau).mean())
            rows.append(r)
    write("broad_summary.csv", rows)
    mf = os.path.join(RES, "broad_matched.csv")
    if os.path.exists(mf):
        M = read(mf)
        cr = []
        for rx in dict.fromkeys(r["receiver"] for r in M):
            L = sorted((float(r["snr_db"]), float(r["bler"])) for r in M if r["receiver"] == rx)
            cr.append({"receiver": rx, "snr_at_bler_0p1": snr_at([x for x, _ in L], [y for _, y in L])})
        write("broad_matched_snr01.csv", cr)
    sf = os.path.join(RES, "broad_s4_bler.csv")
    if os.path.exists(sf):
        write("broad_s4.csv", read(sf))


# ============================================================================ figure exports
def _num(v):
    try:
        x = float(v)
        return "nan" if x != x else f"{x:.6g}"
    except (TypeError, ValueError):
        return v


def export_figures():
    """Tidy one-file-per-figure CSVs (numbers only) for the PGFPlots sources."""
    def rd(name):
        p = os.path.join(OUT, name)
        return read(p) if os.path.exists(p) else []

    # columns that are probabilities: their plotted interval must stay inside [0, 1]
    PROB = {"AINR", "DiscNRX", "LSLIN", "LSLMMSE", "PCSI", "Broad"}

    def tidy(r, prob_all=False):
        """Numbers only; '-' removed from names; a missing CI (single seed) becomes a
        zero-width interval at the mean so the plotting code is seed-count agnostic.
        Student-t intervals over few seeds can reach outside [0, 1]; for probabilities the
        *plotted* band is clipped there (the unclipped values stay in the summary CSVs)."""
        out = {}
        for k, v in r.items():
            if k == "cond":
                continue
            out[k.replace("-", "")] = _num(v)
        for k in list(out):
            if k.endswith("_lo") or k.endswith("_hi"):
                base = k[:-3]
                if out[k] == "nan" and base in out:
                    out[k] = out[base]
                elif prob_all or base in PROB or base.startswith(("bler", "auc", "rolling", "tpr")):
                    try:
                        out[k] = _num(min(1.0, max(0.0, float(out[k]))))
                    except (TypeError, ValueError):
                        pass
        return out

    # severity curves: one file per (axis, SNR)
    for tag in ("ds", "v"):
        rows = rd(f"severity_{tag}.csv")
        for snr in sorted({r["snr_db"] for r in rows}):
            write(f"fig_sev_{tag}_{float(snr):g}dB.csv", [tidy(r) for r in rows if r["snr_db"] == snr])
    # model mismatch versus severity
    mm = rd("model_mismatch.csv")
    if mm:
        by = {r["cond"]: r for r in mm}
        for tag, sev in (("ds", DS_SEV), ("v", V_SEV)):
            write(f"fig_mismatch_{tag}.csv", [{"severity": x, "e_time": _num(by[c]["e_time"]), "e_taps": _num(by[c]["e_taps"]),
                                                "e_total": _num(by[c]["e_total"])} for c, x in sev if c in by])
    # inversion traces: wide format per code
    tr = rd("inversion_trace.csv")
    for cfgname in sorted({r["config"] for r in tr}):
        steps = sorted({int(r["step"]) for r in tr if r["config"] == cfgname})
        wide = []
        for st in steps:
            row = {"step": st}
            for r in tr:
                if r["config"] == cfgname and int(r["step"]) == st:
                    row[f"info_{r['init']}"] = _num(r["info_ber_mean"]); row[f"coded_{r['init']}"] = _num(r["coded_ber_mean"])
                    row[f"ginfo_{r['init']}"] = _num(r["grad_info_mean"]); row[f"gcoded_{r['init']}"] = _num(r["grad_coded_mean"])
            wide.append(row)
        write(f"fig_inversion_{cfgname}.csv", wide)
    # ROC curves (5 dB, single slot): vertical averaging over seeds on a fixed FPR grid
    files = seeds_of("shift_slots_seed*.csv")
    if files:
        fpr_grid = np.linspace(0, 1, 101)
        stat_dir = {"resid": 1, "comb": 1, "conf_d": -1, "tdisp": 1}
        per = defaultdict(list)
        streams = {}
        for s, f in files.items():
            D = defaultdict(lambda: defaultdict(list))
            for r in read(f):
                if float(r["snr_db"]) != 5.0:
                    continue
                for k in ("resid", "tdisp", "conf_d"):
                    D[(r["stream"], r["cond"])][k].append(float(r[k]))
            val = {k: (np.mean(D[("val", "S1")][k]), np.std(D[("val", "S1")][k], ddof=1)) for k in ("resid", "tdisp", "conf_d")}
            def z(stream, cond, k):
                x = np.asarray(D[(stream, cond)][k]); mu, sd = val[k]
                return (x - mu) / sd
            def score(stream, cond, st):
                if st == "comb":
                    return np.maximum(z(stream, cond, "resid"), z(stream, cond, "tdisp"))
                return stat_dir[st] * z(stream, cond, st)
            for cond in ("S2", "V90", "V250"):
                for st in stat_dir:
                    sn, sp = score("test", "S1", st), score("test", cond, st)
                    thr = np.unique(np.concatenate([sn, sp, [np.inf, -np.inf]]))[::-1]
                    fpr = np.array([(sn >= t).mean() for t in thr]); tpr = np.array([(sp >= t).mean() for t in thr])
                    per[(cond, st)].append(np.interp(fpr_grid, fpr, tpr))
            if s == min(files):
                streams = {k: (score("test", "S1", k), score("test", "S2", k), score("test", "V250", k)) for k in ("resid", "tdisp", "conf_d")}
        rows = []
        for i, x in enumerate(fpr_grid):
            row = {"fpr": f"{x:.3f}"}
            for (cond, st), L in per.items():
                row[f"{cond}_{st}"] = f"{np.mean([l[i] for l in L]):.4f}"
            rows.append(row)
        write("fig_roc_5dB.csv", rows)
        # drift-onset timeline: 150 matched slots, then 150 S2 slots (and V250), first seed
        tl = []
        for i in range(300):
            row = {"slot": i + 1}
            for k in ("resid", "tdisp", "conf_d"):
                a, b, c = streams[k]
                row[f"S2_{k}"] = f"{(a[i] if i < 150 else b[i - 150]):.4f}"
                row[f"V250_{k}"] = f"{(a[i] if i < 150 else c[i - 150]):.4f}"
            tl.append(row)
        write("fig_timeline_5dB.csv", tl)
    # training convergence (seed 0 plus any finished seeds), every 500 steps, last entry per step kept
    conv = []
    logs = {0: os.path.join(ROOT, "results", "ainr_training_log.csv")}
    for s in range(1, 5):
        logs[s] = os.path.join(ROOT, "results", "seeds", f"seed{s}", "ainr_training_log.csv")
    series = {}
    for s, p in logs.items():
        if not os.path.exists(p):
            continue
        last = {}
        for r in read(p):
            st = int(float(r["step"]))
            last[st] = r
        if max(last) < 50000:
            continue
        series[s] = last
    if series:
        steps = sorted(set.intersection(*[set(v) for v in series.values()]))
        for st in [x for x in steps if x % 500 == 0 or x == steps[0]]:
            row = {"step": st, "n_seeds": len(series)}
            for k in ("ce", "kl_channel", "kl_noise", "expected_ll"):
                v = [float(series[s][st][k]) for s in series]
                row[k] = f"{np.mean(v):.6g}"
                row[k + "_lo"] = f"{min(v):.6g}"; row[k + "_hi"] = f"{max(v):.6g}"
            row["neg_ell"] = f"{-float(row['expected_ll']):.6g}"
            conv.append(row)
        write("fig_training.csv", conv)

    # robustness versus training budget: one file per receiver (x = training steps)
    bud = rd("budget_summary.csv")
    if bud:
        steps = sorted({int(float(r["step"])) for r in bud})
        for rx, tag in (("AINR", "ainr"), ("DiscNRX", "disc")):
            rows = []
            for st in steps:
                row = {"step": st}
                for r in bud:
                    if r["receiver"] == rx and int(float(r["step"])) == st:
                        m, lo, hi = (float(r["bler_mean"]), float(r["ci_lo"]), float(r["ci_hi"]))
                        # log axis (ymin 5e-4): zero-error points are not drawn, and a band edge
                        # outside the axis is clamped to it -- 0 is not a valid log coordinate
                        # and would break the fill-between path.
                        if m != m or m <= 0:
                            v = lo_v = hi_v = "nan"
                        else:
                            v = f"{m:.6g}"
                            lo_v = f"{max(lo if lo == lo else m, 5e-4):.6g}"
                            hi_v = f"{min(hi if hi == hi else m, 1.0):.6g}"
                        row.update({r["cond"]: v, r["cond"] + "_lo": lo_v, r["cond"] + "_hi": hi_v})
                rows.append(row)
            write(f"fig_budget_{tag}.csv", rows)
    # adaptation rolling BLER: wide format per (cond, SNR)
    ar = rd("adapt_rolling.csv")
    for key in sorted({(r["cond"], r["snr_db"]) for r in ar}):
        slots = sorted({int(r["slot"]) for r in ar if (r["cond"], r["snr_db"]) == key})
        wide = []
        for s in slots:
            row = {"slot": s}
            for r in ar:
                if (r["cond"], r["snr_db"]) == key and int(r["slot"]) == s:
                    c = f"{r['receiver']}_{r['mode']}"
                    row[c] = _num(r["rolling_bler"]); row[c + "_lo"] = _num(r["ci_lo"]); row[c + "_hi"] = _num(r["ci_hi"])
            wide.append(tidy(row, prob_all=True))      # every column but `slot` is a rolling BLER
        write(f"fig_adapt_{key[0]}_{float(key[1]):g}dB.csv", wide)


# ============================================================================ text macros
def export_macros():
    """Write FINAL_OJCOMS/tables/results_macros.tex: every number quoted in the text is a
    macro generated from the data files, so text, tables and figures cannot disagree."""
    M = {}

    def rd(name):
        p = os.path.join(OUT, name)
        return read(p) if os.path.exists(p) else []

    def put(name, value, fmt="{:.3f}"):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro names must be letters only: {name}")
        M[name] = value if isinstance(value, str) else (fmt.format(value) if value == value else "n/a")
        if re.fullmatch(r"-[0-9.]+", M[name]):
            M[name] = "\\ensuremath{" + M[name] + "}"            # typographic minus in text and math

    def pfmt(p):
        """p-values: never print '0.00', which would read as exactly zero."""
        return "\\ensuremath{<0.001}" if p < 0.001 else f"{p:.3f}"

    alpha = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
             "7": "seven", "8": "eight", "9": "nine"}
    CN = {"S1": "Sone", "S2": "Stwo", "DS200": "DStwohundred", "DS300": "DSthreehundred",
          "DS600": "DSsixhundred", "DS1000": "DSthousand", "CDLA": "CdlA", "CDLB": "CdlB", "CDLD": "CdlD",
          "CDLE": "CdlE", "V30": "Vthirty", "V90": "Vninety", "V150": "Vonefifty", "V250": "Vtwofifty",
          "V400": "Vfourhundred", "16QAM-K1824": "Qsixteen", "64QAM-K2736": "Qsixtyfour", "16QAM-K576": "Qshort"}
    cname = lambda c: CN[c]
    rxname = {"AINR": "Ainr", "DiscNRX": "Disc", "LS-LIN": "Lslin", "LS-LMMSE": "Lmmse", "PCSI": "Pcsi"}
    # BLER per (condition, receiver, SNR) and the number of seeds behind it
    B = rd("shift_bler_summary.csv")
    for r in B:
        snr = alpha[str(int(float(r["snr_db"])))[0]] if float(r["snr_db"]) < 10 else "ten"
        put(f"bler{cname(r['cond'])}{rxname[r['receiver']]}{snr}", float(r["bler_mean"]))
    if B:
        put("nSeedsShift", str(int(float(B[0]["n_seeds"]))))
        put("nBlocksCond", str(int(float(B[0]["pooled_blocks"])) // max(1, int(float(B[0]["n_seeds"])))))
    # family F1: paired AINR - DiscNRX BLER differences over seeds (5 dB, shifted conditions)
    F1 = rd("paired_F1_ainr_vs_disc.csv")
    if F1:
        diffs = [(float(r["mean_diff_AINR_minus_Disc"]), r["cond"], float(r["p_holm"])) for r in F1]
        put("fOneNcond", str(len(F1)))
        put("fOneNsig", str(sum(p < 0.05 for _, _, p in diffs)))
        put("fOneNainrLower", str(sum(d < 0 for d, _, _ in diffs)))        # AINR lower mean BLER
        put("fOneNdiscLower", str(sum(d > 0 for d, _, _ in diffs)))
        put("fOneMinP", pfmt(min(p for _, _, p in diffs)))
        d, c, _ = max(diffs, key=lambda t: abs(t[0]))
        pretty = (c.replace("DS", "DS\\,") if c.startswith("DS") else c[1:] + "~km/h" if c.startswith("V")
                  else "CDL-" + c[3] if c.startswith("CDL") else c)
        put("fOneMaxAbsDiff", abs(d)); put("fOneMaxCond", pretty)
        put("fOneMaxSign", "lower" if d < 0 else "higher")
    # single-slot AUC at 5 dB for key statistics
    for r in rd("detector_auc_summary.csv"):
        if r["snr_db"] == "5.0" and r["window"] in ("1", "8") and r["stat"] in ("resid", "tdisp", "comb", "conf_a", "conf_d", "cf"):
            w = {"1": "Slot", "8": "Win"}[r["window"]]       # single slot / 8-slot window
            put(f"auc{r['stat'].replace('_', '')}{cname(r['cond'])}{w}", float(r["auc_mean"]))
    # detector relevance: mean Spearman rho over seeds at 5 dB (per statistic)
    R_ = rd("detector_relevance.csv")
    for st, nm in (("comb", "Comb"), ("tdisp", "Tdisp"), ("conf_d", "Confd"), ("conf_a", "Confa"),
                   ("resid", "Resid"), ("crcfail_d", "Crcfaild"), ("cf", "Cf"), ("ymag", "Ymag")):
        v = [float(r["spearman_rho"]) for r in R_
             if r["stat"] == st and float(r["snr_db"]) == 5.0 and r["spearman_rho"] not in ("", "nan")]
        if v:
            put("rho" + nm, float(np.mean(v)), "{:.2f}")
    # adaptation
    for r in rd("adapt_summary.csv"):
        snr = alpha[str(int(float(r["snr_db"])))]
        base = f"ad{cname(r['cond'])}{snr}{rxname[r['receiver']]}{r['mode'].capitalize()}"
        put(base + "First", float(r["first200"])); put(base + "Last", float(r["last200"]))
        put(base + "Upd", float(r["accepted_updates"]), "{:.0f}"); put(base + "Fp", float(r["false_pass_updates"]), "{:.0f}")
        put(base + "Pass", float(r["initial_pass_rate"]), "{:.2f}"); put(base + "Ms", float(r["median_update_ms"]), "{:.0f}")
    # can adaptation start? per (seed, receiver) at the severe shift (S2, 3 dB), CRC-gated mode
    G = [r for r in rd("adapt_summary_per_seed.csv")
         if r["mode"] == "gated" and float(r["snr_db"]) == 3.0]
    if G:
        started = [r for r in G if float(r["initial_pass_rate"]) > 0]
        recovered = [r for r in started if float(r["last200"]) < 0.1]
        put("adaptRunsN", str(len(G)))
        put("adaptStartN", str(len(started)))
        put("adaptRecovN", str(len(recovered)))
        put("adaptStuckN", str(len(G) - len(started)))
        if recovered:
            put("adaptRecovBest", min(float(r["last200"]) for r in recovered))
            put("adaptRecovWorst", max(float(r["last200"]) for r in recovered))
        stuck = [r for r in G if float(r["initial_pass_rate"]) == 0]
        if stuck:
            put("adaptStuckBler", min(float(r["last200"]) for r in stuck))
        for rx, tag in (("AINR", "Ainr"), ("DiscNRX", "Disc")):
            per_rx = [r for r in G if r["receiver"] == rx]
            put(f"adaptSeeds{tag}N", str(len(per_rx)))
            put(f"adaptStart{tag}N", str(sum(float(r["initial_pass_rate"]) > 0 for r in per_rx)))
    # 5 dB: how many (seed, receiver) runs each update mode brings below BLER 0.01
    P5 = [r for r in rd("adapt_summary_per_seed.csv") if float(r["snr_db"]) == 5.0]
    if P5:
        put("recFiveRunsN", str(len({(r["seed"], r["receiver"]) for r in P5})))
        for md, tag in (("gated", "Gated"), ("ungated", "Ungated"), ("trig", "Trig")):
            put(f"recFive{tag}N", str(sum(float(r["last200"]) < 0.01 for r in P5 if r["mode"] == md)))
        ung5 = [float(r["false_pass_updates"]) for r in P5 if r["mode"] == "ungated"]
        if ung5:
            put("ungFiveFpRunsN", str(sum(f > 0 for f in ung5)))
            put("ungFiveFpMax", max(ung5), "{:.0f}")
    # ungated self-training: updates whose own CRC passed although the block was in error
    U = [r for r in rd("adapt_summary_per_seed.csv")
         if r["mode"] == "ungated" and float(r["snr_db"]) == 3.0]
    if U:
        fps = [float(r["false_pass_updates"]) for r in U]
        put("ungRunsN", str(len(U)))
        put("ungFpRunsN", str(sum(f > 0 for f in fps)))
        put("ungFpMax", max(fps), "{:.0f}")
    # inversion
    for r in rd("inversion_summary.csv"):
        base = "inv" + cname(r["config"]) + r["init"].capitalize()
        put(base + "Info", float(r["final_info_ber_mean"])); put(base + "Coded", float(r["final_coded_ber_mean"]), "{:.4f}")
        put(base + "Bler", float(r["post_bp_bler_mean"])); put(base + "Ginfo", float(r["final_grad_info_mean"]), "{:.3f}")
    # model mismatch (dB)
    mm = rd("model_mismatch.csv")
    for r in mm:
        put(f"mm{cname(r['cond'])}", float(r["e_total_db"]), "{:.1f}")
        put(f"mmTime{cname(r['cond'])}", float(r["e_time"]), "{:.3f}")
    ds = [r for r in mm if r["cond"].startswith("DS")]
    if ds:
        worst = max(ds, key=lambda r: float(r["e_total_db"]))
        put("mmDSmax", float(worst["e_total_db"]), "{:.1f}")
        put("mmDSmaxCond", worst["cond"].replace("DS", "") + "~ns")
    # ablation shift
    for r in rd("ablation_shift_summary.csv"):
        put(f"abl{r['variant'].replace('_', '').capitalize()}{cname(r['cond'])}", float(r["bler_mean"]))
    for v, tag in (("no_detach", "Nodetach"), ("pure_vfe", "Purevfe")):
        P = [r for r in rd("ablation_paired.csv") if r["variant"] == v]
        if P:
            put(f"ablPair{tag}Ncond", str(len(P)))
            put(f"ablPair{tag}Nsig", str(sum(float(r["p_holm"]) < 0.05 for r in P)))
            put(f"ablPair{tag}Nworse", str(sum(float(r["mean_diff_vs_full"]) > 0 for r in P)))
            put(f"ablPair{tag}MinP", pfmt(min(float(r["p_holm"]) for r in P)))
    abl_seeds = {r["seed"] for r in rd("ablation_all.csv") if r.get("variant") == "full"}
    if abl_seeds:
        put("nSeedsAbl", str(len(abl_seeds)))
    # matched: SNR at BLER = 0.1 (mean over seeds) and gaps to the AINR
    cr = rd("matched_snr01.csv")
    if cr:
        for qam in ("16", "64"):
            mean = {}
            for rx in ("AINR", "DiscNRX", "LS-LIN", "LS-LMMSE", "PCSI"):
                v = [float(r["snr_at_bler_0p1"]) for r in cr if r["qam"] == qam and r["receiver"] == rx]
                v = [x for x in v if x == x]
                if v:
                    m, lo, hi, n = tci(v); mean[rx] = m
                    q = "Qsixteen" if qam == "16" else "Qsixtyfour"
                    # two decimals, as in the table: gaps of < 0.1 dB must not round to a
                    # contradiction between the quoted SNRs and the quoted gap
                    put(f"snrTen{q}{rxname[rx]}", m, "{:.2f}")
                    if n > 1:
                        put(f"snrTen{q}{rxname[rx]}Ci", (hi - lo) / 2, "{:.2f}")
            q = "Qsixteen" if qam == "16" else "Qsixtyfour"
            for rx in ("LS-LIN", "LS-LMMSE"):
                if rx in mean and "AINR" in mean:
                    put(f"gap{q}{rxname[rx]}", mean[rx] - mean["AINR"], "{:.2f}")   # dB the classical rx needs more
            if "PCSI" in mean and "AINR" in mean:
                put(f"gap{q}ToPcsi", mean["AINR"] - mean["PCSI"], "{:.2f}")         # dB the AINR is from perfect CSI
        seeds16 = {r["seed"] for r in cr if r["qam"] == "16" and r["receiver"] == "AINR"}
        put("nSeedsMatched", str(len(seeds16)))
    # S4
    snrword = {0: "zero", 5: "five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twentyfive"}
    for r in rd("s4_summary.csv"):
        put(f"sfour{rxname[r['receiver']]}{snrword[int(float(r['snr_db']))]}", float(r["bler_mean"]))
    for r in rd("s4_adapt_summary.csv"):
        base = f"sfourAd{rxname[r['receiver']]}{'five' if float(r['snr_db']) == 5 else 'fifteen'}"
        put(base + "Before", float(r["bler_before"])); put(base + "After", float(r["bler_after"]))
        put(base + "Fp", float(r["false_pass_total"]), "{:.0f}")
    # complexity (median latency, ms)
    cpath = os.path.join(RES, "complexity.csv")
    if os.path.exists(cpath):
        lab = {("detection", "AINR"): "latDetAinr", ("detection", "DiscNRX"): "latDetDisc",
               ("detection", "LS-LIN"): "latDetLslin", ("detection", "LS-LMMSE"): "latDetLmmse",
               ("cnn", "AINR"): "latCnnAinr", ("ldpc", "shared"): "latLdpc", ("monitor", "AINR S=1"): "latMonOne",
               ("monitor", "AINR S=4"): "latMonFour", ("crc", "shared"): "latCrc", ("adapt", "AINR"): "latAdaptAinr",
               ("adapt", "DiscNRX"): "latAdaptDisc"}
        for r in read(cpath):
            k = lab.get((r["stage"], r["receiver"]))
            if k and r.get("median_ms") not in (None, "", "nan"):
                b = "One" if r["batch"] in ("1", "1.0") else "Batch"
                put(k + b, float(r["median_ms"]), "{:.1f}")
    # broadly trained DiscNRX (one training seed)
    for r in rd("broad_summary.csv"):
        snr = "five" if float(r["snr_db"]) == 5 else "ten"
        put(f"brBler{cname(r['cond'])}{snr}", float(r["bler_Broad"]))
        if r.get("auc_conf_b") not in (None, ""):
            put(f"brAuc{cname(r['cond'])}{snr}", float(r["auc_conf_b"]))
    _bm = {r["receiver"]: float(r["snr_at_bler_0p1"]) for r in rd("broad_matched_snr01.csv")}
    for rx, v in _bm.items():
        put(f"brSnrTen{rxname.get(rx, 'Broad')}", v, "{:.2f}")
    if "Broad" in _bm and "DiscNRX" in _bm:                 # matched-channel cost of broad training
        put("brSnrTenGap", _bm["Broad"] - _bm["DiscNRX"], "{:.2f}")
    for r in rd("broad_s4.csv"):
        put(f"brSfour{rxname.get(r['receiver'], 'Broad')}{snrword[int(float(r['snr_db']))]}", float(r["bler"]))
    # budget
    B_ = rd("budget_summary.csv")
    if B_:
        put("nSeedsBudget", str(max(int(float(r["n_seeds"])) for r in B_)))
    for r in B_:
        step = {12000: "Twelve", 25000: "Twentyfive", 50000: "Fifty"}.get(int(r["step"]))
        if step:
            put(f"bud{rxname[r['receiver']]}{step}{cname(r['cond'])}", float(r["bler_mean"]))
    path = os.path.join(ROOT, "FINAL_OJCOMS", "tables", "results_macros.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write("% Auto-generated by analyze_results.py -- do not edit by hand.\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    os.replace(path + ".tmp", path)
    print(f"  -> FINAL_OJCOMS/tables/results_macros.tex ({len(M)} macros)")


PARTS = {"shift": analyze_shift, "adapt": analyze_adapt, "inversion": analyze_inversion,
         "ablation": analyze_ablation, "budget": analyze_budget, "mismatch": analyze_mismatch,
         "matched": analyze_matched, "s4": analyze_s4, "broad": analyze_broad,
         "figures": export_figures, "macros": export_macros}
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=list(PARTS))
    a = ap.parse_args()
    for p in a.parts:
        print(f"== {p}")
        PARTS[p]()
