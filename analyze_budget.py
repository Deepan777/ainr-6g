"""Stopping-criterion analysis, implementing FINAL_OJCOMS/audits/ANALYSIS_PROTOCOL_BUDGET.md.

Written before the held-out seeds finished training, so the analysis could not be
shaped by its own result. Reads results/ojcoms/dense_eval.csv.

Usage:  python analyze_budget.py [--csv PATH] [--selftest]
"""
import argparse, csv, itertools, os, sys
import numpy as np
from scipy import stats

DEV_SEEDS, HELD_OUT = [2, 3, 4], [5, 6, 7, 8]
CAP_DB = 0.30           # matched-cost cap (protocol section 5)
RISE_FRAC = 0.20        # R1 constant
KNEE_DB = 0.01          # R2 constant, dB per 1000 steps
DEFAULT_STEP = 50000
# signal -> direction that means "stop now"
SIGNALS = {"fe": "rise", "ell": "rise", "klh": "rise", "kls": "rise",
           "ptol": "rise", "mbler": "fall"}
PRIMARY = "fe"


def smooth3(v):
    """Centred 3-point moving average, endpoints held."""
    v = np.asarray(v, float)
    if len(v) < 3:
        return v.copy()
    out = v.copy()
    out[1:-1] = (v[:-2] + v[1:-1] + v[2:]) / 3.0
    return out


def rule_R1(steps, vals, direction):
    """Smallest step at which the signal has moved away from its running extreme by
    more than RISE_FRAC of the trajectory's total range."""
    s = smooth3(vals)
    if not np.all(np.isfinite(s)) or len(s) < 2:
        return None
    # Robust range (protocol section 7 amendment, 2026-09-15): max - min is dominated
    # by the pre-convergence transient, which stops R1 ever firing on the free energy.
    rng = float(np.nanpercentile(s, 90) - np.nanpercentile(s, 10))
    if rng <= 0:
        return None
    if direction == "fall":
        s = -s
    run_min = s[0]
    for t, x in zip(steps, s):
        run_min = min(run_min, x)
        if x - run_min > RISE_FRAC * rng:
            return t
    return None


def rule_R2(steps, snr):
    """Knee: smallest step at which matched-accuracy improvement falls below
    KNEE_DB per 1000 steps. Improvement = decrease in the SNR at BLER 0.1."""
    steps, snr = np.asarray(steps, float), np.asarray(snr, float)
    for i in range(1, len(steps)):
        d = steps[i] - steps[i - 1]
        if d <= 0:
            continue
        improve_per_1k = (snr[i - 1] - snr[i]) * 1000.0 / d
        if np.isfinite(improve_per_1k) and improve_per_1k < KNEE_DB:
            return steps[i]
    return None


def load(path):
    runs = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["seed"]), r["receiver"])
            runs.setdefault(key, []).append(r)
    out = {}
    for key, rows in runs.items():
        rows.sort(key=lambda r: int(r["step"]))
        d = {"step": [int(r["step"]) for r in rows]}
        for c in ("snr_at_0p1", "bler_S1_5db", "bler_S2", "bler_DS1000",
                  "fe", "ell", "klh", "kls", "ptol"):
            d[c] = [float(r[c]) if r.get(c) not in ("", None) else np.nan for r in rows]
        d["mbler"] = d["bler_S1_5db"]
        out[key] = d
    return out


def evaluate(run, sig):
    """Apply the rule for `sig`, return (selected_step, cost_db, s2_at_selected, capped_ok)."""
    steps, snr = run["step"], np.asarray(run["snr_at_0p1"], float)
    sel = rule_R2(steps, snr) if sig == "mbler" else rule_R1(steps, run[sig], SIGNALS[sig])
    if sel is None:
        return None, np.nan, np.nan, False
    i = steps.index(sel)
    best = np.nanmin(snr)
    cost = float(snr[i] - best)
    return sel, cost, float(run["bler_S2"][i]), bool(cost <= CAP_DB + 1e-9)


def oracle(run):
    """Lowest S2 BLER among snapshots within the matched-cost cap."""
    snr = np.asarray(run["snr_at_0p1"], float)
    best = np.nanmin(snr)
    ok = [i for i in range(len(snr)) if np.isfinite(snr[i]) and snr[i] - best <= CAP_DB + 1e-9]
    return float(np.nanmin([run["bler_S2"][i] for i in ok])) if ok else np.nan


def holm(pvals, labels):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, prev = len(pvals), {}, 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * pvals[i]))
        out[labels[i]] = adj
        prev = adj
    return out


def export(path, outdir="FINAL_OJCOMS/data"):
    """Write the figure data for the trade-off frontier.

    fig_frontier.csv    per training step: mean matched cost and mean OOD BLER over
                        the held-out seeds, with a 95 % interval over seeds.
    fig_frontier_sel.csv the snapshot each criterion selects, for marking on the figure.
    """
    runs = load(path)
    ho = [s for s in HELD_OUT if (s, "ainr") in runs]
    if not ho:
        print("no held-out seeds yet -- nothing to export")
        return
    os.makedirs(outdir, exist_ok=True)
    steps = sorted({t for s in ho for t in runs[(s, "ainr")]["step"]})
    rows = []
    for t in steps:
        cost, s2 = [], []
        for s in ho:
            r = runs[(s, "ainr")]
            if t not in r["step"]:
                continue
            i = r["step"].index(t)
            snr = np.asarray(r["snr_at_0p1"], float)
            cost.append(snr[i] - np.nanmin(snr))
            s2.append(r["bler_S2"][i])
        if not cost:
            continue
        s2 = np.asarray(s2, float)
        lo, hi = (np.nanpercentile(s2, 2.5), np.nanpercentile(s2, 97.5)) if len(s2) > 1 else (np.nan, np.nan)
        rows.append({"step": t, "cost": float(np.nanmean(cost)),
                     "S2": float(np.nanmean(s2)),
                     "S2_lo": float(max(lo, 5e-4)) if np.isfinite(lo) else "nan",
                     "S2_hi": float(hi) if np.isfinite(hi) else "nan"})
    p1 = os.path.join(outdir, "fig_frontier.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "cost", "S2", "S2_lo", "S2_hi"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # The figure marks only the primary signal's selections; pgfplots has no row
    # filter defined in this project's style file, so filter here instead.
    sel_rows = []
    for sig in [PRIMARY]:
        for s in ho:
            run = runs[(s, "ainr")]
            if np.all(np.isnan(run[sig])):
                continue
            step, cost, s2, ok = evaluate(run, sig)
            if step is not None:
                sel_rows.append({"seed": s, "step": step,
                                 "cost": round(cost, 4), "S2": s2, "within_cap": int(ok)})
    p2 = os.path.join(outdir, "fig_frontier_sel.csv")
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "step", "cost", "S2", "within_cap"])
        w.writeheader()
        for r in sel_rows:
            w.writerow(r)
    print(f"wrote {p1} ({len(rows)} steps) and {p2} ({len(sel_rows)} selections)")


def report(path):
    runs = load(path)
    have = sorted({s for (s, rx) in runs if rx == "ainr"})
    print(f"AINR runs present: seeds {have}")
    ho = [s for s in HELD_OUT if (s, "ainr") in runs]
    if not ho:
        print("No held-out seeds evaluated yet -- nothing to report.")
        return
    print(f"Held-out seeds available: {ho}\n")

    rows, praw, labels = {}, [], []
    for sig in SIGNALS:
        per = []
        for s in ho:
            run = runs[(s, "ainr")]
            if not np.all(np.isnan(run[sig])):
                sel, cost, s2, ok = evaluate(run, sig)
                i50 = run["step"].index(DEFAULT_STEP) if DEFAULT_STEP in run["step"] else -1
                per.append({"seed": s, "sel": sel, "cost": cost, "s2": s2, "ok": ok,
                            "s2_default": run["bler_S2"][i50], "oracle": oracle(run)})
        if not per:
            continue
        rows[sig] = per
        a = np.array([p["s2"] if p["ok"] else p["s2_default"] for p in per], float)
        b = np.array([p["s2_default"] for p in per], float)
        if len(a) >= 2 and np.any(a != b):
            t, p2 = stats.ttest_rel(a, b)
            p = p2 / 2 if t < 0 else 1 - p2 / 2      # one-sided: criterion lower
        else:
            p = 1.0
        praw.append(float(p)); labels.append(sig)
        print(f"--- {sig} ---")
        for q in per:
            print(f"   seed{q['seed']}: step={q['sel']} cost={q['cost']:+.2f} dB "
                  f"{'OK' if q['ok'] else 'OVER CAP'}  S2={q['s2']:.3f} "
                  f"(default {q['s2_default']:.3f}, oracle {q['oracle']:.3f})")
        print(f"   mean S2 selected={np.mean(a):.3f} vs default={np.mean(b):.3f}  "
              f"one-sided p={p:.4f}\n")

    if PRIMARY in rows:
        i = labels.index(PRIMARY)
        print(f"PRIMARY ENDPOINT ({PRIMARY}, declared in advance): p={praw[i]:.4f} "
              f"-- {'beats' if praw[i] < 0.05 else 'does NOT beat'} training to convergence")
    sec = [(l, p) for l, p in zip(labels, praw) if l != PRIMARY]
    if sec:
        adj = holm([p for _, p in sec], [l for l, _ in sec])
        print("\nSecondary signals (Holm-corrected):")
        for l, _ in sec:
            print(f"   {l:6s} p_adj={adj[l]:.4f}")


def selftest():
    """Rules must behave correctly on constructed trajectories."""
    steps = list(range(1000, 11000, 1000))
    flat = [5.0] * 10
    assert rule_R1(steps, flat, "rise") is None, "flat signal must not trigger"
    v = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]                 # rises from index 5
    got = rule_R1(steps, v, "rise")
    assert got in (6000, 7000), got                     # smoothing may shift by one
    got = rule_R1(steps, [-x for x in v], "fall")
    assert got in (6000, 7000), got
    snr = [3.0, 2.0, 1.5, 1.45, 1.445, 1.4449, 1.4448, 1.4447, 1.4446, 1.4445]
    assert rule_R2(steps, snr) == 5000, rule_R2(steps, snr)
    assert abs(holm([0.01, 0.04], ["a", "b"])["a"] - 0.02) < 1e-9
    r = {"step": steps, "snr_at_0p1": [1.5] * 5 + [1.0] * 5,
         "bler_S2": [0.01] * 5 + [0.5] * 5}
    assert abs(oracle(r) - 0.5) < 1e-9, "cap must exclude the high-cost early snapshots"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ojcoms/dense_eval.csv")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.export:
        export(a.csv)
    else:
        report(a.csv)
