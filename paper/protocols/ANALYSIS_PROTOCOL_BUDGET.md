# Analysis Protocol — Training-Budget Stopping Criterion

Frozen: 2026-09-14, before any dense-snapshot checkpoint existed (the training driver
`run_dense.py` was launched the same hour and the first snapshot had not been evaluated).
Companion to `ANALYSIS_PROTOCOL.md`, which governs the drift-detection study. Purpose: fix
the candidate signals, the selection rules, the held-out split and the success metric
**before** the trajectory data exists. Deviations must be listed in §7 with a reason.

## 1. What had been seen when this protocol was written

* The three-point budget study on seeds 2–4 (12k/25k/50k steps), reported in
  `results/ojcoms/budget_curve.csv`, `pilot_reliance.csv`, `pilot_reliance_margin.csv`,
  `budget_stop_signal.csv`. Specifically: the matched crossing falls 1.13 → 0.86 dB while
  S2 BLER rises 0.005 → 0.544; margin-controlled pilot tolerance rises +1.71 dB (6/6 pairs);
  matched free energy and pilot tolerance each correlate with S2 BLER at Spearman 0.783
  over nine AINR snapshots.
* **Not seen:** every dense snapshot (1000–50000 steps), and everything from training
  seeds 5, 6, 7 and 8, which did not exist.

Seeds 2–4 are therefore **development** seeds: their coarse behaviour informed this design.
Seeds 5–8 are **held-out**; the primary test is computed on them alone.

## 2. Snapshots

26 per model per seed: every 1000 steps to 10000, then every 2500 to 50000. Both AINR and
the parameter-matched DiscNRX are snapshotted from the same training run, so the pair shares
its data order and channel realisations.

## 3. Candidate signals (matched data only)

A stopping criterion may use **no out-of-distribution data**. Every candidate is computed on
held-out S1 (matched) validation slots at 5 dB. The direction that is expected to indicate
"stop" is fixed here, a priori.

| Signal | Definition | Direction meaning "stop now" |
|---|---|---|
| `fe` | mean free energy F on matched validation slots (AINR only) | rising |
| `ell` | mean expected log-likelihood E[log p(Y\|c,h,σ²)] | rising (less negative) |
| `klh` | mean KL[q(h)‖p(h)] | rising |
| `kls` | mean KL[q(σ²)‖p(σ²)] | rising |
| `ptol` | margin-controlled pilot-noise tolerance: extra pilot-noise multiplier (dB) at which matched BLER reaches 0.5, measured at the snapshot's own BLER-0.1 operating point | rising |
| `mbler` | matched BLER at 5 dB — the trivial accuracy-only control | falling (never stops) |

`fe`, `ell`, `klh`, `kls` exist only for AINR. `ptol` and `mbler` exist for both receivers;
they are the signals available to a purely discriminative receiver. That asymmetry is the
point of the experiment and is declared here, not after the fact.

## 4. Selection rules

For a signal s(t) over the snapshot steps t, with direction d fixed in §3:

* **R1 (turning point).** Choose the smallest t for which s has risen (fallen, if d is
  falling) by more than 20 % of its total range over the run, evaluated on the trajectory
  smoothed with a centred 3-snapshot moving average.
* **R2 (knee).** Choose the smallest t at which the matched-accuracy improvement per 1000
  steps falls below 0.01 dB — a conventional early-stopping rule that uses accuracy only.

R1 is applied to `fe`, `ell`, `klh`, `kls`, `ptol`. R2 is applied to `mbler`. The 20 % and
0.01 dB constants are fixed now and will not be tuned; if a rule degenerates (selects the
first or last snapshot in every run) that outcome is reported as the result.

## 5. Success metric

Each run's dense trajectory defines a frontier of (matched cost, S2 BLER), where matched
cost is the snapshot's SNR at BLER 0.1 minus the best such SNR achieved anywhere in that run.
A criterion selects one point on it.

* **Matched-cost cap.** 0.30 dB, fixed from the 0.27 dB measured in the three-point study.
  A criterion whose selected snapshot exceeds the cap is scored as a failure for that run.
* **Primary endpoint.** S2 BLER at the selected snapshot versus S2 BLER at 50000 steps
  (the default of simply training to convergence), paired across the four held-out seeds,
  one-sided (the criterion is predicted to be lower).
* **Secondary endpoint.** Regret: S2 BLER at the selected snapshot minus the lowest S2 BLER
  of any snapshot within the cap in that run (the oracle).
* **Multiplicity.** The primary endpoint is declared for `fe` alone. All other signals are
  secondary and reported with Holm correction across the five remaining signals.

## 6. Pre-declared negative outcome

If `fe` does not beat the 50k default on the held-out seeds, that is reported as a negative
result in the manuscript, and the budget study is reported as the trade-off curve plus its
pilot-reliance mechanism only, with the explicit statement that no validation-only stopping
criterion was found. The three-point study already shows the free-energy spread **between**
seeds (97.6) exceeds its range **within** a seed across budgets (49.0), so an absolute
threshold is known in advance not to be usable; only the within-run trend is tested.

## 7. Deviations and clarifications

**Clarification, 2026-09-15** (no dense-snapshot data had been analysed at this point:
`dense_eval.csv` held one smoke-test row, and the four held-out seeds had not finished
training). §4 says R1 fires when the signal "has risen by more than 20 % of its total range".
That wording does not say what the rise is measured *from*. The implementation
(`analyze_budget.py`, written before the held-out data existed) measures it from the running
extreme: R1 selects the smallest t for which

    s(t) - min_{t' <= t} s(t')  >  0.20 x (max s - min s)

over the 3-snapshot smoothed trajectory, with the sign flipped for a falling direction. Range
is taken over the whole run. This is fixed now and will not be revisited.

`analyze_budget.py --selftest` checks the rules on constructed trajectories: a flat signal
never triggers R1, a signal rising at a known index triggers within one snapshot of it, R2
fires at the first knee, the Holm correction matches its definition, and the matched-cost cap
correctly excludes cheap-but-inaccurate early snapshots from the oracle.

**Amendment, 2026-09-15 — R1 range made robust to the pre-convergence transient.**

*Found on development data.* Free-energy trajectories were computed for development seeds
2-4 (`exp_dev_fe.py` -> `results/ojcoms/dev_fe_trajectory.csv`). R1 as written **never fires**
on the `fe` signal in any of the three runs. The cause is a scaling artefact, not an absence
of signal: free energy is ~36 500 at step 1000, falls to ~14 170 by step 2000 and is flat
thereafter, so max - min is dominated by that single early transient. A threshold of 20 % of
that range demands a rise of ~4 500, while the post-convergence signal spans only ~140.

*The signal is real.* Re-evaluating one fixed checkpoint (seed 2, step 25000) under six
different evaluation seeds gives a free-energy standard deviation of 6.8, against an
across-step standard deviation of 36.6 over the 25 post-convergence snapshots of the same run
— a ratio of 5.35. The trajectory is therefore resolvable structure, not estimator noise.

*The amendment.* The range in R1 is computed as the 10th-to-90th percentile interval of the
smoothed trajectory instead of max - min. Nothing else changes: the 20 % constant, the
direction table, the smoothing, the cap, the endpoints and the multiplicity rule all stand.
Snapshots from the transient can still be selected, and are still scored against the 0.30 dB
matched-cost cap, which rejects them on their own merits (step 1000 sits 1.1 dB above the
best matched accuracy of its run).

*Why this is legitimate.* Seeds 2-4 are development data under section 1, which exists for
exactly this purpose. At the time of the amendment `results/ojcoms/dense_eval.csv` contained
**one** row — seed 2, step 1000, a development seed — and **no** held-out seed had been
evaluated, so no out-of-distribution outcome for any dense snapshot was available to tune
against. The amended rule is non-degenerate on the development seeds (fe selects 10000, 9000
and 4000 steps) and its selections are unchanged for thresholds of 0.15, 0.20 and 0.25,
so it is not balanced on the constant.

*(no other deviations)*

**Correction note, 2026-09-17 (later the same day).** The noise check quoted in the amendment above
(standard deviation 6.8, ratio 5.35) was run interactively and not saved. It was repeated with a saved
script (`exp_dev_fe_noise.py` -> `results/ojcoms/dev_fe_noise.csv`) and gives a standard deviation of 5.3
against the same across-step value of 36.6 (ratio 6.9); the difference is attributable to GPU
non-determinism. The conclusion that the trajectory carries structure beyond estimator noise is unchanged,
and the manuscript reports the saved values. This note was added after the held-out evaluation; it concerns
development data only and changes no rule.

