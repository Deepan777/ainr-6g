# Analysis Protocol (frozen before the held-out seeds are examined)

Frozen: 2026-09-11, 16:00 IST. Purpose: fix every statistic, threshold rule and summary
**before** results of training seeds 1–4 exist, so they serve as an untouched held-out
test. Deviations after this point must be listed in §8 with a reason.

## 1. What had been seen when this protocol was written

* Seed 0 (50k-step checkpoints): per-slot BLER and all drift statistics for the 15
  conditions at 5 and 10 dB (`shift_slots_seed0.csv`), seen as single-slot AUCs and BLER
  tables. Adaptation at S2 for 3 and 5 dB (`adapt_S2_snr{3,5}_seed0.csv`).
* Controlled inversion (all 45 runs).
* Ablation seed 1 (12k-step checkpoints, pre-existing): matched BLER; shift BLER of `full` only.
* The 12k-step multi-seed log (`multiseed.log`).
* **Not seen:** anything from training seeds 1–4 at 50k steps; ablation seeds 2–3; snapshots.

The temporal-dispersion statistic was defined (in `exp_common.ainr_monitor`) before any
OJ-COMS result existed. Its motivation: the likelihood assumes a block-constant channel,
so time variation should appear as unequal per-symbol residuals. Seed 0 is development
data for it; seeds 1–4 are the test.

## 2. Statistics (per slot)

| Symbol | Definition | Direction (decided a priori) |
|---|---|---|
| `resid` | ℰ(Y) = mean over REs of E_q\|Y − G(c,h)\|², S = 4 posterior samples | one-sided, high = shift |
| `tdisp` | Var_ℓ(r_ℓ)/Mean_ℓ(r_ℓ)², r_ℓ = per-OFDM-symbol mean residual | one-sided, high = shift |
| `conf_a`, `conf_d` | mean over coded bits of \|tanh(LLR/2)\| (AINR, DiscNRX) | one-sided, low = shift |
| `crcfail_a`, `crcfail_d` | 1 − CRC pass (receiver-side CRC16) | one-sided, high = shift |
| `cf`, `ct` | lag-1 frequency correlation / pilot-symbol correlation of LS pilot estimates | **two-sided** (\|z\|) |
| `ymag` | mean \|Y\| over the slot | **two-sided** (\|z\|) |
| `comb` | max(z_resid, z_tdisp) | one-sided, high = shift |

z-scores use the mean and standard deviation of the statistic on the **S1 validation stream
at the same SNR and seed**. Two-sided statistics are scored as \|z\|.

## 3. Detection metrics

* AUC (Mann–Whitney) of each statistic, shifted-condition test slots versus S1 test slots,
  per seed and SNR.
* Fixed false-alarm operating points α ∈ {1 %, 5 %, 10 %}: threshold τ_α = (1−α) empirical
  quantile of the S1 **validation** stream (128 slots per seed and SNR). Report TPR on the
  shifted test stream and the **realised** FPR on the S1 test stream. The validation stream
  is small, so the realised FPR can differ from α; both are reported.
* Windows W ∈ {1, 8, 32}: the statistic is averaged over non-overlapping groups of W test
  slots of one condition. Validation thresholds use windows of the same length. Slots are
  i.i.d. draws, so windows model monitoring over W slots of a stationary condition.
* Relevance: a condition is *harmful* for a receiver at an SNR if its BLER exceeds the S1
  BLER of that receiver by ≥ 0.01. Report Spearman's ρ between the per-condition median
  z-statistic and ΔBLER over the 14 shifted conditions, per seed.
* Severity curves: delay spread {100 (S1), 200, 300, 600, 1000} ns (CDL-C); speed
  {3 (S1), 30, 90, 150, 250, 400} km/h (CDL-C 100 ns). BLER, median statistic and AUC versus
  severity.

## 4. BLER summaries

* Per seed: BLER with Clopper–Pearson 95 % intervals (exact counts reported).
* Across seeds (n = 5 for 16-QAM): mean and 95 % t-interval of seed-level BLERs.
* SNR at BLER = 0.1: linear interpolation of log10(BLER) between the two bracketing SNR
  points, per seed; mean and 95 % t-interval.
* AINR versus DiscNRX: per-seed paired differences, mean and 95 % t-interval; sign count.
  No difference is claimed unless the interval excludes 0. Because both detection paths
  are identical in architecture and loss, any difference is reported as between-run
  variation, not as an effect of the generative model.
* Classical baselines are deterministic given the channel draws; they are evaluated on the
  same slot streams.

## 5. Adaptation summaries (per seed × SNR × receiver × mode)

* First-200 and last-200 slot BLER; rolling BLER (window 50).
* Recovery slot: first slot t such that the rolling-50 BLER stays below 0.10 for the rest of
  the stream (∞ if never).
* Accepted updates, rejected updates (CRC fail), false-pass updates (CRC pass but block
  wrong; ground truth used for reporting only), median update latency.
* Initial CRC pass rate of the un-adapted receiver (first 100 slots of mode `none`) as the
  explanatory variable for bootstrapping.
* Across seeds: mean and 95 % t-interval; mode comparisons as paired differences.

## 6. Ablation and training-budget summaries

* Ablation (seeds 1–3, 12k steps): matched BLER at 0/1/3/5 dB and shift BLER at 5 dB (S1,
  DS600, DS1000, S2, V250). Paired differences `no_detach − full` and `pure_vfe − full`
  per seed. Three seeds is descriptive only; conclusions stated with that caveat.
* Budget (seeds 2–4): BLER versus training step (12k, 25k, 50k), both receivers, 5 dB.
  Hypothesis H_B (formed from ablation seed 1 vs seed 0; *not* yet tested): shift BLER
  increases with training steps while S1 BLER decreases. Descriptive; supported only if the
  direction holds in all three seeds for DS1000 and S2.

## 7. Statistical tests

Pre-specified tests only, all two-sided, α = 0.05, Holm correction within each family:
* Family F1: paired t-tests of AINR − DiscNRX BLER per shifted condition at 5 dB (n = 5).
* Family F2: paired t-tests of `no_detach − full` and `pure_vfe − full` per shift condition
  (n = 3; low power stated).

Everything else is descriptive (intervals, no p-values). With 5 seeds, the power to detect
small differences is low; non-significant results are reported as "not distinguishable",
never as "equal".

## 8. Deviations log

| Date | Deviation | Reason |
|---|---|---|
| 2026-09-11 | Windowed thresholds (W > 1) are the (1−α) quantiles of 4,000 bootstrap window means drawn from the 128 S1 validation slots, not of disjoint validation windows | 128 validation slots give only 16 (W = 8) or 4 (W = 32) disjoint windows, too few for 1–10 % quantiles; the bootstrap still uses validation data only. Realised FPRs on the S1 test stream are reported and run above nominal (e.g. 0.09 at α = 0.05, W = 8, seed 0). |
| 2026-09-11 | Added hypothesis N2′ (ordering full < no_detach < pure_vfe of shift BLER) after seeing ablation seed 1 | Formed on seed 1; seeds 2–3 are its test. Reported as exploratory if they disagree. |
| 2026-09-11 | Matched SNR grids changed from uniform 0.5 dB (16-QAM −2…8 dB; 64-QAM 4…14 dB) to 0.1 dB steps across the waterfalls (16-QAM −1, −0.5, 0…2, 2.5…3.5, 4…6 dB; 64-QAM 3, 3.5, 4…6.5, 7, 7.5, 7.6…9.5, 10 dB); seed-0 runs repeated on the new grids (64-QAM coarse crossings: AINR 5.44, DiscNRX 5.42, LS-LMMSE 5.52, LS-LIN 8.48, PCSI 5.09 dB) | The first seed-0 run placed only two points on each learned receiver's waterfall, so the SNR at BLER 0.1 was set by interpolation between 1.0 and ≈0.05. The change affects resolution only; the coarse run is kept (`*_coarse.csv`) and its crossings (AINR 0.87, DiscNRX 0.86, LS-LMMSE 1.06, LS-LIN 4.90, PCSI 0.56 dB) are recorded in the experiment ledger. |
| 2026-09-12 | Post-hoc diagnostic: seed 1's ungated run (DiscNRX, S2, 3 dB) was replayed slot by slot to inspect the decoded bits | The pre-registered analysis counts false-pass updates but does not explain them. The replay uses the same checkpoint, stream seed, learning rate and mode as the logged run and only reads its decisions; it adds no experimental condition. It explains a recorded number (900 of 1000 updates accepted on erroneous blocks): the receiver saturates to the all-zero codeword, whose transport-block CRC is also zero. Reported as an explanation, not as a separate result. |
| 2026-09-12 | Presentation of the pre-registered recovery slot changed in the adaptation table: instead of listing one slot per seed, the cell gives how many runs recovered out of all runs and the median slot among them | The per-seed list grew with the seed count and overflowed the table's text block (3.85 pt at four seeds, worse at five). The metric itself is unchanged, the per-seed slots remain in `data/adapt_summary_per_seed.csv`, and the cell format is stated in the table caption. |
| 2026-09-12 | Added descriptive per-run counts for the adaptation family: how many (seed, receiver) runs had a non-zero initial CRC pass rate, how many recovered, and how many fell below BLER 0.01 per mode at 5 dB | The protocol summarises adaptation by BLER, updates, false passes and recovery slot averaged over seeds. Where runs diverge (one seed recovers, another collapses) a mean describes no actual run, so the counts are reported alongside the means, and the figure shows the seed interval for the ungated mode as well. No hypothesis test is performed on these counts. |
| 2026-09-11 | Added a broadly trained DiscNRX (E15/T05) as an extra control, not in the original design | Answers reviewer risk R12 (shifts may be in-distribution for a receiver trained on randomised channels). Its analysis is descriptive: BLER per condition with Clopper–Pearson intervals, the paired difference to the narrow DiscNRX on identical slots, and the AUC of its confidence statistic with the same S1-validation thresholds as §3. One training seed only; no hypothesis test. |
