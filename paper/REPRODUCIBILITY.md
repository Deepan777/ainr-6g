# Reproducibility Guide

This document lists the exact commands that regenerate every result, table, and figure of the
manuscript *Optimization-Dependent Robustness and Checkpoint Selection in Neural OFDM Receivers*
(under review at *Array*, Elsevier). All commands are run from the repository root unless stated
otherwise.

Section and figure numbers below are those of the Array manuscript. In the released tree the
manuscript's `figures/`, `tables/` and `data/` directories appear under `paper/`.

## 1. Environment

| Component | Version used |
|---|---|
| OS | Windows 11 Home (build 26200) |
| Python | 3.11.5 |
| PyTorch | 2.9.1 + CUDA 12.6, cuDNN 9.10.02 |
| Sionna | 2.0.1 |
| NumPy / SciPy / OmegaConf / psutil | 2.4.4 / 1.17.1 / 2.3.0 / 7.2.2 |
| GPU / driver | NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) / 596.08 |
| CPU / RAM | AMD Ryzen 7 6800H (16 logical cores) / 15.2 GB |
| LaTeX | MiKTeX 25.12 (pdfTeX 4.23); `elsarticle.cls` for the Array build |

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt          # Linux/macOS: venv/bin/pip
venv/Scripts/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
venv/Scripts/pip install psutil
```

All hyperparameters are in `config/config.yaml`. Every script records its random seeds; stream
seeds are fixed in the code, so reruns reproduce the block streams. GPU non-determinism (cuDNN
kernels) can change individual decisions, but not the reported statistics beyond their stated
intervals.

## 2. Training (Section 4.2)

| Model | Command | Output |
|---|---|---|
| 16-QAM, seed 0 | `python train.py` | `results/checkpoints/` |
| 16-QAM, seeds 1–4 | `python train.py paths.results_dir=results/seeds/seed<s> paths.checkpoint_dir=results/seeds/seed<s>/checkpoints train.seed=<s> train.session_steps=50000 train.snapshot_steps=[12000,25000]` | `results/seeds/seed<s>/` |
| 64-QAM, seed 0 | `python train.py phy.n_bits_per_symbol=6 paths.checkpoint_dir=results/qam64/checkpoints paths.results_dir=results/qam64` | `results/qam64/` |
| Ablation variants | `python exp_ablation.py --seed <s> [--variants full no_detach pure_vfe]` | `results/ablation/[seed<s>/]` |
| Broadly trained DiscNRX (CDL A–E, 30–1000 ns, 0–120 km/h) | `python train_broad.py` | `results/broad/` |

`train.py` is session-based and resume-safe: rerunning the same command continues from the last
checkpoint (written atomically every 1000 steps). The supervisor `python pipeline.py` runs the
complete experiment plan (`pipeline_steps.py`) in order, skips finished steps, and resumes
interrupted ones; `python pipeline.py --status` lists their state.

`train.py` offsets the DiscNRX data-stream seed by 7919, so the AINR and the DiscNRX of the same
seed index differ in initialization and in training data. The seed index is therefore not a
pairing factor, and the AINR-vs-DiscNRX comparison uses an unpaired Welch test (Section 4.8).
Comparisons between variants or between checkpoints of one trajectory are paired.

## 3. Experiments

| Paper item | Command | Raw output (`results/ojcoms/`) |
|---|---|---|
| Controlled inversion (Fig. 8) | `python exp_inversion.py` | `inversion_summary.csv`, `inversion_trace.csv` |
| Coded-bit vs information-bit latent (Fig. 8) | `python exp_latent_target.py` | `inversion_*` |
| Matched 16-QAM BLER (Section 5.1) | `python exp_matched.py --qam 16 --seeds <s>` | `matched_qam16_S1_seed<s>.csv`, `..._classical.csv` |
| 64-QAM BLER | `python exp_matched.py --qam 64 --seeds 0` | `matched_qam64_S1_seed0.csv` |
| Shift severity and all monitors (Sections 5.2, 5.6) | `python exp_shift_detect.py --seeds <s>` | `shift_slots_seed<s>.csv` (one row per slot) |
| Monitor AUCs, bootstrap CIs, operating points | `python exp_roc_stats.py` | `detector_*` |
| Likelihood representation error (Fig. 5) | `python exp_model_mismatch.py` | `model_mismatch.csv` |
| Genie attribution of the residual (Section 5.5) | `python exp_attribution.py` | `attribution.csv` |
| Pilot-overhead probe (Section 5.5) | `python exp_pilot_eval.py` | `pilot_reliance.csv`, `pilot_reliance_margin.csv`, `pilot1_matched.csv` |
| Online adaptation, real CRC16 (Section 5.7) | `python exp_adapt.py --cond S2 --snr <3\|5> --seeds <s>` | `adapt_S2_snr<snr>_seed<s>.csv` |
| Multi-seed adaptation, gating and LR ablations | `python exp_stage_d_stats.py`, `python exp_stage_d_harsh.py` | `adapt_*` |
| Ray-traced scene S4 (Section 5.2) | `python s4_gen.py` (channels), `python exp_s4.py --seeds <s>` | `s4_bler_*.csv`, `s4_adapt_seed<s>.csv` |
| S4 with confidence intervals | `python exp_s4_ci.py` | `s4_bler_*.csv` |
| Training budget vs robustness | `python exp_budget.py --seeds <s>` | `budget_shift_seed<s>.csv` |
| Ablation incl. the stop-gradient (Section 5.8) | `python exp_ablation.py --seed <s>` | `ablation_{bler,shift,summary}_seed<s>.csv` |
| Broadly trained DiscNRX control (Table 5) | `python exp_broad.py` | `broad_shift_slots.csv`, `broad_matched.csv`, `broad_s4_bler.csv` |
| Complexity (Table 9) | `python exp_complexity.py` (GPU otherwise idle) | `complexity.csv`, `complexity_env.json` |
| Inference latency protocol (Table 9) | `python exp_latency.py` | `complexity*.csv` |
| Matched BLER confidence intervals | `python ci_bler.py` | `results/stage_b_ci.csv` |
| Cross-seed training stability | `python multiseed.py` | printed |

Every experiment writes its results in pieces and resumes after an interruption; a `--quick` flag
runs a small smoke test that never overwrites real outputs.

## 3b. Training-budget study and adaptation probes (Sections 4.5, 4.6, 5.3, 5.4, 5.7)

The analysis plan is `paper/protocols/ANALYSIS_PROTOCOL_BUDGET.md` (written before any snapshot was
evaluated; one dated amendment made on development data). Seeds 2–4 are development data, seeds 5–8
are held out.

```bash
python run_dense_all.py            # trains seeds 2-8 with 26 snapshots each (AINR pass first, then
                                   # DiscNRX), then evaluates every snapshot; resumable at any point
                                   #   -> results/dense/seed<s>/checkpoints/, results/ojcoms/dense_eval.csv
python analyze_budget.py --selftest   # checks the stopping rules on constructed trajectories
python analyze_budget.py              # the pre-registered test, printed per signal and seed
python make_budget_macros.py          # paper/tables/budget_macros.tex, tab_supp_budget.tex,
                                      #   data/fig_frontier_*.csv (Fig. 3 and Section 5.3)
python exp_scramble_probe.py          # NR-style scrambling replay -> results/ojcoms/scramble_probe.csv
python exp_vfe_probe.py               # label-free free-energy adaptation -> results/ojcoms/vfe_probe_S2_snr3.csv
python exp_vfe_probe.py --seeds 1 3 --modes vfe --lr 3e-4 --tag _lr3e4   # learning-rate checks
python exp_vfe_probe.py --seeds 1 3 --modes vfe --lr 1e-3 --tag _lr1e3
```

The dense snapshots are 643 MB and are **not** in this repository; `results/ojcoms/dense_eval.csv`
holds their complete per-snapshot evaluation, which is what every downstream analysis reads. The
snapshots are available from the corresponding author on request.

Development-only diagnostics, not quoted as results: `exp_dev_fe.py` (free-energy trajectories of
seeds 2-4 that motivated the protocol amendment); `exp_budget_curve.py`, `exp_budget_margin.py`,
`exp_budget_stop.py` (the three-snapshot exploration that preceded the protocol);
`exp_dev_fe_noise.py` (the estimator-noise check behind the amendment).

On the machine used, the GPU is clocked down under sustained load, so the training driver pauses
2 s every 25 steps (`train.cooldown_every`, `train.cooldown_seconds`); this changes wall-clock time
only.

`train.lr_schedule=const` exists for a constant-learning-rate control that asks whether the
degradation is an artefact of the decaying schedule. That control is incomplete and is not
reported; no number in the manuscript depends on it.

## 3c. Independent revalidation and the shift-family controls (Sections 4.5, 5.3)

Checkpoints are *chosen* on the trajectory sweep and *measured* on fresh streams whose seeds are
disjoint from every earlier evaluation, so the reported values for the chosen checkpoints are not
biased by their selection.

```bash
python exp_highprec.py             # re-evaluates t_val, t_match, t_star, t_R1, t_R2 per trajectory
                                   #   on disjoint streams -> results/ojcoms/highprec.csv
python analyze_highprec.py         # gamma_0.1, Delta, kappa with block-bootstrap intervals
                                   #   -> results/ojcoms/highprec_summary.csv
python exp_traj_extra.py           # validation over the full training distribution D_tr (giving
                                   #   t_val) and Doppler along the trajectories -> traj_extra.csv
python exp_traj_doppler.py         # Doppler dose-response at 30, 90, 150 km/h -> traj_doppler.csv
python make_revalidation_macros.py # paper/tables/reval_macros.tex, data/fig_family_*.csv,
                                   #   data/fig_disc_traj.csv (Fig. 4, Supplementary Fig. S5)
```

The three selection criteria are distinct and are kept distinct throughout: `t_match` minimises the
matched SNR requirement, `t_val` minimises validation loss on the training distribution, and
`t*_eps` is the oracle — minimum shifted error subject to a matched cost of at most eps = 0.30 dB.

## 3d. Checkpoint selection from matched data only (Sections 4.6, 5.4)

Both protocols were written before any held-out trajectory was scored:
`paper/protocols/ANALYSIS_PROTOCOL_SOUP.md` and `paper/protocols/ANALYSIS_PROTOCOL_SOURCEONLY.md`.
The networks use GroupNorm, so an averaged model is the elementwise mean of the selected
checkpoints and no normalization statistics need recomputation.

```bash
python exp_soup.py                 # six averaging schemes (ALL, EARLY, MID, TAIL10/20/35)
                                   #   -> results/ojcoms/soup.csv
python exp_sourceonly.py           # four matched-data scores (effective rank, log-determinant,
                                   #   feature norm, sharpness) -> results/ojcoms/sourceonly.csv
python make_selection_macros.py    # paper/tables/selection_macros.tex, data/fig_soup.csv,
                                   #   data/fig_soup_wide.csv and the three supplementary
                                   #   tables tab_supp_{soup,sourceonly,selseed}.tex
```

Both results are retrospective, over a completed trajectory. They are not a prospective stopping
rule: the pre-registered test of online stopping rules (section 3b of this guide) is null, and the manuscript
says so.

## 4. Analysis, tables, figures, manuscript

```bash
python analyze_results.py          # statistics per paper/protocols/ANALYSIS_PROTOCOL.md
                                   #   -> paper/data/*.csv and paper/tables/results_macros.tex
python make_tables.py              # paper/tables/tab_*.tex
python make_budget_macros.py       # paper/tables/budget_macros.tex
python make_revalidation_macros.py # paper/tables/reval_macros.tex
python make_selection_macros.py    # paper/tables/selection_macros.tex
python paper/figures/make_figures.py   # standalone vector PDFs in paper/figures/pdf/
# manuscript build (in the manuscript directory, not released here):
#   pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every figure is a PGFPlots/TikZ source in `paper/figures/` that reads a CSV in `paper/data/`; no
value is typed into plot code. Every number in the text is a macro in `results_macros.tex`,
`budget_macros.tex`, `reval_macros.tex`, or `selection_macros.tex`, generated from the raw files.

## 5. Script-to-result map

| Figure / table | Data file (`paper/data/` unless noted) | Produced by |
|---|---|---|
| Table 1 (closest work), Table 2 (parameters) | – | typed |
| Fig. 1 (architecture), Algorithm 1 | – | `figures/fig_architecture.tex` |
| Table 3 (matched detection) | `matched_summary.csv`, `fig_matched_16.csv` | `exp_matched.py` |
| Fig. 3 (trajectory and frontier), graphical abstract | `fig_frontier_seeds.csv`, `fig_frontier_traj.csv`, `fig_frontier_conv.csv`, `fig_frontier_front.csv`, `fig_frontier_sel_{fe,mbler}.csv` | `run_dense_all.py` → `make_budget_macros.py` |
| Fig. 4 (shift-family dependence, incl. Doppler) | `fig_family_{freq,dop,v150}.csv` | `exp_traj_extra.py`, `exp_traj_doppler.py` → `make_revalidation_macros.py` |
| Fig. 2 (severity) | `fig_sev_ds_5dB.csv`, `fig_sev_v_10dB.csv` | `exp_shift_detect.py` |
| Table 4 (ray-traced scene S4) | `s4_summary.csv` | `exp_s4.py`, `exp_s4_ci.py` |
| Fig. 5 (representation error) | `fig_mismatch_{ds,v}.csv` | `exp_model_mismatch.py` |
| Table 5 (broad-training control) | `broad_summary.csv`, `broad_matched_snr01.csv`, `broad_s4.csv` | `train_broad.py`, `exp_broad.py` |
| Table 6, Fig. 6 (shift monitoring) | `detector_auc_summary.csv`, `detector_relevance.csv`, `fig_sev_{ds,v}_5dB.csv` | `exp_shift_detect.py`, `exp_roc_stats.py` |
| Table 7, Fig. 7 (adaptation) | `adapt_summary.csv`, `fig_adapt_S2_{3,5}dB.csv` | `exp_adapt.py`, `exp_stage_d_stats.py` |
| Scrambling and label-free probes | `results/ojcoms/scramble_probe.csv`, `results/ojcoms/vfe_probe_S2_snr3*.csv` | `exp_scramble_probe.py`, `exp_vfe_probe.py` |
| Fig. 8 (latent parameterization) | `fig_inversion_16QAM-K1824.csv` | `exp_inversion.py`, `exp_latent_target.py` → `analyze_results.py` |
| Table 8 (ablation) | `ablation_all.csv` | `exp_ablation.py` |
| Table 9 (complexity, latency) | `results/ojcoms/complexity.csv` | `exp_complexity.py`, `exp_latency.py` |
| Revalidated factors quoted in Section 5.3 | `results/ojcoms/highprec_summary.csv` | `exp_highprec.py` → `analyze_highprec.py` |
| Checkpoint-averaging frontier | `fig_soup.csv` | `exp_soup.py` → `make_selection_macros.py` |
| Source-only score selections | `results/ojcoms/sourceonly.csv` | `exp_sourceonly.py` → `make_selection_macros.py` |
| Discriminative-baseline trajectory | `fig_disc_traj.csv` | `make_revalidation_macros.py` |
| Supplementary: training curves, ROC, onset, S4 BLER, discriminative checkpoints | `fig_training.csv`, `fig_roc_5dB.csv`, `fig_timeline_5dB.csv`, `fig_s4.csv`, `fig_budget_{ainr,disc}.csv`, `budget_summary.csv` | `train.py` logs, `exp_shift_detect.py`, `exp_s4.py`, `exp_budget.py` |
| Supplementary: stopping selections per seed | `results/ojcoms/dense_eval.csv` | `make_budget_macros.py` |
| Supplementary: protocol amendment evidence | `results/ojcoms/dev_fe_trajectory.csv`, `results/ojcoms/dev_fe_noise.csv` | `exp_dev_fe.py`, `exp_dev_fe_noise.py` |
| Supplementary: all four source-only scores | `results/ojcoms/sourceonly.csv` | `exp_sourceonly.py` -> `make_selection_macros.py` |
| Supplementary: all six averaging windows, averaging frontier, per-trajectory selections | `results/ojcoms/soup.csv`, `fig_soup_wide.csv` | `exp_soup.py` -> `make_selection_macros.py` |

## 6. Known sources of non-reproducibility

* Training runs that were interrupted and resumed (seed 1, and several runs of the training-budget
  study) replay at most the last 1000 steps with the same data-stream seed, so they are not
  bit-identical to uninterrupted runs.
* cuDNN convolution kernels are not deterministic; reruns agree statistically, not bit for bit.
* Latencies depend on the GPU clock, which was thermally limited during long runs; the complexity
  script records the clock state at measurement time.
