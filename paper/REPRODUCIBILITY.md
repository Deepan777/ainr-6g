# Reproducibility Guide

This document lists the exact commands that regenerate every result, table, and figure of
the manuscript *An Active-Inference-Inspired Neural Receiver for Coded OFDM: Generative
Drift Monitoring and CRC-Gated Online Adaptation*. All commands are run from the
repository root unless stated otherwise.

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
| LaTeX | MiKTeX 25.12 (pdfTeX 4.23), `IEEEoj.cls` of 11 Jan 2024 |

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt          # Linux/macOS: venv/bin/pip
venv/Scripts/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
venv/Scripts/pip install psutil
```

All hyperparameters are in `config/config.yaml`. Every script records its random seeds;
stream seeds are fixed in the code, so reruns reproduce the block streams. GPU
non-determinism (cuDNN kernels) can change individual decisions, but not the reported
statistics beyond their stated intervals.

## 2. Training (Section V-B)

| Model | Command | Output |
|---|---|---|
| 16-QAM, seed 0 | `python train.py` | `results/checkpoints/` |
| 16-QAM, seeds 1–4 | `python train.py paths.results_dir=results/seeds/seed<s> paths.checkpoint_dir=results/seeds/seed<s>/checkpoints train.seed=<s> train.session_steps=50000 train.snapshot_steps=[12000,25000]` | `results/seeds/seed<s>/` |
| 64-QAM, seed 0 | `python train.py phy.n_bits_per_symbol=6 paths.checkpoint_dir=results/qam64/checkpoints paths.results_dir=results/qam64` | `results/qam64/` |
| Ablation variants | `python exp_ablation.py --seed <s> [--variants full no_detach pure_vfe]` | `results/ablation/[seed<s>/]` |
| Broadly trained DiscNRX (CDL A–E, 30–1000 ns, 0–120 km/h) | `python train_broad.py` | `results/broad/` |

`train.py` is session-based and resume-safe: rerunning the same command continues from
the last checkpoint (written atomically every 1000 steps). The supervisor
`python pipeline.py` runs the complete experiment plan (`pipeline_steps.py`) in order,
skips finished steps, and resumes interrupted ones; `python pipeline.py --status` lists
their state.

## 3. Experiments

| Paper item | Command | Raw output (`results/ojcoms/`) |
|---|---|---|
| Controlled inversion (Sec. IV-D, Fig. 2) | `python exp_inversion.py` | `inversion_summary.csv`, `inversion_trace.csv` |
| Matched 16-QAM BLER (Sec. VI-A) | `python exp_matched.py --qam 16 --seeds <s>` | `matched_qam16_S1_seed<s>.csv`, `..._classical.csv` |
| 64-QAM BLER (Sec. VI-B) | `python exp_matched.py --qam 64 --seeds 0` | `matched_qam64_S1_seed0.csv` |
| Shift severity, all detectors (Sec. VI-C/D) | `python exp_shift_detect.py --seeds <s>` | `shift_slots_seed<s>.csv` (one row per slot) |
| Likelihood representation error (Fig. 4) | `python exp_model_mismatch.py` | `model_mismatch.csv` |
| Online adaptation, real CRC16 (Sec. VI-E) | `python exp_adapt.py --cond S2 --snr <3|5> --seeds <s>` | `adapt_S2_snr<snr>_seed<s>.csv` |
| Ray-traced scene S4 (Sec. VI-F) | `python s4_gen.py` (channels), `python exp_s4.py --seeds <s>` | `s4_bler_*.csv`, `s4_adapt_seed<s>.csv` |
| Training budget vs robustness | `python exp_budget.py --seeds <s>` | `budget_shift_seed<s>.csv` |
| Ablation incl. shift (Sec. VI-G) | `python exp_ablation.py --seed <s>` | `ablation_{bler,shift,summary}_seed<s>.csv` |
| Broadly trained DiscNRX vs narrow receivers | `python exp_broad.py` | `broad_shift_slots.csv`, `broad_matched.csv`, `broad_s4_bler.csv` |
| Complexity and latency (Sec. VI-H) | `python exp_complexity.py` (GPU otherwise idle) | `complexity.csv`, `complexity_env.json` |

Every experiment writes its results in pieces and resumes after an interruption; a
`--quick` flag runs a small smoke test that never overwrites real outputs.

## 4. Analysis, tables, figures, manuscript

```bash
python analyze_results.py          # statistics per ANALYSIS_PROTOCOL.md -> FINAL_OJCOMS/data/*.csv
                                   #   and FINAL_OJCOMS/tables/results_macros.tex (all numbers in the text)
python make_tables.py              # FINAL_OJCOMS/tables/tab_*.tex
python FINAL_OJCOMS/figures/make_figures.py      # standalone vector PDFs in FINAL_OJCOMS/figures/pdf/
cd FINAL_OJCOMS && pdflatex main && bibtex main && pdflatex main && pdflatex main
python check_build.py              # page count, overfull boxes, undefined references, pending markers
```

Every figure is a PGFPlots/TikZ source in `FINAL_OJCOMS/figures/` that reads a CSV file in
`FINAL_OJCOMS/data/`; no value is typed into plot code. Every number in the text is a
macro in `tables/results_macros.tex`, generated from the same CSV files.

## 5. Script-to-result map

| Figure / table | Data file | Produced by |
|---|---|---|
| Fig. 1 (architecture), Algorithm 1 | – | `figures/fig_architecture.tex` |
| Fig. 2 (inversion) | `data/fig_inversion_16QAM-K1824.csv` | `exp_inversion.py` → `analyze_results.py` |
| Fig. 3 (severity) | `data/fig_sev_ds_5dB.csv`, `data/fig_sev_v_10dB.csv` | `exp_shift_detect.py` |
| Fig. 4 (representation error) | `data/fig_mismatch_{ds,v}.csv` | `exp_model_mismatch.py` |
| Fig. 5, Table 4 (detection) | `data/fig_sev_*_5dB.csv`, `data/detector_auc_summary.csv`, `data/detector_relevance.csv` | `exp_shift_detect.py` |
| Fig. 6 (ROC), Fig. 7 (onset) | `data/fig_roc_5dB.csv`, `data/fig_timeline_5dB.csv` | `exp_shift_detect.py` |
| Fig. 8, Table 5 (adaptation) | `data/fig_adapt_S2_{3,5}dB.csv`, `data/adapt_summary.csv` | `exp_adapt.py` |
| Table 7 (ablation) | `data/ablation_all.csv` | `exp_ablation.py` |
| Table 3 (matched), Table 6 (S4), Table 8 (complexity) | `data/matched_summary.csv`, `data/s4_summary.csv`, `results/ojcoms/complexity.csv` | `exp_matched.py`, `exp_s4.py`, `exp_complexity.py` |
| Training curves (supplement) | `data/fig_training.csv` | `train.py` logs |

## 6. Known sources of non-reproducibility

* Training runs that were interrupted and resumed (seed 1) replay at most the last 1000
  steps with the same data-stream seed, so they are not bit-identical to uninterrupted runs.
* cuDNN convolution kernels are not deterministic; reruns agree statistically, not bit for bit.
* Latencies depend on the GPU clock, which was thermally limited during long runs; the
  complexity script records the clock state at measurement time.
