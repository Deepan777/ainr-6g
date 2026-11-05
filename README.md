# Optimization-Dependent Robustness and Checkpoint Selection in Neural OFDM Receivers

Code, configurations, raw numerical results, analysis protocols, and trained
checkpoints for the paper *"Optimization-Dependent Robustness and Checkpoint
Selection in Neural OFDM Receivers"* (under review at *Array*, Elsevier).

**Authors:** A. Helen Sharmila, P. Deepanramkumar (corresponding author,
deepanramkumar.p@vit.ac.in) — School of Computer Science and Engineering,
Vellore Institute of Technology, Vellore, Tamil Nadu, India.

> **Note on earlier versions of this repository.** This repository previously
> described an active-inference receiver whose generative reconstruction residual
> was presented as a label-free channel-drift detector. That claim did not
> survive measurement against a matched discriminative baseline and is reported
> as a negative result in the current paper. The code is the same code; the
> conclusions changed. See *What the paper reports* below.

## What the paper studies

Learned OFDM receivers are optimized on a training distribution and then judged
on channels it does not contain. This paper treats robustness as a property of
the **optimization trajectory** rather than of the trained architecture alone,
and poses the choice of deployed weights as a constrained problem: minimise
error under shift subject to a bounded loss of matched-channel accuracy.

A variational receiver supplies the evidence. Channel and noise posteriors and a
physics-based likelihood run alongside a discriminatively trained detector, and a
**stop-gradient** keeps the detection path numerically identical to the
discriminative baseline. Detection parity is therefore a construction, not a
finding, which is what makes the generative path's separate contribution
measurable.

## What the paper reports

Positive results:

- Continued optimization improved the matched SNR requirement at block error
  rate 0.1 by 0.50 dB while raising the block error rate under a compound
  delay-spread and channel-family shift about 70-fold. Re-evaluated on streams
  disjoint from those used to select them, checkpoints near the matched optimum
  retained 12.7 to 271 times lower shifted-channel error.
- Once the trajectory is complete, matched data alone suffice to find those
  checkpoints. A representation statistic selected the same checkpoint as the
  oracle in all seven trajectories, meeting the pre-registered criterion on all
  four held-out ones; averaging a mid-training window (iterations 1.0e4 to
  2.5e4 of 5.0e4) met it in six of seven, at about half the training cost. Both
  protocols were fixed in writing before any held-out trajectory was scored
  (`paper/protocols/`).

Negative results, reported as such:

- The generative path improves neither detection nor generic shift detection:
  the model-fit statistic does not beat the soft-output confidence of an equally
  accurate discriminative receiver, and every learned statistic also fires on
  shifts that cost no accuracy.
- The pre-registered test of rules that stop training **online** is null; the
  two selection results above are retrospective, over a completed trajectory.
- Label-free free-energy adaptation flattens the bit posterior rather than
  recovering accuracy.
- The degradation is **shift-dependent**: it is not observed under Doppler, at
  any of the speeds tested.
- CRC-gated recovery operates only while correctly decoded blocks remain.

The evidence is simulation-based, on a standard-compliant LDPC coding chain with
a simplified pilot pattern. Limitations are stated in the paper's scope
subsection.

## Requirements

- NVIDIA GPU with CUDA (results were produced on an RTX 3060 Laptop, 6 GB).
- Python 3.10+, [Sionna](https://nvlabs.github.io/sionna/) 2.x (PyTorch
  backend), PyTorch 2.x — see `requirements.txt`.

```bash
bash setup.sh              # creates a venv and installs all dependencies
source venv/bin/activate   # (Windows: venv/Scripts/activate)
python -c "import sionna, torch; print('OK', torch.cuda.is_available())"
```

All hyperparameters live in `config/config.yaml`; nothing is hard-coded.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/channel/sionna_channel.py` | 5G NR OFDM link (CDL channel, LDPC, QAM, pilots) |
| `src/variational_posterior.py` | The inference network q(bits, channel, noise \| Y) |
| `src/generative_model.py` | The parameter-free physics likelihood p(Y \| c, h, sigma^2) |
| `src/vfe.py` | Variational free-energy objective |
| `src/ainr.py` | The receiver (posterior + physics + BP decoder + adaptation) |
| `src/baselines/` | LMMSE and parameter-matched discriminative NRX baselines |
| `train.py` | Offline training (resume-safe sessions) |
| `pipeline.py` | Supervisor that runs the whole experiment plan in order |
| `results/ojcoms/` | Raw per-slot and per-checkpoint outputs (CSV/JSON) |
| `results/` | Trained checkpoints (`*.pt`) |
| `paper/data/` | One CSV per figure; no value is typed into plot code |
| `paper/figures/`, `paper/tables/` | PGFPlots/TikZ sources and generated macro files |
| `paper/protocols/` | The analysis protocols, as written before the data were scored |
| `paper/REPRODUCIBILITY.md` | Exact commands that regenerate every number |

## Reproducing the results

`paper/REPRODUCIBILITY.md` is the authoritative guide: it lists the environment
versions, every command in order, and a map from each figure and table to the
script and raw file behind it. The short version:

```bash
python train.py              # re-run until it prints ALL DONE
python pipeline.py           # runs the full experiment plan, skipping finished steps
python analyze_results.py    # statistics -> paper/data/*.csv, paper/tables/results_macros.tex
python make_tables.py && python paper/figures/make_figures.py
```

Every number quoted in the paper is a LaTeX macro generated from a raw result
file by `analyze_results.py`, `make_budget_macros.py`,
`make_revalidation_macros.py`, or `make_selection_macros.py`. No number is typed
by hand.

| Script | Paper item |
|--------|------------|
| `exp_matched.py` | Table 3, matched detection; Supplementary matched BLER curves |
| `run_dense_all.py` → `make_budget_macros.py` | Fig. 2, the robustness–accuracy frontier along the trajectory |
| `exp_highprec.py` → `analyze_highprec.py` → `make_revalidation_macros.py` | Independent re-evaluation of the selected checkpoints |
| `exp_traj_doppler.py`, `exp_traj_extra.py` | Fig. 3, shift-family dependence incl. the Doppler null result |
| `exp_shift_detect.py` | Fig. 4 severity, Fig. 6 and Table 6 monitoring |
| `s4_gen.py` + `exp_s4.py` | Table 4, ray-traced Munich scene |
| `exp_model_mismatch.py`, `exp_attribution.py` | Fig. 5, genie attribution of the reconstruction error |
| `train_broad.py` + `exp_broad.py` | Table 5, broad-training control |
| `exp_pilot_eval.py` | Pilot-overhead probe |
| `exp_adapt.py`, `exp_vfe_probe.py`, `exp_scramble_probe.py` | Fig. 7 and Table 7, CRC-gated and label-free adaptation |
| `exp_inversion.py`, `exp_latent_target.py` | Fig. 8, coded-bit vs information-bit latent |
| `exp_ablation.py` | Table 8, ablation incl. the stop-gradient |
| `exp_complexity.py`, `exp_latency.py` | Table 9, complexity and latency |
| `exp_soup.py`, `exp_sourceonly.py` → `make_selection_macros.py` | The two checkpoint-selection results |

Every experiment writes its output in pieces and resumes after an interruption.
A `--quick` flag runs a minutes-long smoke test that never overwrites real
output.

## What is and is not included

Included: all source, all configurations, the raw CSV/JSON behind every reported
number, the analysis protocols, and the final trained checkpoints for every seed
and variant (≈63 MB).

Not included: the dense training-budget snapshots — 26 snapshots per trajectory
for seeds 2–8, 643 MB — which exceed what is reasonable for a Git repository.
Their per-snapshot evaluation is released in full as
`results/ojcoms/dense_eval.csv`, so the trajectory analysis and both selection
results can be reproduced from the CSVs without them. The snapshots themselves
are available from the corresponding author on request.

Also not included: the scripts of an unfinished constant-learning-rate control,
which is not reported in the paper, and the plotting scripts of a superseded
earlier submission.

## Citation

Citation details will be added on publication.

## License

MIT — see `LICENSE`.
