# AINR-6G: Active-Inference Neural Receiver for 6G OFDM

Code for the paper *"An active-inference neural receiver for 6G OFDM: Joint
detection, label-free channel-drift monitoring, and self-supervised online
adaptation"* (submitted to *Physical Communication*, Elsevier).

**Authors:** A. Helen Sharmila, P. Deepanramkumar (corresponding author,
deepanramkumar.p@vit.ac.in) — School of Computer Science and Engineering,
Vellore Institute of Technology, Vellore, Tamil Nadu, India.

The AINR recasts OFDM detection as variational inference under an explicit,
physics-based generative model of the received resource grid. It jointly
infers the transmitted coded bits, the channel response, and the noise
variance by minimizing a hybrid variational free-energy objective. Because the
receiver keeps a generative model, its reconstruction residual is a built-in,
label-free channel-drift detector, and its CRC verdict gates a self-supervised
online-adaptation rule.

## Requirements

- NVIDIA GPU with CUDA (the paper's results were produced on an RTX 3060
  Laptop, 6 GB).
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
| `src/ainr.py` | The AINR receiver (posterior + physics + BP decoder + adaptation) |
| `src/baselines/` | LMMSE and parameter-matched discriminative NRX baselines |
| `train.py` | Offline training (resume-safe sessions) |
| `evaluate.py` | Matched / drift / adaptation evaluation stages |

## Reproducing the paper's results

Train both neural receivers (checkpoints land in `results/checkpoints/`):

```bash
python train.py            # re-run until it prints ALL DONE (50k steps)
```

Then each experiment script reproduces one artifact of the paper:

| Script | Paper artifact |
|--------|----------------|
| `evaluate.py --stage all` | Matched/drift BLER sweeps (Figs. 4–5, drift timeline) |
| `ci_bler.py` | Matched BLER confidence intervals (Fig. 4, Table 2) |
| `exp_latent_target.py` | Controlled inversion experiment (Fig. 2) |
| `exp_roc_stats.py` | Multi-run drift AUC, bootstrap CIs, operating points (Fig. 6, Table 3) |
| `s4_gen.py` + `exp_s4_ci.py` | Ray-traced S4 evaluation with CIs (Fig. 8, Table 4) |
| `exp_stage_d_stats.py` | Multi-seed online adaptation + LR sweep (Fig. 10a) |
| `exp_stage_d_harsh.py` | Harsh-drift CRC-gating comparison (Fig. 10b) |
| `exp_ablation.py` | Five-variant budget-matched ablation (Table 5) |
| `exp_latency.py` | Latency protocol (Table 6) |
| `multiseed.py` | Cross-seed training-stability check |
| `make_new_figures.py`, `make_paper_figures.py` | Render all figures from the result CSVs |

`run_all_experiments.sh` runs the exp suite sequentially (single-GPU safe).
Every script accepts `--quick` for a minutes-long smoke test where applicable.

Trained checkpoints and the raw result CSVs behind the published figures are
available from the corresponding author upon reasonable request.

## Citation

If you use this code, please cite the paper (citation details will be added
upon publication).

## License

MIT — see `LICENSE`.
