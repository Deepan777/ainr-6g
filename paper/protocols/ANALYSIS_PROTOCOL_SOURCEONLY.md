# Analysis protocol — source-only checkpoint scores

Written 2026-09-21, **before** any score was computed. Experiment: `exp_sourceonly.py`. Output:
`results/ojcoms/sourceonly.csv`.

## Question

Source-only scorers rank checkpoints for out-of-distribution robustness using in-distribution
representations alone \[TopoGeoScore, arXiv:2605.08870; representation geometry, arXiv:2602.03951].
They are formulated for image classification, where class-conditional embeddings are available. A
neural receiver has no classes, so the scorers are not transferable as published. Three analogues
that are computable for a receiver are tested here, together with a weight-space flatness measure
of the kind used in the same literature.

## Scores, all computed from matched data only

For each checkpoint, encoder features are collected on a fixed set of matched slots.

| Name | Definition |
|---|---|
| `effrank` | participation ratio of the eigenvalues of the feature covariance, $(\sum\lambda)^2/\sum\lambda^2$ |
| `logdet` | log-determinant of the regularized feature covariance, a global spectral-complexity proxy |
| `featnorm` | mean Euclidean norm of the feature vectors |
| `sharpness` | mean increase of the matched coded-bit cross-entropy under Gaussian weight perturbation of relative scale 0.01, averaged over 5 draws |

## Direction and selection

The sign of each score is not assumed. It is fixed on the development trajectories (seeds 2, 3, 4)
as the direction of the rank correlation between the score and $P_{\mathrm{S2}}$, and then applied
unchanged to the held-out trajectories (seeds 5, 6, 7, 8). This mirrors the protocol already used
for rules R1 and R2.

Each score selects, among the checkpoints of a trajectory satisfying $\Delta \le 0.30$ dB, the one
that optimizes the score in the fixed direction. Both quantities are available without shifted data.

## Pre-registered decision rule

A selection **succeeds** for a trajectory if $P_{\mathrm{S2}} \le P_{\mathrm{S2}}(t_{\mathrm{val}})/10$.
The **primary endpoint** is whether any score succeeds in at least 3 of the 4 held-out
trajectories. Scores are evaluated on the discovery sweep; any score meeting the endpoint is then
re-measured on the independent streams before it is reported as a working criterion.

If no score meets the endpoint, it is reported as a negative, with the correlations given so that
the reader can see how close each came. No score invented after seeing these results will be
presented as anything other than exploratory.
