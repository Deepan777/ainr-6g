# Analysis protocol — checkpoint averaging along the optimization trajectory

Written 2026-09-21, **before** any averaged model was evaluated. The schemes, the streams, the
sample sizes, and the decision rule below are fixed in advance so that the outcome cannot be chosen
after the fact. Experiment: `exp_soup.py`. Output: `results/ojcoms/soup.csv`.

## Question

The paper establishes that robust checkpoints exist within a small matched-accuracy tolerance but
that no matched-data rule reliably selects one. Averaging weights along the trajectory is the
standard remedy in the machine-learning robustness literature and requires no shifted data. Does it
reach the robust region of the frontier for a neural receiver?

## Models

The networks use GroupNorm, so no normalization statistics have to be recomputed after averaging;
an averaged model is the elementwise mean of the selected checkpoints' tensors.

Six averaging schemes, fixed here, over the 26 snapshots of each of the 7 AINR trajectories:

| Name | Snapshots averaged |
|---|---|
| `ALL` | every snapshot from 1000 to 50000 |
| `EARLY` | 5000 to 15000 |
| `MID` | 10000 to 25000 |
| `TAIL10` | 10000 to 50000 |
| `TAIL20` | 20000 to 50000 |
| `TAIL35` | 35000 to 50000 |

Two single checkpoints are evaluated on the same streams as references: the training-distribution
validation optimum `t_val` and the oracle `t_star` of the discovery sweep, both taken from
`results/ojcoms/highprec.csv`.

## Evaluation

Streams are disjoint from every earlier experiment (matched grid seeded 97000 + 100*SNR, S2 98001,
1000 ns 98002).

- Matched: SNR grid 0.4 to 2.4 dB in 0.1-dB steps, 512 blocks per point, `gamma_0.1` by
  interpolation of the crossing of BLER 0.1.
- Shifted: S2 and 1000-ns delay spread at 5 dB, 960 blocks each.
- `Delta` is measured against the best `gamma_0.1` among all models of the same trajectory
  evaluated here, which is the same convention as the high-precision re-evaluation.

## Pre-registered decision rule

A scheme **succeeds** for a trajectory if it satisfies both
1. `Delta <= 0.30` dB, the tolerance used throughout the paper, and
2. `P_S2 <= P_S2(t_val) / 10`.

The **primary endpoint** is whether any single scheme, fixed across trajectories, succeeds in at
least 4 of the 7 trajectories. The scheme with the most successes is reported whatever the outcome.

If no scheme meets the endpoint, the result is reported as a negative alongside the existing null
for rules R1 and R2, and no post-hoc scheme is introduced to rescue it. Any scheme examined after
seeing these results will be labelled exploratory.
