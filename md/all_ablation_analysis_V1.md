# Full-Suite Ablation Analysis

## Scope

This note summarizes the completed full-suite run over all seven available datasets.

Datasets:

- `PROTEINS`
- `DD`
- `ENZYMES`
- `MUTAG`
- `AIDS`
- `Mutagenicity`

Compared models:

- `PlainGNN`
- `NodeResGNN`
- `NodeCrossGNN`
- `GraphResGNN`
- `GraphCrossGNN`

## Dataset Winners

- `PROTEINS`: `GraphResGNN` with `0.72232 ± 0.03673`
- `DD`: `NodeResGNN` with `0.72066 ± 0.02593`
- `ENZYMES`: `GraphResGNN` with `0.31167 ± 0.07465`
- `MUTAG`: `NodeCrossGNN` with `0.73969 ± 0.06762`
- `AIDS`: `GraphResGNN` with `0.91600 ± 0.01210`
- `Mutagenicity`: `NodeCrossGNN` with `0.80333 ± 0.01589`

## Overall Ranking Signals

- `GraphResGNN`: wins `3/7`, average rank `2.33`
- `NodeCrossGNN`: wins `2/7`, average rank `2.00`
- `NodeResGNN`: wins `1/7`, average rank `3.17`
- `GraphCrossGNN`: wins `0/7`, average rank `3.50`
- `PlainGNN`: wins `0/7`, average rank `4.00`

## Cross vs Plain and Residual

- `PROTEINS`: best cross vs plain `-0.00899`, best cross vs best residual `-0.02605`, `NodeCrossGNN - NodeResGNN = +0.00182`, `GraphCrossGNN - GraphResGNN = -0.03596`
- `DD`: best cross vs plain `+0.00339`, best cross vs best residual `-0.00082`, `NodeCrossGNN - NodeResGNN = -0.00082`, `GraphCrossGNN - GraphResGNN = +0.01621`
- `ENZYMES`: best cross vs plain `+0.05667`, best cross vs best residual `-0.01000`, `NodeCrossGNN - NodeResGNN = +0.03500`, `GraphCrossGNN - GraphResGNN = -0.01000`
- `MUTAG`: best cross vs plain `+0.01607`, best cross vs best residual `+0.00541`, `NodeCrossGNN - NodeResGNN = +0.00541`, `GraphCrossGNN - GraphResGNN = -0.01607`
- `AIDS`: best cross vs plain `+0.02800`, best cross vs best residual `-0.04250`, `NodeCrossGNN - NodeResGNN = +0.00450`, `GraphCrossGNN - GraphResGNN = -0.04500`
- `Mutagenicity`: best cross vs plain `+0.02191`, best cross vs best residual `+0.00115`, `NodeCrossGNN - NodeResGNN = +0.01361`, `GraphCrossGNN - GraphResGNN = +0.00069`

## Aggregated Conclusions

- Best cross model beats `PlainGNN` on `5` completed datasets.
- Best cross model beats the best residual baseline on `2` completed datasets.
- `NodeCrossGNN` beats `NodeResGNN` on `5` completed datasets.
- `GraphCrossGNN` beats `GraphResGNN` on `2` completed datasets.

## Interpretation

- `Cross` is not the strongest default family in the full suite. It only beats the best residual baseline on 2 completed datasets: `MUTAG`, `Mutagenicity`.
- `Residual` remains the strongest default family across the active benchmark package, especially on the topic-facing protein-oriented datasets.
- `Cross` still has selective value. It wins outright on selected datasets such as `MUTAG` and `Mutagenicity`, which means the idea is useful, but not universally dominant.
- `PlainGNN` never wins the full-suite benchmark. It stays competitive on `PROTEINS` and `DD`, but the stronger information-flow variants dominate the top ranks.
- The most defensible final claim is that cross-residual design is a meaningful alternative information-flow mechanism whose gains are dataset-dependent, while residual reuse remains the stronger default baseline.
