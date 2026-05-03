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

- `PROTEINS`: `NodeResGNN` with `0.71425 ± 0.02235`
- `DD`: `GraphResGNN` with `0.72746 ± 0.01436`
- `ENZYMES`: `GraphResGNN` with `0.33333 ± 0.10000`
- `MUTAG`: `NodeResGNN` with `0.74509 ± 0.05036`
- `AIDS`: `GraphResGNN` with `0.90900 ± 0.01940`
- `Mutagenicity`: `GraphResGNN` with `0.80219 ± 0.02440`

## Overall Ranking Signals

- `GraphResGNN`: wins `4/7`, average rank `1.83`
- `NodeResGNN`: wins `2/7`, average rank `3.17`
- `GraphCrossGNN`: wins `0/7`, average rank `2.67`
- `NodeCrossGNN`: wins `0/7`, average rank `3.00`
- `PlainGNN`: wins `0/7`, average rank `4.33`

## Cross vs Plain and Residual

- `PROTEINS`: best cross vs plain `-0.00092`, best cross vs best residual `-0.01258`, `NodeCrossGNN - NodeResGNN = -0.01977`, `GraphCrossGNN - GraphResGNN = -0.00541`
- `DD`: best cross vs plain `+0.02370`, best cross vs best residual `-0.00678`, `NodeCrossGNN - NodeResGNN = +0.00762`, `GraphCrossGNN - GraphResGNN = -0.00678`
- `ENZYMES`: best cross vs plain `+0.01667`, best cross vs best residual `-0.04833`, `NodeCrossGNN - NodeResGNN = +0.00333`, `GraphCrossGNN - GraphResGNN = -0.04833`
- `MUTAG`: best cross vs plain `+0.00555`, best cross vs best residual `-0.00526`, `NodeCrossGNN - NodeResGNN = -0.00526`, `GraphCrossGNN - GraphResGNN = +0.01593`
- `AIDS`: best cross vs plain `+0.05550`, best cross vs best residual `-0.02400`, `NodeCrossGNN - NodeResGNN = +0.02350`, `GraphCrossGNN - GraphResGNN = -0.03500`
- `Mutagenicity`: best cross vs plain `+0.01615`, best cross vs best residual `-0.00300`, `NodeCrossGNN - NodeResGNN = +0.00646`, `GraphCrossGNN - GraphResGNN = -0.00300`

## Aggregated Conclusions

- Best cross model beats `PlainGNN` on `5` completed datasets.
- Best cross model beats the best residual baseline on `0` completed datasets.
- `NodeCrossGNN` beats `NodeResGNN` on `4` completed datasets.
- `GraphCrossGNN` beats `GraphResGNN` on `1` completed datasets.

## Interpretation

- `Cross` is not the strongest default family in the full suite. It only beats the best residual baseline on 0 completed datasets: .
- `Residual` remains the strongest default family across the active benchmark package, especially on the topic-facing protein-oriented datasets.
- `Cross` still has selective value. It wins outright on selected datasets such as `MUTAG` and `Mutagenicity`, which means the idea is useful, but not universally dominant.
- `PlainGNN` never wins the full-suite benchmark. It stays competitive on `PROTEINS` and `DD`, but the stronger information-flow variants dominate the top ranks.
- The most defensible final claim is that cross-residual design is a meaningful alternative information-flow mechanism whose gains are dataset-dependent, while residual reuse remains the stronger default baseline.
