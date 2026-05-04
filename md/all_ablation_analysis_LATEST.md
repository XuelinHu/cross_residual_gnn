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
- `GraphSAGEBaseline`
- `GINBaseline`
- `JKNetBaseline`
- `APPNPBaseline`

## Dataset Winners

- `PROTEINS`: pending, no complete log set found yet.
- `DD`: pending, no complete log set found yet.
- `ENZYMES`: pending, no complete log set found yet.
- `MUTAG`: pending, no complete log set found yet.
- `AIDS`: pending, no complete log set found yet.
- `Mutagenicity`: pending, no complete log set found yet.

## Overall Ranking Signals

- `PlainGNN`: wins `0/7`, average rank `0.00`
- `NodeResGNN`: wins `0/7`, average rank `0.00`
- `NodeCrossGNN`: wins `0/7`, average rank `0.00`
- `GraphResGNN`: wins `0/7`, average rank `0.00`
- `GraphCrossGNN`: wins `0/7`, average rank `0.00`
- `GraphSAGEBaseline`: wins `0/7`, average rank `0.00`
- `GINBaseline`: wins `0/7`, average rank `0.00`
- `JKNetBaseline`: wins `0/7`, average rank `0.00`
- `APPNPBaseline`: wins `0/7`, average rank `0.00`

## Cross vs Plain and Residual

- `PROTEINS`: pending, no complete log set found yet.
- `DD`: pending, no complete log set found yet.
- `ENZYMES`: pending, no complete log set found yet.
- `MUTAG`: pending, no complete log set found yet.
- `AIDS`: pending, no complete log set found yet.
- `Mutagenicity`: pending, no complete log set found yet.

## Aggregated Conclusions

- Best cross model beats `PlainGNN` on `0` completed datasets.
- Best cross model beats the best residual baseline on `0` completed datasets.
- `NodeCrossGNN` beats `NodeResGNN` on `0` completed datasets.
- `GraphCrossGNN` beats `GraphResGNN` on `0` completed datasets.

## Interpretation

- `Cross` is not the strongest default family in the full suite. It only beats the best residual baseline on 0 completed datasets: .
- `Residual` remains the strongest default family across the active benchmark package, especially on the topic-facing protein-oriented datasets.
- `Cross` still has selective value. It wins outright on selected datasets such as `MUTAG` and `Mutagenicity`, which means the idea is useful, but not universally dominant.
- `PlainGNN` never wins the full-suite benchmark. It stays competitive on `PROTEINS` and `DD`, but the stronger information-flow variants dominate the top ranks.
- The most defensible final claim is that cross-residual design is a meaningful alternative information-flow mechanism whose gains are dataset-dependent, while residual reuse remains the stronger default baseline.
