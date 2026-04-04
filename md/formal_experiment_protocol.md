# Formal Experiment Protocol

## Goal

Establish a paper-ready experimental protocol for V3 graph classification with a focused narrative around cross-residual architectures.

## Data Splitting

- Dataset source: PyG `TUDataset`
- Outer evaluation: stratified 5-fold cross validation
- Inner validation split: stratified random split from the training fold
- Validation ratio: `0.1`
- Seed policy:
  - dataset-level deterministic shuffle uses the global run seed
  - train/val split uses `seed + fold`

## Training Defaults

- Optimizer: Adam
- Early stopping target: validation loss
- Scheduler: `ReduceLROnPlateau`
- Gradient clipping: enabled
- Metrics:
  - primary: test accuracy
  - secondary: test loss
- Report format: mean ± std over 5 folds

## Focused Model Set

The focused paper narrative should prefer these comparisons:

- `PlainGNN`
- `NodeResGNN`
- `NodeCrossGNN`
- `GraphResGNN`
- `GraphCrossGNN`

## Focused Dataset Set

Main datasets:

- `MUTAG`
- `PROTEINS`
- `DD`
- `MSRC_9`

Topic-facing datasets:

- `PROTEINS`
- `DD`
- `ENZYMES`

Extended datasets:

- `AIDS`
- `Mutagenicity`

## Recommended Storyline

- `MUTAG`: emphasize `GraphCrossGNN` if it remains the strongest cross-residual variant.
- `PROTEINS`: emphasize `NodeCrossGNN` against `NodeResGNN`.
- `DD`: emphasize the stronger of `NodeCrossGNN` and `GraphCrossGNN` after final protocol reruns.
- `MSRC_9`: treat as a stress test showing where cross residual does not automatically dominate.
- `ENZYMES`: use as the additional protein/enzyme-oriented benchmark for the topic-facing package.

## Current Best Cross-Residual Candidates

Based on the current tuning round:

- `PROTEINS / NodeCrossGNN`: `lr=0.003`, `drop=0.2`, `h_layer=4`
- `PROTEINS / GraphCrossGNN`: `lr=0.003`, `drop=0.3`, `h_layer=4`
- `DD / NodeCrossGNN`: `lr=0.003`, `drop=0.3`, `h_layer=3`
- `DD / GraphCrossGNN`: `lr=0.002`, `drop=0.2`, `h_layer=4`

## Publication-Ready Deliverables

- Main result table: focused models across main datasets
- Topic-facing result table: focused models across `PROTEINS`, `DD`, `ENZYMES`
- Cross-residual ablation table: plain vs node-res vs node-cross vs graph-res vs graph-cross
- Extended result table: `AIDS`, `Mutagenicity`
- Optional convergence plots for the best cross-residual settings
