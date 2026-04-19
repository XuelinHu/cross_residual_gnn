# Current Experiment Inventory And Gaps

## 1. Current directory and version status

- `achivement_V0`
  - historical archive only
  - no longer part of the active runtime versioning
- `logs/V1`, `records/V1`
  - older formal version
  - includes early main results and early sensitivity scans
- `logs/V2`, `records/V2`
  - previous formal benchmark version
  - contains the complete main benchmark and supplementary gate studies
- `logs/V3`, `records/V3`
  - current consolidated paper-facing version
  - now includes accepted `V2` outputs and all new residual studies

Current active state:

- `V3` is now the single paper-facing version
- accepted `V2` benchmark and supplementary outputs have been copied into `V3`
- new residual-mode and residual-parameter experiments are also stored in `V3`

## 2. What is complete right now

### 2.1 Main benchmark matrix

The full main benchmark is now available in `V3`.

Coverage:

- datasets: `6`
  - `AIDS`
  - `DD`
  - `ENZYMES`
  - `MUTAG`
  - `Mutagenicity`
  - `PROTEINS`
- methods: `9`
  - `PlainGNN`
  - `NodeResGNN`
  - `NodeCrossGNN`
  - `GraphResGNN`
  - `GraphCrossGNN`
  - `GraphSAGEBaseline`
  - `GINBaseline`
  - `JKNetBaseline`
  - `APPNPBaseline`
- operators: `4`
  - `GCNConv`
  - `GATConv`
  - `GINConv`
  - `SAGEConv`
- folds: `5`

Main benchmark size:

- `6 datasets x 9 methods x 4 operators x 5 folds = 720`
- current status in `V3`: `720 / 720`

### 2.2 Supplementary ablations already complete

#### A. Main architecture ablation

Already covered by the main benchmark matrix:

- plain vs residual vs cross
- node-level vs graph-level
- internal methods vs external baselines

#### B. AIDS gate ablation

Completed and consolidated into `V3`.

Target:

- `AIDS + GraphResGNN + GINConv`

Settings:

- `learnable gate`
- `fixed gate = 0.0`
- `fixed gate = 0.5`
- `fixed gate = 1.0`

Fold coverage:

- all four settings are complete on `5/5` folds

#### C. Cross + gate ablation

Completed and consolidated into `V3`.

Targets:

- `AIDS + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GINConv`
- `ENZYMES + NodeCrossGNN + GATConv`
- `MUTAG + GraphCrossGNN + GATConv`
- `Mutagenicity + NodeCrossGNN + GATConv`

Settings:

- `learnable gate`
- `fixed gate = 0.0`
- `fixed gate = 0.5`
- `fixed gate = 1.0`

Coverage:

- `6 targets x 4 settings x 5 folds = 120`
- current status in `V3`: `120 / 120`

#### D. Residual-mode ablation

Completed in `V3`.

Targets:

- `AIDS + GraphResGNN + GINConv`
- `PROTEINS + GraphResGNN + SAGEConv`
- `AIDS + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GATConv`

Settings:

- `learnable`
- `topk_0p25`
- `topk_0p5`
- `sparse_0p02`
- `sparse_0p05`

Coverage:

- `4 targets x 5 settings x 5 folds = 100`
- current status in `V3`: `100 / 100`

#### E. Residual parameter sweeps

Completed in `V3`.

Targets:

- `AIDS + GraphResGNN + GINConv`
- `PROTEINS + GraphResGNN + SAGEConv`
- `AIDS + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GATConv`

Newly added settings:

- `topk_0p75`
- `sparse_0p1`

Coverage:

- `4 targets x 2 settings x 5 folds = 40`
- current status in `V3`: `40 / 40`

## 3. Parameter experiments already done

### 3.1 Original sensitivity scans

Stored in `V1`.

Targets:

- datasets:
  - `PROTEINS`
  - `DD`
  - `ENZYMES`
- methods:
  - `NodeCrossGNN`
  - `GraphCrossGNN`
- operator:
  - `GCNConv`

Scanned parameters:

- `lr`
- `drop`
- `h_layer`

Important limitation:

- these runs were executed only on `fold 0`
- they are useful for trend inspection
- they are not a full 5-fold paper-grade sensitivity benchmark

### 3.2 AIDS supplementary parameter scans

Stored in `V2`, and the accepted artifacts are now also available through `V3`.

Target:

- `AIDS + GraphResGNN + GINConv`

Scanned parameters:

- `lr`
- `drop`
- `weight_decay`
- `h_layer`
- `dim`
- `gate_init`

Important limitation:

- these runs are also only `fold 0`
- they are useful as local sensitivity evidence
- they should not be treated as a full benchmark-level parameter study

### 3.3 New 5-fold residual parameter sweeps

Stored in `V3`.

Target family:

- `AIDS + GraphResGNN + GINConv`
- `PROTEINS + GraphResGNN + SAGEConv`
- `AIDS + NodeCrossGNN + GATConv`
- `DD + NodeCrossGNN + GATConv`

Scanned parameters:

- `topk_ratio = 0.25, 0.5, 0.75`
- `sparse_lambda = 0.02, 0.05, 0.1`

Important note:

- the earlier `0.25 / 0.5 / 0.02 / 0.05` points came from the residual-mode ablation
- the later `0.75 / 0.1` points were added by the dedicated residual parameter sweep

## 4. Current experiment matrix

### 4.1 Main benchmark matrix

| category | status | location |
|---|---:|---|
| 6 datasets | complete | `V3` |
| 9 methods | complete | `V3` |
| 4 operators | complete | `V3` |
| 5 folds | complete | `V3` |
| total 720 jobs | `720/720` | `logs/V3` |

### 4.2 Gate ablation matrix

| category | status | location |
|---|---:|---|
| AIDS gate ablation | `20/20` | `V3` |
| cross + gate ablation | `120/120` | `V3` |

### 4.3 Residual ablation matrix

| category | status | location |
|---|---:|---|
| residual mode comparison | `100/100` | `V3` |
| residual parameter sweeps | `40/40` | `V3` |

### 4.4 Sensitivity matrix

| category | status | limitation | location |
|---|---:|---|---|
| original cross sensitivity | complete | fold 0 only | `V1` |
| AIDS supplementary sensitivity | complete | fold 0 only | `V2` and `V3` |
| new residual parameter sweeps | complete | representative targets only | `V3` |

## 5. What is still missing

These are the real remaining gaps.

### Gap 1. No compact residual summary report yet

Training is already done, but there is still no single report that merges:

- `residual_mode_*`
- `residual_param_*`

into one clean table for writing.

This is now mainly a reporting gap, not a training gap.

### Gap 2. Old parameter studies are still partly single-fold

The older sensitivity studies remain weaker than the main benchmark because they are not full 5-fold studies.

This is acceptable for support evidence, but not ideal if you want very strong parameter claims.

### Gap 3. Residual-mode coverage is still selective

Current residual-mode and residual-parameter experiments cover representative targets, not the full benchmark matrix.

This is enough for a paper draft, but not enough for a claim like:

- top-k or sparse residual is universally best everywhere

## 6. Recommended next steps

### Must do

1. Generate a single residual summary report from `V3`
2. Use `V3` as the only paper-facing experiment version
3. Start the middle draft

### Should do

1. Decide whether the paper centers on:
   - `cross + gate`
   - or `residual mode design`
2. Keep old single-fold sensitivity studies only as supporting evidence

### Optional

1. Extend residual-mode ablation to more datasets and operators
2. Clean duplicate runtime files in `logs/V3`
